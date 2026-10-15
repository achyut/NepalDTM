"""Build the compact 8 m stack for one or more nationwide tiles (same layers and encoding as build_stack_hma.py).

Tile = 24 km core + 1 km buffer (3250 x 3250 px). Layers: hma, cop, dsm, src (1 HMA, 2 Copernicus fill), wc,
chm_mean, chm_q3, chm_max. Tiles without HMA data use Copernicus as the surface (src = 2).
Output: data/nw/stack/<id>.tif (uint16, encoded as in fuse_hma.encode). Skips tiles already built.

Run:
    PROJ_NETWORK=ON ~/workspace/super-resolution-satellite/.venv/bin/python -I nw_stack.py --ids E0336N3048
    PROJ_NETWORK=ON ~/workspace/super-resolution-satellite/.venv/bin/python -I nw_stack.py --all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stack_hma import CHM, COP, CRS, HMA, RES, fill_from_cop, geoid_offset, quadkeys, signed, warp  # noqa: E402
from fuse_hma import encode  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
NW = DATA / "nw"
BUF = 1000
ENC = {"hma": "h2", "cop": "h2", "dsm": "h2", "src": "u8", "wc": "u8", "chm_mean": "q4", "chm_q3": "u8", "chm_max": "u8"}


def build(t, stac):
    x0, y0, x1, y1 = t["core"]
    T = from_origin(x0 - BUF, y1 + BUF, RES, RES)
    W = H = int((x1 - x0 + 2 * BUF) / RES)
    w, s, e, n = t["lonlat"]
    L = {}
    tm = {}
    t0 = time.time()
    L["cop"] = warp([COP.format(lat=la, lon=lo) for la in range(math.floor(s), math.floor(n) + 1)
                     for lo in range(math.floor(w), math.floor(e) + 1)], T, W, H, Resampling.bilinear)
    tm["cop"] = time.time() - t0
    t0 = time.time()
    hma = np.full((H, W), np.nan, "float32")
    if t["hma"]:
        for attempt in range(2):  # an expired signed URL or a dropped read returns all-NaN: retry once
            hma = warp([signed(HMA.format(t=k)) for k in t["hma"]], T, W, H, Resampling.bilinear, -9999.0)
            if np.isfinite(hma).any():
                break
        z = geoid_offset(T, W, H)
        # PROJ returns inf (silently) when the EGM2008 grid is unreachable; the grid is installed locally in
        # ~/.local/share/proj/us_nga_egm08_25.tif, but never write a tile without it.
        if not np.isfinite(z).all() or np.abs(z).max() > 100:
            raise RuntimeError("geoid offset not applied (EGM2008 grid missing)")
        hma = hma + z
        d = hma - L["cop"]
        if np.isfinite(d).any():
            hma[np.abs(d - np.nanmedian(d)) > 40] = np.nan  # blunders: cloud, snow, matching failures
    tm["hma"] = time.time() - t0
    L["hma"] = hma.astype("float32")
    L["dsm"] = fill_from_cop(hma, L["cop"]) if np.isfinite(hma).sum() > 1000 else L["cop"].copy()
    L["src"] = np.where(np.isfinite(hma), 1, 2).astype("float32")
    t0 = time.time()
    items = list(stac.search(collections=["esa-worldcover"], bbox=t["lonlat"], datetime="2021").items())
    L["wc"] = warp([it.assets["map"].href for it in items], T, W, H, Resampling.mode, 0)
    tm["wc"] = time.time() - t0
    t0 = time.time()
    qs = quadkeys(t["lonlat"])
    for name, rs in (("chm_mean", Resampling.average), ("chm_q3", Resampling.q3), ("chm_max", Resampling.max)):
        L[name] = warp([CHM.format(q=q) for q in qs], T, W, H, rs, 255)
    tm["chm"] = time.time() - t0
    path = NW / "stack" / f"{t['id']}.tif"
    tmp = path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", driver="GTiff", width=W, height=H, count=len(ENC), dtype="uint16", crs=CRS,
                       transform=T, nodata=0, compress="deflate", predictor=2, tiled=True, blockxsize=512,
                       blockysize=512) as dst:
        for i, (k, enc) in enumerate(ENC.items(), 1):
            dst.write(encode(L[k], enc), i)
            dst.set_band_description(i, f"{k}:{enc}")
    tmp.replace(path)
    return dict(id=t["id"], hma_valid=float(np.isfinite(hma).mean()), cop_valid=float(np.isfinite(L["cop"]).mean()),
                mb=path.stat().st_size / 1e6, **{f"s_{k}": round(v) for k, v in tm.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--part", default="0/1", help="k/n: this worker takes every n-th tile starting at k")
    ap.add_argument("--reverse", action="store_true", help="work from the end of the tile list (second machine)")
    args = ap.parse_args()
    tiles = json.load(open(NW / "tiles.json"))
    if not args.all:
        tiles = [t for t in tiles if t["id"] in set(args.ids or [])]
    if args.reverse:
        tiles = tiles[::-1]
    k, n = map(int, args.part.split("/"))
    tiles = tiles[k::n]
    (NW / "stack").mkdir(parents=True, exist_ok=True)
    stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                     modifier=planetary_computer.sign_inplace)
    log = NW / "stack_log.jsonl"
    for i, t in enumerate(tiles, 1):
        if (NW / "stack" / f"{t['id']}.tif").exists():
            continue
        t0 = time.time()
        try:
            r = build(t, stac)
        except Exception as e:  # keep going; failed tiles are retried on the next run
            print(f"  {i}/{len(tiles)} {t['id']} FAILED: {str(e)[:200]}", flush=True)
            continue
        r["s_total"] = round(time.time() - t0)
        with open(log, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"  {i}/{len(tiles)} {t['id']}: HMA {r['hma_valid']:.0%}, {r['mb']:.0f} MB, {r['s_total']} s "
              f"(cop {r['s_cop']}, hma {r['s_hma']}, wc {r['s_wc']}, chm {r['s_chm']})", flush=True)


if __name__ == "__main__":
    main()
