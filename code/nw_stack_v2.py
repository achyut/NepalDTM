#!/usr/bin/env python3
"""Build stack_v2/<id>.tif: the v1 stack with the photon-adjusted HMA surface and three extra layers.

Changes from data/nw/stack (nw_stack.py):
  hma          per-pixel median of the jointly adjusted strips (nw_strip_adjust.py) instead of HMA_DEM8m_MOS;
               same blunder filter against Copernicus (|d - median d| > 40 m) as nw_stack.py
  dsm, src     rebuilt from the new hma with the same Copernicus fill (fill_from_cop)
  strip_count  number of adjusted strips per pixel (EXP024)
  strip_nmad   their NMAD, NaN below 2 strips (EXP024)
  chmv2        CHMv2 10 m canopy height averaged onto the 8 m grid (EXP002/EXP022)
cop, wc, chm_mean/q3/max are copied unchanged. Same uint16 encodings as fuse_hma.encode.

Run:
    ../.venv/bin/python -I -u nw_stack_v2.py [--part k/n]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stack_hma import fill_from_cop, warp  # noqa: E402
from fuse_hma import encode, load_stack  # noqa: E402
from nw_stack import ENC  # noqa: E402

NW = Path(__file__).resolve().parent / "data" / "nw"
ENC2 = ENC | {"strip_count": "u8", "strip_nmad": "q4", "chmv2": "q4"}
_CHM_BOUNDS: dict[Path, tuple] = {}


def chmv2_tiles(crs, T, W, H) -> list[Path]:
    """CHMv2 10 m tiles (EPSG:3857) that overlap this stack tile; warping all 12 per tile is slow."""
    from rasterio.warp import transform_bounds
    if not _CHM_BOUNDS:
        for f in sorted((NW / "chmv2").glob("*.tif")):
            with rasterio.open(f) as s:
                _CHM_BOUNDS[f] = transform_bounds(s.crs, crs, *s.bounds)
    x0, y1 = T * (0, 0)
    x1, y0 = T * (W, H)
    return [f for f, b in _CHM_BOUNDS.items() if b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0]


def build(p: Path) -> dict:
    prof, L = load_stack(p)
    T, W, H = prof["transform"], prof["width"], prof["height"]
    with rasterio.open(NW / "adjust" / "out" / p.name) as s:
        hma, cnt, nmad = (s.read(i).astype("float32") for i in (1, 2, 3))
    d = hma - L["cop"]
    if np.isfinite(d).any():
        # EXP030: Copernicus is itself wrong at cliffs, so keep cells where >=2 adjusted strips agree (NMAD < 5 m)
        # (re-admitted cells: 3.75 m RMSE vs 47.8 m for the Copernicus fill on Langtang)
        bad = (np.abs(d - np.nanmedian(d)) > 40) & ~((cnt >= 2) & (nmad < 5))
        hma[bad] = np.nan
        cnt[bad] = 0
        nmad[bad] = np.nan
    L["hma"] = hma
    L["dsm"] = fill_from_cop(hma, L["cop"]) if np.isfinite(hma).sum() > 1000 else L["cop"].copy()
    L["src"] = np.where(np.isfinite(hma), 1, 2).astype("float32")
    L["strip_count"] = np.where(np.isfinite(hma), cnt, 0).astype("float32")
    L["strip_nmad"] = nmad
    L["chmv2"] = warp(chmv2_tiles(prof["crs"], T, W, H), T, W, H, Resampling.average, 255)
    out = NW / "stack_v2" / p.name
    tmp = out.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", driver="GTiff", width=W, height=H, count=len(ENC2), dtype="uint16",
                       crs=prof["crs"], transform=T, nodata=0, compress="deflate", predictor=2, tiled=True,
                       blockxsize=512, blockysize=512) as dst:
        for i, (k, enc) in enumerate(ENC2.items(), 1):
            dst.write(encode(L[k], enc), i)
            dst.set_band_description(i, f"{k}:{enc}")
    tmp.replace(out)
    return {"hma_valid": float(np.isfinite(hma).mean()), "chmv2_valid": float(np.isfinite(L["chmv2"]).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    (NW / "stack_v2").mkdir(exist_ok=True)
    k, n = map(int, a.part.split("/"))
    todo = sorted((NW / "stack").glob("E*N*.tif"))[k::n]
    for i, p in enumerate(todo, 1):
        if (NW / "stack_v2" / p.name).exists() or not (NW / "adjust" / "out" / p.name).exists():
            continue
        t0 = time.time()
        r = build(p)
        print(f"  {i}/{len(todo)} {p.stem}: HMA {r['hma_valid']:.0%}, CHMv2 {r['chmv2_valid']:.0%} "
              f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
