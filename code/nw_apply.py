"""Apply the national model to every tile stack and write the 24 km core of each tile.

Per tile (data/nw/out/<id>_*.tif, COG, EPSG:32645, 8 m, heights EGM2008 m):
  _dem.tif   DSM (surface)                                    float32, LERC max error 1 cm
  _dtm.tif   DTM (bare earth) = DSM - predicted residual        float32, LERC max error 1 cm
  _aux.tif   band 1 source (1 HMA 8 m, 2 Copernicus 30 m fill), band 2 expected |error| in dm (255 = >= 25.5 m)
Then data/nw/nepal_{dem,dtm,aux}_8m.vrt mosaics. Tiles already written are skipped.

Run:
    OMP_NUM_THREADS=2 ~/workspace/super-resolution-satellite/.venv/bin/python -I nw_apply.py --part 0/2
    ~/workspace/super-resolution-satellite/.venv/bin/python -I nw_apply.py --vrt-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuse_hma import EXTRA, FEATURES, STACK_DIR, SUFFIX, load_stack, terrain  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
NW = DATA / "nw"
BUFPX = 125  # 1 km buffer at 8 m
CORE = 3000
OUT = f"out{SUFFIX}"


def cog(path, prof, a, dtype, **opt):
    p = dict(driver="COG", width=CORE, height=CORE, count=a.shape[0], dtype=dtype, crs=prof["crs"],
             transform=prof["transform"] * prof["transform"].translation(BUFPX, BUFPX), BLOCKSIZE=512,
             OVERVIEWS="AUTO", **opt)
    tmp = path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **p) as dst:
        dst.write(a.astype(dtype))
    tmp.replace(path)


def apply(path, m, u):
    prof, L = load_stack(path)
    F = {**terrain(L["dsm"]), "chm_mean": L["chm_mean"], "chm_q3": L["chm_q3"], "chm_max": L["chm_max"],
         "wc": L["wc"], "dsm_minus_cop": L["dsm"] - L["cop"], "src": L["src"], "dsm": L["dsm"],
         **{k: L[k] for k in EXTRA}}
    c = (slice(BUFPX, BUFPX + CORE), slice(BUFPX, BUFPX + CORE))
    G = np.column_stack([F[k][c].ravel() for k in FEATURES]).astype("float32")
    v = np.all(np.isfinite(G[:, [i for i, k in enumerate(FEATURES) if k not in EXTRA]]), 1)  # v2 layers may be NaN
    corr = np.full(G.shape[0], np.nan, "float32")
    unc = np.full(G.shape[0], np.nan, "float32")
    for a in range(0, G.shape[0], 2_000_000):
        sl = slice(a, a + 2_000_000)
        idx = np.flatnonzero(v[sl]) + a
        if idx.size:
            corr[idx] = m.predict(G[idx]); unc[idx] = u.predict(G[idx])
    corr, unc = corr.reshape(CORE, CORE), np.maximum(unc.reshape(CORE, CORE), 0)
    dsm = L["dsm"][c]
    dtm = dsm - corr
    out = NW / OUT
    lerc = dict(compress="LERC_DEFLATE", MAX_Z_ERROR=0.01, nodata=np.nan)
    cog(out / f"{path.stem}_dem.tif", prof, dsm[None], "float32", **lerc)
    cog(out / f"{path.stem}_dtm.tif", prof, dtm[None], "float32", **lerc)
    err = np.where(np.isfinite(unc), np.clip(np.round(unc * 10), 0, 254), 255)
    src = np.where(np.isfinite(dsm), L["src"][c], 0)
    cog(out / f"{path.stem}_aux.tif", prof, np.stack([src, err]), "uint8", compress="deflate", nodata=0)
    return float(np.isfinite(dtm).mean()), float(np.nanmean(corr)) if np.isfinite(corr).any() else np.nan


def vrt():
    """National mosaics as VRT XML (no GDAL CLI here). All tiles share one aligned 8 m grid, so each is a plain
    offset; relative paths keep the VRTs valid if data/nw is moved."""
    for kind in ("dem", "dtm", "aux"):
        files = sorted((NW / OUT).glob(f"*_{kind}.tif"))
        if not files:
            continue
        info = []
        for f in files:
            with rasterio.open(f) as s:
                info.append((f, s.transform.c, s.transform.f, s.count, s.dtypes[0], s.nodata, s.crs.to_wkt()))
        x0, y1 = min(i[1] for i in info), max(i[2] for i in info)
        x1 = max(i[1] for i in info) + CORE * 8
        y0 = min(i[2] for i in info) - CORE * 8
        W, H = int(round((x1 - x0) / 8)), int(round((y1 - y0) / 8))
        dt = {"float32": "Float32", "uint8": "Byte"}[info[0][4]]
        nod = "nan" if info[0][5] is None or np.isnan(info[0][5]) else str(int(info[0][5]))
        bands = []
        for b in range(1, info[0][3] + 1):
            src = "".join(
                f'<SimpleSource><SourceFilename relativeToVRT="1">{OUT}/{f.name}</SourceFilename><SourceBand>{b}</SourceBand>'
                f'<SrcRect xOff="0" yOff="0" xSize="{CORE}" ySize="{CORE}"/>'
                f'<DstRect xOff="{int(round((cx - x0) / 8))}" yOff="{int(round((y1 - cy) / 8))}" xSize="{CORE}" ySize="{CORE}"/>'
                f"</SimpleSource>" for f, cx, cy, *_ in info)
            bands.append(f'<VRTRasterBand dataType="{dt}" band="{b}"><NoDataValue>{nod}</NoDataValue>{src}</VRTRasterBand>')
        xml = (f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}"><SRS>{info[0][6]}</SRS>'
               f"<GeoTransform>{x0}, 8, 0, {y1}, 0, -8</GeoTransform>{''.join(bands)}</VRTDataset>")
        (NW / f"nepal_{kind}_8m{SUFFIX}.vrt").write_text(xml)
        print(f"-> nepal_{kind}_8m{SUFFIX}.vrt ({len(files)} tiles, {W} x {H} px)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--vrt-only", action="store_true")
    args = ap.parse_args()
    if not args.vrt_only:
        (NW / OUT).mkdir(exist_ok=True)
        m = joblib.load(NW / f"model_residual{SUFFIX}.joblib")
        u = joblib.load(NW / f"model_error{SUFFIX}.joblib")
        k, n = map(int, args.part.split("/"))
        stacks = sorted((NW / STACK_DIR).glob("E*N*.tif"))[k::n]
        for i, p in enumerate(stacks, 1):
            if (NW / OUT / f"{p.stem}_aux.tif").exists():
                continue
            valid, mc = apply(p, m, u)
            print(f"  {i}/{len(stacks)} {p.stem}: valid {valid:.0%}, mean removed {mc:+.2f} m", flush=True)
    vrt()


if __name__ == "__main__":
    main()
