"""Compare G with public and earlier terrain models on the Chitwan test boxes.

Every product is scored on the same held-out ICESat-2 ground photons (tracks never used to build v2), on the
photons where all products have a value. Heights are EGM2008 (GEM-15 converted from EGM96).
Products: Copernicus GLO-30, FABDEM V1-2, GEDTM30 (when its host serves it), GEM-15 DTM, NepalDTM v1,
NepalDTM v2 (M), bicubic 1 m, G 1 m.
Out: data/gen/apply/compare.json.  Run: python -u g/g_compare.py (from code/)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "YES")
from build_stack_hma import COP, CRS, warp  # noqa: E402
from fetch_external_dems import GEDTM30  # noqa: E402
from gem15_fill import egm96_to_08, gem_sample  # noqa: E402
from nw_validate import FABDEM_COG  # noqa: E402
from validate_v1_v2 import NW, VRT, sample, window  # noqa: E402

OUT = ROOT / "data" / "gen" / "apply"
BOXES = tuple(sys.argv[1:]) or ("chitwan_fast_s24", "churia_fast_s24")  # apply runs of the final model


def externals(T, W, H) -> dict:
    """Copernicus GLO-30, FABDEM and GEDTM30 bilinear on the 8 m grid (as validate_v1_v2.externals). GEDTM30 is
    skipped if its host fails, so one flaky server does not block the others."""
    w, s, e, n = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform_bounds(*(T * (0, H)), *(T * (W, 0)))
    cells = [(la, lo) for la in range(math.floor(s), math.floor(n) + 1) for lo in range(math.floor(w), math.floor(e) + 1)]
    out = {"copernicus": warp([COP.format(lat=la, lon=lo) for la, lo in cells], T, W, H, Resampling.bilinear)}
    tile = lambda la, lo: f"N{la // 10 * 10:02d}E{lo // 10 * 10:03d}-N{la // 10 * 10 + 10:02d}E{lo // 10 * 10 + 10:03d}"
    out["fabdem"] = warp(["/vsicurl/" + FABDEM_COG.format(z=tile(la, lo), la=la, lo=lo) for la, lo in cells],
                         T, W, H, Resampling.bilinear)
    try:
        g = np.full((H, W), np.nan, "float32")
        with rasterio.open("/vsicurl/" + GEDTM30) as src:
            reproject(rasterio.band(src, 1), g, dst_transform=T, dst_crs=CRS, resampling=Resampling.bilinear,
                      src_nodata=src.nodata, dst_nodata=np.nan)
        out["gedtm30"] = g
    except Exception as ex:
        print(f"  GEDTM30 skipped: {ex}", flush=True)
    return out


def stats(e):
    md = np.median(e)
    return {"n": int(e.size), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "nmad": float(1.4826 * np.median(np.abs(e - md))), "median": float(md)}


def main():
    res = {}
    to = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    for name in BOXES:
        rep = json.loads((OUT / f"{name}_report.json").read_text())
        b = rep["bbox"]
        d = []
        for f in sorted((NW / "atl03").glob("*.parquet")):
            p = pd.read_parquet(f, columns=["lon", "lat", "h", "holdout"])
            p = p[p.holdout.astype(bool) & p.lon.between(b[0], b[2]) & p.lat.between(b[1], b[3])]
            if len(p):
                d.append(p)
        d = pd.concat(d)
        lon, lat, h = d.lon.values, d.lat.values, d.h.values.astype("float32")
        x, y = to.transform(lon, lat)
        with rasterio.open(OUT / f"{name}_v2_8m.tif") as s:
            T8, W8, H8, bounds = s.transform, s.width, s.height, tuple(s.bounds)
        Z = {k: sample(v, T8, x, y) for k, v in externals(T8, W8, H8).items()}
        lo, la = int(np.floor(np.mean(lon))), int(np.floor(np.mean(lat)))
        Z["gem15"] = gem_sample(NW / "gem15" / f"w{lo}s{la}_dtm.tif", lon, lat) + egm96_to_08(lon, lat)
        for v in ("v1", "v2"):
            a, T = window(VRT[v]["dtm"], bounds)
            Z[f"nepaldtm_{v}"] = sample(a, T, x, y)
        for k, f in (("bicubic_1m", "bicubic_1m"), ("g_1m", "G_1m")):
            with rasterio.open(OUT / f"{name}_{f}.tif") as s:
                Z[k] = sample(s.read(1).astype("float32"), s.transform, x, y)
        E = {k: z - h for k, z in Z.items()}
        m = np.all([np.isfinite(e) for e in E.values()], 0)
        res[name] = {"bbox": b, "g_checkpoint_step": rep["step"], "photons_in_box": int(h.size),
                     "scores": {k: stats(e[m]) for k, e in E.items()}}
        for k, s_ in res[name]["scores"].items():
            print(f"{name:8s} {k:14s} RMSE {s_['rmse']:6.2f} NMAD {s_['nmad']:5.2f} "
                  f"median {s_['median']:+6.2f} n={s_['n']:,}", flush=True)
    (OUT / "compare.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
