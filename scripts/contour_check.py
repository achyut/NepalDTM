"""Check NepalDTM against the Survey Department's national contour lines, with Copernicus as the baseline.

The contours (code/data/control/survey/geonode_eleva_ln.geojson, 35,054 lines from 0 to 8,320 m, mostly at a
20 or 40 m interval) are independent of everything the model was fitted to, and unlike the ICESat-2 tracks they
cover the whole country. They are map products, not point surveys: a contour carries its own error, roughly
+-10-20 m in the mountains, and a small share of the lines are plainly mislabelled (a few hundred sit on
terrain kilometres away from their stated height). So this ranks NepalDTM against Copernicus GLO-30 on the
same points; it is not an absolute accuracy, and the ICESat-2 holdout in validation_report.json remains the
headline measure.

Because of those bad lines the summary leads with the median and the NMAD, which outliers do not move, and
reports the share of points beyond 50 m rather than letting RMSE speak for the set.

Every contour vertex inside Nepal becomes a check point: points are grouped by 24 km tile for NepalDTM and by
1-degree tile for Copernicus, so each file is read once.

Run from the repository root, after the _dtm tiles are in code/data/nw/out:
    .venv/bin/python -I scripts/contour_check.py [--stride 4]

Writes validation/contour_check.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import shapely
from pyproj import Transformer
from rasterio.windows import from_bounds
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parent.parent
VRT = ROOT / "code" / "data" / "nw" / "nepal_dtm_8m.vrt"
CONTOURS = ROOT / "code" / "data" / "control" / "survey" / "geonode_eleva_ln.geojson"
NEPAL = ROOT / "code" / "data" / "control" / "nepal_boundary.geojson"
OUT = ROOT / "validation" / "contour_check.json"
COP = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"
       "/Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM.tif")
TILE = 24000  # m, the processing grid
BANDS = [(0, 300, "terai <300 m"), (300, 1000, "siwalik 300-1000 m"), (1000, 2500, "hills 1-2.5 km"),
         (2500, 4000, "mountains 2.5-4 km"), (4000, 9000, "high himal >4 km")]


def points(stride):
    """Contour vertices as lon, lat, height, every `stride`-th vertex."""
    fs = json.loads(CONTOURS.read_text())["features"]
    xs, ys, zs = [], [], []
    for f in fs:
        z = float(f["properties"]["cov"])
        g = f["geometry"]
        for ln in (g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]):
            c = np.array([q[:2] for q in ln if len(q) >= 2], dtype="float64")[::stride]
            if len(c):
                xs.append(c[:, 0]); ys.append(c[:, 1]); zs.append(np.full(len(c), z))
    lon, lat, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    print(f"{len(fs):,} contours -> {len(lon):,} points (every {stride} vertices)", flush=True)
    return lon, lat, z


def sample_tiles(path_of, keys, coords, shape_of_key):
    """Sample rasters grouped by key, one windowed read each. coords are in each raster's own CRS."""
    out = np.full(len(coords[0]), np.nan)
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    for n, (k, idx) in enumerate(sorted(groups.items()), 1):
        idx = np.array(idx)
        try:
            with rasterio.open(path_of(k)) as src:
                win = from_bounds(*shape_of_key(k), src.transform).round_offsets().round_lengths()
                arr = src.read(1, window=win, boundless=True, fill_value=np.nan).astype("float64")
                nodata = src.nodata
                T = src.window_transform(win)
        except rasterio.RasterioIOError:
            continue
        if nodata is not None:
            arr[arr == nodata] = np.nan
        col = ((coords[0][idx] - T.c) / T.a).astype(int)
        row = ((coords[1][idx] - T.f) / T.e).astype(int)
        ok = (col >= 0) & (row >= 0) & (col < arr.shape[1]) & (row < arr.shape[0])
        vals = np.full(len(idx), np.nan)
        vals[ok] = arr[row[ok], col[ok]]
        out[idx] = vals
        if n % 25 == 0:
            print(f"  {n}/{len(groups)}", flush=True)
    return out


def stats(d):
    d = d[np.isfinite(d)]
    if len(d) < 10:
        return None
    med = float(np.median(d))
    return {"n": int(len(d)), "median": round(med, 2),
            "nmad": round(float(1.4826 * np.median(np.abs(d - med))), 2),
            "share_beyond_50m": round(float((np.abs(d) > 50).mean()), 4),
            "mean": round(float(d.mean()), 2), "rmse": round(float(np.sqrt((d ** 2).mean())), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4, help="use every Nth contour vertex")
    args = ap.parse_args()

    lon, lat, z = points(args.stride)
    outline = shape(json.loads(NEPAL.read_text())["features"][0]["geometry"])
    keep = shapely.contains_xy(outline, lon, lat)
    lon, lat, z = lon[keep], lat[keep], z[keep]
    print(f"{len(lon):,} points inside Nepal", flush=True)

    with rasterio.open(VRT) as src:
        crs = src.crs
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    print("sampling NepalDTM", flush=True)
    ours = sample_tiles(lambda k: VRT, [((int(a // TILE)), int(b // TILE)) for a, b in zip(x, y)], (x, y),
                        lambda k: (k[0] * TILE, k[1] * TILE, (k[0] + 1) * TILE, (k[1] + 1) * TILE))
    print("sampling Copernicus GLO-30", flush=True)
    with rasterio.Env(GDAL_HTTP_USERAGENT="nepaldtm-contour-check/1.0", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        cop = sample_tiles(lambda k: COP.format(lat=k[1], lon=k[0]),
                           [(int(np.floor(a)), int(np.floor(b))) for a, b in zip(lon, lat)], (lon, lat),
                           lambda k: (k[0], k[1], k[0] + 1, k[1] + 1))

    both = np.isfinite(ours) & np.isfinite(cop)
    print(f"{int(both.sum()):,} points with both models", flush=True)
    do, dc = ours[both] - z[both], cop[both] - z[both]
    zb = z[both]
    report = {
        "description": "Model height minus the Survey Department contour height at every "
                       f"{args.stride}th contour vertex inside Nepal, on the points where both models have a "
                       "value. Metres; negative = model lower. Contours are map products with their own error "
                       "(about +-10-20 m in the mountains) and a few hundred lines are mislabelled, so read the "
                       "median and NMAD, and read this as NepalDTM against Copernicus rather than as an "
                       "absolute accuracy. The ICESat-2 block holdout in validation_report.json is the headline.",
        "generated": datetime.date.today().isoformat(),
        "source": "Survey Department of Nepal, national contour lines (geonode_eleva_ln)",
        "all": {"nepaldtm": stats(do), "copernicus": stats(dc)},
        "by_elevation": {name: {"nepaldtm": stats(do[(zb >= lo) & (zb < hi)]),
                                "copernicus": stats(dc[(zb >= lo) & (zb < hi)])}
                         for lo, hi, name in BANDS},
    }
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\n{'group':24s} {'n':>9s} {'NepalDTM med/NMAD':>22s} {'Copernicus med/NMAD':>22s}")
    for name, g in [("all", report["all"])] + list(report["by_elevation"].items()):
        o, c = g["nepaldtm"], g["copernicus"]
        if o and c:
            print(f"{name:24s} {o['n']:9,d} {o['median']:+9.1f} / {o['nmad']:<9.1f} {c['median']:+9.1f} / {c['nmad']:<9.1f}")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
