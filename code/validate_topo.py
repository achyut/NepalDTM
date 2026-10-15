"""Independent check of DEM/DTM grids against Survey Department topographic data.

- Contours (geonode:eleva_ln, 1:50k maps, 20 m interval): densified to ~4 m spacing and sampled.
- Photogrammetric spot heights (GIID_Topo_ENTM_wgs841, 1994 aerial strips).
Both are on Nepal's mean-sea-level datum; compared directly with EGM2008 grids (difference is ~1 m).
Map contours carry roughly half-interval (~10 m) error in steep terrain, so they test shape and blunders,
not decimetre accuracy.

Run:
    ~/workspace/super-resolution-satellite/.venv/bin/python -I validate_topo.py --aoi bhotekoshi \
        --contours <contours.geojson> --spots ~/workspace/geoportal/wfs_downloads/GIID_Topo_ENTM_wgs841.geojson
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_altimetry import AOIS  # noqa: E402
from fuse_hma import decode  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def densify(coords, step):
    out = []
    for a, b in zip(coords[:-1], coords[1:]):
        n = max(int(np.hypot(*(b - a)) / step), 1)
        t = np.arange(n)[:, None] / n
        out.append(a + t * (b - a))
    out.append(coords[-1:])
    return np.concatenate(out)


def read(path, band):
    """Band by number or by name; compact-stack bands ('name:enc') are decoded to metres."""
    with rasterio.open(path) as s:
        names = [(d or "").split(":") for d in s.descriptions]
        i = int(band) if band.isdigit() else [n[0] for n in names].index(band) + 1
        a = s.read(i)
        enc = names[i - 1][1] if len(names[i - 1]) == 2 else None
        return (decode(a, enc) if enc else a.astype("float32")), s.transform, s.res[0]


def sample(grid, x, y):
    a, T, _ = grid
    r, c = rowcol(T, x, y, op=float)
    return ndimage.map_coordinates(a, [np.asarray(r) - .5, np.asarray(c) - .5], order=1, mode="constant", cval=np.nan)


def stats(e):
    e = e[np.isfinite(e)]
    if not e.size:
        return None
    return dict(n=int(e.size), bias=float(np.mean(e)), median=float(np.median(e)), rmse=float(np.sqrt(np.mean(e ** 2))),
                nmad=float(1.4826 * np.median(np.abs(e - np.median(e)))), le90=float(np.percentile(np.abs(e), 90)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--contours", required=True)
    ap.add_argument("--spots")
    ap.add_argument("--grids", nargs="+", help="name=path:band entries; default: this AOI's cop/dsm/dtm")
    args = ap.parse_args()
    grids = {}
    for g in args.grids or [f"copernicus={DATA}/{args.aoi}_stack8m_c.tif:cop",
                            f"dem_8m={DATA}/{args.aoi}_dem_8m.tif:1", f"dtm_8m={DATA}/{args.aoi}_dtm_8m.tif:1"]:
        name, rest = g.split("=", 1)
        path, band = rest.rsplit(":", 1)
        grids[name] = (path, band)
    with rasterio.open(next(iter(grids.values()))[0]) as s:
        crs = s.crs
    to = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    w, s_, e, n = AOIS[args.aoi]

    fc = json.load(open(args.contours))
    xs, ys, zs = [], [], []
    for f in fc["features"]:
        g = f["geometry"]
        lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        for ln in lines:
            c = np.asarray([p[:2] for p in ln], float)
            if len(c) < 2:
                continue
            x, y = to.transform(c[:, 0], c[:, 1])
            d = densify(np.column_stack([x, y]), 4.0)
            xs.append(d[:, 0]); ys.append(d[:, 1]); zs.append(np.full(len(d), float(f["properties"]["cov"])))
    cx, cy, cz = map(np.concatenate, (xs, ys, zs))
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(cx, cy)
    inside = (lon >= w) & (lon <= e) & (lat >= s_) & (lat <= n)
    cx, cy, cz = cx[inside], cy[inside], cz[inside]
    print(f"contours: {len(fc['features'])} lines -> {cx.size} samples at 4 m spacing")

    report = {"contours": {}, "spots": {}}
    slope = None
    loaded = {name: read(path, b) for name, (path, b) in grids.items()}
    for name, g in loaded.items():
        z = sample(g, cx, cy)
        if slope is None and name.startswith("dem"):
            gy, gx = np.gradient(g[0], g[2])
            slope = sample((np.degrees(np.arctan(np.hypot(gx, gy))), g[1], g[2]), cx, cy)
        report["contours"][name] = {"all": stats(z - cz), "_z": z}
        print(f"  {name:12s} vs contours: {report['contours'][name]['all']}")
    if slope is not None:
        for name in grids:
            z = report["contours"][name].pop("_z")
            for lab, m in (("slope<20", slope < 20), ("slope20-40", (slope >= 20) & (slope < 40)), ("slope>=40", slope >= 40)):
                report["contours"][name][lab] = stats((z - cz)[m])
                st = report["contours"][name][lab]
                if st:
                    print(f"    {name:12s} {lab:10s} n={st['n']:7d} median={st['median']:+6.2f} nmad={st['nmad']:6.2f} rmse={st['rmse']:6.2f}")
    else:
        for name in grids:
            report["contours"][name].pop("_z", None)

    if args.spots:
        sp = json.load(open(args.spots))["features"]
        pts = np.array([[*f["geometry"]["coordinates"][:2], f["properties"]["Elevation"]] for f in sp], float)
        m = (pts[:, 0] >= w) & (pts[:, 0] <= e) & (pts[:, 1] >= s_) & (pts[:, 1] <= n)
        px, py = to.transform(pts[m, 0], pts[m, 1])
        print(f"spot heights inside AOI: {m.sum()}")
        for name, g in loaded.items():
            report["spots"][name] = stats(sample(g, px, py) - pts[m, 2])
            print(f"  {name:12s} vs spot heights: {report['spots'][name]}")

    out = DATA / f"{args.aoi}_topo_validation.json"
    out.write_text(json.dumps(report, indent=2))
    print("->", out)


if __name__ == "__main__":
    main()
