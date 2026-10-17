"""Are the Survey Department contours more accurate than NepalDTM? Judge both against ICESat-2.

The contours are an official product, so it is worth asking whether the tiles should be corrected towards
them. That only makes sense if the contours are the more accurate of the two, and neither the contours nor
the model can settle that about itself. ICESat-2 can: ATL08 measures the ground directly with a laser, to
well under a metre on gentle ground, and it is the reference the model's 6.10 m headline is already measured
against.

Every ICESat-2 ground return that falls within --radius metres of a contour vertex is a place where both
sources claim a height for nearly the same spot. Comparing each against the laser gives their errors on the
same ground. The horizontal offset costs height on a slope (12 m across a 30-degree slope is about 7 m), so
results are reported by slope band and the gentlest band is the cleanest read.

    .venv/bin/python -I scripts/contour_vs_icesat2.py [--radius 12]

Writes validation/contour_vs_icesat2.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
VRT = ROOT / "code" / "data" / "nw" / "nepal_dtm_8m.vrt"
CONTOURS = ROOT / "code" / "data" / "control" / "survey" / "geonode_eleva_ln.geojson"
ALT = ROOT / "code" / "data" / "nw" / "alt"
OUT = ROOT / "validation" / "contour_vs_icesat2.json"
SLOPE_BANDS = [(0, 5, "slope <5 deg"), (5, 15, "slope 5-15 deg"), (15, 30, "slope 15-30 deg"), (30, 90, "slope >=30 deg")]


def contour_points(crs):
    """Every contour vertex as grid x, y and its stated height."""
    fs = json.loads(CONTOURS.read_text())["features"]
    xs, ys, zs = [], [], []
    for f in fs:
        z = float(f["properties"]["cov"])
        g = f["geometry"]
        for ln in (g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]):
            c = np.array([q[:2] for q in ln if len(q) >= 2], dtype="float64")
            if len(c):
                xs.append(c[:, 0]); ys.append(c[:, 1]); zs.append(np.full(len(c), z))
    lon, lat, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    print(f"{len(fs):,} contours -> {len(x):,} vertices", flush=True)
    return np.column_stack([x, y]), z


def icesat2(crs):
    """Good-quality ATL08 ground returns: the same filter the model's training used.

    ATL08 heights are above the WGS84 ellipsoid; the tiles are above the EGM2008 geoid, which in Nepal sits
    39 m (east) to 59 m (west) higher. Without this the comparison carries both a 45 m offset and the 20 m
    spread of that difference across the country.
    """
    xs, ys, hs, sl = [], [], [], []
    to_grid = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    to_geoid = Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True).transform
    for p in sorted(ALT.glob("*_atl08.npz")):
        a = np.load(p)
        ok = (a["h"] < 1e30) & (a["snow"] == 1) & a["strong"].astype(bool) & (a["msw"] <= 1)
        if not ok.any():
            continue
        lon, lat = a["lon"][ok], a["lat"][ok]
        _, _, h = to_geoid(lon, lat, a["h"][ok])
        x, y = to_grid(lon, lat)
        xs.append(x); ys.append(y); hs.append(h); sl.append(a["slope"][ok])
    x, y, h, s = map(np.concatenate, (xs, ys, hs, sl))
    s = np.degrees(np.arctan(np.abs(s)))  # ATL08 terrain_slope is a gradient
    good = np.isfinite(h)
    print(f"{int(good.sum()):,} ICESat-2 ground returns (EGM2008)", flush=True)
    return np.column_stack([x, y])[good], h[good], s[good]


def stats(d):
    d = d[np.isfinite(d)]
    if len(d) < 30:
        return None
    med = float(np.median(d))
    return {"n": int(len(d)), "median": round(med, 2),
            "nmad": round(float(1.4826 * np.median(np.abs(d - med))), 2),
            "rmse": round(float(np.sqrt((d ** 2).mean())), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=12.0, help="metres: how close a contour vertex must be")
    args = ap.parse_args()

    with rasterio.open(VRT) as src:
        crs = src.crs
    cxy, cz = contour_points(crs)
    pxy, ph, pslope = icesat2(crs)

    print("matching", flush=True)
    tree = cKDTree(cxy)
    dist, idx = tree.query(pxy, distance_upper_bound=args.radius, workers=-1)
    hit = np.isfinite(dist)
    print(f"{int(hit.sum()):,} ICESat-2 returns within {args.radius:g} m of a contour vertex", flush=True)

    pxy, ph, pslope, cz_hit = pxy[hit], ph[hit], pslope[hit], cz[idx[hit]]

    # NepalDTM at the same points.
    with rasterio.open(VRT) as src:
        ours = np.array([v[0] for v in src.sample(map(tuple, pxy))], dtype="float64")

    d_contour, d_model = cz_hit - ph, ours - ph
    both = np.isfinite(d_contour) & np.isfinite(d_model)
    d_contour, d_model, pslope = d_contour[both], d_model[both], pslope[both]

    report = {
        "description": f"Height minus the ICESat-2 ATL08 ground return, in metres, at every good-quality return "
                       f"within {args.radius:g} m of a Survey Department contour vertex. The contour and NepalDTM "
                       "are judged on the same ground, so the smaller spread is the more accurate source. Part of "
                       "each spread is the horizontal offset between the return and the vertex, which costs more "
                       "height the steeper the ground, so the gentlest slope band is the cleanest comparison. "
                       "ICESat-2 ellipsoid heights are converted to EGM2008, the datum of the tiles and contours.",
        "generated": datetime.date.today().isoformat(),
        "radius_m": args.radius,
        "all": {"contour": stats(d_contour), "nepaldtm": stats(d_model)},
        "by_slope": {name: {"contour": stats(d_contour[(pslope >= lo) & (pslope < hi)]),
                            "nepaldtm": stats(d_model[(pslope >= lo) & (pslope < hi)])}
                     for lo, hi, name in SLOPE_BANDS},
    }
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\n{'group':20s} {'n':>9s} {'contour med/NMAD':>20s} {'NepalDTM med/NMAD':>20s}")
    for name, g in [("all", report["all"])] + list(report["by_slope"].items()):
        c, o = g["contour"], g["nepaldtm"]
        if c and o:
            print(f"{name:20s} {c['n']:9,d} {c['median']:+8.1f} / {c['nmad']:<8.1f} {o['median']:+8.1f} / {o['nmad']:<8.1f}")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
