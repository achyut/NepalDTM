"""Build site/data/summits.geojson: the mountain summits in NepalDTM, for the map's Show peaks layer.

1. The 8 m bare-earth mosaic (code/data/nw/nepal_dtm_8m.vrt) is reduced to a 96 m grid, keeping the highest
   8 m cell in each block, so summits keep their height.
2. A summit is a regional maximum that stands at least 100 m above the highest col (saddle) connecting it to
   higher ground: its topographic prominence. The h-maxima transform finds exactly these; repeating it at
   200, 300, 500, 1,000, 1,500 and 2,000 m gives each summit a prominence class ("at least X m").
   Without the threshold every rock and terrace edge would count as a summit.
3. Each summit is placed on the highest 8 m cell around it, with that cell's bare-earth height.
4. Summits inside Nepal or within 1 km of its border are kept, so the peaks on the border ridge (Everest,
   Makalu, Cho Oyu, Kangchenjunga) are included. Where an OpenStreetMap natural=peak with a name lies within 300 m, its name
   and recorded height are attached (data (c) OpenStreetMap contributors, ODbL).

Properties: h (NepalDTM bare-earth height, m), prom (prominence of at least this many m), name and osm
(OpenStreetMap name and height) where matched.

Run from the repository root after downloading the _dtm tiles:
    .venv/bin/python -I scripts/summits.py [--reuse]

Steps 1-2 take about 20 minutes and are saved to code/data/nw/summit_candidates.npz; --reuse starts from that file.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape
from shapely.ops import transform as reproject
from shapely.prepared import prep
from skimage.morphology import h_maxima

ROOT = Path(__file__).resolve().parent.parent
VRT = ROOT / "code" / "data" / "nw" / "nepal_dtm_8m.vrt"
NEPAL = ROOT / "code" / "data" / "control" / "nepal_boundary.geojson"
OUT = ROOT / "site" / "data" / "summits.geojson"
CANDIDATES = ROOT / "code" / "data" / "nw" / "summit_candidates.npz"
BORDER = 1000  # m: keep summits this close outside Nepal (border peaks)
BLOCK = 12  # 8 m cells per side of a grid cell: 96 m
TILE = 3000  # cells per side of a 24 km tile
PROMINENCE = [100, 200, 300, 500, 1000, 1500, 2000]
NAME_RADIUS = 300  # m
QUERY = '[out:json][timeout:120];area["ISO3166-1"="NP"][admin_level=2]->.np;node["natural"="peak"]["name"](area.np);out body;'


def coarse_max(src):
    """Highest 8 m cell in each 96 m block, tile by tile."""
    g = np.full((src.height // BLOCK, src.width // BLOCK), np.nan, dtype="float32")
    k = TILE // BLOCK
    for row in range(0, src.height, TILE):
        for col in range(0, src.width, TILE):
            a = src.read(1, window=Window(col, row, TILE, TILE))
            if np.isfinite(a).any():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    g[row // BLOCK:row // BLOCK + k, col // BLOCK:col // BLOCK + k] = \
                        np.nanmax(a.reshape(k, BLOCK, k, BLOCK), axis=(1, 3))
    return g


def osm_peaks():
    req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                                 data=urllib.parse.urlencode({"data": QUERY}).encode(),
                                 headers={"User-Agent": "nepaldtm-summits/1.0 (https://github.com/achyut/NepalDTM)"})
    out = []
    for n in json.load(urllib.request.urlopen(req, timeout=180))["elements"]:
        t = n["tags"]
        m = re.search(r"\d+(\.\d+)?", (t.get("ele") or "").replace(",", ""))
        ele = float(m.group()) if m and 50 <= float(m.group()) <= 8900 else None
        out.append((n["lon"], n["lat"], t.get("name:en") or t["name"], ele))
    return out


def candidates(src):
    """Summits on the 96 m grid: (row, col) of each and its prominence class."""
    t0 = time.time()
    g = coarse_max(src)
    print(f"96 m grid {g.shape[1]} x {g.shape[0]} in {time.time() - t0:.0f} s", flush=True)
    dem = np.where(np.isfinite(g), g, -1000.0).astype("float64")
    labels, n = ndimage.label(h_maxima(dem, PROMINENCE[0]).astype(bool), structure=np.ones((3, 3)))
    idx = np.arange(1, n + 1)
    prom = np.full(n, PROMINENCE[0])
    for h in PROMINENCE[1:]:
        survived = np.asarray(ndimage.maximum(h_maxima(dem, h).astype(bool), labels, idx)) > 0
        prom[survived] = h
        print(f"  prominence >= {h} m: {int(survived.sum())}", flush=True)
    rc = np.array(ndimage.maximum_position(dem, labels, idx))
    print(f"{n} summits with prominence >= {PROMINENCE[0]} m ({time.time() - t0:.0f} s)", flush=True)
    np.savez(CANDIDATES, rc=rc, prom=prom)
    return rc, prom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help=f"start from {CANDIDATES.name}")
    args = ap.parse_args()
    t0 = time.time()
    with rasterio.open(VRT) as src:
        if args.reuse:
            saved = np.load(CANDIDATES)
            rc, prom = saved["rc"], saved["prom"]
        else:
            rc, prom = candidates(src)
        peaks = [tuple(p) for p in rc]

        # Place each on the highest 8 m cell of its 96 m block and the eight around it.
        to_ll = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform
        to_utm = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
        nepal = prep(reproject(to_utm, shape(json.load(open(NEPAL))["features"][0]["geometry"])).buffer(BORDER))
        summits = []
        for (r, c), p in zip(peaks, prom):
            r0, c0 = max(0, (r - 1) * BLOCK), max(0, (c - 1) * BLOCK)
            w = src.read(1, window=Window(c0, r0, 3 * BLOCK, 3 * BLOCK), boundless=True, fill_value=np.nan)
            if not np.isfinite(w).any():
                continue
            i, j = np.unravel_index(np.nanargmax(w), w.shape)
            x, y = src.xy(r0 + i, c0 + j)
            lon, lat = to_ll(x, y)
            if nepal.contains(Point(x, y)):
                summits.append({"x": x, "y": y, "lon": lon, "lat": lat, "h": float(w[i, j]), "prom": int(p)})
        print(f"{len(summits)} inside Nepal or within {BORDER} m of its border", flush=True)

        # One OpenStreetMap name per summit, and each named peak used once: nearest pairs first.
        names = osm_peaks()
        tree = cKDTree([to_utm(lon, lat) for lon, lat, _, _ in names])
        d, k = tree.query([(s["x"], s["y"]) for s in summits], distance_upper_bound=NAME_RADIUS)
        used = set()
        for si in np.argsort(d):
            if np.isfinite(d[si]) and k[si] not in used:
                used.add(k[si])
                _, _, name, ele = names[k[si]]
                summits[si]["name"] = name
                if ele is not None:
                    summits[si]["osm"] = ele
        print(f"{len(used)} of {len(names)} named OpenStreetMap peaks matched within {NAME_RADIUS} m", flush=True)

    summits.sort(key=lambda s: -s["h"])
    features = [{"type": "Feature",
                 "properties": {"h": round(s["h"]), "prom": s["prom"],
                                **({"name": s["name"]} if "name" in s else {}), **({"osm": s["osm"]} if "osm" in s else {})},
                 "geometry": {"type": "Point", "coordinates": [round(s["lon"], 5), round(s["lat"], 5)]}} for s in summits]
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":"), ensure_ascii=False))
    print(f"{len(features)} summits -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} KB) in {time.time() - t0:.0f} s")
    for f in features[:8]:
        print("  ", f["properties"])


if __name__ == "__main__":
    main()
