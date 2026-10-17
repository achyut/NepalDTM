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

5. Each summit gets the source of its 8 m cell from the _aux tiles (read from R2): 1 = HMA 8 m stereo,
   2 = Copernicus 30 m fill.
6. Summits with an OpenStreetMap height are compared with it (most come from published surveys), overall and by
   source, and the result is written to validation/summit_check.json.

Properties: h (NepalDTM bare-earth height, m), prom (prominence of at least this many m), src (1 HMA 8 m,
2 Copernicus fill), name and osm (OpenStreetMap name and height) where matched.

Run from the repository root after downloading the _dtm tiles:
    .venv/bin/python -I scripts/summits.py [--reuse]

Steps 1-2 take about 20 minutes and are saved to code/data/nw/summit_candidates.npz; --reuse starts from that file.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import time
import urllib.parse
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from scipy import ndimage
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import Point, shape
from shapely.prepared import prep
from skimage.morphology import h_maxima

ROOT = Path(__file__).resolve().parent.parent
VRT = ROOT / "code" / "data" / "nw" / "nepal_dtm_8m.vrt"
NEPAL = ROOT / "code" / "data" / "control" / "nepal_boundary.geojson"
OUT = ROOT / "site" / "data" / "summits.geojson"
CANDIDATES = ROOT / "code" / "data" / "nw" / "summit_candidates.npz"
CHECK = ROOT / "validation" / "summit_check.json"
AUX = "/vsicurl/https://nepaldtm-data.achyutpdl.com/cog/{}_aux.tif"
BORDER = 1000  # m: keep summits this close outside Nepal (border peaks)
BLOCK = 12  # 8 m cells per side of a grid cell: 96 m
TILE = 3000  # cells per side of a 24 km tile
PROMINENCE = [100, 200, 300, 500, 1000, 1500, 2000]
NAME_RADIUS = 300  # m
# Named peaks in a box around Nepal, so border peaks mapped just across the line (Kangchenjunga) still match.
QUERY = '[out:json][timeout:120];node["natural"="peak"]["name"](26.2,79.9,30.6,88.4);out body;'


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


def tile_id(x, y):
    """The 24 km processing tile holding a UTM 45N point, named as nw_tiles.py names it."""
    return f"E{int(x // 24000 * 24):04d}N{int(y // 24000 * 24):04d}"


def sources(summits):
    """Band 1 of the _aux tile at each summit: 1 HMA 8 m stereo, 2 Copernicus 30 m fill (0 if unreadable)."""
    by_tile = {}
    for i, s in enumerate(summits):
        by_tile.setdefault(tile_id(s["x"], s["y"]), []).append(i)
    out = [0] * len(summits)

    def read(tile):
        try:
            with rasterio.open(AUX.format(tile)) as src:
                for i in by_tile[tile]:
                    r, c = src.index(summits[i]["x"], summits[i]["y"])
                    out[i] = int(src.read(1, window=Window(c, r, 1, 1))[0, 0])
        except rasterio.RasterioIOError:
            pass

    with rasterio.Env(GDAL_HTTP_USERAGENT="nepaldtm-summits/1.0", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), \
            ThreadPoolExecutor(8) as pool:
        list(pool.map(read, by_tile))
    return out


def summit_check(summits):
    """NepalDTM summit heights against OpenStreetMap heights, overall and by source."""
    def stats(rows):
        d = np.array([s["h"] - s["osm"] for s in rows])
        return {"n": len(d), "median": round(float(np.median(d)), 1), "median_abs": round(float(np.median(np.abs(d))), 1),
                "rmse": round(float(np.sqrt(np.mean(d ** 2))), 1), "share_more_than_50m_low": round(float(np.mean(d < -50)), 3)}
    rows = [s for s in summits if "osm" in s]
    return {
        "description": "NepalDTM bare-earth height of each summit (highest 8 m cell) minus the OpenStreetMap height of "
                       f"the named peak within {NAME_RADIUS} m. Metres; negative = NepalDTM lower.",
        "generated": datetime.date.today().isoformat(),
        "all": stats(rows),
        "hma_8m": stats([s for s in rows if s["src"] == 1]),
        "copernicus_fill": stats([s for s in rows if s["src"] == 2]),
    }


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
        outline = shape(json.loads(NEPAL.read_text())["features"][0]["geometry"])
        nepal = prep(shapely.transform(outline, lambda c: np.column_stack(to_utm(c[:, 0], c[:, 1]))).buffer(BORDER))
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

    for s, src in zip(summits, sources(summits)):
        s["src"] = src
    print(f"source read for {sum(s['src'] > 0 for s in summits)} of {len(summits)} summits", flush=True)
    check = summit_check(summits)
    CHECK.write_text(json.dumps(check, indent=1) + "\n")
    print(f"summit check -> {CHECK.relative_to(ROOT)}: {json.dumps({k: check[k] for k in ('all', 'hma_8m', 'copernicus_fill')})}", flush=True)

    summits.sort(key=lambda s: -s["h"])
    features = [{"type": "Feature",
                 "properties": {"h": round(s["h"]), "prom": s["prom"], "src": s["src"],
                                **({"name": s["name"]} if "name" in s else {}), **({"osm": s["osm"]} if "osm" in s else {})},
                 "geometry": {"type": "Point", "coordinates": [round(s["lon"], 5), round(s["lat"], 5)]}} for s in summits]
    # The check rides along (a GeoJSON foreign member, ignored by map libraries) so the map can quote it.
    OUT.write_text(json.dumps({"type": "FeatureCollection", "check": check, "features": features},
                              separators=(",", ":"), ensure_ascii=False))
    print(f"{len(features)} summits -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} KB) in {time.time() - t0:.0f} s")
    for f in features[:8]:
        print("  ", f["properties"])


if __name__ == "__main__":
    main()
