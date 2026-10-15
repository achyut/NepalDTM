"""Nationwide processing grid: 24 km tiles (3000 x 3000 px at 8 m) on EPSG:32645, built with a 1 km buffer so
terrain features and gap filling have context; only the core is kept in the national mosaic.

West Nepal lies in UTM 44N; using 45N everywhere keeps one seamless grid (scale error <= 0.5% at 80 deg E).
Writes data/nw/tiles.json: id, core bounds (m), lon/lat bounds, HMA 8 m mosaic tiles that intersect, share of
the tile inside Nepal.

Run:
    ~/workspace/super-resolution-satellite/.venv/bin/python -I nw_tiles.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, box, shape
from shapely.ops import transform as stransform

DATA = Path(__file__).resolve().parent / "data"
NEPAL = DATA / "control" / "nepal_boundary.geojson"
CRS = "EPSG:32645"
SIZE = 24000
BUF = 1000


def hma_footprints():
    r = subprocess.run(["curl", "-s", "https://cmr.earthdata.nasa.gov/search/granules.json?short_name=HMA_DEM8m_MOS"
                        "&bounding_box=79.5,26.0,88.7,31.0&page_size=200"], capture_output=True, text=True, check=True)
    out = {}
    for g in json.loads(r.stdout)["feed"]["entry"]:
        p = list(map(float, g["polygons"][0][0].split()))
        out[int(g["title"].split("tile-")[1].split(".")[0])] = Polygon(zip(p[1::2], p[0::2]))
    return out


def main() -> None:
    to_m = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform
    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform
    nep_ll = shape(json.load(open(NEPAL))["features"][0]["geometry"])
    nep = stransform(to_m, nep_ll)
    hma = hma_footprints()
    x0, y0, x1, y1 = nep.bounds
    tiles = []
    for x in np.arange(np.floor(x0 / SIZE) * SIZE, x1, SIZE):
        for y in np.arange(np.floor(y0 / SIZE) * SIZE, y1, SIZE):
            core = box(x, y, x + SIZE, y + SIZE)
            if not nep.intersects(core):
                continue
            ll = stransform(to_ll, core.buffer(BUF, join_style=2))
            tiles.append(dict(id=f"E{int(x / 1000):04d}N{int(y / 1000):04d}", core=[x, y, x + SIZE, y + SIZE],
                              lonlat=list(ll.bounds), nepal_share=round(nep.intersection(core).area / core.area, 4),
                              hma=sorted(k for k, p in hma.items() if p.intersects(ll))))
    (DATA / "nw").mkdir(parents=True, exist_ok=True)
    (DATA / "nw" / "tiles.json").write_text(json.dumps(tiles, indent=1))
    area = sum(t["nepal_share"] for t in tiles) * SIZE ** 2 / 1e6
    print(f"{len(tiles)} tiles covering {area:,.0f} km2 of Nepal; {sum(not t['hma'] for t in tiles)} with no HMA tile")


if __name__ == "__main__":
    main()
