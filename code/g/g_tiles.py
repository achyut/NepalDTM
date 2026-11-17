#!/usr/bin/env python3
"""G 1 m test tiles: zooms 15-17 over the generated boxes, as a small lossless-WebP Terrarium PMTiles overlay.

Zoom 17 is ~1.06 m per pixel over Chitwan, the finest the 1 m product supports. Each zoom-17 tile is reprojected
(bilinear) from the G COGs; zooms 16 and 15 are 2 x 2 means of the zoom below. Encoding and archive layout match
nw_tiles_rgb.py, so the website decodes these tiles with the same code.

Run: ../.venv/bin/python -I -u g/g_tiles.py chitwan churia
Writes data/gen/apply/NepalDTM_G_test.webp.pmtiles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nw_tiles_rgb import SIZE, encode, halve, merc_bounds, tile_xy  # noqa: E402

APPLY = Path(__file__).resolve().parent.parent / "data" / "gen" / "apply"
OUT = APPLY / "NepalDTM_G_test.webp.pmtiles"
Z0, Z1 = 15, 17


def main(names):
    cogs = [APPLY / f"{n}_G_1m.tif" for n in names]
    tiles, boxes = {}, []
    for cog in cogs:
        with rasterio.open(cog) as src:
            lon0, lat0, lon1, lat1 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            boxes.append((lon0, lat0, lon1, lat1))
            x0, y0 = tile_xy(lon0, lat1, Z1)
            x1, y1 = tile_xy(lon1, lat0, Z1)
            # interior tiles only: a tile half outside the box would mix G with nodata at its edge
            for x in range(x0 + 1, x1):
                for y in range(y0 + 1, y1):
                    a = np.full((SIZE, SIZE), np.nan, "float32")
                    reproject(rasterio.band(src, 1), a, src_nodata=np.nan, dst_nodata=np.nan,
                              dst_transform=from_bounds(*merc_bounds(Z1, x, y), SIZE, SIZE), dst_crs="EPSG:3857",
                              resampling=Resampling.bilinear)
                    if np.isfinite(a).all():
                        tiles[(Z1, x, y)] = a
        print(f"{cog.name}: {sum(1 for k in tiles if k[0] == Z1)} zoom-{Z1} tiles so far", flush=True)
    for z in range(Z1 - 1, Z0 - 1, -1):
        for px, py in {(x // 2, y // 2) for (zz, x, y) in tiles if zz == z + 1}:
            kids = [tiles.get((z + 1, 2 * px + i, 2 * py + j)) for j in (0, 1) for i in (0, 1)]
            if all(k is not None for k in kids):  # full parents only, for the same reason as above
                tiles[(z, px, py)] = halve(np.block([[kids[0], kids[1]], [kids[2], kids[3]]]))
    lon0 = min(b[0] for b in boxes); lat0 = min(b[1] for b in boxes)
    lon1 = max(b[2] for b in boxes); lat1 = max(b[3] for b in boxes)
    with open(OUT, "wb") as f:
        w = Writer(f)
        for (z, x, y) in sorted(tiles, key=lambda k: zxy_to_tileid(*k)):
            w.write_tile(zxy_to_tileid(z, x, y), encode(tiles[(z, x, y)]))
        w.finalize(
            {"tile_type": TileType.WEBP, "tile_compression": Compression.NONE,
             "internal_compression": Compression.GZIP, "min_zoom": Z0, "max_zoom": Z1,
             "min_lon_e7": int(lon0 * 1e7), "min_lat_e7": int(lat0 * 1e7),
             "max_lon_e7": int(lon1 * 1e7), "max_lat_e7": int(lat1 * 1e7),
             "center_zoom": Z0, "center_lon_e7": int((lon0 + lon1) / 2 * 1e7), "center_lat_e7": int((lat0 + lat1) / 2 * 1e7)},
            {"name": "NepalDTM G (test)", "description": "Generated 1 m bare-earth detail over test boxes, Terrarium, 0.1 m",
             "attribution": "NepalDTM G (generated, not measured), Achyut Paudel, CC BY 4.0", "format": "webp",
             "encoding": "terrarium", "minzoom": Z0, "maxzoom": Z1, "boxes": boxes},
        )
    print(f"Done: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(tiles):,} tiles); boxes {boxes}")


if __name__ == "__main__":
    main(sys.argv[1:])
