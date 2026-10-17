"""Build the web map tiles: the national bare-earth mosaic as a Terrarium terrain-RGB PMTiles archive.

Zoom 14 (8.4 m per pixel over Nepal, the finest zoom the 8 m grid supports) is rendered from
data/nw/nepal_dtm_8m.vrt, one zoom-9 block (32 x 32 tiles) per worker. Zooms 13 down to 5 are averaged
from the zoom below, 2 x 2 pixels to 1, so no level re-reads the 8 m data.

Heights are rounded to 0.1 m (far below the model's ~6 m error) and stored as lossless WebP:
about half the size of PNG, with every height decoded exactly. Lossy WebP or AVIF must not be used:
an error in the red channel moves the height by 256 m.

Terrarium: height = R * 256 + G + B / 256 - 32768. Pixels with no data inside a tile take the nearest
valid height; tiles with no data at all are left out.

Run after nw_apply.py:
    cd code
    ../.venv/bin/python -I nw_tiles_rgb.py [--workers 10]

Writes data/nw/NepalDTM.webp.pmtiles, which the website reads from R2.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.transform import from_bounds
from scipy import ndimage

DATA = Path(__file__).resolve().parent / "data" / "nw"
VRT = DATA / "nepal_dtm_8m.vrt"
TILES = DATA / "tiles_webp"
OUT = DATA / "NepalDTM.webp.pmtiles"
MIN_ZOOM, BLOCK_ZOOM, MAX_ZOOM = 5, 9, 14
SIZE = 256
SPAN = 2 ** (MAX_ZOOM - BLOCK_ZOOM)  # zoom-14 tiles per block side
HALF = 20037508.342789244  # half the web mercator world, m


def merc_bounds(z, x, y):
    s = 2 * HALF / 2 ** z
    return -HALF + x * s, HALF - (y + 1) * s, -HALF + (x + 1) * s, HALF - y * s


def tile_xy(lon, lat, z):
    n = 2 ** z
    r = math.radians(lat)
    return int((lon + 180) / 360 * n), int((1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2 * n)


def halve(a):
    """2 x 2 mean that ignores missing pixels."""
    h, w = a.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a.reshape(h // 2, 2, w // 2, 2), axis=(1, 3)).astype("float32")


def encode(h):
    """Fill gaps from the nearest height, round to 0.1 m, Terrarium-encode, lossless WebP."""
    valid = np.isfinite(h)
    if not valid.all():
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        h = h[tuple(idx)]
    e = np.clip(np.round(h.astype("float64"), 1) + 32768, 0, 65535.996)
    rgb = np.stack([e // 256, np.floor(e % 256), np.floor((e * 256) % 256)], -1).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, "WEBP", lossless=True, quality=50, method=1, exact=True)
    return buf.getvalue()


def save(z, x, y, h):
    if not np.isfinite(h).any():
        return False
    p = TILES / str(z) / str(x) / f"{y}.webp"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(encode(h))
    return True


def block(bx, by):
    """Render one zoom-9 block at zoom 14, save zooms 14..9, return the zoom-9 heights."""
    n = SIZE * SPAN
    a = np.full((n, n), np.nan, dtype="float32")
    with rasterio.Env(GDAL_CACHEMAX=512), rasterio.open(VRT) as src:
        reproject(rasterio.band(src, 1), a, src_nodata=src.nodata, dst_nodata=np.nan,
                  dst_transform=from_bounds(*merc_bounds(BLOCK_ZOOM, bx, by), n, n), dst_crs="EPSG:3857",
                  resampling=Resampling.bilinear, num_threads=2)
    written = 0
    for z in range(MAX_ZOOM, BLOCK_ZOOM - 1, -1):
        k = 2 ** (z - BLOCK_ZOOM)
        for i in range(k):
            for j in range(k):
                written += save(z, bx * k + i, by * k + j, a[j * SIZE:(j + 1) * SIZE, i * SIZE:(i + 1) * SIZE])
        if z > BLOCK_ZOOM:
            a = halve(a)
    return bx, by, a, written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    if not VRT.exists():
        raise SystemExit(f"missing {VRT}; run nw_apply.py first")

    with rasterio.open(VRT) as src:
        lon0, lat0, lon1, lat1 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    x0, y0 = tile_xy(lon0, lat1, BLOCK_ZOOM)
    x1, y1 = tile_xy(lon1, lat0, BLOCK_ZOOM)
    blocks = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    print(f"{VRT.name}: lon {lon0:.3f}..{lon1:.3f}, lat {lat0:.3f}..{lat1:.3f}; "
          f"{len(blocks)} zoom-{BLOCK_ZOOM} blocks, {args.workers} workers", flush=True)

    t0 = time.time()
    level = {}
    written = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for n, f in enumerate(as_completed([pool.submit(block, *b) for b in blocks]), 1):
            bx, by, a, w = f.result()
            written += w
            if np.isfinite(a).any():
                level[(bx, by)] = a
            print(f"  block {n}/{len(blocks)} ({bx},{by}): {w} tiles, {time.time() - t0:.0f} s", flush=True)

    empty = np.full((SIZE, SIZE), np.nan, dtype="float32")
    for z in range(BLOCK_ZOOM - 1, MIN_ZOOM - 1, -1):
        parents = {(x // 2, y // 2) for x, y in level}
        nxt = {}
        for px, py in parents:
            quad = np.block([[level.get((2 * px, 2 * py), empty), level.get((2 * px + 1, 2 * py), empty)],
                             [level.get((2 * px, 2 * py + 1), empty), level.get((2 * px + 1, 2 * py + 1), empty)]])
            nxt[(px, py)] = halve(quad)
            written += save(z, px, py, nxt[(px, py)])
        level = nxt
    print(f"{written:,} tiles in {time.time() - t0:.0f} s; packing {OUT.name}", flush=True)

    files = sorted(((zxy_to_tileid(int(p.parts[-3]), int(p.parts[-2]), int(p.stem)), p) for p in TILES.glob("*/*/*.webp")))
    with open(OUT, "wb") as f:
        w = Writer(f)
        for tid, p in files:
            w.write_tile(tid, p.read_bytes())
        w.finalize(
            {"tile_type": TileType.WEBP, "tile_compression": Compression.NONE,
             "internal_compression": Compression.GZIP, "min_zoom": MIN_ZOOM, "max_zoom": MAX_ZOOM,
             "min_lon_e7": int(lon0 * 1e7), "min_lat_e7": int(lat0 * 1e7),
             "max_lon_e7": int(lon1 * 1e7), "max_lat_e7": int(lat1 * 1e7),
             "center_zoom": 7, "center_lon_e7": int((lon0 + lon1) / 2 * 1e7), "center_lat_e7": int((lat0 + lat1) / 2 * 1e7)},
            {"name": "NepalDTM", "description": "Bare-earth terrain of Nepal, Terrarium-encoded, heights rounded to 0.1 m",
             "attribution": "NepalDTM, Achyut Paudel, CC BY 4.0", "type": "baselayer", "format": "webp",
             "encoding": "terrarium", "minzoom": MIN_ZOOM, "maxzoom": MAX_ZOOM},
        )
    print(f"Done: {OUT} ({OUT.stat().st_size / 1e9:.2f} GB, {len(files):,} tiles)")


if __name__ == "__main__":
    main()
