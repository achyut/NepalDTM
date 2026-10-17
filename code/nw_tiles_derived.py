"""Build the slope, aspect and roughness tiles the web map draws.

The map used to work these out in the browser, a few milliseconds per tile for slope and roughness but up to
150 ms for the shaded ones, which is most of why it felt slow. They are fixed properties of the terrain, so
they belong in tiles built once, the way the terrain itself is.

Each measure is stored as a value, not a colour: the number is Terrarium-encoded exactly like a height, and
MapLibre's color-relief layer maps it to a colour on the graphics card. The palette and the light and dark
themes stay in the page, so a colour can change without rebuilding anything.

  slope      Horn's 3 x 3 gradient, degrees from flat (0 to 90)
  aspect     the compass direction the slope faces, degrees from north (0 to 360; 999 marks flat ground)
  roughness  the largest height difference in the 3 x 3 window, metres

A measure depends on the width of the pixel it is measured over, so each zoom is computed from that zoom's own
heights rather than by averaging the zoom below. That matches what the browser produced and keeps the honest
reading that slopes look gentler when you zoom out.

Run after nw_apply.py has written data/nw/nepal_dtm_8m.vrt:
    cd code
    ../.venv/bin/python -I nw_tiles_derived.py [--workers 10] [--measures slope,aspect,roughness]

Writes data/nw/NepalDTM.{slope,aspect,roughness}.pmtiles.
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
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

DATA = Path(__file__).resolve().parent / "data" / "nw"
VRT = DATA / "nepal_dtm_8m.vrt"
TILES = DATA / "tiles_derived"
MIN_ZOOM, BLOCK_ZOOM, MAX_ZOOM = 5, 9, 14
SIZE = 256
SPAN = 2 ** (MAX_ZOOM - BLOCK_ZOOM)  # zoom-14 tiles per block side
HALO = 64  # zoom-14 pixels of overlap, so gradients stay right at a block edge down to zoom 9
HALF = 20037508.342789244  # half the web mercator world, m
FLAT_SLOPE = 2.0  # degrees: below this the aspect is meaningless
FLAT_ASPECT = 999.0  # the value that marks it
# How finely each measure is stored. These are for drawing: the colour ramps span tens of degrees or metres,
# so tenths would only cost space. The readout under the cursor still works the values out from the heights,
# so it stays exact. Rounding to 0.5 degrees makes a tile 39% smaller.
STEP = {"slope": 0.5, "aspect": 1.0, "roughness": 0.5}
MEASURES = tuple(STEP)


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


def pixel_metres(z, rows, y0):
    """Width of a pixel, per grid row: web mercator stretches with latitude."""
    n = 2 ** z * SIZE
    lat = np.arctan(np.sinh(np.pi * (1 - 2 * (y0 + rows + 0.5) / n)))
    return (40075016.686 / n) * np.cos(lat)


def measure(a, kind, cell, out):
    """One measure over the array's interior, written into `out` (which is 2 rows and columns smaller).

    The 3 x 3 window is taken as slices of `a`, which are views rather than copies: rolling the whole array
    nine times needs nine times its memory, and at 8 k square per worker that was enough to make the machine
    swap.
    """
    v = [a[dy:a.shape[0] - 2 + dy, dx:a.shape[1] - 2 + dx] for dy in (0, 1, 2) for dx in (0, 1, 2)]
    nw, n_, ne, w_, c_, e_, sw, s_, se = v
    if kind == "roughness":
        np.copyto(out, nw)
        hi = out
        for s in v[1:]:
            np.fmax(hi, s, out=hi)
        lo = np.array(nw)
        for s in v[1:]:
            np.fmin(lo, s, out=lo)
        hi -= lo
        return out
    # Horn's 3 x 3 gradient, rows running north to south.
    cell8 = 8 * cell[1:-1]
    dzdx = ((ne + 2 * e_ + se) - (nw + 2 * w_ + sw)) / cell8
    dzdy = ((sw + 2 * s_ + se) - (nw + 2 * n_ + ne)) / cell8
    np.hypot(dzdx, dzdy, out=out)
    np.degrees(np.arctan(out, out=out), out=out)
    if kind == "slope":
        return out
    aspect = np.degrees(np.arctan2(-dzdx, dzdy))
    aspect += 360.0
    np.mod(aspect, 360.0, out=aspect)
    return np.where(out < FLAT_SLOPE, FLAT_ASPECT, aspect)


def encode(v, step):
    """Terrarium: value = R * 256 + G + B / 256 - 32768. Gaps take the nearest value."""
    bad = ~np.isfinite(v)
    if bad.any():
        if bad.all():
            return None
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        v = v[tuple(idx)]
    e = np.clip(np.round(v.astype("float64") / step) * step + 32768, 0, 65535.996)
    rgb = np.stack([e // 256, np.floor(e % 256), np.floor((e * 256) % 256)], -1).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, "WEBP", lossless=True, quality=50, method=1, exact=True)
    return buf.getvalue()


def save(kind, z, x, y, v):
    if not np.isfinite(v).any():
        return 0
    data = encode(v, STEP[kind])
    if data is None:
        return 0
    p = TILES / kind / str(z) / str(x) / f"{y}.webp"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return 1


def block(bx, by, kinds):
    """Render one zoom-9 block for every measure, zooms 14 down to 9; return its zoom-9 heights."""
    n = SIZE * SPAN
    a = np.full((n + 2 * HALO, n + 2 * HALO), np.nan, dtype="float32")
    x0, y0, x1, y1 = merc_bounds(BLOCK_ZOOM, bx, by)
    pad = (x1 - x0) * HALO / n
    with rasterio.Env(GDAL_CACHEMAX=512), rasterio.open(VRT) as src:
        reproject(rasterio.band(src, 1), a, src_nodata=src.nodata, dst_nodata=np.nan,
                  dst_transform=from_bounds(x0 - pad, y0 - pad, x1 + pad, y1 + pad, a.shape[1], a.shape[0]),
                  dst_crs="EPSG:3857", resampling=Resampling.bilinear, num_threads=2)
    written = 0
    halo = HALO
    for z in range(MAX_ZOOM, BLOCK_ZOOM - 1, -1):
        if np.isfinite(a).any():
            k = 2 ** (z - BLOCK_ZOOM)
            rows = np.arange(a.shape[0]) - halo
            cell = pixel_metres(z, rows, by * SIZE * k)[:, None]
            scratch = np.empty((a.shape[0] - 2, a.shape[1] - 2), dtype="float32")
            for kind in kinds:
                full = measure(a, kind, cell, scratch)
                core = full[halo - 1:halo - 1 + SIZE * k, halo - 1:halo - 1 + SIZE * k]
                for i in range(k):
                    for j in range(k):
                        written += save(kind, z, bx * k + i, by * k + j,
                                        core[j * SIZE:(j + 1) * SIZE, i * SIZE:(i + 1) * SIZE])
        if z > BLOCK_ZOOM:
            a = halve(a)
            halo //= 2
    return bx, by, a[halo:halo + SIZE, halo:halo + SIZE].copy(), written


def pack(kind, lon0, lat0, lon1, lat1):
    out = DATA / f"NepalDTM.{kind}.pmtiles"
    files = sorted((zxy_to_tileid(int(p.parts[-3]), int(p.parts[-2]), int(p.stem)), p)
                   for p in (TILES / kind).glob("*/*/*.webp"))
    with open(out, "wb") as f:
        w = Writer(f)
        for tid, p in files:
            w.write_tile(tid, p.read_bytes())
        w.finalize(
            {"tile_type": TileType.WEBP, "tile_compression": Compression.NONE,
             "internal_compression": Compression.GZIP, "min_zoom": MIN_ZOOM, "max_zoom": MAX_ZOOM,
             "min_lon_e7": int(lon0 * 1e7), "min_lat_e7": int(lat0 * 1e7),
             "max_lon_e7": int(lon1 * 1e7), "max_lat_e7": int(lat1 * 1e7),
             "center_zoom": 7, "center_lon_e7": int((lon0 + lon1) / 2 * 1e7),
             "center_lat_e7": int((lat0 + lat1) / 2 * 1e7)},
            {"name": f"NepalDTM {kind}", "description": f"{kind} of the NepalDTM bare-earth terrain, "
                                                        "Terrarium-encoded as a value, not a colour",
             "attribution": "NepalDTM, Achyut Paudel, CC BY 4.0", "type": "overlay", "format": "webp",
             "encoding": "terrarium", "measure": kind, "minzoom": MIN_ZOOM, "maxzoom": MAX_ZOOM},
        )
    print(f"{out.name}: {out.stat().st_size / 1e9:.2f} GB, {len(files):,} tiles", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--measures", default=",".join(MEASURES))
    args = ap.parse_args()
    kinds = [k for k in args.measures.split(",") if k in MEASURES]
    if not VRT.exists():
        raise SystemExit(f"missing {VRT}; run nw_apply.py first")

    with rasterio.open(VRT) as src:
        lon0, lat0, lon1, lat1 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    x0, y0 = tile_xy(lon0, lat1, BLOCK_ZOOM)
    x1, y1 = tile_xy(lon1, lat0, BLOCK_ZOOM)
    blocks = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    print(f"{len(blocks)} zoom-{BLOCK_ZOOM} blocks, {args.workers} workers, measures: {', '.join(kinds)}", flush=True)

    t0 = time.time()
    level, written = {}, 0
    with ProcessPoolExecutor(args.workers) as pool:
        for n, f in enumerate(as_completed([pool.submit(block, x, y, kinds) for x, y in blocks]), 1):
            bx, by, a, w = f.result()
            written += w
            if np.isfinite(a).any():
                level[(bx, by)] = a
            print(f"  block {n}/{len(blocks)} ({bx},{by}): {w} tiles, {time.time() - t0:.0f} s", flush=True)

    # Zooms 8 down to 5, from the assembled coarser heights.
    empty = np.full((SIZE, SIZE), np.nan, dtype="float32")
    for z in range(BLOCK_ZOOM - 1, MIN_ZOOM - 1, -1):
        nxt = {}
        for px, py in {(x // 2, y // 2) for x, y in level}:
            quad = np.block([[level.get((2 * px, 2 * py), empty), level.get((2 * px + 1, 2 * py), empty)],
                             [level.get((2 * px, 2 * py + 1), empty), level.get((2 * px + 1, 2 * py + 1), empty)]])
            nxt[(px, py)] = halve(quad)
        for (px, py), a in nxt.items():
            pad = np.pad(a, 1, mode="edge")
            cell = pixel_metres(z, np.arange(SIZE + 2) - 1, py * SIZE)[:, None]
            scratch = np.empty((SIZE, SIZE), dtype="float32")
            for kind in kinds:
                written += save(kind, z, px, py, measure(pad, kind, cell, scratch))
        level = nxt
    print(f"{written:,} tiles in {time.time() - t0:.0f} s; packing", flush=True)
    for kind in kinds:
        pack(kind, lon0, lat0, lon1, lat1)


if __name__ == "__main__":
    main()
