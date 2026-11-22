#!/usr/bin/env python3
"""G 1 m map tiles for a province run: zooms 15-17 over every finished 10 km tile, as a lossless-WebP Terrarium
PMTiles overlay (same encoding as gen/g_tiles.py, so the website decodes it unchanged).

Scales to a province: map tiles are rendered one at a time, in parallel, straight from a VRT mosaic of the G tiles
(zoom 17 bilinear, zooms 16 and 15 averaged), and the archive writer streams tile data to disk. Only map tiles fully
covered by G are kept, so none mixes G with nodata.

Run: python -u g/g_tiles_region.py (from code/) bagmati
Writes data/gen/region/<region>/NepalDTM_G_<region>.webp.pmtiles
"""
from __future__ import annotations

import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nw_tiles_webp import SIZE, encode, merc_bounds, tile_xy  # noqa: E402

Z0, Z1 = 15, 17
_src = None


def _init(vrt):
    global _src
    _src = rasterio.open(vrt)


def render(zxy):
    z, x, y = zxy
    a = np.full((SIZE, SIZE), np.nan, "float32")
    reproject(rasterio.band(_src, 1), a, src_nodata=np.nan, dst_nodata=np.nan,
              dst_transform=from_bounds(*merc_bounds(z, x, y), SIZE, SIZE), dst_crs="EPSG:3857",
              resampling=Resampling.bilinear if z == Z1 else Resampling.average)
    return zxy, (encode(a) if np.isfinite(a).all() else None)


def build_vrt(cogs, vrt):
    """VRT mosaic of same-grid 1 m tiles (written directly: the system gdalbuildvrt may lack the LERC codec)."""
    srcs = [rasterio.open(f) for f in cogs]
    t0 = srcs[0].transform
    x0 = min(s.bounds.left for s in srcs); y1 = max(s.bounds.top for s in srcs)
    x1 = max(s.bounds.right for s in srcs); y0 = min(s.bounds.bottom for s in srcs)
    W, H = round((x1 - x0) / t0.a), round((y1 - y0) / -t0.e)
    parts = [f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}"><SRS>{srcs[0].crs.to_wkt()}</SRS>',
             f"<GeoTransform>{x0}, {t0.a}, 0, {y1}, 0, {t0.e}</GeoTransform>",
             '<VRTRasterBand dataType="Float32" band="1"><NoDataValue>nan</NoDataValue>']
    for f, s in zip(cogs, srcs):
        c, r = round((s.bounds.left - x0) / t0.a), round((y1 - s.bounds.top) / -t0.e)
        parts.append(f'<ComplexSource><SourceFilename relativeToVRT="0">{f}</SourceFilename><SourceBand>1</SourceBand>'
                     f'<SrcRect xOff="0" yOff="0" xSize="{s.width}" ySize="{s.height}"/>'
                     f'<DstRect xOff="{c}" yOff="{r}" xSize="{s.width}" ySize="{s.height}"/><NODATA>nan</NODATA></ComplexSource>')
        s.close()
    Path(vrt).write_text("".join(parts) + "</VRTRasterBand></VRTDataset>")


def main(region):
    d = ROOT / "data" / "gen" / "region" / region
    cogs = sorted(f for f in glob.glob(str(d / "tiles" / "G_*.tif")) if not f.endswith(".part.tif"))
    vrt = d / "mosaic.vrt"
    build_vrt(cogs, vrt)
    # candidate map tiles: interior zoom-17 tiles of each G tile, and their parents
    cand = set()
    for f in cogs:
        with rasterio.open(f) as s:
            lon0, lat0, lon1, lat1 = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
        x0, y0 = tile_xy(lon0, lat1, Z1)
        x1, y1 = tile_xy(lon1, lat0, Z1)
        cand |= {(Z1, x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}
    for z in range(Z1 - 1, Z0 - 1, -1):
        cand |= {(z, x // 2, y // 2) for (zz, x, y) in cand if zz == z + 1}
    order = sorted(cand, key=lambda k: zxy_to_tileid(*k))
    print(f"{region}: {len(cogs)} G tiles, {len(order):,} candidate map tiles", flush=True)
    with rasterio.open(vrt) as s:
        lon0, lat0, lon1, lat1 = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
    out = d / f"NepalDTM_G_{region}.webp.pmtiles"
    n = 0
    with open(out.with_suffix(".part"), "wb") as f, Pool(int(os.environ.get("WORKERS", "6")), _init, (str(vrt),)) as pool:
        w = Writer(f)
        for i, (zxy, data) in enumerate(pool.imap(render, order, chunksize=64)):  # imap keeps tile-id order
            if data is not None:
                w.write_tile(zxy_to_tileid(*zxy), data); n += 1
            if i % 20000 == 0:
                print(f"  {i:,}/{len(order):,} rendered, {n:,} kept", flush=True)
        w.finalize(
            {"tile_type": TileType.WEBP, "tile_compression": Compression.NONE,
             "internal_compression": Compression.GZIP, "min_zoom": Z0, "max_zoom": Z1,
             "min_lon_e7": int(lon0 * 1e7), "min_lat_e7": int(lat0 * 1e7),
             "max_lon_e7": int(lon1 * 1e7), "max_lat_e7": int(lat1 * 1e7),
             "center_zoom": Z0, "center_lon_e7": int((lon0 + lon1) / 2 * 1e7),
             "center_lat_e7": int((lat0 + lat1) / 2 * 1e7)},
            {"name": f"NepalDTM-G 1 m ({region})",
             "description": f"Generated 1 m bare-earth DTM over {region.title()} Province, Terrarium, 0.1 m",
             "attribution": "NepalDTM-G (generated, not measured), Achyut Paudel, CC BY 4.0", "format": "webp",
             "encoding": "terrarium", "minzoom": Z0, "maxzoom": Z1, "g_tiles": len(cogs),
             "bounds": [lon0, lat0, lon1, lat1]},
        )
    out.with_suffix(".part").rename(out)
    print(f"Done: {out} ({out.stat().st_size / 1e9:.2f} GB, {n:,} map tiles); bounds {lon0:.3f} {lat0:.3f} "
          f"{lon1:.3f} {lat1:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bagmati")
