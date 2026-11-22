#!/usr/bin/env python3
"""NepalDTM-G national map tiles, read straight from R2: zooms 15-17, lossless-WebP Terrarium, in a few PMTiles
archives split by longitude (each small enough to build on a Colab disk).

All provinces' 10 km G tiles go into one VRT over /vsis3/, so where two provinces meet their masked tiles fill each
other and no map tile is lost along a border. Work is done one zoom-15 block at a time (16 + 4 + 1 map tiles from
the same 1 m data, so the GDAL cache stays hot); only map tiles fully covered by G are kept.

Needs R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET. Resumable: bands already in R2 are skipped.
Run (from code/): python -u g/g_tiles_nepal.py [--bands 4] [--workers 12]
Writes <bucket>/nepal/NepalDTM_G_nepal_<k>.webp.pmtiles
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "g")]

PROVINCES = ["bagmati", "koshi", "madhesh", "gandaki", "lumbini", "karnali", "sudurpashchim"]
NAT_X0, NAT_Y0, CORE = -192000.0, 3408000.0, 52 * 24 * 8  # national 8 m grid origin; 9,984 m tile core
Z0, Z1 = 15, 17


def gdal_env():
    ep = os.environ["R2_ENDPOINT"].replace("https://", "").rstrip("/")
    os.environ.update({"AWS_S3_ENDPOINT": ep, "AWS_HTTPS": "YES", "AWS_VIRTUAL_HOSTING": "FALSE",
                       "AWS_REGION": "auto", "AWS_ACCESS_KEY_ID": os.environ["R2_ACCESS_KEY_ID"],
                       "AWS_SECRET_ACCESS_KEY": os.environ["R2_SECRET_ACCESS_KEY"],
                       "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
                       "VSI_CACHE": "TRUE", "VSI_CACHE_SIZE": "200000000", "GDAL_CACHEMAX": "1024",
                       "GDAL_HTTP_MAX_RETRY": "8", "GDAL_HTTP_RETRY_DELAY": "2"})


def s3():
    import boto3
    return boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"], region_name="auto",
                        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"])


def list_tiles(c, bucket):
    keys = []
    for p in PROVINCES:
        for page in c.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{p}/tiles/"):
            keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".tif")]
    return keys


def core_bounds(key):
    ti, tj = (int(v) for v in Path(key).stem.split("_")[1:3])
    x0, y1 = NAT_X0 + tj * CORE, NAT_Y0 - ti * CORE
    return x0, y1 - CORE, x0 + CORE, y1


def write_vrt(keys, bucket, path):
    from rasterio.crs import CRS
    b = [core_bounds(k) for k in keys]
    x0, y0 = min(v[0] for v in b), min(v[1] for v in b)
    x1, y1 = max(v[2] for v in b), max(v[3] for v in b)
    W, H, n = round(x1 - x0), round(y1 - y0), round(CORE)
    parts = [f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}"><SRS>{CRS.from_epsg(32645).to_wkt()}</SRS>',
             f"<GeoTransform>{x0}, 1.0, 0, {y1}, 0, -1.0</GeoTransform>",
             '<VRTRasterBand dataType="Float32" band="1"><NoDataValue>nan</NoDataValue>']
    for k, (bx0, _, _, by1) in zip(keys, b):
        parts.append(f'<ComplexSource><SourceFilename relativeToVRT="0">/vsis3/{bucket}/{k}</SourceFilename>'
                     f'<SourceBand>1</SourceBand><SourceProperties RasterXSize="{n}" RasterYSize="{n}" '
                     f'DataType="Float32" BlockXSize="512" BlockYSize="512"/>'
                     f'<SrcRect xOff="0" yOff="0" xSize="{n}" ySize="{n}"/>'
                     f'<DstRect xOff="{round(bx0 - x0)}" yOff="{round(y1 - by1)}" xSize="{n}" ySize="{n}"/>'
                     f'<NODATA>nan</NODATA></ComplexSource>')
    Path(path).write_text("".join(parts) + "</VRTRasterBand></VRTDataset>")


def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    return int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


_src = None


def _init(vrt):
    global _src
    gdal_env()
    import rasterio
    _src = rasterio.open(vrt)


def render_block(xy):
    """All map tiles of one zoom-15 block: returns [(z, x, y, webp bytes)] for the fully covered ones."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject
    from nw_tiles_webp import SIZE, encode, merc_bounds
    bx, by = xy
    out = []
    for z in (Z1, Z1 - 1, Z0):
        k = 2 ** (z - Z0)
        for x in range(bx * k, bx * k + k):
            for y in range(by * k, by * k + k):
                a = np.full((SIZE, SIZE), np.nan, "float32")
                try:
                    reproject(rasterio.band(_src, 1), a, src_nodata=np.nan, dst_nodata=np.nan,
                              dst_transform=from_bounds(*merc_bounds(z, x, y), SIZE, SIZE), dst_crs="EPSG:3857",
                              resampling=Resampling.bilinear if z == Z1 else Resampling.average)
                except Exception as e:  # one bad read must not stop a multi-hour job; the tile is left out
                    print(f"  read failed {z}/{x}/{y}: {e}", flush=True)
                    continue
                if np.isfinite(a).all():
                    out.append((z, x, y, encode(a)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--work", default="/content/nepal_tiles")
    ap.add_argument("--test", type=int, default=0, help="render only this many blocks of band 0, keep it local")
    a = ap.parse_args()
    from pmtiles.tile import Compression, TileType, zxy_to_tileid
    from pmtiles.writer import Writer
    from pyproj import Transformer
    bucket, c = os.environ["R2_BUCKET"], s3()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    keys = list_tiles(c, bucket)
    vrt = work / "nepal.vrt"
    write_vrt(keys, bucket, vrt)
    to_ll = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)
    blocks = set()
    for k in keys:
        x0, y0, x1, y1 = core_bounds(k)
        lon0, lat0, lon1, lat1 = to_ll.transform_bounds(x0, y0, x1, y1)
        tx0, ty0 = lonlat_to_tile(lon0, lat1, Z0)
        tx1, ty1 = lonlat_to_tile(lon1, lat0, Z0)
        blocks |= {(x, y) for x in range(tx0, tx1 + 1) for y in range(ty0, ty1 + 1)}
    xs = sorted({x for x, _ in blocks})
    cuts = [xs[round(i * len(xs) / a.bands)] for i in range(a.bands)] + [xs[-1] + 1]
    print(f"{len(keys)} G tiles from R2, {len(blocks):,} zoom-15 blocks, {a.bands} bands", flush=True)
    for k in range(a.bands):
        key = f"nepal/NepalDTM_G_nepal_{k}.webp.pmtiles"
        try:
            c.head_object(Bucket=bucket, Key=key); print(f"band {k}: already in R2"); continue
        except c.exceptions.ClientError:
            pass
        todo = sorted(b for b in blocks if cuts[k] <= b[0] < cuts[k + 1])
        if a.test:
            todo = sorted(todo, key=lambda b: (b[0] - todo[len(todo) // 2][0]) ** 2 + (b[1] - todo[len(todo) // 2][1]) ** 2)[:a.test]
        out = work / Path(key).name
        n, lon = 0, []
        with open(out, "wb") as f, Pool(a.workers, _init, (str(vrt),)) as pool:
            w = Writer(f)
            for i, tiles in enumerate(pool.imap_unordered(render_block, todo, chunksize=4)):
                for z, x, y, data in tiles:
                    w.write_tile(zxy_to_tileid(z, x, y), data); n += 1
                if i % 500 == 0:
                    print(f"  band {k}: {i:,}/{len(todo):,} blocks, {n:,} map tiles", flush=True)
            n2 = 2 ** Z0
            lon0, lon1 = cuts[k] / n2 * 360 - 180, cuts[k + 1] / n2 * 360 - 180
            w.finalize(
                {"tile_type": TileType.WEBP, "tile_compression": Compression.NONE,
                 "internal_compression": Compression.GZIP, "min_zoom": Z0, "max_zoom": Z1,
                 "min_lon_e7": int(lon0 * 1e7), "min_lat_e7": int(26.3 * 1e7),
                 "max_lon_e7": int(lon1 * 1e7), "max_lat_e7": int(30.5 * 1e7),
                 "center_zoom": Z0, "center_lon_e7": int((lon0 + lon1) / 2 * 1e7), "center_lat_e7": int(28.2 * 1e7)},
                {"name": f"NepalDTM-G 1 m, Nepal part {k + 1} of {a.bands}",
                 "description": "Generated 1 m bare-earth DTM of Nepal, Terrarium, 0.1 m",
                 "attribution": "NepalDTM-G (generated, not measured), Achyut Paudel, CC BY 4.0", "format": "webp",
                 "encoding": "terrarium", "minzoom": Z0, "maxzoom": Z1, "g_tiles": len(keys)})
        print(f"band {k}: {n:,} map tiles, {out.stat().st_size / 1e9:.1f} GB; uploading", flush=True)
        if a.test:
            print("test: not uploading", out); return
        c.upload_file(str(out), bucket, key, ExtraArgs={"ContentType": "application/vnd.pmtiles"})
        out.unlink()
        print(f"band {k}: uploaded {key}", flush=True)
    print("national map tiles done", flush=True)


if __name__ == "__main__":
    main()
