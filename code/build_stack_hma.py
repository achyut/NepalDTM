"""Build an 8 m co-registered stack for a mountain AOI on the NASA HMA 8 m stereo DEM.

Layers (EPSG:32645, 8 m, heights in EGM2008 metres; written compactly to <aoi>_stack8m_c.tif):
  hma        HMA DEM 8 m mosaic (WorldView stereo, DSM), ellipsoid -> EGM2008; NaN where missing/blunder
  cop        Copernicus GLO-30 DSM, bilinear
  dsm        hma where valid, else cop + smoothly interpolated local (hma - cop) offset
  src        1 = HMA, 2 = Copernicus fill
  slope, aspect_sin, aspect_cos, tri, tpi   terrain shape from dsm
  wc         ESA WorldCover 2021 class
  chm_mean, chm_q3, chm_max   Meta/WRI 1 m canopy height aggregated to 8 m
  dsm_minus_cop

Run:
    PROJ_NETWORK=ON ~/workspace/super-resolution-satellite/.venv/bin/python -I build_stack_hma.py --aoi bhotekoshi
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_altimetry import AOIS  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
CRS = "EPSG:32645"
RES = 8.0
HMA = "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/HMA/HMA_DEM8m_MOS/1/2002/01/28/HMA_DEM8m_MOS_20170716_tile-{t}.tif"
HMA_TILES = {"bhotekoshi": [675], "khumbu": [677]}
COP = "/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM/Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM.tif"
CHM = "/vsicurl/https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float/chm/{q}.tif"
WARP_MEM = 512


def token() -> str:
    line = next(l.strip() for l in (Path.home() / ".earthdata_token").read_text().splitlines() if l.strip())
    return line.split("=", 1)[1].strip() if line.startswith("EARTHDATA_TOKEN=") else line


def signed(url: str) -> str:
    # GDAL forwards the bearer header to the CloudFront redirect, which rejects it; resolve the redirect first.
    r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{redirect_url}", "-H", f"Authorization: Bearer {token()}", url],
                       capture_output=True, text=True, check=True)
    return "/vsicurl/" + r.stdout.strip()


def grid(aoi):
    w, s, e, n = transform_bounds("EPSG:4326", CRS, *AOIS[aoi])
    w, s = math.floor(w / RES) * RES, math.floor(s / RES) * RES
    e, n = math.ceil(e / RES) * RES, math.ceil(n / RES) * RES
    W, H = int((e - w) / RES), int((n - s) / RES)
    return from_origin(w, n, RES, RES), W, H, (w, s, e, n)


def warp(paths, transform, W, H, resampling, src_nodata=None):
    dst = np.full((H, W), np.nan, "float32")
    for p in paths:
        with rasterio.open(p) as src:
            tmp = np.full((H, W), np.nan, "float32")
            reproject(rasterio.band(src, 1), tmp, src_nodata=src_nodata if src_nodata is not None else src.nodata,
                      dst_transform=transform, dst_crs=CRS, dst_nodata=np.nan, resampling=resampling,
                      warp_mem_limit=WARP_MEM)
            fill = np.isnan(dst)
            dst[fill] = tmp[fill]
    return dst


def quadkeys(bb, z=9):
    def tile(lon, lat):
        x = int((lon + 180) / 360 * 2 ** z)
        y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2 ** z)
        return x, y
    x0, y1 = tile(bb[0], bb[1]); x1, y0 = tile(bb[2], bb[3])
    out = []
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            out.append("".join(str(((x >> i) & 1) + 2 * ((y >> i) & 1)) for i in range(z - 1, -1, -1)))
    return out


def geoid_offset(transform, W, H, step=16):
    """z such that h_egm2008 = h_ellipsoid + z, on the grid (geoid is smooth: compute coarse, interpolate)."""
    rows, cols = np.mgrid[0:H + step:step, 0:W + step:step]
    x, y = transform * (cols + .5, rows + .5)
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(x, y)
    _, _, z = Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True).transform(lon, lat, np.zeros_like(lon))
    rr, cc = np.mgrid[0:H, 0:W]
    return ndimage.map_coordinates(z, [rr / step, cc / step], order=1, mode="nearest").astype("float32")


def fill_from_cop(hma, cop, sigma_px=25):
    """Gaps: cop plus the local hma-cop offset, spread by normalized Gaussian convolution."""
    d = np.where(np.isfinite(hma), hma - cop, 0.0)
    m = np.isfinite(hma).astype("float32")
    num = ndimage.gaussian_filter(d, sigma_px)
    den = ndimage.gaussian_filter(m, sigma_px)
    glob = float(np.nanmedian(hma - cop))
    off = np.where(den > 1e-3, num / np.maximum(den, 1e-3), glob)
    return np.where(np.isfinite(hma), hma, cop + off).astype("float32")


def shape(z):
    gy, gx = np.gradient(z, RES)
    gy = -gy  # rows run north to south
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.arctan2(-gx, -gy)  # downslope direction, 0 = north
    pad = np.pad(z, 1, mode="edge")
    tri = np.mean([np.abs(pad[1 + dy:pad.shape[0] - 1 + dy, 1 + dx:pad.shape[1] - 1 + dx] - z)
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx], axis=0)
    tpi = z - ndimage.uniform_filter(z, 13)  # ~100 m neighbourhood: ridge (+) vs valley (-)
    return [a.astype("float32") for a in (slope, np.sin(aspect), np.cos(aspect), tri, tpi)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="bhotekoshi", choices=list(HMA_TILES))
    args = ap.parse_args()
    transform, W, H, bounds = grid(args.aoi)
    print(f"grid {W} x {H} at {RES} m, bounds {bounds}")
    L = {}

    print("hma"); hma_ell = warp([signed(HMA.format(t=t)) for t in HMA_TILES[args.aoi]], transform, W, H, Resampling.bilinear, -9999.0)
    lon0, lat0, lon1, lat1 = AOIS[args.aoi]
    print("copernicus")
    L["cop"] = warp([COP.format(lat=la, lon=lo) for la in range(int(lat0), int(lat1) + 1) for lo in range(int(lon0), int(lon1) + 1)],
                    transform, W, H, Resampling.bilinear)
    if np.isfinite(hma_ell).mean() < 0.3:
        raise SystemExit(f"HMA read looks broken: only {np.isfinite(hma_ell).mean():.1%} valid; rerun")
    z = geoid_offset(transform, W, H)
    hma = hma_ell + z
    d = hma - L["cop"]
    print(f"  geoid offset on grid: {np.nanmin(z):+.2f} .. {np.nanmax(z):+.2f} m")
    print(f"  median(HMA_ellipsoid - cop) = {np.nanmedian(hma_ell - L['cop']):+.2f} m;"
          f" after geoid: median(HMA - cop) = {np.nanmedian(d):+.2f} m, MAD {np.nanmedian(np.abs(d - np.nanmedian(d))):.2f} m")
    # Blunders (cloud, snow, matching failures) and big real change: drop where far from Copernicus.
    blunder = np.abs(d - np.nanmedian(d)) > 40
    hma[blunder] = np.nan
    print(f"  HMA valid {np.isfinite(hma).mean():.1%} (blunders removed {blunder.mean():.1%})")
    L["hma"] = hma.astype("float32")
    L["dsm"] = fill_from_cop(hma, L["cop"])
    L["src"] = np.where(np.isfinite(hma), 1, 2).astype("float32")

    print("worldcover")
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    items = list(cat.search(collections=["esa-worldcover"], bbox=AOIS[args.aoi], datetime="2021").items())
    L["wc"] = warp([it.assets["map"].href for it in items], transform, W, H, Resampling.mode, 0)

    qs = quadkeys(AOIS[args.aoi])
    print("canopy tiles", qs)
    for name, rs in (("chm_mean", Resampling.average), ("chm_q3", Resampling.q3), ("chm_max", Resampling.max)):
        L[name] = warp([CHM.format(q=q) for q in qs], transform, W, H, rs, 255)
        print(f"  {name} nan {np.isnan(L[name]).mean():.1%}")

    # Compact storage (disk is tight): heights as 0.1 m integers, classes/canopy as small integers; 0 = nodata.
    # Terrain-shape features are recomputed by fuse_hma.py.
    from fuse_hma import encode
    enc = {"hma": "h2", "cop": "h2", "dsm": "h2", "src": "u8", "wc": "u8", "chm_mean": "q4", "chm_q3": "u8", "chm_max": "u8"}
    path = OUT / f"{args.aoi}_stack8m_c.tif"
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=len(enc), dtype="uint16", crs=CRS,
                       transform=transform, nodata=0, compress="deflate", predictor=2, tiled=True,
                       blockxsize=512, blockysize=512) as dst:
        for i, (k, e) in enumerate(enc.items(), 1):
            dst.write(encode(L[k], e), i)
            dst.set_band_description(i, f"{k}:{e}")
    for k, v in L.items():
        print(f"{k:13s} min={np.nanmin(v):9.2f} mean={np.nanmean(v):9.2f} max={np.nanmax(v):9.2f} nan={np.isnan(v).mean():.1%}")
    print("->", path, f"{path.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
