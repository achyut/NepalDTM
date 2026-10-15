"""Clip free global 30 m DEMs to an AOI for benchmarking (no full downloads).

- FABDEM V1-2 (Copernicus with forests and buildings removed; Hawker et al. 2022; Univ. of Bristol,
  CC BY-NC-SA 4.0). Single tiles are read out of the 10x10 degree zip with HTTP range requests.
- GEDTM30 v1.2 (OpenGeoHub ensemble DTM trained on ICESat-2 and GEDI; Ho et al. 2025, CC BY 4.0), public COG.
- Copernicus GLO-30 is already in the stacks.

All heights are EGM2008 metres. Output: data/external/<aoi>_<name>.tif in EPSG:4326, ~500 m beyond the AOI.

Run:
    ~/workspace/super-resolution-satellite/.venv/bin/python -I fetch_external_dems.py --aoi bhotekoshi
"""

from __future__ import annotations

import argparse
import math
import sys
import zipfile
from pathlib import Path

import fsspec
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_altimetry import AOIS  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
FABDEM_ZIP = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/{}_FABDEM_V1-2.zip"
GEDTM30 = "https://s3.opengeohub.org/global/dtm/v1.2/gedtm_rf_m_30m_s_20060101_20151231_go_epsg.4326.3855_v1.2.tif"
PAD = 0.005


def save(path, a, transform, crs):
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype="float32",
                       crs=crs, transform=transform, nodata=np.nan, compress="deflate", predictor=3) as dst:
        dst.write(a.astype("float32"), 1)
    print("->", path, f"{a.shape[1]}x{a.shape[0]}")


def fabdem(bounds, out):
    w, s, e, n = bounds
    ds, mem = [], []  # keep the MemoryFiles alive while their datasets are open (else merge segfaults)
    for lat in range(math.floor(s), math.floor(n) + 1):
        for lon in range(math.floor(w), math.floor(e) + 1):
            la, lo = lat // 10 * 10, lon // 10 * 10
            z = f"N{la:02d}E{lo:03d}-N{la + 10:02d}E{lo + 10:03d}"
            name = f"N{lat:02d}E{lon:03d}_FABDEM_V1-2.tif"
            with fsspec.open(FABDEM_ZIP.format(z), block_size=2 ** 22).open() as f, zipfile.ZipFile(f) as zf:
                mem.append(MemoryFile(zf.read(name)))
                ds.append(mem[-1].open())
            print("  read", name)
    a, t = merge(ds, bounds=bounds, nodata=np.nan, dtype="float32")
    save(out, a[0], t, ds[0].crs)


def gedtm(bounds, out):
    with rasterio.open(GEDTM30) as src:
        win = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
        # v1.2 values are metres: the file's scale tag (0.1) is wrong; the project's cog_list.csv gives scale 1,
        # and raw values match FABDEM to ~2 m (Khumbu median 5310 vs 5308).
        a = src.read(1, window=win, masked=True).astype("float32").filled(np.nan)
        save(out, a, src.window_transform(win), src.crs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="bhotekoshi", choices=list(AOIS))
    args = ap.parse_args()
    w, s, e, n = AOIS[args.aoi]
    b = (w - PAD, s - PAD, e + PAD, n + PAD)
    (DATA / "external").mkdir(exist_ok=True)
    fabdem(b, DATA / "external" / f"{args.aoi}_fabdem.tif")
    gedtm(b, DATA / "external" / f"{args.aoi}_gedtm30.tif")


if __name__ == "__main__":
    main()
