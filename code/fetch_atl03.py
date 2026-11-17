#!/usr/bin/env python3
"""ICESat-2 ATL03 ground photons over Nepal via SlideRule (server-side subsetting, ATL08 labels).

Per half-degree cell (same cells as data/nw/alt/*_atl08.npz), every photon that ATL08 classifies as ground,
high-confidence (cnf 4) and quality_ph == 0. Along-track cleaning: per (rgt, cycle, gt), photons are binned
every 10 m in x_atc and dropped if more than 2 m from their bin median (residual canopy/noise returns).
Heights: WGS84 ellipsoid -> EGM2008. holdout = True for ~20 % of reference ground tracks (stable hash of rgt);
those tracks are never used for training M or G.

Output: data/nw/atl03/<cell>.parquet  (lon, lat, h, t, rgt, cycle, gt, x_atc, solar_elev, holdout)

Python's OpenSSL here does not see the corporate root; SlideRule also sets trust_env=False, so the keychain
bundle (~/.certs/macos_bundle.pem, exported with `security find-certificate -a -p`) is passed explicitly.

Run:
    ../.venv/bin/python -I -u fetch_atl03.py            # all cells, skips existing
    ../.venv/bin/python -I -u fetch_atl03.py --cells E85.5N28.0
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

NW = Path(__file__).resolve().parent / "data" / "nw"
OUT = NW / "atl03"
BUNDLE = os.path.expanduser("~/.certs/macos_bundle.pem")


def init():
    from sliderule import sliderule
    from sliderule.session import Session
    if os.path.exists(BUNDLE):
        orig = Session.__init__

        def patched(self, *a, **k):
            k["ssl_verify"] = BUNDLE
            orig(self, *a, **k)
        Session.__init__ = patched
    sliderule.init(verbose=False)
    return sliderule


def cells() -> list[str]:
    return sorted(p.name.split("_")[0] for p in (NW / "alt").glob("*_atl08.npz"))


def bounds(cell: str):
    lon, lat = float(cell[1:cell.index("N")]), float(cell[cell.index("N") + 1:])
    return lon, lat, lon + 0.5, lat + 0.5


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["quality_ph"] == 0]
    keys = ["rgt", "cycle", "gt"]
    b = (df["x_atc"] // 10).astype("int64")
    med = df.groupby(keys + [b])["height"].transform("median")
    return df[(df["height"] - med).abs() <= 2.0]


def request(sr, w, s, e, n) -> pd.DataFrame | None:
    poly = [{"lon": w, "lat": s}, {"lon": e, "lat": s}, {"lon": e, "lat": n}, {"lon": w, "lat": n},
            {"lon": w, "lat": s}]
    df = sr.run("atl03x", {"poly": poly, "cnf": 4, "atl08_class": ["atl08_ground"]})
    return None if df is None or len(df) == 0 else df


def fetch(sr, cell: str, geo: Transformer) -> int:
    dest = OUT / f"{cell}.parquet"
    if dest.exists():
        return -1
    w, s, e, n = bounds(cell)
    df = request(sr, w, s, e, n)
    if df is None:
        # Dense Terai/Siwalik cells come back empty as one request (server-side size limit); quarter them.
        parts = [request(sr, a, b, a + 0.25, b + 0.25) for a in (w, w + 0.25) for b in (s, s + 0.25)]
        if any(p is None for p in parts):
            # Every cell here has ATL08 segments, so an empty answer is a server failure, not a real result:
            # raise so main() retries, and never write an empty file that would be skipped next time.
            raise RuntimeError("empty response")
        df = pd.concat(parts)
    df = clean(pd.DataFrame(df.drop(columns="geometry")).assign(lon=df.geometry.x.values, lat=df.geometry.y.values))
    _, _, h = geo.transform(df["lon"].values, df["lat"].values, df["height"].values.astype("float64"))
    if not np.isfinite(h).all():
        raise RuntimeError("geoid offset not applied (EGM2008 grid missing)")
    out = pd.DataFrame({
        "lon": df["lon"].values, "lat": df["lat"].values, "h": h.astype("float32"),
        "t": df.index.values.astype("datetime64[ns]"), "rgt": df["rgt"].values.astype("int16"),
        "cycle": df["cycle"].values.astype("int16"), "gt": df["gt"].values.astype("int16"),
        "x_atc": df["x_atc"].values, "solar_elev": df["solar_elevation"].values.astype("float32"),
        "holdout": ((df["rgt"].values.astype("int64") * 2654435761) % 100) < 20,
    })
    tmp = dest.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(dest)
    return len(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sr = init()
    geo = Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True)
    todo = args.cells or cells()
    total = 0
    for i, c in enumerate(todo, 1):
        t0 = time.time()
        for attempt in range(3):
            try:
                n = fetch(sr, c, geo)
                break
            except Exception as e:  # transient server/network errors: retry, then move on
                n = None
                print(f"  {c} attempt {attempt + 1} failed: {str(e)[:150]}", flush=True)
                time.sleep(30)
        if n is None:
            continue
        total += max(n, 0)
        print(f"  {i}/{len(todo)} {c}: {'skip' if n < 0 else f'{n:,} photons'} ({time.time() - t0:.0f} s)", flush=True)
    print(f"DONE {total:,} photons", flush=True)


if __name__ == "__main__":
    main()
