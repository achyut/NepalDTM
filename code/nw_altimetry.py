"""Nationwide ICESat-2 ATL08 (20 m) and GEDI L2A ground points via SlideRule (server-side subsetting).

Same fields and filters as fetch_altimetry.py / fetch_gedi_sliderule.py, so nationwide training data matches the
AOI runs: ATL08 terrain/h_te_best_fit_20m with snow cover, msw_flag, strong beam (spots 1, 3, 5); GEDI
elev_lowestmode with quality, degrade, sensitivity, power-beam flags. Heights are WGS84 ellipsoidal.

Nepal is fetched in 0.5 x 0.5 degree cells (one SlideRule request each, resumable):
    data/nw/alt/<cell>_atl08.npz, <cell>_gedi.npz

Run:
    ~/workspace/super-resolution-satellite/.venv/bin/python -I nw_altimetry.py            # all cells
    ~/workspace/super-resolution-satellite/.venv/bin/python -I nw_altimetry.py --bbox 86.76 27.86 86.92 28.02 --out x
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sliderule import sliderule

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "nw" / "alt"
NEPAL = DATA / "control" / "nepal_boundary.geojson"
FILL = 1e30


def poly(w, s, e, n):
    return [{"lon": w, "lat": s}, {"lon": e, "lat": s}, {"lon": e, "lat": n}, {"lon": w, "lat": n}, {"lon": w, "lat": s}]


def atl08(bb):
    df = sliderule.run("atl08x", {"poly": poly(*bb), "atl08_fields": [
        "terrain/h_te_best_fit_20m", "canopy/h_canopy_20m", "latitude_20m", "longitude_20m", "msw_flag", "night_flag"]})
    if df is None or len(df) == 0:
        return None
    k = lambda c: np.stack(df[c].to_numpy()).astype("float64")  # (n, 5) per 100 m segment
    lat, lon, h, can = k("latitude_20m"), k("longitude_20m"), k("terrain/h_te_best_fit_20m"), k("canopy/h_canopy_20m")
    rep = lambda c: np.repeat(df[c].to_numpy()[:, None], 5, 1)
    t = (df.index.astype("int64").to_numpy() / 1e9 - 1514764800)  # seconds since 2018-01-01 (ATLAS epoch), as delta_time
    out = dict(lat=lat, lon=lon, h=h, canopy=can, snow=rep("segment_snowcover"), msw=rep("msw_flag"),
               night=rep("night_flag"), strong=np.isin(rep("spot"), (1, 3, 5)), t=np.repeat(t[:, None], 5, 1),
               unc=rep("h_te_uncertainty"), slope=rep("terrain_slope"))
    out = {a: b.ravel() for a, b in out.items()}
    ok = (out["lat"] < FILL) & (out["lon"] < FILL)
    w, s, e, n = bb
    ok &= (out["lon"] >= w) & (out["lon"] <= e) & (out["lat"] >= s) & (out["lat"] <= n)
    out = {a: b[ok] for a, b in out.items()}
    out["h"] = np.where(out["h"] < FILL, out["h"], 3.4e38)  # keep the ATL08 fill convention of fetch_altimetry.py
    out["canopy"] = np.where(out["canopy"] < FILL, out["canopy"], 3.4e38)
    return out


def gedi(bb):
    df = sliderule.run("gedi02ax", {"poly": poly(*bb), "degrade_filter": True, "l2_quality_filter": True})
    if df is None or len(df) == 0:
        return None
    beam = df["beam"].to_numpy()
    flags = df["flags"].to_numpy().astype(int) if "flags" in df else np.zeros(len(df), int)
    t = df.index.astype("int64").to_numpy() / 1e9 - 1514764800
    return dict(lat=df.geometry.y.to_numpy(), lon=df.geometry.x.to_numpy(), h=df["elevation_lm"].to_numpy(),
                canopy=(df["elevation_hr"] - df["elevation_lm"]).to_numpy(), t=t,
                quality=np.ones(len(df), int), bias_flag=np.zeros(len(df), int), degrade=np.zeros(len(df), int),
                sens=df["sensitivity"].to_numpy(), power=np.isin(beam, (5, 6, 8, 11)), flags=flags)


def cells():
    from shapely.geometry import box, shape
    nep = shape(json.load(open(NEPAL))["features"][0]["geometry"]).buffer(0.02)
    out = []
    for w in np.arange(80.0, 88.5, 0.5):
        for s in np.arange(26.0, 30.5, 0.5):
            if nep.intersects(box(w, s, w + .5, s + .5)):
                out.append((float(w), float(s), float(w + .5), float(s + .5)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float)
    ap.add_argument("--out")
    args = ap.parse_args()
    sliderule.init(verbose=False)
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [(args.out, tuple(args.bbox))] if args.bbox else [(f"E{w:.1f}N{s:.1f}", (w, s, e, n)) for w, s, e, n in cells()]
    print(f"{len(todo)} cells")
    for i, (name, bb) in enumerate(todo, 1):
        for kind, fn in (("atl08", atl08), ("gedi", gedi)):
            p = OUT / f"{name}_{kind}.npz"
            if p.exists():
                continue
            for attempt in range(3):
                try:
                    t0 = time.time()
                    d = fn(bb)
                    break
                except Exception as e:  # SlideRule occasionally drops a request; retry
                    print(f"  retry {name} {kind}: {str(e)[:120]}", flush=True)
                    time.sleep(20)
            else:
                continue
            if d is None:
                d = {"lat": np.zeros(0)}
            np.savez_compressed(p, **d)
            print(f"  {i}/{len(todo)} {name} {kind}: {d['lat'].size} points, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
