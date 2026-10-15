"""Fetch ICESat-2 ATL08 (20 m) and GEDI L2A ground/canopy points over an AOI.

Granules are streamed with earthaccess + h5py and only the AOI slice of each
beam is read, so nothing is downloaded whole (GEDI granules are ~2 GB each).
Heights are WGS84 ellipsoidal; convert before comparing with Copernicus (EGM2008).

Run:
    ~/workspace/super-resolution-satellite/.venv/bin/python fetch_altimetry.py --aoi bheemdatt
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import earthaccess
import h5py
import numpy as np

OUT = Path(__file__).resolve().parent / "data"
AOIS = {
    # NISAR GCOV crop in super-resolution-satellite/data/nisar/bheemdatt, plus ~500 m
    "bheemdatt": (80.113, 28.920, 80.227, 29.020),
    # Bhote Koshi / Trishuli corridor: Rasuwagadhi - Syabrubesi - Dhunche - Betrawati - Trishuli (HMA 8 m tile 675)
    "bhotekoshi": (85.15, 27.90, 85.42, 28.30),
    # Khumbu / Everest: University of Leeds dGPS control points (BGS NGDC item 75529)
    "khumbu": (86.76, 27.86, 86.92, 28.02),
}


def load_token(path: Path = Path.home() / ".earthdata_token") -> None:
    if os.environ.get("EARTHDATA_TOKEN") or not path.exists():
        return
    line = next((l.strip() for l in path.read_text().splitlines() if l.strip()), "")
    os.environ["EARTHDATA_TOKEN"] = line.split("=", 1)[1].strip() if line.startswith("EARTHDATA_TOKEN=") else line


def window(lat, lon, bb):
    w, s, e, n = bb
    idx = np.flatnonzero((lon >= w) & (lon <= e) & (lat >= s) & (lat <= n))
    return (idx.min(), idx.max() + 1, idx) if idx.size else None


def read_atl08(fobj, bb) -> list[dict]:
    rows = []
    with h5py.File(fobj, "r") as f:
        strong = {"forward": "r", "backward": "l"}  # strong beam side depends on sc_orient
        orient = int(f["orbit_info/sc_orient"][0]) if "orbit_info/sc_orient" in f else -1
        for beam in [k for k in f if k.startswith("gt")]:
            ls = f[f"{beam}/land_segments"]
            lat, lon = ls["latitude_20m"][:], ls["longitude_20m"][:]  # (n, 5)
            hit = window(lat.min(1), lon.min(1), (bb[0] - .01, bb[1] - .01, bb[2] + .01, bb[3] + .01))
            if hit is None:
                continue
            a, b, _ = hit
            g = lambda k: ls[k][a:b]
            h20 = g("terrain/h_te_best_fit_20m")
            c20 = g("canopy/h_canopy_20m")
            seg = dict(
                night=g("night_flag"), snow=g("segment_snowcover"), unc=g("terrain/h_te_uncertainty"),
                cloud=g("cloud_flag_atm"), msw=g("msw_flag"), slope=g("terrain/terrain_slope"),
                cover=g("canopy/segment_cover"), t=g("delta_time"),
            )
            strong_beam = beam.endswith(strong["forward" if orient == 1 else "backward"]) if orient in (0, 1) else False
            for j in range(5):
                rows.append(dict(
                    lat=lat[a:b, j], lon=lon[a:b, j], h=h20[:, j], canopy=c20[:, j],
                    strong=np.full(b - a, strong_beam), **seg,
                ))
    return rows


def read_gedi(fobj, bb) -> list[dict]:
    rows = []
    with h5py.File(fobj, "r") as f:
        for beam in [k for k in f if k.startswith("BEAM")]:
            B = f[beam]
            lat, lon = B["lat_lowestmode"][:], B["lon_lowestmode"][:]
            hit = window(lat, lon, bb)
            if hit is None:
                continue
            a, b, _ = hit
            g = lambda k: B[k][a:b]
            rows.append(dict(
                lat=lat[a:b], lon=lon[a:b], h=g("elev_lowestmode"), canopy=g("rh")[:, 98],
                quality=g("l2a_quality_flag_rel3"), bias_flag=g("elevation_bias_flag"), degrade=g("degrade_flag"),
                sens=g("sensitivity"),
                t=g("delta_time"), power=np.full(b - a, beam in ("BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011")),
            ))
    return rows


def concat(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {k: np.concatenate([np.asarray(r[k]) for r in rows]) for k in keys}


def fetch(short_name, version, reader, bb, workers=4):
    granules = earthaccess.search_data(short_name=short_name, version=version, bounding_box=bb)
    print(f"{short_name}: {len(granules)} granules", flush=True)

    def one(i_g):
        i, g = i_g
        # Open one granule at a time and close it: earthaccess file objects keep
        # their read cache alive, and holding ~100 of them exhausts memory.
        for attempt in range(3):
            try:
                fobj = earthaccess.open([g], show_progress=False)[0]
                try:
                    rows = reader(fobj, bb)
                finally:
                    fobj.close()
                print(f"  {i + 1}/{len(granules)} {sum(r['lat'].size for r in rows)} pts", flush=True)
                return rows
            except Exception as e:  # retry transient 5xx; one bad granule shouldn't sink the run
                err = e
        print(f"  skip {i + 1}: {err}", flush=True)
        return []

    with ThreadPoolExecutor(workers) as ex:
        rows = [r for rs in ex.map(one, enumerate(granules)) for r in rs]
    return concat(rows) if rows else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="bheemdatt", choices=list(AOIS))
    ap.add_argument("--only", choices=["atl08", "gedi"])
    args = ap.parse_args()
    bb = AOIS[args.aoi]
    load_token()
    earthaccess.login(strategy="environment")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (sn, ver, reader) in {"atl08": ("ATL08", "007", read_atl08), "gedi": ("GEDI02_A", "003", read_gedi)}.items():
        if args.only and name != args.only:
            continue
        d = fetch(sn, ver, reader, bb)
        if d is None:
            print(f"{name}: no points")
            continue
        np.savez_compressed(OUT / f"{args.aoi}_{name}.npz", **d)
        print(f"{name}: {d['lat'].size} points -> {OUT / f'{args.aoi}_{name}.npz'}")


if __name__ == "__main__":
    main()
