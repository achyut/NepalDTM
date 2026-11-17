#!/usr/bin/env python3
"""National joint adjustment of the HMA 8 m strips against ICESat-2 ground photons (M-v2 surface).

Stages (each skips work already on disk):
  ties   per stack tile: photon ties (training tracks, stable open ground) and strip-overlap ties in the core
         -> data/nw/adjust/ties/<id>.npz
  solve  one Huber-IRLS solve for every strip's plane (offset + tilt, centred on the strip)
         -> data/nw/adjust/params.npz, solve.json
  apply  per stack tile: adjusted strip median, count, NMAD on the stack grid (3250 px incl. buffer)
         -> data/nw/adjust/out/<id>.tif (float32, 3 bands)

Validated on Langtang by EXP027 (gated 2 m 8.41 -> 5.88 m; held-out photons 4.03 -> 2.61 m).

Run:
    PROJ_NETWORK=ON ../.venv/bin/python -I -u nw_strip_adjust.py ties [--part k/n]
    PROJ_NETWORK=ON ../.venv/bin/python -I -u nw_strip_adjust.py solve
    PROJ_NETWORK=ON ../.venv/bin/python -I -u nw_strip_adjust.py apply [--part k/n]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strip_adjust as sa  # noqa: E402
from build_stack_hma import CRS  # noqa: E402

STACK = sa.NW / "stack"


def stacks(part: str):
    k, n = map(int, part.split("/"))
    return sorted(STACK.glob("*.tif"))[k::n]


def ties(part: str) -> None:
    (sa.ADJ / "ties").mkdir(parents=True, exist_ok=True)
    sa.strip_index()
    todo = stacks(part)
    for i, p in enumerate(todo, 1):
        out = sa.ADJ / "ties" / f"{p.stem}.npz"
        if out.exists():
            continue
        t0 = time.time()
        d = sa.tile_ties(p, seed=i)
        np.savez_compressed(out, **d)
        print(f"  {i}/{len(todo)} {p.stem}: {d['pk'].size:,} photon, {d['ok'].size:,} overlap ties "
              f"({time.time() - t0:.0f} s)", flush=True)


def solve() -> None:
    files, centre, _ = sa.strip_index()
    parts = [dict(np.load(f)) for f in sorted((sa.ADJ / "ties").glob("*.npz"))]
    allt = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    rng = np.random.default_rng(0)
    for pre, key, cap in (("p", "pk", 25_000_000), ("o", "ok", 25_000_000)):  # fits in memory; still ~10^4 per unknown
        n = allt[key].size
        if n > cap:
            sel = np.sort(rng.choice(n, cap, replace=False))
            for k in [k for k in allt if k.startswith(pre)]:
                allt[k] = allt[k][sel]
    print(f"{len(parts)} tiles, {allt['pk'].size:,} photon ties, {allt['ok'].size:,} overlap ties, "
          f"{len(files)} strips", flush=True)
    params, info = sa.solve_global(allt, centre)
    np.savez(sa.ADJ / "params.npz", params=params, centre=centre)
    info["offset_m"] = {q: float(np.percentile(params[:, 0], q)) for q in (5, 50, 95)}
    (sa.ADJ / "solve.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


def apply(part: str) -> None:
    (sa.ADJ / "out").mkdir(parents=True, exist_ok=True)
    pz = np.load(sa.ADJ / "params.npz")
    params, centre = pz["params"], pz["centre"]
    todo = stacks(part)
    for i, p in enumerate(todo, 1):
        out = sa.ADJ / "out" / f"{p.stem}.tif"
        if out.exists():
            continue
        t0 = time.time()
        T, med, cnt, nmad = sa.tile_composite(p, params, centre)
        tmp = out.with_suffix(".tmp.tif")
        with rasterio.open(tmp, "w", driver="GTiff", width=med.shape[1], height=med.shape[0], count=3,
                           dtype="float32", crs=CRS, transform=T, nodata=np.nan, compress="deflate",
                           predictor=3, tiled=True, blockxsize=512, blockysize=512) as dst:
            for b, (a, nm) in enumerate(((med, "height"), (cnt, "count"), (nmad, "nmad")), 1):
                dst.write(a, b)
                dst.set_band_description(b, nm)
        tmp.replace(out)
        print(f"  {i}/{len(todo)} {p.stem}: valid {np.isfinite(med).mean():.0%} ({time.time() - t0:.0f} s)",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ties", "solve", "apply"])
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    {"ties": lambda: ties(a.part), "solve": solve, "apply": lambda: apply(a.part)}[a.stage]()


if __name__ == "__main__":
    main()
