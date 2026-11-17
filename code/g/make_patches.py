"""Phase 3 (G): clean 1 m training patches from the swissALTI3D 0.5 m and USGS 3DEP 1 m tiles.

Each patch is 256 x 256 at 1 m (the G target); the 8 m input (32 x 32) is made at training time by the error
model (g/error_model.py), so only the clean truth is stored here. Swiss 0.5 m tiles are averaged 2 x 2 to 1 m.
Patches with any nodata are skipped. Heights are stored as int16 centimetres above the patch minimum (relief up
to 327 m per patch; steeper patches are kept at reduced precision via a per-patch scale), plus the minimum.

Out: data/gen/patches/<region>_<k>.npz with z (n,256,256) int16, zmin (n,) f4, scale (n,) f4, slope_mean (n,),
relief (n,), src (n,) str.  Run: ../.venv/bin/python -I -u g/make_patches.py [--workers 8]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "gen" / "raw"
OUT = ROOT / "data" / "gen" / "patches"
P = 256


def patches_from(arr: np.ndarray, nodata) -> list[np.ndarray]:
    out = []
    H, W = arr.shape
    for r in range(0, H - P + 1, P):
        for c in range(0, W - P + 1, P):
            a = arr[r:r + P, c:c + P]
            if nodata is not None and (a == nodata).any() or not np.isfinite(a).all() or (a < -500).any():
                continue
            out.append(a.astype("float32"))
    return out


def one_tile(path: Path) -> list[np.ndarray]:
    with rasterio.open(path) as s:
        a = s.read(1)
        nod = s.nodata
        res = s.res[0]
    if res < 0.75:  # 0.5 m -> 1 m block mean
        valid = a != nod
        h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
        a, valid = a[:h, :w], valid[:h, :w]
        blk = a.reshape(h // 2, 2, w // 2, 2)
        ok = valid.reshape(h // 2, 2, w // 2, 2).all((1, 3))
        a = np.where(ok, blk.mean((1, 3)), nod).astype("float32")
    return patches_from(a, nod)


def encode(ps: list[np.ndarray], src: str) -> dict:
    z = np.stack(ps)
    zmin = z.min((1, 2))
    rel = z - zmin[:, None, None]
    relief = rel.max((1, 2))
    scale = np.maximum(1.0, np.ceil(relief * 100 / 32000))  # cm per unit
    q = np.round(rel * 100 / scale[:, None, None]).astype("int16")
    gy, gx = np.gradient(z, axis=(1, 2))
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).mean((1, 2))
    return {"z": q, "zmin": zmin.astype("f4"), "scale": scale.astype("f4"), "slope_mean": slope.astype("f4"),
            "relief": relief.astype("f4"), "src": np.full(len(ps), src)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=2048)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {}
    for f in sorted((RAW / "swiss").glob("*.tif")):
        groups.setdefault("swiss", []).append(f)
    for f in sorted((RAW / "3dep").glob("*.tif")):
        groups.setdefault(f.name.split("__")[0], []).append(f)
    total = 0
    for region, files in groups.items():
        if any(OUT.glob(f"{region}_*.npz")):
            print(f"{region}: already done, skipping", flush=True)
            continue
        buf, k = [], 0
        with ProcessPoolExecutor(a.workers) as ex:
            for i, ps in enumerate(ex.map(one_tile, files, chunksize=4), 1):
                buf += ps
                while len(buf) >= a.shard:
                    np.savez(OUT / f"{region}_{k:03d}.npz", **encode(buf[:a.shard], region))
                    buf, k, total = buf[a.shard:], k + 1, total + a.shard
                if i % 200 == 0 or i == len(files):
                    print(f"{region}: {i}/{len(files)} tiles, {total + len(buf):,} patches", flush=True)
        if buf:
            np.savez(OUT / f"{region}_{k:03d}.npz", **encode(buf, region))
            total += len(buf)
    print(f"done: {total:,} new patches", flush=True)


if __name__ == "__main__":
    main()
