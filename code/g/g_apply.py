#!/usr/bin/env python3
"""Apply G to a real area: v2 8 m DTM -> generated 1 m DTM over a box, written as a COG.

The box is cut into 256 m windows (32 x 32 cells at 8 m) with STRIDE overlap. Each window gets the median of ENS
DDIM samples, each projected so its 8 m block means equal the v2 cells exactly (sample(project=True)). Overlaps are
blended with a raised-cosine weight, so seams fade over (256 - STRIDE) m.

Score: held-out ICESat-2 photons inside the box (never used to fit v2), for v2 8 m (nearest), bicubic 1 m and G 1 m.
G cannot add accuracy the 8 m input does not have; the check is that it does not lose any while adding texture.

Run: ../.venv/bin/python -I -u g/g_apply.py --name chitwan --bbox 84.40 27.55 84.50 27.65 [--ens 4] [--ckpt ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "g"))
sys.path.insert(0, str(ROOT))
from g_model import CK, DEV, UNet, condition, project_area, sample  # noqa: E402

NW = ROOT / "data" / "nw"
OUT = ROOT / "data" / "gen" / "apply"
DTM = NW / "nepal_dtm_8m_v2.vrt"
WIN, STRIDE = 32, 24  # window and step in 8 m cells (256 m windows, 64 m overlap)


def feather(n=256, ov=(WIN - STRIDE) * 8):
    r = np.ones(n, "float32")
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ov + 2)[1:-1])
    r[:ov], r[-ov:] = ramp, ramp[::-1]
    return np.outer(r, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"))
    ap.add_argument("--ens", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--ckpt")
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps per sample")
    ap.add_argument("--stride", type=int, default=24, help="window step in 8 m cells (24 = 192 m, 64 m overlap)")
    ap.add_argument("--fp16", action="store_true", help="half-precision sampling (MPS/CUDA autocast)")
    a = ap.parse_args()
    global STRIDE
    STRIDE = a.stride
    OUT.mkdir(parents=True, exist_ok=True)
    p = Path(a.ckpt) if a.ckpt else sorted(CK.glob("ema_*.pt"))[-1]
    net = UNet().to(DEV)
    s = torch.load(p, map_location=DEV)
    net.load_state_dict(s["ema"]); net.eval()
    print(f"checkpoint {p.name} (step {s['step']}), device {DEV}", flush=True)

    with rasterio.open(DTM) as src:
        crs = src.crs
        b = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform_bounds(*a.bbox)
        w = from_bounds(*b, src.transform).round_offsets().round_lengths()
        m8 = src.read(1, window=w).astype("float32")
        T8 = src.window_transform(w)
    H8, W8 = m8.shape
    # whole windows only: trim to a multiple of the stride plus one window
    ny, nx = (H8 - WIN) // STRIDE + 1, (W8 - WIN) // STRIDE + 1
    H8, W8 = (ny - 1) * STRIDE + WIN, (nx - 1) * STRIDE + WIN
    m8 = m8[:H8, :W8]
    bad = ~np.isfinite(m8)
    if bad.any():  # rare voids: nearest valid height, flagged in the output mask
        from scipy import ndimage
        m8 = m8[tuple(ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True))]
    print(f"{a.name}: {W8 * 8 / 1000:.1f} x {H8 * 8 / 1000:.1f} km, {ny * nx} windows x {a.ens} samples", flush=True)

    acc = np.zeros((H8 * 8, W8 * 8), "float32"); wsum = np.zeros_like(acc)
    fw = feather(ov=(WIN - STRIDE) * 8)
    jobs = [(i, j) for i in range(ny) for j in range(nx)]
    t0 = time.time()
    for k in range(0, len(jobs), a.bs):
        chunk = jobs[k:k + a.bs]
        x = torch.from_numpy(np.stack([m8[i * STRIDE:i * STRIDE + WIN, j * STRIDE:j * STRIDE + WIN] for i, j in chunk]))
        x = x[:, None].to(DEV).repeat_interleave(a.ens, 0)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=a.fp16):
            z = sample(net, x, n_steps=a.steps).float()
        z = z.view(len(chunk), a.ens, 256, 256).median(1).values.cpu().numpy()
        for (i, j), zz in zip(chunk, z):
            r, c = i * STRIDE * 8, j * STRIDE * 8
            acc[r:r + 256, c:c + 256] += zz * fw
            wsum[r:r + 256, c:c + 256] += fw
        if (k // a.bs) % 10 == 0:
            done = k + len(chunk)
            print(f"  {done}/{len(jobs)} windows, {(time.time() - t0) / done * (len(jobs) - done) / 60:.0f} min left", flush=True)
    g1 = project_area(acc / wsum, m8).astype("float32")  # block means = M after median and blending
    g1[np.kron(bad[:H8, :W8], np.ones((8, 8), bool))] = np.nan

    T1 = from_origin(T8.c, T8.f, 1.0, 1.0)
    prof = dict(driver="COG", width=g1.shape[1], height=g1.shape[0], count=1, dtype="float32", crs=crs, transform=T1,
                nodata=np.nan, compress="deflate", predictor=3, blocksize=512)
    f1 = OUT / f"{a.name}_G_1m.tif"
    with rasterio.open(f1, "w", **prof) as o:
        o.write(g1, 1)
    up = condition(torch.from_numpy(m8)[None, None])[0] + torch.from_numpy(m8)[None, None].mean()
    bic = up[0, 0].numpy().astype("float32")
    with rasterio.open(OUT / f"{a.name}_bicubic_1m.tif", "w", **prof) as o:
        o.write(bic, 1)
    with rasterio.open(OUT / f"{a.name}_v2_8m.tif", "w", **(prof | dict(width=W8, height=H8, transform=T8))) as o:
        o.write(np.where(bad, np.nan, m8).astype("float32"), 1)
    print(f"-> {f1} ({g1.shape[1]} x {g1.shape[0]} px)", flush=True)

    # held-out photons in the box
    import pandas as pd
    to = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    hs, xs, ys = [], [], []
    for f in sorted((NW / "atl03").glob("*.parquet")):
        d = pd.read_parquet(f, columns=["lon", "lat", "h", "holdout"])
        d = d[d.holdout.astype(bool) & d.lon.between(a.bbox[0], a.bbox[2]) & d.lat.between(a.bbox[1], a.bbox[3])]
        if len(d):
            x_, y_ = to.transform(d.lon.values, d.lat.values)
            hs.append(d.h.values); xs.append(x_); ys.append(y_)
    R = {"checkpoint": p.name, "step": int(s["step"]), "bbox": a.bbox, "ens": a.ens}
    if hs:
        h, x_, y_ = map(np.concatenate, (hs, xs, ys))
        def at(arr, T):
            inv = ~T
            c = np.floor(inv.a * x_ + inv.b * y_ + inv.c).astype(int); r = np.floor(inv.d * x_ + inv.e * y_ + inv.f).astype(int)
            ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
            v = np.full(x_.size, np.nan, "float32"); v[ok] = arr[r[ok], c[ok]]
            return v
        E = {"v2 8 m": at(np.where(bad, np.nan, m8), T8) - h, "bicubic 1 m": at(bic, T1) - h, "G 1 m": at(g1, T1) - h}
        m = np.all([np.isfinite(e) for e in E.values()], 0)
        R["photons"] = {}
        for k, e in E.items():
            e = e[m]; md = np.median(e)
            R["photons"][k] = {"n": int(e.size), "rmse": float(np.sqrt(np.mean(e ** 2))),
                               "nmad": float(1.4826 * np.median(np.abs(e - md))), "median": float(md)}
            print(f"  held-out photons {k:12s} RMSE {R['photons'][k]['rmse']:.2f}  NMAD {R['photons'][k]['nmad']:.2f}  "
                  f"median {md:+.2f}  (n={e.size:,})", flush=True)
    # texture: share of relief G adds below 8 m, against bicubic
    R["subgrid_std_m"] = {"bicubic": float(np.nanstd(bic - np.kron(m8, np.ones((8, 8))))),
                          "G": float(np.nanstd(g1 - np.kron(m8, np.ones((8, 8)))))}
    (OUT / f"{a.name}_report.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R["subgrid_std_m"]))


if __name__ == "__main__":
    main()
