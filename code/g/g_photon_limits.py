"""Why point photons cannot score 1 m detail: two tests for the G paper.

1. Jitter test on held-out lidar: true heights at random points, compared with the true 1 m surface and with bicubic
   of its 8 m block means, each read at the point displaced by a Gaussian horizontal error. If the perfect surface
   scores no better than bicubic at realistic errors, photon RMSE cannot reward correct 1 m relief.
2. Detail scan in Nepal: photon RMSE of bicubic + a (G - bicubic) for a in 0..1 on the final-model apply runs.

Out: data/gen/apply/photon_limits.json.  Run: python -u g/g_photon_limits.py (from code/)
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from pyproj import Transformer
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "data" / "gen"
APPLY = GEN / "apply"
BOXES = {"chitwan_fast_s24": (84.40, 27.55, 84.50, 27.65), "churia_fast_s24": (84.45, 27.46, 84.50, 27.51)}
SIGMAS = (0, 1, 2, 3, 4, 6)
ALPHAS = (0, 0.25, 0.5, 0.75, 1.0)


def jitter_test(n_patches=400, pts=200, seed=0):
    meta = np.load(GEN / "patches_meta.npz", allow_pickle=True)
    n = int(meta["n"])
    mm = np.memmap(GEN / "patches_all.int16", "int16", "r", shape=(n, 256, 256))
    rng = np.random.default_rng(seed)
    acc = {s: {"truth": [], "bicubic": []} for s in SIGMAS}
    for i in np.arange(0, n, 20)[:n_patches]:  # held-out patches: every 20th
        z = mm[i].astype("f4") * meta["scale"][i] / 100
        m8 = z.reshape(32, 8, 32, 8).mean((1, 3))
        bic = F.interpolate(torch.from_numpy(m8)[None, None], scale_factor=8, mode="bicubic", align_corners=False)[0, 0].numpy()
        py, px = rng.uniform(20, 236, pts), rng.uniform(20, 236, pts)
        h = map_coordinates(z, [py, px], order=1)
        for s in SIGMAS:
            qy, qx = py + rng.normal(0, s, pts), px + rng.normal(0, s, pts)
            acc[s]["truth"].append(map_coordinates(z, [qy, qx], order=1) - h)
            acc[s]["bicubic"].append(map_coordinates(bic, [qy, qx], order=1) - h)
    return {str(s): {k: float(np.sqrt(np.mean(np.concatenate(v) ** 2))) for k, v in d.items()} for s, d in acc.items()}


def detail_scan():
    out = {}
    for name, b in BOXES.items():
        ps = []
        for f in sorted((ROOT / "data" / "nw" / "atl03").glob("*.parquet")):
            p = pd.read_parquet(f, columns=["lon", "lat", "h", "holdout"])
            p = p[p.holdout.astype(bool) & p.lon.between(b[0], b[2]) & p.lat.between(b[1], b[3])]
            if len(p):
                ps.append(p)
        p = pd.concat(ps)
        h = p.h.values

        def at(suffix):
            with rasterio.open(APPLY / f"{name}_{suffix}.tif") as s:
                x, y = Transformer.from_crs("EPSG:4326", s.crs, always_xy=True).transform(p.lon.values, p.lat.values)
                r, c = (np.array(v) for v in rasterio.transform.rowcol(s.transform, x, y))
                z = s.read(1)
            ok = (r >= 0) & (r < z.shape[0]) & (c >= 0) & (c < z.shape[1])
            o = np.full(len(h), np.nan)
            o[ok] = z[r[ok], c[ok]]
            return o

        g, bi, m = at("G_1m"), at("bicubic_1m"), at("v2_8m")
        k = np.isfinite(g + bi + m)
        rmse = lambda e: float(np.sqrt(np.mean(e[k] ** 2)))
        out[name] = {"n": int(k.sum()), "M": rmse(m - h),
                     "alpha": {str(a): rmse(bi + a * (g - bi) - h) for a in ALPHAS}}
    return out


if __name__ == "__main__":
    R = {"jitter_rmse_m": jitter_test(), "detail_scan_rmse_m": detail_scan()}
    (APPLY / "photon_limits.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))
