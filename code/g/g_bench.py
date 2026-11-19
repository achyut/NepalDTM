"""Speed-up benchmark for G sampling: which cheaper settings keep the full-quality result?

Every setting is scored on the same held-out lidar patches as the paper's evaluation (gen/g_model.py evaluate: 256
patches, degradation seeded per patch, diffusion noise seeded), and timed on the GPU. The 50-step, 4-sample setting is
the full-quality reference and is also written as the checkpoint's eval_XXXXXX.json, so it doubles as the final
evaluation. A setting is acceptable if its RMSE is within TOL of the reference and its slope correlation within TOL_SC.

Settings: DDIM steps (50 / 20 / 10), samples per window (4 / 2 / 1), float16 autocast.
Window overlap (stride) is not tested here: it does not change a window, only how many windows cover an area;
the seam check for a wider stride is a separate run of g/g_apply.py --stride.
Out: data/checkpoints/g_model/bench_XXXXXX.json.   Run: python -u g/g_bench.py [--ckpt ...] (from code/)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from g_model import CK, DEV, Patches, UNet, condition, sample

SETTINGS = [  # (steps, samples, fp16)
    (50, 4, False), (20, 4, False), (20, 2, False), (10, 2, False), (20, 1, False),
    (50, 4, True), (20, 2, True), (10, 2, True),
]
TOL, TOL_SC = 0.01, 0.005  # metres RMSE, slope-correlation points
BS = 8  # windows per GPU batch (as gen/g_apply.py)


def score(net, ds, idx, steps, ens, fp16):
    torch.manual_seed(0)
    errs, sc, t_gpu = [], [], 0.0
    e1, sc1 = [], []  # first sample alone (the paper's 'G, 1 sample' column)
    for k in range(0, len(idx), BS):
        Z, M = zip(*(ds[int(j)] for j in idx[k:k + BS]))
        z, m8 = torch.stack(Z).to(DEV), torch.stack(M).to(DEV)
        if DEV.type == "mps":
            torch.mps.synchronize()
        t = time.time()
        with torch.no_grad(), torch.autocast(DEV.type, dtype=torch.float16, enabled=fp16):
            S = sample(net, m8.repeat_interleave(ens, 0), n_steps=steps).float()
        if DEV.type == "mps":
            torch.mps.synchronize()
        t_gpu += time.time() - t
        S = S.view(len(Z), ens, 256, 256)
        g = S.median(1).values
        for b in range(len(Z)):
            e = (S[b, 0] - z[b, 0]).cpu().numpy(); e1.append(e.ravel())
            gz = np.hypot(*np.gradient(z[b, 0].cpu().numpy())); gc = np.hypot(*np.gradient(S[b, 0].cpu().numpy()))
            sc1.append(float(np.corrcoef(gz.ravel(), gc.ravel())[0, 1]))
            e = (g[b] - z[b, 0]).cpu().numpy()
            errs.append(e.ravel())
            gz = np.hypot(*np.gradient(z[b, 0].cpu().numpy())); gc = np.hypot(*np.gradient(g[b].cpu().numpy()))
            sc.append(float(np.corrcoef(gz.ravel(), gc.ravel())[0, 1]))
    e, f = np.concatenate(errs), np.concatenate(e1)
    one = {"rmse": float(np.sqrt(np.mean(f ** 2))), "nmad": float(1.4826 * np.median(np.abs(f - np.median(f)))),
           "slope_corr": float(np.nanmean(sc1))}
    return {"one_sample": one, "steps": steps, "samples": ens, "fp16": fp16, "rmse": float(np.sqrt(np.mean(e ** 2))),
            "nmad": float(1.4826 * np.median(np.abs(e - np.median(e)))), "slope_corr": float(np.nanmean(sc)),
            "s_per_window": t_gpu / len(idx)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--quick", action="store_true", help="8 patches, 2 cheap settings: a smoke test, writes nothing")
    a = ap.parse_args()
    global SETTINGS
    if a.quick:
        a.n, SETTINGS = 8, [(10, 2, False), (10, 2, True)]
    p = Path(a.ckpt) if a.ckpt else sorted(CK.glob("ema_*.pt"))[-1]
    s = torch.load(p, map_location=DEV)
    net = UNet().to(DEV); net.load_state_dict(s["ema"]); net.eval()
    ds = Patches("val")
    idx = np.random.default_rng(0).choice(len(ds), min(a.n, len(ds)), replace=False)  # = evaluate()'s patches
    print(f"checkpoint {p.name} (step {s['step']}), {len(idx)} held-out patches, device {DEV}", flush=True)

    # baselines on the same patches (cheap, no diffusion)
    base = {}
    for name in ("input_8m_nearest", "bicubic"):
        errs, sc = [], []
        for j in idx:
            z, m8 = ds[int(j)]
            z, m8 = z[None].to(DEV), m8[None].to(DEV)
            up, _, off = condition(m8)
            c = F.interpolate(m8, scale_factor=8, mode="nearest") if name == "input_8m_nearest" else up + off
            e = (c - z)[0, 0].cpu().numpy(); errs.append(e.ravel())
            gz = np.hypot(*np.gradient(z[0, 0].cpu().numpy())); gc = np.hypot(*np.gradient(c[0, 0].cpu().numpy()))
            sc.append(float(np.corrcoef(gz.ravel(), gc.ravel())[0, 1]))
        e = np.concatenate(errs)
        base[name] = {"rmse": float(np.sqrt(np.mean(e ** 2))), "nmad": float(1.4826 * np.median(np.abs(e - np.median(e)))),
                      "slope_corr": float(np.nanmean(sc))}
        print(f"  {name:18s} RMSE {base[name]['rmse']:.3f}  slope corr {base[name]['slope_corr']:.3f}", flush=True)

    rows = []
    for steps, ens, fp16 in SETTINGS:
        r = score(net, ds, idx, steps, ens, fp16)
        rows.append(r)
        print(f"  steps {steps:2d}  samples {ens}  fp16 {str(fp16):5s}  RMSE {r['rmse']:.3f}  slope corr "
              f"{r['slope_corr']:.3f}  {r['s_per_window']:.2f} s/window", flush=True)
    ref = rows[0]
    for r in rows:
        r["speedup"] = ref["s_per_window"] / r["s_per_window"]
        r["ok"] = bool(r["rmse"] <= ref["rmse"] + TOL and r["slope_corr"] >= ref["slope_corr"] - TOL_SC)
    best = max((r for r in rows if r["ok"]), key=lambda r: r["speedup"])
    out = {"ckpt": str(p), "step": int(s["step"]), "n": len(idx), "device": str(DEV), "tol_rmse": TOL,
           "tol_slope_corr": TOL_SC, "baselines": base, "settings": rows, "fastest_ok": best}
    if a.quick:
        print(json.dumps(out["fastest_ok"], indent=1)); return
    (CK / f"bench_{s['step']:06d}.json").write_text(json.dumps(out, indent=2))
    # the reference row is the standard evaluation of this checkpoint
    (CK / f"eval_{s['step']:06d}.json").write_text(json.dumps({"ckpt": str(p), "n": len(idx), "results": {
        **base, "G_1_sample": ref["one_sample"],
        "G_median_4": {k: ref[k] for k in ("rmse", "nmad", "slope_corr")}}}, indent=2))
    print(f"fastest acceptable: {best['steps']} steps, {best['samples']} samples, fp16 {best['fp16']}: "
          f"{best['speedup']:.1f}x ({best['s_per_window']:.2f} s/window), RMSE {best['rmse']:.3f} vs {ref['rmse']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
