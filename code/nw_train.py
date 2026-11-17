"""Nationwide DSM -> DTM correction model: one model for all of Nepal (no per-tile seams).

1. Ground points from data/nw/alt (same filters as fuse_hma.load_points), ellipsoid -> EGM2008, assigned to the
   tile whose core contains them; features sampled from that tile's stack (terrain recomputed as in fuse_hma).
2. Held-out scoring, two ways:
     pass    GroupKFold by satellite pass (comparable with the AOI runs)
     spatial GroupKFold by 48 km blocks: whole regions unseen in training (the honest national figure)
3. Final residual model on all points + expected-error model on the spatial out-of-fold |error|.

Outputs: data/nw/points.npz (features, errors, out-of-fold predictions), data/nw/model_*.joblib,
data/nw/train_report.json.

Run:
    OMP_NUM_THREADS=5 ~/workspace/super-resolution-satellite/.venv/bin/python -I nw_train.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from pyproj import Transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuse_hma import EXTRA, FEATURES, STACK_DIR, SUFFIX, WC, load_stack, model, sampler, stats, terrain  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
NW = DATA / "nw"
SIZE = 24000
ZONES = ((-1e9, 300, "terai <300 m"), (300, 1000, "siwalik/inner terai 300-1000 m"), (1000, 2500, "hills 1-2.5 km"),
         (2500, 4000, "mountains 2.5-4 km"), (4000, 1e9, "high himal >4 km"))


def load_points():
    A, G = [], []
    for p in sorted((NW / "alt").glob("*_atl08.npz")):
        a = dict(np.load(p))
        if a["lat"].size == 0:
            continue
        ok = (a["h"] < 1e30) & (a["snow"] == 1) & a["strong"].astype(bool) & (a["msw"] <= 1)
        A.append({k: a[k][ok] for k in ("lat", "lon", "h", "t")})
    for p in sorted((NW / "alt").glob("*_gedi.npz")):
        g = dict(np.load(p))
        if g["lat"].size == 0:
            continue
        ok = (g["sens"] > 0.95) & (g["sens"] <= 1) & g["power"].astype(bool)
        G.append({k: g[k][ok] for k in ("lat", "lon", "h", "t")})
    cat = lambda L, k: np.concatenate([d[k] for d in L])
    p = {k: np.concatenate([cat(A, k), cat(G, k)]) for k in ("lat", "lon", "h", "t")}
    p["src"] = np.r_[np.zeros(sum(d["lat"].size for d in A), int), np.ones(sum(d["lat"].size for d in G), int)]
    # cells overlap only on their edges; drop exact duplicates
    _, u = np.unique(np.c_[p["lat"], p["lon"], p["t"]], axis=0, return_index=True)
    p = {k: v[np.sort(u)] for k, v in p.items()}
    h0 = p["h"].astype("float64")
    _, _, p["h"] = Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True).transform(p["lon"], p["lat"], h0)
    if not np.isfinite(p["h"]).all() or np.nanmedian(np.abs(p["h"] - h0)) < 1:
        raise SystemExit("geoid not applied")
    p["day"] = (p["t"] // 86400).astype(int) * 2 + p["src"]
    print(f"points: ICESat-2 {int((p['src'] == 0).sum()):,}, GEDI {int((p['src'] == 1).sum()):,}", flush=True)
    return p


def assemble():
    p = load_points()
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True).transform(p["lon"], p["lat"])
    tid = np.array([f"E{int(a // SIZE * SIZE / 1000):04d}N{int(b // SIZE * SIZE / 1000):04d}" for a, b in zip(x, y)])
    cols = {k: np.full(x.size, np.nan, "float32") for k in FEATURES + ["cop", "hma"]}
    stacks = sorted((NW / STACK_DIR).glob("E*N*.tif"))
    done = np.zeros(x.size, bool)
    for i, path in enumerate(stacks, 1):
        sel = np.flatnonzero(tid == path.stem)
        if sel.size == 0:
            continue
        prof, L = load_stack(path)
        F = {**terrain(L["dsm"]), "chm_mean": L["chm_mean"], "chm_q3": L["chm_q3"], "chm_max": L["chm_max"],
             "wc": L["wc"], "dsm_minus_cop": L["dsm"] - L["cop"], "src": L["src"], "dsm": L["dsm"],
             "cop": L["cop"], "hma": L["hma"], **{k: L[k] for k in EXTRA}}
        S = sampler(prof["transform"], x[sel], y[sel])
        for k in cols:
            cols[k][sel] = S(F[k], 0 if k in ("wc", "src") else 1)
        done[sel] = True
        if i % 25 == 0:
            print(f"  sampled {i}/{len(stacks)} tiles", flush=True)
    res = cols["dsm"] - p["h"]
    # v2 layers may be missing (strip_nmad needs >=2 strips; chmv2 has gaps): HistGBR takes NaN, so only the
    # v1 features are required
    ok = done & np.isfinite(res) & np.all([np.isfinite(cols[k]) for k in FEATURES if k not in EXTRA], 0) & (np.abs(res) < 50)
    print(f"usable points {ok.sum():,} of {x.size:,} (no stack {(~done).sum():,}, |residual|>=50 m "
          f"{(done & (np.abs(res) >= 50)).sum():,})", flush=True)
    out = dict(lon=p["lon"][ok], lat=p["lat"][ok], h=p["h"][ok], day=p["day"][ok], tile=tid[ok],
               block=(x[ok] // (2 * SIZE)).astype(int) * 1000 + (y[ok] // (2 * SIZE)).astype(int), res=res[ok],
               err_cop=(cols["cop"] - p["h"])[ok], err_hma=(cols["hma"] - p["h"])[ok],
               **{k: cols[k][ok] for k in FEATURES if k != "src"}, src_f=cols["src"][ok], src=p["src"][ok])
    np.savez_compressed(NW / f"points{SUFFIX}.npz", **out)
    return out


def subsets(d):
    s0 = d["src"] == 0
    out = {"all": s0, "tree": s0 & (d["wc"] == 10), "non-tree": s0 & (d["wc"] != 10),
           "slope<20": s0 & (d["slope"] < 20), "slope20-40": s0 & (d["slope"] >= 20) & (d["slope"] < 40),
           "slope>=40": s0 & (d["slope"] >= 40), "on-HMA": s0 & (d["src_f"] == 1),
           "cop-filled": s0 & (d["src_f"] == 2)}
    for lo, hi, nm in ZONES:
        out[nm] = s0 & (d["dsm"] >= lo) & (d["dsm"] < hi)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="reuse data/nw/points.npz")
    ap.add_argument("--max-train", type=int, default=3_000_000, help="cap training points per fit (speed)")
    args = ap.parse_args()
    pp = NW / f"points{SUFFIX}.npz"
    d = dict(np.load(pp)) if args.reuse and pp.exists() else assemble()
    # FEATURES' "src" (DSM source: 1 HMA, 2 Copernicus) was overwritten by the sensor id (0 ICESat-2, 1 GEDI) when
    # both were stored under "src"; keep them apart: d["src"] = sensor, d["src_f"] = DSM source.
    X = np.column_stack([d["src_f"] if k == "src" else d[k] for k in FEATURES]).astype("float32")
    res, src = d["res"], d["src"]
    w = np.where(src == 0, 1.0, 0.5)
    rng = np.random.default_rng(0)
    report = {"points": dict(icesat2=int((src == 0).sum()), gedi=int((src == 1).sum()),
                             passes=int(np.unique(d["day"]).size), tiles=int(np.unique(d["tile"]).size),
                             blocks=int(np.unique(d["block"]).size))}
    print(report["points"], flush=True)
    oof = {}
    for scheme, groups in (("pass", d["day"]), ("spatial", d["block"])):
        pred = np.full(res.size, np.nan, "float32")
        nsp = min(5, np.unique(groups).size)
        if nsp < 2:
            print(f"  {scheme}: only one group, skipped", flush=True)
            oof[scheme] = pred
            continue
        for k, (tr, te) in enumerate(GroupKFold(n_splits=nsp).split(X, res, groups), 1):
            t0 = time.time()
            if tr.size > args.max_train:
                tr = rng.choice(tr, args.max_train, replace=False)
            m = model().fit(X[tr], res[tr], sample_weight=w[tr])
            pred[te] = m.predict(X[te])
            print(f"  {scheme} fold {k}/5: {tr.size:,} train, {te.size:,} test, {time.time() - t0:.0f} s", flush=True)
        oof[scheme] = pred
    d["err_pass"], d["err_spatial"] = res - oof["pass"], res - oof["spatial"]
    report["heldout_icesat2"] = {}
    for sn, sel in subsets(d).items():
        for nm, e in (("copernicus", d["err_cop"]), ("fusion/pass", d["err_pass"]), ("fusion/spatial", d["err_spatial"])):
            ee = e[sel]
            ee = ee[np.isfinite(ee)]
            if ee.size >= 30:
                report["heldout_icesat2"][f"{nm}/{sn}"] = st = stats(ee)
                print(f"  {nm:15s} {sn:32s} n={st['n']:8,d} bias={st['bias']:+6.2f} nmad={st['nmad']:5.2f} "
                      f"rmse={st['rmse']:6.2f} le90={st['le90']:6.2f}", flush=True)
    np.savez_compressed(NW / f"points{SUFFIX}.npz", **d)

    t0 = time.time()
    tr = np.arange(res.size)
    if tr.size > args.max_train:
        tr = rng.choice(tr, args.max_train, replace=False)
    final = model().fit(X[tr], res[tr], sample_weight=w[tr])
    s0 = np.flatnonzero(src == 0)
    unc = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=80,
                                        categorical_features=[FEATURES.index("wc")])
    e_unc = np.where(np.isfinite(d["err_spatial"]), d["err_spatial"], d["err_pass"])  # spatial if it ran
    unc.fit(X[s0], np.abs(e_unc[s0]))
    joblib.dump(final, NW / f"model_residual{SUFFIX}.joblib")
    joblib.dump(unc, NW / f"model_error{SUFFIX}.joblib")
    report["final_fit_s"] = round(time.time() - t0)
    report["features"] = FEATURES
    (NW / f"train_report{SUFFIX}.json").write_text(json.dumps(report, indent=2))
    print("->", NW / f"train_report{SUFFIX}.json", flush=True)


if __name__ == "__main__":
    main()
