"""Phase 2 (G): the error model of the measured 8 m product M, and degrade() to imitate it on clean lidar.

EXP026 showed that super-resolution trained on clean block means does not transfer to the real product, so G is
trained on inputs that carry M's actual errors. Fitted from Langtang (v2 DEM vs the gated 2 m composite, stereo and
void cells separately) and from held-out photons (DTM by land cover, validation_v1_v2.json):
  blur_px        Gaussian sigma (8 m px) that best maps the 8 m block mean of the truth to M (matching softness)
  sigma(slope)   std of the remaining stereo error per slope bin (robust, NMAD)
  corr_px        e-folding length of that error (exponential fit to its autocorrelation)
  void_p(slope)  share of void cells per slope bin;  void_sigma  error in voids (re-admitted fill, EXP030)
  forest_bias, forest_sigma   extra DTM error under forest, from held-out photons (tree minus non-tree)
Check (plan): degraded lidar patches must reproduce Nepal's NMAD per slope bin within 10%.

Out: data/g/error_model.json.  Run: PROJ_NETWORK=ON ../.venv/bin/python -I -u g/error_model.py [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, optimize

ROOT = Path(__file__).resolve().parent.parent
NW = ROOT / "data" / "nw"
OUT = ROOT / "data" / "gen" / "error_model.json"
SLOPE_BINS = (0, 10, 20, 30, 40, 50, 90)


def nmad(e):
    e = e[np.isfinite(e)]
    return float(1.4826 * np.median(np.abs(e - np.median(e)))) if e.size else float("nan")


def slope_deg(z, res):
    gy, gx = np.gradient(z, res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def langtang_pairs():
    """(truth at 8 m = 2 m composite block mean, M = v2 DEM, src, frac) on the v2 stack grid of the main tile."""
    os.environ["NW_VERSION"] = "v2"
    sys.path.insert(0, str(ROOT))
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from build_stack_hma import CRS, geoid_offset
    from fuse_hma import load_stack
    ref_p = next((NW / "zenodo" / "langtang").rglob("*_median_gated_v1.0.tif"))
    prof, L = load_stack(NW / "stack_v2" / "E0336N3120.tif")
    T, W, H = prof["transform"], prof["width"], prof["height"]
    ref = np.full((H, W), np.nan, "float32"); frac = ref.copy(); m = ref.copy()
    with rasterio.open(ref_p) as s:
        a = s.read(1).astype("float32"); v = (a != s.nodata).astype("float32"); a[v == 0] = np.nan
        reproject(a, ref, src_transform=s.transform, src_crs=s.crs, dst_transform=T, dst_crs=CRS, resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
        reproject(v, frac, src_transform=s.transform, src_crs=s.crs, dst_transform=T, dst_crs=CRS, resampling=Resampling.average, dst_nodata=np.nan)
    ref += geoid_offset(T, W, H)
    with rasterio.open(NW / "nepal_dem_8m_v2.vrt") as s:
        reproject(rasterio.band(s, 1), m, dst_transform=T, dst_crs=CRS, resampling=Resampling.nearest, dst_nodata=np.nan)
    rows, cols = np.where(frac >= 0.999)
    sl = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    return ref[sl], m[sl], L["src"][sl], frac[sl]


def fit():
    ref, M, src, frac = langtang_pairs()
    full = (frac >= 0.999) & np.isfinite(ref) & np.isfinite(M)
    stereo = full & (src == 1)
    void = full & (src == 2)
    # a bounding box of fully covered cells: fill the few gaps so the filters run
    reff = np.where(np.isfinite(ref), ref, np.nanmedian(ref))
    slope = slope_deg(reff, 8.0)

    def loss(s):
        e = (M - ndimage.gaussian_filter(reff, s[0]))[stereo]
        return float(np.mean(np.abs(e - np.median(e))))
    blur = float(optimize.minimize_scalar(lambda s: loss([s]), bounds=(0.01, 3.0), method="bounded").x)
    r = np.where(stereo, M - ndimage.gaussian_filter(reff, blur), np.nan)
    r -= np.nanmedian(r)
    sig = {}
    for lo, hi in zip(SLOPE_BINS[:-1], SLOPE_BINS[1:]):
        b = stereo & (slope >= lo) & (slope < hi)
        sig[f"{lo}-{hi}"] = {"nmad": nmad(r[b]), "n": int(b.sum()),
                             "void_p": float((void & (slope >= lo) & (slope < hi)).sum() / max(1, (full & (slope >= lo) & (slope < hi)).sum()))}
    # autocorrelation of the stereo error (FFT, masked), first 40 px
    z = np.where(np.isfinite(r), r, 0.0); w = np.isfinite(r).astype(float)
    F = lambda a: np.fft.rfft2(a, s=(2 * a.shape[0], 2 * a.shape[1]))
    num = np.fft.irfft2(F(z) * np.conj(F(z)))[:40, :40]
    den = np.fft.irfft2(F(w) * np.conj(F(w)))[:40, :40]
    ac = num / np.maximum(den, 1)
    lag = np.hypot(*np.mgrid[:40, :40])
    prof = np.array([ac[(lag >= k - .5) & (lag < k + .5)].mean() for k in range(20)]) / ac[0, 0]
    corr = float(optimize.curve_fit(lambda d, L: np.exp(-d / L), np.arange(20), prof, p0=[2.0], bounds=(0.1, 100))[0][0])
    model = {"blur_px": blur, "corr_px": corr, "slope_bins": sig, "autocorr": prof.round(4).tolist(),
             "void_sigma": nmad((M - reff)[void]), "stereo_bias_m": float(np.nanmedian((M - reff)[stereo]))}
    vp = NW / "validation_v1_v2.json"
    if vp.exists():
        P = json.loads(vp.read_text())["photons_holdout_dtm"]
        t, n = P["tree"]["v2"], P["non-tree"]["v2"]
        model["forest_bias"] = t["median"] - n["median"]
        model["forest_sigma"] = float(np.sqrt(max(t["nmad"] ** 2 - n["nmad"] ** 2, 0)))
        model["photon_nmad"] = {k: v["v2"]["nmad"] for k, v in P.items() if v.get("v2")}
    else:
        model["forest_bias"] = model["forest_sigma"] = None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(model, indent=2) + "\n")
    print(json.dumps({k: v for k, v in model.items() if k != "autocorr"}, indent=1))
    return model


def degrade(truth_1m: np.ndarray, model: dict, forest: np.ndarray | None = None, rng=None) -> np.ndarray:
    """Clean 1 m DTM (H, W; multiples of 8) -> imitation of M at 8 m (H/8, W/8)."""
    rng = rng or np.random.default_rng()
    h, w = truth_1m.shape[0] // 8, truth_1m.shape[1] // 8
    z8 = truth_1m[:h * 8, :w * 8].reshape(h, 8, w, 8).mean((1, 3))
    m = ndimage.gaussian_filter(z8, model["blur_px"])
    sl = slope_deg(z8, 8.0)
    sig = np.zeros_like(z8)
    vp = np.zeros_like(z8)
    for k, b in model["slope_bins"].items():
        lo, hi = map(float, k.split("-"))
        sel = (sl >= lo) & (sl < hi)
        sig[sel] = b["nmad"] if np.isfinite(b["nmad"]) else 0
        vp[sel] = b["void_p"]
    n = ndimage.gaussian_filter(rng.standard_normal(z8.shape), model["corr_px"] / 1.414)
    n /= n.std() + 1e-9
    m = m + sig * n
    # voids: blobs with probability void_p; heights there come from a much coarser, noisier fill
    blob = ndimage.gaussian_filter(rng.random(z8.shape), 2.0)
    void = blob > np.quantile(blob, 1 - float(np.clip(vp.mean(), 0, 0.9)))
    coarse = ndimage.gaussian_filter(z8, 30 / 8 / 2) + model["void_sigma"] * ndimage.gaussian_filter(rng.standard_normal(z8.shape), 3)
    m = np.where(void, coarse, m)
    if forest is not None and model.get("forest_bias") is not None:
        f8 = forest[:h * 8, :w * 8].reshape(h, 8, w, 8).mean((1, 3))
        m = m + f8 * (model["forest_bias"] + model["forest_sigma"] * ndimage.gaussian_filter(rng.standard_normal(z8.shape), 1.0))
    return m.astype("float32")


def check(model: dict, n_patches=3000):
    """Degraded lidar patches: NMAD of (degraded - block mean) per slope bin vs Nepal's (target: within 10%)."""
    pats = sorted((ROOT / "data" / "gen" / "patches").glob("*.npz"))
    rng = np.random.default_rng(0)
    E = {k: [] for k in model["slope_bins"]}
    for p in pats:
        d = np.load(p)
        idx = rng.choice(len(d["zmin"]), min(len(d["zmin"]), n_patches // len(pats) + 1), replace=False)
        for i in idx:
            z = d["z"][i].astype("float32") * d["scale"][i] / 100 + d["zmin"][i]
            m = degrade(z, model, rng=rng)
            z8 = z.reshape(32, 8, 32, 8).mean((1, 3))
            e = m - ndimage.gaussian_filter(z8, model["blur_px"])
            sl = slope_deg(z8, 8.0)
            for k in E:
                lo, hi = map(float, k.split("-"))
                E[k].append(e[(sl >= lo) & (sl < hi)])
    rep = {}
    for k, b in model["slope_bins"].items():
        got = nmad(np.concatenate(E[k])) if E[k] else float("nan")
        rep[k] = {"nepal": b["nmad"], "degraded": got, "rel": got / b["nmad"] - 1 if b["nmad"] else float("nan")}
        print(f"  slope {k:6s} Nepal NMAD {b['nmad']:.2f}  degraded {got:.2f}  ({rep[k]['rel']:+.0%})")
    ok = all(abs(v["rel"]) < 0.10 for v in rep.values() if np.isfinite(v["rel"]) and model["slope_bins"][k]["n"] > 1000)
    print("CHECK", "PASSED" if ok else "FAILED (needs per-bin noise calibration)")
    return rep, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    mdl = fit() if not (a.check and OUT.exists()) else json.loads(OUT.read_text())
    if a.check:
        check(mdl)
