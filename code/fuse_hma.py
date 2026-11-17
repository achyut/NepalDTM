"""8 m DEM (surface) and DTM (bare earth) for a mountain AOI from HMA 8 m + ICESat-2/GEDI + canopy.

1. Ground points: ICESat-2 ATL08 (strong beams, snow-free, clear or thin cloud) and GEDI L2A, ellipsoid -> EGM2008.
2. Co-register the HMA-based DSM horizontally to ICESat-2 open ground (grid search of the shift).
3. residual = DSM - ground, learned from canopy height, land cover, terrain shape and DSM source with
   gradient boosting; scored with whole satellite passes held out (GroupKFold).
4. DTM = DSM - predicted residual. A second model maps features to the held-out absolute error,
   giving a per-pixel expected error.

Run:
    PROJ_NETWORK=ON ~/workspace/super-resolution-satellite/.venv/bin/python -I fuse_hma.py --aoi bhotekoshi
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

DATA = Path(__file__).resolve().parent / "data"
RES = 8.0
WC = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 70: "snow", 80: "water", 90: "wetland", 100: "moss"}
FEATURES = ["chm_mean", "chm_q3", "chm_max", "wc", "slope", "aspect_sin", "aspect_cos", "tri", "tpi", "tpi_l",
            "dsm_minus_cop", "src", "dsm"]
# National v2 (NW_VERSION=v2): stack_v2/ holds the photon-adjusted strip median (nw_strip_adjust.py) plus strip
# agreement (EXP024) and CHMv2 canopy height (EXP002/EXP022); outputs get a _v2 suffix so v1 stays intact.
V2 = os.environ.get("NW_VERSION") == "v2"
EXTRA = ["strip_count", "strip_nmad", "chmv2"] if V2 else []
FEATURES = FEATURES + EXTRA
SUFFIX = "_v2" if V2 else ""
STACK_DIR = "stack_v2" if V2 else "stack"


# Compact stack encodings (uint16, 0 = nodata): heights at 0.1 m ("dm") or 0.2 m ("h2", fits up to 13,107 m);
# "q4" quarter metres + 1; "u8" integer + 1.
SCALE = {"dm": (0.1, 0), "h2": (0.2, 0), "q4": (0.25, 1), "u8": (1.0, 1)}


def encode(a, enc):
    step, off = SCALE[enc]
    return np.where(np.isfinite(a), np.round(a / step) + off, 0).astype("uint16")


def decode(a, enc):
    step, off = SCALE[enc]
    return np.where(a == 0, np.nan, (a.astype("float32") - off) * step).astype("float32")


def load_stack(path):
    with rasterio.open(path) as s:
        prof, L = s.profile, {}
        for i, d in enumerate(s.descriptions, 1):
            k, enc = d.split(":")
            L[k] = decode(s.read(i), enc)
    return prof, L


def terrain(z):
    gy, gx = np.gradient(z, RES)
    gy = -gy
    aspect = np.arctan2(-gx, -gy)
    pad = np.pad(z, 1, mode="edge")
    tri = np.mean([np.abs(pad[1 + dy:pad.shape[0] - 1 + dy, 1 + dx:pad.shape[1] - 1 + dx] - z)
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx], axis=0)
    return dict(slope=np.degrees(np.arctan(np.hypot(gx, gy))), aspect_sin=np.sin(aspect), aspect_cos=np.cos(aspect),
                tri=tri, tpi=z - ndimage.uniform_filter(z, 13), tpi_l=z - ndimage.uniform_filter(z, 63))


def load_points(aoi):
    pts = []
    a = dict(np.load(DATA / f"{aoi}_atl08.npz"))
    # No h_te_uncertainty cut: in steep terrain it grows with slope (median 66 m here) and would keep only flat
    # valley floors. Strong beams, snow-free, clear/thin cloud (msw <= 1); blunders are cut later at |residual| 50 m.
    ok = (a["h"] < 1e30) & (a["snow"] == 1) & a["strong"].astype(bool) & (a["msw"] <= 1) & (np.abs(a["lat"]) <= 90)
    pts.append(dict(lat=a["lat"][ok], lon=a["lon"][ok], h=a["h"][ok], t=a["t"][ok], src=np.zeros(ok.sum(), int)))
    print(f"ATL08: {ok.sum()} / {ok.size} kept (strong beams, snow-free, msw_flag <= 1)")
    g = dict(np.load(DATA / f"{aoi}_gedi.npz"))
    ok = (g["quality"] == 1) & (g["bias_flag"] == 0) & (g["degrade"] == 0) & (g["sens"] > 0.95) & (g["sens"] <= 1) & g["power"]
    pts.append(dict(lat=g["lat"][ok], lon=g["lon"][ok], h=g["h"][ok], t=g["t"][ok], src=np.ones(ok.sum(), int)))
    print(f"GEDI : {ok.sum()} / {ok.size} kept (power beams, sensitivity > 0.95)")
    p = {k: np.concatenate([d[k] for d in pts]) for k in pts[0]}
    h0 = p["h"].astype("float64")
    _, _, p["h"] = Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True).transform(p["lon"], p["lat"], h0)
    # Without the EGM2008 grid (PROJ_NETWORK=ON or a local copy) PROJ silently returns ellipsoid heights.
    if np.nanmedian(np.abs(p["h"] - h0)) < 1:
        raise SystemExit("geoid not applied: run with PROJ_NETWORK=ON")
    p["day"] = (p["t"] // 86400).astype(int) * 2 + p["src"]
    return p


def sampler(transform, x, y):
    r, c = rowcol(transform, x, y, op=float)
    rc = [np.asarray(r) - 0.5, np.asarray(c) - 0.5]
    return lambda band, order=1, dr=0.0, dc=0.0: ndimage.map_coordinates(
        band, [rc[0] + dr, rc[1] + dc], order=order, mode="constant", cval=np.nan)


def nmad(e):
    e = e[np.isfinite(e)]
    return float(1.4826 * np.median(np.abs(e - np.median(e))))


def stats(e):
    return dict(n=int(e.size), bias=float(np.mean(e)), median=float(np.median(e)), rmse=float(np.sqrt(np.mean(e ** 2))),
                nmad=nmad(e), le90=float(np.percentile(np.abs(e), 90)))


def model():
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=40,
                                         l2_regularization=1.0, categorical_features=[FEATURES.index("wc")],
                                         loss="absolute_error")


def write(path, prof, bands):
    p = prof | dict(driver="COG", count=len(bands), dtype="float32", nodata=np.nan, compress="deflate",
                    predictor=3, BLOCKSIZE=512, OVERVIEWS="AUTO", BIGTIFF="IF_SAFER")
    for k in ("blockxsize", "blockysize", "tiled", "interleave", "zlevel"):
        p.pop(k, None)
    with rasterio.open(path, "w", **p) as dst:
        for i, (name, a) in enumerate(bands.items(), 1):
            dst.write(np.round(a, 2).astype("float32"), i)
            dst.set_band_description(i, name)
    print("->", path, f"{path.stat().st_size / 1e6:.0f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="bhotekoshi")
    ap.add_argument("--heldout-only", action="store_true",
                    help="stop after the held-out evaluation and save per-point errors (no rasters written)")
    args = ap.parse_args()
    prof, L = load_stack(DATA / f"{args.aoi}_stack8m_c.tif")
    T = prof["transform"]
    p = load_points(args.aoi)
    x, y = Transformer.from_crs("EPSG:4326", prof["crs"], always_xy=True).transform(p["lon"], p["lat"])
    S = sampler(T, x, y)
    report = {"grid": dict(width=prof["width"], height=prof["height"], res_m=RES, crs=str(prof["crs"]))}

    # --- 1. horizontal co-registration of HMA to ICESat-2 open ground
    terr0 = terrain(L["dsm"])
    open_ground = (p["src"] == 0) & (S(L["chm_max"], 0) < 2) & (S(terr0["slope"]) > 10) & (S(L["src"], 0) == 1)
    base = S(L["dsm"]) - p["h"]
    print(f"\nco-registration on {open_ground.sum()} ICESat-2 open-ground points on HMA, slope > 10 deg")
    best = (nmad(base[open_ground]), 0.0, 0.0)
    for dr in np.arange(-3, 3.01, 0.25):
        for dc in np.arange(-3, 3.01, 0.25):
            e = S(L["dsm"], 1, dr, dc)[open_ground] - p["h"][open_ground]
            v = nmad(e)
            if v < best[0]:
                best = (v, dr, dc)
    print(f"  NMAD before {nmad(base[open_ground]):.2f} m -> after {best[0]:.2f} m with shift "
          f"rows {best[1]:+.2f} px, cols {best[2]:+.2f} px ({best[2] * RES:+.1f} m E, {-best[1] * RES:+.1f} m N)")
    report["coregistration"] = dict(nmad_before=nmad(base[open_ground]), nmad_after=best[0],
                                    shift_east_m=best[2] * RES, shift_north_m=-best[1] * RES, n=int(open_ground.sum()))
    # Best match samples the DSM at (r + dr, c + dc); ndimage.shift(a, s)(r) = a(r - s), so shift by (-dr, -dc).
    hma0 = L["hma"].copy()
    if best[1] or best[2]:
        s = (-best[1], -best[2])
        L["dsm"] = ndimage.shift(L["dsm"], s, order=1, mode="nearest")
        m = ndimage.shift(np.isfinite(hma0).astype("float32"), s, order=0, mode="constant") > 0.5
        L["hma"] = np.where(m, ndimage.shift(np.where(np.isfinite(hma0), hma0, 0), s, order=1, mode="nearest"), np.nan)
        L["src"] = ndimage.shift(L["src"], s, order=0, mode="nearest").astype("float32")
    # integer bias of the HMA-based surface vs ICESat-2 open ground is left to the model (it learns the offset)

    # --- 2. features
    terr = terrain(L["dsm"])
    F = {**terr, "chm_mean": L["chm_mean"], "chm_q3": L["chm_q3"], "chm_max": L["chm_max"], "wc": L["wc"],
         "dsm_minus_cop": L["dsm"] - L["cop"], "src": L["src"], "dsm": L["dsm"]}
    S = sampler(T, x, y)
    f = {k: S(F[k], 0 if k in ("wc", "src") else 1) for k in FEATURES}
    res = f["dsm"] - p["h"]
    raw_cop = S(L["cop"]) - p["h"]
    raw_hma = S(hma0) - p["h"]
    ok = np.isfinite(res) & np.all([np.isfinite(f[k]) for k in FEATURES], 0) & (np.abs(res) < 50)
    print(f"\npoints inside grid with all features: {ok.sum()}  (dropped |residual| >= 50 m: {(np.abs(res) >= 50).sum()})")
    res, src, day = res[ok], p["src"][ok], p["day"][ok]
    raw_cop, raw_hma = raw_cop[ok], raw_hma[ok]
    X = np.column_stack([f[k][ok] for k in FEATURES])
    wc, slope, dsrc = f["wc"][ok], f["slope"][ok], f["src"][ok]
    report["points"] = dict(atl08=int((src == 0).sum()), gedi=int((src == 1).sum()), passes=int(np.unique(day).size))
    print(f"  ICESat-2 {report['points']['atl08']}, GEDI {report['points']['gedi']}, passes {report['points']['passes']}")

    print("\nDSM minus altimeter ground by land cover (ICESat-2):")
    report["by_class"] = {}
    for cls, nm in WC.items():
        m = (wc == cls) & (src == 0)
        if m.sum() >= 30:
            st = stats(res[m]); report["by_class"][nm] = st
            print(f"  {nm:7s} n={st['n']:6d} median={st['median']:+6.2f} nmad={st['nmad']:5.2f} rmse={st['rmse']:5.2f}")

    # --- 3. held-out evaluation
    pred = np.full(res.size, np.nan)
    pred_off = np.full(res.size, np.nan)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, res, day), 1):
        t0 = time.time()
        m = model()
        m.fit(X[tr], res[tr], sample_weight=np.where(src[tr] == 0, 1.0, 0.5))
        pred[te] = m.predict(X[te])
        pred_off[te] = np.median(res[tr][src[tr] == 0])
        print(f"  fold {k}/5: {len(tr)} train, {len(te)} test, {time.time() - t0:.0f} s", flush=True)
    err = res - pred  # held-out DTM error vs ground
    report["heldout_icesat2"] = {}
    print("\nHeld-out passes, error vs ICESat-2 ground (m):")
    subsets = (("all", src == 0), ("tree", (src == 0) & (wc == 10)), ("non-tree", (src == 0) & (wc != 10)),
               ("slope<20", (src == 0) & (slope < 20)), ("slope20-40", (src == 0) & (slope >= 20) & (slope < 40)),
               ("slope>=40", (src == 0) & (slope >= 40)), ("on-HMA", (src == 0) & (dsrc == 1)), ("strip-filled", (src == 0) & (dsrc == 3)),
               ("cop-filled", (src == 0) & (dsrc == 2)))
    methods = (("copernicus raw", raw_cop), ("HMA raw (valid only)", raw_hma),
               ("DSM co-registered, one offset", res - pred_off), ("fusion DTM", err))
    for nm, e_all in methods:
        for sn, sel in subsets:
            e = e_all[sel]
            e = e[np.isfinite(e)]
            if e.size < 30:
                continue
            st = stats(e); report["heldout_icesat2"][f"{nm}/{sn}"] = st
            print(f"  {nm:30s} {sn:10s} n={st['n']:6d} bias={st['bias']:+6.2f} nmad={st['nmad']:5.2f} rmse={st['rmse']:6.2f} le90={st['le90']:6.2f}")
    # per-point held-out errors, for benchmarking other DEMs on the same points (benchmark_external.py)
    np.savez_compressed(DATA / f"{args.aoi}_heldout.npz", lon=p["lon"][ok], lat=p["lat"][ok], h=p["h"][ok], src=src,
                        day=day, wc=wc, slope=slope, dsrc=dsrc, err_fusion=err, err_cop=raw_cop)
    if args.heldout_only:
        return

    # --- 4. final models and grids
    m = model().fit(X, res, sample_weight=np.where(src == 0, 1.0, 0.5))
    u = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=80,
                                      categorical_features=[FEATURES.index("wc")])
    sel = src == 0
    u.fit(X[sel], np.abs(err[sel]))
    H, W = L["dsm"].shape
    corr = np.full(H * W, np.nan, "float32")
    unc = np.full(H * W, np.nan, "float32")
    cols = [F[k].ravel() for k in FEATURES]
    for a in range(0, H * W, 2_000_000):
        G = np.column_stack([c[a:a + 2_000_000] for c in cols])
        v = np.all(np.isfinite(G), 1)
        idx = np.flatnonzero(v) + a
        corr[idx] = m.predict(G[v]); unc[idx] = u.predict(G[v])
    corr, unc = corr.reshape(H, W), np.maximum(unc.reshape(H, W), 0)
    dtm = (L["dsm"] - corr).astype("float32")
    report["grid_correction"] = dict(mean=float(np.nanmean(corr)), p5=float(np.nanpercentile(corr, 5)),
                                     p95=float(np.nanpercentile(corr, 95)))
    report["expected_abs_error"] = dict(mean=float(np.nanmean(unc)), p50=float(np.nanpercentile(unc, 50)),
                                        p95=float(np.nanpercentile(unc, 95)))
    # how much terrain detail finer than Copernicus' 30 m the surface carries
    hp = L["dsm"] - ndimage.gaussian_filter(L["dsm"], 30 / RES / 2.355 * 2)
    hp_cop = L["cop"] - ndimage.gaussian_filter(L["cop"], 30 / RES / 2.355 * 2)
    report["detail_finer_than_30m_rms"] = dict(dsm8m=float(np.nanstd(hp)), cop30m_resampled=float(np.nanstd(hp_cop)))
    print(f"\ncorrection mean {report['grid_correction']['mean']:+.2f} m (p5 {report['grid_correction']['p5']:+.2f}, "
          f"p95 {report['grid_correction']['p95']:+.2f}); expected |error| median {report['expected_abs_error']['p50']:.2f} m")
    print(f"sub-30 m detail (RMS of high-pass): 8 m DSM {np.nanstd(hp):.2f} m vs Copernicus {np.nanstd(hp_cop):.2f} m")

    base = dict(prof)
    write(DATA / f"{args.aoi}_dem_8m.tif", base, {"dsm_egm2008": L["dsm"], "source_1hma_2cop": L["src"]})
    write(DATA / f"{args.aoi}_dtm_8m.tif", base, {"dtm_egm2008": dtm, "removed_m": corr, "expected_abs_error_m": unc})
    (DATA / f"{args.aoi}_report.json").write_text(json.dumps(report, indent=2))
    print("->", DATA / f"{args.aoi}_report.json")


if __name__ == "__main__":
    main()
