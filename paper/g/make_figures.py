#!/usr/bin/env python3
"""Figures for the G paper (reports/g_v1). Reads the training log, the checkpoint evaluations and the Chitwan/Churia
apply outputs; writes PDF + PNG into figures/.

Run from the repo root: python paper/g/make_figures.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "code" / "data" / "checkpoints" / "g_model"
APPLY = ROOT / "code" / "data" / "gen" / "apply"
OUT = Path(__file__).resolve().parent / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def hillshade(z, px=1.0, az=315, alt=45):
    gy, gx = np.gradient(z, px)
    s, a = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    az, alt = np.radians(az), np.radians(alt)
    return np.clip(np.sin(alt) * np.cos(s) + np.cos(alt) * np.sin(s) * np.cos(az - a), 0, 1)


def fig_loss():
    steps, loss = [], []
    logs = [ROOT / "code" / "data" / "gen" / "g_train.log", CK / "train.log"]  # steps 0-24k, then from the resume
    for line in "\n".join(p.read_text() for p in logs if p.exists()).splitlines():
        m = re.match(r"step (\d+)\s+loss ([\d.]+)", line)
        if m:
            steps.append(int(m.group(1))); loss.append(float(m.group(2)))
    s, l = np.array(steps), np.array(loss)
    o = np.argsort(s, kind="stable"); s, l = s[o], l[o]
    _, last = np.unique(s[::-1], return_index=True)  # one value per step (restarts re-log a few steps)
    keep = len(s) - 1 - last
    s, l = s[keep], l[keep]
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.plot(s / 1000, l, lw=0.8, color="#003893")
    ax.set_xlabel("training step (thousands)"); ax.set_ylabel("v-prediction loss (running mean)")
    ax.set_ylim(0, min(0.3, l.max() * 1.05))
    save(fig, "fig01_training_loss")


def fig_eval():
    rows = []
    for f in sorted(CK.glob("eval_0*.json")):
        d = json.loads(f.read_text())
        step = int(f.stem.split("_")[1])
        if step < 1000:
            continue
        r = d["results"]
        gm = next(k for k in r if k.startswith("G_median"))
        rows.append((step, d["n"], r["bicubic"]["rmse"], r[gm]["rmse"], r["G_1_sample"]["rmse"],
                     r["bicubic"].get("slope_corr"), r[gm].get("slope_corr")))
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 2.8))
    x = np.arange(len(rows)); w = 0.27
    lab = [f"{s // 1000}k\n(n={n})" for s, n, *_ in rows]
    ax[0].bar(x - w, [r[2] for r in rows], w, label="bicubic", color="#9aa7b8")
    ax[0].bar(x, [r[4] for r in rows], w, label="G, 1 sample", color="#7fa3d6")
    ax[0].bar(x + w, [r[3] for r in rows], w, label="G, median of 4", color="#003893")
    ax[0].set_xticks(x, lab); ax[0].set_ylabel("RMSE against 1 m lidar (m)"); ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_ylim(1.0, max(r[2] for r in rows) * 1.08)
    gain = [100 * (1 - r[3] / r[2]) for r in rows]
    ax[1].bar(x - w / 2, gain, w, color="#003893", label="RMSE gain of G over bicubic (%)")
    sc = [(r[5], r[6]) for r in rows]
    ax2 = ax[1].twinx()
    ax2.plot(x + w / 2, [b for b, _ in sc], "o", color="#9aa7b8", label="slope-map corr., bicubic")
    ax2.plot(x + w / 2, [g for _, g in sc], "o", color="#dc143c", label="slope-map corr., G")
    ax[1].set_xticks(x, lab); ax[1].set_ylabel("RMSE gain over bicubic (%)"); ax2.set_ylabel("slope-map correlation")
    ax2.set_ylim(0.5, 0.75); ax2.spines["right"].set_visible(True)
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "fig02_heldout_patches")


def fig_compare(name, r0, c0, n, zoom, out, note):
    g = rasterio.open(APPLY / f"{name}_G_1m.tif").read(1)
    b = rasterio.open(APPLY / f"{name}_bicubic_1m.tif").read(1)
    m = rasterio.open(APPLY / f"{name}_v2_8m.tif").read(1)
    m1 = np.kron(m[r0 // 8:(r0 + n) // 8, c0 // 8:(c0 + n) // 8], np.ones((8, 8)))
    crops = [("v2 8 m (input M)", m1), ("bicubic 1 m", b[r0:r0 + n, c0:c0 + n]), ("G 1 m", g[r0:r0 + n, c0:c0 + n])]
    fig, ax = plt.subplots(2, 3, figsize=(7.5, 5.2))
    zr, zc, zn = zoom
    for i, (t, z) in enumerate(crops):
        ax[0, i].imshow(hillshade(z), cmap="gray", vmin=0, vmax=1)
        ax[0, i].add_patch(plt.Rectangle((zc, zr), zn, zn, fill=False, ec="#dc143c", lw=0.8))
        ax[0, i].set_title(f"{t}, {n / 1000:.1f} km", fontsize=8)
        ax[1, i].imshow(hillshade(z[zr:zr + zn, zc:zc + zn]), cmap="gray", vmin=0, vmax=1)
        ax[1, i].set_title(f"{t}, {zn} m", fontsize=8)
        for a in ax[:, i]:
            a.set_xticks([]); a.set_yticks([])
    fig.text(0.5, 0.005, note, ha="center", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, out)


def fig_seams():
    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    for name, lab, col in (("chitwan", "Chitwan, block-constant projection", "#dc143c"),
                           ("churia", "Churia, smooth projection", "#003893")):
        g = rasterio.open(APPLY / f"{name}_G_1m.tif").read(1).astype("f8")
        b = rasterio.open(APPLY / f"{name}_bicubic_1m.tif").read(1).astype("f8")
        for z, ls, l2 in ((g, "-", lab), (b, ":", None)):
            c = np.nanmean(np.abs(np.diff(z, 2, axis=1)), 0)
            k = np.arange(c.size) + 1
            p = np.array([np.nanmean(c[(k % 8) == q]) for q in range(8)])
            ax.plot(range(8), p / np.median(p), ls, color=col, marker="o" if ls == "-" else None, ms=3,
                    label=l2 if l2 else f"bicubic ({name})")
    ax.set_xlabel("column position within the 8 m cell (1 m pixels)")
    ax.set_ylabel("roughness / median")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig05_seam_diagnostic")


def fig_location():
    """Nepal on v2 relief, with the test boxes and the error-model site."""
    import json as _json
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling as R
    vrt = ROOT / "code" / "data" / "nw" / "nepal_dtm_8m_v2.vrt"
    W, H = 1400, 720
    with rasterio.open(vrt) as s, WarpedVRT(s, crs="EPSG:4326", resampling=R.average) as v:
        b = v.bounds
        z = v.read(1, out_shape=(H, W), resampling=R.average).astype("f4")
    z[~np.isfinite(z) | (z < -500)] = np.nan
    ext = (b.left, b.right, b.bottom, b.top)
    px = (b.right - b.left) / W * 111e3 * np.cos(np.radians(28))
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.imshow(np.where(np.isfinite(z), hillshade(np.nan_to_num(z, nan=np.nanmin(z)), px), np.nan), cmap="gray",
              extent=ext, vmin=0, vmax=1)
    from matplotlib.colors import LinearSegmentedColormap
    land = LinearSegmentedColormap.from_list("land", plt.cm.terrain(np.linspace(0.25, 1.0, 256)))  # no sea blues
    ax.imshow(z, cmap=land, alpha=0.5, extent=ext, vmin=0, vmax=7000)
    nep = _json.loads((Path(__file__).resolve().parent / "nepal.geojson").read_text())
    for f in nep["features"]:
        polys = [f["geometry"]["coordinates"]] if f["geometry"]["type"] == "Polygon" else f["geometry"]["coordinates"]
        for p in polys:
            r = np.array(p[0]); ax.plot(r[:, 0], r[:, 1], color="#003893", lw=0.8)
    lt = next((ROOT / "code" / "data" / "nw" / "zenodo" / "langtang").rglob("*_median_gated_v1.0.tif"))
    from rasterio.warp import transform_bounds
    with rasterio.open(lt) as s:
        lb = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
    boxes = [("Chitwan floodplain", (84.40, 27.55, 84.50, 27.65), "#dc143c", (8, 6)),
             ("Churia hills", (84.45, 27.46, 84.50, 27.51), "#ff8c00", (8, -12)),
             ("Langtang (error model)", lb, "#003893", (8, 6))]
    for name, (w, s_, e, n), col, off in boxes:
        ax.add_patch(plt.Rectangle((w, s_), e - w, n - s_, fill=False, ec=col, lw=1.6))
        ax.annotate(name, (e, n if off[1] > 0 else s_), xytext=off, textcoords="offset points", fontsize=7, color=col,
                    weight="bold")
    ax.set_xlim(80, 88.3); ax.set_ylim(26.3, 30.5)
    ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude (°N)")
    save(fig, "fig06_location_map")


def fig_training_examples():
    """One representative lidar patch per training site, shaded."""
    meta = np.load(ROOT / "code" / "data" / "gen" / "patches_meta.npz", allow_pickle=True)
    n = int(meta["n"])
    mm = np.memmap(ROOT / "code" / "data" / "gen" / "patches_all.int16", "int16", "r", shape=(n, 256, 256))
    sites = [("swiss", "Swiss Alps (swissALTI3D)"), ("san_juans_co", "San Juan Mts (3DEP)"),
             ("cascades_olympics_wa", "Cascades and Olympics (3DEP)"), ("smokies_nc_tn", "Great Smoky Mts (3DEP)")]
    fig, ax = plt.subplots(1, 4, figsize=(7.5, 2.3))
    for a, (key, label) in zip(ax, sites):
        idx = np.where(meta["src"] == key)[0]
        rel = meta["relief"][idx]
        i = idx[np.argsort(rel)[int(0.8 * len(idx))]]  # an 80th-percentile-relief patch: typical of steep ground
        z = mm[i].astype("f4") * meta["scale"][i] / 100
        a.imshow(hillshade(z), cmap="gray", vmin=0, vmax=1)
        a.set_title(f"{label}\nrelief {meta['relief'][i]:.0f} m, mean slope {meta['slope_mean'][i]:.0f}°", fontsize=7)
        a.set_xticks([]); a.set_yticks([])
    fig.text(0.5, 0.01, "Each patch is 256 × 256 m at 1 m.", ha="center", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig07_training_sites")


def fig_detail_maps():
    """Churia: where G adds relief beyond bicubic, and held-out photon errors of G."""
    import pandas as pd
    from pyproj import Transformer
    g = rasterio.open(APPLY / "churia_G_1m.tif")
    gz = g.read(1); bz = rasterio.open(APPLY / "churia_bicubic_1m.tif").read(1)
    d = gz - bz
    T, crs = g.transform, g.crs
    ext = (T.c, T.c + T.a * gz.shape[1], T.f + T.e * gz.shape[0], T.f)
    hs = hillshade(np.nan_to_num(gz, nan=np.nanmean(gz)))
    hs_ = []
    for f in sorted((ROOT / "code" / "data" / "nw" / "atl03").glob("*.parquet")):
        p = pd.read_parquet(f, columns=["lon", "lat", "h", "holdout"])
        p = p[p.holdout.astype(bool) & p.lon.between(84.45, 84.50) & p.lat.between(27.46, 27.51)]
        if len(p):
            hs_.append(p)
    p = pd.concat(hs_)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(p.lon.values, p.lat.values)
    inv = ~T
    c = np.floor(inv.a * x + inv.b * y + inv.c).astype(int); r = np.floor(inv.d * x + inv.e * y + inv.f).astype(int)
    ok = (r >= 0) & (r < gz.shape[0]) & (c >= 0) & (c < gz.shape[1])
    err = gz[r[ok], c[ok]] - p.h.values[ok]
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.9))
    v = np.nanpercentile(np.abs(d), 98)
    im = ax[0].imshow(d, cmap="RdBu_r", vmin=-v, vmax=v, extent=ext)
    ax[0].set_title("G minus bicubic (m): relief G adds", fontsize=8)
    fig.colorbar(im, ax=ax[0], shrink=0.75, label="m")
    ax[1].imshow(hs, cmap="gray", vmin=0, vmax=1, extent=ext)
    sc = ax[1].scatter(x[ok], y[ok], c=err, s=2, cmap="RdBu_r", vmin=-20, vmax=20)
    ax[1].set_title(f"G minus held-out photons (m), n = {ok.sum():,}", fontsize=8)
    fig.colorbar(sc, ax=ax[1], shrink=0.75, label="m")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
        a.plot([ext[0] + 300, ext[0] + 1300], [ext[2] + 300] * 2, color="k", lw=2)
        a.text(ext[0] + 800, ext[2] + 380, "1 km", ha="center", fontsize=7)
    fig.tight_layout()
    save(fig, "fig08_churia_detail_and_photons")


def fig_error_model():
    em = json.loads((ROOT / "code" / "data" / "gen" / "error_model.json").read_text())
    bins = list(em["slope_bins"])
    nm = [em["slope_bins"][k]["nmad"] for k in bins]
    vo = [100 * em["slope_bins"][k]["void_p"] for k in bins]
    deg = [0.39, 0.50, 0.66, 0.89, 1.31, 2.16]  # degraded-lidar NMAD from error_model.check()
    labels = [b.replace("50-90", "≥50") for b in bins]
    x = np.arange(len(bins)); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.bar(x - w / 2, nm, w, color="#003893", label="M at Langtang")
    ax.bar(x + w / 2, deg, w, color="#7fa3d6", label="degraded lidar")
    ax.set_xticks(x, [f"{l}°" for l in labels]); ax.set_xlabel("slope"); ax.set_ylabel("stereo error NMAD (m)")
    a2 = ax.twinx(); a2.plot(x, vo, "o-", color="#dc143c", ms=3, label="void share (%)")
    a2.set_ylabel("void share (%)"); a2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper left")
    save(fig, "fig09_error_model")


if __name__ == "__main__":
    fig_location()
    fig_training_examples()
    fig_detail_maps()
    fig_error_model()
    fig_loss()
    fig_eval()
    fig_compare("chitwan", 600, 4000, 1500, (500, 500, 400), "fig03_chitwan_floodplain",
                "Generated with the block-constant projection; the faint 8 m grid is the artefact of Section 5.4.")
    fig_compare("churia", 1500, 1500, 2000, (700, 700, 400), "fig04_churia_hills",
                "Generated with the smooth projection.")
    fig_seams()
    print("figures ->", OUT)
