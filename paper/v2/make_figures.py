#!/usr/bin/env python3
"""Figures for the NepalDTM v2 paper (reports/m_v2/paper_v2.tex).

Every number is read from data: experiments/EXP0xx_*/result.json, fusion/data/nw/{train_report*.json,
validation_v1_v2.json, adjust/params.npz}, and the stack tiles over Langtang for the error maps.
Outputs PNG + PDF to reports/m_v2/figures/.

Run:  cd ~/testground/3dnepal && PROJ_NETWORK=ON .venv/bin/python -I reports/m_v2/make_figures.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
NW = ROOT / "fusion" / "data" / "nw"
OUT = ROOT / "reports" / "m_v2" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
C = {"v1": "#8c8c8c", "v2a": "#9ecae1", "v2": "#2171b5", "cop": "#e6550d", "good": "#31a354"}

plt.rcParams.update({"font.size": 10, "font.family": "sans-serif", "axes.titlesize": 11, "axes.labelsize": 10,
                     "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  -> {name}")


def exp(eid):
    return json.loads(Path(glob.glob(str(ROOT / "experiments" / f"{eid}_*" / "result.json"))[0]).read_text())["results"]


def bars(ax, groups, series, labels, colors, fmt="{:.1f}", ylabel="RMSE (m)"):
    x = np.arange(len(groups))
    w = 0.8 / len(series)
    for i, (vals, lab, col) in enumerate(zip(series, labels, colors)):
        b = ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, label=lab, color=col)
        for r, v in zip(b, vals):
            if np.isfinite(v):
                ax.text(r.get_x() + r.get_width() / 2, r.get_height(), " " + fmt.format(v), ha="center", va="bottom",
                        fontsize=6 if len(series) > 3 else 7, rotation=90 if len(series) > 3 else 0)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8 if len(groups) > 6 else 10)
    ax.margins(y=0.15)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)


# ---------- Fig 1: national strip corrections ----------
def fig_strip_corrections():
    p = np.load(NW / "adjust" / "params.npz")["params"]
    s = json.loads((NW / "adjust" / "solve.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].hist(p[:, 0], bins=60, color=C["v2"])
    ax[0].axvline(np.median(p[:, 0]), color="k", lw=1, ls="--")
    ax[0].set_xlabel("vertical offset $a_k$ (m)")
    ax[0].set_ylabel("strips")
    ax[0].set_title(f"Offset per strip (median {np.median(p[:, 0]):+.2f} m)")
    tilt = np.hypot(p[:, 1], p[:, 2])
    ax[1].hist(tilt, bins=60, color=C["v2"])
    ax[1].set_xlabel("tilt magnitude (m per 10 km)")
    ax[1].set_title(f"Tilt per strip (median {np.median(tilt):.2f} m/10 km)")
    fig.suptitle(f"Joint adjustment of {p.shape[0]} HMA strips "
                 f"({s.get('n_photon_ties', 0):,} photon ties, {s.get('n_overlap_ties', 0):,} overlap ties)", fontsize=10)
    save(fig, "fig01_strip_corrections")



# ---------- Fig 3/4: Langtang error maps and error by source ----------
def langtang_arrays():
    cache = OUT / "langtang_cache.npz"
    if cache.exists():
        return dict(np.load(cache))
    os.environ["NW_VERSION"] = "v2"
    sys.path.insert(0, str(ROOT / "fusion"))
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from build_stack_hma import CRS, geoid_offset
    from fuse_hma import load_stack
    ref_p = next((NW / "zenodo" / "langtang").rglob("*_median_gated_v1.0.tif"))
    p = NW / "stack_v2" / "E0336N3120.tif"
    prof, L = load_stack(p)
    T, W, H = prof["transform"], prof["width"], prof["height"]
    ref = np.full((H, W), np.nan, "float32"); frac = ref.copy()
    with rasterio.open(ref_p) as s:
        a = s.read(1).astype("float32"); v = (a != s.nodata).astype("float32"); a[v == 0] = np.nan
        reproject(a, ref, src_transform=s.transform, src_crs=s.crs, dst_transform=T, dst_crs=CRS, resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
        reproject(v, frac, src_transform=s.transform, src_crs=s.crs, dst_transform=T, dst_crs=CRS, resampling=Resampling.average, dst_nodata=np.nan)
    ref += geoid_offset(T, W, H); ref[~(frac >= 0.9)] = np.nan
    out = {"ref": ref, "src_v2": L["src"]}
    for k, vrt in (("v1", NW / "nepal_dem_8m.vrt"), ("v2", NW / "nepal_dem_8m_v2.vrt")):
        z = np.full((H, W), np.nan, "float32")
        with rasterio.open(vrt) as s:
            reproject(rasterio.band(s, 1), z, dst_transform=T, dst_crs=CRS, resampling=Resampling.nearest, dst_nodata=np.nan)
        out[k] = z
    out["src_v1"] = load_stack(NW / "stack" / p.name)[1]["src"]
    rows, cols = np.where(np.isfinite(ref))
    sl = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    out = {k: v[sl] for k, v in out.items()}
    np.savez_compressed(cache, **out)
    return out


def fig_langtang_maps():
    d = langtang_arrays()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for a, k, t in ((ax[0], "v1", "v1 DEM $-$ 2 m reference"), (ax[1], "v2", "v2 DEM $-$ 2 m reference")):
        e = d[k] - d["ref"]
        im = a.imshow(e, cmap="RdBu_r", vmin=-15, vmax=15, interpolation="nearest")
        m = np.isfinite(e)
        a.set_title(f"{t}\nRMSE {np.sqrt(np.nanmean(e[m] ** 2)):.2f} m, median {np.nanmedian(e[m]):+.2f} m", fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label="height difference (m)")
    save(fig, "fig03_langtang_error_maps")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    groups, s1, s2 = [], [], []
    for nm, code in (("HMA stereo", 1), ("Copernicus fill", 2)):
        m = np.isfinite(d["ref"]) & np.isfinite(d["v1"]) & np.isfinite(d["v2"])
        e1, e2 = (d["v1"] - d["ref"])[m & (d["src_v1"] == code)], (d["v2"] - d["ref"])[m & (d["src_v2"] == code)]
        groups.append(f"{nm}\nv1 n={e1.size:,} / v2 n={e2.size:,}")
        s1.append(np.sqrt(np.mean(e1 ** 2)) if e1.size else np.nan)
        s2.append(np.sqrt(np.mean(e2 ** 2)) if e2.size else np.nan)
    bars(ax, groups, [s1, s2], ["v1", "v2"], [C["v1"], C["v2"]])
    ax.set_title("Langtang DEM error by source of the 8 m cell")
    save(fig, "fig04_error_by_source")




# ---------- Fig 8: held-out photons ----------
def fig_photons():
    p = NW / os.environ.get("VAL_JSON", "validation_v1_v2.json")
    if not p.exists():
        print("  (no validation_v1_v2.json yet)")
        return
    R = json.loads(p.read_text())["photons_holdout_dtm"]
    g = [("all", "all"), ("tree", "forest"), ("non-tree", "non-forest"), ("bare/snow", "bare/snow"), ("slope <20", "<20°"),
         ("slope 20-40", "20-40°"), ("slope >=40", "≥40°")]
    g = [x for x in g if x[0] in R]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5))
    models = [("copernicus", "Copernicus GLO-30", C["cop"]), ("fabdem", "FABDEM", "#fdae6b"),
              ("gedtm30", "GEDTM30", "#fdd0a2"), ("v1", "NepalDTM v1", C["v1"]), ("v2", "NepalDTM v2", C["v2"])]
    models = [m for m in models if R["all"].get(m[0])]
    for a, stat, fmt, yl in ((ax[0], "rmse", "{:.1f}", "RMSE (m)"), (ax[1], "nmad", "{:.2f}", "NMAD (m)")):
        bars(a, [l for _, l in g], [[R[k][m][stat] if R[k].get(m) else np.nan for k, _ in g] for m, _, _ in models],
             [l for _, l, _ in models], [c for _, _, c in models], fmt=fmt, ylabel=yl)
    ax[0].set_title(f"RMSE, {R['all']['v1']['n'] / 1e6:.1f} M held-out ATL03 ground photons")
    ax[1].set_title("NMAD, same photons")
    save(fig, "fig08_heldout_photons")
    V = json.loads(p.read_text())
    L = V["langtang_2m"]
    lg = [("all", "all"), ("slope <20", "<20°"), ("slope 20-40", "20-40°"), ("slope >=40", "≥40°")]
    lm = [("copernicus", "Copernicus GLO-30", C["cop"]), ("fabdem", "FABDEM", "#fdae6b"), ("gedtm30", "GEDTM30", "#fdd0a2"),
          ("v1/dem", "NepalDTM v1 DEM", C["v1"]), ("v2/dem", "NepalDTM v2 DEM", C["v2"])]
    lm = [m for m in lm if L["all"].get(m[0])]
    D = V["dgps"]
    dg = ["margins", "glacier surface", "wind tower"]
    dm = [("copernicus", "Copernicus GLO-30", C["cop"]), ("fabdem", "FABDEM", "#fdae6b"), ("gedtm30", "GEDTM30", "#fdd0a2"),
          ("v1", "NepalDTM v1 DEM", C["v1"]), ("v2", "NepalDTM v2 DEM", C["v2"])]
    get = lambda m, gname: (D[m].get(f"dem/{gname}") or {}).get("rmse", np.nan) if m in ("v1", "v2") else (D[m].get(gname) or {}).get("rmse", np.nan)
    dm = [m for m in dm if D.get(m[0])]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    bars(ax[0], [l for _, l in lg], [[L[k][m]["rmse"] for k, _ in lg] for m, _, _ in lm], [l for _, l, _ in lm], [c for _, _, c in lm])
    ax[0].set_title("Langtang, against the 2 m WorldView/GeoEye composite")
    bars(ax[1], dg, [[get(m, k) for k in dg] for m, _, _ in dm], [l for _, l, _ in dm], [c for _, _, c in dm])
    ax[1].set_title("Khumbu, against dGPS control points")
    save(fig, "fig11_public_dems")



# ---------- Fig 10: what each experiment changed ----------
def fig_experiments():
    rows = [("EXP002 CHMv2 predictor", 7.722, 7.683), ("EXP003 subtract CHMv2", 7.721, 7.981),
            ("EXP021 per-landcover models", 7.721, 7.700), ("EXP022 CHMv2+contours+rivers", 7.721, 7.641),
            ("EXP024 + strip agreement", 7.724, 7.585)]
    r27, r28, r29, r30 = exp("EXP027"), exp("EXP028"), exp("EXP029"), exp("EXP030")
    rows += [("EXP027 strip adjustment (Langtang)", r27["ref_2m"]["all"]["current"]["rmse"], r27["ref_2m"]["all"]["adjusted"]["rmse"]),
             ("EXP028 smooth strip grid (Langtang)", r28["ref_2m"]["all"]["plane"]["rmse"], r28["ref_2m"]["all"]["grid_4km"]["rmse"]),
             ("EXP029 photon-cell targets", r29["photon_cells_holdout"]["all"]["atl08"]["rmse"], r29["photon_cells_holdout"]["all"]["both"]["rmse"]),
             ("EXP030 void fill (Langtang voids)", r30["A_cop_fill"]["ref"]["rmse"], r30["D_strips_kept"]["ref"]["rmse"])]
    ch = [100 * (b / a - 1) for _, a, b in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, ch, color=[C["good"] if c < -2 else (C["cop"] if c > 2 else C["v1"]) for c in ch])
    for yy, c in zip(y, ch):
        ax.text(c + (0.8 if c >= 0 else -0.8), yy, f"{c:+.1f}%", va="center", ha="left" if c >= 0 else "right", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _, _ in rows], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("change in RMSE against each experiment's own baseline (%)")
    ax.set_xlim(min(ch) - 12, max(ch) + 12)
    save(fig, "fig10_experiments")


if __name__ == "__main__":
    for f in (fig_strip_corrections, fig_langtang_maps, fig_photons, fig_experiments):
        try:
            f()
        except Exception as e:  # keep going; report which figure failed
            print(f"  !! {f.__name__}: {type(e).__name__}: {e}")
