#!/usr/bin/env python3
"""GEM-15 as void filler (national): on cells where v2 falls back to Copernicus, is GEM-15 better?

Stages:
  fetch   download GEM-15 dsm+dtm COGs for the 1-degree cells over Nepal into data/nw/gem15/ (public, no auth)
  score   held-out ATL03 photons (25% sample, as validate_v1_v2) on v2 Copernicus-fill cells (aux band 1 == 2):
          Copernicus vs GEM-15 dsm vs GEM-15 dsm debiased vs GEM-15 dtm vs current v2 DTM, by land cover and slope.
          Debias = per 1-degree tile median(GEM dsm - v2 DEM) over v2 HMA cells (no validation photons used).
          Gate: best GEM-15 arm >= 10% lower RMSE than Copernicus on fill cells -> adopt as filler.
GEM-15 heights (EGM96) are converted to EGM2008 like the photons and the other DEMs.

Run: .venv/bin/python -I -u fusion/gem15_fill.py {fetch|score}
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "YES")
os.environ["NW_VERSION"] = "v2"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

from build_stack_hma import CRS  # noqa: E402
from validate_v1_v2 import NW, window, sample, worldcover, externals  # noqa: E402
from strip_adjust import grid  # noqa: E402

BASE = "https://storage.googleapis.com/climate-and-sustainability/gem-15"
OUT = NW / "gem15"
CELLS = [(lo, la) for lo in range(80, 89) for la in range(26, 31)]  # Nepal: 80.0-88.2 E, 26.3-30.5 N
BARE = (60, 70)


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    have = []
    for lo, la in CELLS:
        g = f"p{la // 10 * 10}_p{lo // 10 * 10}"
        try:
            item = json.load(urllib.request.urlopen(f"{BASE}/stac/gem-15/{g}/w{lo}s{la}/w{lo}s{la}.json", timeout=60))
        except Exception:
            continue  # cell not mapped (e.g. Tibet side)
        have.append((lo, la))
        for band in ("dsm", "dtm"):
            f = OUT / f"w{lo}s{la}_{band}.tif"
            if f.exists() and f.stat().st_size == item["assets"][band]["file:size"]:
                continue
            print(f"  {f.name} {item['assets'][band]['file:size'] / 1e9:.2f} GB", flush=True)
            subprocess.run(["curl", "-sSfL", "--retry", "5", "-C", "-", "--connect-timeout", "30", "-k",
                            "-o", str(f), item["assets"][band]["href"]], check=True)
    print(f"{len(have)} GEM-15 cells over Nepal: {have}")


def egm96_to_08(lon, lat):
    n96 = Transformer.from_crs("EPSG:4326", "EPSG:4326+5773", always_xy=True).transform(lon, lat, np.zeros_like(lon))[2]
    n08 = Transformer.from_crs("EPSG:4326", "EPSG:4326+3855", always_xy=True).transform(lon, lat, np.zeros_like(lon))[2]
    return n08 - n96  # add to EGM96 heights (transformer returns -N)


def gem_sample(path, lon, lat):
    with rasterio.open(path) as s:
        a = s.read(1)
        x, y = Transformer.from_crs("EPSG:4326", s.crs, always_xy=True).transform(lon, lat)
        inv = ~s.transform
        c = inv.a * x + inv.b * y + inv.c - 0.5
        r = inv.d * x + inv.e * y + inv.f - 0.5
    return map_coordinates(a, [r, c], order=1, mode="constant", cval=np.nan, prefilter=False).astype("float32")


def score():
    to = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    rng = np.random.default_rng(0)
    cells = {(lo, la) for lo, la in CELLS if (OUT / f"w{lo}s{la}_dsm.tif").exists()}
    cols = {k: [] for k in ("h", "cop", "gdsm", "gdtm", "v2dtm", "v2dem", "src", "wc", "slope", "cell")}
    for f in sorted((NW / "atl03").glob("*.parquet")):
        d = pd.read_parquet(f, columns=["lon", "lat", "h", "holdout"])
        d = d[d.holdout.astype(bool)]
        d = d.iloc[rng.random(len(d)) < 0.25]
        if len(d) == 0:
            continue
        x, y = to.transform(d.lon.values, d.lat.values)
        b = (x.min() - 16, y.min() - 16, x.max() + 16, y.max() + 16)
        a, T = window(NW / "nepal_aux_8m_v2.vrt", b)
        with rasterio.open(NW / "nepal_aux_8m_v2.vrt") as s:
            from rasterio.windows import from_bounds
            w = from_bounds(*b, s.transform).round_offsets().round_lengths()
            src = s.read(1, window=w, boundless=True, fill_value=0).astype("float32")
        srcp = sample(src, T, x, y)
        keep = srcp > 0  # inside the v2 product
        if not keep.any():
            continue
        dtm, Td = window(NW / "nepal_dtm_8m_v2.vrt", b)
        dem, Te = window(NW / "nepal_dem_8m_v2.vrt", b)
        gy, gx = np.gradient(np.where(np.isfinite(dtm), dtm, np.nanmedian(dtm)), 8.0)
        sl = np.degrees(np.arctan(np.hypot(gx, gy)))
        T0, W0, H0 = grid(b)
        cop = sample(externals_cop(T0, W0, H0), T0, x, y)
        wc = sample(worldcover(T0, W0, H0).astype("float32"), T0, x, y)
        lo_, la_ = np.floor(d.lon.values).astype(int), np.floor(d.lat.values).astype(int)
        gd = np.full(x.size, np.nan, "float32"); gt = gd.copy()
        for cl in set(zip(lo_, la_)):
            if cl not in cells:
                continue
            m = (lo_ == cl[0]) & (la_ == cl[1])
            off = egm96_to_08(d.lon.values[m], d.lat.values[m])
            gd[m] = gem_sample(OUT / f"w{cl[0]}s{cl[1]}_dsm.tif", d.lon.values[m], d.lat.values[m]) + off
            gt[m] = gem_sample(OUT / f"w{cl[0]}s{cl[1]}_dtm.tif", d.lon.values[m], d.lat.values[m]) + off
        for k, v in (("h", d.h.values), ("cop", cop), ("gdsm", gd), ("gdtm", gt), ("v2dtm", sample(dtm, Td, x, y)),
                     ("v2dem", sample(dem, Te, x, y)), ("src", srcp), ("wc", wc), ("slope", sample(sl, Td, x, y)),
                     ("cell", lo_ * 100 + la_)):
            cols[k].append(np.asarray(v)[keep])
    C = {k: np.concatenate(v) for k, v in cols.items()}
    # per-tile GEM-15 bias from v2 HMA cells (src==1), independent of the photon heights
    bias = {}
    for cl in np.unique(C["cell"]):
        m = (C["cell"] == cl) & (C["src"] == 1) & np.isfinite(C["gdsm"]) & np.isfinite(C["v2dem"])
        bias[int(cl)] = float(np.median((C["gdsm"] - C["v2dem"])[m])) if m.sum() > 500 else 0.0
    C["gdsm_db"] = C["gdsm"] - np.vectorize(bias.get)(C["cell"])

    def st(e):
        e = e[np.isfinite(e)]
        if e.size < 50:
            return None
        md = np.median(e)
        return {"n": int(e.size), "rmse": float(np.sqrt(np.mean(e ** 2))), "nmad": float(1.4826 * np.median(np.abs(e - md))),
                "median": float(md), "p95abs": float(np.percentile(np.abs(e), 95))}
    arms = {"Copernicus (current fill)": "cop", "GEM-15 dsm": "gdsm", "GEM-15 dsm debiased": "gdsm_db",
            "GEM-15 dtm": "gdtm", "v2 DTM (current product)": "v2dtm"}
    fill = (C["src"] == 2) & np.all([np.isfinite(C[v]) for v in arms.values()], 0)
    groups = {"fill: all": fill, "fill: tree": fill & (C["wc"] == 10), "fill: non-tree": fill & (C["wc"] != 10),
              "fill: bare/snow": fill & np.isin(C["wc"], BARE), "fill: slope <20": fill & (C["slope"] < 20),
              "fill: slope 20-40": fill & (C["slope"] >= 20) & (C["slope"] < 40), "fill: slope >=40": fill & (C["slope"] >= 40)}
    hma = (C["src"] == 1) & np.all([np.isfinite(C[v]) for v in arms.values()], 0)
    groups["HMA cells (reference)"] = hma
    R = {"gem_coverage_of_fill_photons": float(np.isfinite(C["gdsm"][C["src"] == 2]).mean()), "tile_bias": bias, "groups": {}}
    print(f"GEM-15 covers {R['gem_coverage_of_fill_photons']:.1%} of fill-cell photons")
    for g, m in groups.items():
        R["groups"][g] = {a: st((C[v] - C["h"])[m]) for a, v in arms.items()}
        print(f"\n{g} (n={m.sum():,})")
        for a, r in R["groups"][g].items():
            if r:
                print(f"  {a:26s} RMSE {r['rmse']:6.2f}  NMAD {r['nmad']:5.2f}  median {r['median']:+6.2f}  p95 {r['p95abs']:6.2f}")
    f = R["groups"]["fill: all"]
    best = min(("GEM-15 dsm", "GEM-15 dsm debiased", "GEM-15 dtm"), key=lambda a: f[a]["rmse"])
    gain = 1 - f[best]["rmse"] / f["Copernicus (current fill)"]["rmse"]
    R["best_gem_arm"], R["gain_vs_copernicus"], R["gate_passed"] = best, gain, bool(gain >= 0.10)
    print(f"\nbest GEM-15 arm on fill cells: {best} ({gain:+.1%} RMSE vs Copernicus) -> gate {'PASSED' if gain >= 0.10 else 'FAILED'}")
    (NW / "gem15_fill.json").write_text(json.dumps(R, indent=1, default=float))


def externals_cop(T, W, H):
    import math
    from rasterio.enums import Resampling
    from build_stack_hma import COP, warp
    w, s, e, n = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform_bounds(*(T * (0, H)), *(T * (W, 0)))
    return warp([COP.format(lat=la, lon=lo) for la in range(math.floor(s), math.floor(n) + 1)
                 for lo in range(math.floor(w), math.floor(e) + 1)], T, W, H, Resampling.bilinear)


if __name__ == "__main__":
    {"fetch": fetch, "score": score}[sys.argv[1]]()
