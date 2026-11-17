"""v1 vs v2 national products, scored on independent references (fills the paper's pending numbers).

1. Held-out ICESat-2 ground photons (tracks with holdout=True; never used by the strip adjustment), DTM error by
   land cover, slope and elevation zone. A random 25% of held-out photons per 0.5 deg cell keeps memory bounded.
2. Langtang gated 2 m composite (Shean et al.), averaged to 8 m, cells >=90% covered; DEM and DTM.
3. Khumbu dGPS control points (nw_validate.dgps), DEM and DTM.
Chamoli is in India, outside the national VRTs; it is covered by the transfer experiment.

Out: data/nw/validation_v1_v2.json.  Run: .venv/bin/python -I -u fusion/validate_v1_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stack_hma import CRS, geoid_offset  # noqa: E402
from nw_validate import dgps, st  # noqa: E402
from strip_adjust import NW, grid  # noqa: E402


def _worldcover(T, W, H):
    """ESA WorldCover 2021 classes on the grid, from the Planetary Computer STAC (mode resampling)."""
    import planetary_computer
    import pystac_client
    from build_stack_hma import warp
    from pyproj import Transformer
    ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform_bounds(*(T * (0, H)), *(T * (W, 0)))
    stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                     modifier=planetary_computer.sign_inplace)
    items = list(stac.search(collections=["esa-worldcover"], bbox=ll, datetime="2021").items())
    return warp([it.assets["map"].href for it in items], T, W, H, Resampling.mode, 0)


def worldcover(T, W, H, tries=6):
    """Planetary Computer STAC drops requests now and then (crashed a run at 80/82 cells): retry with backoff."""
    import time
    for k in range(tries):
        try:
            return _worldcover(T, W, H)
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"  worldcover retry {k + 1}: {type(e).__name__}", flush=True)
            time.sleep(15 * (k + 1))

VRT = {v: {k: NW / f"nepal_{k}_8m{s}.vrt" for k in ("dem", "dtm")} for v, s in (("v1", ""), ("v2", "_v2"))}
EXT = ("copernicus", "fabdem", "gedtm30")  # public DEMs scored on the same references (all EGM2008)
BARE = (60, 70, 100)  # WorldCover bare/sparse, snow/ice, moss/lichen (EXP031 rule: DTM = DEM there)
ZONES = ((-1e9, 300, "terai <300 m"), (300, 1000, "siwalik 300-1000 m"), (1000, 2500, "hills 1-2.5 km"),
         (2500, 4000, "mountains 2.5-4 km"), (4000, 1e9, "high himal >4 km"))


def window(path, b):
    with rasterio.open(path) as s:
        w = from_bounds(*b, s.transform).round_offsets().round_lengths()
        a = s.read(1, window=w, boundless=True, fill_value=np.nan).astype("float32")
        return a, s.window_transform(w)


def externals(T, W, H) -> dict:
    """Copernicus GLO-30, FABDEM V1-2 and GEDTM30 v1.2 bilinearly resampled to the 8 m grid."""
    import math
    from pyproj import Transformer
    from build_stack_hma import COP, warp
    from fetch_external_dems import GEDTM30
    from nw_validate import FABDEM_COG
    w, s, e, n = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform_bounds(*(T * (0, H)), *(T * (W, 0)))
    cells = [(la, lo) for la in range(math.floor(s), math.floor(n) + 1) for lo in range(math.floor(w), math.floor(e) + 1)]
    out = {"copernicus": warp([COP.format(lat=la, lon=lo) for la, lo in cells], T, W, H, Resampling.bilinear)}
    fab = []
    for la, lo in cells:
        z = f"N{la // 10 * 10:02d}E{lo // 10 * 10:03d}-N{la // 10 * 10 + 10:02d}E{lo // 10 * 10 + 10:03d}"
        fab.append("/vsicurl/" + FABDEM_COG.format(z=z, la=la, lo=lo))
    try:
        out["fabdem"] = warp(fab, T, W, H, Resampling.bilinear)
    except Exception as ex:
        print(f"  FABDEM unavailable: {ex}", flush=True)
        out["fabdem"] = np.full((H, W), np.nan, "float32")
    g = np.full((H, W), np.nan, "float32")
    with rasterio.open("/vsicurl/" + GEDTM30 if not GEDTM30.startswith("/vsi") else GEDTM30) as src:
        reproject(rasterio.band(src, 1), g, dst_transform=T, dst_crs=CRS, resampling=Resampling.bilinear,
                  src_nodata=src.nodata, dst_nodata=np.nan)
    out["gedtm30"] = g
    return out


def sample(a, T, x, y):
    inv = ~T
    c = np.floor(inv.a * x + inv.b * y + inv.c).astype(int)
    r = np.floor(inv.d * x + inv.e * y + inv.f).astype(int)
    ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
    z = np.full(x.size, np.nan, "float32")
    z[ok] = a[r[ok], c[ok]]
    return z


def photons(R, limit=None):
    from pyproj import Transformer
    to = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    rng = np.random.default_rng(0)
    E = {v: [] for v in list(VRT) + ["v2_bare_rule"] + list(EXT)}
    G = {"wc": [], "slope": [], "z": []}
    files = sorted((NW / "atl03").glob("*.parquet"))
    if limit:
        files = files[len(files) // 2:len(files) // 2 + limit]
    for i, f in enumerate(files, 1):
        d = pd.read_parquet(f, columns=["lon", "lat", "h", "holdout"])
        d = d[d.holdout.astype(bool)]
        d = d.iloc[rng.random(len(d)) < 0.25]
        if len(d) == 0:
            continue
        x, y = to.transform(d.lon.values, d.lat.values)
        b = (x.min() - 16, y.min() - 16, x.max() + 16, y.max() + 16)
        cols = {}
        for v in VRT:
            a, T = window(VRT[v]["dtm"], b)
            inv = ~T
            c = np.floor(inv.a * x + inv.b * y + inv.c).astype(int)
            r = np.floor(inv.d * x + inv.e * y + inv.f).astype(int)
            ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
            z = np.full(x.size, np.nan, "float32")
            z[ok] = a[r[ok], c[ok]]
            cols[v] = z
            if v == "v1":
                gy, gx = np.gradient(np.where(np.isfinite(a), a, np.nanmedian(a)), 8.0)
                sl = np.degrees(np.arctan(np.hypot(gx, gy)))
                s_ = np.full(x.size, np.nan, "float32"); s_[ok] = sl[r[ok], c[ok]]
                G["slope"].append(s_); G["z"].append(z.copy())
        T0, W0, H0 = grid(b)
        wa = worldcover(T0, W0, H0)
        inv = ~T0
        c = np.clip(np.floor(inv.a * x + inv.b * y + inv.c).astype(int), 0, W0 - 1)
        r = np.clip(np.floor(inv.d * x + inv.e * y + inv.f).astype(int), 0, H0 - 1)
        wcp = wa[r, c].astype("float32")
        G["wc"].append(wcp)
        for v in VRT:
            E[v].append(cols[v] - d.h.values)
        if "v2" in VRT:
            a, T = window(VRT["v2"]["dem"], b)
            dem2 = sample(a, T, x, y)
            E["v2_bare_rule"].append(np.where(np.isin(wcp, BARE), dem2, cols["v2"]) - d.h.values)
        else:
            E["v2_bare_rule"].append(np.full(x.size, np.nan, "float32"))
        X = externals(T0, W0, H0)
        for k in EXT:
            E[k].append(sample(X[k], T0, x, y) - d.h.values)
        if i % 10 == 0:
            print(f"  photons {i}/{len(files)}", flush=True)
    E = {v: np.concatenate(e) for v, e in E.items()}
    G = {k: np.concatenate(g) for k, g in G.items()}
    common = np.all([np.isfinite(E[v]) for v in VRT], 0)  # every model scored on photons all our versions cover
    for k in EXT:
        print(f"  {k} coverage of common photons: {np.isfinite(E[k][common]).mean():.1%}", flush=True)
    common &= np.all([np.isfinite(E[k]) for k in EXT if np.isfinite(E[k]).any()], 0)
    groups = {"all": common, "tree": common & (G["wc"] == 10), "non-tree": common & (G["wc"] != 10),
              "bare/snow": common & np.isin(G["wc"], BARE),
              "slope <20": common & (G["slope"] < 20), "slope 20-40": common & (G["slope"] >= 20) & (G["slope"] < 40),
              "slope >=40": common & (G["slope"] >= 40)}
    for lo, hi, nm in ZONES:
        groups[nm] = common & (G["z"] >= lo) & (G["z"] < hi)
    R["photons_holdout_dtm"] = {g: {v: st(E[v][m]) for v in E} for g, m in groups.items()}
    for g, r in R["photons_holdout_dtm"].items():
        print(f"  {g:20s} " + " | ".join(f"{v} rmse {s['rmse']:.2f} nmad {s['nmad']:.2f} n {s['n']:,}"
                                          for v, s in r.items() if s), flush=True)


def langtang(R):
    ref_p = next((NW / "zenodo" / "langtang").rglob("*_median_gated_v1.0.tif"))
    with rasterio.open(ref_p) as s:
        b = s.bounds
        T, W, H = grid((b.left, b.bottom, b.right, b.top))
        ref = np.full((H, W), np.nan, "float32")
        frac = np.full((H, W), np.nan, "float32")
        a = s.read(1).astype("float32")
        v = (a != s.nodata).astype("float32")
        a[v == 0] = np.nan
        reproject(a, ref, src_transform=s.transform, src_crs=s.crs, dst_transform=T, dst_crs=CRS,
                  resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
        reproject(v, frac, src_transform=s.transform, src_crs=s.crs, dst_transform=T, dst_crs=CRS,
                  resampling=Resampling.average, dst_nodata=np.nan)
    ref = ref + geoid_offset(T, W, H)  # ellipsoidal -> EGM2008, same convention as EXP027
    ref[~(frac >= 0.9)] = np.nan
    gy, gx = np.gradient(np.where(np.isfinite(ref), ref, np.nanmedian(ref)), 8.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    S = {}
    for v in VRT:
        for k in ("dem", "dtm"):
            z = np.full((H, W), np.nan, "float32")
            with rasterio.open(VRT[v][k]) as s:
                reproject(rasterio.band(s, 1), z, dst_transform=T, dst_crs=CRS, resampling=Resampling.bilinear,
                          dst_nodata=np.nan)
            S[f"{v}/{k}"] = z
    X = externals(T, W, H)
    for k in EXT:
        if np.isfinite(X[k]).any():
            S[k] = X[k]
    common = np.isfinite(ref) & np.all([np.isfinite(z) for z in S.values()], 0)
    R["langtang_2m"] = {}
    for g, m in (("all", common), ("slope <20", common & (slope < 20)),
                 ("slope 20-40", common & (slope >= 20) & (slope < 40)), ("slope >=40", common & (slope >= 40))):
        R["langtang_2m"][g] = {k: st((z - ref)[m]) for k, z in S.items()}
        print(f"  langtang {g:12s} " + " | ".join(f"{k} {s['rmse']:.2f}" for k, s in R["langtang_2m"][g].items() if s))


def dgps_external() -> dict:
    p = NW.parent / "control" / "khumbu_All_GCPs.csv"
    d = pd.read_csv(p)
    d["Location"] = d["Location"].ffill()
    d = d[d.Easting.between(4e5, 6e5)]
    T, W, H = grid((d.Easting.min() - 200, d.Northing.min() - 200, d.Easting.max() + 200, d.Northing.max() + 200))
    X = externals(T, W, H)
    out = {}
    for k in EXT:
        z = sample(X[k], T, d.Easting.values, d.Northing.values)
        out[k] = {gn: st((z - d["Ortho. Hgt."].values)[d.Location.str.contains(key).values])
                  for gn, key in (("margins", "margins"), ("glacier surface", "surface"), ("wind tower", "Wind"))}
        print(f"  {k}: " + ", ".join(f"{g} {s['rmse']:.2f}" for g, s in out[k].items() if s), flush=True)
    return out


def main():
    global VRT
    test = "--test" in sys.argv  # v1 only, 2 photon cells: checks the code before v2 exists
    if test:
        VRT = {"v1": VRT["v1"]}
    R = {}
    print("[held-out photons]", flush=True)
    photons(R, 2 if test else None)
    print("[langtang]", flush=True)
    langtang(R)
    print("[dgps]", flush=True)
    R["dgps"] = {}
    for v in VRT:
        r = {}
        dgps(r, VRT[v])
        R["dgps"][v] = r.get("dgps")
        print(f"  {v}: {r.get('dgps')}", flush=True)
    R["dgps"].update(dgps_external())
    out = NW / ("validation_test.json" if test else "validation_v1_v2.json")
    out.write_text(json.dumps(R, indent=2, default=float) + "\n")
    print("->", out)


if __name__ == "__main__":
    main()
