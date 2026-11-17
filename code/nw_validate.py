"""Validate the national 8 m DEM/DTM (run after nw_train.py and nw_apply.py).

Every number in the outputs is computed here from data; nothing is typed in by hand.

1. Held-out ICESat-2 errors (out-of-fold from nw_train.py): by zone, province, land cover, slope, DSM source.
   Headline = spatial blocks (48 km regions unseen in training); pass-held-out for comparison with AOI runs.
2. Same points, other models: Copernicus GLO-30, GEDTM30 v1.2 (trained on ICESat-2, so optimistic), FABDEM V1-2.
3. Khumbu dGPS control points (independent, sub-cm, 2015-16).
4. Survey Department contours (data/control/contours_wfs_*.geojson; independent, +-10-20 m in mountains).
5. Agreement with the validated AOI DTMs (Bhote Koshi 8 m, Khumbu 8 m, Bheemdatt 20 m).
6. Seams: height step across tile edges vs the step between neighbouring pixels inside tiles.
7. Horizontal shift of the DEM vs ICESat-2 open ground, per province (should be ~0).
8. Coverage inside Nepal: share of HMA 8 m vs Copernicus fill, expected-error distribution, gaps.

Outputs: data/nw/validation_report.json and data/nw/VALIDATION.md (tables generated from the JSON).

Run:
    ~/workspace/super-resolution-satellite/.venv/bin/python -I nw_validate.py
"""

from __future__ import annotations

import json
import math
import sys
import zipfile
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.transform import rowcol
from rasterio.warp import reproject
from rasterio.windows import from_bounds
from scipy import ndimage
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_external_dems import FABDEM_ZIP, GEDTM30  # noqa: E402
FABDEM_COG = "https://huggingface.co/buckets/links-ads/fabdem/resolve/tiles/{z}_FABDEM_V1-2/N{la:02d}E{lo:03d}_FABDEM_V1-2.tif"
from fuse_hma import SUFFIX, WC, stats  # noqa: E402  (NW_VERSION=v2 -> _v2 tiles, VRTs, points, reports)
from validate_topo import densify  # noqa: E402
from build_stack_hma import COP, warp  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
NW = DATA / "nw"
CTRL = DATA / "control"
CRS = "EPSG:32645"
CORE = 3000
ZONES = ((-1e9, 300, "terai <300 m"), (300, 1000, "siwalik 300-1000 m"), (1000, 2500, "hills 1-2.5 km"),
         (2500, 4000, "mountains 2.5-4 km"), (4000, 1e9, "high himal >4 km"))


def nmad(e):
    e = e[np.isfinite(e)]
    return float(1.4826 * np.median(np.abs(e - np.median(e)))) if e.size else float("nan")


def st(e):
    e = e[np.isfinite(e)]
    return stats(e) if e.size >= 30 else None


def sample_grid(a, T, x, y):
    r, c = rowcol(T, x, y, op=float)
    return ndimage.map_coordinates(a, [np.asarray(r) - .5, np.asarray(c) - .5], order=1, mode="constant", cval=np.nan)


def provinces(lon, lat):
    fc = json.load(open(CTRL / "nepal_provinces.geojson"))["features"]
    out = np.full(lon.size, "outside Nepal", dtype=object)
    for f in fc:
        g = shape(f["geometry"])
        out[shapely.contains_xy(g, lon, lat)] = f["properties"].get("Name_E", str(f["properties"]))
    return out


def external(lon, lat):
    """GEDTM30 and FABDEM at the points, read one 1-degree cell at a time."""
    g30, fab = np.full(lon.size, np.nan, "float32"), np.full(lon.size, np.nan, "float32")
    cell = (np.floor(lon).astype(int) * 1000 + np.floor(lat).astype(int))
    with rasterio.open(GEDTM30) as gs:
        for c in np.unique(cell):
            lo, la = c // 1000, c % 1000
            m = cell == c
            win = from_bounds(lo - .01, la - .01, lo + 1.01, la + 1.01, gs.transform).round_offsets().round_lengths()
            a = gs.read(1, window=win, masked=True).astype("float32").filled(np.nan)  # metres (see fetch_external_dems)
            g30[m] = sample_grid(a, gs.window_transform(win), lon[m], lat[m])
            z = f"N{la // 10 * 10:02d}E{lo // 10 * 10:03d}-N{la // 10 * 10 + 10:02d}E{lo // 10 * 10 + 10:03d}"
            try:
                url = FABDEM_COG.format(z=z, la=la, lo=lo)
                with rasterio.open(url) as fs:
                    fab[m] = sample_grid(fs.read(1).astype("float32"), fs.transform, lon[m], lat[m])
            except Exception:
                pass  # no FABDEM tile (all water/outside)
            print(f"  external DEMs: cell E{lo} N{la}, {m.sum():,} points", flush=True)
    return g30, fab


def heldout(d, R):
    s0 = d["src"] == 0
    lon, lat, h = d["lon"][s0], d["lat"][s0], d["h"][s0]
    g30, fab = external(lon, lat)
    errs = {"ours_spatial": d["err_spatial"][s0], "ours_pass": d["err_pass"][s0], "copernicus": d["err_cop"][s0],
            "gedtm30": g30 - h, "fabdem": fab - h}
    common = np.all([np.isfinite(e) for e in errs.values()], 0)
    prov = provinces(lon, lat)
    wc, slope, dsm, srcf = d["wc"][s0], d["slope"][s0], d["dsm"][s0], d["src_f"][s0]
    groups = {"all": np.ones(len(h), bool)}
    for lo, hi, nm in ZONES:
        groups[f"zone: {nm}"] = (dsm >= lo) & (dsm < hi)
    for p in sorted(set(prov) - {"outside Nepal"}):
        groups[f"province: {p}"] = prov == p
    groups["forest"], groups["non-forest"] = wc == 10, wc != 10
    groups["slope <20"], groups["slope 20-40"], groups["slope >=40"] = slope < 20, (slope >= 20) & (slope < 40), slope >= 40
    groups["on HMA 8 m"], groups["Copernicus-filled"] = srcf == 1, srcf == 2
    common &= prov != "outside Nepal"  # every group: points inside Nepal that all models cover
    R["heldout"] = {"n_common": int(common.sum()), "groups": {}}
    for gname, m in groups.items():
        R["heldout"]["groups"][gname] = {k: st(e[m & common]) for k, e in errs.items()}


def dgps(R, vrt):
    p = CTRL / "khumbu_All_GCPs.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d["Location"] = d["Location"].ffill()
    d = d[d.Easting.between(4e5, 6e5)]
    z0 = d["Ortho. Hgt."].values
    R["dgps"] = {}
    for kind in ("dem", "dtm"):
        with rasterio.open(vrt[kind]) as s:
            z = np.array([v[0] for v in s.sample(zip(d.Easting, d.Northing))], "float64")
        z[~np.isfinite(z)] = np.nan
        for gn, key in (("margins", "margins"), ("glacier surface", "surface"), ("wind tower", "Wind")):
            R["dgps"][f"{kind}/{gn}"] = st((z - z0)[d.Location.str.contains(key).values])
    if not any(R["dgps"].values()):
        R["dgps"] = None  # Khumbu tiles not built yet


def read_window(path, bounds):
    with rasterio.open(path) as s:
        win = from_bounds(*bounds, s.transform).round_offsets().round_lengths()
        a = s.read(1, window=win, boundless=True, fill_value=np.nan).astype("float32")
        return a, s.window_transform(win)


def contours(R, vrt):
    R["contours"] = {}
    to = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    for p in sorted(CTRL.glob("contours_wfs_*.geojson")):
        xs, ys, zs = [], [], []
        for f in json.load(open(p))["features"]:
            g = f["geometry"]
            for ln in (g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]):
                c = np.asarray([q[:2] for q in ln], float)
                if len(c) >= 2:
                    dd = densify(np.column_stack(to.transform(c[:, 0], c[:, 1])), 4.0)
                    xs.append(dd[:, 0]); ys.append(dd[:, 1]); zs.append(np.full(len(dd), float(f["properties"]["cov"])))
        x, y, z = map(np.concatenate, (xs, ys, zs))
        b = (x.min() - 100, y.min() - 100, x.max() + 100, y.max() + 100)
        e = {}
        for kind in ("dem", "dtm"):
            a, T = read_window(vrt[kind], b)
            e[kind] = sample_grid(a, T, x, y) - z
        # Copernicus on the same 8 m grid and points: the contours carry +-10-20 m in mountains, so the
        # numbers only mean something relative to this baseline
        w, s_, e_, n_ = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform_bounds(*b)
        cop = warp([COP.format(lat=la, lon=lo) for la in range(math.floor(s_), math.floor(n_) + 1)
                    for lo in range(math.floor(w), math.floor(e_) + 1)], T, a.shape[1], a.shape[0], Resampling.bilinear)
        e["copernicus"] = sample_grid(cop, T, x, y) - z
        m = np.all([np.isfinite(v) for v in e.values()], 0)
        out = {k: st(v[m]) for k, v in e.items()}
        R["contours"][p.stem.replace("contours_wfs_", "")] = out


def aoi_agreement(R, vrt):
    R["aoi_agreement"] = {}
    for name, f in (("bhotekoshi 8 m", "bhotekoshi_dtm_8m.tif"), ("khumbu 8 m", "khumbu_dtm_8m.tif"),
                    ("bheemdatt 20 m", "bheemdatt_dtm_fused.tif")):
        if not (DATA / f).exists():
            continue
        with rasterio.open(DATA / f) as a, rasterio.open(vrt["dtm"]) as v:
            ref = a.read(1).astype("float32")
            o = np.full(ref.shape, np.nan, "float32")
            reproject(rasterio.band(v, 1), o, dst_transform=a.transform, dst_crs=a.crs,
                      resampling=Resampling.average if a.res[0] > 10 else Resampling.bilinear,
                      src_nodata=np.nan, dst_nodata=np.nan)
        d = o - ref
        m = np.isfinite(d)
        R["aoi_agreement"][name] = None if m.sum() < 100 else dict(
            overlap=float(m.mean()), median=float(np.median(d[m])), nmad=nmad(d[m]),
            p95_abs=float(np.percentile(np.abs(d[m]), 95)))


def seams(R):
    tiles = {p.name[:10]: p for p in (NW / f"out{SUFFIX}").glob("*_dtm.tif")}
    xy = {k: (int(k[1:5]), int(k[6:10])) for k in tiles}
    inv = {v: k for k, v in xy.items()}
    across, inside, steps, worst = [], [], [], []
    for k, (x, y) in xy.items():
        with rasterio.open(tiles[k]) as s:
            A = s.read(1)
        for nb, sl_a, sl_b in (((x + 24, y), (slice(None), slice(-2, None)), (slice(None), slice(0, 2))),
                               ((x, y + 24), (slice(0, 2), slice(None)), (slice(-2, None), slice(None)))):
            if nb not in inv:
                continue
            with rasterio.open(tiles[inv[nb]]) as s:
                B = s.read(1)
            a, b = A[sl_a], B[sl_b]
            if a.shape[1] == 2:  # east neighbour: columns
                edge_a, in_a, edge_b, in_b = a[:, 1], a[:, 0], b[:, 0], b[:, 1]
            else:  # north neighbour: A's top rows meet B's bottom rows
                edge_a, in_a, edge_b, in_b = a[0], a[1], b[1], b[0]
            dx = edge_b - edge_a
            m = np.isfinite(dx)
            if m.sum() < 100:
                continue
            across.append(np.abs(dx[m])); inside.append(np.abs((edge_a - in_a)[m])); inside.append(np.abs((in_b - edge_b)[m]))
            steps.append(float(np.median(dx[m])))
            worst.append((abs(float(np.median(dx[m]))), k, inv[nb]))
    if not across:
        R["seams"] = None
        return
    ac, ins = np.concatenate(across), np.concatenate(inside)
    R["seams"] = dict(n_edges=len(steps), median_abs_step_across=float(np.median(ac)),
                      median_abs_step_inside=float(np.median(ins)), p99_across=float(np.percentile(ac, 99)),
                      p99_inside=float(np.percentile(ins, 99)), median_signed_step_per_edge_p95=float(np.percentile(np.abs(steps), 95)),
                      worst_edges=[dict(step=w[0], a=w[1], b=w[2]) for w in sorted(worst, reverse=True)[:10]])


def shift(R, d):
    s = (d["src"] == 0) & (d["wc"] != 10) & (d["slope"] > 15)
    lon, lat, h = d["lon"][s], d["lat"][s], d["h"][s]
    prov = provinces(lon, lat)
    x, y = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(lon, lat)
    rng = np.random.default_rng(0)
    keep = np.zeros(lon.size, bool)
    for p in set(prov) - {"outside Nepal"}:
        idx = np.flatnonzero(prov == p)
        keep[rng.choice(idx, min(idx.size, 30000), replace=False)] = True
    shifts = [(de, dn) for de in range(-16, 17, 4) for dn in range(-16, 17, 4)]
    E = np.full((len(shifts), lon.size), np.nan, "float32")
    tid = np.array([f"E{int(a // 24000 * 24):04d}N{int(b // 24000 * 24):04d}" for a, b in zip(x, y)])
    for t in np.unique(tid[keep]):
        p = NW / f"out{SUFFIX}" / f"{t}_dem.tif"
        if not p.exists():
            continue
        m = keep & (tid == t)
        with rasterio.open(p) as src:
            a, T = src.read(1), src.transform
        for i, (de, dn) in enumerate(shifts):
            E[i, m] = sample_grid(a, T, x[m] + de, y[m] + dn) - h[m]
    R["shift_vs_icesat2"] = {}
    for p in sorted(set(prov[keep]) - {"outside Nepal"}):
        m = keep & (prov == p)
        v = [nmad(E[i, m]) for i in range(len(shifts))]
        if not np.isfinite(v).any():
            continue
        b = int(np.nanargmin(v))
        R["shift_vs_icesat2"][p] = dict(n=int(m.sum()), shift_east_m=shifts[b][0], shift_north_m=shifts[b][1],
                                        nmad_at_best=v[b], nmad_at_zero=v[shifts.index((0, 0))])


def coverage(R):
    nep = shape(json.load(open(CTRL / "nepal_boundary.geojson"))["features"][0]["geometry"])
    nep_m = shapely.transform(nep, lambda c: np.column_stack(
        Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(c[:, 0], c[:, 1])))
    n = dict(px=0, hma=0, cop=0, gap=0)
    err = np.zeros(256, np.int64)
    for p in sorted((NW / f"out{SUFFIX}").glob("*_aux.tif")):
        with rasterio.open(p) as s:
            src, e = s.read(1), s.read(2)
            inside = rasterize([(nep_m, 1)], out_shape=src.shape, transform=s.transform).astype(bool)
        n["px"] += int(inside.sum()); n["hma"] += int((inside & (src == 1)).sum())
        n["cop"] += int((inside & (src == 2)).sum()); n["gap"] += int((inside & (src == 0)).sum())
        err += np.bincount(e[inside & (src > 0)], minlength=256)
    if not n["px"]:
        R["coverage"] = None
        return
    cdf = np.cumsum(err[:255]) / max(err[:255].sum(), 1)
    q = lambda f: float(np.searchsorted(cdf, f) / 10)
    R["coverage"] = dict(km2_in_tiles=n["px"] * 64 / 1e6, share_hma_8m=n["hma"] / n["px"], share_cop_fill=n["cop"] / n["px"],
                         share_gap=n["gap"] / n["px"], expected_abs_error_m=dict(p50=q(.5), p90=q(.9)))


def markdown(R):
    L = ["# National 8 m DEM/DTM: validation", "", "Generated by `fusion/nw_validate.py` from `validation_report.json`.", ""]
    f = lambda s, k="rmse": "–" if not s else f"{s[k]:.2f}"
    if R.get("heldout"):
        L += [f"## Held-out ICESat-2 ground ({R['heldout']['n_common']:,} points inside Nepal, common to all models)", "",
              "RMSE in m (NMAD in brackets). *Ours, spatial* = 48 km regions unseen in training (headline). "
              "GEDTM30 was trained on ICESat-2, so its score is optimistic.", "",
              "| Group | n | Ours, spatial | Ours, pass | Copernicus | GEDTM30 | FABDEM |", "|---|---|---|---|---|---|---|"]
        for g, v in R["heldout"]["groups"].items():
            if v["ours_spatial"]:
                L.append(f"| {g} | {v['ours_spatial']['n']:,} | " + " | ".join(
                    f"{f(v[k])} ({f(v[k], 'nmad')})" for k in ("ours_spatial", "ours_pass", "copernicus", "gedtm30", "fabdem")) + " |")
        L.append("")
    if R.get("contours"):
        L += ["## Survey Department contours (1:50k, 20 m interval)", "",
              "RMSE (NMAD) in m at the same samples. The contours carry +-10-20 m in mountains: compare against "
              "Copernicus, not against zero.", "", "| Area | n | Copernicus | Our DEM | Our DTM |", "|---|---|---|---|---|"]
        for a, v in R["contours"].items():
            if v.get("dtm"):
                L.append(f"| {a} | {v['dtm']['n']:,} | " + " | ".join(
                    f"{f(v[k])} ({f(v[k], 'nmad')})" for k in ("copernicus", "dem", "dtm")) + " |")
        L.append("")
    for sec, title in (("dgps", "Khumbu dGPS control points"),
                       ("aoi_agreement", "Agreement with validated AOI DTMs"), ("shift_vs_icesat2", "Horizontal shift vs ICESat-2"),
                       ("seams", "Tile seams"), ("coverage", "Coverage inside Nepal")):
        if R.get(sec) is not None:
            L += [f"## {title}", "", "```json", json.dumps(R[sec], indent=1), "```", ""]
    (NW / f"VALIDATION{SUFFIX}.md").write_text("\n".join(L))


def main() -> None:
    R = {}
    vrt = {k: NW / f"nepal_{k}_8m{SUFFIX}.vrt" for k in ("dem", "dtm", "aux")}
    d = dict(np.load(NW / f"points{SUFFIX}.npz", allow_pickle=False))
    steps = (("held-out + external DEMs", lambda: heldout(d, R)), ("dGPS", lambda: dgps(R, vrt)),
             ("contours", lambda: contours(R, vrt)), ("AOI agreement", lambda: aoi_agreement(R, vrt)),
             ("seams", lambda: seams(R)), ("shift", lambda: shift(R, d)), ("coverage", lambda: coverage(R)))
    for name, fn in steps:
        print(f"[{name}]", flush=True)
        try:
            fn()
        except Exception as e:  # one failing check shouldn't hide the others; it is recorded in the report
            R[f"error: {name}"] = repr(e)[:500]
            print(f"  FAILED: {e!r}"[:300], flush=True)
        (NW / f"validation_report{SUFFIX}.json").write_text(json.dumps(R, indent=1, default=float))
    markdown(R)
    print("->", NW / f"validation_report{SUFFIX}.json", NW / f"VALIDATION{SUFFIX}.md")


if __name__ == "__main__":
    main()
