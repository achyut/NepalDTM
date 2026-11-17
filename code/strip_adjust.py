#!/usr/bin/env python3
"""Joint adjustment of the NSIDC HMA 8 m stereo strips against ICESat-2 ground photons.

Each strip k gets a plane correction c_k(x, y) = a_k + b_k * dx + c_k * dy (dx, dy in 10 km from the AOI
centre). Unknowns are solved together by robust (Huber IRLS) sparse least squares from:
  1. photon ties: strip_k(x, y) + c_k - h_photon = 0, at training-track photons (holdout == False) on stable
     open ground: WorldCover not tree / snow / water, slope < 35 deg, |residual| < 25 m (cloud, change);
  2. overlap ties: (strip_k + c_k) - (strip_j + c_j) = 0 at subsampled cells where both are valid;
  3. a weak prior pulling every parameter to zero (keeps isolated strips defined).
The composite is the per-pixel median of the corrected strips, with count and NMAD.

All heights are EGM2008 on the national 8 m grid (EPSG:32645). Strips are WGS84 ellipsoidal.
Library module: used by nw_strip_adjust.py for the national run and by the Langtang test gate.
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from scipy import sparse
from scipy.sparse.linalg import lsqr

from build_stack_hma import CRS, geoid_offset

NW = Path(__file__).resolve().parent / "data" / "nw"
STRIPS = NW / "hma_strips"
STABLE_WC_EXCLUDE = (10, 70, 80)  # tree cover, snow/ice, water


def grid(bounds, res=8.0):
    x0, y0 = np.floor(bounds[0] / res) * res, np.floor(bounds[1] / res) * res
    x1, y1 = np.ceil(bounds[2] / res) * res, np.ceil(bounds[3] / res) * res
    W, H = int((x1 - x0) / res), int((y1 - y0) / res)
    return from_origin(x0, y1, res, res), W, H


def strips_in(bounds) -> list[Path]:
    out = []
    for f in sorted(glob.glob(str(STRIPS / "*" / "*.tif"))):
        with rasterio.open(f) as s:
            b = transform_bounds(s.crs, CRS, *s.bounds)
        if b[0] < bounds[2] and b[2] > bounds[0] and b[1] < bounds[3] and b[3] > bounds[1]:
            out.append(Path(f))
    return out


def warp_strips(files, T, W, H, z_geoid) -> np.ndarray:
    """(K, H, W) float32 strip heights on the grid, EGM2008, NaN outside."""
    A = np.full((len(files), H, W), np.nan, "float32")
    for k, f in enumerate(files):
        with rasterio.open(f) as s:
            reproject(rasterio.band(s, 1), A[k], dst_transform=T, dst_crs=CRS, resampling=Resampling.bilinear,
                      src_nodata=s.nodata, dst_nodata=np.nan)
        A[k] += z_geoid
    return A


def load_photons(bounds, holdout: bool | None) -> pd.DataFrame:
    w, s, e, n = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform_bounds(*bounds)
    parts = []
    for f in sorted((NW / "atl03").glob("*.parquet")):
        lon, lat = float(f.stem[1:f.stem.index("N")]), float(f.stem[f.stem.index("N") + 1:])
        if lon > e or lon + 0.5 < w or lat > n or lat + 0.5 < s:
            continue
        d = pd.read_parquet(f, columns=["lon", "lat", "h", "rgt", "t", "holdout"])
        d = d[(d.lon >= w) & (d.lon <= e) & (d.lat >= s) & (d.lat <= n)]
        if holdout is not None:
            d = d[d.holdout == holdout]
        parts.append(d)
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["lon", "lat", "h"])
    x, y = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(d.lon.values, d.lat.values)
    return d.assign(x=x, y=y)


def pix(T, x, y):
    inv = ~T
    c = np.floor(inv.a * x + inv.b * y + inv.c).astype(np.int64)
    r = np.floor(inv.d * x + inv.e * y + inv.f).astype(np.int64)
    return r, c


def solve(A, T, phot, stable, n_overlap=4000, iters=6, prior=1e-3, max_photons=200_000, verbose=True):
    """Return params (K, 3): offset, tilt per 10 km east, tilt per 10 km north."""
    K, H, W = A.shape
    cx, cy = T * (W / 2, H / 2)
    R, C, V, B, KIND = [], [], [], [], []
    eq = 0

    def add(k, x, y, d, kind, j=None):
        """Vectorised: one equation per element; strip k (+1) and optionally strip j (-1)."""
        nonlocal eq
        n = d.size
        e = np.arange(eq, eq + n)
        dx, dy = (x - cx) / 1e4, (y - cy) / 1e4
        co = np.stack([np.ones(n), dx, dy], 1)
        R.append(np.repeat(e, 3)); C.append((3 * k + np.arange(3))[None].repeat(n, 0).ravel()); V.append(co.ravel())
        if j is not None:
            R.append(np.repeat(e, 3)); C.append((3 * j + np.arange(3))[None].repeat(n, 0).ravel())
            V.append(-co.ravel())
        B.append(-d); KIND.append(np.full(n, kind)); eq += n

    rng = np.random.default_rng(0)
    r, c = pix(T, phot.x.values, phot.y.values)
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    r, c, h = r[ok], c[ok], phot.h.values[ok]
    st = stable[r, c]
    r, c, h = r[st], c[st], h[st]
    if r.size > max_photons:
        sel = rng.choice(r.size, max_photons, replace=False)
        r, c, h = r[sel], c[sel], h[sel]
    xs, ys = T * (c + 0.5, r + 0.5)
    n_ph = 0
    for k in range(K):
        v = A[k, r, c]
        m = np.isfinite(v)
        if m.sum() < 20:
            continue
        d = v[m] - h[m]
        m2 = np.abs(d - np.median(d)) < 25
        add(k, xs[m][m2], ys[m][m2], d[m2], 0)
        n_ph += int(m2.sum())
    valid = np.isfinite(A)
    n_ov = 0
    for k in range(K):
        for j in range(k + 1, K):
            both = valid[k] & valid[j] & stable
            if both.sum() < 200:
                continue
            ii = np.flatnonzero(both.ravel())
            ii = rng.choice(ii, min(n_overlap, ii.size), replace=False)
            rr, cc = np.unravel_index(ii, (H, W))
            d = A[k, rr, cc] - A[j, rr, cc]
            keep = np.abs(d - np.median(d)) < 25
            x2, y2 = T * (cc[keep] + 0.5, rr[keep] + 0.5)
            add(k, np.asarray(x2), np.asarray(y2), d[keep], 1, j=j)
            n_ov += int(keep.sum())
    e = np.arange(eq, eq + 3 * K)
    R.append(e); C.append(np.arange(3 * K)); V.append(np.ones(3 * K)); B.append(np.zeros(3 * K))
    KIND.append(np.full(3 * K, 2)); eq += 3 * K
    M = sparse.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(eq, 3 * K))
    b = np.concatenate(B)
    kind = np.concatenate(KIND)
    base_w = np.where(kind == 0, 1.0, np.where(kind == 1, 0.5, prior))
    w = base_w.copy()
    for it in range(iters):
        sw = np.sqrt(w)
        x = lsqr(sparse.diags(sw) @ M, sw * b, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
        res = M @ x - b
        s = 1.4826 * np.median(np.abs(res[kind < 2])) + 1e-6
        u = np.abs(res) / (1.5 * s)
        w = base_w * np.where(u <= 1, 1.0, 1.0 / u)  # Huber
        if verbose:
            print(f"    IRLS {it + 1}: photon NMAD {1.4826 * np.median(np.abs(res[kind == 0])):.2f} m, "
                  f"overlap NMAD {1.4826 * np.median(np.abs(res[kind == 1])) if (kind == 1).any() else 0:.2f} m",
                  flush=True)
    return x.reshape(K, 3), {"n_photon_ties": n_ph, "n_overlap_ties": n_ov, "n_strips": K}


def apply_planes(A, T, params) -> np.ndarray:
    K, H, W = A.shape
    cx, cy = T * (W / 2, H / 2)
    cols, rows = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    x, y = T * (cols, rows)
    dx, dy = (x - cx) / 1e4, (y - cy) / 1e4
    out = np.empty_like(A)
    for k in range(K):
        out[k] = A[k] + (params[k, 0] + params[k, 1] * dx + params[k, 2] * dy).astype("float32")
    return out


def composite(A) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cnt = np.isfinite(A).sum(0).astype("float32")
    with np.errstate(all="ignore"):
        med = np.nanmedian(A, 0)
        nmad = 1.4826 * np.nanmedian(np.abs(A - med[None]), 0)
    nmad[cnt < 2] = np.nan
    return med.astype("float32"), cnt, nmad.astype("float32")


def stable_mask(wc, slope) -> np.ndarray:
    return ~np.isin(wc, STABLE_WC_EXCLUDE) & (slope < 35) & np.isfinite(slope)


def run_aoi(bounds, wc, slope, verbose=True):
    T, W, H = grid(bounds)
    files = strips_in(bounds)
    if verbose:
        print(f"  {len(files)} strips on {W}x{H} px", flush=True)
    t0 = time.time()
    A = warp_strips(files, T, W, H, geoid_offset(T, W, H))
    # blunders against the per-pixel median of all strips (cloud, snow, matching failures, as in nw_stack)
    med0 = np.nanmedian(A, 0)
    A[np.abs(A - med0[None]) > 40] = np.nan
    if verbose:
        print(f"  warped in {time.time() - t0:.0f} s", flush=True)
    phot = load_photons(bounds, holdout=False)
    stable = stable_mask(wc, slope)
    params, info = solve(A, T, phot, stable, verbose=verbose)
    Ac = apply_planes(A, T, params)
    raw = composite(A)
    adj = composite(Ac)
    info["files"] = [f.name for f in files]
    info["params"] = params.round(3).tolist()
    return T, (W, H), raw, adj, info


# ----------------------------------------------------------------------------- national
# One plane per strip, centred on the strip itself (not on a tile), so the same correction holds in every tile
# the strip crosses. Ties are gathered per tile core (no double counting), solved once for all strips.
ADJ = NW / "adjust"


def strip_index() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Strip file names, centres (K, 2) and bounds (K, 4), all in EPSG:32645. Cached."""
    p = ADJ / "strip_index.json"
    if p.exists():
        d = json.loads(p.read_text())
        return d["files"], np.array(d["centre"]), np.array(d["bounds"])
    files, cen, bnd = [], [], []
    for f in sorted(glob.glob(str(STRIPS / "*" / "*.tif"))):
        with rasterio.open(f) as s:
            b = transform_bounds(s.crs, CRS, *s.bounds)
        files.append(str(Path(f).relative_to(STRIPS)))
        cen.append([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        bnd.append(list(b))
    ADJ.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"files": files, "centre": cen, "bounds": bnd}))
    return files, np.array(cen), np.array(bnd)


def tile_strips(T, W, H, bnd) -> np.ndarray:
    x0, y1 = T * (0, 0)
    x1, y0 = T * (W, H)
    return np.flatnonzero((bnd[:, 0] < x1) & (bnd[:, 2] > x0) & (bnd[:, 1] < y1) & (bnd[:, 3] > y0))


def tile_surface(stack_path):
    """Warped strips (K, H, W) for one stack tile, with blunders removed, plus grid and global strip ids."""
    from fuse_hma import load_stack, terrain
    prof, L = load_stack(stack_path)
    T, W, H = prof["transform"], prof["width"], prof["height"]
    files, _, bnd = strip_index()
    ids = tile_strips(T, W, H, bnd)
    A = warp_strips([STRIPS / files[k] for k in ids], T, W, H, geoid_offset(T, W, H)) if ids.size else \
        np.empty((0, H, W), "float32")
    if ids.size:
        with np.errstate(all="ignore"):
            med0 = np.nanmedian(A, 0)
        A[np.abs(A - med0[None]) > 40] = np.nan
        keep = np.isfinite(A).reshape(ids.size, -1).any(1)
        A, ids = A[keep], ids[keep]
    dsm = L["dsm"]
    slope = terrain(np.where(np.isfinite(dsm), dsm, np.nanmedian(dsm)))["slope"]
    return T, W, H, A, ids, L, slope


def tile_ties(stack_path, core_px=125, n_overlap=1500, max_photons=40_000, seed=0):
    """Photon and overlap ties inside the tile core. Returns dict of arrays (global strip ids)."""
    T, W, H, A, ids, L, slope = tile_surface(stack_path)
    out = {k: np.empty(0, t) for k, t in (("pk", "int32"), ("px", "f8"), ("py", "f8"), ("pd", "f4"),
                                          ("ok", "int32"), ("oj", "int32"), ("ox", "f8"), ("oy", "f8"),
                                          ("od", "f4"))}
    if ids.size == 0:
        return out
    stable = stable_mask(L["wc"], slope)
    core = np.zeros((H, W), bool)
    core[core_px:H - core_px, core_px:W - core_px] = True
    stable &= core
    rng = np.random.default_rng(seed)
    x0, y1 = T * (0, 0)
    x1, y0 = T * (W, H)
    phot = load_photons((x0, y0, x1, y1), holdout=False)
    r, c = pix(T, phot.x.values, phot.y.values)
    m = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    r, c, h = r[m], c[m], phot.h.values[m]
    m = stable[r, c]
    r, c, h = r[m], c[m], h[m]
    if r.size > max_photons:
        sel = rng.choice(r.size, max_photons, replace=False)
        r, c, h = r[sel], c[sel], h[sel]
    xs, ys = T * (c + 0.5, r + 0.5)
    P = {k: [] for k in ("pk", "px", "py", "pd")}
    for a, k in enumerate(ids):
        v = A[a, r, c]
        f = np.isfinite(v)
        if f.sum() < 20:
            continue
        d = v[f] - h[f]
        g = np.abs(d - np.median(d)) < 25
        P["pk"].append(np.full(g.sum(), k, "int32")); P["px"].append(np.asarray(xs)[f][g])
        P["py"].append(np.asarray(ys)[f][g]); P["pd"].append(d[g].astype("float32"))
    O = {k: [] for k in ("ok", "oj", "ox", "oy", "od")}
    valid = np.isfinite(A)
    for a in range(ids.size):
        for b in range(a + 1, ids.size):
            both = valid[a] & valid[b] & stable
            if both.sum() < 200:
                continue
            ii = rng.choice(np.flatnonzero(both.ravel()), min(n_overlap, int(both.sum())), replace=False)
            rr, cc = np.unravel_index(ii, (H, W))
            d = A[a, rr, cc] - A[b, rr, cc]
            g = np.abs(d - np.median(d)) < 25
            xx, yy = T * (cc[g] + 0.5, rr[g] + 0.5)
            O["ok"].append(np.full(g.sum(), ids[a], "int32")); O["oj"].append(np.full(g.sum(), ids[b], "int32"))
            O["ox"].append(np.asarray(xx)); O["oy"].append(np.asarray(yy)); O["od"].append(d[g].astype("float32"))
    for k, v in (P | O).items():
        if v:
            out[k] = np.concatenate(v)
    return out


def solve_global(ties: dict, centre: np.ndarray, iters=8, prior=1e-3, verbose=True):
    """Huber-IRLS solve for (K, 3) planes from concatenated tile ties; planes centred on each strip."""
    K = len(centre)

    def coeff(k, x, y):
        return np.stack([np.ones(k.size), (x - centre[k, 0]) / 1e4, (y - centre[k, 1]) / 1e4], 1)

    np_, no = ties["pk"].size, ties["ok"].size
    cp = coeff(ties["pk"], ties["px"], ties["py"])
    ck = coeff(ties["ok"], ties["ox"], ties["oy"])
    cj = coeff(ties["oj"], ties["ox"], ties["oy"])
    e_p = np.arange(np_)
    e_o = np.arange(np_, np_ + no)
    e_r = np.arange(np_ + no, np_ + no + 3 * K)
    rows = np.concatenate([np.repeat(e_p, 3), np.repeat(e_o, 3), np.repeat(e_o, 3), e_r])
    cols = np.concatenate([(3 * ties["pk"][:, None] + np.arange(3)).ravel(),
                           (3 * ties["ok"][:, None] + np.arange(3)).ravel(),
                           (3 * ties["oj"][:, None] + np.arange(3)).ravel(), np.arange(3 * K)])
    vals = np.concatenate([cp.ravel(), ck.ravel(), -cj.ravel(), np.ones(3 * K)])
    M = sparse.csr_matrix((vals, (rows, cols)), shape=(np_ + no + 3 * K, 3 * K))
    b = np.concatenate([-ties["pd"], -ties["od"], np.zeros(3 * K)]).astype("float64")
    kind = np.concatenate([np.zeros(np_), np.ones(no), np.full(3 * K, 2)])
    base_w = np.where(kind == 0, 1.0, np.where(kind == 1, 0.5, prior))
    w = base_w.copy()
    for it in range(iters):
        sw = np.sqrt(w)
        x = lsqr(sparse.diags(sw) @ M, sw * b, atol=1e-10, btol=1e-10, iter_lim=50000)[0]
        res = M @ x - b
        s = 1.4826 * np.median(np.abs(res[kind < 2])) + 1e-6
        u = np.abs(res) / (1.5 * s)
        w = base_w * np.where(u <= 1, 1.0, 1.0 / u)
        if verbose:
            print(f"  IRLS {it + 1}: photon NMAD {1.4826 * np.median(np.abs(res[kind == 0])):.2f} m, "
                  f"overlap NMAD {1.4826 * np.median(np.abs(res[kind == 1])):.2f} m", flush=True)
    return x.reshape(K, 3), {"n_photon_ties": int(np_), "n_overlap_ties": int(no), "n_strips": int(K),
                             "strips_with_photons": int(np.unique(ties["pk"]).size)}


def tile_composite(stack_path, params, centre):
    """Adjusted per-pixel median, count and NMAD on the stack tile grid (3250 px incl. buffer)."""
    T, W, H, A, ids, L, _ = tile_surface(stack_path)
    if ids.size == 0:
        nan = np.full((H, W), np.nan, "float32")
        return T, nan, np.zeros((H, W), "float32"), nan
    cols, rows = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    x, y = T * (cols, rows)
    for a, k in enumerate(ids):
        A[a] += (params[k, 0] + params[k, 1] * (x - centre[k, 0]) / 1e4
                 + params[k, 2] * (y - centre[k, 1]) / 1e4).astype("float32")
    med, cnt, nmad = composite(A)
    return T, med, cnt, nmad


# ----------------------------------------------------------------------------- smooth (grid) correction
# EXP029: a plane per strip leaves along-strip warping (photon misfit 1.22 m NMAD nationally). Here each strip
# gets a bilinear correction surface on knots every `knot` metres over its own bounds, with a first-difference
# smoothness penalty between neighbouring knots. Same ties as the plane solve.

def aoi_ties(A, T, phot, stable, n_overlap=4000, max_photons=200_000, seed=0) -> dict:
    """Photon and overlap ties on an AOI stack (local strip ids 0..K-1), same filters as tile_ties."""
    K, H, W = A.shape
    rng = np.random.default_rng(seed)
    r, c = pix(T, phot.x.values, phot.y.values)
    m = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    r, c, h = r[m], c[m], phot.h.values[m]
    m = stable[r, c]
    r, c, h = r[m], c[m], h[m]
    if r.size > max_photons:
        sel = rng.choice(r.size, max_photons, replace=False)
        r, c, h = r[sel], c[sel], h[sel]
    xs, ys = (np.asarray(v) for v in T * (c + 0.5, r + 0.5))
    P = {k: [] for k in ("pk", "px", "py", "pd")}
    for k in range(K):
        v = A[k, r, c]
        f = np.isfinite(v)
        if f.sum() < 20:
            continue
        d = v[f] - h[f]
        g = np.abs(d - np.median(d)) < 25
        P["pk"].append(np.full(g.sum(), k, "int32")); P["px"].append(xs[f][g]); P["py"].append(ys[f][g])
        P["pd"].append(d[g])
    O = {k: [] for k in ("ok", "oj", "ox", "oy", "od")}
    valid = np.isfinite(A)
    for k in range(K):
        for j in range(k + 1, K):
            both = valid[k] & valid[j] & stable
            if both.sum() < 200:
                continue
            ii = rng.choice(np.flatnonzero(both.ravel()), min(n_overlap, int(both.sum())), replace=False)
            rr, cc = np.unravel_index(ii, (H, W))
            d = A[k, rr, cc] - A[j, rr, cc]
            g = np.abs(d - np.median(d)) < 25
            xx, yy = T * (cc[g] + 0.5, rr[g] + 0.5)
            O["ok"].append(np.full(g.sum(), k, "int32")); O["oj"].append(np.full(g.sum(), j, "int32"))
            O["ox"].append(np.asarray(xx)); O["oy"].append(np.asarray(yy)); O["od"].append(d[g])
    return {k: np.concatenate(v) if v else np.empty(0) for k, v in (P | O).items()}


class KnotGrid:
    """Per-strip bilinear knot grids; global unknown index = offset[k] + local knot index."""

    def __init__(self, bounds_k: np.ndarray, knot: float):
        self.knot = knot
        self.x0 = bounds_k[:, 0] - knot / 2
        self.y0 = bounds_k[:, 1] - knot / 2
        self.nx = np.ceil((bounds_k[:, 2] - self.x0) / knot).astype(int) + 2
        self.ny = np.ceil((bounds_k[:, 3] - self.y0) / knot).astype(int) + 2
        self.off = np.r_[0, np.cumsum(self.nx * self.ny)]
        self.n = int(self.off[-1])

    def weights(self, k, x, y):
        """(n, 4) global indices and bilinear weights."""
        u = np.clip((x - self.x0[k]) / self.knot, 0, self.nx[k] - 1.001)
        v = np.clip((y - self.y0[k]) / self.knot, 0, self.ny[k] - 1.001)
        i, j = np.floor(u).astype(int), np.floor(v).astype(int)
        fu, fv = u - i, v - j
        base = self.off[k] + j * self.nx[k] + i
        idx = np.stack([base, base + 1, base + self.nx[k], base + self.nx[k] + 1], 1)
        w = np.stack([(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv], 1)
        return idx, w

    def smooth_rows(self):
        """First differences between east and north neighbouring knots of the same strip."""
        a, b = [], []
        for k in range(len(self.nx)):
            g = self.off[k] + np.arange(self.nx[k] * self.ny[k]).reshape(self.ny[k], self.nx[k])
            a += [g[:, :-1].ravel(), g[:-1, :].ravel()]
            b += [g[:, 1:].ravel(), g[1:, :].ravel()]
        return np.concatenate(a), np.concatenate(b)


def solve_grid(ties: dict, grid_: KnotGrid, smooth=3.0, prior=1e-3, iters=6, verbose=True):
    """Huber-IRLS solve for all knot values. smooth weights the neighbour differences (per knot pair)."""
    np_, no = ties["pk"].size, ties["ok"].size
    ip, wp = grid_.weights(ties["pk"].astype(int), ties["px"], ties["py"]) if np_ else (np.empty((0, 4), int), np.empty((0, 4)))
    ik, wk = grid_.weights(ties["ok"].astype(int), ties["ox"], ties["oy"])
    ij, wj = grid_.weights(ties["oj"].astype(int), ties["ox"], ties["oy"])
    sa_, sb_ = grid_.smooth_rows()
    ns, n = sa_.size, grid_.n
    e_p, e_o = np.arange(np_), np_ + np.arange(no)
    e_s, e_r = np_ + no + np.arange(ns), np_ + no + ns + np.arange(n)
    rows = np.concatenate([np.repeat(e_p, 4), np.repeat(e_o, 4), np.repeat(e_o, 4), e_s, e_s, e_r])
    cols = np.concatenate([ip.ravel(), ik.ravel(), ij.ravel(), sa_, sb_, np.arange(n)])
    vals = np.concatenate([wp.ravel(), wk.ravel(), -wj.ravel(), np.ones(ns), -np.ones(ns), np.ones(n)])
    M = sparse.csr_matrix((vals, (rows, cols)), shape=(np_ + no + ns + n, n))
    b = np.concatenate([-np.asarray(ties["pd"], float), -np.asarray(ties["od"], float), np.zeros(ns + n)])
    kind = np.concatenate([np.zeros(np_), np.ones(no), np.full(ns, 3), np.full(n, 2)])
    base_w = np.select([kind == 0, kind == 1, kind == 3], [1.0, 0.5, smooth ** 2], prior)
    w = base_w.copy()
    for it in range(iters):
        sw = np.sqrt(w)
        x = lsqr(sparse.diags(sw) @ M, sw * b, atol=1e-10, btol=1e-10, iter_lim=50000)[0]
        res = M @ x - b
        obs = kind < 2
        s = 1.4826 * np.median(np.abs(res[obs])) + 1e-6
        u = np.abs(res) / (1.5 * s)
        w = np.where(obs, base_w * np.where(u <= 1, 1.0, 1.0 / u), base_w)
        if verbose:
            print(f"    IRLS {it + 1}: photon NMAD {1.4826 * np.median(np.abs(res[kind == 0])):.2f} m, "
                  f"overlap NMAD {1.4826 * np.median(np.abs(res[kind == 1])):.2f} m", flush=True)
    return x


def apply_grid(A, T, grid_: KnotGrid, x) -> np.ndarray:
    K, H, W = A.shape
    cols, rows = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    px, py = T * (cols.ravel(), rows.ravel())
    px, py = np.asarray(px), np.asarray(py)
    out = np.empty_like(A)
    for k in range(K):
        m = np.isfinite(A[k].ravel())
        corr = np.zeros(H * W, "float32")
        if m.any():
            idx, w = grid_.weights(np.full(m.sum(), k), px[m], py[m])
            corr[m] = (x[idx] * w).sum(1)
        out[k] = A[k] + corr.reshape(H, W)
    return out
