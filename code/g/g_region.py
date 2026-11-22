"""Apply G to a whole province in resumable 10 km tiles without seams.

Windows sit on one global lattice of the 8 m v2 grid (every STRIDE cells), and each window's starting noise is
seeded by its lattice position. With deterministic DDIM sampling a window then gives the same 1 m surface whichever
tile computes it, so tiles that share windows agree exactly at their edges. Each tile computes every window that
touches its core, blends them, crops to the core and masks to the province.

  python g/g_region.py --region Bagmati --steps 10 --ens 2
Output: data/gen/region/<region>/tiles/G_<row>_<col>.tif (1 m, LERC 1 cm), progress.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box, shape
from shapely.ops import transform as stransform

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "g"))
from g_apply import DTM, WIN, feather  # noqa: E402
from g_model import CK, DEV, UNet, project_area, sample  # noqa: E402

PROV = ROOT / "data" / "control" / "nepal_provinces.geojson"
NAT_T = Affine(8.0, 0.0, -192000.0, 0.0, -8.0, 3408000.0)  # national v2 8 m grid (EPSG:32645)


class R2:
    """Minimal R2 (S3 API) client for sharing tiles between machines; credentials from the environment."""

    def __init__(self, prefix):
        import boto3
        self.prefix, self.bucket = prefix.rstrip("/") + "/", os.environ["R2_BUCKET"]
        self.s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"], region_name="auto",
                               aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                               aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"])

    def exists(self, fname):
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self.prefix + fname)
            return True
        except self.s3.exceptions.ClientError:
            return False

    def put(self, path):
        self.s3.upload_file(str(path), self.bucket, self.prefix + path.name)


def auto_bs(net, ens, steps, half=False, cap=64, probe=4):
    """Windows per batch that fit the free GPU memory: measure a few windows, keep generous headroom."""
    if DEV.type != "cuda":
        return 8
    x = torch.zeros(probe * ens, 1, WIN, WIN, device=DEV)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=half):
        sample(net, x, n_steps=2)
    per = max((torch.cuda.max_memory_allocated() - base) / probe, 1)
    del x; torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    share = 0.5 if total > 30e9 else 0.35  # small GPUs (T4, L4) fragment sooner
    return int(max(1, min(cap, share * free / per)))


def published(base, fname):
    """True if base/fname answers 200 to a HEAD request."""
    import urllib.request
    try:
        with urllib.request.urlopen(urllib.request.Request(base.rstrip("/") + "/" + fname, method="HEAD", headers={"User-Agent": "curl/8"}), timeout=20) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="Bagmati")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ens", type=int, default=4)
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--tile", type=int, default=52, help="tile core in strides (52 x 24 x 8 m = 9.98 km)")
    ap.add_argument("--fp32", action="store_true", help="full precision on CUDA (default there: float16 autocast, "
                    "several times faster on T4/L4 tensor cores; held-out scores identical)")
    ap.add_argument("--bs", default="auto", help="windows per GPU batch, or auto (fit to free GPU memory)")
    ap.add_argument("--ckpt")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new tiles (testing)")
    ap.add_argument("--dtm", help="8 m input on the national v2 grid (default: the national VRT); may be a subset")
    ap.add_argument("--prov", help="province GeoJSON (default: data/control/nepal_provinces.geojson)")
    ap.add_argument("--out", help="output folder (default: data/gen/region/<region>)")
    ap.add_argument("--reverse", action="store_true", help="process tiles last-first (a second machine)")
    ap.add_argument("--r2-prefix", help="also upload each tile to this R2 key prefix and skip tiles already there "
                    "(needs R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET)")
    ap.add_argument("--delete-uploaded", action="store_true", help="delete each local tile once it is in R2")
    ap.add_argument("--skip-url", help="public base URL; skip tiles another machine has already published there")
    a = ap.parse_args()
    S, TILE = a.stride, a.tile * a.stride
    out = Path(a.out) if a.out else ROOT / "data" / "gen" / "region" / a.region.lower()
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    r2 = R2(a.r2_prefix) if a.r2_prefix else None

    p = Path(a.ckpt) if a.ckpt else sorted(CK.glob("ema_*.pt"))[-1]
    net = UNet().to(DEV)
    s = torch.load(p, map_location=DEV)
    net.load_state_dict(s["ema"]); net.eval()
    half = DEV.type == "cuda" and not a.fp32
    if DEV.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    bs = auto_bs(net, a.ens, a.steps, half) if a.bs == "auto" else int(a.bs)
    print(f"batch size {bs} windows, {'float16' if half else 'float32'}", flush=True)

    src = rasterio.open(a.dtm or DTM)
    # tiles, windows and noise seeds are indexed on the national 8 m grid, so a subset input gives identical tiles
    ocol, orow = (round(v) for v in ~NAT_T * (src.transform.c, src.transform.f))
    assert abs(src.transform.a - 8) < 1e-6 and (ocol, orow) == (~NAT_T * (src.transform.c, src.transform.f)), \
        "input must lie on the national 8 m grid"
    feats = json.load(open(a.prov or PROV))["features"]
    geom = shape(next(f["geometry"] for f in feats if f["properties"]["Name_E"].lower() == a.region.lower()))
    geom = stransform(Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform, geom)
    inv = ~NAT_T
    c0, r0 = inv * (geom.bounds[0], geom.bounds[3])
    c1, r1 = inv * (geom.bounds[2], geom.bounds[1])
    tiles = [(ti, tj) for ti in range(int(r0) // TILE, int(r1) // TILE + 1)
             for tj in range(int(c0) // TILE, int(c1) // TILE + 1)
             if geom.intersects(box(*rasterio.windows.bounds(Window(tj * TILE, ti * TILE, TILE, TILE), NAT_T)))]
    name = lambda t: f"G_{t[0]:03d}_{t[1]:03d}.tif"
    todo = [t for t in tiles if not (out / "tiles" / name(t)).exists()]
    if a.reverse:
        todo = todo[::-1]
    print(f"{a.region}: {len(tiles)} tiles of {TILE * 8 / 1000:.2f} km, {len(todo)} to do; checkpoint {p.name} "
          f"(step {s['step']}), {a.steps} steps x {a.ens} samples, device {DEV}", flush=True)

    fw = feather(ov=(WIN - S) * 8)
    t0, nwin = time.time(), 0
    for n, (ti, tj) in enumerate(todo[:a.limit or None]):
        if r2 and r2.exists(name((ti, tj))) or a.skip_url and published(a.skip_url, name((ti, tj))):
            continue  # done by another machine
        rc0, cc0 = ti * TILE, tj * TILE  # tile core, national 8 m grid
        i0, i1 = -(-(rc0 - WIN + 1) // S), (rc0 + TILE - 1) // S  # windows touching the core
        j0, j1 = -(-(cc0 - WIN + 1) // S), (cc0 + TILE - 1) // S
        R, C = i0 * S, j0 * S
        H, W = (i1 - i0) * S + WIN, (j1 - j0) * S + WIN
        m8 = src.read(1, window=Window(C - ocol, R - orow, W, H), boundless=True, fill_value=np.nan).astype("float32")
        m8[m8 < -1000] = np.nan  # nodata of a subset written with a numeric fill
        bad = ~np.isfinite(m8)
        if bad.all():
            continue
        if bad.any():  # voids and off-grid: nearest valid height, masked in the output
            from scipy import ndimage
            m8 = m8[tuple(ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True))]
        jobs = [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)
                if not bad[(i - i0) * S:(i - i0) * S + WIN, (j - j0) * S:(j - j0) * S + WIN].all()]
        acc = np.zeros((H * 8, W * 8), "float32"); wsum = np.zeros_like(acc)
        k = 0
        while k < len(jobs):
            chunk = jobs[k:k + bs]
            x = torch.from_numpy(np.stack([m8[(i - i0) * S:(i - i0) * S + WIN, (j - j0) * S:(j - j0) * S + WIN]
                                           for i, j in chunk]))[:, None].to(DEV).repeat_interleave(a.ens, 0)
            noise = torch.cat([torch.randn(a.ens, 1, 256, 256, generator=torch.Generator().manual_seed(i * 100003 + j))
                               for i, j in chunk])
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=half):
                    z = sample(net, x, n_steps=a.steps, noise=noise).float()
            except torch.OutOfMemoryError:
                # results do not depend on the batch split (noise is seeded per window), so shrink and retry
                del x, noise
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print(f"  GPU out of memory: batch size now {bs} for the rest of the run", flush=True)
                continue
            k += len(chunk)
            z = z.view(len(chunk), a.ens, 256, 256).median(1).values.cpu().numpy()
            for (i, j), zz in zip(chunk, z):
                r, c = (i - i0) * S * 8, (j - j0) * S * 8
                acc[r:r + 256, c:c + 256] += zz * fw
                wsum[r:r + 256, c:c + 256] += fw
        nwin += len(jobs)
        dr, dc = (rc0 - R) * 8, (cc0 - C) * 8
        with np.errstate(invalid="ignore"):
            full = acc / wsum
        cov = wsum > 0  # cells outside every window (void-only windows skipped) stay at M before projection
        full[~cov] = np.kron(m8, np.ones((8, 8), "float32"))[~cov]
        g1 = project_area(full, m8)[dr:dr + TILE * 8, dc:dc + TILE * 8]
        g1[np.kron(bad[rc0 - R:rc0 - R + TILE, cc0 - C:cc0 - C + TILE], np.ones((8, 8), bool))] = np.nan
        x0, y0 = NAT_T * (cc0, rc0)
        T1 = from_origin(x0, y0, 1.0, 1.0)
        g1[geometry_mask([geom], g1.shape, T1)] = np.nan
        prof = dict(driver="COG", width=g1.shape[1], height=g1.shape[0], count=1, dtype="float32", crs=src.crs,
                    transform=T1, nodata=np.nan, compress="LERC_DEFLATE", max_z_error=0.01, blocksize=512)
        f = out / "tiles" / name((ti, tj))
        with rasterio.open(f.with_suffix(".part.tif"), "w", **prof) as o:
            o.write(g1.astype("float32"), 1)
        f.with_suffix(".part.tif").rename(f)
        if r2:
            r2.put(f)
            if a.delete_uploaded:
                f.unlink()
        el = time.time() - t0
        left = len(todo) - n - 1
        print(f"  tile {ti},{tj}: {len(jobs)} windows; {n + 1}/{len(todo)} done, {el / nwin:.2f} s/window, "
              f"~{el / (n + 1) * left / 3600:.1f} h left", flush=True)
        (out / "progress.json").write_text(json.dumps(dict(tiles=len(tiles), done=len(tiles) - left,
                                                           s_per_window=el / nwin, steps=a.steps, ens=a.ens,
                                                           ckpt=p.name)))
    print(f"done {time.strftime('%c')}", flush=True)


if __name__ == "__main__":
    main()
