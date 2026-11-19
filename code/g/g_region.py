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
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="Bagmati")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ens", type=int, default=4)
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--tile", type=int, default=52, help="tile core in strides (52 x 24 x 8 m = 9.98 km)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--ckpt")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new tiles (testing)")
    a = ap.parse_args()
    S, TILE = a.stride, a.tile * a.stride
    out = ROOT / "data" / "gen" / "region" / a.region.lower()
    (out / "tiles").mkdir(parents=True, exist_ok=True)

    p = Path(a.ckpt) if a.ckpt else sorted(CK.glob("ema_*.pt"))[-1]
    net = UNet().to(DEV)
    s = torch.load(p, map_location=DEV)
    net.load_state_dict(s["ema"]); net.eval()

    src = rasterio.open(DTM)
    feats = json.load(open(PROV))["features"]
    geom = shape(next(f["geometry"] for f in feats if f["properties"]["Name_E"].lower() == a.region.lower()))
    geom = stransform(Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform, geom)
    inv = ~src.transform
    c0, r0 = inv * (geom.bounds[0], geom.bounds[3])
    c1, r1 = inv * (geom.bounds[2], geom.bounds[1])
    tiles = [(ti, tj) for ti in range(int(r0) // TILE, int(r1) // TILE + 1)
             for tj in range(int(c0) // TILE, int(c1) // TILE + 1)
             if geom.intersects(box(*rasterio.windows.bounds(Window(tj * TILE, ti * TILE, TILE, TILE), src.transform)))]
    todo = [t for t in tiles if not (out / "tiles" / f"G_{t[0]:03d}_{t[1]:03d}.tif").exists()]
    print(f"{a.region}: {len(tiles)} tiles of {TILE * 8 / 1000:.2f} km, {len(todo)} to do; checkpoint {p.name} "
          f"(step {s['step']}), {a.steps} steps x {a.ens} samples, device {DEV}", flush=True)

    fw = feather(ov=(WIN - S) * 8)
    t0, nwin = time.time(), 0
    for n, (ti, tj) in enumerate(todo[:a.limit or None]):
        rc0, cc0 = ti * TILE, tj * TILE  # tile core, 8 m grid
        i0, i1 = -(-(rc0 - WIN + 1) // S), (rc0 + TILE - 1) // S  # windows touching the core
        j0, j1 = -(-(cc0 - WIN + 1) // S), (cc0 + TILE - 1) // S
        R, C = i0 * S, j0 * S
        H, W = (i1 - i0) * S + WIN, (j1 - j0) * S + WIN
        m8 = src.read(1, window=Window(C, R, W, H), boundless=True, fill_value=np.nan).astype("float32")
        bad = ~np.isfinite(m8)
        if bad.all():
            continue
        if bad.any():  # voids and off-grid: nearest valid height, masked in the output
            from scipy import ndimage
            m8 = m8[tuple(ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True))]
        jobs = [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)
                if not bad[(i - i0) * S:(i - i0) * S + WIN, (j - j0) * S:(j - j0) * S + WIN].all()]
        acc = np.zeros((H * 8, W * 8), "float32"); wsum = np.zeros_like(acc)
        for k in range(0, len(jobs), a.bs):
            chunk = jobs[k:k + a.bs]
            x = torch.from_numpy(np.stack([m8[(i - i0) * S:(i - i0) * S + WIN, (j - j0) * S:(j - j0) * S + WIN]
                                           for i, j in chunk]))[:, None].to(DEV).repeat_interleave(a.ens, 0)
            noise = torch.cat([torch.randn(a.ens, 1, 256, 256, generator=torch.Generator().manual_seed(i * 100003 + j))
                               for i, j in chunk])
            with torch.no_grad():
                z = sample(net, x, n_steps=a.steps, noise=noise).float()
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
        x0, y0 = src.transform * (cc0, rc0)
        T1 = from_origin(x0, y0, 1.0, 1.0)
        g1[geometry_mask([geom], g1.shape, T1)] = np.nan
        prof = dict(driver="COG", width=g1.shape[1], height=g1.shape[0], count=1, dtype="float32", crs=src.crs,
                    transform=T1, nodata=np.nan, compress="LERC_DEFLATE", max_z_error=0.01, blocksize=512)
        f = out / "tiles" / f"G_{ti:03d}_{tj:03d}.tif"
        with rasterio.open(f.with_suffix(".part.tif"), "w", **prof) as o:
            o.write(g1.astype("float32"), 1)
        f.with_suffix(".part.tif").rename(f)
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
