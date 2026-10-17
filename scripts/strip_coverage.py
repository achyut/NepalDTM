"""How much of Nepal's Copernicus-filled area could the free HMA 8 m strips cover?

Two thirds of the delivered grid has no 8 m stereo surface and falls back to Copernicus GLO-30, which is where
NepalDTM is least accurate (6.71 m against 4.00 m on the mosaic) and where summits read furthest low. The HMA
8 m mosaic is not the only 8 m product: NASA also publishes the individual along-track (HMA_DEM8m_AT) and
cross-track (HMA_DEM8m_CT) strips the mosaic was built from, and those cover ground the mosaic left out.

This asks the cheap question first, from granule metadata alone and with nothing downloaded: if every strip
over Nepal were used, how much of the gap would close? A trial in the Bhote Koshi raised 8 m coverage from
80.2% to 84.9% there, so a national figure decides whether the full download (tens of gigabytes) is worth it.

It needs the mosaic coverage raster (0.005 degree, 1 = no mosaic, 2 = mosaic) from the earlier build.

    .venv/bin/python -I scripts/strip_coverage.py --coverage <nepal_hma_coverage.tif>

Writes validation/strip_coverage.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
import shapely
from rasterio.features import rasterize
from shapely.geometry import Polygon, shape

ROOT = Path(__file__).resolve().parent.parent
NEPAL = ROOT / "code" / "data" / "control" / "nepal_boundary.geojson"
OUT = ROOT / "validation" / "strip_coverage.json"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
BBOX = "79.5,26.0,88.7,31.0"


def granules(short_name):
    """Footprint polygons of every granule of a collection touching Nepal."""
    out, page = [], 1
    while True:
        q = urllib.parse.urlencode({"short_name": short_name, "bounding_box": BBOX,
                                    "page_size": 2000, "page_num": page})
        with urllib.request.urlopen(f"{CMR}?{q}", timeout=120) as r:
            entries = json.load(r)["feed"]["entry"]
        for g in entries:
            for poly in g.get("polygons") or []:
                v = list(map(float, poly[0].split()))
                out.append(Polygon(zip(v[1::2], v[0::2])))  # CMR gives lat lon pairs
        if len(entries) < 2000:
            break
        page += 1
    print(f"{short_name}: {len(out)} granule footprints", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", required=True, help="nepal_hma_coverage.tif (1 = no mosaic, 2 = mosaic)")
    args = ap.parse_args()

    with rasterio.open(args.coverage) as src:
        cov = src.read(1)
        transform, shape_ = src.transform, cov.shape
    nepal = cov > 0
    mosaic = cov == 2
    gap = cov == 1
    print(f"Nepal cells {nepal.sum():,}; mosaic {mosaic.sum() / nepal.sum():.1%}; gap {gap.sum() / nepal.sum():.1%}", flush=True)

    strips = {k: granules(k) for k in ("HMA_DEM8m_AT", "HMA_DEM8m_CT")}
    covered = {}
    for name, polys in strips.items():
        covered[name] = rasterize(((p, 1) for p in polys), out_shape=shape_, transform=transform,
                                  dtype="uint8", all_touched=False).astype(bool)
    both = covered["HMA_DEM8m_AT"] | covered["HMA_DEM8m_CT"]

    def share(mask):
        return round(float(mask.sum() / nepal.sum()), 4)

    report = {
        "description": "Share of Nepal that the HMA 8 m strips could add on top of the 8 m mosaic, from granule "
                       "footprints in NASA CMR. A footprint is where a strip exists, not where it has valid "
                       "pixels: stereo matching fails on cloud, snow and deep shadow, so the usable gain is "
                       "smaller than the figure here. In the Bhote Koshi trial 24 of 26 strips passed the "
                       "5 m NMAD match and coverage went from 80.2% to 84.9%.",
        "generated": datetime.date.today().isoformat(),
        "granules": {k: len(v) for k, v in strips.items()},
        "mosaic_share": share(mosaic),
        "gap_share": share(gap),
        "strip_share_of_nepal": share(both & nepal),
        "gap_covered_by_strips": round(float((both & gap).sum() / gap.sum()), 4),
        "share_after_strips": share((mosaic | (both & gap)) & nepal),
        "by_collection": {k: {"gap_covered": round(float((v & gap).sum() / gap.sum()), 4)} for k, v in covered.items()},
    }
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
