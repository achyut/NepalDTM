"""The land around Nepal, for the frame of the map.

The viewer used to end at the national boundary with flat colour all around, which left Nepal looking cut
out of the page. This writes site/data/neighbours.geojson, which the viewer draws very faintly outside the
border:

  * one polygon for the surrounding land, built as the frame minus Nepal itself. It is cut against our own
    boundary rather than against Natural Earth's, because the two disagree by up to a kilometre and any
    difference shows as a pale gap running along the border.
  * the lines between the neighbours, away from Nepal's own border, so the land reads as several countries.
    The stretch of each country's outline that follows Nepal is dropped: our border is already drawn there,
    and Natural Earth's version of it would sit beside ours as a second, crooked line.
  * points for the country names, one wherever a country takes up a side of the frame. India wraps Nepal
    on three sides and wants its name on all three; a label placed from a polygon instead repeats itself
    once per tile.

The map's own frame is the margins below (site/js/terrain.js MAX_BOUNDS), and the land is cut further out
than that: tilted in 3D the view reaches past the frame, and the clip's straight edge should not be the
thing it reaches.

Source: Natural Earth 1:10m admin 0 countries (public domain).

    python3 scripts/neighbours.py <ne_10m_admin_0_countries.geojson>
"""

import json
import math
import sys
from pathlib import Path

from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
MARGIN_NS_KM = 50.0    # how far past the border the map may be panned, north and south
MARGIN_EW_KM = 175.0   # and east and west: the panel takes a fifth of the canvas, and the whole country
                       # still has to fit beside it, which costs more room than 50 km leaves
CLIP_KM = 280.0        # how far out the land is kept, so its straight edge stays out of view
SIMPLIFY_DEG = 0.0015  # about 150 m: finer than the frame is ever drawn at
SHARED_DEG = 0.04      # about 4 km: how far from our border a neighbour's own outline is dropped
LABEL_MIN_SQ_DEG = 0.1 # a side of the frame has to hold about this much of a country to be worth naming


def frame(nepal, ns_km, ew_km=None):
    """Nepal's bounding box grown by `ns_km` north and south and `ew_km` east and west."""
    west, south, east, north = nepal.bounds
    dlat = ns_km / 110.574
    dlng = (ew_km if ew_km is not None else ns_km) / (111.320 * math.cos(math.radians((south + north) / 2)))
    return box(west - dlng, south - dlat, east + dlng, north + dlat)


def sides(nepal, visible):
    """The four strips of the frame around Nepal's bounding box: west, east, south, north."""
    west, south, east, north = nepal.bounds
    fw, fs, fe, fn = visible.bounds
    return [box(fw, fs, west, fn), box(east, fs, fe, fn), box(west, fs, east, south), box(west, north, east, fn)]


def main(source):
    nepal = unary_union([shape(f["geometry"])
                         for f in json.loads((ROOT / "site/data/nepal.geojson").read_text())["features"]])
    window = frame(nepal, CLIP_KM)
    visible = frame(nepal, MARGIN_NS_KM, MARGIN_EW_KM)
    along_border = nepal.boundary.buffer(SHARED_DEG)

    land = window.difference(nepal).simplify(SIMPLIFY_DEG, preserve_topology=True)
    features = [{"type": "Feature", "properties": {"kind": "land"}, "geometry": mapping(land)}]

    for f in json.loads(Path(source).read_text())["features"]:
        name = f["properties"]["NAME"]
        if name == "Nepal":
            continue
        country = shape(f["geometry"]).intersection(window)
        if country.is_empty:
            continue
        edge = country.boundary.difference(along_border).difference(window.boundary.buffer(1e-6))
        if not edge.is_empty:
            features.append({"type": "Feature", "properties": {"name": name, "kind": "edge"},
                             "geometry": mapping(edge.simplify(SIMPLIFY_DEG, preserve_topology=True))})
        # Names go in the band the map can actually reach, not in the wider clip, which would park them
        # somewhere nobody can pan to; and one to a side, so India reads west, south and east.
        band = country.difference(nepal).intersection(visible)
        for side in sides(nepal, visible):
            here = band.intersection(side)
            if here.area < LABEL_MIN_SQ_DEG:
                continue
            parts = here.geoms if here.geom_type == "MultiPolygon" else [here]
            widest = max(parts, key=lambda g: g.area)
            features.append({"type": "Feature", "properties": {"name": name, "kind": "label"},
                             "geometry": mapping(widest.representative_point())})

    out = ROOT / "site/data/neighbours.geojson"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")))
    w, s, e, n = visible.bounds
    print(f"MAX_BOUNDS [[{w:.2f}, {s:.2f}], [{e:.2f}, {n:.2f}]]")
    named = ", ".join(sorted(f["properties"]["name"] for f in features if f["properties"]["kind"] == "label"))
    print(f"{out.relative_to(ROOT)}: labelled {named} — {out.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main(sys.argv[1])
