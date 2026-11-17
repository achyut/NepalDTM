"""Phase 3 (G): download high-resolution lidar DTMs to train the 1 m generator.

swissALTI3D 0.5 m (swisstopo, open data): 1 km tiles over the high Alps, newest year per tile.
USGS 3DEP 1 m (open): 10 km tiles in four Nepal-like settings (steep rock, steep forest, forested mid-hills,
glaciers). Both are bare-earth DTMs, the target G has to produce.

Resumable (skips files already complete), parallel curl, stops at the size budget.
Out: data/gen/raw/{swiss,3dep}/*.tif and data/gen/raw/manifest.json.

Run: ../.venv/bin/python -I -u g/fetch_training.py
"""
from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "gen" / "raw"
CA = os.path.expanduser("~/.certs/macos_bundle.pem")
SWISS_BBOX = (7.0, 45.85, 8.6, 46.6)  # Valais / Bernese Alps: 400-4600 m, glaciers, cliffs
SWISS_MAX = 5000
USGS = {  # name: (bbox, max tiles)
    "san_juans_co": ((-108.0, 37.6, -107.3, 38.1), 12),
    "cascades_olympics_wa": ((-123.9, 47.5, -120.8, 48.9), 14),
    "smokies_nc_tn": ((-84.0, 35.4, -83.0, 35.8), 10),
    "alaska_range": ((-150.5, 63.0, -146.0, 63.8), 10),
}
BUDGET_GB = 85


def get_json(url):
    r = subprocess.run(["curl", "-sfL", "--retry", "5", "--retry-all-errors", "--connect-timeout", "30", "--max-time", "120", "--cacert", CA, url], capture_output=True, check=True)
    return json.loads(r.stdout)


def swiss_items():
    url = (f"https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"
           f"?bbox={','.join(map(str, SWISS_BBOX))}&limit=100")
    best = {}
    while url:
        d = get_json(url)
        for f in d["features"]:
            _, year, tile = f["id"].split("_")
            a = next((a for k, a in f["assets"].items() if k.endswith(".tif") and "_0.5_" in k), None)
            if a and (tile not in best or year > best[tile][0]):
                best[tile] = (year, a["href"])
        url = next((l["href"] for l in d.get("links", []) if l["rel"] == "next"), None)
    tiles = sorted(best)
    step = max(1, len(tiles) // SWISS_MAX)
    return [(best[t][1], OUT / "swiss" / Path(best[t][1]).name, None) for t in tiles[::step][:SWISS_MAX]]


def usgs_items():
    out = []
    for name, (bb, n) in USGS.items():
        url = ("https://tnmaccess.nationalmap.gov/api/v1/products?datasets="
               + quote("Digital Elevation Model (DEM) 1 meter") + f"&bbox={','.join(map(str, bb))}&max=500")
        items = get_json(url)["items"]
        newest = {}
        for i in items:  # one product per 10 km cell, the newest project
            cell = i["title"].split()[4] if len(i["title"].split()) > 4 else i["title"]
            if cell not in newest or i["publicationDate"] > newest[cell]["publicationDate"]:
                newest[cell] = i
        cells = sorted(newest)
        step = max(1, len(cells) // n)
        for c in cells[::step][:n]:
            i = newest[c]
            out.append((i["downloadURL"], OUT / "3dep" / f"{name}__{Path(i['downloadURL']).name}", i.get("sizeInBytes") or None))
    return out


def fetch(job):
    url, dst, size = job
    if dst.exists() and (size is None or dst.stat().st_size == size):
        return dst, "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    r = subprocess.run(["curl", "-sfL", "--retry", "5", "--retry-all-errors", "--connect-timeout", "30", "--max-time", "3600", "--cacert", CA, "-o", str(tmp), url])
    if r.returncode:
        tmp.unlink(missing_ok=True)
        return dst, f"FAIL {r.returncode}"
    tmp.rename(dst)
    return dst, "ok"


def used_gb():
    return sum(p.stat().st_size for p in OUT.rglob("*.tif")) / 1e9


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = usgs_items()
    print(f"3DEP: {len(jobs)} tiles, {sum(j[2] or 0 for j in jobs) / 1e9:.1f} GB", flush=True)
    sw = []
    for name, workers in (("3dep", 4), ("swiss", 8)):
        if name == "swiss":  # long paged listing: done after 3DEP so downloads start at once
            sw = swiss_items()
            print(f"swissALTI3D: {len(sw)} tiles", flush=True)
        batch = jobs if name == "3dep" else sw
        (OUT / "manifest.json").write_text(json.dumps([{"url": u, "file": str(d.relative_to(OUT))} for u, d, _ in jobs + sw], indent=1))
        done = 0
        with ThreadPoolExecutor(workers) as ex:
            for dst, status in ex.map(fetch, batch):
                done += 1
                if status.startswith("FAIL") or done % 50 == 0 or name == "3dep":
                    print(f"[{name} {done}/{len(batch)}] {dst.name} {status}  ({used_gb():.1f} GB)", flush=True)
                if used_gb() > BUDGET_GB:
                    print("budget reached, stopping", flush=True)
                    return
    print(f"done, {used_gb():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
