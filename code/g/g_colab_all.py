#!/usr/bin/env python3
"""Colab driver: generate every remaining province, then the national map tiles. One command, resumable.

For each province not yet marked done in R2 (<province>/DONE): download its 8 m input, run g_region (each tile is
uploaded to <province>/tiles/ as it finishes and tiles already there are skipped), mark it done and free the disk.
Then build the national map archives (g_tiles_nepal.py). After any stop, run the same command again.

Needs R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET.
Run (from code/): python -u g/g_colab_all.py [--bs 32] [--provinces Koshi Madhesh ...]
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import boto3

DATA = "https://nepaldtm-data.achyutpdl.com/g/colab"
ALL = ["Koshi", "Madhesh", "Gandaki", "Lumbini", "Karnali", "Sudurpashchim"]  # Bagmati is done
HERE = Path(__file__).resolve().parent


def fetch(url, path):
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 16 << 20)
    tmp.rename(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--provinces", nargs="*", default=ALL)
    ap.add_argument("--no-map", action="store_true", help="skip the national map tiles")
    a = ap.parse_args()
    bucket = os.environ["R2_BUCKET"]
    c = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"], region_name="auto",
                     aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                     aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"])
    done = lambda p: "Contents" in c.list_objects_v2(Bucket=bucket, Prefix=f"{p}/DONE")
    inp = Path("/content/in"); inp.mkdir(exist_ok=True)
    fetch(f"{DATA}/ema_150000.pt", inp / "ema_150000.pt")
    for prov in a.provinces:
        p = prov.lower()
        if done(p):
            print(f"== {prov}: done already", flush=True); continue
        print(f"== {prov}: downloading input", flush=True)
        d = inp / p; d.mkdir(exist_ok=True)
        fetch(f"{DATA}/{p}/{p}_v2_dtm_8m.tif", d / f"{p}_v2_dtm_8m.tif")
        fetch(f"{DATA}/{p}/{p}.geojson", d / f"{p}.geojson")
        out = Path(f"/content/out/{p}")
        r = subprocess.run([sys.executable, "-u", str(HERE / "g_region.py"), "--region", prov,
                            "--dtm", str(d / f"{p}_v2_dtm_8m.tif"), "--prov", str(d / f"{p}.geojson"),
                            "--ckpt", str(inp / "ema_150000.pt"), "--steps", "10", "--ens", "2", "--bs", str(a.bs),
                            "--out", str(out), "--r2-prefix", f"{p}/tiles", "--delete-uploaded"], cwd=HERE.parent)
        if r.returncode != 0:
            sys.exit(f"{prov}: g_region failed ({r.returncode}); run the command again to resume")
        c.put_object(Bucket=bucket, Key=f"{p}/DONE", Body=b"done\n")
        shutil.rmtree(out, ignore_errors=True); shutil.rmtree(d, ignore_errors=True)
        print(f"== {prov}: done and marked in R2", flush=True)
    if not a.no_map:
        subprocess.run([sys.executable, "-u", str(HERE / "g_tiles_nepal.py")], cwd=HERE.parent, check=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
