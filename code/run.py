"""Run the NepalDTM build from download through validation.

The science stays in the step scripts. This file only checks the two things
those scripts assume, then runs them in order and stops if one fails.

  prepare    EGM2008 geoid grid, and a NASA Earthdata token for the HMA surface
  tiles      24 km grid (nw_tiles.py)
  altimetry  ICESat-2 ATL08 and GEDI L2A (nw_altimetry.py)
  stack      HMA, Copernicus, canopy, and land cover on the 8 m grid (nw_stack.py)
  train      bare-earth correction (nw_train.py)
  apply      national surface, bare earth, and error tiles (nw_apply.py)
  validate   held-out scores (nw_validate.py)

Groups:

  python run.py download     prepare, tiles, altimetry, stack
  python run.py fuse         train, apply
  python run.py validate     validate
  python run.py all          every step, in order

A national run downloads tens of gigabytes. Cells and tiles that already
exist are skipped by the step scripts, so stopping and starting again is safe.

  python run.py stack --part 0/2
  python run.py apply --part 1/2
  python run.py train --reuse --max-train 3000000
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
EGM_URL = "https://cdn.proj.org/us_nga_egm08_25.tif"
EGM_NAME = "us_nga_egm08_25.tif"

# name -> (what it does, script args). None means the check in prepare().
STEPS = (
    ("prepare", "EGM2008 geoid grid and Earthdata token", None),
    ("tiles", "24 km tile grid", ["nw_tiles.py"]),
    ("altimetry", "ICESat-2 ATL08 and GEDI L2A ground points", ["nw_altimetry.py"]),
    ("stack", "HMA, Copernicus, canopy, and land cover stacked at 8 m", ["nw_stack.py", "--all"]),
    ("train", "bare-earth correction model", ["nw_train.py"]),
    ("apply", "surface, bare-earth, and error tiles", ["nw_apply.py"]),
    ("validate", "held-out ICESat-2, dGPS, contours, seams", ["nw_validate.py"]),
)
ORDER = [name for name, _, _ in STEPS]
GROUPS = {
    "download": ["prepare", "tiles", "altimetry", "stack"],
    "fuse": ["train", "apply"],
    "validate": ["validate"],
    "all": ORDER,
}


def egm_ok() -> bool:
    from pyproj import Transformer

    _, _, z = Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True).transform(85.2, 28.0, 0.0)
    return math.isfinite(z) and abs(z) > 1


def prepare() -> None:
    boundary = ROOT / "data" / "control" / "nepal_boundary.geojson"
    if not boundary.is_file():
        raise SystemExit(f"missing {boundary}")
    token = Path.home() / ".earthdata_token"
    if not token.is_file() or not token.read_text().strip():
        raise SystemExit(
            "missing ~/.earthdata_token\n"
            "Create a token at https://urs.earthdata.nasa.gov/ and save that one line in the file. "
            "The HMA 8 m surface will not download without it. Do not commit the file."
        )
    if egm_ok():
        print("EGM2008 grid is already usable")
        return
    from pyproj import datadir

    dest = Path(datadir.get_user_data_dir(True)) / EGM_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {EGM_NAME} -> {dest}")
    urllib.request.urlretrieve(EGM_URL, dest)
    if not egm_ok():
        raise SystemExit("EGM2008 grid was downloaded but a test height at 85.2E, 28.0N is still not finite")
    print("EGM2008 grid is usable")


def command(name: str, args: argparse.Namespace) -> list[str]:
    script = dict((n, spec) for n, _, spec in STEPS)[name]
    cmd = [PY, "-I", *script]
    if name == "train":
        cmd += ["--max-train", str(args.max_train)]
        if args.reuse:
            cmd.append("--reuse")
    if name in ("stack", "apply") and args.part != "0/1":
        cmd += ["--part", args.part]
    return cmd


def run_step(name: str, args: argparse.Namespace) -> None:
    title = dict((n, text) for n, text, _ in STEPS)[name]
    print(f"\n== {name}: {title}", flush=True)
    if name == "prepare":
        prepare()
        return
    cmd = command(name, args)
    print(" ", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def selected(args: argparse.Namespace) -> list[str]:
    names = GROUPS.get(args.step, [args.step])
    if args.only:
        unknown = [n for n in args.only if n not in ORDER]
        if unknown:
            raise SystemExit(f"unknown step: {', '.join(unknown)}")
        names = [n for n in names if n in args.only]
    if args.start:
        if args.start not in names:
            raise SystemExit(f"{args.start} is not in this run ({', '.join(names)})")
        names = names[names.index(args.start):]
    if not names:
        raise SystemExit("no steps selected")
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", choices=[*ORDER, *GROUPS])
    ap.add_argument("--start", choices=ORDER, help="start at this step and continue to the end of the group")
    ap.add_argument("--only", nargs="+", choices=ORDER, help="run only these steps, still in pipeline order")
    ap.add_argument("--part", default="0/1", help="k/n shard for stack and apply, for example 0/2")
    ap.add_argument("--max-train", type=int, default=3_000_000, help="training points per fit")
    ap.add_argument("--reuse", action="store_true", help="train from an existing data/nw/points.npz")
    args = ap.parse_args()
    if not args.step:
        ap.print_help()
        return
    for name in selected(args):
        run_step(name, args)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
