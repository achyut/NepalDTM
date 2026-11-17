#!/bin/zsh
# NepalDTM-G: generated 1 m terrain from NepalDTM v2 (8 m). Run after the v2 build (run_v2.sh), from code/:
#
#   ./run_g.sh                         # data, error model, training (long: ~1.2 s/step on one Apple M4 Max GPU)
#   ./run_g.sh apply NAME W S E N      # generate 1 m over a lon/lat box with the latest checkpoint
#
# 1. g/fetch_training.py   swissALTI3D 0.5 m and USGS 3DEP 1 m lidar tiles (about 80 GB)
# 2. g/make_patches.py     256 x 256 m patches at 1 m (75,521), then g/g_model.py prep packs them for training
# 3. g/error_model.py      error model of the v2 DEM fitted at Langtang (needs the Langtang 2 m composite in data/nw/)
# 4. g/g_model.py train    conditional diffusion model; checkpoints in data/checkpoints/g_model/
#    g/g_model.py eval     held-out lidar patches against bicubic
# 5. g/g_apply.py          1 m COG over a box, scored against held-out ICESat-2 photons
#    g/g_tiles.py          zoom 15-17 Terrarium tiles of the generated boxes for the web map
set -e
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}
export PROJ_NETWORK=ON
if [[ $1 == apply ]]; then
    $PY -I -u g/g_apply.py --name $2 --bbox $3 $4 $5 $6 --ens 4
    exit
fi
$PY -I -u g/fetch_training.py
$PY -I -u g/make_patches.py
$PY -I -u g/g_model.py prep
$PY -I -u g/error_model.py
$PY -I -u g/error_model.py --check
$PY -I -u g/g_model.py train --steps 150000
$PY -I -u g/g_model.py eval --n 256 --ens 4
