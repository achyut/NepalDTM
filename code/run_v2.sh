#!/bin/zsh
# NepalDTM v2: the full national build, after the v1 inputs (HMA strips, Copernicus, ICESat-2 ATL08/GEDI) are in
# data/nw/ (see README). Each step skips work already on disk, so the script can be re-run after an interruption.
#
#   cd code && ./run_v2.sh
#
# 1. fetch_atl03.py        ICESat-2 ATL03 ground photons over Nepal (SlideRule), with held-out tracks marked
# 2. nw_strip_adjust.py    photon ties per strip, joint adjustment of all HMA strips to the photons (a plane per strip, Huber IRLS + LSQR),
#                          then the adjusted strips per 24 km tile
# 3. nw_stack_v2.py        per-tile stack from the adjusted strips (blunder filter keeps strips that agree in voids)
# 4. nw_train.py           DSM -> DTM residual model, NW_VERSION=v2
# 5. nw_apply.py           national DEM, DTM and aux COGs + VRT mosaics (suffix _v2)
# 6. nw_validate.py        held-out ICESat-2, Langtang 2 m and Khumbu dGPS report (validation_report_v2.json)
#    validate_v1_v2.py     v1, v2 and public DEMs (Copernicus, FABDEM, GEDTM30) on the same held-out photons
# 7. nw_tiles_rgb.py       lossless WebP Terrarium PMTiles for the web map (NepalDTM_v2.webp.pmtiles)
set -e
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}
N=${PARTS:-3}
export PROJ_NETWORK=ON NW_VERSION=v2

$PY -I -u fetch_atl03.py
$PY -I -u nw_strip_adjust.py ties
$PY -I -u nw_strip_adjust.py solve
$PY -I -u nw_strip_adjust.py apply
for k in $(seq 0 $((N - 1))); do $PY -I -u nw_stack_v2.py --part $k/$N & done; wait
$PY -I -u nw_train.py
for k in $(seq 0 $((N - 1))); do $PY -I -u nw_apply.py --part $k/$N & done; wait
$PY -I -u nw_apply.py --vrt-only
$PY -I -u nw_validate.py
$PY -I -u validate_v1_v2.py
$PY -I -u nw_tiles_rgb.py
