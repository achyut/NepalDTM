# NepalDTM

**Achyut Paudel** (achyut.pdl@gmail.com)

NepalDTM is a bare-earth elevation model of Nepal on an 8 m grid. The High Mountain Asia 8 m stereo mosaic is the surface where it exists. Copernicus GLO-30 fills the gaps. A model trained on 6,230,548 ICESat-2 points and 7,035,904 GEDI points predicts the height of vegetation and buildings and subtracts it.

On 4,463,463 held-out ICESat-2 points inside Nepal, with whole 48 km blocks left out of training, the RMSE is 6.10 m. Copernicus GLO-30 on the same points is 9.07 m. Inside the delivered grid, 37% of the area uses the 8 m mosaic and 63% is Copernicus fill, so the file spacing is not 8 m terrain detail everywhere.

The paper is [`paper/paper.tex`](paper/paper.tex), with the figures in [`paper/figures`](paper/figures). The longer guide is [`paper/whitepaper.tex`](paper/whitepaper.tex). Both quote [`validation/validation_report.json`](validation/validation_report.json).

## Results

Spatial holdout, 4,463,463 ICESat-2 points inside Nepal:

| | NepalDTM | Copernicus GLO-30 |
|---|---:|---:|
| RMSE | 6.10 m | 9.07 m |
| Bias | −0.15 m | +3.87 m |

Below 300 m the spatial-holdout RMSE is 1.96 m, against 4.53 m for Copernicus. From 2.5 to 4 km it is 8.27 m, against 12.88 m. Those elevation rows are on the 6,227,829-point common set, which includes the tile buffer. Khumbu differential GPS agrees with the surface tile to 2.49 m RMSE at glacier margins (83 points). The bare-earth tile on the glacier surface is a different comparison: RMSE 6.01 m, bias −4.85 m.

## View it

The website and map viewer are at https://nepaldtm.achyutpdl.com. The map shows only Nepal, in 2D or 3D, as hillshade (optionally coloured by height), Swiss relief, color relief, slope, aspect, or roughness. Swiss relief follows the hand-drawn tradition of Eduard Imhof and the ideas behind [Eduard](https://dilpreet.co/projects/eduard): it shades three widths of landform and leans on the broadest, brightens crests, and lets low ground fade into haze, so ridges and valleys read as whole shapes instead of a field of 8 m bumps. Slope, aspect, and roughness are computed in the browser from the map tiles, in background workers, so they depend on the zoom; zoom 14 (8.4 m pixels) is the finest. Swiss relief and the cast shadows run as WebGL2 shaders (`site/js/gpu.js`): the tiles go straight to the graphics card and the shader reads the heights out of them, which measured 2.4 ms against 28.1 ms for Swiss relief and 4.6 ms against 64.4 ms for a shadow tile on integrated graphics. `site/js/dem.js` holds the same maths on the CPU, used where WebGL2 is missing and for the exact values under the cursor. A Places button lists 27 places across the seven provinces, each with a camera angle, and a View button opens the 2D/3D switch, height exaggeration, and rotate and tilt, each slider with its own buttons; a Light button opens the lighting. The light can come from a fixed direction (north-west by convention, or any of the four) or from the real sun for a chosen year, date and Nepal time, with presets for 6am, 9am, noon and 3pm. The date and time sliders each have a play button, so a day runs in about six seconds and a year in thirty. With the real sun, the shading follows at once and the shadows cast by ridges fill in when the slider is let go. Show peaks marks summits found in NepalDTM between a minimum and a maximum height. Clicking a point reads the bare-earth, surface, slope, aspect, source, and expected-error values from the 8 m tiles, and the profile tool draws an elevation profile between two points.

The summits file, `site/data/summits.geojson`, is built by `scripts/summits.py` from the 8 m bare-earth mosaic: every summit that rises at least 100 m above the col joining it to higher ground (its topographic prominence), with a prominence class, the source of its height (HMA 8 m or Copernicus fill), and, where an OpenStreetMap peak lies within 300 m, its name and recorded height (© OpenStreetMap contributors, ODbL). The map labels peaks with the recorded height where there is one.

Summits read low, because a sharp top is narrower than the patch a stereo or radar height represents. Against 406 named OpenStreetMap peaks with recorded heights, NepalDTM's summit is a median 21.2 m lower: 16.3 m on the HMA mosaic and 32.8 m in Copernicus fill. `scripts/summits.py` writes this check to [`validation/summit_check.json`](validation/summit_check.json).

The site is static HTML and JavaScript in [`site/`](site), served by a Cloudflare Worker. `scripts/build.mjs` copies it to `dist/` and fills every number on the home page from [`validation/validation_report.json`](validation/validation_report.json). To run it locally or deploy it, with [Wrangler](https://developers.cloudflare.com/workers/wrangler/) installed:

```bash
wrangler dev       # builds, then serves http://localhost:8787
wrangler deploy    # builds and publishes to nepaldtm.achyutpdl.com
```

The tiles are in the `nepal-dem-8m` R2 bucket, served at `nepaldtm-data.achyutpdl.com`. Its CORS rule is [`infra/r2-cors.json`](infra/r2-cors.json): `wrangler r2 bucket cors set nepal-dem-8m --file infra/r2-cors.json`.

## Data

| Product | URL |
|---|---|
| Map (PMTiles, zoom 5–14, lossless WebP, heights to 0.1 m, 3.0 GB) | https://nepaldtm-data.achyutpdl.com/NepalDTM.webp.pmtiles |
| Cloud Optimized GeoTIFF tiles | `https://nepaldtm-data.achyutpdl.com/cog/{tile}_{dtm,dem,aux}.tif`, e.g. [`E0360N3072_dtm.tif`](https://nepaldtm-data.achyutpdl.com/cog/E0360N3072_dtm.tif) |

Heights are metres above the EGM2008 geoid, on UTM zone 45N. Each tile has a surface file, a bare-earth file, and an error file. The code writes VRT mosaics that point at a local `out/` folder.

## Reproduce

```bash
git clone https://github.com/achyut/NepalDTM.git
cd NepalDTM
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r code/requirements-nw.txt
cd code
../.venv/bin/python -I run.py all
```

| Command | What it runs |
|---|---|
| `python run.py download` | Geoid grid, Earthdata token check, tile grid, ICESat-2 and GEDI, then the 8 m stacks |
| `python run.py fuse` | Train the bare-earth correction and write the tiles |
| `python run.py validate` | Held-out scores, differential GPS, contours, and seams |
| `python run.py all` | Every step |
| `python nw_tiles_rgb.py` | Map tiles: `data/nw/NepalDTM.webp.pmtiles` from the bare-earth mosaic |

A national run downloads tens of gigabytes. Cells and tiles already on disk are skipped. Put a NASA Earthdata token in `~/.earthdata_token`. That file is not part of this repository.

## Citation

Please cite the paper. The arXiv identifier below is a placeholder until the paper is announced. Replace `XXXX.XXXXX` with the identifier arXiv assigns.

Paudel, A. (2026). NepalDTM: A national 8-metre bare-earth digital terrain model of Nepal from multi-source satellite data fusion. https://github.com/achyut/NepalDTM

```bibtex
@article{paudel2026nepaldtm,
  author        = {Paudel, Achyut},
  title         = {{NepalDTM}: A National 8-Metre Bare-Earth Digital Terrain Model of {Nepal} from Multi-Source Satellite Data Fusion},
  year          = {2026},
  journal       = {arXiv preprint arXiv:XXXX.XXXXX},
  archiveprefix = {arXiv},
  eprint        = {XXXX.XXXXX},
  primaryclass  = {physics.geo-ph},
  url           = {https://github.com/achyut/NepalDTM}
}
```

## Data credits

- Copernicus DEM: European Space Agency, doi:10.5270/ESA-c5d3d65
- ICESat-2 ATL08: Neuenschwander and Pitts (2019); the mission is Markus et al. (2017)
- GEDI L2A: Dubayah et al. (2020)
- HMA DEM 8 m: Shean (2017), NASA NSIDC, doi:10.5067/KXOVQ9L172S2
- Canopy height: Tolan et al. (2024)
- ESA WorldCover 2021 v200: Zanaga et al. (2022), doi:10.5281/zenodo.7254221
- EGM2008: Pavlis et al. (2012)
- Khumbu dGPS: Quincey, King, and Watson (2016), BGS NGDC accession 75529
- FABDEM V1-2: Hawker et al. (2022)
- GEDTM30: Ho et al. (2025)

## License

Code is MIT. See `LICENSE`. The NepalDTM tiles are CC BY 4.0. Input datasets keep their own licences.

## Survey Department data

`code/data/control/survey/` holds Survey Department layers used as independent references (not inputs): the
41 named peak heights and the 50 pass heights. The national contour lines
(`geonode_eleva_ln.geojson`, 35,054 lines, 156 MB) are too large for this repository and are not in it. They
come from the National Geoportal's WFS service as `geonode_eleva_ln`; the copy that service returns is
truncated mid-file, so a repaired copy (all but the last line recovered) is kept in the private
`nepalgeoportal` R2 bucket at `wfs-downloads/geonode_eleva_ln.repaired.geojson`. That bucket is not public:
the Survey Department layers are licensed data, so they are referenced here, not redistributed.
Three checks read these layers:

| Script | Writes | What it measures |
|---|---|---|
| `scripts/contour_check.py` | `validation/contour_check.json` | NepalDTM and Copernicus against 776,216 national contour points |
| `scripts/contour_vs_icesat2.py` | `validation/contour_vs_icesat2.json` | Which of the contours and NepalDTM is closer to the ICESat-2 laser ground |
| `scripts/strip_coverage.py` | `validation/strip_coverage.json` | How much of the Copernicus-filled gap the free HMA 8 m strips could close |

The contours are a check, not a correction. Judged against 60,183 ICESat-2 ground returns within 12 m of a
contour vertex, the contours carry a +3.2 m bias and a 13.0 m spread (NMAD) against NepalDTM's −0.4 m and
5.3 m; on gentle ground it is 10.2 m against 4.0 m. They are 1:50,000 map products at a 20 m interval, and a
few hundred lines are mislabelled, so correcting the tiles towards them would make the tiles worse. On the
contours themselves NepalDTM still beats Copernicus nationwide (median −2.7 m, NMAD 14.6 m against +4.4 m and
17.4 m), which is an independent confirmation of the model.

Heights in ICESat-2 ATL08 are above the WGS84 ellipsoid; the tiles and the contours are above the EGM2008
geoid, which in Nepal sits 39 m (east) to 59 m (west) higher. Any comparison has to convert first.
