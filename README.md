# NepalDTM

**Achyut Paudel** (achyut.pdl@gmail.com)

NepalDTM is a national surface (DEM) and bare-earth (DTM) elevation model of Nepal on an 8 m grid. This branch holds **v2** (November 2026), the current release. v1 (October 2026) is kept below for reference; its code and paper are on `main`.

Map with a v1/v2 switch: https://nepaldtm.achyutpdl.com/map

## v2 (November 2026)

v2 adjusts all 512 HMA stereo strips jointly to ICESat-2 ATL03 ground photons (a plane per strip, robust least
squares), rebuilds the stack from the adjusted strips, and keeps strips that agree inside voids instead of falling
back to Copernicus there. The DSM to DTM model is retrained on the new surface.

On 19.8 million held-out ATL03 photons (tracks never used in the adjustment or training), against the same points:

| | v2 | v1 | Copernicus GLO-30 | FABDEM | GEDTM30 |
|---|---:|---:|---:|---:|---:|
| DTM RMSE | **5.26 m** | 6.93 m | 8.69 m | 8.40 m | 8.97 m |

Over every 8 m cell of the Langtang 2 m stereo composite the DEM error falls from 8.62 m (v1) to 4.71 m
(Copernicus 13.96 m). These photon RMSEs are on a different point set from the v1 table above, so compare columns
within this table only.

- Paper: EarthArXiv, https://doi.org/10.31223/X5B23B (PDF [`paper/v2/paper_v2.pdf`](paper/v2/paper_v2.pdf), source [`paper/v2/paper_v2.tex`](paper/v2/paper_v2.tex))
- Validation: [`validation/validation_report_v2.json`](validation/validation_report_v2.json),
  [`validation/validation_v1_v2.json`](validation/validation_v1_v2.json)
- Build: [`code/run_v2.sh`](code/run_v2.sh) runs the whole v2 chain after the v1 inputs are in `code/data/nw/`
- Data: Zenodo, [doi:10.5281/zenodo.24483836](https://doi.org/10.5281/zenodo.24483836) (v2.1, DEM, DTM and aux COG tiles, and
  `NepalDTM_v2.1.webp.pmtiles` for web maps); map with a version switch at https://nepaldtm.achyutpdl.com/map

## v1 results (October 2026)

v1 used the High Mountain Asia 8 m stereo mosaic where it exists and Copernicus GLO-30 elsewhere (37% / 63% of the area), with a model trained on 6,230,548 ICESat-2 points and 7,035,904 GEDI points removing vegetation and buildings. The v1 paper is on EarthArXiv, https://doi.org/10.31223/X5DR5M (source [`paper/paper.tex`](paper/paper.tex)) and the guide [`paper/whitepaper.tex`](paper/whitepaper.tex); both quote [`validation/validation_report.json`](validation/validation_report.json).

Spatial holdout, 4,463,463 ICESat-2 points inside Nepal:

| | NepalDTM | Copernicus GLO-30 |
|---|---:|---:|
| RMSE | 6.10 m | 9.07 m |
| Bias | −0.15 m | +3.87 m |

Below 300 m the spatial-holdout RMSE is 1.96 m, against 4.53 m for Copernicus. From 2.5 to 4 km it is 8.27 m, against 12.88 m. Those elevation rows are on the 6,227,829-point common set, which includes the tile buffer. Khumbu differential GPS agrees with the surface tile to 2.49 m RMSE at glacier margins (83 points). The bare-earth tile on the glacier surface is a different comparison: RMSE 6.01 m, bias −4.85 m.

## View it

```bash
python3 serve.py
```

Then open http://127.0.0.1:8000/

## Data

| Product | Archive (Zenodo, DOI) |
|---|---|
| v2.1 data (recommended): DTM, DEM and aux COG tiles (zipped by layer), WebP PMTiles, validation | [doi:10.5281/zenodo.24483836](https://doi.org/10.5281/zenodo.24483836) |
| v1.1 data: DTM, DEM and aux COG tiles (zipped by layer), WebP PMTiles, validation | [doi:10.5281/zenodo.24483831](https://doi.org/10.5281/zenodo.24483831) |
| v2.0 data (superseded: gap-fill rings) | [doi:10.5281/zenodo.24327134](https://doi.org/10.5281/zenodo.24327134) |
| v1.0 data (superseded: gap-fill rings; numbers in the v1 paper) | [doi:10.5281/zenodo.24327066](https://doi.org/10.5281/zenodo.24327066) |

Zenodo is the archived copy to download and cite. The web map and quick single-tile access stream from a mirror at
`https://nepaldtm-data.achyutpdl.com` (v1 COGs under `cog/`, v2 under `v2/out_v2/`), e.g.
`gdalinfo /vsicurl/https://nepaldtm-data.achyutpdl.com/cog/E0360N3072_dtm.tif`.

The COG prefixes are folders of files, not browsable pages. Heights are metres above the EGM2008 geoid, on UTM zone 45N. Each tile has a surface file, a bare-earth file, and an aux file. In v2 the aux file's band 1 is the source (1 = adjusted HMA stereo, 2 = Copernicus fill) and band 2 is the expected error in decimetres. The code writes VRT mosaics that point at the local output folder.

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
| `bash run_v2.sh` | v2 after the v1 inputs exist: ATL03 photons, strip ties, joint adjustment, restack, retrain, apply, validate, tiles |

A national run downloads tens of gigabytes. Cells and tiles already on disk are skipped. Put a NASA Earthdata token in `~/.earthdata_token`. That file is not part of this repository.

## Citation

If you use v2, please cite the v2 paper (EarthArXiv, https://doi.org/10.31223/X5B23B).

Paudel, A. (2026). NepalDTM v2: Improving on NepalDTM v1 with ICESat-2-adjusted stereo strips for a national 8 m terrain model of Nepal. EarthArXiv. https://doi.org/10.31223/X5B23B

```bibtex
@misc{paudel2026nepaldtmv2,
  author       = {Paudel, Achyut},
  title        = {{NepalDTM} v2: Improving on {NepalDTM} v1 with {ICESat-2}-Adjusted Stereo Strips for a National 8~m Terrain Model of {Nepal}},
  year         = {2026},
  howpublished = {EarthArXiv preprint},
  doi          = {10.31223/X5B23B},
  url          = {https://doi.org/10.31223/X5B23B},
  note         = {Code: https://github.com/achyut/NepalDTM (branch v2)}
}
```

v1 (EarthArXiv, https://doi.org/10.31223/X5DR5M):

```bibtex
@misc{paudel2026nepaldtm,
  author       = {Paudel, Achyut},
  title        = {{NepalDTM}: A National 8-Metre Bare-Earth Digital Terrain Model of {Nepal} from Multi-Source Satellite Data Fusion},
  year         = {2026},
  howpublished = {EarthArXiv preprint, version 1},
  doi          = {10.31223/X5DR5M},
  url          = {https://doi.org/10.31223/X5DR5M}
}
```

## Data credits

- Copernicus DEM: European Space Agency, doi:10.5270/ESA-c5d3d65
- ICESat-2 ATL08: Neuenschwander and Pitts (2019); the mission is Markus et al. (2017)
- GEDI L2A: Dubayah et al. (2020)
- HMA DEM 8 m: Shean (2017), NASA NSIDC, doi:10.5067/KXOVQ9L172S2
- HMA 8 m stereo strips (v2): Shean (2017), along-track doi:10.5067/GSACB044M4PK, cross-track doi:10.5067/0MCWJJH5ABYO
- ICESat-2 ATL03 photons (v2): Neumann et al. (2019), doi:10.1016/j.rse.2019.111325, accessed with SlideRule (Shean et al. 2023, doi:10.21105/joss.04982)
- Langtang 2 m stereo composite (v2 validation): Shean and Bhushan (2026), doi:10.5281/zenodo.22842748
- Canopy height: Tolan et al. (2024)
- ESA WorldCover 2021 v200: Zanaga et al. (2022), doi:10.5281/zenodo.7254221
- EGM2008: Pavlis et al. (2012)
- Khumbu dGPS: Quincey, King, and Watson (2016), BGS NGDC accession 75529
- FABDEM V1-2: Hawker et al. (2022)
- GEDTM30: Ho et al. (2025)

## License

Code is MIT. See `LICENSE`. The NepalDTM tiles are CC BY 4.0. Input datasets keep their own licences.
