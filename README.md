# NepalDTM

**Achyut Paudel** (achyut.pdl@gmail.com)

NepalDTM is a bare-earth elevation model of Nepal on an 8 m grid. The High Mountain Asia 8 m stereo mosaic is the surface where it exists. Copernicus GLO-30 fills the gaps. A model trained on 6,230,548 ICESat-2 points and 7,035,904 GEDI points predicts the height of vegetation and buildings and subtracts it.

On 4,463,463 held-out ICESat-2 points inside Nepal, with whole 48 km blocks left out of training, the RMSE is 6.10 m. Copernicus GLO-30 on the same points is 9.07 m. Inside the delivered grid, 37% of the area uses the 8 m mosaic and 63% is Copernicus fill, so the file spacing is not 8 m terrain detail everywhere.

The paper is [`paper/paper.tex`](paper/paper.tex), with the figures in [`paper/figures`](paper/figures). The longer guide is [`paper/whitepaper.tex`](paper/whitepaper.tex). Both quote [`validation/validation_report.json`](validation/validation_report.json).

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

- Paper: [`paper/v2/paper_v2.pdf`](paper/v2/paper_v2.pdf) (source [`paper/v2/paper_v2.tex`](paper/v2/paper_v2.tex))
- Validation: [`validation/validation_report_v2.json`](validation/validation_report_v2.json),
  [`validation/validation_v1_v2.json`](validation/validation_v1_v2.json)
- Build: [`code/run_v2.sh`](code/run_v2.sh) runs the whole v2 chain after the v1 inputs are in `code/data/nw/`
- Data: `https://nepaldtm-data.achyutpdl.com/v2/` (DEM, DTM and aux COGs in `out_v2/`, VRT mosaics, and
  `NepalDTM_v2.webp.pmtiles` for web maps); map with a v1/v2 switch at https://nepaldtm.achyutpdl.com/map

## NepalDTM-G: generated 1 m (experimental)

NepalDTM-G adds 1 m terrain detail to the measured v2 model with a conditional diffusion model trained on Swiss and
US lidar, degraded with an error model fitted to v2. Every 8 x 8 m block of the 1 m output averages back to the v2
cell, so G never changes the measured model at 8 m. **Its detail is generated, not measured.**

On two test areas in Chitwan, G keeps v2's accuracy against held-out ICESat-2 photons (floodplain 1.99 m against
1.98 m RMSE on 117,703 photons; Churia hills 11.67 m against 11.63 m on 1,610 photons) while adding sub-8 m relief.

- Paper: [`paper/g/paper_g.pdf`](paper/g/paper_g.pdf) (source [`paper/g/paper_g.tex`](paper/g/paper_g.tex))
- Code: [`code/g/`](code/g), run with [`code/run_g.sh`](code/run_g.sh) after the v2 build
- Map: https://nepaldtm.achyutpdl.com/map?v=g (test area only, zooms 15 to 17)

## Results

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

| Product | URL |
|---|---|
| NepalDTM-G 1 m test overlay (Churia) | https://nepaldtm-data.achyutpdl.com/g/NepalDTM_G_test.webp.pmtiles |
| v2 web map tiles | https://nepaldtm-data.achyutpdl.com/v2/NepalDTM_v2.webp.pmtiles |
| v1 web map tiles | https://nepaldtm-data.achyutpdl.com/NepalDTM.pmtiles |
| v1 Cloud Optimized GeoTIFF tiles | prefix `https://nepaldtm-data.achyutpdl.com/cog/` |

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

A national run downloads tens of gigabytes. Cells and tiles already on disk are skipped. Put a NASA Earthdata token in `~/.earthdata_token`. That file is not part of this repository.

## Citation

NepalDTM-G is work in progress; the paper source is in `paper/g/`. Cite it, and the v2 model it downscales:

```bibtex
@misc{paudel2026nepaldtmg,
  author       = {Paudel, Achyut},
  title        = {{NepalDTM-G}: Mass-Preserving Generative Downscaling of {Nepal}'s National 8~m Terrain Model to 1~m},
  year         = {2026},
  howpublished = {Preprint in preparation},
  url          = {https://github.com/achyut/NepalDTM/tree/g}
}

@misc{paudel2026nepaldtmv2,
  author       = {Paudel, Achyut},
  title        = {{NepalDTM} v2: Photon-Controlled Joint Adjustment of Stereo Strips for a National Terrain Model of {Nepal}},
  year         = {2026},
  howpublished = {EarthArXiv preprint (DOI pending)},
  url          = {https://nepaldtm.achyutpdl.com/paper/paper_v2.pdf}
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
