# Publishing NepalDTM on arXiv

The submission is `paper/paper.tex` plus `paper/references.bib`.
`paper/whitepaper.tex` is the longer guide. Leave it out of the arXiv upload, or add it as an ancillary file. It is not the article.

arXiv compiles the TeX. You need an arXiv account. A first submission in `physics.geo-ph` often needs an endorser. This file cannot submit for you.

## 1. Make the upload

From `paper/`:

```bash
tectonic paper.tex
```

or, if you have TeX Live:

```bash
pdflatex paper
bibtex paper
pdflatex paper
pdflatex paper
```

Zip only these two files:

- `paper.tex`
- `references.bib`

Do not put the Earthdata token, the R2 secret, or `code/data/nw/` in the zip.

## 2. Start the submission

1. Sign in at https://arxiv.org/user/login
2. Choose **START NEW SUBMISSION**
3. Licence: grant arXiv the standard non-exclusive licence to distribute, and select **CC BY 4.0** so the paper matches the tiles
4. Upload the zip
5. Set the main TeX file to `paper.tex` if arXiv asks
6. Check the compiled PDF, then submit the metadata below

## 3. Values to paste

Fields marked FILL are yours. Everything else is ready to copy.

| Field | Value |
|---|---|
| Title | NepalDTM: A National 8-Metre Bare-Earth Digital Terrain Model of Nepal from Multi-Source Satellite Data Fusion |
| Authors | Paudel, Achyut |
| Author email | achyut.pdl@gmail.com |
| Affiliation | FILL: your institution, or leave blank if you are submitting independently |
| Abstract | Paste the abstract from the next section |
| Comments | 324 tiles. Code: https://github.com/achyut/NepalDTM |
| MSC class | FILL: optional. 86A30 is a possible geophysics class if you want one |
| ACM class | leave blank |
| Report number | leave blank |
| Journal reference | leave blank until a journal version exists |
| DOI | leave blank. arXiv assigns one after the announcement |
| Primary category | physics.geo-ph |
| Cross-list | eess.IV |
| License | arXiv non-exclusive distribution licence, plus CC BY 4.0 |

### Abstract

NepalDTM is an open bare-earth elevation model of Nepal on an 8 m grid. Where the NASA High Mountain Asia 8 m stereo mosaic exists it is the surface; elsewhere the surface is Copernicus GLO-30. A histogram gradient-boosting model, trained on 6,230,548 ICESat-2 and 7,035,904 GEDI ground points, predicts how far that surface sits above the ground. On 4,463,463 held-out ICESat-2 points inside Nepal, leaving out whole 48 km blocks, the root-mean-square error is 6.10 m, against 9.07 m for Copernicus GLO-30, 8.47 m for GEDTM30, and 7.78 m for FABDEM. The bias is -0.15 m. Inside the delivered grid, 37% of the area uses the 8 m mosaic and 63% is filled from Copernicus, so the posted spacing is not 8 m terrain detail everywhere. Tiles, the validation files, and the code that rebuilds the model are public.

## 4. After arXiv announces it

1. Copy the identifier, which looks like `2601.01234`
2. Replace every `XXXX.XXXXX` in `README.md` and `CITATION.cff`
3. The public tile URL is already in the paper and does not change

## 5. What not to upload

- `~/.earthdata_token`
- any R2 access key or secret
- `code/data/nw/` and other downloaded rasters
- `.venv/`
