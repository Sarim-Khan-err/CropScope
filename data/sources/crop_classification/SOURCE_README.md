# Field-level Sentinel-2 time series for crop-type classification in Pakistan

Processed satellite image time series and crop labels for 85,618 surveyed
agricultural fields across 79 districts of Pakistan, covering two cropping
seasons: Rabi 2022 (October to April) and Kharif 2023 (May to November).

This is the exact data behind the paper "Conformal Crop Classification Across
Regions with Sentinel-2 Time Series". Every number in that paper can be
regenerated from these arrays plus the code repository linked below.

## Files

| File | Shape | Size | Description |
|---|---|---|---|
| `X_rabi.npy` | (26428, 7, 28) float32 | 20 MB | Monthly feature tensor, Rabi |
| `X_kharif.npy` | (59190, 7, 28) float32 | 46 MB | Monthly feature tensor, Kharif |
| `y_rabi.npy` | (26428,) int64 | 0.2 MB | Crop label index, Rabi |
| `y_kharif.npy` | (59190,) int64 | 0.5 MB | Crop label index, Kharif |
| `meta_rabi.csv` | 26428 rows | 1.5 MB | Field ID, crop, season, province, district |
| `meta_kharif.csv` | 59190 rows | 3.4 MB | Same, Kharif |
| `label_map.csv` | 10 rows | 1 KB | Crop name to label index |

## The feature tensor

Each field is one row of `X`, shaped `(T=7 months, C=28 channels)`.

The 7 time steps are monthly median composites over the growing season. Monthly
rather than single-date, because monsoon cloud during Kharif makes individual
acquisitions unreliable. Note that `t=0` means October for Rabi and May for
Kharif, so the two seasons are not directly comparable along the time axis and
should be modelled separately.

The 28 channels are 14 variables, each stored as its within-field mean and its
within-field standard deviation:

- 10 Sentinel-2 surface reflectance bands (harmonised L2A, Scene Classification
  Layer cloud masking)
- 4 vegetation and water indices: NDVI, EVI, NDWI, SAVI

Each index is computed per pixel and only then averaged over the field, so the
index channels are not recoverable from the band means. An ablation in the paper
found them accuracy-neutral, but they are kept for completeness.

## Labels

Ground truth is the Asian Development Bank crop-type survey of Pakistan. Labels
are field-level, one crop per delineated field per season.

Both seasons are steeply imbalanced, which is a real property of Pakistani
agriculture and not a sampling artefact. Wheat is 79.2% of Rabi and Rice is
54.9% of Kharif. Macro-F1 is the appropriate metric, not accuracy.

**Rabi, 5 crops:** Wheat 20,932 (79.2%), Rapeseed Mustard 2,218 (8.4%),
Barseem 1,281 (4.8%), Maize 1,134 (4.3%), Sugarcane 745 (2.8%).

**Kharif, 7 crops:** Rice 32,477 (54.9%), Cotton 12,179 (20.6%), Sugarcane
7,487 (12.6%), Maize 4,178 (7.1%), Sorghum 1,189 (2.0%), Jantar 887 (1.5%),
Sesame 793 (1.3%).

These arrays are the unfiltered survey. The paper drops classes below 1% of a
season as off-season rarities, which in Rabi means 118 further fields spread
over Cotton (101), Sorghum (12), Jantar (4), and Rice (1). Those rows are still
present in `X_rabi.npy` and `meta_rabi.csv`, so filter them yourself if you want
to match the paper exactly. Kharif needs no filtering. `label_map.csv` holds the
union of both seasons' crops, so some indices are unused within a given season.

Districts are 63 in Rabi and 72 in Kharif, 56 of them shared, for a union of 79.

## The District column matters

`meta_*.csv` carries a `District` and a `Province` for every field. These are
not decoration. Geospatial samples are spatially autocorrelated, so a random
train/test split puts neighbouring fields on both sides and inflates accuracy.
Any evaluation on this dataset should group by `District`, holding whole
districts out rather than random fields. The paper reports both, and the gap
between them is roughly 0.03 macro-F1 on Rabi and 0.07 on Kharif.

## Provenance and processing

Started as 135,408 raw (field, season, crop) boundary polygons. Fields missing
more than half their monthly composites were dropped and shorter gaps
interpolated, leaving 85,618 fields, about 63% of the raw polygons. That drop is
close to spatially uniform, with per-district retention of median 0.64 for Rabi
and 0.65 for Kharif and a floor of 0.43, so it does not preferentially thin any
region.

Normalisation is deliberately **not** applied here. The arrays hold raw
reflectance and index values, roughly -150 to 8600, and a few negative
reflectances survive atmospheric correction. Fit your scaler on your training
fold only, as fitting it on everything leaks test statistics. There are no NaNs
left: gaps were interpolated during preprocessing.

## Privacy

The metadata contains district and province names only. There are no field
geometries, no coordinates, and no personal or farm-holder identifiers. The `ID`
column is the survey's internal field key.

The field boundary shapefiles are not included in this deposit. They remain the
property of the Asian Development Bank.

## Loading

```python
import numpy as np, pandas as pd

X = np.load("X_kharif.npy")            # (59190, 7, 28)
y = np.load("y_kharif.npy")            # (59190,)
meta = pd.read_csv("meta_kharif.csv")  # ID, crop_type, Season, Province, District
names = pd.read_csv("label_map.csv")

# Group by district, never split randomly
from sklearn.model_selection import StratifiedGroupKFold
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for tr, te in cv.split(X, y, groups=meta["District"]):
    ...
```

To use the arrays with the paper's code, place them in
`data/processed_monthly/` inside the repository.

## Code

The full pipeline, trained models, and evaluation scripts:
https://github.com/rms-ali247/conformal-crop-sits

## Citation

```bibtex
@inproceedings{ali2026conformal,
  title     = {Conformal Crop Classification Across Regions with Sentinel-2 Time Series},
  author    = {Ali, Rao Muhammad Saeed and Fraz, Muhammad Moazam and
               Ullah, Sami and Zafar, Zuhair},
  booktitle = {TBD},
  year      = {2026}
}
```

Please also cite this deposit by its DOI, and acknowledge the Asian Development
Bank crop-type survey of Pakistan as the source of the ground truth.

## License

CC-BY-4.0. See `LICENSE`.
