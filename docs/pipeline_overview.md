# Pipeline Overview

## Stages

The pipeline runs sequentially through 5 stages. Each stage has a numbered notebook in `notebooks/` that orchestrates the work, calling Python modules from `src/`.

```
01 Download Satellite   ─→  02 Download Morphometrics  ─→  03 Alignment
        │                            │                          │
   S2 composite              Building density            Validate CRS,
   GHSL built-up             Morphometric features       resolution, extent
   Reference labels          Multi-scale density         match across inputs
        │                            │                          │
        └────────────────────────────┴──────────────────────────┘
                                     │
                              04 Grid Sampling
                                     │
                         Balanced train/val/test grid
                         with class-2 (slum) oversampling
                                     │
                              05 Extract Patches
                                     │
                         128×128 GeoTIFF tiles per modality
                         (S2_, DEN_, RF_ prefixes)
                                     │
                                06 Train
                                     │
                         MBCNN with Dice+Focal loss
                         Saves checkpoints to models/
                                     │
                                07 Inference
                                     │
                         Sliding window + Hann blending
                         Full-image classification map
                                     │
                           08 Change Detection
                                     │
                         Compare 2020 vs 2025 predictions
                         Loss / Stable / Gain analysis
```

## Notebooks → Source Modules

| Notebook | Modules used |
|----------|-------------|
| `01_download_satellite.ipynb` | `src.downloads.gee_auth`, `src.downloads.download_s2`, `src.downloads.download_ghsl`, `src.downloads.download_utils`, `src.downloads.data_validator` |
| `02_download_morphometrics.ipynb` | `src.downloads.density_calculator_optimized`, `src.downloads.multiscale_density_calculator`, `src.downloads.morphometric_calculator` |
| `03_alignment.ipynb` | `src.preprocessing.validate_alignment` |
| `04_grid_sampling.ipynb` | `src.preprocessing.grid_sampling_undersample` |
| `05_extract_patches.ipynb` | `src.preprocessing.extract_patch` |
| `06_train.ipynb` | `src.training.mbcnn`, `src.training.data_utils`, `src.training.losses` |
| `07_inference.ipynb` | `src.inference.inference`, `src.inference.data_utils`, `src.training.mbcnn` |
| `08_change_detection.ipynb` | Direct rasterio/geopandas (no custom modules) |

## Inputs and Outputs per Stage

| Stage | Input | Output | Location |
|-------|-------|--------|----------|
| 01 Download S2 | AOI boundary (`data/boundaries/`) | S2 composite, GHSL, reference labels (`.tif`) | `datasets/raw/` |
| 02 Morphometrics | Building footprints + reference raster | Density rasters (`.tif`) | `datasets/raw/` |
| 03 Alignment | All rasters from stages 01-02 | Alignment report (pass/fail) | stdout |
| 04 Grid Sampling | Reference label raster + AOI | Grid GeoJSON with train/val/test splits | `datasets/raw/` |
| 05 Extract Patches | S2 + density + reference rasters + grid | 128×128 patches per modality | `datasets/patches/{train,val,test}/` |
| 06 Train | Patches from stage 05 | Model weights (`.h5`) | `models/` |
| 07 Inference | Full rasters + model weights | Classification map (`.tif`) | `datasets/predictions/` |
| 08 Change Detection | Two prediction maps (different years) | Change map + statistics | `datasets/predictions/` |
