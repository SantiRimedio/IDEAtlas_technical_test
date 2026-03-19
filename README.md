# Techo: Informal Settlement Detection from Satellite Imagery

Deep learning pipeline for detecting and monitoring informal settlements in Latin American cities using Sentinel-2 satellite imagery and building morphometric features.

## Overview

This project uses a Multi-Branch Convolutional Neural Network (MBCNN) to classify urban areas into three categories: background, formal urban, and informal settlements. The model combines Sentinel-2 multispectral imagery with building density features computed from OpenStreetMap/Google Buildings footprints.

Current focus area: Buenos Aires Metropolitan Area (AMBA), Argentina.

## Setup

```bash
pip install -r requirements.txt
```

For data downloads, authenticate Google Earth Engine:
```python
import ee
ee.Authenticate()
ee.Initialize(project='your-project-id')
```

## Project Structure

```
src/                    Python modules (downloads, preprocessing, training, inference)
notebooks/              Jupyter notebooks numbered by pipeline stage (01-08)
data/boundaries/        AOI boundary files (GeoJSON)
models/                 Trained model weights (.h5)
datasets/               Large data files - not in git (see docs/download_guide.md)
docs/                   Detailed documentation per topic
```

## Pipeline

Run notebooks in order:

1. **01_download_satellite** - Download S2 composites, GHSL, reference labels via GEE
2. **02_download_morphometrics** - Compute building density at multiple scales
3. **03_alignment** - Validate spatial alignment across all input rasters
4. **04_grid_sampling** - Create balanced train/val/test grid with class-2 oversampling
5. **05_extract_patches** - Extract 128x128 patches per modality
6. **06_train** - Train MBCNN with Dice+Focal loss
7. **07_inference** - Run sliding-window inference on full images
8. **08_change_detection** - Compare predictions across years

See [docs/pipeline_overview.md](docs/pipeline_overview.md) for detailed stage descriptions.

## Documentation

- [Pipeline Overview](docs/pipeline_overview.md) - Stage-by-stage flow with inputs/outputs
- [Data Format](docs/data_format.md) - Patch naming, band order, normalization specs
- [Model Architecture](docs/model_architecture.md) - MBCNN/MTCNN structure and training config
- [Download Guide](docs/download_guide.md) - How to obtain and place dataset files

## Key Technical Details

- **Input**: S2 bands B2/B3/B4/B8/B11/B12 (normalized to [0,1]) + building density raster
- **Patch size**: 128x128 pixels at 10m resolution
- **Classes**: 0=background, 1=formal urban, 2=informal settlement
- **Inference**: Hann-window blended sliding window (stride = patch_size/2)
