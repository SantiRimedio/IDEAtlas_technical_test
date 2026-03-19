# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Informal settlement detection pipeline for Latin American cities using Sentinel-2 imagery + building morphometric features. Multi-Branch CNN (MBCNN) produces 3-class segmentation: background, formal urban, informal settlement. Current focus: Buenos Aires Metropolitan Area (AMBA).

## Setup

```bash
pip install -r requirements.txt
```

GEE authentication required for downloads: `ee.Authenticate()` then `ee.Initialize(project='...')`

## Repository Layout

```
src/downloads/          GEE downloaders (S2, GHSL, buildings), density calculators
src/preprocessing/      Grid generation, patch extraction, raster alignment validation
src/training/           MBCNN model (mbcnn.py), data loaders, loss functions
src/inference/          Sliding-window inference, post-processing utilities
notebooks/              01-08 numbered by pipeline stage
data/boundaries/        AOI boundary GeoJSONs (tracked in git)
models/                 Trained .h5 weights (tracked in git)
datasets/               Large rasters + patches (gitignored, see docs/download_guide.md)
docs/                   Detailed docs per topic (saves context tokens - read these first)
```

## Key Documentation (read instead of source when possible)

- `docs/pipeline_overview.md` - Stage flow, notebook→module mapping, inputs/outputs per stage
- `docs/data_format.md` - Patch naming (S2_/DEN_/RF_), band order, normalization, class labels
- `docs/model_architecture.md` - MBCNN/MTCNN structure, loss functions, training config
- `docs/download_guide.md` - How to obtain datasets and where to place them

## Quick Reference

**CLI patch extraction:**
```bash
python src/preprocessing/extract_patch.py \
  --raster input.tif --grid grid.gpkg --output output_dir/ \
  --prefix S2 --patch_size 128 --min_valid_ratio 0.95
```

**Notebook imports pattern** (all notebooks use this):
```python
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, PROJECT_ROOT)
from src.training.mbcnn import mbcnn
```

## Critical Conventions

- S2 normalization: if `max > 3`, divide by 10000. Guard in `src/training/data_utils.py:norm_s2()`
- Class indices: 0-based in training one-hot, 1-based in inference output (`final_pred + 1`)
- All rasters saved with LZW compression
- Inference uses Hann window blending (stride = patch_size // 2)
- Patch validation: reject patches with <95% valid pixels
- S2 harmonization: -1000 offset for processing baseline >= 04.00
