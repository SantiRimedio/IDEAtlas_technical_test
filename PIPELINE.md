# Techo Pipeline — Local Execution Guide

This document explains how to run the end-to-end Techo pipeline locally on your machine.

## Overview

The pipeline consists of 8 stages:

1. **Download** (skip locally — data already exists)
2. **Morphometrics** (skip locally — density data already exists)
3. **Alignment** — Validate spatial consistency of rasters
4. **Grid Sampling** — Create balanced 3-class training grid
5. **Patch Extraction** — Extract 128×128 patches from rasters
6. **Training** — Train MBCNN model on patches
7. **Inference** — Run sliding-window inference on full rasters
8. **Change Detection** (optional) — Compare predictions across years

## Setup

### Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies (one-time)
pip install -r requirements.txt
```

### Directory Structure

The pipeline expects this structure:

```
Techo/
├── data/
│   └── boundaries/                  # AOI GeoJSONs
│       ├── partidos_amba_IA_BID_2025.geojson
│       └── renabap-datos-barrios.geojson
├── datasets/
│   ├── raw/                         # Downloaded rasters
│   │   ├── s2_2020_partidos_ambas-006.tif
│   │   ├── s2_2025_partidos_ambas-005.tif
│   │   ├── s2_2025_building_density_updated.tif
│   │   ├── reference_labels_partidos_AMBA.tif
│   │   ├── ghsl_builtup_partidos_amba.tif
│   │   └── density_multiscale_AMBA/
│   ├── grids/                       # Generated grids
│   ├── patches/                     # Extracted patches
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── predictions/                 # Inference outputs
│   └── reports/                     # Validation reports
├── models/                          # Model weights
├── logs/                            # Training logs
├── scripts/                         # Pipeline scripts
│   ├── data_audit.py
│   ├── 03_alignment.py
│   ├── 04_grid_sampling.py
│   ├── 05_extract_patches.py
│   ├── 06_train.py
│   └── 07_inference.py
└── src/
    ├── config.py                    # Central configuration
    ├── downloads/                   # Download modules
    ├── preprocessing/               # Grid & patch modules
    ├── training/                    # Model training
    └── inference/                   # Inference
```

## Running the Pipeline

### Step 1: Data Audit (Recommended)

Before proceeding, audit the data for quality issues:

```bash
python scripts/data_audit.py
```

**Output:**
- Console summary with data quality checks
- `datasets/reports/data_audit_report.json` with detailed analysis

**Checks:**
- Raster metadata (CRS, resolution, bands)
- NaN/Inf values
- Value ranges and normalization status
- Patch consistency and naming

### Step 2: Alignment Validation

Validate that all rasters are properly aligned:

```bash
python scripts/03_alignment.py
```

**Output:**
- Console validation report
- `datasets/reports/alignment_report.json`

**Checks:**
- CRS consistency
- Resolution consistency
- Bounds overlap
- Grid alignment

**Expected result:**
```
✅ PERFECT_ALIGNMENT
```

If errors occur, the report will indicate which rasters need reprojection/realignment.

### Step 3: Generate Grid

Create a balanced 3-class training grid:

```bash
python scripts/04_grid_sampling.py
```

**Output:**
- `datasets/grids/grid_balanced.geojson` (patch grid with train/val/test splits)
- Console statistics showing class distribution

**Key outputs:**
- Total patches by class
- Split sizes (70% train, 15% val, 15% test)
- Average class purity per split

### Step 4: Extract Patches

Extract 128×128 patches from rasters using the grid:

```bash
python scripts/05_extract_patches.py
```

**Output:**
```
datasets/patches/
├── train/     [S2_*.tif, DEN_*.tif, RF_*.tif]
├── val/       [S2_*.tif, DEN_*.tif, RF_*.tif]
└── test/      [S2_*.tif, DEN_*.tif, RF_*.tif]
```

Also saves validation logs:
- `S2_validation_log.json`
- `DEN_validation_log.json`
- `RF_validation_log.json`

**Expected result:**
- 765 total patches (534 train / 114 val / 117 test, but will vary based on grid)
- All patch_ids consistent across S2/DEN/RF modalities

### Step 5: Train Model

Train MBCNN on extracted patches:

```bash
python scripts/06_train.py
```

**Output:**
- `models/latest_model.h5` (trained weights)
- `logs/training_log.csv` (epoch-by-epoch history)
- Console training progress and test evaluation

**Configuration:**
- Batch size: 8
- Epochs: 100 (early stopping with patience=20)
- Loss: Dice + 3.0×Focal loss (weighted by class frequency)
- Optimizer: Adam (lr=1e-4)

**Metrics printed:**
- Overall accuracy
- Per-class precision, recall, F1
- Class 2 (slums) analysis: precision, recall, F1

**Expected result:**
```
Test accuracy: 0.85+
Class 2 F1: 0.60+
```

### Step 6: Inference

Run sliding-window inference on full rasters:

```bash
python scripts/07_inference.py
```

**Output:**
- `datasets/predictions/prediction_map.tif` (classification raster)
- Console statistics showing class distribution

**Method:**
- Patch size: 128×128 pixels
- Stride: 64 pixels (50% overlap)
- Window: Hann 2D window for smooth blending
- Clip: To AOI boundary

**Output classes:**
- 0: Background
- 1: Formal urban
- 2: Informal settlement (slums)

## Configuration

All paths and parameters are defined in `src/config.py`. Key settings:

```python
# Model
PATCH_SIZE = 128
N_CLASSES = 3
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4

# Data validation
MIN_VALID_RATIO = 0.95  # Minimum valid pixels in patch

# Raster paths (automatically resolved relative to PROJECT_ROOT)
S2_2020 = datasets/raw/s2_2020_partidos_ambas-006.tif
S2_2025 = datasets/raw/s2_2025_partidos_ambas-005.tif
DENSITY = datasets/raw/s2_2025_building_density_updated.tif
REFERENCE_LABELS = datasets/raw/reference_labels_partidos_AMBA.tif
AOI = data/boundaries/partidos_amba_IA_BID_2025.geojson
```

To modify:
1. Edit `src/config.py`
2. Re-run relevant pipeline stages

## Data Format

### Patches

**Sentinel-2 patches (S2_*.tif):**
- Shape: (128, 128, 10 bands)
- Bands: B2, B3, B4, B8, B11, B12 (+ 4 derived indices)
- Range: 0–10000 (raw) or 0–1 (normalized)
- The `norm_s2()` function in `data_utils.py` handles normalization

**Density patches (DEN_*.tif):**
- Shape: (128, 128, 1 band)
- Range: 0–1 (normalized building density)

**Reference label patches (RF_*.tif):**
- Shape: (128, 128, 1 band)
- Values: 0 (background), 1 (formal), 2 (informal settlement)
- Stored as uint8, converted to one-hot (128, 128, 3) during training

### Class Labels

| Class | Name | Value |
|-------|------|-------|
| 0 | Background | 0 |
| 1 | Formal urban | 1 |
| 2 | Informal settlement | 2 |

## Troubleshooting

### "File not found" errors

Check that all input files exist:

```bash
ls -la datasets/raw/
ls -la data/boundaries/
```

Use the data audit script to identify missing files:

```bash
python scripts/data_audit.py
```

### NaN/Inf values in patches

The audit script will report patches with >5% invalid pixels. These are skipped during extraction.

If many patches are skipped:
1. Check raster metadata: `python scripts/03_alignment.py`
2. Check value ranges in audit report

### Training is slow

If no GPU is detected:
```python
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")
```

Training on CPU will be 10–50× slower. Consider:
- Using cloud GPU (Google Colab, AWS, GCP)
- Reducing batch size to fit GPU memory
- Reducing number of patches (sample from grids)

### Out of memory (OOM)

Reduce `BATCH_SIZE` in `src/config.py`:

```python
BATCH_SIZE = 4  # was 8
```

Rerun training script.

### Model not improving

Common causes:
1. **Data quality** — Check audit report for NaN patterns
2. **Class imbalance** — Check class distribution in grid report
3. **Hyperparameters** — Adjust in `scripts/06_train.py`:
   - `FOCAL_ALPHA` (higher = more weight on minority class)
   - `FOCAL_GAMMA` (higher = more focus on hard examples)
   - `LEARNING_RATE` (lower = more stable, slower learning)

## Replicability

This pipeline is designed for replicability across machines:

1. **All paths are relative** to `PROJECT_ROOT` (defined in `src/config.py`)
2. **Random seeds are fixed** (`RANDOM_SEED = 42`)
3. **Dependencies are pinned** in `requirements.txt`

To reproduce results on another machine:

```bash
# Clone repo
git clone <repo> Techo
cd Techo

# Install dependencies
pip install -r requirements.txt

# Place data in expected directories
# (see Directory Structure section above)

# Run pipeline
python scripts/data_audit.py
python scripts/03_alignment.py
python scripts/04_grid_sampling.py
python scripts/05_extract_patches.py
python scripts/06_train.py
python scripts/07_inference.py
```

Results will be identical given the same random seed.

## Next Steps

After running the pipeline:

1. **Analyze results:**
   - Review confusion matrix in training output
   - Check class 2 (slums) precision/recall/F1
   - Examine spatial patterns in `prediction_map.tif`

2. **Iterate:**
   - Adjust hyperparameters in `src/config.py` or script configs
   - Retrain: `python scripts/06_train.py`
   - Re-run inference: `python scripts/07_inference.py`

3. **Validate on new areas:**
   - Prepare new AOI and rasters
   - Update paths in `src/config.py`
   - Run grid generation, patch extraction, and inference
   - (Optional) Fine-tune model on new area data

4. **Export results:**
   - Predictions: `datasets/predictions/prediction_map.tif`
   - Model: `models/latest_model.h5`
   - Logs: `logs/training_log.csv`

## References

- **Model:** Multi-Branch CNN (MBCNN) for multi-input semantic segmentation
- **Loss:** Combined Dice + Categorical Focal Loss
- **Data:** Sentinel-2 + building density morphometrics
- **Inference:** Sliding window with Hann window blending
- **Classes:** Background, formal urban, informal settlement

See `docs/` for detailed documentation:
- `docs/pipeline_overview.md`
- `docs/model_architecture.md`
- `docs/data_format.md`
