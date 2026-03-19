# Data Format Specification

## Patch Files

Patches are 128×128 pixel GeoTIFF tiles stored in `datasets/patches/{train,val,test}/`.

### File Naming

| Prefix | Content | Bands | Value Range |
|--------|---------|-------|-------------|
| `S2_` | Sentinel-2 multispectral | 6 (B2, B3, B4, B8, B11, B12) | [0, 1] float32 |
| `DEN_` | Building density features | 1+ | [0, 1] float32 |
| `RF_` | Reference labels (ground truth) | 1 | Integer class IDs |

Patch numbering uses `patch_id` from the sampling grid: `{PREFIX}_{patch_id:06d}.tif`

### Class Labels

| Class | Meaning |
|-------|---------|
| 0 | Background / No data |
| 1 | Non-informal urban |
| 2 | Informal settlement (slum) |

Training uses 0-indexed one-hot encoding. Inference output uses 1-indexed classes (`final_pred + 1`).

## Sentinel-2 Normalization

Raw S2 values are divided by 10000 to get [0,1] reflectance. The guard check is:
```python
if np.max(s2) > 3:
    s2 = s2 / 10000
```

### S2 Harmonization

For images with processing baseline >= 04.00, a -1000 offset correction is applied before normalization. This is handled automatically by `S2Downloader` in `src/downloads/download_s2.py`.

### Band Order

Bands are stored in this order: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2).

## Patch Validation

During extraction (`src/preprocessing/extract_patch.py`), patches are validated:
- Minimum 95% valid pixels required (configurable via `--min_valid_ratio`)
- Invalid pixels: NaN, Inf, nodata values, negative values (unless `--allow_negative`)
- Rejected patches are logged to `{PREFIX}_validation_log.json`

## Grid Format

Sampling grids are GeoJSON files with required columns:
- `patch_id`: Unique integer identifier
- `set`: One of `train`, `val`, `test`
- `geometry`: Polygon matching patch footprint

Default split ratio: 70% train / 15% val / 15% test, with oversampling of class-2 patches.

## Raster Conventions

- All rasters use LZW compression
- CRS is preserved from the reference raster (typically EPSG:4326 or UTM)
- AOI clipping uses `rasterio.mask` with crop=True
