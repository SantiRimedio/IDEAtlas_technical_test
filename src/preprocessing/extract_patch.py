#!/usr/bin/env python3
"""
extract_patches.py

Extracts fixed-size patches using patch_id as patch numbers, with:
- Randomized patch numbering
- Automatic train/val/test directory organization
- Support for any number of bands
- Configurable filename prefix
- DATA VALIDATION: Skips patches with invalid data

Usage:
    python extract_patches.py --raster input.tif --grid grid.gpkg --output output_dir --prefix S2
"""

import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
import os
from tqdm import tqdm
import sys

def validate_grid(grid: gpd.GeoDataFrame) -> bool:
    """Check if grid contains required columns."""
    required_cols = {'set', 'patch_id'}
    if not required_cols.issubset(grid.columns):
        print(f"Error: Grid missing required columns {required_cols - set(grid.columns)}")
        return False
    if not all(grid['set'].isin(['train', 'val', 'test'])):
        print("Error: 'set' column must contain only 'train', 'val', or 'test'")
        return False
    return True

def is_patch_valid(data: np.ndarray, nodata: float = None, 
                   min_valid_ratio: float = 0.95,
                   allow_negative: bool = False) -> tuple[bool, dict]:
    """
    Validate if a patch contains sufficient valid data.
    
    Args:
        data: Numpy array of shape (bands, height, width)
        nodata: NoData value to check against
        min_valid_ratio: Minimum ratio of valid pixels required (0.0-1.0)
        allow_negative: Whether to allow negative values as valid
        
    Returns:
        tuple: (is_valid: bool, stats: dict)
    """
    total_pixels = data.shape[1] * data.shape[2]  # height * width
    total_values = data.size  # bands * height * width
    
    stats = {
        'total_pixels_per_band': total_pixels,
        'total_values': total_values,
        'valid_values': 0,
        'invalid_breakdown': {
            'infinite': 0,
            'nan': 0,
            'nodata': 0,
            'negative': 0,
            'zero': 0
        },
        'valid_ratio': 0.0,
        'value_range': {'min': None, 'max': None, 'mean': None}
    }
    
    # Create validity mask
    valid_mask = np.ones_like(data, dtype=bool)
    
    # Check for infinite values
    inf_mask = np.isinf(data)
    if inf_mask.any():
        valid_mask &= ~inf_mask
        stats['invalid_breakdown']['infinite'] = inf_mask.sum()
    
    # Check for NaN values  
    nan_mask = np.isnan(data)
    if nan_mask.any():
        valid_mask &= ~nan_mask
        stats['invalid_breakdown']['nan'] = nan_mask.sum()
    
    # Check for NoData values
    if nodata is not None:
        if np.isfinite(nodata):
            nodata_mask = (data == nodata)
        elif np.isneginf(nodata):  # NoData is -inf
            nodata_mask = np.isneginf(data)
        elif np.isposinf(nodata):  # NoData is +inf
            nodata_mask = np.isposinf(data)
        else:  # NoData is NaN
            nodata_mask = np.isnan(data)
            
        if nodata_mask.any():
            valid_mask &= ~nodata_mask
            stats['invalid_breakdown']['nodata'] = nodata_mask.sum()
    
    # Check for negative values (if not allowed)
    if not allow_negative:
        negative_mask = (data < 0) & np.isfinite(data)
        if negative_mask.any():
            valid_mask &= ~negative_mask
            stats['invalid_breakdown']['negative'] = negative_mask.sum()
    
    # Check for zero values (optional - uncomment if zeros are invalid)
    # zero_mask = (data == 0)
    # if zero_mask.any():
    #     valid_mask &= ~zero_mask
    #     stats['invalid_breakdown']['zero'] = zero_mask.sum()
    
    # Calculate statistics
    valid_count = valid_mask.sum()
    stats['valid_values'] = int(valid_count)
    stats['valid_ratio'] = float(valid_count) / total_values
    
    # Calculate value range for valid data
    if valid_count > 0:
        valid_data = data[valid_mask]
        stats['value_range']['min'] = float(np.min(valid_data))
        stats['value_range']['max'] = float(np.max(valid_data))
        stats['value_range']['mean'] = float(np.mean(valid_data))
    
    # Determine if patch is valid
    is_valid = stats['valid_ratio'] >= min_valid_ratio
    
    return is_valid, stats

def extract_patches(
    raster_path: str,
    grid_path: str,
    output_root: str,
    patch_size: int = 128,
    nodata: float = None,
    verbose: bool = True,
    prefix: str = "S1",
    min_valid_ratio: float = 0.95,
    allow_negative: bool = False,
    save_validation_log: bool = True
) -> None:
    """
    Extract patches using patch_id as patch numbers, with data validation.
    
    Args:
        raster_path: Path to input raster
        grid_path: Path to grid file with FID and set assignments
        output_root: Root output directory
        patch_size: Patch size in pixels (default: 128)
        nodata: Custom nodata value (default: read from raster)
        verbose: Show progress bars (default: True)
        prefix: Filename prefix for patches (default: "S1")
        min_valid_ratio: Minimum ratio of valid pixels required (default: 0.95)
        allow_negative: Whether negative values are valid (default: False)
        save_validation_log: Save validation statistics to log file (default: True)
    """
    try:
        # Load and validate grid
        grid = gpd.read_file(grid_path)
        if not validate_grid(grid):
            sys.exit(1)

        # Create output directories
        for set_name in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_root, set_name), exist_ok=True)

        # Prepare validation log
        validation_log = []
        
        with rasterio.open(raster_path) as src:
            # Get raster metadata
            num_bands = src.count
            dtype = src.dtypes[0]
            raster_nodata = src.nodatavals[0]
            nodata = nodata if nodata is not None else raster_nodata

            if verbose:
                print(f"Raster Info: {num_bands} band(s), dtype: {dtype}, nodata: {nodata}")
                print(f"Using prefix: {prefix}")
                print(f"Validation criteria:")
                print(f"  - Min valid ratio: {min_valid_ratio:.1%}")
                print(f"  - Allow negative values: {allow_negative}")

            # Process each dataset split
            set_counts = {'train': 0, 'val': 0, 'test': 0}
            set_skipped = {'train': 0, 'val': 0, 'test': 0}
            
            for set_name in ['train', 'val', 'test']:
                set_patches = grid[grid['set'] == set_name]
                if len(set_patches) == 0:
                    continue

                if verbose:
                    print(f"\nProcessing {len(set_patches)} {set_name} patches...")
                    pbar = tqdm(set_patches.iterrows(), total=len(set_patches))
                else:
                    pbar = set_patches.iterrows()

                for _, row in pbar:
                    patch_id = row['patch_id'] 
                    window = src.window(*row['geometry'].bounds)
                    
                    # Read data with padding if needed
                    data = src.read(
                        range(1, num_bands + 1),
                        window=window,
                        out_shape=(num_bands, patch_size, patch_size),
                        boundless=True,
                        fill_value=nodata
                    )

                    # VALIDATE PATCH DATA
                    is_valid, stats = is_patch_valid(
                        data, 
                        nodata=nodata, 
                        min_valid_ratio=min_valid_ratio,
                        allow_negative=allow_negative
                    )
                    
                    # Log validation results
                    log_entry = {
                        'patch_id': patch_id,
                        'set': set_name,
                        'is_valid': is_valid,
                        'valid_ratio': stats['valid_ratio'],
                        'stats': stats
                    }
                    validation_log.append(log_entry)

                    if is_valid:
                        # Save patch using configurable prefix
                        output_path = os.path.join(output_root, set_name, f"{prefix}_{patch_id}.tif")
                        with rasterio.open(
                            output_path,
                            'w',
                            driver='GTiff',
                            height=patch_size,
                            width=patch_size,
                            count=num_bands,
                            dtype=dtype,
                            crs=src.crs,
                            transform=rasterio.windows.transform(window, src.transform),
                            nodata=nodata,
                            compress='lzw'  # Add compression to save space
                        ) as dst:
                            dst.write(data)
                        
                        set_counts[set_name] += 1
                        
                        if verbose and hasattr(pbar, 'set_description'):
                            pbar.set_description(f"{set_name}: {set_counts[set_name]} valid")
                    else:
                        set_skipped[set_name] += 1
                        if verbose and hasattr(pbar, 'set_description'):
                            pbar.set_description(f"{set_name}: {set_skipped[set_name]} skipped")

        # Print final statistics
        if verbose:
            print(f"\n{'='*60}")
            print(f"EXTRACTION SUMMARY")
            print(f"{'='*60}")
            
            total_extracted = sum(set_counts.values())
            total_skipped = sum(set_skipped.values())
            total_processed = total_extracted + total_skipped
            
            print(f"Total patches processed: {total_processed}")
            print(f"Total patches extracted: {total_extracted} ({total_extracted/total_processed:.1%})")
            print(f"Total patches skipped: {total_skipped} ({total_skipped/total_processed:.1%})")
            print()
            
            for set_name in ['train', 'val', 'test']:
                extracted = set_counts[set_name] 
                skipped = set_skipped[set_name]
                total = extracted + skipped
                if total > 0:
                    print(f"{set_name:>5}: {extracted:>4} extracted, {skipped:>4} skipped "
                          f"({extracted/total:.1%} valid)")

        # Save validation log
        if save_validation_log and validation_log:
            import json
            log_path = os.path.join(output_root, f"{prefix}_validation_log.json")
            with open(log_path, 'w') as f:
                json.dump(validation_log, f, indent=2)
            
            if verbose:
                print(f"\nValidation log saved to: {log_path}")
                
                # Quick stats from log
                invalid_patches = [p for p in validation_log if not p['is_valid']]
                if invalid_patches:
                    print(f"\nTop reasons for patch rejection:")
                    reasons = {}
                    for patch in invalid_patches:
                        for reason, count in patch['stats']['invalid_breakdown'].items():
                            if count > 0:
                                reasons[reason] = reasons.get(reason, 0) + 1
                    
                    for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {reason}: {count} patches")

    except Exception as e:
        print(f"\nError during extraction: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract patches using FID as patch numbers with data validation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--raster', required=True, help='Input raster path')
    parser.add_argument('--grid', required=True, help='Grid file with set assignments')
    parser.add_argument('--output', required=True, help='Root output directory')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size in pixels')
    parser.add_argument('--nodata', type=float, help='Override nodata value')
    parser.add_argument('--quiet', action='store_true', help='Disable progress output')
    parser.add_argument('--prefix', default='S1', help='Filename prefix for patches (e.g., S1, S2, MORPHO, TARGET)')
    parser.add_argument('--min_valid_ratio', type=float, default=0.95, 
                      help='Minimum ratio of valid pixels required (0.0-1.0)')
    parser.add_argument('--allow_negative', action='store_true', 
                      help='Allow negative values as valid data')
    parser.add_argument('--no_validation_log', action='store_true',
                      help='Skip saving validation log file')
    
    args = parser.parse_args()
    
    extract_patches(
        raster_path=args.raster,
        grid_path=args.grid,
        output_root=args.output,
        patch_size=args.patch_size,
        nodata=args.nodata,
        verbose=not args.quiet,
        prefix=args.prefix,
        min_valid_ratio=args.min_valid_ratio,
        allow_negative=args.allow_negative,
        save_validation_log=not args.no_validation_log
    )