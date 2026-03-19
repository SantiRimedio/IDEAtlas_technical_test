#!/usr/bin/env python3
"""
05_extract_patches.py

Extract 128x128 patches from rasters using the balanced grid.
Creates train/val/test subdirectories with S2, DEN, RF modalities.

Validates patches for NaN content and data quality.

Output:
    datasets/patches/
    ├── train/ [S2_*.tif, DEN_*.tif, RF_*.tif]
    ├── val/   [S2_*.tif, DEN_*.tif, RF_*.tif]
    └── test/  [S2_*.tif, DEN_*.tif, RF_*.tif]
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.extract_patch import extract_patches
from src.config import (
    S2_2020, DENSITY, REFERENCE_LABELS,
    GRID_BALANCED, PATCHES_DIR, PATCH_SIZE, MIN_VALID_RATIO
)

def main():
    """Extract patches from rasters."""
    print("\n" + "=" * 80)
    print("📦 PATCH EXTRACTION")
    print("=" * 80)

    # Ensure grid exists
    if not GRID_BALANCED.exists():
        print(f"\n❌ Grid file not found: {GRID_BALANCED}")
        print("Run scripts/04_grid_sampling.py first")
        sys.exit(1)

    print(f"\nGrid: {GRID_BALANCED}")
    print(f"Output directory: {PATCHES_DIR}")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE} pixels")
    print(f"Min valid ratio: {MIN_VALID_RATIO:.0%}")

    # Extract S2 patches
    print(f"\n1️⃣  Extracting Sentinel-2 patches...")
    extract_patches(
        raster_path=str(S2_2020),
        grid_path=str(GRID_BALANCED),
        output_root=str(PATCHES_DIR),
        patch_size=PATCH_SIZE,
        verbose=True,
        prefix="S2",
        min_valid_ratio=MIN_VALID_RATIO,
        allow_negative=False,
        save_validation_log=True
    )

    # Extract DEN patches
    print(f"\n2️⃣  Extracting Density patches...")
    extract_patches(
        raster_path=str(DENSITY),
        grid_path=str(GRID_BALANCED),
        output_root=str(PATCHES_DIR),
        patch_size=PATCH_SIZE,
        verbose=True,
        prefix="DEN",
        min_valid_ratio=MIN_VALID_RATIO,
        allow_negative=False,
        save_validation_log=True
    )

    # Extract RF (reference labels) patches
    print(f"\n3️⃣  Extracting Reference Label patches...")
    extract_patches(
        raster_path=str(REFERENCE_LABELS),
        grid_path=str(GRID_BALANCED),
        output_root=str(PATCHES_DIR),
        patch_size=PATCH_SIZE,
        verbose=True,
        prefix="RF",
        min_valid_ratio=MIN_VALID_RATIO,
        allow_negative=True,  # Allow negative values for label encoding
        save_validation_log=True
    )

    print(f"\n✅ Patch extraction complete!")
    print(f"   Output directory: {PATCHES_DIR}")

if __name__ == "__main__":
    main()
