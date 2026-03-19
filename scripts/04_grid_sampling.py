#!/usr/bin/env python3
"""
04_grid_sampling.py

Generate a balanced 3-class grid from reference labels.
Creates stratified train/val/test splits with class balancing.

Output:
    - datasets/grids/grid_balanced.geojson
    - Console statistics
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.grid_sampling_undersample import create_triclass_balanced_grid
from src.config import REFERENCE_LABELS, AOI, GRID_BALANCED, PATCH_SIZE

def main():
    """Generate balanced grid."""
    print("\n" + "=" * 80)
    print("📊 BALANCED GRID GENERATION")
    print("=" * 80)

    # Ensure output directory exists
    GRID_BALANCED.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nInput raster: {REFERENCE_LABELS}")
    print(f"AOI: {AOI}")
    print(f"Output: {GRID_BALANCED}")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE} pixels")

    # Generate grid
    output_path = create_triclass_balanced_grid(
        input_raster=str(REFERENCE_LABELS),
        output_path=str(GRID_BALANCED),
        patch_size=PATCH_SIZE,
        target_patches_per_class=None,  # Balance to rarest class
        class_ratios=None,  # Equal balance [1, 1, 1]
        min_class_pct=30.0,
        splits=(0.7, 0.15, 0.15),
        random_seed=42,
        verbose=True,
        aoi_path=str(AOI)
    )

    print(f"\n✅ Grid saved to: {output_path}")

if __name__ == "__main__":
    main()
