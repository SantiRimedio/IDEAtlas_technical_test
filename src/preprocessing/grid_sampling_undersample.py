"""
grid_sampling_triclass_balanced.py

Creates a sampling grid with balanced representation across three classes.
Ensures Class 0, Class 1, and Class 2 have similar numbers of patches.

Strategy:
1. Find patches dominated by each class
2. Sample equal (or specified ratio) numbers from each class
3. Stratified train/val/test splits

Usage:
    from grid_sampling_triclass_balanced import create_triclass_balanced_grid
    
    grid_path = create_triclass_balanced_grid(
        input_raster='path/to/raster.tif',
        output_path='path/to/output.geojson',
        target_patches_per_class=1000,   # OR
        class_ratios=[1, 1, 1],          # Equal representation
        min_class_pct=10.0,              # Dominant class threshold
        splits=(0.7, 0.15, 0.15),
        random_seed=42
    )
"""

import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box
import random
import argparse
import warnings
import pandas as pd


def create_grid_with_class_dominance(
    raster_path: str, 
    patch_size: int = 128,
    min_class_pct: float = 30.0,
    aoi_path: str = None
) -> gpd.GeoDataFrame:
    """
    Creates a grid with dominant class identification.
    
    Args:
        raster_path: Path to input raster
        patch_size: Size of grid cells in pixels
        min_class_pct: Min percentage for a class to be considered "dominant"
        aoi_path: Optional path to AOI shapefile/geojson
        
    Returns:
        GeoDataFrame with geometry, class labels, and class percentages
    """
    # Load AOI if provided
    aoi = None
    if aoi_path:
        aoi = gpd.read_file(aoi_path)
        print(f"📍 AOI loaded: {len(aoi)} feature(s)")
    
    with rasterio.open(raster_path) as src:
        height, width = src.shape
        crs = src.crs
        transform = src.transform

        grid_cells = []
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                win_height = min(patch_size, height - y)
                win_width = min(patch_size, width - x)
                window = Window(x, y, win_width, win_height)
                bounds = rasterio.windows.bounds(window, transform)
                grid_cells.append(box(*bounds))

        grid = gpd.GeoDataFrame({
            'geometry': grid_cells,
            'temp_id': range(1, len(grid_cells) + 1)
        }, crs=crs)

        # Apply AOI filter first
        if aoi is not None:
            if aoi.crs != grid.crs:
                print(f"   Reprojecting AOI from {aoi.crs} to {grid.crs}")
                aoi = aoi.to_crs(grid.crs)
            
            grid_original_len = len(grid)
            grid = gpd.sjoin(grid, aoi, how='inner', predicate='intersects')
            grid = grid.drop(columns=['index_right'], errors='ignore')
            grid = grid.drop_duplicates(subset=['temp_id'])
            grid = grid[~grid['geometry'].isna()].copy()
            grid = grid[~grid.geometry.is_empty].copy()
            print(f"   Filtered by AOI: {len(grid)}/{grid_original_len} patches kept")
        
        # Initialize columns
        grid['dominant_class'] = -1
        grid['class_0_pct'] = 0.0
        grid['class_1_pct'] = 0.0
        grid['class_2_pct'] = 0.0
        grid['dominant_pct'] = 0.0
        grid['class_counts'] = None
        grid['is_pure'] = False

        # Read raster data
        with rasterio.open(raster_path) as src:
            for idx in grid.index:
                geom = grid.loc[idx, 'geometry']
                window = src.window(*geom.bounds)
                data = src.read(1, window=window)
                valid_data = data[data >= 0]
                
                if len(valid_data) > 0:
                    classes, counts = np.unique(valid_data, return_counts=True)
                    total = counts.sum()
                    class_count_dict = dict(zip(classes, counts))
                    
                    # Calculate percentages for each class
                    for cls in [0, 1, 2]:
                        pct = (class_count_dict.get(cls, 0) / total) * 100
                        grid.at[idx, f'class_{cls}_pct'] = round(pct, 2)
                    
                    # Determine dominant class
                    dominant_class = classes[np.argmax(counts)]
                    dominant_pct = (counts.max() / total) * 100
                    
                    grid.at[idx, 'dominant_class'] = int(dominant_class)
                    grid.at[idx, 'dominant_pct'] = round(dominant_pct, 2)
                    grid.at[idx, 'class_counts'] = {int(k): int(v) for k, v in class_count_dict.items()}
                    grid.at[idx, 'is_pure'] = dominant_pct >= min_class_pct

        return grid[grid['dominant_class'] != -1].copy()


def balance_and_split_triclass(
    grid: gpd.GeoDataFrame,
    target_patches_per_class: int = None,
    class_ratios: list = None,
    sampling_mode: str = 'dominant',
    splits: tuple = (0.7, 0.15, 0.15),
    random_seed: int = None
) -> gpd.GeoDataFrame:
    """
    Balance three classes and create stratified splits.
    
    Args:
        grid: Input GeoDataFrame
        target_patches_per_class: Absolute number of patches per class (overrides ratios)
        class_ratios: Relative ratios [class0, class1, class2] (e.g., [1, 2, 1])
        sampling_mode: 'dominant' (use only dominant_class) or 'presence' (≥X% presence)
        splits: Train/val/test proportions
        random_seed: Random seed for reproducibility
        
    Returns:
        GeoDataFrame with set assignments and patch_id
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    if not np.isclose(sum(splits), 1.0):
        raise ValueError("Split proportions must sum to 1.0")
    
    # Separate patches by dominant class
    class_0_patches = grid[grid['dominant_class'] == 0].copy()
    class_1_patches = grid[grid['dominant_class'] == 1].copy()
    class_2_patches = grid[grid['dominant_class'] == 2].copy()
    
    available_counts = {
        0: len(class_0_patches),
        1: len(class_1_patches),
        2: len(class_2_patches)
    }
    
    print(f"\n📊 Available patches by dominant class:")
    for cls in [0, 1, 2]:
        print(f"  Class {cls}: {available_counts[cls]:,} patches")
    
    # Determine target counts for each class
    if target_patches_per_class is not None:
        target_counts = {0: target_patches_per_class, 
                        1: target_patches_per_class, 
                        2: target_patches_per_class}
        print(f"\n🎯 Target: {target_patches_per_class:,} patches per class")
    elif class_ratios is not None:
        # Use the rarest class as base
        min_available = min(available_counts.values())
        base_count = min_available
        
        total_ratio = sum(class_ratios)
        target_counts = {
            0: int(base_count * class_ratios[0] / min(class_ratios)),
            1: int(base_count * class_ratios[1] / min(class_ratios)),
            2: int(base_count * class_ratios[2] / min(class_ratios))
        }
        print(f"\n🎯 Target ratios [Class0:Class1:Class2] = {class_ratios}")
        print(f"   Based on rarest class: {min_available:,} patches")
    else:
        # Default: equal to the rarest class
        min_count = min(available_counts.values())
        target_counts = {0: min_count, 1: min_count, 2: min_count}
        print(f"\n🎯 Balancing to rarest class: {min_count:,} patches each")
    
    # Sample from each class
    sampled_patches = []
    
    for cls, patches in [(0, class_0_patches), (1, class_1_patches), (2, class_2_patches)]:
        target = target_counts[cls]
        available = len(patches)
        
        if target > available:
            print(f"  ⚠️  Class {cls}: Requested {target:,} but only {available:,} available. Using all.")
            sampled = patches
        else:
            sampled = patches.sample(n=target, random_state=random_seed)
            print(f"  ✅ Class {cls}: Sampled {target:,} from {available:,} available")
        
        sampled_patches.append(sampled)
    
    # Combine all sampled patches
    final_grid = gpd.GeoDataFrame(
        pd.concat(sampled_patches, ignore_index=True),
        crs=grid.crs
    )
    
    # Assign randomized patch IDs
    ids = list(range(1, len(final_grid) + 1))
    random.shuffle(ids)
    final_grid['patch_id'] = ids
    
    # Stratified split by dominant class
    final_grid['set'] = ''
    
    for cls, group in final_grid.groupby('dominant_class'):
        indices = group.index.tolist()
        random.shuffle(indices)
        
        n = len(indices)
        n_train = round(splits[0] * n)
        n_val = round(splits[1] * n)
        n_test = n - n_train - n_val
        
        final_grid.loc[indices[:n_train], 'set'] = 'train'
        final_grid.loc[indices[n_train:n_train + n_val], 'set'] = 'val'
        final_grid.loc[indices[n_train + n_val:], 'set'] = 'test'
    
    return final_grid.drop(columns=['temp_id'], errors='ignore')


def print_statistics_triclass(grid: gpd.GeoDataFrame):
    """Print detailed statistics about the final grid."""
    print("\n" + "="*60)
    print("📈 FINAL DATASET STATISTICS")
    print("="*60)
    
    # Overall split sizes
    print("\n1. Split sizes:")
    split_counts = grid.groupby('set').size()
    split_pcts = (split_counts / len(grid) * 100).round(1)
    for split in ['train', 'val', 'test']:
        if split in split_counts.index:
            print(f"   {split.capitalize():6s}: {split_counts[split]:4d} patches ({split_pcts[split]:5.1f}%)")
    
    # Class distribution across all splits
    print("\n2. Overall class distribution:")
    class_counts = grid['dominant_class'].value_counts().sort_index()
    for cls, count in class_counts.items():
        pct = round(count / len(grid) * 100, 1)
        print(f"   Class {cls}: {count:4d} patches ({pct:5.1f}%)")
    
    # Class distribution within each split
    print("\n3. Class distribution per split:")
    for split in ['train', 'val', 'test']:
        split_data = grid[grid['set'] == split]
        if len(split_data) > 0:
            print(f"   {split.capitalize()}:")
            class_dist = split_data['dominant_class'].value_counts().sort_index()
            for cls, count in class_dist.items():
                pct = round(count / len(split_data) * 100, 1)
                print(f"      Class {cls}: {count:4d} ({pct:5.1f}%)")
    
    # Average class percentages
    print("\n4. Average class purity per split:")
    for split in ['train', 'val', 'test']:
        split_data = grid[grid['set'] == split]
        if len(split_data) > 0:
            print(f"   {split.capitalize()}:")
            for cls in [0, 1, 2]:
                class_patches = split_data[split_data['dominant_class'] == cls]
                if len(class_patches) > 0:
                    avg_purity = class_patches['dominant_pct'].mean()
                    print(f"      Class {cls}: {avg_purity:.1f}% average dominance")
    
    print("\n" + "="*60)


def create_triclass_balanced_grid(
    input_raster: str,
    output_path: str,
    patch_size: int = 128,
    target_patches_per_class: int = None,
    class_ratios: list = None,
    min_class_pct: float = 30.0,
    splits: tuple = (0.7, 0.15, 0.15),
    random_seed: int = 42,
    verbose: bool = True,
    aoi_path: str = None
) -> str:
    """
    Main function to create three-way balanced grid.
    
    Args:
        input_raster: Path to input raster file
        output_path: Path where output GeoJSON will be saved
        patch_size: Size of grid cells in pixels
        target_patches_per_class: Absolute number of patches per class
        class_ratios: Relative ratios [class0, class1, class2] (e.g., [1, 2, 1])
                     If None and target_patches_per_class is None, balances to rarest class
        min_class_pct: Min percentage for class to be considered "dominant"
        splits: Tuple of (train, val, test) proportions
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress messages
        aoi_path: Optional path to AOI file
        
    Returns:
        Path to the saved output file
        
    Examples:
        # Equal balance (1:1:1)
        >>> grid_path = create_triclass_balanced_grid(
        ...     input_raster='labels.tif',
        ...     output_path='grid_equal.geojson',
        ...     class_ratios=[1, 1, 1]
        ... )
        
        # Custom ratio (Class 0 : Class 1 : Class 2 = 1:2:1)
        >>> grid_path = create_triclass_balanced_grid(
        ...     input_raster='labels.tif',
        ...     output_path='grid_custom.geojson',
        ...     class_ratios=[1, 2, 1]
        ... )
        
        # Absolute target (1000 patches per class)
        >>> grid_path = create_triclass_balanced_grid(
        ...     input_raster='labels.tif',
        ...     output_path='grid_absolute.geojson',
        ...     target_patches_per_class=1000
        ... )
    """
    def _print(msg):
        if verbose:
            print(msg)
    
    _print("🔍 Creating grid and calculating class distributions...")
    grid = create_grid_with_class_dominance(
        input_raster, 
        patch_size,
        min_class_pct,
        aoi_path
    )
    
    _print(f"\n✅ Initial grid created: {len(grid)} total patches")
    _print(f"   Patches by dominant class:")
    for cls in [0, 1, 2]:
        n = len(grid[grid['dominant_class'] == cls])
        _print(f"     Class {cls}: {n:,} patches")
    
    _print(f"\n🎯 Balancing classes...")
    grid = balance_and_split_triclass(
        grid,
        target_patches_per_class=target_patches_per_class,
        class_ratios=class_ratios,
        splits=splits,
        random_seed=random_seed
    )
    
    if verbose:
        print_statistics_triclass(grid)
    
    _print(f"\n💾 Saving to {output_path}")
    grid.to_file(output_path, driver='GeoJSON')
    _print(f"✅ Done! Final dataset: {len(grid)} patches saved.")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Create three-way balanced grid (Class 0, 1, 2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', required=True, help='Input reference raster path')
    parser.add_argument('--output', required=True, help='Output path to GeoJSON grid')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size in pixels')
    parser.add_argument('--target_patches', type=int, 
                        help='Absolute number of patches per class')
    parser.add_argument('--class_ratios', type=int, nargs=3,
                        help='Relative ratios for Class 0, 1, 2 (e.g., 1 2 1)')
    parser.add_argument('--min_class_pct', type=float, default=30.0,
                        help='Min percentage for class to be dominant')
    parser.add_argument('--splits', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        help='Train/val/test split proportions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--aoi', type=str, help='Optional path to AOI shapefile/geojson')
    args = parser.parse_args()

    create_triclass_balanced_grid(
        input_raster=args.input,
        output_path=args.output,
        patch_size=args.patch_size,
        target_patches_per_class=args.target_patches,
        class_ratios=args.class_ratios,
        min_class_pct=args.min_class_pct,
        splits=tuple(args.splits),
        random_seed=args.seed,
        verbose=True,
        aoi_path=args.aoi
    )


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()