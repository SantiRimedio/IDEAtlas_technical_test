"""
stack_density_patches.py

Stacks the three available density rasters into 3-band DEN3 patches:
  Band 1: building density        (s2_2025_building_density_updated.tif)
  Band 2: local/regional ratio    (density_multiscale_AMBA/ratio_local_regional.tif)
  Band 3: density variance        (density_multiscale_AMBA/variance.tif)

For patches outside the coverage of ratio/variance rasters, band 2 and 3
are filled with 0.0 (these are peripheral AMBA areas).

Output patches are saved as DEN3_<patch_id>.tif alongside existing DEN_*.tif files.

Usage:
    python -m src.preprocessing.stack_density_patches
    python -m src.preprocessing.stack_density_patches --splits train val test
    python -m src.preprocessing.stack_density_patches --dry-run  # preview only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PATCHES_TRAIN_DIR, PATCHES_VAL_DIR, PATCHES_TEST_DIR,
    DENSITY, DENSITY_MULTISCALE_DIR,
)

# ---------------------------------------------------------------------------
# Source rasters (band order determines DEN3 band order)
# ---------------------------------------------------------------------------
DENSITY_SOURCES = [
    DENSITY,                                         # band 1: original density
    DENSITY_MULTISCALE_DIR / "ratio_local_regional.tif",  # band 2: local/regional ratio
    DENSITY_MULTISCALE_DIR / "variance.tif",         # band 3: density variance
]

SPLIT_DIRS = {
    "train": PATCHES_TRAIN_DIR,
    "val":   PATCHES_VAL_DIR,
    "test":  PATCHES_TEST_DIR,
}

PATCH_SIZE = 128


def extract_band_at_patch(
    raster_path: Path,
    bounds: tuple,
    out_shape: tuple = (PATCH_SIZE, PATCH_SIZE),
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Extract a single band from raster_path clipped to the geographic bounds
    of an existing patch. Resamples to out_shape using bilinear interpolation.
    Fills out-of-coverage areas with fill_value (boundless read).

    Parameters
    ----------
    raster_path : Path
    bounds      : (left, bottom, right, top) in raster CRS units
    out_shape   : (height, width) of output array
    fill_value  : value used where raster has no data (out of extent)

    Returns
    -------
    np.ndarray of shape out_shape, dtype float32
    """
    with rasterio.open(raster_path) as src:
        window = src.window(*bounds)
        data = src.read(
            1,
            window=window,
            out_shape=out_shape,
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=fill_value,
        )
    return data.astype(np.float32)


def stack_patch(den_path: Path, dry_run: bool = False) -> Path:
    """
    Build a 3-band DEN3 patch from an existing DEN patch.

    Parameters
    ----------
    den_path : Path to existing single-band DEN_*.tif
    dry_run  : if True, return output path without writing

    Returns
    -------
    Path to written DEN3_*.tif (or would-be path if dry_run)
    """
    patch_id = den_path.stem.replace("DEN_", "")
    out_path  = den_path.parent / f"DEN3_{patch_id}.tif"

    if dry_run:
        return out_path

    # Read reference geometry from existing DEN patch
    with rasterio.open(den_path) as ref:
        bounds   = ref.bounds
        transform = ref.transform
        crs      = ref.crs
        profile  = ref.profile.copy()

    # Extract each band at the same geographic extent
    bands = []
    for source in DENSITY_SOURCES:
        band = extract_band_at_patch(source, bounds)
        bands.append(band)

    stack = np.stack(bands, axis=0)  # (3, 128, 128)

    # Write 3-band GeoTIFF
    profile.update(
        count=3,
        dtype="float32",
        compress="lzw",
        nodata=None,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stack)

    return out_path


def run(splits: list, dry_run: bool = False) -> dict:
    """
    Process all requested splits. Returns per-split stats.
    """
    stats = {}

    for split in splits:
        patches_dir = SPLIT_DIRS[split]
        den_patches  = sorted(patches_dir.glob("DEN_*.tif"))

        if not den_patches:
            print(f"[{split}] No DEN_*.tif patches found in {patches_dir}")
            stats[split] = {"total": 0, "written": 0, "skipped": 0}
            continue

        written = 0
        skipped = 0

        print(f"\n[{split}] Processing {len(den_patches)} DEN patches...")

        for i, den_path in enumerate(den_patches, 1):
            out_path = den_path.parent / f"DEN3_{den_path.stem.replace('DEN_', '')}.tif"

            # Skip if already exists (allow resuming)
            if not dry_run and out_path.exists():
                skipped += 1
                continue

            if dry_run:
                # Verify source rasters are readable at this location
                with rasterio.open(den_path) as ref:
                    bounds = ref.bounds
                b1 = extract_band_at_patch(DENSITY_SOURCES[0], bounds)
                b2 = extract_band_at_patch(DENSITY_SOURCES[1], bounds)
                b3 = extract_band_at_patch(DENSITY_SOURCES[2], bounds)
                # Check what fraction of b2/b3 is non-zero (indicates coverage)
                cov2 = float((b2 > 0).mean())
                cov3 = float((b3 > 0).mean())
                if i <= 5 or i == len(den_patches):
                    print(f"  [{i:3d}/{len(den_patches)}] {den_path.name}  "
                          f"ratio_cov={cov2:.1%}  var_cov={cov3:.1%}  "
                          f"→ {out_path.name}")
                written += 1
                continue

            try:
                stack_patch(den_path)
                written += 1
                if i % 20 == 0 or i == len(den_patches):
                    print(f"  {i}/{len(den_patches)} patches written...")
            except Exception as e:
                print(f"  ERROR on {den_path.name}: {e}")
                skipped += 1

        action = "would write" if dry_run else "written"
        print(f"[{split}] Done. {written} {action}, {skipped} skipped.")
        stats[split] = {"total": len(den_patches), "written": written, "skipped": skipped}

    return stats


def print_summary(stats: dict):
    total_w = sum(v["written"] for v in stats.values())
    total_t = sum(v["total"]   for v in stats.values())
    print(f"\n{'='*50}")
    print(f"  Summary: {total_w}/{total_t} DEN3 patches created")
    for split, s in stats.items():
        print(f"  {split:5s}: {s['written']:3d}/{s['total']:3d} written, "
              f"{s['skipped']:3d} skipped")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Stack 3 density rasters into DEN3 patches alongside existing DEN patches"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to process (default: all three)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview coverage stats without writing any files"
    )
    args = parser.parse_args()

    missing = [str(p) for p in DENSITY_SOURCES if not p.exists()]
    if missing:
        print("ERROR: Missing source rasters:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    print("Source rasters:")
    for i, src in enumerate(DENSITY_SOURCES, 1):
        with rasterio.open(src) as r:
            print(f"  Band {i}: {src.name}  {r.width}x{r.height}  {r.crs}")

    if args.dry_run:
        print("\n[DRY RUN] Sampling first/last patches to check coverage...")

    stats = run(args.splits, dry_run=args.dry_run)
    print_summary(stats)

    if not args.dry_run:
        # Verify a written patch
        for split in args.splits:
            sample = sorted(SPLIT_DIRS[split].glob("DEN3_*.tif"))
            if sample:
                with rasterio.open(sample[0]) as s:
                    arr = s.read()
                    print(f"\nVerify {sample[0].name}: "
                          f"shape={arr.shape}, "
                          f"band ranges: "
                          f"[{arr[0].min():.3f},{arr[0].max():.3f}] "
                          f"[{arr[1].min():.3f},{arr[1].max():.3f}] "
                          f"[{arr[2].min():.3f},{arr[2].max():.3f}]")
                break


if __name__ == "__main__":
    main()
