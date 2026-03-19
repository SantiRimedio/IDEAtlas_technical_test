"""
make_large_grid.py

Creates a large, full-coverage grid for AMBA — no undersampling.

Problem with the original grid (117 tiles):
  The original sampler forced 1:1:1 class balance by undersampling the
  majority classes down to the rarest (39 class-2 tiles).  This threw away
  7,807 useful class-1 patches.

New strategy — keep everything, oversample class-2 at train time:
  1. Relabel patches: any patch with ≥ CLASS2_PRESENCE_PCT class-2 pixels
     gets labelled "class-2" regardless of dominant class.
     (Informal settlements are embedded in urban fabric and are rarely the
     majority pixel class in a 128×128 tile — the dominance criterion missed
     most of them.)
  2. Keep ALL patches from every group — no undersampling.
  3. Stratified train/val/test split (70/15/15) within each group.
  4. Class imbalance is handled at training time via:
       - DiceLoss class weights  (already in experiment_runner)
       - Oversampling class-2 patches in the training array
       - Data augmentation  (next step)

Result: ~8 000+ patches vs 117.

Usage:
    python -m src.preprocessing.make_large_grid
    python -m src.preprocessing.make_large_grid --dry-run   # just print counts
    python -m src.preprocessing.make_large_grid --cls2-pct 3.0
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.grid_sampling_undersample import create_grid_with_class_dominance
from src.config import (
    REFERENCE_LABELS, AOI, GRIDS_DIR, PATCH_SIZE, RANDOM_SEED
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CLASS2_PRESENCE_PCT = 5.0   # patches with ≥ this % class-2 pixels → group "2"
MIN_CLASS_PCT       = 10.0  # threshold for class-0/1 dominance (passed to scanner)
SPLITS              = (0.70, 0.15, 0.15)

OUTPUT_PATH = GRIDS_DIR / "grid_large.geojson"


def make_large_grid(
    cls2_presence_pct: float = CLASS2_PRESENCE_PCT,
    min_class_pct: float = MIN_CLASS_PCT,
    splits: tuple = SPLITS,
    random_seed: int = RANDOM_SEED,
    dry_run: bool = False,
    output_path: Path = OUTPUT_PATH,
) -> gpd.GeoDataFrame | None:
    """
    Build and save the full-coverage grid (no undersampling).

    Parameters
    ----------
    cls2_presence_pct : patches with ≥ this % class-2 pixels are labelled group-2
    min_class_pct     : passed to the raster scanner for class-0/1 dominance
    splits            : (train, val, test) proportions
    dry_run           : print counts and return without writing
    output_path       : where to save the output GeoJSON

    Returns
    -------
    GeoDataFrame (or None on dry-run)
    """
    np.random.seed(random_seed)

    # ------------------------------------------------------------------
    # Step 1: full raster scan
    # ------------------------------------------------------------------
    print("Scanning reference labels raster (takes ~2 min)...")
    grid = create_grid_with_class_dominance(
        str(REFERENCE_LABELS),
        patch_size=PATCH_SIZE,
        min_class_pct=min_class_pct,
        aoi_path=str(AOI),
    )
    print(f"  Total patches in AOI: {len(grid):,}")

    # ------------------------------------------------------------------
    # Step 2: presence-based re-labelling
    # ------------------------------------------------------------------
    grid = grid.copy()
    grid["group"] = grid["dominant_class"].astype(int)
    mask_cls2 = grid["class_2_pct"] >= cls2_presence_pct
    grid.loc[mask_cls2, "group"] = 2

    g0 = grid[grid["group"] == 0]
    g1 = grid[grid["group"] == 1]
    g2 = grid[grid["group"] == 2]

    print(f"\nPresence-based groups (class-2 threshold = {cls2_presence_pct}%):")
    print(f"  Group 0 (background-dominant, <{cls2_presence_pct}% cls-2): {len(g0):,}")
    print(f"  Group 1 (formal-dominant,     <{cls2_presence_pct}% cls-2): {len(g1):,}")
    print(f"  Group 2 (any patch with      ≥{cls2_presence_pct}% cls-2): {len(g2):,}")
    print(f"  Total  : {len(grid):,}  (keeping ALL — no undersampling)")

    if dry_run:
        # Projected split counts
        for grp, name in [(0, "background"), (1, "formal"), (2, "informal≥5%")]:
            n = len(grid[grid["group"] == grp])
            print(f"\n  [{name}] {n} patches  →  "
                  f"train ~{round(splits[0]*n)}, val ~{round(splits[1]*n)}, test ~{round(splits[2]*n)}")
        print("\n[DRY RUN] No file written.")
        return None

    # ------------------------------------------------------------------
    # Step 3: keep ALL patches — stratified split by group
    # ------------------------------------------------------------------
    final = grid.copy()
    final["set"] = ""

    for grp, subset in final.groupby("group"):
        idx = subset.index.tolist()
        np.random.shuffle(idx)
        n = len(idx)
        n_train = round(splits[0] * n)
        n_val   = round(splits[1] * n)
        final.loc[idx[:n_train],              "set"] = "train"
        final.loc[idx[n_train:n_train+n_val], "set"] = "val"
        final.loc[idx[n_train+n_val:],        "set"] = "test"

    # Sequential zero-padded patch IDs (6 digits) so filenames sort correctly
    # extract_patch.py uses f"{prefix}_{patch_id}.tif" — string IDs give S2_000001.tif
    final = final.reset_index(drop=True)
    final["patch_id"] = [f"{i+1:06d}" for i in range(len(final))]
    final["dominant_class"] = final["group"]
    final = final.drop(columns=["group", "temp_id"], errors="ignore")
    final = gpd.GeoDataFrame(final, crs=grid.crs)

    # ------------------------------------------------------------------
    # Step 4: summary + save
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  LARGE GRID SUMMARY  (no undersampling)")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        sub = final[final["set"] == split]
        cls2_count = (sub["class_2_pct"] >= cls2_presence_pct).sum()
        print(f"  {split:5s}: {len(sub):5d} patches  "
              f"({cls2_count:4d} with ≥{cls2_presence_pct:.0f}% cls-2,  "
              f"{cls2_count/len(sub)*100:.1f}%)")
    print(f"  {'Total':5s}: {len(final):5d} patches")
    print(f"\n  vs original grid_balanced.geojson: 117 patches  "
          f"({len(final)/117:.0f}× more data)")
    print("=" * 60)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_file(str(output_path), driver="GeoJSON")
    print(f"\nSaved → {output_path}")
    print(f"\nNext step: extract patches for the new grid.")
    print(f"  python -m src.preprocessing.extract_patches_for_grid \\")
    print(f"      --grid {output_path} \\")
    print(f"      --output datasets/patches_large/")

    return final


def main():
    parser = argparse.ArgumentParser(
        description="Build a full-coverage AMBA grid (no undersampling)"
    )
    parser.add_argument("--cls2-pct",    type=float, default=CLASS2_PRESENCE_PCT,
                        help=f"Min class-2 %% to label a patch as class-2 (default {CLASS2_PRESENCE_PCT})")
    parser.add_argument("--min-cls-pct", type=float, default=MIN_CLASS_PCT,
                        help=f"Min %% for class-0/1 dominance (default {MIN_CLASS_PCT})")
    parser.add_argument("--output",      type=Path,  default=OUTPUT_PATH,
                        help=f"Output GeoJSON path (default {OUTPUT_PATH})")
    parser.add_argument("--seed",        type=int,   default=RANDOM_SEED)
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print projected counts without writing any file")
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        make_large_grid(
            cls2_presence_pct=args.cls2_pct,
            min_class_pct=args.min_cls_pct,
            random_seed=args.seed,
            dry_run=args.dry_run,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
