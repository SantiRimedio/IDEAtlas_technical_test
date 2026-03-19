"""
experiment_runner.py

Lightweight hyperparameter experiment runner for MBCNN loss function tuning.
Trains each ExperimentConfig, evaluates on test set, and logs results to CSV.

Usage:
    # Run all defined experiments
    python -m src.evaluation.experiment_runner

    # Run a single named experiment (for testing)
    python -m src.evaluation.experiment_runner --run baseline_combined

    # Run only a specific block (a, b, c, or d)
    python -m src.evaluation.experiment_runner --block a
"""

import os
import sys
import json
import csv
import time
import argparse
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("SM_FRAMEWORK", "tf.keras")

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)

import segmentation_models as sm

from src.config import (
    PATCHES_TRAIN_DIR, PATCHES_VAL_DIR, PATCHES_TEST_DIR,
    PATCHES_LARGE_TRAIN_DIR, PATCHES_LARGE_VAL_DIR, PATCHES_LARGE_TEST_DIR,
    N_CLASSES, PATCH_SIZE, S2_BANDS, DEN_BANDS, CLASS_NAMES
)
from src.training.data_utils import load_data, calculate_class_weights
from src.training.mbcnn import mbcnn
from src.evaluation.evaluate import (
    load_test_data, evaluate_model,
    print_evaluation_report, save_evaluation_json
)

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    name: str
    # Loss components
    use_dice: bool = True
    use_focal: bool = True
    focal_alpha: float = 0.85
    focal_gamma: float = 2.5
    dice_focal_ratio: float = 3.0     # combined = dice + ratio*focal
    class2_weight_multiplier: float = 1.0  # multiplies the class-2 weight from balanced calculation
    normalize_class_weights: bool = True   # if False, use raw (un-renormalized) weights
    # Training schedule
    epochs: int = 40
    lr: float = 1e-4
    dropout_rate: float = 0.3
    batch_size: int = 8
    early_stopping_patience: int = 12
    reduce_lr_patience: int = 6
    # Monitoring
    monitor: str = "val_f1"            # "val_f1" or "val_class2_recall"
    monitor_mode: str = "max"
    # Input modality
    den_prefix: str = "DEN"            # "DEN" (1-band) or "DEN3" (3-band multiscale)
    den_bands: int = 1                 # must match den_prefix band count
    # Dataset
    use_large_grid: bool = False       # True → load from patches_large/ (~8 000 tiles)
    class2_oversample_factor: int = 1  # repeat class-2 training patches N× (1 = off)
    # Meta
    block: str = "a"
    notes: str = ""


# ---------------------------------------------------------------------------
# Experiment definitions  (10 structured runs)
# ---------------------------------------------------------------------------

BLOCK_A = [
    ExperimentConfig(
        name="baseline_combined",
        use_dice=True, use_focal=True,
        focal_alpha=0.85, focal_gamma=2.5, dice_focal_ratio=3.0,
        class2_weight_multiplier=1.0, block="a",
        notes="Control: current best config (dice + focal)"
    ),
    ExperimentConfig(
        name="pure_dice",
        use_dice=True, use_focal=False,
        class2_weight_multiplier=1.0, block="a",
        notes="Is focal loss contributing?"
    ),
    ExperimentConfig(
        name="pure_focal",
        use_dice=False, use_focal=True,
        focal_alpha=0.85, focal_gamma=2.5,
        class2_weight_multiplier=1.0, block="a",
        notes="Is dice loss contributing?"
    ),
]

# ---------------------------------------------------------------------------
# Block A diagnosis revealed three problems:
#   1. val_f1 monitoring is dominated by class 1 (~82% of pixels) and gives
#      no gradient signal toward learning class 2.
#   2. Weight renormalization (/ sum * 3) caps class-2 weight at ≈2.59
#      regardless of the multiplier — essentially no boost for large mult.
#   3. Single-scalar focal alpha does nothing for per-class imbalance.
#
# Block B tests the two critical fixes independently:
#   B4 — just the monitoring change (same weights as pure_dice A2)
#   B5 — monitoring change + raw (un-normalized) 10x class-2 boost
#   B6 — monitoring change + raw 50x class-2 boost  (more aggressive)
# ---------------------------------------------------------------------------

BLOCK_B = [
    ExperimentConfig(
        name="dice_recall_monitor",
        use_dice=True, use_focal=False,
        class2_weight_multiplier=1.0,
        normalize_class_weights=True,     # same as A2 (pure_dice)
        monitor="val_class2_recall",
        block="b",
        notes="Pure dice, same weights as A2 — only monitoring changes to val_class2_recall"
    ),
    ExperimentConfig(
        name="dice_cls2_10x_raw",
        use_dice=True, use_focal=False,
        class2_weight_multiplier=10.0,
        normalize_class_weights=False,    # raw weights: cls2 ≈ 134
        monitor="val_class2_recall",
        block="b",
        notes="Pure dice, raw 10x class-2 boost (cls2_weight≈134), val_class2_recall monitor"
    ),
    ExperimentConfig(
        name="dice_cls2_50x_raw",
        use_dice=True, use_focal=False,
        class2_weight_multiplier=50.0,
        normalize_class_weights=False,    # raw weights: cls2 ≈ 670
        monitor="val_class2_recall",
        block="b",
        notes="Pure dice, raw 50x class-2 boost (cls2_weight≈670), val_class2_recall monitor"
    ),
]

# ---------------------------------------------------------------------------
# Block C: 3-band multiscale density (DEN3) experiments.
# Both runs replicate the exact Colab winning config (dice + 3×focal,
# alpha=0.85, gamma=2.5, val_f1 monitor) but now with 3-band DEN3 input.
#
# C7 — baseline_combined_den3: direct replication of Colab winner locally
# C8 — pure_dice_den3: best 1-band loss (A2) applied to 3-band input
#       — isolates whether DEN3 helps independent of loss choice
# ---------------------------------------------------------------------------

BLOCK_C = [
    ExperimentConfig(
        name="baseline_combined_den3",
        use_dice=True, use_focal=True,
        focal_alpha=0.85, focal_gamma=2.5, dice_focal_ratio=3.0,
        class2_weight_multiplier=1.0,
        normalize_class_weights=True,
        monitor="val_f1",
        den_prefix="DEN3", den_bands=3,
        block="c",
        notes="Colab winning config (dice+3×focal, val_f1 monitor) with 3-band DEN3 input"
    ),
    ExperimentConfig(
        name="pure_dice_den3",
        use_dice=True, use_focal=False,
        class2_weight_multiplier=1.0,
        normalize_class_weights=True,
        monitor="val_f1",
        den_prefix="DEN3", den_bands=3,
        block="c",
        notes="Best 1-band loss config (pure_dice A2) applied to 3-band DEN3 — isolates DEN3 contribution"
    ),
]

# ---------------------------------------------------------------------------
# Block D: large-grid retraining (8 000 tiles, 70× more data).
#
# A-C analysis conclusions:
#   - best loss: pure_dice (A2) with val_class2_recall monitoring
#   - DEN3 hurts (Block C rejected)
#   - class-2 is rare even in large grid (~7% of training patches)
#   - main bottleneck was data scarcity — now fixed with full AOI tiling
#
# D9 — large_pure_dice:
#   Exact replica of A2 (pure_dice) but trained on patches_large.
#   Baseline to quantify the data-size improvement alone.
#
# D10 — large_pure_dice_oversample:
#   Same as D9 but training array has class-2 patches repeated 5× so
#   class-2 frequency rises from ~7% → ~26%.  Tests whether oversampling
#   on top of the larger dataset further helps class-2 recall.
# ---------------------------------------------------------------------------

BLOCK_D = [
    ExperimentConfig(
        name="large_pure_dice",
        use_dice=True, use_focal=False,
        class2_weight_multiplier=1.0,
        normalize_class_weights=True,
        monitor="val_class2_recall",
        epochs=40,
        use_large_grid=True,
        class2_oversample_factor=1,
        block="d",
        notes="Pure dice on large grid (8 000 tiles) — quantify data-size gain"
    ),
    ExperimentConfig(
        name="large_pure_dice_oversample",
        use_dice=True, use_focal=False,
        class2_weight_multiplier=1.0,
        normalize_class_weights=True,
        monitor="val_class2_recall",
        epochs=40,
        use_large_grid=True,
        class2_oversample_factor=5,   # class-2 train patches repeated 5×
        block="d",
        notes="Pure dice + 5× class-2 oversampling on large grid"
    ),
]

ALL_EXPERIMENTS = BLOCK_A + BLOCK_B + BLOCK_C + BLOCK_D


# ---------------------------------------------------------------------------
# Data loading (done once for all experiments)
# ---------------------------------------------------------------------------

def _load_split_aligned(split_dir: Path, den_prefix: str):
    """
    Load one data split (train/val/test) with proper patch-ID alignment.

    Uses RF_*.tif files as the master list of patch IDs, then loads matching
    S2_<id>.tif and <den_prefix>_<id>.tif.  Patches missing any modality are
    silently skipped.  This is robust to non-contiguous patch IDs caused by
    the raster-validity filter in extract_patch.py.

    Returns
    -------
    s2  : np.ndarray (N, H, W, S2_BANDS), float32  — S2 normalized
    den : np.ndarray (N, H, W, den_bands), float32
    rf  : np.ndarray (N, H, W, N_CLASSES), float32  — one-hot
    rf_has_cls2 : np.ndarray (N,), bool — True if patch contains any class-2 pixel
    """
    import rasterio as rio
    from keras.utils import to_categorical
    from src.training.data_utils import norm_s2

    rf_files = sorted(split_dir.glob("RF_*.tif"))
    if not rf_files:
        raise FileNotFoundError(f"No RF_*.tif files in {split_dir}")

    s2_list, den_list, rf_list, cls2_mask = [], [], [], []

    for rf_path in rf_files:
        pid = rf_path.stem.replace("RF_", "")
        s2_path = split_dir / f"S2_{pid}.tif"
        den_path = split_dir / f"{den_prefix}_{pid}.tif"

        if not s2_path.exists() or not den_path.exists():
            continue

        with rio.open(s2_path) as ds:
            s2 = ds.read().astype(np.float32)
            s2 = np.moveaxis(s2, 0, -1)
            s2 = np.nan_to_num(s2, nan=0.0)
            s2 = norm_s2(s2)

        with rio.open(den_path) as ds:
            den = ds.read().astype(np.float32)
            den = np.moveaxis(den, 0, -1)
            den = np.nan_to_num(den, nan=0.0)

        with rio.open(rf_path) as ds:
            rf_raw = ds.read(1).astype(np.int32)

        rf_oh = to_categorical(rf_raw, num_classes=N_CLASSES).astype(np.float32)
        has_cls2 = bool(np.any(rf_raw == 2))

        s2_list.append(s2)
        den_list.append(den)
        rf_list.append(rf_oh)
        cls2_mask.append(has_cls2)

    return (
        np.array(s2_list, dtype=np.float32),
        np.array(den_list, dtype=np.float32),
        np.array(rf_list, dtype=np.float32),
        np.array(cls2_mask, dtype=bool),
    )


def _oversample_class2(s2, den, rf, has_cls2, factor: int):
    """
    Repeat class-2 training patches `factor` additional times.

    factor=5 means class-2 patches appear 5+1=6× total → ~6× frequency boost.
    Returns shuffled arrays.
    """
    if factor <= 0:
        return s2, den, rf

    idx_cls2 = np.where(has_cls2)[0]
    if len(idx_cls2) == 0:
        print("  Warning: no class-2 patches found — oversampling skipped")
        return s2, den, rf

    extra = np.tile(idx_cls2, factor)
    all_idx = np.concatenate([np.arange(len(s2)), extra])
    np.random.shuffle(all_idx)

    before = has_cls2.sum()
    after = (np.concatenate([has_cls2, has_cls2[extra]])).sum()
    print(f"  Oversampled class-2: {before} → {after} patches "
          f"({after/len(all_idx)*100:.1f}% of {len(all_idx)} total train)")
    return s2[all_idx], den[all_idx], rf[all_idx]


def load_all_data(den_prefix: str = "DEN", use_large_grid: bool = False,
                  class2_oversample_factor: int = 1):
    """
    Load train/val/test patches with proper patch-ID alignment.

    Parameters
    ----------
    den_prefix            : "DEN" (1-band) or "DEN3" (3-band multiscale density)
    use_large_grid        : if True, load from patches_large/ (~8 000 tiles)
    class2_oversample_factor : repeat class-2 training patches N extra times
    """
    if use_large_grid:
        train_dir = PATCHES_LARGE_TRAIN_DIR
        val_dir   = PATCHES_LARGE_VAL_DIR
        test_dir  = PATCHES_LARGE_TEST_DIR
        tag = "large grid"
    else:
        train_dir = PATCHES_TRAIN_DIR
        val_dir   = PATCHES_VAL_DIR
        test_dir  = PATCHES_TEST_DIR
        tag = "original grid"

    print(f"\nLoading training data ({tag}, den_prefix='{den_prefix}')...")
    train_s2, train_den, train_rf, train_has_cls2 = _load_split_aligned(
        train_dir, den_prefix
    )

    print("Loading validation data...")
    val_s2, val_den, val_rf, _ = _load_split_aligned(val_dir, den_prefix)

    # Replace any residual NaNs
    for arr in [train_s2, train_den, train_rf, val_s2, val_den, val_rf]:
        np.nan_to_num(arr, copy=False, nan=0.0)

    # Oversample class-2 patches in training set
    if class2_oversample_factor > 1:
        train_s2, train_den, train_rf = _oversample_class2(
            train_s2, train_den, train_rf,
            train_has_cls2, class2_oversample_factor
        )

    print(f"  Train: S2={train_s2.shape}, {den_prefix}={train_den.shape}, RF={train_rf.shape}")
    print(f"  Val  : S2={val_s2.shape},   {den_prefix}={val_den.shape},   RF={val_rf.shape}")

    # Class weights from training labels
    base_class_weights = calculate_class_weights(train_rf)
    print(f"  Base class weights (balanced): {base_class_weights}")

    print(f"Loading test data ({tag}, den_prefix='{den_prefix}')...")
    test_s2, test_den, test_rf, test_ids = load_test_data(
        test_dir, den_prefix=den_prefix
    )

    return {
        "train": ([train_s2, train_den], train_rf),
        "val":   ([val_s2, val_den], val_rf),
        "test":  (test_s2, test_den, test_rf),
        "base_class_weights": base_class_weights,
        "den_prefix": den_prefix,
    }


# ---------------------------------------------------------------------------
# Loss construction
# ---------------------------------------------------------------------------

def build_loss(config: ExperimentConfig, base_class_weights: np.ndarray):
    """Build segmentation-models combined loss matching notebook 06 pattern."""
    # Apply class2 multiplier
    weights = base_class_weights.copy()
    weights[2] *= config.class2_weight_multiplier
    if config.normalize_class_weights:
        # Re-normalize so weights sum to n_classes (keeps scale stable)
        weights = weights / weights.sum() * N_CLASSES
    # else: use raw (un-renormalized) weights — allows much larger class-2 signal
    print(f"  Loss class weights: {np.round(weights, 3)}")

    if config.use_dice and config.use_focal:
        dice_loss = sm.losses.DiceLoss(class_weights=weights)
        focal_loss = sm.losses.CategoricalFocalLoss(
            alpha=config.focal_alpha, gamma=config.focal_gamma
        )
        return dice_loss + (config.dice_focal_ratio * focal_loss)
    elif config.use_dice:
        return sm.losses.DiceLoss(class_weights=weights)
    elif config.use_focal:
        return sm.losses.CategoricalFocalLoss(
            alpha=config.focal_alpha, gamma=config.focal_gamma
        )
    else:
        raise ValueError("At least one of use_dice or use_focal must be True")


# ---------------------------------------------------------------------------
# Single experiment: train + evaluate
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig, data: dict, exp_dir: Path) -> dict:
    """
    Train one experiment config and evaluate on test set.
    Returns the evaluation metrics dict (plus training metadata).
    """
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Build model — use config.den_bands so DEN3 experiments get a 3-channel branch
    input_shapes = {
        0: (PATCH_SIZE, PATCH_SIZE, S2_BANDS),
        1: (PATCH_SIZE, PATCH_SIZE, config.den_bands),
    }
    model = mbcnn(
        CL=N_CLASSES,
        input_shapes=input_shapes,
        dropout_rate=config.dropout_rate,
        batch_norm=True,
        drop_train=True,
    )

    # Build loss
    loss_fn = build_loss(config, data["base_class_weights"])

    model.compile(
        optimizer=Adam(learning_rate=config.lr),
        loss=loss_fn,
        metrics=[
            sm.metrics.FScore(name="f1"),
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(class_id=2, name="class2_precision"),
            tf.keras.metrics.Recall(class_id=2, name="class2_recall"),
        ],
    )

    weights_path = exp_dir / "best_model.weights.h5"
    monitor = config.monitor
    mode = config.monitor_mode
    print(f"  Monitoring: {monitor} (mode={mode})")

    callbacks = [
        ModelCheckpoint(
            str(weights_path), monitor=monitor,
            save_best_only=True, save_weights_only=True,
            mode=mode, verbose=1,
        ),
        EarlyStopping(
            monitor=monitor, patience=config.early_stopping_patience,
            mode=mode, restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", patience=config.reduce_lr_patience,
            factor=0.5, min_lr=1e-6, verbose=1,
        ),
        CSVLogger(str(exp_dir / "training_log.csv"), append=False),
    ]

    train_inputs, train_labels = data["train"]
    val_inputs, val_labels = data["val"]

    t0 = time.time()
    history = model.fit(
        train_inputs, train_labels,
        validation_data=(val_inputs, val_labels),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    train_time_s = time.time() - t0

    epochs_run = len(history.history["loss"])
    monitor_history = history.history.get(config.monitor, [0.0])
    best_val_f1 = (max(monitor_history) if config.monitor_mode == "max"
                   else min(monitor_history))
    best_val_loss = min(history.history.get("val_loss", [float("inf")]))

    print(f"\nTraining done in {train_time_s/60:.1f} min ({epochs_run} epochs)")
    print(f"Best val_f1: {best_val_f1:.4f}, best val_loss: {best_val_loss:.4f}")

    # Evaluate on test set using best weights
    test_s2, test_den, test_rf = data["test"]
    metrics = evaluate_model(model, test_s2, test_den, test_rf,
                             batch_size=config.batch_size)

    # Attach training metadata
    metrics["experiment_name"] = config.name
    metrics["block"] = config.block
    metrics["epochs_run"] = epochs_run
    metrics["best_val_f1"] = float(best_val_f1)
    metrics["best_val_loss"] = float(best_val_loss)
    metrics["train_time_min"] = round(train_time_s / 60, 2)
    metrics["weights_path"] = str(weights_path)
    # Config snapshot
    metrics["monitor"] = config.monitor
    metrics["normalize_class_weights"] = config.normalize_class_weights
    metrics["class2_weight_multiplier"] = config.class2_weight_multiplier
    metrics["use_dice"] = config.use_dice
    metrics["use_focal"] = config.use_focal
    metrics["den_prefix"] = config.den_prefix
    metrics["den_bands"] = config.den_bands
    metrics["use_large_grid"] = config.use_large_grid
    metrics["class2_oversample_factor"] = config.class2_oversample_factor

    # Save full metrics JSON
    save_evaluation_json(metrics, exp_dir / "evaluation.json")

    return metrics


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------

RESULTS_COLS = [
    "experiment_name", "block",
    # Key metrics (class-2 focus)
    "class2_f1", "class2_precision", "class2_recall", "class2_iou",
    # Overall metrics
    "mean_iou", "macro_f1", "weighted_f1", "accuracy",
    # Other classes
    "class0_f1", "class0_iou", "class1_f1", "class1_iou",
    # Training info  (best_val_f1 stores best value of whichever metric was monitored)
    "best_val_f1", "best_val_loss", "epochs_run", "train_time_min",
    # Config snapshot for traceability
    "monitor", "normalize_class_weights", "class2_weight_multiplier",
    "use_dice", "use_focal", "den_prefix", "den_bands",
    "use_large_grid", "class2_oversample_factor",
    "weights_path",
]


def append_results_csv(metrics: dict, results_path: Path):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = results_path.exists()
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    print(f"Results appended to {results_path}")


def print_results_summary(results_path: Path):
    """Print comparison table from results CSV, sorted by class2_f1."""
    if not results_path.exists():
        print("No results CSV found yet.")
        return

    rows = []
    with open(results_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        return

    rows.sort(key=lambda r: float(r.get("class2_f1", 0)), reverse=True)

    print("\n" + "=" * 90)
    print("  EXPERIMENT RESULTS SUMMARY  (sorted by class-2 F1)")
    print("=" * 90)
    header = (
        f"  {'Name':<22} {'Blk':<4} "
        f"{'cls2_F1':>8} {'cls2_P':>8} {'cls2_R':>8} {'cls2_IoU':>9} "
        f"{'mIoU':>7} {'macF1':>7} "
        f"{'epochs':>7} {'time_m':>7}"
    )
    print(header)
    print("  " + "-" * 86)
    for r in rows:
        print(
            f"  {r.get('experiment_name','?'):<22} {r.get('block','?'):<4} "
            f"{float(r.get('class2_f1',0)):.4f}   "
            f"{float(r.get('class2_precision',0)):.4f}   "
            f"{float(r.get('class2_recall',0)):.4f}   "
            f"{float(r.get('class2_iou',0)):.4f}    "
            f"{float(r.get('mean_iou',0)):.4f}  "
            f"{float(r.get('macro_f1',0)):.4f}  "
            f"{r.get('epochs_run','?'):>7}  "
            f"{r.get('train_time_min','?'):>7}"
        )
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_experiments(
    configs: list,
    output_dir: Path = EXPERIMENTS_DIR,
    skip_existing: bool = True,
):
    """
    Load data once per den_prefix, then train + evaluate each experiment config.

    Experiments are grouped by den_prefix ("DEN" or "DEN3") so the data load
    happens once per group rather than once per experiment.
    """
    output_dir = Path(output_dir)
    results_path = output_dir / "results.csv"

    # Check which experiments already have results (to resume interrupted runs)
    done_names = set()
    if skip_existing and results_path.exists():
        with open(results_path) as f:
            for row in csv.DictReader(f):
                done_names.add(row.get("experiment_name", ""))

    # Group configs by (den_prefix, use_large_grid, class2_oversample_factor)
    # so data is loaded once per unique combination.
    prefix_groups: dict = {}
    for config in configs:
        key = (config.den_prefix, config.use_large_grid, config.class2_oversample_factor)
        prefix_groups.setdefault(key, []).append(config)

    total = len(configs)
    global_i = 0  # global counter across all prefixes for display

    for (den_prefix, use_large_grid, oversample_factor), group_configs in prefix_groups.items():
        # Filter out already-done experiments before loading data
        pending = [c for c in group_configs if c.name not in done_names]
        if not pending:
            for config in group_configs:
                global_i += 1
                print(f"\n[{global_i}/{total}] Skipping '{config.name}' (already in results.csv)")
            continue

        data = load_all_data(
            den_prefix=den_prefix,
            use_large_grid=use_large_grid,
            class2_oversample_factor=oversample_factor,
        )

        for config in group_configs:
            global_i += 1
            if config.name in done_names:
                print(f"\n[{global_i}/{total}] Skipping '{config.name}' (already in results.csv)")
                continue

            print(f"\n{'='*60}")
            print(f"  [{global_i}/{total}] Running experiment: {config.name}  (block {config.block})")
            print(f"  DEN input: {config.den_prefix} ({config.den_bands} band(s))")
            print(f"  Notes: {config.notes}")
            print(f"{'='*60}")

            exp_dir = output_dir / f"exp_{global_i:02d}_{config.name}"

            try:
                metrics = run_experiment(config, data, exp_dir)
                append_results_csv(metrics, results_path)
                print_evaluation_report(metrics)
            except Exception as e:
                print(f"\nERROR in experiment '{config.name}': {e}")
                traceback.print_exc()
                print("Continuing with next experiment...\n")

    print("\n\nAll experiments complete.")
    print_results_summary(results_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run MBCNN loss-function experiments")
    parser.add_argument(
        "--run", type=str, default=None,
        help="Run only the experiment with this name (for testing)"
    )
    parser.add_argument(
        "--block", type=str, default=None,
        choices=["a", "b", "c", "d"],
        help="Run only experiments from this block"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=EXPERIMENTS_DIR,
        help="Directory to store experiment artifacts"
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Re-run even if experiment already appears in results.csv"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Just print the results summary table, no training"
    )
    args = parser.parse_args()

    if args.summary:
        print_results_summary(args.output_dir / "results.csv")
        return

    configs = ALL_EXPERIMENTS

    if args.run:
        configs = [c for c in configs if c.name == args.run]
        if not configs:
            print(f"No experiment named '{args.run}' found.")
            return

    if args.block:
        configs = [c for c in configs if c.block == args.block]

    run_all_experiments(
        configs,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
