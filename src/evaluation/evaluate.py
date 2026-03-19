"""
evaluate.py

Standalone test-set evaluation for trained MBCNN models.
Computes per-class IoU, F1, Precision, Recall and confusion matrix on test patches.

Usage (CLI):
    python -m src.evaluation.evaluate --weights models/partidos_amba_best_overall_f1.weights.h5

Usage (Python):
    from src.evaluation.evaluate import load_test_data, evaluate_model, print_evaluation_report
    s2, den, rf, patch_ids = load_test_data()
    model = build_and_load_model(weights_path)
    metrics = evaluate_model(model, s2, den, rf)
    print_evaluation_report(metrics)
"""

import os
import sys
import argparse
import json
import csv
from pathlib import Path

import numpy as np
import rasterio as rio
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score
)

# Resolve project root so this module works when called from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PATCHES_TEST_DIR, MODELS_DIR, N_CLASSES, PATCH_SIZE,
    S2_BANDS, DEN_BANDS, CLASS_NAMES
)
from src.training.data_utils import norm_s2
from src.training.mbcnn import mbcnn


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(
    test_dir: Path = PATCHES_TEST_DIR,
    patch_size: int = PATCH_SIZE,
    den_prefix: str = "DEN",
):
    """
    Load aligned test patches (S2, DEN/DEN3, RF) by matching on patch IDs from RF files.

    Parameters
    ----------
    test_dir   : directory containing test patches
    patch_size : expected patch size (used for shape validation)
    den_prefix : file prefix for density input — "DEN" (1-band) or "DEN3" (3-band)

    Returns
    -------
        s2   : np.ndarray shape (N, patch_size, patch_size, S2_BANDS), float32
        den  : np.ndarray shape (N, patch_size, patch_size, den_bands), float32
        rf   : np.ndarray shape (N, patch_size, patch_size), int   — class labels 0/1/2
        ids  : list[str] — patch IDs loaded
    """
    test_dir = Path(test_dir)

    # Build set of patch IDs that have RF labels (ground truth)
    rf_files = sorted(test_dir.glob("RF_*.tif"))
    if not rf_files:
        raise FileNotFoundError(f"No RF_*.tif files found in {test_dir}")

    patch_ids = [f.stem.replace("RF_", "") for f in rf_files]

    s2_list, den_list, rf_list = [], [], []

    for pid in patch_ids:
        s2_path = test_dir / f"S2_{pid}.tif"
        den_path = test_dir / f"{den_prefix}_{pid}.tif"
        rf_path = test_dir / f"RF_{pid}.tif"

        if not s2_path.exists():
            print(f"  Warning: S2_{pid}.tif missing, skipping patch {pid}")
            continue
        if not den_path.exists():
            print(f"  Warning: {den_prefix}_{pid}.tif missing, skipping patch {pid}")
            continue

        # Load S2
        with rio.open(s2_path) as ds:
            s2 = ds.read().astype(np.float32)          # (bands, H, W)
            s2 = np.moveaxis(s2, 0, -1)                # (H, W, bands)
            s2 = np.nan_to_num(s2, nan=0.0)
            s2 = norm_s2(s2)

        # Load DEN
        with rio.open(den_path) as ds:
            den = ds.read().astype(np.float32)          # (bands, H, W)
            den = np.moveaxis(den, 0, -1)               # (H, W, bands)
            den = np.nan_to_num(den, nan=0.0)

        # Load RF (integer class labels)
        with rio.open(rf_path) as ds:
            rf = ds.read(1).astype(np.int32)            # (H, W)

        s2_list.append(s2)
        den_list.append(den)
        rf_list.append(rf)

    if not s2_list:
        raise RuntimeError("No valid aligned patches found.")

    print(f"Loaded {len(s2_list)} aligned test patches from {test_dir} "
          f"(den_prefix='{den_prefix}')")
    return (
        np.array(s2_list, dtype=np.float32),
        np.array(den_list, dtype=np.float32),
        np.array(rf_list, dtype=np.int32),
        patch_ids[:len(s2_list)],
    )


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_and_load_model(
    weights_path: Path,
    dropout_rate: float = 0.3,
    n_classes: int = N_CLASSES,
    patch_size: int = PATCH_SIZE,
    s2_bands: int = S2_BANDS,
    den_bands: int = DEN_BANDS,
) -> object:
    """Reconstruct MBCNN and load weights."""
    input_shapes = {
        0: (patch_size, patch_size, s2_bands),
        1: (patch_size, patch_size, den_bands),
    }
    model = mbcnn(
        CL=n_classes,
        input_shapes=input_shapes,
        dropout_rate=dropout_rate,
        batch_norm=True,
        drop_train=False,
    )
    model.load_weights(str(weights_path))
    print(f"Loaded weights: {weights_path}")
    return model


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_iou(y_true_flat, y_pred_flat, n_classes):
    """Per-class IoU from flattened arrays."""
    iou = {}
    for cls in range(n_classes):
        tp = np.sum((y_true_flat == cls) & (y_pred_flat == cls))
        fp = np.sum((y_true_flat != cls) & (y_pred_flat == cls))
        fn = np.sum((y_true_flat == cls) & (y_pred_flat != cls))
        denom = tp + fp + fn
        iou[cls] = float(tp / denom) if denom > 0 else 0.0
    return iou


def evaluate_model(
    model,
    s2: np.ndarray,
    den: np.ndarray,
    rf: np.ndarray,
    n_classes: int = N_CLASSES,
    batch_size: int = 8,
) -> dict:
    """
    Run model prediction on test patches and compute all metrics.

    Args:
        model    : compiled Keras model
        s2       : (N, H, W, 10) float32
        den      : (N, H, W, 1)  float32
        rf       : (N, H, W)     int — ground truth classes 0/1/2
        n_classes: number of classes
        batch_size: inference batch size

    Returns:
        metrics dict with all scalar metrics (suitable for CSV row)
    """
    # Run prediction in batches
    n = s2.shape[0]
    pred_probs = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_preds = model.predict(
            [s2[start:end], den[start:end]], verbose=0
        )
        pred_probs.append(batch_preds)

    pred_probs = np.concatenate(pred_probs, axis=0)       # (N, H, W, C)
    pred_classes = np.argmax(pred_probs, axis=-1)          # (N, H, W)

    # Flatten for sklearn
    y_true = rf.flatten()
    y_pred = pred_classes.flatten()

    # Per-class IoU
    iou = _compute_iou(y_true, y_pred, n_classes)
    mean_iou = float(np.mean(list(iou.values())))

    # Classification report (F1, precision, recall per class)
    labels = list(range(n_classes))
    class_name_list = [CLASS_NAMES.get(i, str(i)) for i in labels]
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_name_list,
        output_dict=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    # Build flat metrics dict
    metrics = {"mean_iou": mean_iou}
    for cls_idx, cls_name in enumerate(class_name_list):
        prefix = f"class{cls_idx}"
        metrics[f"{prefix}_iou"] = iou[cls_idx]
        if cls_name in report:
            metrics[f"{prefix}_f1"] = report[cls_name]["f1-score"]
            metrics[f"{prefix}_precision"] = report[cls_name]["precision"]
            metrics[f"{prefix}_recall"] = report[cls_name]["recall"]
        else:
            metrics[f"{prefix}_f1"] = 0.0
            metrics[f"{prefix}_precision"] = 0.0
            metrics[f"{prefix}_recall"] = 0.0

    metrics["macro_f1"] = report.get("macro avg", {}).get("f1-score", 0.0)
    metrics["weighted_f1"] = report.get("weighted avg", {}).get("f1-score", 0.0)
    metrics["accuracy"] = float(np.mean(y_true == y_pred))

    # Store confusion matrix as JSON string for CSV compatibility
    metrics["confusion_matrix"] = json.dumps(cm.tolist())
    metrics["confusion_matrix_norm"] = json.dumps(
        [[round(v, 4) for v in row] for row in cm_norm.tolist()]
    )

    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_evaluation_report(metrics: dict, class_names: dict = None):
    """Pretty-print evaluation report to stdout."""
    if class_names is None:
        class_names = CLASS_NAMES

    print("\n" + "=" * 60)
    print("  MBCNN TEST-SET EVALUATION REPORT")
    print("=" * 60)

    print(f"\n  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  Mean IoU      : {metrics['mean_iou']:.4f}")
    print(f"  Macro F1      : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1   : {metrics['weighted_f1']:.4f}")

    print("\n  Per-class metrics:")
    print(f"  {'Class':<22} {'IoU':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}")
    print("  " + "-" * 54)
    for cls_idx, cls_name in class_names.items():
        p = f"class{cls_idx}"
        print(
            f"  {cls_name:<22} "
            f"{metrics.get(p+'_iou', 0):.4f}  "
            f"{metrics.get(p+'_f1', 0):.4f}  "
            f"{metrics.get(p+'_precision', 0):.4f}  "
            f"{metrics.get(p+'_recall', 0):.4f}"
        )

    # Confusion matrix
    if "confusion_matrix" in metrics:
        cm = json.loads(metrics["confusion_matrix"])
        cm_norm = json.loads(metrics["confusion_matrix_norm"])
        print("\n  Confusion Matrix (raw counts):")
        header = "  " + " " * 18 + "".join(f"  pred_{i}" for i in range(len(cm)))
        print(header)
        for i, (row, row_norm) in enumerate(zip(cm, cm_norm)):
            name = class_names.get(i, str(i))
            counts = "  ".join(f"{v:6d}" for v in row)
            pcts = "  ".join(f"({p:.0%})" for p in row_norm)
            print(f"  true_{i} {name:<14} {counts}")
        print("\n  Confusion Matrix (row-normalized %):")
        print(header)
        for i, row_norm in enumerate(cm_norm):
            name = class_names.get(i, str(i))
            pcts = "  ".join(f"{v:6.1%}" for v in row_norm)
            print(f"  true_{i} {name:<14} {pcts}")

    print("\n" + "=" * 60)


def save_evaluation_csv(metrics: dict, output_path: Path):
    """Save metrics dict as a single-row CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Exclude large JSON fields for the summary CSV
    row = {k: v for k, v in metrics.items()
           if k not in ("confusion_matrix", "confusion_matrix_norm")}

    file_exists = output_path.exists()
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Metrics appended to {output_path}")


def save_evaluation_json(metrics: dict, output_path: Path):
    """Save full metrics dict (including confusion matrices) as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Full metrics saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate MBCNN on test patches")
    parser.add_argument(
        "--weights", type=Path,
        default=MODELS_DIR / "partidos_amba_best_overall_f1.weights.h5",
        help="Path to model weights .h5 file",
    )
    parser.add_argument(
        "--test-dir", type=Path, default=PATCHES_TEST_DIR,
        help="Directory containing test patches",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to save evaluation JSON",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    args = parser.parse_args()

    s2, den, rf, patch_ids = load_test_data(args.test_dir)
    model = build_and_load_model(args.weights, dropout_rate=args.dropout_rate)
    metrics = evaluate_model(model, s2, den, rf, batch_size=args.batch_size)
    print_evaluation_report(metrics)

    if args.output:
        save_evaluation_json(metrics, args.output)


if __name__ == "__main__":
    main()
