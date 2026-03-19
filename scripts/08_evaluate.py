#!/usr/bin/env python3
"""
08_evaluate.py

Evaluate inference predictions against reference labels.

Features:
- Compute pixel-level accuracy and class-wise metrics
- Generate confusion matrix
- Calculate area statistics (pixels per class)
- Compare prediction map to reference labels
- Generate evaluation report

Output:
    datasets/reports/
    ├── evaluation_report.json (detailed metrics)
    └── evaluation_summary.txt (human-readable summary)
"""

import sys
import os
from pathlib import Path
import numpy as np
import rasterio
import json
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import geopandas as gpd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    REFERENCE_LABELS, PREDICTIONS_DIR, REPORTS_DIR, CLASS_NAMES
)

def evaluate_predictions():
    """Evaluate inference predictions against reference labels."""
    print("\n" + "=" * 80)
    print("📊 EVALUATION PIPELINE")
    print("=" * 80)

    # Check inputs
    prediction_map = PREDICTIONS_DIR / "prediction_map.tif"

    print(f"\n📋 INPUT FILES:")
    inputs = {
        'Reference Labels': REFERENCE_LABELS,
        'Prediction Map': prediction_map
    }

    for name, path in inputs.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {name}: {path.name}")
        if not exists:
            print(f"      ERROR: File not found!")
            return False

    print(f"\n🔄 LOADING RASTERS...")

    # Load prediction map
    with rasterio.open(str(prediction_map)) as src:
        pred_data = src.read(1)
        pred_profile = src.profile
        print(f"   ✅ Prediction map shape: {pred_data.shape}")

    # Load reference labels
    with rasterio.open(str(REFERENCE_LABELS)) as src:
        ref_data = src.read(1)
        ref_profile = src.profile
        print(f"   ✅ Reference labels shape: {ref_data.shape}")

    # Ensure same spatial extent
    min_h = min(pred_data.shape[0], ref_data.shape[0])
    min_w = min(pred_data.shape[1], ref_data.shape[1])

    pred_clipped = pred_data[:min_h, :min_w]
    ref_clipped = ref_data[:min_h, :min_w]

    print(f"   ℹ️  Using common extent: {min_h}×{min_w}")

    # Flatten for evaluation
    pred_flat = pred_clipped.flatten()
    ref_flat = ref_clipped.flatten()

    # Remove NoData values (0 or -1, depending on reference)
    valid_mask = (ref_flat > 0) & (ref_flat <= 3)
    pred_valid = pred_flat[valid_mask]
    ref_valid = ref_flat[valid_mask]

    print(f"   ℹ️  Valid pixels (>0 and <=3): {len(ref_valid):,} / {len(ref_flat):,}")

    # Compute metrics
    print(f"\n📈 COMPUTING METRICS...")

    accuracy = accuracy_score(ref_valid, pred_valid)
    print(f"   ✅ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Confusion matrix
    cm = confusion_matrix(ref_valid, pred_valid, labels=[1, 2, 3])
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"   {cm}")

    # Per-class metrics
    print(f"\n🎯 PER-CLASS METRICS:")
    report = classification_report(
        ref_valid, pred_valid,
        labels=[1, 2, 3],
        target_names=[CLASS_NAMES.get(i, f"Class {i}") for i in [1, 2, 3]],
        digits=4
    )
    print(report)

    # Area statistics
    print(f"\n📐 AREA STATISTICS (pixels):")
    for class_id in [1, 2, 3]:
        ref_count = np.sum(ref_valid == class_id)
        pred_count = np.sum(pred_valid == class_id)
        ref_pct = ref_count / len(ref_valid) * 100
        pred_pct = pred_count / len(pred_valid) * 100
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")

        print(f"\n   {class_name} (Class {class_id}):")
        print(f"      Reference: {ref_count:>12,} pixels ({ref_pct:>6.2f}%)")
        print(f"      Predicted: {pred_count:>12,} pixels ({pred_pct:>6.2f}%)")
        print(f"      Difference: {pred_count - ref_count:>11,} pixels ({(pred_pct - ref_pct):>6.2f}%)")

    # Save evaluation report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    eval_report = {
        'overall_accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'area_statistics': {}
    }

    for class_id in [1, 2, 3]:
        ref_count = np.sum(ref_valid == class_id)
        pred_count = np.sum(pred_valid == class_id)
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")

        eval_report['area_statistics'][class_name] = {
            'class_id': class_id,
            'reference_pixels': int(ref_count),
            'predicted_pixels': int(pred_count),
            'reference_percent': float(ref_count / len(ref_valid) * 100),
            'predicted_percent': float(pred_count / len(pred_valid) * 100)
        }

    report_path = REPORTS_DIR / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(eval_report, f, indent=2)

    print(f"\n✅ Evaluation report saved: {report_path}")

    # Save summary
    summary_path = REPORTS_DIR / "evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")

        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")

        f.write("Per-Class Metrics:\n")
        f.write(report + "\n\n")

        f.write("Area Statistics:\n")
        for class_id in [1, 2, 3]:
            ref_count = np.sum(ref_valid == class_id)
            pred_count = np.sum(pred_valid == class_id)
            class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")

            f.write(f"\n{class_name} (Class {class_id}):\n")
            f.write(f"  Reference: {ref_count:,} pixels ({ref_count/len(ref_valid)*100:.2f}%)\n")
            f.write(f"  Predicted: {pred_count:,} pixels ({pred_count/len(pred_valid)*100:.2f}%)\n")

    print(f"✅ Evaluation summary saved: {summary_path}")

    return True

if __name__ == "__main__":
    try:
        success = evaluate_predictions()
        if success:
            print(f"\n" + "=" * 80)
            print("✅ EVALUATION COMPLETE")
            print("=" * 80 + "\n")
        else:
            print(f"\n" + "=" * 80)
            print("❌ EVALUATION FAILED")
            print("=" * 80 + "\n")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
