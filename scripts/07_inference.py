#!/usr/bin/env python3
"""
07_inference.py

Run inference on full rasters using trained model.

Features:
- Sliding window inference with Hann window blending
- Clip output to AOI boundary
- Save predictions as GeoTIFF + GeoJSON + interactive map

Output:
    datasets/predictions/
    ├── prediction_map.tif (GeoTIFF)
    ├── prediction_map_class2.geojson (polygons)
    └── prediction_map_map.html (interactive leaflet map)
"""

import sys
import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from src.inference.inference import full_inference_mbcnn
from src.training.mbcnn import mbcnn
from src.config import (
    S2_2025, DENSITY, AOI, LATEST_MODEL,
    PREDICTIONS_DIR, PATCH_SIZE, N_CLASSES
)

def main():
    """Run inference pipeline."""
    print("\n" + "=" * 80)
    print("🔮 INFERENCE PIPELINE")
    print("=" * 80)

    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✅ GPU available: {len(gpus)} device(s)")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"⚠️  {e}")
    else:
        print(f"\n⚠️  No GPU detected. Using CPU (slower).")

    # Check inputs
    print(f"\n📋 INPUT FILES:")
    inputs = {
        'S2': S2_2025,
        'Density': DENSITY,
        'Model': LATEST_MODEL,
        'AOI': AOI
    }

    for name, path in inputs.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {name}: {path.name}")
        if not exists:
            print(f"      ERROR: File not found!")
            return

    # Prepare output directory
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREDICTIONS_DIR / "prediction_map.tif"

    print(f"\n📁 OUTPUT DIRECTORY: {PREDICTIONS_DIR}")
    print(f"   Output file: {output_path.name}")

    # Run inference
    print(f"\n" + "=" * 80)
    print("🚀 RUNNING INFERENCE")
    print("=" * 80)

    print(f"\nModel: {LATEST_MODEL.name}")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Stride (Hann window): {PATCH_SIZE // 2}")
    print(f"N_CLASSES: {N_CLASSES}")

    try:
        # Build and load trained model
        print(f"\n📦 Building model architecture...")
        input_shapes = {
            'S2': (PATCH_SIZE, PATCH_SIZE, 10),
            'density': (PATCH_SIZE, PATCH_SIZE, 1)
        }
        model = mbcnn(CL=N_CLASSES, input_shapes=input_shapes)
        print(f"   ✅ Model architecture created")

        print(f"\n📦 Loading trained weights...")
        model.load_weights(str(LATEST_MODEL))
        print(f"   ✅ Weights loaded successfully")

        full_inference_mbcnn(
            N_CLASSES=N_CLASSES,
            image_sources=[str(S2_2025), str(DENSITY)],
            model=model,
            save_path=str(output_path),
            aoi_path=str(AOI),
            batch_size=32
        )

        print(f"\n✅ Inference completed!")
        print(f"   Output: {output_path}")

        # Summary
        if output_path.exists():
            with rasterio.open(output_path) as src:
                data = src.read(1)
                print(f"\n📊 OUTPUT STATISTICS:")
                print(f"   Shape: {data.shape}")
                print(f"   Unique classes: {np.unique(data)}")
                for cls in range(N_CLASSES):
                    count = np.sum(data == cls)
                    pct = 100 * count / (data.shape[0] * data.shape[1])
                    print(f"   Class {cls}: {count:,} pixels ({pct:.1f}%)")

    except Exception as e:
        print(f"\n❌ Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n" + "=" * 80)
    print("✨ INFERENCE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
