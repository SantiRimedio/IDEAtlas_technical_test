#!/usr/bin/env python3
"""
06_train.py

Train MBCNN model on extracted patches.

Features:
- Multi-input architecture (S2 + DEN)
- Balanced Dice + Focal loss
- GPU support with memory growth
- Model checkpointing, CSV logging, early stopping
- Detailed metrics and evaluation

Output:
    models/latest_model.h5
    logs/training_log.csv
    Console training history and evaluation
"""

import sys
import os
from pathlib import Path
import math
import numpy as np
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set TF logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
import segmentation_models as sm

from src.training.data_utils import (
    load_data, calculate_class_weights, patch_class_proportion
)
from src.training.mbcnn import mbcnn
from src.training.losses import CombinedDiceFocalLoss
from src.config import (
    PATCHES_TRAIN_DIR, PATCHES_VAL_DIR, PATCHES_TEST_DIR,
    MODELS_DIR, LOGS_DIR, LATEST_MODEL, TRAINING_LOG,
    PATCH_SIZE, N_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    RANDOM_SEED
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training configuration."""

    # Model
    PATCH_SIZE = PATCH_SIZE
    N_CLASSES = N_CLASSES
    S2_BANDS = 10
    DEN_BANDS = 1
    DROPOUT_RATE = 0.3

    # Training
    BATCH_SIZE = BATCH_SIZE
    EPOCHS = EPOCHS
    LEARNING_RATE = LEARNING_RATE

    # Loss function
    FOCAL_ALPHA = 0.85
    FOCAL_GAMMA = 2.5
    DICE_FOCAL_RATIO = 3.0

    # Input shapes
    INPUT_SHAPES = {
        0: (PATCH_SIZE, PATCH_SIZE, S2_BANDS),
        1: (PATCH_SIZE, PATCH_SIZE, DEN_BANDS),
    }

    # Paths
    CHECKPOINT_PATH = str(MODELS_DIR / "checkpoints")
    LOG_PATH = str(LOGS_DIR)

    # Random seed
    RANDOM_SEED = RANDOM_SEED


config = TrainingConfig()

# ============================================================================
# GPU SETUP
# ============================================================================

def setup_gpu():
    """Configure GPU for training."""
    print("\n" + "=" * 80)
    print("🔧 GPU SETUP")
    print("=" * 80)

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU detected: {len(gpus)} device(s)")
            print(f"   Memory growth enabled")
        except RuntimeError as e:
            print(f"⚠️  GPU setup error: {e}")
    else:
        print("⚠️  No GPU detected. Training will use CPU.")
        print("   This will be slow. Consider using GPU if available.")

    print(f"TensorFlow version: {tf.__version__}")
    return len(gpus) > 0

# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_data():
    """Load and validate training data."""
    print("\n" + "=" * 80)
    print("📦 LOADING TRAINING DATA")
    print("=" * 80)

    print(f"\nTrain dir: {PATCHES_TRAIN_DIR}")
    print(f"Val dir:   {PATCHES_VAL_DIR}")
    print(f"Test dir:  {PATCHES_TEST_DIR}")

    # Load S2 data
    print("\n1. Loading Sentinel-2 patches...")
    train_s2 = load_data(
        str(PATCHES_TRAIN_DIR), config.PATCH_SIZE, config.PATCH_SIZE, 'S2'
    )
    val_s2 = load_data(
        str(PATCHES_VAL_DIR), config.PATCH_SIZE, config.PATCH_SIZE, 'S2'
    )
    test_s2 = load_data(
        str(PATCHES_TEST_DIR), config.PATCH_SIZE, config.PATCH_SIZE, 'S2'
    )

    print(f"   Train S2 shape: {train_s2.shape}")
    print(f"   Val S2 shape: {val_s2.shape}")
    print(f"   Test S2 shape: {test_s2.shape}")

    # Load DEN data
    print("\n2. Loading Density patches...")
    train_den = load_data(
        str(PATCHES_TRAIN_DIR), config.PATCH_SIZE, config.PATCH_SIZE, 'DEN'
    )
    val_den = load_data(
        str(PATCHES_VAL_DIR), config.PATCH_SIZE, config.PATCH_SIZE, 'DEN'
    )
    test_den = load_data(
        str(PATCHES_TEST_DIR), config.PATCH_SIZE, config.PATCH_SIZE, 'DEN'
    )

    print(f"   Train DEN shape: {train_den.shape}")
    print(f"   Val DEN shape: {val_den.shape}")
    print(f"   Test DEN shape: {test_den.shape}")

    # Load labels
    print("\n3. Loading reference labels...")
    train_labels = load_data(
        str(PATCHES_TRAIN_DIR), config.PATCH_SIZE, config.PATCH_SIZE, 'RF',
        n_classes=config.N_CLASSES
    )
    val_labels = load_data(
        str(PATCHES_VAL_DIR), config.PATCH_SIZE, config.PATCH_SIZE, 'RF',
        n_classes=config.N_CLASSES
    )
    test_labels = load_data(
        str(PATCHES_TEST_DIR), config.PATCH_SIZE, config.PATCH_SIZE, 'RF',
        n_classes=config.N_CLASSES
    )

    print(f"   Train labels shape: {train_labels.shape}")
    print(f"   Val labels shape: {val_labels.shape}")
    print(f"   Test labels shape: {test_labels.shape}")

    # Clean NaN/Inf
    print("\n4. Cleaning data...")
    for arr in [train_s2, val_s2, test_s2, train_den, val_den, test_den]:
        arr[:] = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    for arr in [train_labels, val_labels, test_labels]:
        arr[:] = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    print("   ✅ Data cleaned")

    # Fix cardinality
    print("\n5. Fixing data cardinality...")
    min_train = min(len(train_s2), len(train_den), len(train_labels))
    min_val = min(len(val_s2), len(val_den), len(val_labels))
    min_test = min(len(test_s2), len(test_den), len(test_labels))

    train_s2, train_den, train_labels = train_s2[:min_train], train_den[:min_train], train_labels[:min_train]
    val_s2, val_den, val_labels = val_s2[:min_val], val_den[:min_val], val_labels[:min_val]
    test_s2, test_den, test_labels = test_s2[:min_test], test_den[:min_test], test_labels[:min_test]

    print(f"   Train: {len(train_s2)} samples")
    print(f"   Val: {len(val_s2)} samples")
    print(f"   Test: {len(test_s2)} samples")

    # Calculate class weights
    print("\n6. Calculating class weights...")
    class_weights = calculate_class_weights(train_labels)
    print(f"   Class weights: {class_weights}")

    # Class distribution
    print("\n7. Training set class distribution:")
    train_classes = np.argmax(train_labels, axis=-1).flatten()
    for cls in range(config.N_CLASSES):
        count = np.sum(train_classes == cls)
        pct = 100 * count / len(train_classes)
        print(f"   Class {cls}: {count:,} pixels ({pct:.1f}%)")

    return {
        'train': ([train_s2, train_den], train_labels),
        'val': ([val_s2, val_den], val_labels),
        'test': ([test_s2, test_den], test_labels),
        'class_weights': class_weights
    }

# ============================================================================
# MODEL CREATION
# ============================================================================

def create_model(class_weights):
    """Create and compile MBCNN model."""
    print("\n" + "=" * 80)
    print("🧠 MODEL CREATION")
    print("=" * 80)

    print(f"\nArchitecture: Multi-Branch CNN (MBCNN)")
    print(f"Input shapes: {config.INPUT_SHAPES}")
    print(f"Classes: {config.N_CLASSES}")

    # Create model
    print("\nBuilding model...")
    model = mbcnn(
        CL=config.N_CLASSES,
        input_shapes=config.INPUT_SHAPES,
        dropout_rate=config.DROPOUT_RATE,
        batch_norm=True,
        drop_train=True
    )

    print(f"✅ Model created")
    print(f"   Parameters: {model.count_params():,}")

    # Create loss function
    print(f"\nConfiguring loss function:")
    print(f"   Focal alpha: {config.FOCAL_ALPHA}")
    print(f"   Focal gamma: {config.FOCAL_GAMMA}")
    print(f"   Dice/Focal ratio: {config.DICE_FOCAL_RATIO}")

    dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
    focal_loss = sm.losses.CategoricalFocalLoss(
        alpha=config.FOCAL_ALPHA,
        gamma=config.FOCAL_GAMMA
    )

    combined_loss = dice_loss + (config.DICE_FOCAL_RATIO * focal_loss)

    # Compile
    print(f"\nCompiling model...")
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss=combined_loss,
        metrics=[
            sm.metrics.FScore(name='f1'),
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(class_id=2, name='class2_precision'),
            tf.keras.metrics.Recall(class_id=2, name='class2_recall')
        ]
    )

    print(f"✅ Model compiled")
    print(f"   Learning rate: {config.LEARNING_RATE}")

    return model

# ============================================================================
# CALLBACKS
# ============================================================================

def setup_callbacks():
    """Setup training callbacks."""
    print("\n" + "=" * 80)
    print("📋 CALLBACKS")
    print("=" * 80)

    # Create directories
    Path(config.CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
    Path(config.LOG_PATH).mkdir(parents=True, exist_ok=True)

    # Clean old files
    for f in Path(config.CHECKPOINT_PATH).glob("*.h5"):
        f.unlink()
    for f in Path(config.LOG_PATH).glob("*.csv"):
        f.unlink()

    print(f"\nCheckpoint dir: {config.CHECKPOINT_PATH}")
    print(f"Log dir: {config.LOG_PATH}")

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(config.CHECKPOINT_PATH, "best_model.weights.h5"),
            monitor="val_f1",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        CSVLogger(
            os.path.join(config.LOG_PATH, "training_log.csv")
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_f1",
            mode="max",
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("\n✅ Callbacks configured:")
    print("   • ModelCheckpoint (best model by val_f1)")
    print("   • CSVLogger")
    print("   • ReduceLROnPlateau")
    print("   • EarlyStopping")

    return callbacks

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, data, callbacks):
    """Train the model."""
    print("\n" + "=" * 80)
    print("🚀 TRAINING")
    print("=" * 80)

    train_images, train_labels = data['train']
    val_images, val_labels = data['val']

    print(f"\nTraining configuration:")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Epochs: {config.EPOCHS}")
    print(f"   Training samples: {len(train_labels)}")
    print(f"   Validation samples: {len(val_labels)}")

    print(f"\n{'='*80}")

    history = model.fit(
        train_images,
        train_labels,
        batch_size=config.BATCH_SIZE,
        steps_per_epoch=math.ceil(len(train_labels) / config.BATCH_SIZE),
        epochs=config.EPOCHS,
        callbacks=callbacks,
        validation_data=(val_images, val_labels),
        validation_steps=math.ceil(len(val_labels) / config.BATCH_SIZE),
        verbose=1
    )

    print(f"\n{'='*80}")
    print("✅ Training completed!")

    return history

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, data):
    """Evaluate model on test set."""
    print("\n" + "=" * 80)
    print("📊 EVALUATION ON TEST SET")
    print("=" * 80)

    test_images, test_labels = data['test']

    print(f"\nGenerating predictions on {len(test_labels)} test samples...")
    predictions = model.predict(test_images, batch_size=config.BATCH_SIZE, verbose=0)

    pred_classes = np.argmax(predictions, axis=-1)
    true_classes = np.argmax(test_labels, axis=-1)

    pred_flat = pred_classes.flatten()
    true_flat = true_classes.flatten()

    # Overall metrics
    print(f"\n{'OVERALL METRICS':^50}")
    print(f"{'='*50}")

    total = len(true_flat)
    accuracy = np.mean(pred_flat == true_flat)

    print(f"Samples: {total:,}")
    print(f"Accuracy: {accuracy:.4f}")

    # Confusion matrix
    conf_matrix = confusion_matrix(true_flat, pred_flat)
    print(f"\nConfusion Matrix:")
    print(conf_matrix)

    # Per-class metrics
    print(f"\n{'PER-CLASS METRICS':^50}")
    print(f"{'='*50}")

    class_report = classification_report(
        true_flat, pred_flat,
        target_names=[f"Class_{i}" for i in range(config.N_CLASSES)],
        output_dict=True
    )

    for cls in range(config.N_CLASSES):
        metrics = class_report[f"Class_{cls}"]
        print(f"\nClass {cls}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1-score']:.4f}")

    # Class 2 focus
    print(f"\n{'CLASS 2 (SLUMS) ANALYSIS':^50}")
    print(f"{'='*50}")

    true_c2 = np.sum(true_flat == 2)
    pred_c2 = np.sum(pred_flat == 2)
    correct_c2 = np.sum((true_flat == 2) & (pred_flat == 2))

    prec_c2 = correct_c2 / pred_c2 if pred_c2 > 0 else 0
    rec_c2 = correct_c2 / true_c2 if true_c2 > 0 else 0
    f1_c2 = 2 * (prec_c2 * rec_c2) / (prec_c2 + rec_c2) if (prec_c2 + rec_c2) > 0 else 0

    print(f"True pixels: {true_c2:,}")
    print(f"Predicted pixels: {pred_c2:,}")
    print(f"Correctly identified: {correct_c2:,}")
    print(f"\nPrecision: {prec_c2:.4f}")
    print(f"Recall: {rec_c2:.4f}")
    print(f"F1-Score: {f1_c2:.4f}")

    return {
        'accuracy': accuracy,
        'class2_f1': f1_c2,
        'class_report': class_report
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline."""
    print("\n" + "=" * 80)
    print("TECHO TRAINING PIPELINE")
    print("=" * 80)

    # Setup
    has_gpu = setup_gpu()

    # Load data
    data = load_training_data()

    # Create model
    model = create_model(data['class_weights'])

    # Setup callbacks
    callbacks = setup_callbacks()

    # Train
    history = train_model(model, data, callbacks)

    # Evaluate
    metrics = evaluate_model(model, data)

    # Save final model
    print(f"\n" + "=" * 80)
    print("💾 SAVING MODEL")
    print("=" * 80)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(LATEST_MODEL))
    print(f"✅ Model saved to: {LATEST_MODEL}")

    # Summary
    print(f"\n" + "=" * 80)
    print("✨ TRAINING SUMMARY")
    print("=" * 80)
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Class 2 F1: {metrics['class2_f1']:.4f}")
    print(f"Model: {LATEST_MODEL}")
    print(f"Logs: {LOGS_DIR}/training_log.csv")
    print("=" * 80)

if __name__ == "__main__":
    main()
