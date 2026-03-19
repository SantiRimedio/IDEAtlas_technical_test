"""
config.py

Central configuration module for the Techo pipeline.
All paths are defined relative to PROJECT_ROOT for reproducibility across machines.

Usage:
    from src.config import *
    print(S2_2020)  # /Users/santi/DataspellProjects/Techo/datasets/raw/s2_2020_partidos_ambas-006.tif
"""

from pathlib import Path

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets"
RAW_DIR = DATA_DIR / "raw"
PATCHES_DIR = DATA_DIR / "patches"
PREDICTIONS_DIR = DATA_DIR / "predictions"
GRIDS_DIR = DATA_DIR / "grids"
REPORTS_DIR = DATA_DIR / "reports"
BOUNDARIES_DIR = PROJECT_ROOT / "data" / "boundaries"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# ============================================================================
# RASTER DATA PATHS (input data for pipeline)
# ============================================================================

# Sentinel-2 composites
S2_2020 = RAW_DIR / "s2_2020_partidos_ambas-006.tif"
S2_2025 = RAW_DIR / "s2_2025_partidos_ambas-005.tif"

# Building density (morphometric)
DENSITY = RAW_DIR / "s2_2025_building_density_updated.tif"
DENSITY_MULTISCALE_DIR = RAW_DIR / "density_multiscale_AMBA"

# Reference labels (ground truth for training)
REFERENCE_LABELS = RAW_DIR / "reference_labels_partidos_AMBA.tif"

# GHSL built-up data
GHSL = RAW_DIR / "ghsl_builtup_partidos_amba.tif"

# ============================================================================
# BOUNDARY/AOI DATA
# ============================================================================

AOI = BOUNDARIES_DIR / "partidos_amba_IA_BID_2025.geojson"
AOI_RENABAP = BOUNDARIES_DIR / "renabap-datos-barrios.geojson"

# ============================================================================
# INTERMEDIATE OUTPUTS (grid, patches)
# ============================================================================

# Grid file (output from notebook 04)
GRID_BALANCED = GRIDS_DIR / "grid_balanced.geojson"

# Patches directory structure:
# PATCHES_DIR/
#   ├── train/  [S2_*.tif, DEN_*.tif, RF_*.tif]
#   ├── val/    [S2_*.tif, DEN_*.tif, RF_*.tif]
#   └── test/   [S2_*.tif, DEN_*.tif, RF_*.tif]

PATCHES_TRAIN_DIR = PATCHES_DIR / "train"
PATCHES_VAL_DIR = PATCHES_DIR / "val"
PATCHES_TEST_DIR = PATCHES_DIR / "test"

# Large grid patches (full AOI, no undersampling, ~8 000 patches)
PATCHES_LARGE_DIR = DATA_DIR / "patches_large"
PATCHES_LARGE_TRAIN_DIR = PATCHES_LARGE_DIR / "train"
PATCHES_LARGE_VAL_DIR = PATCHES_LARGE_DIR / "val"
PATCHES_LARGE_TEST_DIR = PATCHES_LARGE_DIR / "test"

# ============================================================================
# MODEL PATHS
# ============================================================================

# Latest trained model (will be updated after training)
LATEST_MODEL = MODELS_DIR / "latest_model.weights.h5"

# Reference pre-trained models
PRETRAINED_BUENOS_AIRES = MODELS_DIR / "buenos_aires_s2_morph_mbcnn.weights.h5"
PRETRAINED_PARTIDOS = MODELS_DIR / "partidos_amba_best_overall_f1.weights.h5"
PRETRAINED_PRECISION = MODELS_DIR / "partidos_amba_best_precision.weights.h5"
PRETRAINED_RECALL = MODELS_DIR / "partidos_amba_best_recall.weights.h5"

# ============================================================================
# PREDICTIONS & OUTPUTS
# ============================================================================

PREDICTIONS_DIR_2025 = PREDICTIONS_DIR / "2025"
PREDICTIONS_DIR_2020 = PREDICTIONS_DIR / "2020"
PREDICTIONS_CHANGE = PREDICTIONS_DIR / "change_detection"

# ============================================================================
# REPORTS & LOGS
# ============================================================================

ALIGNMENT_REPORT = REPORTS_DIR / "alignment_report.json"
DATA_AUDIT_REPORT = REPORTS_DIR / "data_audit_report.json"
TRAINING_LOG = LOGS_DIR / "training_log.csv"
TRAINING_HISTORY = LOGS_DIR / "training_history.pkl"

# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

# Data format & validation
PATCH_SIZE = 128
N_CLASSES = 3
S2_BANDS = 10
DEN_BANDS = 1

# Training hyperparameters
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
MIN_VALID_RATIO = 0.95  # Minimum valid pixel ratio for patch validation

# Class labels
CLASS_NAMES = {
    0: "background",
    1: "formal_urban",
    2: "informal_settlement"
}

# Default random seed
RANDOM_SEED = 42

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories():
    """Ensure all required directories exist."""
    dirs = [
        RAW_DIR, PATCHES_DIR, PATCHES_TRAIN_DIR, PATCHES_VAL_DIR, PATCHES_TEST_DIR,
        PREDICTIONS_DIR, PREDICTIONS_DIR_2025, PREDICTIONS_DIR_2020, PREDICTIONS_CHANGE,
        GRIDS_DIR, REPORTS_DIR, BOUNDARIES_DIR, MODELS_DIR, LOGS_DIR, SCRIPTS_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def get_patch_dirs(split: str = None):
    """
    Get patch directories for a given split (train/val/test) or all.

    Args:
        split: 'train', 'val', 'test', or None for all

    Returns:
        dict or Path
    """
    if split == 'train':
        return PATCHES_TRAIN_DIR
    elif split == 'val':
        return PATCHES_VAL_DIR
    elif split == 'test':
        return PATCHES_TEST_DIR
    else:
        return {
            'train': PATCHES_TRAIN_DIR,
            'val': PATCHES_VAL_DIR,
            'test': PATCHES_TEST_DIR
        }

# ============================================================================
# VALIDATION
# ============================================================================

def validate_required_files():
    """Check that all required input files exist."""
    required_files = {
        'S2 2020': S2_2020,
        'S2 2025': S2_2025,
        'Density': DENSITY,
        'Reference Labels': REFERENCE_LABELS,
        'GHSL': GHSL,
        'AOI': AOI,
    }

    missing = []
    for name, path in required_files.items():
        if not path.exists():
            missing.append(f"{name}: {path}")

    if missing:
        print("❌ Missing required files:")
        for item in missing:
            print(f"   - {item}")
        return False

    print("✅ All required input files found")
    return True

# Initialize directories on import
ensure_directories()
