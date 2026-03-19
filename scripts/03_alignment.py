#!/usr/bin/env python3
"""
03_alignment.py

Validate spatial alignment of rasters used in the pipeline.

Checks:
- CRS consistency
- Resolution consistency
- Bounds overlap
- Grid alignment

Output:
    - Console report
    - datasets/reports/alignment_report.json
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.validate_alignment import RasterAlignmentValidator
from src.config import S2_2020, DENSITY, REFERENCE_LABELS, ALIGNMENT_REPORT

def main():
    """Validate raster alignment."""
    print("\n" + "=" * 80)
    print("🔍 RASTER ALIGNMENT VALIDATION")
    print("=" * 80)

    # Rasters to validate
    rasters = [
        str(S2_2020),
        str(DENSITY),
        str(REFERENCE_LABELS),
    ]

    # Check existence
    print("\nChecking file existence:")
    for raster in rasters:
        path = Path(raster)
        status = "✅" if path.exists() else "❌"
        print(f"  {status} {path.name}")

    # Validate
    validator = RasterAlignmentValidator(verbose=True)
    success = validator.validate_rasters(rasters)

    # Save report
    ALIGNMENT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    validator.save_report(str(ALIGNMENT_REPORT))

    # Exit code
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
