#!/usr/bin/env python3
"""
data_audit.py

Comprehensive data quality audit for the Techo pipeline.
Checks rasters and patches for NaN, value ranges, normalization status, and consistency.

Usage:
    python scripts/data_audit.py

Output:
    - Console summary
    - datasets/reports/data_audit_report.json (detailed report)
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import rasterio
from glob import glob
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    S2_2020, S2_2025, DENSITY, REFERENCE_LABELS, GHSL,
    PATCHES_DIR, PATCHES_TRAIN_DIR, PATCHES_VAL_DIR, PATCHES_TEST_DIR,
    DATA_AUDIT_REPORT, PATCH_SIZE, N_CLASSES, MIN_VALID_RATIO
)

class DataAudit:
    """Comprehensive data quality audit."""

    def __init__(self):
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'rasters': {},
            'patches': {},
            'issues': [],
            'recommendations': [],
            'summary': {}
        }

    def audit_raster(self, raster_path: Path, name: str) -> Dict:
        """
        Audit a single raster file.

        Returns dict with metadata, statistics, and issues.
        """
        info = {
            'name': name,
            'path': str(raster_path),
            'exists': raster_path.exists(),
            'size_mb': None,
            'crs': None,
            'shape': None,
            'dtype': None,
            'nodata': None,
            'bands': None,
            'stats': {},
            'issues': [],
            'normalization_status': None
        }

        if not raster_path.exists():
            info['issues'].append(f"File not found: {raster_path}")
            return info

        try:
            info['size_mb'] = raster_path.stat().st_size / (1024 * 1024)

            with rasterio.open(raster_path) as src:
                info['crs'] = str(src.crs)
                info['shape'] = (src.height, src.width)
                info['dtype'] = src.dtypes[0]
                info['nodata'] = src.nodata
                info['bands'] = src.count

                # Read data to check values
                print(f"  Reading {name}...", end='', flush=True)
                data = src.read()
                print(" ✓")

                # Per-band statistics
                for band_idx in range(data.shape[0]):
                    band_data = data[band_idx].astype(float)
                    valid_data = band_data[np.isfinite(band_data)]

                    if len(valid_data) == 0:
                        info['stats'][f'band_{band_idx}'] = {
                            'all_nan': True,
                            'nan_count': np.isnan(band_data).sum(),
                            'infinite_count': np.isinf(band_data).sum()
                        }
                        info['issues'].append(f"Band {band_idx}: All NaN/invalid")
                        continue

                    nan_count = np.isnan(band_data).sum()
                    inf_count = np.isinf(band_data).sum()
                    negative_count = (band_data < 0).sum()

                    valid_ratio = len(valid_data) / band_data.size

                    info['stats'][f'band_{band_idx}'] = {
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'mean': float(np.mean(valid_data)),
                        'std': float(np.std(valid_data)),
                        'nan_count': int(nan_count),
                        'nan_ratio': float(nan_count / band_data.size),
                        'infinite_count': int(inf_count),
                        'negative_count': int(negative_count),
                        'valid_ratio': float(valid_ratio)
                    }

                    # Flag issues
                    if nan_count > 0:
                        info['issues'].append(
                            f"Band {band_idx}: {nan_count:,} NaN values "
                            f"({info['stats'][f'band_{band_idx}']['nan_ratio']:.2%})"
                        )

                    if inf_count > 0:
                        info['issues'].append(
                            f"Band {band_idx}: {inf_count:,} infinite values"
                        )

                    if negative_count > 0 and 'S2' in name:
                        info['issues'].append(
                            f"Band {band_idx}: {negative_count:,} negative values "
                            f"(unexpected for Sentinel-2)"
                        )

                # Determine normalization status for S2
                if 'S2' in name and info['bands'] == 10:
                    max_val = max([
                        info['stats'][f'band_{i}']['max']
                        for i in range(info['bands'])
                        if f'band_{i}' in info['stats']
                    ])

                    if max_val > 3:
                        info['normalization_status'] = 'RAW (0-10000)'
                        info['recommendation'] = 'Will be normalized by norm_s2() guard'
                    else:
                        info['normalization_status'] = 'NORMALIZED (0-1)'

        except Exception as e:
            info['issues'].append(f"Error reading raster: {str(e)}")

        return info

    def audit_patches(self, split: str, split_dir: Path) -> Dict:
        """Audit patches in a given split directory."""
        info = {
            'split': split,
            'path': str(split_dir),
            'total_files': 0,
            'by_modality': {
                'S2': {'count': 0, 'issues': []},
                'DEN': {'count': 0, 'issues': []},
                'RF': {'count': 0, 'issues': []}
            },
            'patch_ids': {},
            'dimension_issues': [],
            'value_range_issues': [],
            'nan_issues': []
        }

        if not split_dir.exists():
            info['exists'] = False
            return info

        info['exists'] = True

        for modality in ['S2', 'DEN', 'RF']:
            pattern = str(split_dir / f"{modality}_*.tif")
            files = sorted(glob(pattern))
            info['by_modality'][modality]['count'] = len(files)
            info['total_files'] += len(files)

            if not files:
                info['by_modality'][modality]['issues'].append("No files found")
                continue

            # Sample 3 patches to check dimensions and values
            sample_files = files[::max(1, len(files) // 3)][:3]

            for filepath in sample_files:
                try:
                    with rasterio.open(filepath) as src:
                        patch_data = src.read()
                        patch_id = Path(filepath).stem

                        # Check dimensions
                        expected_shape = {
                            'S2': (10, PATCH_SIZE, PATCH_SIZE),
                            'DEN': (1, PATCH_SIZE, PATCH_SIZE),
                            'RF': (1, PATCH_SIZE, PATCH_SIZE)
                        }

                        if patch_data.shape != expected_shape[modality]:
                            info['dimension_issues'].append(
                                f"{modality} {patch_id}: shape {patch_data.shape}, "
                                f"expected {expected_shape[modality]}"
                            )

                        # Check for NaN
                        nan_count = np.isnan(patch_data).sum()
                        if nan_count > 0:
                            nan_ratio = nan_count / patch_data.size
                            if nan_ratio > MIN_VALID_RATIO:
                                info['nan_issues'].append(
                                    f"{modality} {patch_id}: {nan_ratio:.1%} NaN "
                                    f"(exceeds {MIN_VALID_RATIO:.0%} threshold)"
                                )

                        # Check value ranges
                        valid_data = patch_data[np.isfinite(patch_data)]
                        if len(valid_data) > 0:
                            min_val, max_val = np.min(valid_data), np.max(valid_data)

                            if modality == 'S2' and max_val > 3:
                                # RAW S2
                                if max_val > 10000:
                                    info['value_range_issues'].append(
                                        f"{modality} {patch_id}: max={max_val} > 10000 "
                                        f"(check for corruption)"
                                    )
                            elif modality == 'S2' and max_val <= 1:
                                # Normalized S2
                                pass

                except Exception as e:
                    info['by_modality'][modality]['issues'].append(
                        f"Error reading {Path(filepath).name}: {str(e)}"
                    )

        # Check patch_id consistency across modalities
        s2_files = set([Path(f).stem.replace('S2_', '') for f in glob(str(split_dir / "S2_*.tif"))])
        den_files = set([Path(f).stem.replace('DEN_', '') for f in glob(str(split_dir / "DEN_*.tif"))])
        rf_files = set([Path(f).stem.replace('RF_', '') for f in glob(str(split_dir / "RF_*.tif"))])

        all_ids = s2_files | den_files | rf_files
        info['patch_ids']['total_unique'] = len(all_ids)
        info['patch_ids']['mismatches'] = {
            'missing_DEN': list(s2_files - den_files)[:5],  # Show first 5
            'missing_RF': list(s2_files - rf_files)[:5],
            'extra_DEN': list(den_files - s2_files)[:5],
            'extra_RF': list(rf_files - s2_files)[:5]
        }

        return info

    def run(self):
        """Run complete audit."""
        print("\n" + "=" * 80)
        print("🔍 TECHO DATA AUDIT")
        print("=" * 80)

        # Audit rasters
        print("\n📊 RASTER AUDIT")
        print("-" * 80)

        rasters = {
            'S2 2020': S2_2020,
            'S2 2025': S2_2025,
            'Density': DENSITY,
            'Reference Labels': REFERENCE_LABELS,
            'GHSL': GHSL,
        }

        for name, path in rasters.items():
            print(f"\n{name}:")
            info = self.audit_raster(path, name)
            self.report['rasters'][name] = info

            if info['issues']:
                for issue in info['issues'][:3]:  # Show first 3 issues
                    print(f"  ⚠️  {issue}")
            else:
                print(f"  ✅ No data issues found")

            if info['normalization_status']:
                print(f"  📈 Normalization: {info['normalization_status']}")

        # Audit patches
        print("\n\n📦 PATCH AUDIT")
        print("-" * 80)

        splits = {
            'train': PATCHES_TRAIN_DIR,
            'val': PATCHES_VAL_DIR,
            'test': PATCHES_TEST_DIR
        }

        for split_name, split_path in splits.items():
            print(f"\n{split_name.upper()}:")
            info = self.audit_patches(split_name, split_path)
            self.report['patches'][split_name] = info

            if not info['exists']:
                print(f"  ❌ Directory not found: {split_path}")
                continue

            print(f"  Total files: {info['total_files']}")
            for modality in ['S2', 'DEN', 'RF']:
                count = info['by_modality'][modality]['count']
                print(f"    {modality}: {count} patches", end='')
                if info['by_modality'][modality]['issues']:
                    print(f" ⚠️  {info['by_modality'][modality]['issues'][0]}")
                else:
                    print()

            # Check consistency
            mismatches = info['patch_ids']['mismatches']
            if any([mismatches['missing_DEN'], mismatches['missing_RF'],
                    mismatches['extra_DEN'], mismatches['extra_RF']]):
                print(f"  ⚠️  Patch ID mismatches detected")
                if mismatches['missing_DEN']:
                    print(f"     Missing DEN: {len(mismatches['missing_DEN'])} patches")
                if mismatches['missing_RF']:
                    print(f"     Missing RF: {len(mismatches['missing_RF'])} patches")
            else:
                print(f"  ✅ Patch IDs consistent across modalities")

            if info['dimension_issues']:
                print(f"  ⚠️  {len(info['dimension_issues'])} dimension issues")

            if info['nan_issues']:
                print(f"  ⚠️  {len(info['nan_issues'])} patches exceed NaN threshold")

        # Generate summary and recommendations
        self.generate_summary()

        # Save report
        print("\n" + "=" * 80)
        print("💾 Saving detailed report...")
        self.save_report()
        print(f"✅ Report saved to: {DATA_AUDIT_REPORT}")

    def generate_summary(self):
        """Generate audit summary and recommendations."""
        summary = {
            'raster_status': 'OK',
            'patch_status': 'OK',
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }

        # Check rasters
        for name, info in self.report['rasters'].items():
            if info['issues']:
                for issue in info['issues']:
                    if 'All NaN' in issue or 'not found' in issue:
                        summary['critical_issues'].append(f"{name}: {issue}")
                        summary['raster_status'] = 'ERROR'
                    else:
                        summary['warnings'].append(f"{name}: {issue}")

        # Check patches
        for split, info in self.report['patches'].items():
            if not info.get('exists'):
                summary['critical_issues'].append(f"Patches {split} directory not found")
                summary['patch_status'] = 'ERROR'
                summary['recommendations'].append(
                    "Run scripts/04_grid_sampling.py and scripts/05_extract_patches.py"
                )
            else:
                if info['dimension_issues']:
                    summary['warnings'].append(
                        f"Patches {split}: {len(info['dimension_issues'])} dimension issues"
                    )
                if info['nan_issues']:
                    summary['recommendations'].append(
                        f"Patches {split}: Review and re-extract patches with high NaN ratio"
                    )

        # Normalization recommendation
        for name, info in self.report['rasters'].items():
            if 'S2' in name and info.get('normalization_status') == 'RAW (0-10000)':
                summary['recommendations'].append(
                    f"{name} is raw (0-10000). This is OK - norm_s2() will handle it."
                )

        self.report['summary'] = summary

        print("\n" + "=" * 80)
        print("📈 SUMMARY")
        print("=" * 80)
        print(f"Raster status: {summary['raster_status']}")
        print(f"Patch status: {summary['patch_status']}")

        if summary['critical_issues']:
            print(f"\n❌ Critical issues ({len(summary['critical_issues'])}):")
            for issue in summary['critical_issues']:
                print(f"   - {issue}")

        if summary['warnings']:
            print(f"\n⚠️  Warnings ({len(summary['warnings'])}):")
            for warning in summary['warnings'][:5]:  # First 5
                print(f"   - {warning}")

        if summary['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in summary['recommendations'][:5]:  # First 5
                print(f"   - {rec}")

    def save_report(self):
        """Save report to JSON."""
        DATA_AUDIT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_AUDIT_REPORT, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)

if __name__ == "__main__":
    audit = DataAudit()
    audit.run()
