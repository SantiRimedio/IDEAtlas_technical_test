"""
Data Validation for ML Model Compatibility
==========================================

This module validates downloaded data to ensure perfect compatibility with trained ML models.
Checks format, ranges, spatial alignment, and consistency with original preprocessing pipeline.

Key Features:
- Validates S2 data matches harmonize.py output format
- Ensures GHSL data matches create_ref.py processing
- Checks spatial alignment between datasets
- Validates data ranges and types for ML compatibility
- Compares with reference statistics from original pipeline
- Provides detailed validation reports

Author: Adapted for ML inference preprocessing pipeline
Compatible with: Google Colab, original preprocessing pipeline
Dependencies: rasterio, numpy, geopandas (optional)
"""

import os
import numpy as np
import rasterio
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Warning: matplotlib not available. Visualization functions disabled.")


class MLCompatibilityValidator:
    """
    Validates data compatibility with ML models and original preprocessing pipeline.
    """
    
    def __init__(self):
        """Initialize validator with expected formats from original pipeline."""
        
        # Expected S2 format (from harmonize.py + scaling)
        self.s2_expected = {
            'bands': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
            'band_count': 6,
            'dtype': 'float32',
            'value_range_raw': {'min': 0, 'max': 10000},      # Before normalization
            'value_range_normalized': {'min': 0.0, 'max': 1.0}, # After normalization
            'nodata': None,
            'scale': 10  # meters
        }
        
        # Expected GHSL format (from create_ref.py)
        self.ghsl_expected = {
            'band_count': 1,
            'dtype': 'uint8',
            'value_range': {'min': 0, 'max': 1},  # Binary mask
            'values': [0, 1],  # Only these values allowed
            'threshold': 15,   # Threshold used in create_ref.py
            'scale': 10        # Resampled to match S2
        }
        
        # Expected reference labels format (from create_ref.py)
        self.reference_expected = {
            'band_count': 1,
            'dtype': 'uint8',
            'value_range_2class': {'min': 0, 'max': 1},      # Without DUAs
            'value_range_3class': {'min': 0, 'max': 2},      # With DUAs
            'values_2class': [0, 1],     # 0=non-built, 1=built-up
            'values_3class': [0, 1, 2],  # 0=non-built, 1=built-up, 2=DUA
            'scale': 10
        }
        
        self.validation_history = []
    
    def validate_s2_data(self, 
                        filepath: str,
                        is_normalized: bool = True,
                        expected_bands: List[str] = None) -> Dict[str, Any]:
        """
        Validate Sentinel-2 data format and compatibility.
        
        This function ensures S2 data matches the output from harmonize.py.
        
        Parameters:
        -----------
        filepath : str
            Path to S2 raster file
        is_normalized : bool, default True
            Whether data should be normalized [0,1]
        expected_bands : list, optional
            Expected band list (uses default if None)
            
        Returns:
        --------
        dict
            Comprehensive validation results
        """
        print(f"🔍 Validating S2 data: {os.path.basename(filepath)}")
        
        if expected_bands is None:
            expected_bands = self.s2_expected['bands']
        
        validation = {
            'file': os.path.basename(filepath),
            'valid': True,
            'errors': [],
            'warnings': [],
            'details': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with rasterio.open(filepath) as src:
                # Basic file properties
                validation['details']['shape'] = src.shape
                validation['details']['band_count'] = src.count
                validation['details']['dtype'] = str(src.dtypes[0])
                validation['details']['crs'] = str(src.crs)
                validation['details']['nodata'] = src.nodata
                
                # Check band count
                if src.count != len(expected_bands):
                    validation['errors'].append(
                        f"Band count mismatch: expected {len(expected_bands)}, got {src.count}"
                    )
                    validation['valid'] = False
                
                # Check data type
                if src.dtypes[0] != self.s2_expected['dtype']:
                    validation['warnings'].append(
                        f"Data type is {src.dtypes[0]}, expected {self.s2_expected['dtype']}"
                    )
                
                # Check pixel size (approximate)
                pixel_size_x = abs(src.transform[0])
                pixel_size_y = abs(src.transform[4])
                expected_scale = self.s2_expected['scale']
                
                if not (expected_scale * 0.8 <= pixel_size_x <= expected_scale * 1.2):
                    validation['warnings'].append(
                        f"Pixel size X ({pixel_size_x:.1f}m) differs from expected (~{expected_scale}m)"
                    )
                
                # Sample data for range validation
                sample_data = src.read(1, masked=True)  # Read first band
                
                if sample_data.size > 0:
                    data_min = float(sample_data.min())
                    data_max = float(sample_data.max())
                    data_mean = float(sample_data.mean())
                    
                    validation['details']['data_range'] = {
                        'min': data_min,
                        'max': data_max,
                        'mean': data_mean
                    }
                    
                    # Validate value ranges
                    if is_normalized:
                        expected_range = self.s2_expected['value_range_normalized']
                        if data_min < expected_range['min'] - 0.01 or data_max > expected_range['max'] + 0.01:
                            validation['errors'].append(
                                f"Normalized data range [{data_min:.3f}, {data_max:.3f}] "
                                f"outside expected range [{expected_range['min']}, {expected_range['max']}]"
                            )
                            validation['valid'] = False
                        
                        # Check for reasonable mean (shouldn't be too close to 0 or 1)
                        if data_mean < 0.01 or data_mean > 0.9:
                            validation['warnings'].append(
                                f"Unusual mean value: {data_mean:.3f} (may indicate processing issue)"
                            )
                    else:
                        expected_range = self.s2_expected['value_range_raw']
                        if data_min < expected_range['min'] or data_max > expected_range['max']:
                            validation['warnings'].append(
                                f"Raw data range [{data_min:.0f}, {data_max:.0f}] "
                                f"outside typical range [{expected_range['min']}, {expected_range['max']}]"
                            )
                
                # Check for suspicious values
                if sample_data.size > 0:
                    # Check for all zeros
                    zero_percentage = (sample_data == 0).sum() / sample_data.size * 100
                    if zero_percentage > 50:
                        validation['warnings'].append(
                            f"High percentage of zero values: {zero_percentage:.1f}%"
                        )
                    
                    # Check for all identical values
                    unique_values = len(np.unique(sample_data.compressed()))
                    if unique_values < 10:
                        validation['warnings'].append(
                            f"Low unique value count: {unique_values} (may indicate issue)"
                        )
                
        except Exception as e:
            validation['valid'] = False
            validation['errors'].append(f"File reading error: {str(e)}")
        
        # Print validation summary
        self._print_validation_summary(validation, "Sentinel-2")
        
        # Store validation history
        self.validation_history.append(validation)
        
        return validation
    
    def validate_ghsl_data(self, 
                          filepath: str,
                          expected_threshold: int = 15) -> Dict[str, Any]:
        """
        Validate GHSL data format and compatibility with create_ref.py.
        
        Parameters:
        -----------
        filepath : str
            Path to GHSL raster file
        expected_threshold : int, default 15
            Threshold used in create_ref.py
            
        Returns:
        --------
        dict
            Comprehensive validation results
        """
        print(f"🔍 Validating GHSL data: {os.path.basename(filepath)}")
        
        validation = {
            'file': os.path.basename(filepath),
            'valid': True,
            'errors': [],
            'warnings': [],
            'details': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with rasterio.open(filepath) as src:
                # Basic file properties
                validation['details']['shape'] = src.shape
                validation['details']['band_count'] = src.count
                validation['details']['dtype'] = str(src.dtypes[0])
                validation['details']['crs'] = str(src.crs)
                
                # Check band count (should be 1 for binary mask)
                if src.count != self.ghsl_expected['band_count']:
                    validation['errors'].append(
                        f"Band count mismatch: expected {self.ghsl_expected['band_count']}, got {src.count}"
                    )
                    validation['valid'] = False
                
                # Check data type (should be uint8 for binary)
                if src.dtypes[0] not in ['uint8', 'int8']:
                    validation['warnings'].append(
                        f"Data type is {src.dtypes[0]}, expected uint8 for binary data"
                    )
                
                # Sample data for validation
                sample_data = src.read(1, masked=True)
                
                if sample_data.size > 0:
                    unique_values = np.unique(sample_data.compressed())
                    validation['details']['unique_values'] = list(unique_values.astype(int))
                    
                    # Check that only 0 and 1 values exist (binary mask)
                    expected_values = set(self.ghsl_expected['values'])
                    actual_values = set(unique_values)
                    
                    if not actual_values.issubset(expected_values):
                        unexpected = actual_values - expected_values
                        validation['errors'].append(
                            f"Unexpected values found: {list(unexpected)}. Expected only: {list(expected_values)}"
                        )
                        validation['valid'] = False
                    
                    # Calculate built-up statistics
                    total_pixels = sample_data.size
                    built_up_pixels = (sample_data == 1).sum()
                    non_built_pixels = (sample_data == 0).sum()
                    
                    built_up_percentage = (built_up_pixels / total_pixels) * 100
                    
                    validation['details']['statistics'] = {
                        'total_pixels': int(total_pixels),
                        'non_built_pixels': int(non_built_pixels),
                        'built_up_pixels': int(built_up_pixels),
                        'built_up_percentage': round(built_up_percentage, 2),
                        'threshold_used': expected_threshold
                    }
                    
                    # Sanity checks
                    if built_up_percentage < 1:
                        validation['warnings'].append(
                            f"Very low built-up percentage: {built_up_percentage:.1f}% (check AOI or threshold)"
                        )
                    elif built_up_percentage > 80:
                        validation['warnings'].append(
                            f"Very high built-up percentage: {built_up_percentage:.1f}% (check processing)"
                        )
                
        except Exception as e:
            validation['valid'] = False
            validation['errors'].append(f"File reading error: {str(e)}")
        
        # Print validation summary
        self._print_validation_summary(validation, "GHSL")
        
        # Store validation history
        self.validation_history.append(validation)
        
        return validation
    
    def validate_reference_labels(self, 
                                 filepath: str,
                                 has_duas: bool = False) -> Dict[str, Any]:
        """
        Validate reference labels format (output of create_ref.py).
        
        Parameters:
        -----------
        filepath : str
            Path to reference labels file
        has_duas : bool, default False
            Whether DUAs are included (3-class vs 2-class)
            
        Returns:
        --------
        dict
            Comprehensive validation results
        """
        print(f"🔍 Validating reference labels: {os.path.basename(filepath)}")
        
        validation = {
            'file': os.path.basename(filepath),
            'valid': True,
            'errors': [],
            'warnings': [],
            'details': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with rasterio.open(filepath) as src:
                # Basic file properties
                validation['details']['shape'] = src.shape
                validation['details']['band_count'] = src.count
                validation['details']['dtype'] = str(src.dtypes[0])
                validation['details']['has_duas'] = has_duas
                
                # Check band count
                if src.count != self.reference_expected['band_count']:
                    validation['errors'].append(
                        f"Band count mismatch: expected {self.reference_expected['band_count']}, got {src.count}"
                    )
                    validation['valid'] = False
                
                # Sample data for validation
                sample_data = src.read(1, masked=True)
                
                if sample_data.size > 0:
                    unique_values = np.unique(sample_data.compressed())
                    validation['details']['unique_values'] = list(unique_values.astype(int))
                    
                    # Check expected class values
                    if has_duas:
                        expected_values = set(self.reference_expected['values_3class'])
                        expected_range = self.reference_expected['value_range_3class']
                    else:
                        expected_values = set(self.reference_expected['values_2class'])
                        expected_range = self.reference_expected['value_range_2class']
                    
                    actual_values = set(unique_values)
                    
                    if not actual_values.issubset(expected_values):
                        unexpected = actual_values - expected_values
                        validation['errors'].append(
                            f"Unexpected class values: {list(unexpected)}. Expected: {list(expected_values)}"
                        )
                        validation['valid'] = False
                    
                    # Calculate class statistics
                    total_pixels = sample_data.size
                    class_stats = {}
                    class_names = {0: 'non-built', 1: 'built-up', 2: 'DUA'}
                    
                    for class_val in unique_values:
                        count = (sample_data == class_val).sum()
                        percentage = (count / total_pixels) * 100
                        class_stats[int(class_val)] = {
                            'name': class_names.get(int(class_val), f'class_{int(class_val)}'),
                            'pixel_count': int(count),
                            'percentage': round(percentage, 2)
                        }
                    
                    validation['details']['class_statistics'] = class_stats
                    
                    # Sanity checks
                    if 1 in class_stats and class_stats[1]['percentage'] < 1:
                        validation['warnings'].append(
                            f"Very low built-up percentage: {class_stats[1]['percentage']:.1f}%"
                        )
                    
                    if has_duas and 2 in class_stats and class_stats[2]['percentage'] > 50:
                        validation['warnings'].append(
                            f"Very high DUA percentage: {class_stats[2]['percentage']:.1f}%"
                        )
                
        except Exception as e:
            validation['valid'] = False
            validation['errors'].append(f"File reading error: {str(e)}")
        
        # Print validation summary
        self._print_validation_summary(validation, "Reference Labels")
        
        # Store validation history
        self.validation_history.append(validation)
        
        return validation
    
    def validate_spatial_alignment(self, 
                                  file_list: List[str],
                                  tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Validate spatial alignment between multiple raster files.
        
        Parameters:
        -----------
        file_list : list
            List of raster file paths to check
        tolerance : float, default 1e-6
            Tolerance for coordinate comparison
            
        Returns:
        --------
        dict
            Spatial alignment validation results
        """
        print(f"🔍 Validating spatial alignment between {len(file_list)} files...")
        
        validation = {
            'aligned': True,
            'errors': [],
            'warnings': [],
            'file_info': {},
            'comparison_matrix': {},
            'timestamp': datetime.now().isoformat()
        }
        
        if len(file_list) < 2:
            validation['warnings'].append("Need at least 2 files for alignment check")
            return validation
        
        try:
            # Collect file information
            file_info = {}
            for filepath in file_list:
                with rasterio.open(filepath) as src:
                    file_info[filepath] = {
                        'shape': src.shape,
                        'crs': str(src.crs),
                        'transform': src.transform,
                        'bounds': src.bounds,
                        'resolution': (abs(src.transform[0]), abs(src.transform[4]))
                    }
            
            validation['file_info'] = {
                os.path.basename(k): v for k, v in file_info.items()
            }
            
            # Compare each pair of files
            filenames = list(file_info.keys())
            for i in range(len(filenames)):
                for j in range(i + 1, len(filenames)):
                    file1, file2 = filenames[i], filenames[j]
                    info1, info2 = file_info[file1], file_info[file2]
                    
                    comparison_key = f"{os.path.basename(file1)} vs {os.path.basename(file2)}"
                    
                    # Check CRS
                    crs_match = info1['crs'] == info2['crs']
                    
                    # Check transform (pixel alignment)
                    transform_match = all(
                        abs(a - b) < tolerance 
                        for a, b in zip(info1['transform'], info2['transform'])
                    )
                    
                    # Check shape
                    shape_match = info1['shape'] == info2['shape']
                    
                    # Check bounds
                    bounds_match = all(
                        abs(a - b) < tolerance 
                        for a, b in zip(info1['bounds'], info2['bounds'])
                    )
                    
                    pair_aligned = crs_match and transform_match and shape_match and bounds_match
                    
                    validation['comparison_matrix'][comparison_key] = {
                        'aligned': pair_aligned,
                        'crs_match': crs_match,
                        'transform_match': transform_match,
                        'shape_match': shape_match,
                        'bounds_match': bounds_match
                    }
                    
                    if not pair_aligned:
                        validation['aligned'] = False
                        
                        # Add specific error messages
                        if not crs_match:
                            validation['errors'].append(
                                f"CRS mismatch between {os.path.basename(file1)} and {os.path.basename(file2)}: "
                                f"{info1['crs']} vs {info2['crs']}"
                            )
                        if not shape_match:
                            validation['errors'].append(
                                f"Shape mismatch between {os.path.basename(file1)} and {os.path.basename(file2)}: "
                                f"{info1['shape']} vs {info2['shape']}"
                            )
                        if not transform_match:
                            validation['errors'].append(
                                f"Transform mismatch between {os.path.basename(file1)} and {os.path.basename(file2)}"
                            )
                        if not bounds_match:
                            validation['errors'].append(
                                f"Bounds mismatch between {os.path.basename(file1)} and {os.path.basename(file2)}"
                            )
            
        except Exception as e:
            validation['aligned'] = False
            validation['errors'].append(f"Alignment check error: {str(e)}")
        
        # Print summary
        if validation['aligned']:
            print("✅ All files are spatially aligned")
        else:
            print("❌ Files are not aligned:")
            for error in validation['errors']:
                print(f"   - {error}")
        
        return validation
    
    def validate_complete_dataset(self, 
                                 s2_file: str,
                                 ghsl_file: str = None,
                                 reference_file: str = None,
                                 is_s2_normalized: bool = True,
                                 has_duas: bool = False) -> Dict[str, Any]:
        """
        Perform complete validation of a dataset for ML inference.
        
        Parameters:
        -----------
        s2_file : str
            Path to Sentinel-2 file
        ghsl_file : str, optional
            Path to GHSL file
        reference_file : str, optional
            Path to reference labels file
        is_s2_normalized : bool, default True
            Whether S2 data is normalized
        has_duas : bool, default False
            Whether reference includes DUAs
            
        Returns:
        --------
        dict
            Complete dataset validation results
        """
        print("🔍 Performing complete dataset validation...")
        print("="*50)
        
        complete_validation = {
            'dataset_valid': True,
            'individual_validations': {},
            'spatial_alignment': {},
            'summary': {},
            'timestamp': datetime.now().isoformat()
        }
        
        files_to_check = []
        
        # Validate S2 data
        s2_validation = self.validate_s2_data(s2_file, is_s2_normalized)
        complete_validation['individual_validations']['s2'] = s2_validation
        if not s2_validation['valid']:
            complete_validation['dataset_valid'] = False
        files_to_check.append(s2_file)
        
        # Validate GHSL data if provided
        if ghsl_file:
            ghsl_validation = self.validate_ghsl_data(ghsl_file)
            complete_validation['individual_validations']['ghsl'] = ghsl_validation
            if not ghsl_validation['valid']:
                complete_validation['dataset_valid'] = False
            files_to_check.append(ghsl_file)
        
        # Validate reference labels if provided
        if reference_file:
            ref_validation = self.validate_reference_labels(reference_file, has_duas)
            complete_validation['individual_validations']['reference'] = ref_validation
            if not ref_validation['valid']:
                complete_validation['dataset_valid'] = False
            files_to_check.append(reference_file)
        
        # Check spatial alignment between all files
        if len(files_to_check) > 1:
            print("\n" + "="*50)
            alignment_validation = self.validate_spatial_alignment(files_to_check)
            complete_validation['spatial_alignment'] = alignment_validation
            if not alignment_validation['aligned']:
                complete_validation['dataset_valid'] = False
        
        # Create summary
        summary = {
            'total_files_checked': len(files_to_check),
            'individual_validations_passed': sum(
                1 for v in complete_validation['individual_validations'].values() 
                if v['valid']
            ),
            'spatial_alignment_ok': complete_validation['spatial_alignment'].get('aligned', True),
            'ready_for_ml_inference': complete_validation['dataset_valid']
        }
        
        complete_validation['summary'] = summary
        
        # Print final summary
        print("\n" + "="*50)
        print("📊 COMPLETE DATASET VALIDATION SUMMARY")
        print("="*50)
        print(f"✅ Files checked: {summary['total_files_checked']}")
        print(f"✅ Individual validations passed: {summary['individual_validations_passed']}/{summary['total_files_checked']}")
        print(f"✅ Spatial alignment: {'OK' if summary['spatial_alignment_ok'] else 'FAILED'}")
        print(f"🎯 Ready for ML inference: {'YES' if summary['ready_for_ml_inference'] else 'NO'}")
        
        if not complete_validation['dataset_valid']:
            print("\n❌ Dataset validation FAILED. Check individual validation results above.")
        else:
            print("\n✅ Dataset validation PASSED. Ready for ML model inference!")
        
        print("="*50)
        
        return complete_validation
    
    def _print_validation_summary(self, validation: Dict[str, Any], data_type: str):
        """Print formatted validation summary."""
        
        print(f"📊 {data_type} Validation Results:")
        print(f"   File: {validation['file']}")
        print(f"   Status: {'✅ VALID' if validation['valid'] else '❌ INVALID'}")
        
        if 'details' in validation:
            details = validation['details']
            if 'shape' in details:
                print(f"   Shape: {details['shape']}")
            if 'band_count' in details:
                print(f"   Bands: {details['band_count']}")
            if 'dtype' in details:
                print(f"   Data type: {details['dtype']}")
            if 'data_range' in details:
                dr = details['data_range']
                print(f"   Data range: [{dr['min']:.3f}, {dr['max']:.3f}]")
            if 'statistics' in details:
                stats = details['statistics']
                if 'built_up_percentage' in stats:
                    print(f"   Built-up: {stats['built_up_percentage']:.1f}%")
            if 'class_statistics' in details:
                print("   Class distribution:")
                for class_val, stats in details['class_statistics'].items():
                    print(f"     Class {class_val} ({stats['name']}): {stats['percentage']:.1f}%")
        
        # Print errors and warnings
        if validation['errors']:
            print("   ❌ Errors:")
            for error in validation['errors']:
                print(f"     - {error}")
        
        if validation['warnings']:
            print("   ⚠️ Warnings:")
            for warning in validation['warnings']:
                print(f"     - {warning}")
        
        print()
    
    def compare_with_reference_statistics(self, 
                                        validation_results: Dict[str, Any],
                                        reference_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare validation results with reference statistics from original pipeline.
        
        Parameters:
        -----------
        validation_results : dict
            Results from validation functions
        reference_stats : dict
            Reference statistics from original pipeline
            
        Returns:
        --------
        dict
            Comparison results
        """
        print("🔍 Comparing with reference statistics...")
        
        comparison = {
            'matches_reference': True,
            'differences': [],
            'similarities': [],
            'details': {}
        }
        
        # Compare S2 statistics if available
        if 's2' in validation_results.get('individual_validations', {}):
            s2_val = validation_results['individual_validations']['s2']
            if 'data_range' in s2_val.get('details', {}):
                current_range = s2_val['details']['data_range']
                
                if 's2_range' in reference_stats:
                    ref_range = reference_stats['s2_range']
                    
                    # Compare ranges with tolerance
                    min_diff = abs(current_range['min'] - ref_range['min'])
                    max_diff = abs(current_range['max'] - ref_range['max'])
                    
                    if min_diff > 0.01 or max_diff > 0.01:
                        comparison['differences'].append(
                            f"S2 range differs: current [{current_range['min']:.3f}, {current_range['max']:.3f}] "
                            f"vs reference [{ref_range['min']:.3f}, {ref_range['max']:.3f}]"
                        )
                        comparison['matches_reference'] = False
                    else:
                        comparison['similarities'].append("S2 data ranges match reference")
        
        # Compare GHSL statistics if available
        if 'ghsl' in validation_results.get('individual_validations', {}):
            ghsl_val = validation_results['individual_validations']['ghsl']
            if 'statistics' in ghsl_val.get('details', {}):
                current_stats = ghsl_val['details']['statistics']
                
                if 'ghsl_built_up_percentage' in reference_stats:
                    ref_percentage = reference_stats['ghsl_built_up_percentage']
                    current_percentage = current_stats['built_up_percentage']
                    
                    # Compare built-up percentages with tolerance
                    percentage_diff = abs(current_percentage - ref_percentage)
                    
                    if percentage_diff > 5.0:  # 5% tolerance
                        comparison['differences'].append(
                            f"GHSL built-up percentage differs: current {current_percentage:.1f}% "
                            f"vs reference {ref_percentage:.1f}% (diff: {percentage_diff:.1f}%)"
                        )
                        comparison['matches_reference'] = False
                    else:
                        comparison['similarities'].append("GHSL built-up percentages match reference")
        
        # Print comparison results
        if comparison['matches_reference']:
            print("✅ Data matches reference statistics")
        else:
            print("⚠️ Differences found compared to reference:")
            for diff in comparison['differences']:
                print(f"   - {diff}")
        
        if comparison['similarities']:
            print("✅ Similarities with reference:")
            for sim in comparison['similarities']:
                print(f"   - {sim}")
        
        return comparison
    
    def generate_validation_report(self, 
                                  output_file: str = None) -> str:
        """
        Generate comprehensive validation report.
        
        Parameters:
        -----------
        output_file : str, optional
            Output file path (if None, prints to console)
            
        Returns:
        --------
        str
            Report content
        """
        if not self.validation_history:
            return "No validation history available"
        
        report_lines = [
            "=" * 60,
            "DATA VALIDATION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total validations performed: {len(self.validation_history)}",
            "",
            "VALIDATION SUMMARY:",
            "-" * 20
        ]
        
        # Summary statistics
        total_valid = sum(1 for v in self.validation_history if v['valid'])
        total_invalid = len(self.validation_history) - total_valid
        
        report_lines.extend([
            f"✅ Valid datasets: {total_valid}",
            f"❌ Invalid datasets: {total_invalid}",
            f"📊 Success rate: {(total_valid/len(self.validation_history))*100:.1f}%",
            ""
        ])
        
        # Individual validation results
        report_lines.append("INDIVIDUAL VALIDATION RESULTS:")
        report_lines.append("-" * 30)
        
        for i, validation in enumerate(self.validation_history, 1):
            status = "✅ VALID" if validation['valid'] else "❌ INVALID"
            report_lines.extend([
                f"{i}. {validation['file']} - {status}",
                f"   Timestamp: {validation['timestamp']}"
            ])
            
            if validation['errors']:
                report_lines.append("   Errors:")
                for error in validation['errors']:
                    report_lines.append(f"     - {error}")
            
            if validation['warnings']:
                report_lines.append("   Warnings:")
                for warning in validation['warnings']:
                    report_lines.append(f"     - {warning}")
            
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            print(f"📄 Validation report saved to: {output_file}")
        else:
            print(report_content)
        
        return report_content
    
    def create_validation_plots(self, validation_results: Dict[str, Any]) -> None:
        """
        Create visualization plots from validation results.
        
        Parameters:
        -----------
        validation_results : dict
            Results from complete dataset validation
        """
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️ Matplotlib not available. Skipping visualization.")
            return
        
        print("📊 Creating validation plots...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Dataset Validation Results', fontsize=16)
            
            # Plot 1: Validation Status Summary
            ax1 = axes[0, 0]
            individual_vals = validation_results.get('individual_validations', {})
            
            if individual_vals:
                labels = list(individual_vals.keys())
                valid_counts = [1 if v['valid'] else 0 for v in individual_vals.values()]
                invalid_counts = [0 if v['valid'] else 1 for v in individual_vals.values()]
                
                x = np.arange(len(labels))
                width = 0.35
                
                ax1.bar(x - width/2, valid_counts, width, label='Valid', color='green', alpha=0.7)
                ax1.bar(x + width/2, invalid_counts, width, label='Invalid', color='red', alpha=0.7)
                
                ax1.set_xlabel('Dataset Type')
                ax1.set_ylabel('Count')
                ax1.set_title('Validation Status by Dataset')
                ax1.set_xticks(x)
                ax1.set_xticklabels(labels)
                ax1.legend()
            
            # Plot 2: GHSL Class Distribution (if available)
            ax2 = axes[0, 1]
            if 'ghsl' in individual_vals and 'statistics' in individual_vals['ghsl'].get('details', {}):
                stats = individual_vals['ghsl']['details']['statistics']
                
                labels = ['Non-built', 'Built-up']
                sizes = [stats['non_built_pixels'], stats['built_up_pixels']]
                colors = ['lightblue', 'orange']
                
                ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax2.set_title('GHSL Class Distribution')
            else:
                ax2.text(0.5, 0.5, 'GHSL data\nnot available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes)
                ax2.set_title('GHSL Class Distribution')
            
            # Plot 3: Reference Labels Distribution (if available)
            ax3 = axes[1, 0]
            if 'reference' in individual_vals and 'class_statistics' in individual_vals['reference'].get('details', {}):
                class_stats = individual_vals['reference']['details']['class_statistics']
                
                class_names = [stats['name'] for stats in class_stats.values()]
                percentages = [stats['percentage'] for stats in class_stats.values()]
                
                ax3.bar(class_names, percentages, color=['lightblue', 'orange', 'red'][:len(class_names)])
                ax3.set_xlabel('Class')
                ax3.set_ylabel('Percentage (%)')
                ax3.set_title('Reference Labels Distribution')
                ax3.tick_params(axis='x', rotation=45)
            else:
                ax3.text(0.5, 0.5, 'Reference labels\nnot available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax3.transAxes)
                ax3.set_title('Reference Labels Distribution')
            
            # Plot 4: Spatial Alignment Matrix
            ax4 = axes[1, 1]
            spatial_alignment = validation_results.get('spatial_alignment', {})
            
            if 'comparison_matrix' in spatial_alignment:
                comparisons = spatial_alignment['comparison_matrix']
                
                if comparisons:
                    comparison_names = list(comparisons.keys())
                    alignment_status = [1 if comp['aligned'] else 0 for comp in comparisons.values()]
                    
                    colors = ['green' if status else 'red' for status in alignment_status]
                    
                    ax4.barh(comparison_names, alignment_status, color=colors, alpha=0.7)
                    ax4.set_xlabel('Aligned (1) / Not Aligned (0)')
                    ax4.set_title('Spatial Alignment Status')
                    ax4.set_xlim(0, 1.2)
                else:
                    ax4.text(0.5, 0.5, 'Single file\nNo alignment check', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax4.transAxes)
                    ax4.set_title('Spatial Alignment Status')
            
            plt.tight_layout()
            plt.show()
            
            print("✅ Validation plots created successfully")
            
        except Exception as e:
            print(f"❌ Error creating plots: {str(e)}")
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get complete validation history."""
        return self.validation_history.copy()
    
    def clear_validation_history(self) -> None:
        """Clear validation history."""
        self.validation_history = []
        print("🗑️ Validation history cleared")
    
    def save_validation_results(self, 
                               validation_results: Dict[str, Any],
                               output_file: str) -> None:
        """
        Save validation results to JSON file.
        
        Parameters:
        -----------
        validation_results : dict
            Results from validation functions
        output_file : str
            Output JSON file path
        """
        try:
            import json
            
            # Convert numpy types to Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                else:
                    return obj
            
            converted_results = convert_types(validation_results)
            
            with open(output_file, 'w') as f:
                json.dump(converted_results, f, indent=2)
            
            print(f"💾 Validation results saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to save validation results: {str(e)}")


# Convenience functions for direct use
def quick_validate_s2(filepath: str, 
                     is_normalized: bool = True) -> bool:
    """
    Quick validation of S2 file.
    
    Parameters:
    -----------
    filepath : str
        Path to S2 file
    is_normalized : bool, default True
        Whether data is normalized
        
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    validator = MLCompatibilityValidator()
    result = validator.validate_s2_data(filepath, is_normalized)
    return result['valid']


def quick_validate_ghsl(filepath: str) -> bool:
    """
    Quick validation of GHSL file.
    
    Parameters:
    -----------
    filepath : str
        Path to GHSL file
        
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    validator = MLCompatibilityValidator()
    result = validator.validate_ghsl_data(filepath)
    return result['valid']


def quick_validate_dataset(s2_file: str,
                          ghsl_file: str = None,
                          reference_file: str = None) -> bool:
    """
    Quick validation of complete dataset.
    
    Parameters:
    -----------
    s2_file : str
        Path to S2 file
    ghsl_file : str, optional
        Path to GHSL file
    reference_file : str, optional
        Path to reference file
        
    Returns:
    --------
    bool
        True if all files are valid and aligned
    """
    validator = MLCompatibilityValidator()
    result = validator.validate_complete_dataset(s2_file, ghsl_file, reference_file)
    return result['dataset_valid']


def check_ml_compatibility(files: Dict[str, str]) -> Dict[str, Any]:
    """
    Check ML model compatibility for a set of files.
    
    Parameters:
    -----------
    files : dict
        Dictionary with file types as keys and paths as values
        Example: {'s2': 'path/to/s2.tif', 'ghsl': 'path/to/ghsl.tif'}
        
    Returns:
    --------
    dict
        Compatibility check results
    """
    validator = MLCompatibilityValidator()
    
    s2_file = files.get('s2')
    ghsl_file = files.get('ghsl')
    reference_file = files.get('reference')
    
    if not s2_file:
        return {'compatible': False, 'error': 'S2 file is required'}
    
    return validator.validate_complete_dataset(s2_file, ghsl_file, reference_file)


# Example usage and testing
if __name__ == "__main__":
    print("🔍 Data Validator Test")
    print("="*40)
    
    validator = MLCompatibilityValidator()
    
    print("🧪 Testing validation functions...")
    print("   (Note: This test requires actual downloaded files)")
    
    # Example of how to use the validator
    example_usage = """
    # Example usage with real files:
    
    validator = MLCompatibilityValidator()
    
    # Validate individual files
    s2_valid = validator.validate_s2_data('s2_composite.tif', is_normalized=True)
    ghsl_valid = validator.validate_ghsl_data('ghsl_builtup.tif')
    
    # Validate complete dataset
    dataset_valid = validator.validate_complete_dataset(
        s2_file='s2_composite.tif',
        ghsl_file='ghsl_builtup.tif',
        reference_file='reference_labels.tif'
    )
    
    # Generate report
    validator.generate_validation_report('validation_report.txt')
    """
    
    print(example_usage)
    print("✅ Data validator module loaded successfully!")
    print("\n🎯 Key Functions Available:")
    print("   - MLCompatibilityValidator() - Main validation class")
    print("   - quick_validate_s2() - Quick S2 validation")
    print("   - quick_validate_ghsl() - Quick GHSL validation")
    print("   - quick_validate_dataset() - Quick complete validation")
    print("   - check_ml_compatibility() - Check ML compatibility")
    print("\n🚀 Ready to validate your downloaded data!")