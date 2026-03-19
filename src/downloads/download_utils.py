"""
Common Utilities for GEE Data Download Operations
================================================

This module provides common utilities and helper functions for all data download scripts.
Includes robust error handling, progress tracking, spatial validation, and file management.

Key Features:
- AOI geometry creation and validation
- Robust download functions with retry logic
- Spatial alignment checking between datasets
- Progress tracking and logging
- File validation utilities
- Error handling and recovery

Author: Adapted for ML inference preprocessing pipeline
Compatible with: Google Colab, all download scripts
Dependencies: earthengine-api, geopandas, rasterio, shapely
"""

import ee
import os
import time
import requests
import zipfile
import json
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from io import BytesIO
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import geopandas as gpd
    from shapely.geometry import box, Polygon
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("⚠️ Warning: geopandas not available. Some AOI functions will be limited.")


class DownloadManager:
    """
    Manages download operations with robust error handling and progress tracking.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: int = 5):
        """
        Initialize download manager.
        
        Parameters:
        -----------
        max_retries : int, default 3
            Maximum number of download retry attempts
        retry_delay : int, default 5
            Delay between retry attempts (seconds)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.download_history = []
        
    def download_with_retry(self, 
                           image: ee.Image,
                           aoi_geom: ee.Geometry,
                           filename: str,
                           scale: int,
                           max_pixels: int = 1e9) -> str:
        """
        Download Earth Engine image with automatic retry on failure.
        
        Parameters:
        -----------
        image : ee.Image
            Image to download
        aoi_geom : ee.Geometry
            Area of interest geometry
        filename : str
            Output filename
        scale : int
            Spatial resolution in meters
        max_pixels : int, default 1e9
            Maximum number of pixels to download
            
        Returns:
        --------
        str
            Path to downloaded file
        """
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                print(f"📥 Download attempt {attempt + 1}/{self.max_retries}")
                
                # Get download URL
                url = image.getDownloadURL({
                    'scale': scale,
                    'crs': 'EPSG:4326',
                    'region': aoi_geom,
                    'format': 'GEO_TIFF',
                    'maxPixels': max_pixels
                })
                
                # Download with progress tracking
                filepath = self._download_from_url(url, filename)
                
                # Validate download
                if self._validate_download(filepath):
                    download_time = time.time() - start_time
                    
                    # Record successful download
                    self._record_download(filepath, download_time, attempt + 1)
                    
                    print(f"✅ Download successful: {filepath}")
                    print(f"⏱️ Download time: {download_time:.1f} seconds")
                    return filepath
                else:
                    raise Exception("Downloaded file validation failed")
                    
            except Exception as e:
                last_error = e
                print(f"❌ Download attempt {attempt + 1} failed: {str(e)}")
                
                # Clean up failed download
                if os.path.exists(filename):
                    os.remove(filename)
                
                if attempt < self.max_retries - 1:
                    print(f"⏳ Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    print("❌ All download attempts failed")
                    
        # If we get here, all attempts failed
        raise Exception(f"Download failed after {self.max_retries} attempts. Last error: {str(last_error)}")
    
    def _download_from_url(self, url: str, filename: str) -> str:
        """Download file from URL with progress tracking."""
        
        response = requests.get(url, stream=True, timeout=300)
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.reason}")
        
        # Get content length for progress tracking
        total_size = int(response.headers.get('content-length', 0))
        
        print(f"📦 Downloading {total_size / (1024*1024):.1f} MB...")
        
        # Handle zip files
        if response.headers.get('content-type') == 'application/zip':
            return self._extract_from_zip(response.content, filename)
        else:
            # Direct download
            downloaded_size = 0
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Simple progress indication
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        if downloaded_size % (1024 * 1024) == 0:  # Update every MB
                            print(f"📊 Progress: {progress:.1f}%")
            
            return filename
    
    def _extract_from_zip(self, zip_content: bytes, target_filename: str) -> str:
        """Extract tiff file from zip archive."""
        
        with zipfile.ZipFile(BytesIO(zip_content)) as zip_file:
            # Find .tif files
            tif_files = [f for f in zip_file.namelist() if f.endswith('.tif')]
            
            if not tif_files:
                raise Exception("No .tif files found in archive")
            
            # Extract first .tif file
            zip_file.extract(tif_files[0], '.')
            
            # Rename to target filename
            if tif_files[0] != target_filename:
                os.rename(tif_files[0], target_filename)
            
            return target_filename
    
    def _validate_download(self, filepath: str) -> bool:
        """Validate downloaded file."""
        try:
            # Check file exists and has size
            if not os.path.exists(filepath):
                return False
            
            if os.path.getsize(filepath) == 0:
                return False
            
            # Try to open with rasterio
            with rasterio.open(filepath) as src:
                # Check basic properties
                if src.count == 0 or src.width == 0 or src.height == 0:
                    return False
                
                # Try to read a small sample
                sample = src.read(1, window=rasterio.windows.Window(0, 0, min(100, src.width), min(100, src.height)))
                
            return True
            
        except Exception:
            return False
    
    def _record_download(self, filepath: str, download_time: float, attempts: int):
        """Record download statistics."""
        
        try:
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            
            record = {
                'filename': os.path.basename(filepath),
                'file_size_mb': round(file_size, 2),
                'download_time_seconds': round(download_time, 1),
                'attempts': attempts,
                'timestamp': datetime.now().isoformat()
            }
            
            self.download_history.append(record)
            
        except Exception:
            pass  # Don't fail download for logging issues
    
    def get_download_history(self) -> List[Dict]:
        """Get download history."""
        return self.download_history.copy()


class AOIManager:
    """
    Manages Area of Interest (AOI) creation and validation.
    """
    
    @staticmethod
    def create_aoi_geometry(aoi_input: Union[Tuple, List, str, Dict]) -> ee.Geometry:
        """
        Create ee.Geometry from various AOI input formats.
        
        Parameters:
        -----------
        aoi_input : various types
            AOI specification in one of these formats:
            - Tuple/List: (minx, miny, maxx, maxy) bounding box
            - String: path to geojson/shapefile
            - Dict: GeoJSON geometry dict
            - ee.Geometry: passed through unchanged
            
        Returns:
        --------
        ee.Geometry
            Earth Engine geometry object
        """
        if isinstance(aoi_input, ee.Geometry):
            return aoi_input
        
        elif isinstance(aoi_input, (tuple, list)) and len(aoi_input) == 4:
            # Bounding box: (minx, miny, maxx, maxy)
            return ee.Geometry.Rectangle(list(aoi_input))
        
        elif isinstance(aoi_input, str):
            # File path
            if not os.path.exists(aoi_input):
                raise FileNotFoundError(f"AOI file not found: {aoi_input}")
            
            if not GEOPANDAS_AVAILABLE:
                raise ImportError("geopandas required for file-based AOI")
            
            gdf = gpd.read_file(aoi_input)
            if len(gdf) == 0:
                raise ValueError("AOI file contains no geometries")
            
            # Use first geometry or union all geometries
            if len(gdf) == 1:
                geom = gdf.geometry.iloc[0]
            else:
                geom = gdf.geometry.unary_union
            
            return ee.Geometry(geom.__geo_interface__)
        
        elif isinstance(aoi_input, dict):
            # GeoJSON geometry
            return ee.Geometry(aoi_input)
        
        else:
            raise ValueError(f"Unsupported AOI input type: {type(aoi_input)}")
    
    @staticmethod
    def validate_aoi_geometry(aoi_geom: ee.Geometry) -> Dict[str, Any]:
        """
        Validate AOI geometry and return information.
        
        Parameters:
        -----------
        aoi_geom : ee.Geometry
            Geometry to validate
            
        Returns:
        --------
        dict
            Validation results and geometry information
        """
        try:
            # Get geometry info
            bounds = aoi_geom.bounds().getInfo()
            area = aoi_geom.area().getInfo()
            
            # Calculate dimensions
            width_deg = bounds['coordinates'][0][2][0] - bounds['coordinates'][0][0][0]
            height_deg = bounds['coordinates'][0][2][1] - bounds['coordinates'][0][0][1]
            
            # Rough area calculation (not precise for large areas)
            area_km2 = area / 1e6
            
            validation = {
                'valid': True,
                'bounds': {
                    'minx': bounds['coordinates'][0][0][0],
                    'miny': bounds['coordinates'][0][0][1],
                    'maxx': bounds['coordinates'][0][2][0],
                    'maxy': bounds['coordinates'][0][2][1]
                },
                'width_degrees': width_deg,
                'height_degrees': height_deg,
                'area_km2': area_km2,
                'warnings': []
            }
            
            # Add warnings for potential issues
            if area_km2 > 10000:  # > 10,000 km²
                validation['warnings'].append("Large area may cause download timeouts")
            
            if width_deg > 2 or height_deg > 2:  # > 2 degrees
                validation['warnings'].append("Large extent may require tiling")
            
            if area_km2 < 1:  # < 1 km²
                validation['warnings'].append("Very small area - check coordinate units")
            
            return validation
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'bounds': None,
                'area_km2': None,
                'warnings': []
            }
    
    @staticmethod
    def create_tiles_from_aoi(aoi_geom: ee.Geometry, 
                             tile_size_deg: float = 0.5) -> List[ee.Geometry]:
        """
        Split large AOI into smaller tiles for processing.
        
        Parameters:
        -----------
        aoi_geom : ee.Geometry
            Large AOI to split
        tile_size_deg : float, default 0.5
            Tile size in degrees
            
        Returns:
        --------
        list
            List of tile geometries
        """
        bounds = aoi_geom.bounds().getInfo()
        
        minx = bounds['coordinates'][0][0][0]
        miny = bounds['coordinates'][0][0][1]
        maxx = bounds['coordinates'][0][2][0]
        maxy = bounds['coordinates'][0][2][1]
        
        tiles = []
        
        y = miny
        while y < maxy:
            x = minx
            while x < maxx:
                tile_bounds = [
                    x,
                    y,
                    min(x + tile_size_deg, maxx),
                    min(y + tile_size_deg, maxy)
                ]
                
                tile_geom = ee.Geometry.Rectangle(tile_bounds)
                
                # Only include tiles that intersect with original AOI
                if aoi_geom.intersects(tile_geom).getInfo():
                    tiles.append(tile_geom.intersection(aoi_geom))
                
                x += tile_size_deg
            y += tile_size_deg
        
        return tiles


class SpatialValidator:
    """
    Validates spatial alignment and compatibility between datasets.
    """
    
    @staticmethod
    def check_spatial_alignment(file1: str, file2: str, tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Check spatial alignment between two raster files.
        
        Parameters:
        -----------
        file1 : str
            Path to first raster file
        file2 : str
            Path to second raster file
        tolerance : float, default 1e-6
            Tolerance for coordinate comparison
            
        Returns:
        --------
        dict
            Alignment check results
        """
        try:
            with rasterio.open(file1) as src1, rasterio.open(file2) as src2:
                
                # Check CRS
                crs_match = src1.crs == src2.crs
                
                # Check transform (pixel alignment)
                transform_match = all(
                    abs(a - b) < tolerance 
                    for a, b in zip(src1.transform, src2.transform)
                )
                
                # Check bounds
                bounds1 = src1.bounds
                bounds2 = src2.bounds
                bounds_match = all(
                    abs(a - b) < tolerance 
                    for a, b in zip(bounds1, bounds2)
                )
                
                # Check shape
                shape_match = src1.shape == src2.shape
                
                # Calculate overlap
                overlap_bounds = (
                    max(bounds1.left, bounds2.left),
                    max(bounds1.bottom, bounds2.bottom),
                    min(bounds1.right, bounds2.right),
                    min(bounds1.top, bounds2.top)
                )
                
                overlap_area = 0
                if overlap_bounds[2] > overlap_bounds[0] and overlap_bounds[3] > overlap_bounds[1]:
                    overlap_area = (overlap_bounds[2] - overlap_bounds[0]) * (overlap_bounds[3] - overlap_bounds[1])
                
                area1 = (bounds1.right - bounds1.left) * (bounds1.top - bounds1.bottom)
                area2 = (bounds2.right - bounds2.left) * (bounds2.top - bounds2.bottom)
                
                overlap_percentage = (overlap_area / min(area1, area2)) * 100 if min(area1, area2) > 0 else 0
                
                alignment_result = {
                    'aligned': crs_match and transform_match and bounds_match and shape_match,
                    'crs_match': crs_match,
                    'transform_match': transform_match,
                    'bounds_match': bounds_match,
                    'shape_match': shape_match,
                    'overlap_percentage': overlap_percentage,
                    'file1_info': {
                        'crs': str(src1.crs),
                        'shape': src1.shape,
                        'bounds': bounds1,
                        'transform': src1.transform
                    },
                    'file2_info': {
                        'crs': str(src2.crs),
                        'shape': src2.shape,
                        'bounds': bounds2,
                        'transform': src2.transform
                    }
                }
                
                return alignment_result
                
        except Exception as e:
            return {
                'aligned': False,
                'error': str(e),
                'crs_match': False,
                'transform_match': False,
                'bounds_match': False,
                'shape_match': False,
                'overlap_percentage': 0
            }
    
    @staticmethod
    def align_raster_to_reference(input_file: str, 
                                 reference_file: str,
                                 output_file: str,
                                 resampling_method: str = 'nearest') -> str:
        """
        Align one raster to match another raster's grid.
        
        Parameters:
        -----------
        input_file : str
            Path to raster to be aligned
        reference_file : str
            Path to reference raster
        output_file : str
            Path for aligned output raster
        resampling_method : str, default 'nearest'
            Resampling method
            
        Returns:
        --------
        str
            Path to aligned raster
        """
        resampling_map = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic,
            'average': Resampling.average
        }
        
        resampling_enum = resampling_map.get(resampling_method, Resampling.nearest)
        
        with rasterio.open(reference_file) as ref_src:
            ref_transform = ref_src.transform
            ref_crs = ref_src.crs
            ref_shape = (ref_src.height, ref_src.width)
            
            with rasterio.open(input_file) as input_src:
                # Calculate output array
                output_array = np.zeros((input_src.count, ref_shape[0], ref_shape[1]), 
                                      dtype=input_src.dtypes[0])
                
                # Reproject
                reproject(
                    source=rasterio.band(input_src, list(range(1, input_src.count + 1))),
                    destination=output_array,
                    src_transform=input_src.transform,
                    src_crs=input_src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=resampling_enum
                )
                
                # Create output profile
                profile = input_src.profile.copy()
                profile.update({
                    'driver': 'GTiff',
                    'height': ref_shape[0],
                    'width': ref_shape[1],
                    'transform': ref_transform,
                    'crs': ref_crs
                })
                
                # Write aligned raster
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(output_array)
        
        print(f"✅ Aligned raster saved: {output_file}")
        return output_file


class TaskManager:
    """
    Manages Google Earth Engine tasks for large exports.
    """
    
    @staticmethod
    def check_task_status(task: ee.batch.Task) -> Dict[str, Any]:
        """
        Check status of an Earth Engine task.
        
        Parameters:
        -----------
        task : ee.batch.Task
            Task to check
            
        Returns:
        --------
        dict
            Task status information
        """
        status = task.status()
        
        return {
            'id': task.id,
            'state': status['state'],
            'description': status.get('description', 'No description'),
            'creation_timestamp': status.get('creation_timestamp_ms'),
            'start_timestamp': status.get('start_timestamp_ms'),
            'update_timestamp': status.get('update_timestamp_ms'),
            'error_message': status.get('error_message'),
            'progress': status.get('progress', 0)
        }
    
    @staticmethod
    def wait_for_task_completion(task: ee.batch.Task, 
                                timeout_minutes: int = 60,
                                check_interval: int = 30) -> bool:
        """
        Wait for task completion with timeout.
        
        Parameters:
        -----------
        task : ee.batch.Task
            Task to wait for
        timeout_minutes : int, default 60
            Maximum wait time in minutes
        check_interval : int, default 30
            Check interval in seconds
            
        Returns:
        --------
        bool
            True if completed successfully, False if failed/timeout
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        print(f"⏳ Waiting for task completion (timeout: {timeout_minutes} min)...")
        
        while True:
            status_info = TaskManager.check_task_status(task)
            state = status_info['state']
            
            print(f"📊 Task {task.id}: {state}")
            
            if state == 'COMPLETED':
                print("✅ Task completed successfully!")
                return True
            elif state == 'FAILED':
                error_msg = status_info.get('error_message', 'Unknown error')
                print(f"❌ Task failed: {error_msg}")
                return False
            elif state == 'CANCELLED':
                print("⚠️ Task was cancelled")
                return False
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                print(f"⏰ Timeout reached ({timeout_minutes} minutes)")
                return False
            
            # Wait before next check
            time.sleep(check_interval)
    
    @staticmethod
    def get_active_tasks() -> List[Dict[str, Any]]:
        """
        Get list of active Earth Engine tasks.
        
        Returns:
        --------
        list
            List of active task information
        """
        try:
            tasks = ee.batch.Task.list()
            active_tasks = []
            
            for task in tasks:
                if task.active():
                    status_info = TaskManager.check_task_status(task)
                    active_tasks.append(status_info)
            
            return active_tasks
            
        except Exception as e:
            print(f"❌ Error getting active tasks: {str(e)}")
            return []


# Convenience functions for direct use
def create_aoi_from_bounds(bounds: Tuple[float, float, float, float]) -> ee.Geometry:
    """
    Create AOI geometry from bounding box.
    
    Parameters:
    -----------
    bounds : tuple
        Bounding box as (minx, miny, maxx, maxy)
        
    Returns:
    --------
    ee.Geometry
        AOI geometry
    """
    return AOIManager.create_aoi_geometry(bounds)


def create_aoi_from_file(filepath: str) -> ee.Geometry:
    """
    Create AOI geometry from geospatial file.
    
    Parameters:
    -----------
    filepath : str
        Path to shapefile or geojson
        
    Returns:
    --------
    ee.Geometry
        AOI geometry
    """
    return AOIManager.create_aoi_geometry(filepath)


def validate_aoi(aoi_input: Union[Tuple, str, Dict]) -> Dict[str, Any]:
    """
    Validate AOI and return information.
    
    Parameters:
    -----------
    aoi_input : various
        AOI specification
        
    Returns:
    --------
    dict
        Validation results
    """
    try:
        aoi_geom = AOIManager.create_aoi_geometry(aoi_input)
        return AOIManager.validate_aoi_geometry(aoi_geom)
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


def download_ee_image(image: ee.Image,
                     aoi: Union[Tuple, str, Dict, ee.Geometry],
                     filename: str,
                     scale: int,
                     max_retries: int = 3) -> str:
    """
    Download Earth Engine image with robust error handling.
    
    Parameters:
    -----------
    image : ee.Image
        Image to download
    aoi : various
        Area of interest specification
    filename : str
        Output filename
    scale : int
        Spatial resolution in meters
    max_retries : int, default 3
        Maximum retry attempts
        
    Returns:
    --------
    str
        Path to downloaded file
    """
    # Create AOI geometry
    aoi_geom = AOIManager.create_aoi_geometry(aoi)
    
    # Initialize download manager
    manager = DownloadManager(max_retries=max_retries)
    
    # Download with retry
    return manager.download_with_retry(image, aoi_geom, filename, scale)


def export_ee_image_to_drive(image: ee.Image,
                            aoi: Union[Tuple, str, Dict, ee.Geometry],
                            description: str,
                            folder: str,
                            scale: int,
                            wait_for_completion: bool = False) -> Union[ee.batch.Task, bool]:
    """
    Export Earth Engine image to Google Drive.
    
    Parameters:
    -----------
    image : ee.Image
        Image to export
    aoi : various
        Area of interest specification
    description : str
        Export description
    folder : str
        Drive folder name
    scale : int
        Spatial resolution in meters
    wait_for_completion : bool, default False
        Whether to wait for task completion
        
    Returns:
    --------
    ee.batch.Task or bool
        Task object or completion status if waiting
    """
    # Create AOI geometry
    aoi_geom = AOIManager.create_aoi_geometry(aoi)
    
    # Create export task
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        scale=scale,
        region=aoi_geom,
        fileFormat='GeoTIFF',
        maxPixels=1e9,
        crs='EPSG:4326'
    )
    
    # Start task
    task.start()
    print(f"🚀 Export task '{description}' started")
    
    if wait_for_completion:
        return TaskManager.wait_for_task_completion(task)
    else:
        return task


def check_files_alignment(file1: str, file2: str) -> bool:
    """
    Quick check if two raster files are spatially aligned.
    
    Parameters:
    -----------
    file1 : str
        Path to first raster
    file2 : str
        Path to second raster
        
    Returns:
    --------
    bool
        True if files are aligned
    """
    result = SpatialValidator.check_spatial_alignment(file1, file2)
    
    if result['aligned']:
        print(f"✅ Files are spatially aligned")
        return True
    else:
        print(f"❌ Files are not aligned:")
        if not result['crs_match']:
            print(f"   - CRS mismatch: {result['file1_info']['crs']} vs {result['file2_info']['crs']}")
        if not result['shape_match']:
            print(f"   - Shape mismatch: {result['file1_info']['shape']} vs {result['file2_info']['shape']}")
        if not result['bounds_match']:
            print(f"   - Bounds mismatch")
        if result['overlap_percentage'] < 100:
            print(f"   - Overlap: {result['overlap_percentage']:.1f}%")
        return False


def align_to_reference(input_file: str, 
                      reference_file: str,
                      output_file: str,
                      method: str = 'nearest') -> str:
    """
    Align one raster to match another's grid.
    
    Parameters:
    -----------
    input_file : str
        Raster to align
    reference_file : str
        Reference raster
    output_file : str
        Output path
    method : str, default 'nearest'
        Resampling method
        
    Returns:
    --------
    str
        Path to aligned raster
    """
    return SpatialValidator.align_raster_to_reference(
        input_file, reference_file, output_file, method
    )


def get_file_info(filepath: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a raster file.
    
    Parameters:
    -----------
    filepath : str
        Path to raster file
        
    Returns:
    --------
    dict
        File information
    """
    try:
        with rasterio.open(filepath) as src:
            bounds = src.bounds
            
            info = {
                'filename': os.path.basename(filepath),
                'shape': (src.height, src.width),
                'bands': src.count,
                'dtype': str(src.dtypes[0]),
                'crs': str(src.crs),
                'transform': list(src.transform),
                'bounds': {
                    'minx': bounds.left,
                    'miny': bounds.bottom,
                    'maxx': bounds.right,
                    'maxy': bounds.top
                },
                'nodata': src.nodata,
                'compression': src.compression,
                'file_size_mb': round(os.path.getsize(filepath) / (1024*1024), 2)
            }
            
            # Calculate pixel size
            info['pixel_size'] = {
                'x': abs(src.transform[0]),
                'y': abs(src.transform[4])
            }
            
            # Get basic statistics for first band
            sample_data = src.read(1, masked=True)
            if sample_data.size > 0:
                info['statistics'] = {
                    'min': float(sample_data.min()),
                    'max': float(sample_data.max()),
                    'mean': float(sample_data.mean()),
                    'std': float(sample_data.std())
                }
            
            return info
            
    except Exception as e:
        return {
            'filename': os.path.basename(filepath),
            'error': str(e)
        }


def cleanup_downloads(keep_recent: int = 5, pattern: str = "*.tif") -> None:
    """
    Clean up old download files to save space.
    
    Parameters:
    -----------
    keep_recent : int, default 5
        Number of recent files to keep
    pattern : str, default "*.tif"
        File pattern to match
    """
    import glob
    
    files = glob.glob(pattern)
    if len(files) <= keep_recent:
        print(f"📁 Found {len(files)} files, keeping all")
        return
    
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    
    # Keep recent files, remove others
    files_to_remove = files[keep_recent:]
    
    print(f"📁 Cleaning up {len(files_to_remove)} old files...")
    
    for filepath in files_to_remove:
        try:
            os.remove(filepath)
            print(f"   🗑️ Removed: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"   ❌ Failed to remove {filepath}: {str(e)}")
    
    print(f"✅ Cleanup complete. Kept {keep_recent} recent files.")


# Example usage and testing
if __name__ == "__main__":
    print("🛠️ Download Utils Test")
    print("="*40)
    
    # Test AOI creation and validation
    test_bounds = (106.7, -6.4, 107.0, -6.1)
    
    try:
        print("🔍 Testing AOI creation...")
        aoi_geom = create_aoi_from_bounds(test_bounds)
        
        print("🔍 Validating AOI...")
        validation = validate_aoi(test_bounds)
        
        if validation['valid']:
            print(f"✅ AOI validation passed:")
            print(f"   Area: {validation['area_km2']:.1f} km²")
            print(f"   Bounds: {validation['bounds']}")
            if validation['warnings']:
                for warning in validation['warnings']:
                    print(f"   ⚠️ {warning}")
        else:
            print(f"❌ AOI validation failed: {validation.get('error', 'Unknown error')}")
        
        # Test task management
        print("\n📊 Checking active GEE tasks...")
        active_tasks = TaskManager.get_active_tasks()
        print(f"   Found {len(active_tasks)} active tasks")
        
        print("\n✅ All utility tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("Make sure GEE is authenticated before running tests")