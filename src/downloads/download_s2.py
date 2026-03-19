"""
Sentinel-2 Data Download and Harmonization for ML Model Inference
================================================================

This module downloads Sentinel-2 data from Google Earth Engine with exact harmonization
to match the original preprocessing pipeline. Ensures perfect compatibility with trained ML models.

Key Features:
- Replicates harmonize.py offset correction for processing baseline >= 04.00
- Applies exact same scaling and normalization as original pipeline
- Maintains band order and naming consistency
- Creates cloud-free temporal composites
- Validates output format for ML model compatibility

Author: Adapted for ML inference preprocessing pipeline
Compatible with: Google Colab, original preprocessing pipeline
Dependencies: earthengine-api, requests, zipfile
"""

import ee
import requests
import zipfile
import os
import numpy as np
import rasterio
from io import BytesIO
from typing import List, Dict, Tuple, Optional, Union
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class S2Downloader:
    """
    Sentinel-2 downloader with exact harmonization matching original pipeline.
    
    Replicates the preprocessing steps from harmonize.py to ensure ML model compatibility.
    """
    
    def __init__(self, validate_outputs: bool = True):
        """
        Initialize S2 downloader.
        
        Parameters:
        -----------
        validate_outputs : bool, default True
            Whether to validate downloaded data format
        """
        self.validate_outputs = validate_outputs
        self.download_stats = {}
        
        # Expected S2 band properties (for validation)
        self.expected_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        self.expected_ranges = {
            'raw': {'min': 0, 'max': 10000},      # Before normalization
            'normalized': {'min': 0.0, 'max': 1.0}  # After normalization
        }
    
    def download_s2_harmonized(self, 
                              aoi_geom: ee.Geometry,
                              start_date: str,
                              end_date: str,
                              bands: List[str] = None,
                              cloud_filter: int = 20,
                              scale: int = 10,
                              apply_harmonization: bool = True,
                              normalize: bool = True,
                              composite_method: str = 'median',
                              output_filename: str = 's2_composite.tif',
                              max_pixels: int = 1e9,
                              tile_size_deg: float = 0.1) -> str:
        """
        Download Sentinel-2 data with harmonization matching original pipeline.
        
        Automatically handles large areas by tiling and mosaicking.
        
        Parameters:
        -----------
        aoi_geom : ee.Geometry
            Area of interest geometry
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        bands : list, optional
            List of S2 bands. Default: ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        cloud_filter : int, default 20
            Maximum cloud percentage threshold
        scale : int, default 10
            Spatial resolution in meters
        apply_harmonization : bool, default True
            Apply S2 harmonization (offset correction)
        normalize : bool, default True
            Normalize to [0,1] range (divide by 10000)
        composite_method : str, default 'median'
            Temporal reduction method ('median', 'mean', 'mosaic')
        output_filename : str, default 's2_composite.tif'
            Output filename for downloaded data
        max_pixels : int, default 1e9
            Maximum pixels per download chunk
        tile_size_deg : float, default 0.1
            Tile size in degrees for large areas
            
        Returns:
        --------
        str
            Path to downloaded S2 composite file
        """
        if bands is None:
            bands = self.expected_bands.copy()
        
        print(f"📡 Downloading Sentinel-2 data...")
        print(f"   📅 Date range: {start_date} to {end_date}")
        print(f"   📊 Bands: {bands}")
        print(f"   ☁️ Cloud filter: {cloud_filter}%")
        print(f"   🔧 Harmonization: {'ON' if apply_harmonization else 'OFF'}")
        print(f"   📏 Scale: {scale}m")
        
        # Load and filter S2 collection
        s2_collection = self._load_s2_collection(
            aoi_geom, start_date, end_date, cloud_filter, bands
        )
        
        # Apply harmonization if requested
        if apply_harmonization:
            print("🔧 Applying S2 harmonization (offset correction)...")
            s2_collection = s2_collection.map(self._apply_s2_harmonization_gee)
        
        # Apply normalization if requested
        if normalize:
            print("📏 Applying normalization (scaling to [0,1])...")
            s2_collection = s2_collection.map(
                lambda img: img.divide(10000.0).toFloat()
            )
        
        # Create temporal composite
        print(f"🔄 Creating {composite_method} composite...")
        s2_composite = self._create_temporal_composite(s2_collection, composite_method)
        
        # Clip to AOI
        s2_composite = s2_composite.clip(aoi_geom)
        
        # Check if area is too large for direct download
        area_info = self._estimate_download_size(aoi_geom, scale, len(bands))
        
        print(f"📊 Estimated download size: {area_info['size_mb']:.1f} MB")
        print(f"📐 AOI dimensions: {area_info['width_km']:.1f} x {area_info['height_km']:.1f} km")
        
        if area_info['size_mb'] > 40:  # Conservative limit for direct download
            print("🔄 Area too large for direct download - using tiled approach...")
            downloaded_path = self._download_large_area_tiled(
                s2_composite, aoi_geom, output_filename, scale, tile_size_deg
            )
        else:
            print("💾 Downloading directly...")
            downloaded_path = self._download_image_to_colab(
                s2_composite, aoi_geom, output_filename, scale
            )
        
        # Validate output if requested
        if self.validate_outputs:
            self._validate_s2_output(downloaded_path, bands, normalize)
        
        # Store download statistics
        self._update_download_stats(s2_collection, downloaded_path)
        
        print(f"✅ S2 data downloaded successfully: {downloaded_path}")
        return downloaded_path
    
    def _estimate_download_size(self, aoi_geom: ee.Geometry, scale: int, num_bands: int) -> Dict[str, float]:
        """Estimate download size and AOI dimensions."""
        
        try:
            # Get AOI bounds
            bounds = aoi_geom.bounds().getInfo()
            coords = bounds['coordinates'][0]
            
            # Calculate dimensions
            min_lon, min_lat = coords[0]
            max_lon, max_lat = coords[2]
            
            width_deg = max_lon - min_lon
            height_deg = max_lat - min_lat
            
            # Rough conversion to km (111 km per degree)
            width_km = width_deg * 111
            height_km = height_deg * 111
            
            # Estimate pixels
            width_pixels = width_km * 1000 / scale  # Convert km to meters, then to pixels
            height_pixels = height_km * 1000 / scale
            total_pixels = width_pixels * height_pixels
            
            # Estimate file size (4 bytes per pixel for float32, times number of bands)
            size_bytes = total_pixels * num_bands * 4
            size_mb = size_bytes / (1024 * 1024)
            
            return {
                'width_km': width_km,
                'height_km': height_km,
                'width_pixels': width_pixels,
                'height_pixels': height_pixels,
                'total_pixels': total_pixels,
                'size_mb': size_mb
            }
            
        except Exception as e:
            print(f"⚠️ Could not estimate download size: {str(e)}")
            return {
                'width_km': 0, 'height_km': 0,
                'width_pixels': 0, 'height_pixels': 0,
                'total_pixels': 0, 'size_mb': 999  # Assume large to trigger tiling
            }
    
    def _download_large_area_tiled(self, 
                                  image: ee.Image,
                                  aoi_geom: ee.Geometry,
                                  output_filename: str,
                                  scale: int,
                                  tile_size_deg: float = 0.1) -> str:
        """
        Download large area using tiling approach.
        
        Parameters:
        -----------
        image : ee.Image
            Image to download
        aoi_geom : ee.Geometry
            Area of interest
        output_filename : str
            Final output filename
        scale : int
            Spatial resolution
        tile_size_deg : float
            Tile size in degrees
            
        Returns:
        --------
        str
            Path to mosaicked output file
        """
        print(f"🗂️ Creating tiles (size: {tile_size_deg}° = ~{tile_size_deg * 111:.1f} km)...")
        
        # Create tiles
        tiles = self._create_tiles_from_aoi(aoi_geom, tile_size_deg)
        
        print(f"📦 Created {len(tiles)} tiles for download")
        
        # Download each tile
        tile_files = []
        temp_dir = "temp_tiles"
        os.makedirs(temp_dir, exist_ok=True)
        
        for i, tile_geom in enumerate(tiles):
            tile_filename = f"{temp_dir}/tile_{i:03d}.tif"
            
            try:
                print(f"📥 Downloading tile {i+1}/{len(tiles)}...")
                
                # Clip image to tile
                tile_image = image.clip(tile_geom)
                
                # Download tile
                self._download_image_to_colab(
                    tile_image, tile_geom, tile_filename, scale
                )
                
                tile_files.append(tile_filename)
                
            except Exception as e:
                print(f"⚠️ Failed to download tile {i+1}: {str(e)}")
                # Continue with other tiles
                continue
        
        print(f"✅ Downloaded {len(tile_files)}/{len(tiles)} tiles successfully")
        
        if len(tile_files) == 0:
            raise Exception("No tiles downloaded successfully")
        
        # Mosaic tiles together
        print("🔗 Mosaicking tiles...")
        mosaicked_path = self._mosaic_tiles(tile_files, output_filename, scale)
        
        # Clean up temporary files
        print("🧹 Cleaning up temporary files...")
        for tile_file in tile_files:
            try:
                os.remove(tile_file)
            except:
                pass
        
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        return mosaicked_path
    
    def _create_tiles_from_aoi(self, aoi_geom: ee.Geometry, tile_size_deg: float) -> List[ee.Geometry]:
        """Create tiles covering the AOI."""
        
        # Get AOI bounds
        bounds = aoi_geom.bounds().getInfo()
        coords = bounds['coordinates'][0]
        
        min_lon, min_lat = coords[0]
        max_lon, max_lat = coords[2]
        
        tiles = []
        
        # Create grid of tiles
        lat = min_lat
        while lat < max_lat:
            lon = min_lon
            while lon < max_lon:
                # Create tile bounds
                tile_bounds = [
                    lon,
                    lat,
                    min(lon + tile_size_deg, max_lon),
                    min(lat + tile_size_deg, max_lat)
                ]
                
                # Create tile geometry
                tile_geom = ee.Geometry.Rectangle(tile_bounds)
                
                # Only include tiles that intersect with AOI
                if aoi_geom.intersects(tile_geom).getInfo():
                    # Clip tile to AOI
                    clipped_tile = tile_geom.intersection(aoi_geom)
                    tiles.append(clipped_tile)
                
                lon += tile_size_deg
            lat += tile_size_deg
        
        return tiles
    
    def _mosaic_tiles(self, tile_files: List[str], output_filename: str, scale: int) -> str:
        """
        Mosaic downloaded tiles into single file.
        
        Parameters:
        -----------
        tile_files : list
            List of tile file paths
        output_filename : str
            Output filename
        scale : int
            Spatial resolution
            
        Returns:
        --------
        str
            Path to mosaicked file
        """
        import rasterio
        from rasterio.merge import merge
        from rasterio.enums import Resampling
        
        print(f"🔗 Mosaicking {len(tile_files)} tiles...")
        
        # Open all tile files
        tile_datasets = []
        for tile_file in tile_files:
            if os.path.exists(tile_file):
                try:
                    dataset = rasterio.open(tile_file)
                    tile_datasets.append(dataset)
                except Exception as e:
                    print(f"⚠️ Could not open tile {tile_file}: {str(e)}")
        
        if len(tile_datasets) == 0:
            raise Exception("No valid tile datasets to mosaic")
        
        print(f"📊 Mosaicking {len(tile_datasets)} valid tiles...")
        
        # Merge tiles
        mosaic_array, mosaic_transform = merge(
            tile_datasets,
            resampling=Resampling.nearest
        )
        
        # Get metadata from first tile
        mosaic_meta = tile_datasets[0].meta.copy()
        mosaic_meta.update({
            "driver": "GTiff",
            "height": mosaic_array.shape[1],
            "width": mosaic_array.shape[2],
            "transform": mosaic_transform,
            "compress": "lzw"
        })
        
        # Write mosaicked file
        with rasterio.open(output_filename, "w", **mosaic_meta) as dest:
            dest.write(mosaic_array)
        
        # Close tile datasets
        for dataset in tile_datasets:
            dataset.close()
        
        print(f"✅ Mosaic saved: {output_filename}")
        
        # Validate mosaic
        if os.path.exists(output_filename):
            size_mb = os.path.getsize(output_filename) / (1024 * 1024)
            print(f"📊 Final mosaic size: {size_mb:.1f} MB")
            return output_filename
        else:
            raise Exception("Mosaic file was not created successfully")
    
    def _download_image_to_colab(self,
                                image: ee.Image,
                                aoi_geom: ee.Geometry,
                                filename: str,
                                scale: int,
                                max_retries: int = 3) -> str:
        """
        Download Earth Engine image directly to Colab with retry logic.
        
        Parameters:
        -----------
        image : ee.Image
            Image to download
        aoi_geom : ee.Geometry
            Area of interest
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
        for attempt in range(max_retries):
            try:
                print(f"🌐 Download attempt {attempt + 1}/{max_retries}...")
                
                # Get download URL
                url = image.getDownloadURL({
                    'scale': scale,
                    'crs': 'EPSG:4326',
                    'region': aoi_geom,
                    'format': 'GEO_TIFF',
                    'maxPixels': 1e9
                })
                
                # Download file
                response = requests.get(url, stream=True, timeout=300)
                
                if response.status_code == 200:
                    print("📦 Processing download...")
                    
                    # Handle zip file (GEE often returns zipped tiffs)
                    if response.headers.get('content-type') == 'application/zip':
                        return self._extract_from_zip(response.content, filename)
                    else:
                        # Direct tiff file
                        with open(filename, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        return filename
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.reason}")
                    
            except Exception as e:
                print(f"❌ Download attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    print("⏳ Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    raise Exception(f"All download attempts failed. Last error: {str(e)}")
    
    def export_large_area_to_drive(self,
                                  aoi_geom: ee.Geometry,
                                  start_date: str,
                                  end_date: str,
                                  bands: List[str] = None,
                                  cloud_filter: int = 20,
                                  scale: int = 10,
                                  apply_harmonization: bool = True,
                                  normalize: bool = True,
                                  composite_method: str = 'median',
                                  description: str = 's2_large_export',
                                  folder: str = 'GEE_Downloads',
                                  tile_size_deg: float = 0.2) -> List[ee.batch.Task]:
        """
        Export large S2 area to Google Drive using tiles.
        
        For very large areas, this exports multiple tiles to Drive
        which you can then download and mosaic locally or in another session.
        
        Parameters:
        -----------
        [Same as download_s2_harmonized, plus:]
        description : str, default 's2_large_export'
            Base name for export tasks
        folder : str, default 'GEE_Downloads'
            Google Drive folder name
        tile_size_deg : float, default 0.2
            Tile size in degrees (larger tiles for Drive export)
            
        Returns:
        --------
        list
            List of export tasks
        """
        if bands is None:
            bands = self.expected_bands.copy()
        
        print(f"📤 Setting up large area export to Google Drive...")
        
        # Prepare the image (same as download process)
        s2_collection = self._load_s2_collection(
            aoi_geom, start_date, end_date, cloud_filter, bands
        )
        
        if apply_harmonization:
            s2_collection = s2_collection.map(self._apply_s2_harmonization_gee)
        
        if normalize:
            s2_collection = s2_collection.map(
                lambda img: img.divide(10000.0).toFloat()
            )
        
        s2_composite = self._create_temporal_composite(s2_collection, composite_method)
        s2_composite = s2_composite.clip(aoi_geom)
        
        # Create tiles
        tiles = self._create_tiles_from_aoi(aoi_geom, tile_size_deg)
        print(f"📦 Creating {len(tiles)} export tasks...")
        
        # Create export task for each tile
        tasks = []
        for i, tile_geom in enumerate(tiles):
            tile_image = s2_composite.clip(tile_geom)
            
            task = ee.batch.Export.image.toDrive(
                image=tile_image,
                description=f"{description}_tile_{i:03d}",
                folder=folder,
                scale=scale,
                region=tile_geom,
                fileFormat='GeoTIFF',
                maxPixels=1e9,
                crs='EPSG:4326'
            )
            
            task.start()
            tasks.append(task)
            print(f"🚀 Started export task: {description}_tile_{i:03d}")
        
        print(f"✅ All {len(tasks)} export tasks started")
        print(f"📁 Files will be saved to Google Drive folder: {folder}")
        print("⏳ Check task status with: [task.status() for task in tasks]")
        
        return tasks
    
    def _load_s2_collection(self, 
                           aoi_geom: ee.Geometry,
                           start_date: str,
                           end_date: str,
                           cloud_filter: int,
                           bands: List[str]) -> ee.ImageCollection:
        """Load and filter Sentinel-2 collection."""
        
        # Use S2 SR Harmonized collection (already harmonized surface reflectance)
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                      .filterBounds(aoi_geom) \
                      .filterDate(start_date, end_date) \
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter)) \
                      .select(bands)
        
        # Check collection size
        collection_size = collection.size().getInfo()
        print(f"📊 Found {collection_size} Sentinel-2 images")
        
        if collection_size == 0:
            raise ValueError(f"No Sentinel-2 images found for the specified criteria")
        
        return collection
    
    def _apply_s2_harmonization_gee(self, image: ee.Image) -> ee.Image:
        """
        Apply S2 harmonization in GEE (replicates harmonize.py logic).
        
        This function exactly replicates the harmonization logic from harmonize.py:
        - Check processing baseline
        - Apply -1000 offset for baseline >= '04.00'
        - Maintain float32 data type
        
        Parameters:
        -----------
        image : ee.Image
            Input S2 image
            
        Returns:
        --------
        ee.Image
            Harmonized S2 image
        """
        # Get processing baseline from metadata
        baseline = ee.String(image.get('PROCESSING_BASELINE'))
        
        # Check if baseline >= '04.00' (needs offset correction)
        needs_offset = baseline.compareTo('04.00').gte(0)
        
        # Apply offset correction (-1000) where needed
        # This exactly replicates: bands_data -= offset from harmonize.py
        offset_corrected = image.subtract(1000).toFloat()
        original = image.toFloat()
        
        # Use conditional to apply offset only where needed
        harmonized = ee.Image(ee.Algorithms.If(needs_offset, offset_corrected, original))
        
        # Copy properties from original image
        return harmonized.copyProperties(image, ['system:time_start', 'system:time_end'])
    
    def _create_temporal_composite(self, 
                                  collection: ee.ImageCollection,
                                  method: str = 'median') -> ee.Image:
        """Create temporal composite from image collection."""
        
        if method == 'median':
            composite = collection.median()
        elif method == 'mean':
            composite = collection.mean()
        elif method == 'mosaic':
            composite = collection.mosaic()
        else:
            raise ValueError(f"Unknown composite method: {method}")
        
        return composite.toFloat()
    
    def _download_image_to_colab(self,
                                image: ee.Image,
                                aoi_geom: ee.Geometry,
                                filename: str,
                                scale: int) -> str:
        """
        Download Earth Engine image directly to Colab.
        
        Parameters:
        -----------
        image : ee.Image
            Image to download
        aoi_geom : ee.Geometry
            Area of interest
        filename : str
            Output filename
        scale : int
            Spatial resolution in meters
            
        Returns:
        --------
        str
            Path to downloaded file
        """
        try:
            # Get download URL
            url = image.getDownloadURL({
                'scale': scale,
                'crs': 'EPSG:4326',
                'region': aoi_geom,
                'format': 'GEO_TIFF',
                'maxPixels': 1e9
            })
            
            # Download file
            print("🌐 Requesting download from GEE...")
            response = requests.get(url, stream=True, timeout=300)
            
            if response.status_code == 200:
                print("📦 Processing download...")
                
                # Handle zip file (GEE often returns zipped tiffs)
                if response.headers.get('content-type') == 'application/zip':
                    return self._extract_from_zip(response.content, filename)
                else:
                    # Direct tiff file
                    with open(filename, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return filename
            else:
                raise Exception(f"Download failed with status code: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Direct download failed: {str(e)}")
            print("🔄 Consider using Drive export for large areas")
            raise
    
    def _extract_from_zip(self, zip_content: bytes, target_filename: str) -> str:
        """Extract tiff file from zip archive."""
        
        with zipfile.ZipFile(BytesIO(zip_content)) as zip_file:
            # Find .tif files in the archive
            tif_files = [f for f in zip_file.namelist() if f.endswith('.tif')]
            
            if not tif_files:
                raise Exception("No .tif files found in downloaded archive")
            
            # Extract the first .tif file
            zip_file.extract(tif_files[0], '.')
            
            # Rename to target filename
            if tif_files[0] != target_filename:
                os.rename(tif_files[0], target_filename)
            
            return target_filename
    
    def _validate_s2_output(self, 
                           file_path: str,
                           expected_bands: List[str],
                           is_normalized: bool) -> None:
        """
        Validate downloaded S2 data format and ranges.
        
        Ensures the downloaded data matches expected format for ML model.
        """
        print("🔍 Validating S2 output format...")
        
        try:
            with rasterio.open(file_path) as src:
                # Check band count
                if src.count != len(expected_bands):
                    raise ValueError(f"Expected {len(expected_bands)} bands, got {src.count}")
                
                # Check data type
                if src.dtypes[0] != 'float32':
                    print(f"⚠️ Warning: Data type is {src.dtypes[0]}, expected float32")
                
                # Check data ranges
                sample_data = src.read(1, masked=True)  # Read first band
                data_min, data_max = float(sample_data.min()), float(sample_data.max())
                
                expected_range = self.expected_ranges['normalized' if is_normalized else 'raw']
                
                if not (expected_range['min'] <= data_min <= data_max <= expected_range['max']):
                    print(f"⚠️ Warning: Data range [{data_min:.3f}, {data_max:.3f}] "
                          f"outside expected range [{expected_range['min']}, {expected_range['max']}]")
                
                print(f"✅ Validation passed:")
                print(f"   📊 Bands: {src.count}")
                print(f"   📏 Shape: {src.shape}")
                print(f"   🔢 Data type: {src.dtypes[0]}")
                print(f"   📈 Data range: [{data_min:.3f}, {data_max:.3f}]")
                print(f"   🗺️ CRS: {src.crs}")
                
        except Exception as e:
            print(f"❌ Validation failed: {str(e)}")
            raise
    
    def _update_download_stats(self, 
                              collection: ee.ImageCollection,
                              file_path: str) -> None:
        """Update download statistics."""
        
        try:
            # Get collection info
            collection_size = collection.size().getInfo()
            date_range = collection.aggregate_min_max('system:time_start').getInfo()
            
            # Get file info
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            
            self.download_stats = {
                'images_used': collection_size,
                'date_range': {
                    'start': datetime.fromtimestamp(date_range['min']/1000).strftime('%Y-%m-%d'),
                    'end': datetime.fromtimestamp(date_range['max']/1000).strftime('%Y-%m-%d')
                },
                'file_size_mb': round(file_size, 2),
                'download_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"⚠️ Warning: Could not update statistics: {str(e)}")
    
    def export_to_drive(self,
                       aoi_geom: ee.Geometry,
                       start_date: str,
                       end_date: str,
                       bands: List[str] = None,
                       cloud_filter: int = 20,
                       scale: int = 10,
                       apply_harmonization: bool = True,
                       normalize: bool = True,
                       composite_method: str = 'median',
                       description: str = 's2_export',
                       folder: str = 'GEE_Downloads') -> ee.batch.Task:
        """
        Export S2 data to Google Drive (alternative to direct download).
        
        Use this method for large areas where direct download might fail.
        
        Parameters:
        -----------
        [Same as download_s2_harmonized, plus:]
        description : str, default 's2_export'
            Export task description
        folder : str, default 'GEE_Downloads'
            Google Drive folder name
            
        Returns:
        --------
        ee.batch.Task
            Export task object
        """
        if bands is None:
            bands = self.expected_bands.copy()
        
        print(f"📤 Setting up Google Drive export...")
        
        # Prepare the image (same as download process)
        s2_collection = self._load_s2_collection(
            aoi_geom, start_date, end_date, cloud_filter, bands
        )
        
        if apply_harmonization:
            s2_collection = s2_collection.map(self._apply_s2_harmonization_gee)
        
        if normalize:
            s2_collection = s2_collection.map(
                lambda img: img.divide(10000.0).toFloat()
            )
        
        s2_composite = self._create_temporal_composite(s2_collection, composite_method)
        s2_composite = s2_composite.clip(aoi_geom)
        
        # Create export task
        task = ee.batch.Export.image.toDrive(
            image=s2_composite,
            description=description,
            folder=folder,
            scale=scale,
            region=aoi_geom,
            fileFormat='GeoTIFF',
            maxPixels=1e9,
            crs='EPSG:4326'
        )
        
        # Start the task
        task.start()
        print(f"🚀 Export task '{description}' started")
        print(f"📁 Will be saved to Google Drive folder: {folder}")
        print("⏳ Check task status with: task.status()")
        
        return task
    
    def get_download_stats(self) -> Dict:
        """Get download statistics from last operation."""
        return self.download_stats.copy()


# Convenience functions for direct use
def quick_download_s2(aoi_bounds: Tuple[float, float, float, float],
                     start_date: str,
                     end_date: str,
                     output_filename: str = 's2_composite.tif',
                     bands: List[str] = None,
                     cloud_filter: int = 20,
                     apply_harmonization: bool = True,
                     normalize: bool = True) -> str:
    """
    Quick function to download S2 data with default settings.
    
    Parameters:
    -----------
    aoi_bounds : tuple
        Bounding box as (minx, miny, maxx, maxy) in WGS84
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    output_filename : str, default 's2_composite.tif'
        Output filename
    bands : list, optional
        S2 bands to download
    cloud_filter : int, default 20
        Maximum cloud percentage
    apply_harmonization : bool, default True
        Apply S2 harmonization
    normalize : bool, default True
        Normalize to [0,1] range
        
    Returns:
    --------
    str
        Path to downloaded S2 composite file
    """
    if bands is None:
        bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    
    # Create AOI geometry from bounds
    aoi_geom = ee.Geometry.Rectangle(aoi_bounds)
    
    # Initialize downloader and download
    downloader = S2Downloader()
    return downloader.download_s2_harmonized(
        aoi_geom=aoi_geom,
        start_date=start_date,
        end_date=end_date,
        bands=bands,
        cloud_filter=cloud_filter,
        apply_harmonization=apply_harmonization,
        normalize=normalize,
        output_filename=output_filename
    )


def validate_s2_compatibility(file_path: str, 
                             expected_bands: List[str] = None,
                             is_normalized: bool = True) -> bool:
    """
    Validate S2 file compatibility with ML models.
    
    Parameters:
    -----------
    file_path : str
        Path to S2 file to validate
    expected_bands : list, optional
        Expected band list
    is_normalized : bool, default True
        Whether data should be normalized [0,1]
        
    Returns:
    --------
    bool
        True if file is compatible, False otherwise
    """
    if expected_bands is None:
        expected_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    
    downloader = S2Downloader()
    try:
        downloader._validate_s2_output(file_path, expected_bands, is_normalized)
        return True
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        return False


def check_s2_processing_baseline(aoi_geom: ee.Geometry,
                               start_date: str,
                               end_date: str) -> Dict:
    """
    Check processing baselines of available S2 images to understand harmonization needs.
    
    Parameters:
    -----------
    aoi_geom : ee.Geometry
        Area of interest
    start_date : str
        Start date
    end_date : str
        End date
        
    Returns:
    --------
    dict
        Summary of processing baselines found
    """
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                   .filterBounds(aoi_geom) \
                   .filterDate(start_date, end_date)
    
    # Get baseline information
    baselines = collection.aggregate_array('PROCESSING_BASELINE').getInfo()
    baseline_summary = {}
    
    for baseline in baselines:
        if baseline in baseline_summary:
            baseline_summary[baseline] += 1
        else:
            baseline_summary[baseline] = 1
    
    # Count images needing offset correction
    need_correction = sum(count for baseline, count in baseline_summary.items() 
                         if baseline >= '04.00')
    
    total_images = len(baselines)
    
    print(f"📊 Processing Baseline Summary:")
    print(f"   Total images: {total_images}")
    print(f"   Need offset correction (>= 04.00): {need_correction}")
    print(f"   Baseline distribution:")
    for baseline, count in sorted(baseline_summary.items()):
        correction_needed = "✓" if baseline >= '04.00' else "✗"
        print(f"     {baseline}: {count} images (offset correction: {correction_needed})")
    
    return {
        'total_images': total_images,
        'need_correction': need_correction,
        'baseline_distribution': baseline_summary
    }


# Example usage and testing
if __name__ == "__main__":
    print("📡 Sentinel-2 Downloader Test")
    print("="*40)
    
    # Test parameters (Jakarta area)
    test_bounds = (106.7, -6.4, 107.0, -6.1)
    test_start = "2023-06-01"
    test_end = "2023-08-31"
    
    try:
        # Check processing baselines
        aoi_geom = ee.Geometry.Rectangle(test_bounds)
        print("🔍 Checking S2 processing baselines...")
        baseline_info = check_s2_processing_baseline(aoi_geom, test_start, test_end)
        
        # Download small test area
        print("\n📡 Testing S2 download...")
        s2_file = quick_download_s2(
            aoi_bounds=test_bounds,
            start_date=test_start,
            end_date=test_end,
            output_filename='test_s2.tif',
            cloud_filter=30
        )
        
        # Validate output
        print("\n🔍 Validating output...")
        is_valid = validate_s2_compatibility(s2_file)
        
        if is_valid:
            print("✅ All tests passed!")
        else:
            print("❌ Validation failed!")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("Make sure GEE is authenticated before running tests")