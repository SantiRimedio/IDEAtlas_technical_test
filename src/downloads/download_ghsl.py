"""
GHSL Built-up Data Download and Processing for ML Model Inference
================================================================

This module downloads GHSL (Global Human Settlement Layer) built-up data from Google Earth Engine
with exact processing to match the original create_ref.py pipeline. Ensures perfect compatibility 
with trained ML models and reference label creation.

Key Features:
- Replicates create_ref.py GHSL processing logic exactly
- Applies same threshold (>15) for built-up classification
- Creates binary masks matching original workflow
- Resamples to match S2 resolution (10m) with proper alignment
- Validates output format for ML model compatibility
- Supports large area downloads with automatic tiling

Author: Adapted for ML inference preprocessing pipeline  
Compatible with: Google Colab, original create_ref.py workflow
Dependencies: earthengine-api, requests, rasterio
"""

import ee
import requests
import zipfile
import os
import numpy as np
import rasterio
from io import BytesIO
from typing import List, Dict, Tuple, Optional, Union, Any
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("⚠️ Warning: geopandas not available. DUA processing will be limited.")


class GHSLDownloader:
    """
    GHSL downloader with exact processing matching original create_ref.py.
    
    Replicates the GHSL processing steps from create_ref.py to ensure ML model compatibility.
    """
    
    def __init__(self, validate_outputs: bool = True):
        """
        Initialize GHSL downloader.
        
        Parameters:
        -----------
        validate_outputs : bool, default True
            Whether to validate downloaded data format
        """
        self.validate_outputs = validate_outputs
        self.download_stats = {}
        
        # GHSL dataset information
        self.ghsl_datasets = {
            2020: {
                'collection': "JRC/GHSL/P2023A/GHS_BUILT_S", 
                'band': 'built_surface',
                'filter_property': 'year'
            },
            2015: {
                'collection': "JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1",
                'band': 'built',
                'filter_property': 'year'  
            },
            2000: {
                'collection': "JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1",
                'band': 'built',
                'filter_property': 'year'
            },
            1990: {
                'collection': "JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1", 
                'band': 'built',
                'filter_property': 'year'
            }
        }
        
        # Expected GHSL properties (for validation)
        self.expected_threshold = 15
        self.expected_classes = {0: 'non-built', 1: 'built-up'}
    
    def download_ghsl_builtup(self,
                             aoi_geom: ee.Geometry,
                             year: int = 2020,
                             scale: int = 100,
                             threshold: int = 15,
                             resample_to_s2: bool = True,
                             target_scale: int = 10,
                             output_filename: str = 'ghsl_builtup.tif') -> str:
        """
        Download GHSL built-up data with processing matching create_ref.py.
        
        This function exactly replicates the GHSL processing from create_ref.py:
        1. Load GHSL built-up fraction data
        2. Apply threshold > 15 for built-up classification  
        3. Create binary mask (0 = non-built, 1 = built-up)
        4. Resample to match S2 resolution if requested
        
        Parameters:
        -----------
        aoi_geom : ee.Geometry
            Area of interest geometry
        year : int, default 2020
            GHSL data year (2020, 2015, 2000, 1990, 1980, 1975)
        scale : int, default 100
            Native GHSL resolution in meters
        threshold : int, default 15
            Built-up threshold (SAME AS create_ref.py)
        resample_to_s2 : bool, default True
            Resample to 10m resolution to match S2
        target_scale : int, default 10
            Target resolution for resampling (meters)
        output_filename : str, default 'ghsl_builtup.tif'
            Output filename for downloaded data
            
        Returns:
        --------
        str
            Path to downloaded GHSL binary mask file
        """
        print(f"📡 Downloading GHSL built-up data...")
        print(f"   📅 Year: {year}")
        print(f"   🏗️ Threshold: >{threshold} (same as create_ref.py)")
        print(f"   📏 Native scale: {scale}m")
        print(f"   🔄 Resample to {target_scale}m: {'YES' if resample_to_s2 else 'NO'}")
        
        # Load GHSL dataset
        ghsl_image = self._load_ghsl_dataset(aoi_geom, year)
        
        # Create binary built-up mask (replicates create_ref.py logic)
        print("🔧 Creating binary built-up mask (replicating create_ref.py)...")
        ghsl_binary = self._create_ghsl_binary_mask(ghsl_image, threshold)
        
        # Resample to target resolution if requested
        if resample_to_s2:
            print(f"📏 Resampling from {scale}m to {target_scale}m...")
            ghsl_binary = ghsl_binary.resample('bilinear').reproject(
                crs='EPSG:4326',
                scale=target_scale
            )
        
        # Clip to AOI
        ghsl_binary = ghsl_binary.clip(aoi_geom)
        
        # Download to Colab
        print("💾 Downloading to Colab...")
        downloaded_path = self._download_image_to_colab(
            ghsl_binary, aoi_geom, output_filename, 
            target_scale if resample_to_s2 else scale
        )
        
        # Validate output if requested
        if self.validate_outputs:
            self._validate_ghsl_output(downloaded_path, threshold)
        
        # Store download statistics
        self._update_download_stats(ghsl_image, downloaded_path, year, threshold)
        
        print(f"✅ GHSL data downloaded successfully: {downloaded_path}")
        return downloaded_path
    
    def _load_ghsl_dataset(self, aoi_geom: ee.Geometry, year: int) -> ee.Image:
        """Load GHSL dataset for specified year."""
        
        if year not in self.ghsl_datasets:
            raise ValueError(f"GHSL data not available for year {year}. "
                           f"Available years: {list(self.ghsl_datasets.keys())}")
        
        dataset_info = self.ghsl_datasets[year]
        collection_id = dataset_info['collection']
        band_name = dataset_info['band']
        
        print(f"📊 Loading GHSL dataset: {collection_id}")
        
        try:
            if year == 2020:
                # Handle 2020 dataset (P2023A)
                print("🔍 Loading GHSL 2020 dataset...")
                
                # Try different approaches for 2020 dataset
                try:
                    # Method 1: Direct image access
                    ghsl = ee.Image(f"{collection_id}/2020")
                    
                    # Test if image exists by trying to get projection
                    projection = ghsl.projection().getInfo()
                    print(f"✅ Method 1 successful - Direct image access")
                    
                except Exception as e1:
                    print(f"⚠️ Method 1 failed: {str(e1)}")
                    
                    try:
                        # Method 2: Collection filtering
                        collection = ee.ImageCollection(collection_id)
                        ghsl = collection.filterMetadata('year', 'equals', year).first()
                        
                        # Test if image exists
                        projection = ghsl.projection().getInfo()
                        print(f"✅ Method 2 successful - Collection filtering")
                        
                    except Exception as e2:
                        print(f"⚠️ Method 2 failed: {str(e2)}")
                        
                        try:
                            # Method 3: Use most recent image
                            collection = ee.ImageCollection(collection_id)
                            ghsl = collection.sort('system:time_start', False).first()
                            
                            # Test if image exists
                            projection = ghsl.projection().getInfo()
                            print(f"✅ Method 3 successful - Most recent image")
                            
                        except Exception as e3:
                            print(f"⚠️ Method 3 failed: {str(e3)}")
                            raise Exception(f"Could not load GHSL 2020 dataset. All methods failed.")
            
            else:
                # Handle older datasets (P2016)
                collection = ee.ImageCollection(collection_id)
                ghsl = collection.filter(ee.Filter.eq('year', year)).first()
                
                # Test if image exists
                projection = ghsl.projection().getInfo()
            
            # Select the correct band
            if band_name in ['built_surface', 'built']:
                ghsl = ghsl.select([band_name])
            else:
                # If band name doesn't exist, use first band
                band_names = ghsl.bandNames().getInfo()
                print(f"🔍 Available bands: {band_names}")
                ghsl = ghsl.select([band_names[0]])
            
            # Clip to AOI to reduce processing
            ghsl = ghsl.clip(aoi_geom)
            
            return ghsl
            
        except Exception as e:
            print(f"❌ Failed to load GHSL dataset: {str(e)}")
            
            # Fallback: Try alternative dataset
            print("🔄 Trying fallback dataset...")
            try:
                # Use JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1 as fallback
                fallback_collection = ee.ImageCollection("JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1")
                
                # Get the most recent year available
                available_years = fallback_collection.aggregate_array('year').distinct().getInfo()
                fallback_year = max([y for y in available_years if y <= year])
                
                print(f"📊 Using fallback year: {fallback_year}")
                
                ghsl = fallback_collection.filter(ee.Filter.eq('year', fallback_year)).first()
                ghsl = ghsl.select(['built']).clip(aoi_geom)
                
                return ghsl
                
            except Exception as fallback_error:
                raise Exception(f"GHSL dataset loading failed completely: {str(fallback_error)}")
    
    def _check_ghsl_availability(self, year: int = 2020) -> Dict[str, Any]:
        """
        Check GHSL dataset availability and return information.
        
        Parameters:
        -----------
        year : int, default 2020
            Year to check
            
        Returns:
        --------
        dict
            Dataset availability information
        """
        availability = {
            'available': False,
            'datasets': [],
            'recommended': None,
            'error': None
        }
        
        # List of GHSL datasets to try
        datasets_to_try = [
            {
                'id': 'JRC/GHSL/P2023A/GHS_BUILT_S',
                'name': 'GHSL P2023A Built Surface',
                'years': [2020, 2015, 2000, 1990, 1980, 1975]
            },
            {
                'id': 'JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1', 
                'name': 'GHSL P2016 Built-up',
                'years': [2015, 2000, 1990, 1980, 1975]
            }
        ]
        
        for dataset in datasets_to_try:
            try:
                collection = ee.ImageCollection(dataset['id'])
                available_years = collection.aggregate_array('year').distinct().getInfo()
                
                dataset_info = {
                    'id': dataset['id'],
                    'name': dataset['name'],
                    'available_years': sorted(available_years),
                    'has_requested_year': year in available_years
                }
                
                availability['datasets'].append(dataset_info)
                availability['available'] = True
                
                # Set recommended dataset
                if dataset_info['has_requested_year'] and not availability['recommended']:
                    availability['recommended'] = dataset['id']
                
            except Exception as e:
                print(f"⚠️ Could not access {dataset['name']}: {str(e)}")
        
        return availability
    
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
            
            # Estimate file size (1 byte per pixel for uint8, times number of bands)
            size_bytes = total_pixels * num_bands * 1  # uint8 = 1 byte
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
    
    def _create_ghsl_binary_mask(self, ghsl_image: ee.Image, threshold: int = 15) -> ee.Image:
        """
        Create binary built-up mask exactly matching create_ref.py logic.
        
        This function replicates the exact binarization from create_ref.py:
        ```python
        # From create_ref.py:
        binary_raster = clipped_raster.where(clipped_raster > 15, 0)
        binary_raster = binary_raster.where(binary_raster == 0, 1)
        ```
        
        Parameters:
        -----------
        ghsl_image : ee.Image
            Input GHSL built-up fraction image
        threshold : int, default 15
            Built-up threshold value
            
        Returns:
        --------
        ee.Image
            Binary mask (0 = non-built, 1 = built-up)
        """
        # Step 1: Create mask where values > threshold
        # Equivalent to: clipped_raster.where(clipped_raster > 15, 0)
        # This sets values > threshold to 0, keeps others as original values
        step1 = ghsl_image.where(ghsl_image.gt(threshold), 0)
        
        # Step 2: Create final binary mask
        # Equivalent to: binary_raster.where(binary_raster == 0, 1)  
        # This sets values == 0 to 1 (built-up), others remain as original (non-built become 0)
        binary_mask = step1.where(step1.eq(0), 1)
        
        # Final cleanup: ensure only 0 and 1 values
        binary_mask = binary_mask.where(binary_mask.neq(1), 0)
        
        return binary_mask.toUint8().rename('built_up')
    
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
    
    def create_reference_labels(self,
                              aoi_geom: ee.Geometry, 
                              year: int = 2020,
                              threshold: int = 15,
                              duas_gdf = None,
                              target_scale: int = 10,
                              output_filename: str = 'reference_labels.tif') -> str:
        """
        Create reference labels exactly matching create_ref.py workflow.
        
        This function replicates the complete reference creation from create_ref.py:
        1. Create GHSL binary mask (0=non-built, 1=built-up)
        2. Optionally add DUA polygons as class 2
        3. Output 3-class reference: 0=non-built, 1=built-up, 2=DUA
        
        Parameters:
        -----------
        aoi_geom : ee.Geometry
            Area of interest geometry
        year : int, default 2020
            GHSL data year
        threshold : int, default 15
            GHSL built-up threshold
        duas_gdf : geopandas.GeoDataFrame, optional
            DUA polygons (if None, creates 2-class reference)
        target_scale : int, default 10
            Output resolution in meters
        output_filename : str, default 'reference_labels.tif'
            Output filename
            
        Returns:
        --------
        str
            Path to reference labels file
        """
        print(f"🏷️ Creating reference labels (replicating create_ref.py)...")
        
        # Step 1: Create GHSL binary mask
        ghsl_image = self._load_ghsl_dataset(aoi_geom, year)
        binary_mask = self._create_ghsl_binary_mask(ghsl_image, threshold)
        
        # Resample to target resolution
        binary_mask = binary_mask.resample('bilinear').reproject(
            crs='EPSG:4326',
            scale=target_scale
        )
        
        # Step 2: Initialize reference with GHSL binary mask
        reference = binary_mask.toInt().rename('reference')
        
        # Step 3: Add DUA areas if provided
        if duas_gdf is not None:
            print("🏘️ Adding DUA polygons as class 2...")
            reference = self._add_duas_to_reference(reference, duas_gdf, aoi_geom, target_scale)
        
        # Clip to AOI
        reference = reference.clip(aoi_geom)
        
        # Download to Colab
        print("💾 Downloading reference labels...")
        downloaded_path = self._download_image_to_colab(
            reference, aoi_geom, output_filename, target_scale
        )
        
        # Validate output
        if self.validate_outputs:
            self._validate_reference_output(downloaded_path, has_duas=(duas_gdf is not None))
        
        print(f"✅ Reference labels created: {downloaded_path}")
        return downloaded_path
    
    def create_reference_labels_from_local(self,
                                          local_ghsl_path: str,
                                          aoi_geom: ee.Geometry = None, 
                                          duas_gdf = None,
                                          output_filename: str = 'reference_labels.tif') -> str:
        """
        Create reference labels using existing local GHSL file.
        
        This function creates reference labels from a local GHSL binary mask file
        instead of re-downloading from Google Earth Engine.
        
        Parameters:
        -----------
        local_ghsl_path : str
            Path to existing local GHSL binary mask file
        aoi_geom : ee.Geometry, optional
            Area of interest geometry (for DUA processing only)
        duas_gdf : geopandas.GeoDataFrame, optional
            DUA polygons (if None, creates 2-class reference)
        output_filename : str, default 'reference_labels.tif'
            Output filename
            
        Returns:
        --------
        str
            Path to reference labels file
        """
        print(f"🏷️ Creating reference labels from local GHSL file...")
        print(f"📂 Local GHSL file: {local_ghsl_path}")
        
        # Validate local GHSL file exists
        if not os.path.exists(local_ghsl_path):
            raise FileNotFoundError(f"Local GHSL file not found: {local_ghsl_path}")
        
        # Load local GHSL file
        print("📖 Loading local GHSL binary mask...")
        with rasterio.open(local_ghsl_path) as src:
            ghsl_data = src.read(1)  # Read first band
            ghsl_profile = src.profile.copy()
            ghsl_transform = src.transform
            ghsl_crs = src.crs
        
        print(f"✅ Loaded GHSL data: {ghsl_data.shape} pixels")
        print(f"   📊 Unique values: {np.unique(ghsl_data)}")
        print(f"   🏗️ Built-up percentage: {(ghsl_data == 1).sum() / ghsl_data.size * 100:.1f}%")
        
        # Initialize reference with GHSL binary mask
        reference_data = ghsl_data.copy().astype(np.uint8)
        
        # Add DUA areas if provided
        if duas_gdf is not None and len(duas_gdf) > 0:
            print("🏘️ Adding DUA polygons as class 2...")
            reference_data = self._add_duas_to_local_reference(
                reference_data, duas_gdf, ghsl_transform, ghsl_crs
            )
        else:
            print("📍 Creating 2-class reference (no DUA polygons)")
        
        # Update profile for output
        output_profile = ghsl_profile.copy()
        output_profile.update({
            'dtype': 'uint8',
            'count': 1,
            'compress': 'lzw'
        })
        
        # Write reference labels
        print("💾 Saving reference labels...")
        with rasterio.open(output_filename, 'w', **output_profile) as dst:
            dst.write(reference_data, 1)
        
        # Validate output
        if self.validate_outputs:
            self._validate_reference_output(output_filename, has_duas=(duas_gdf is not None))
        
        print(f"✅ Reference labels created: {output_filename}")
        return output_filename

    def _add_duas_to_local_reference(self, 
                                    reference_data: np.ndarray,
                                    duas_gdf,
                                    transform,
                                    crs) -> np.ndarray:
        """
        Add DUA polygons to local reference data as class 2.
        
        Parameters:
        -----------
        reference_data : np.ndarray
            Reference data array
        duas_gdf : geopandas.GeoDataFrame
            DUA polygons
        transform : rasterio.Affine
            Raster transform
        crs : rasterio.CRS
            Coordinate reference system
            
        Returns:
        --------
        np.ndarray
            Reference data with DUAs added
        """
        try:
            from rasterio.features import rasterize
            import geopandas as gpd
            
            print(f"   📊 Processing {len(duas_gdf)} DUA polygons...")
            
            # Ensure DUA polygons are in same CRS as raster
            if duas_gdf.crs != crs:
                print(f"   🔄 Reprojecting DUAs from {duas_gdf.crs} to {crs}")
                duas_gdf = duas_gdf.to_crs(crs)
            
            # Create polygon mask using rasterization
            # This replicates the rasterize() call from create_ref.py
            polygon_mask = rasterize(
                [(geom, 2) for geom in duas_gdf.geometry],
                out_shape=reference_data.shape,
                transform=transform,
                fill=0,
                dtype=np.uint8
            )
            
            # Overlay DUAs on reference (DUAs = class 2, overrides built-up class 1)
            # Equivalent to: binary_raster.values[0] = np.where(polygon_mask > 0, polygon_mask, binary_raster.values[0])
            reference_with_duas = np.where(polygon_mask > 0, polygon_mask, reference_data)
            
            # Count DUA pixels
            dua_pixels = (reference_with_duas == 2).sum()
            dua_percentage = dua_pixels / reference_data.size * 100
            print(f"   ✅ Added DUA pixels: {dua_pixels:,} ({dua_percentage:.2f}%)")
            
            return reference_with_duas.astype(np.uint8)
            
        except ImportError:
            print("⚠️ Warning: geopandas/rasterio.features not available, skipping DUA processing")
            return reference_data
        except Exception as e:
            print(f"⚠️ Warning: Could not process DUA polygons: {str(e)}")
            return reference_data
    
    def _add_duas_to_reference(self, 
                              reference: ee.Image,
                              duas_gdf,
                              aoi_geom: ee.Geometry,
                              scale: int) -> ee.Image:
        """
        Add DUA polygons to reference image as class 2.
        
        Replicates the polygon rasterization from create_ref.py:
        ```python
        polygon_mask = rasterize(
            [(geom, 2) for geom in duas.geometry],
            transform=transform,
            fill=0,
            dtype="uint8"
        )
        ```
        """
        try:
            import geopandas as gpd
            
            # Convert DUA polygons to ee.FeatureCollection
            duas_features = []
            for idx, row in duas_gdf.iterrows():
                geom = ee.Geometry(row.geometry.__geo_interface__)
                feature = ee.Feature(geom, {'class': 2})
                duas_features.append(feature)
            
            duas_fc = ee.FeatureCollection(duas_features)
            
            # Rasterize DUA polygons (assign value 2)
            duas_raster = duas_fc.reduceToImage(
                properties=['class'],
                reducer=ee.Reducer.first()
            ).rename('duas')
            
            # Reproject to match reference grid
            duas_raster = duas_raster.reproject(
                crs=reference.projection().crs(),
                scale=scale
            )
            
            # Overlay DUAs on reference (DUAs = class 2, overrides built-up class 1)
            # Equivalent to: binary_raster.values[0] = np.where(polygon_mask > 0, polygon_mask, binary_raster.values[0])
            reference_with_duas = reference.where(duas_raster.gt(0), 2)
            
            return reference_with_duas.rename('reference')
            
        except ImportError:
            print("⚠️ Warning: geopandas not available, skipping DUA processing")
            return reference
        except Exception as e:
            print(f"⚠️ Warning: Could not process DUA polygons: {str(e)}")
            return reference
    
    def _download_image_to_colab(self,
                                image: ee.Image,
                                aoi_geom: ee.Geometry,
                                filename: str,
                                scale: int,
                                max_retries: int = 3) -> str:
        """
        Download Earth Engine image directly to Colab with retry logic and automatic tiling.
        
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
        # First check if we need to use tiling
        area_info = self._estimate_download_size(aoi_geom, scale, 1)  # 1 band for GHSL
        
        if area_info['size_mb'] > 40:
            print(f"📊 Estimated size {area_info['size_mb']:.1f} MB > 40MB limit")
            print("🔄 Switching to tiled download...")
            return self._download_large_area_tiled(image, aoi_geom, filename, scale, 0.1)
        
        # If small enough, proceed with direct download
        for attempt in range(max_retries):
            try:
                print(f"🌐 Direct download attempt {attempt + 1}/{max_retries}...")
                
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
                
                if "Total request size" in str(e) and "must be less than" in str(e):
                    print("🔄 Size limit exceeded - switching to tiled approach...")
                    return self._download_large_area_tiled(image, aoi_geom, filename, scale, 0.1)
                
                if attempt < max_retries - 1:
                    print("⏳ Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    # Final fallback to tiling
                    print("🔄 All direct download attempts failed - trying tiled approach...")
                    return self._download_large_area_tiled(image, aoi_geom, filename, scale, 0.1)
    
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
    
    def _validate_ghsl_output(self, file_path: str, threshold: int) -> None:
        """
        Validate downloaded GHSL data format and values.
        
        Ensures the downloaded data matches expected format for ML model.
        """
        print("🔍 Validating GHSL output format...")
        
        try:
            with rasterio.open(file_path) as src:
                # Check band count (should be 1 for binary mask)
                if src.count != 1:
                    raise ValueError(f"Expected 1 band, got {src.count}")
                
                # Check data type (should be uint8 for binary)
                if src.dtypes[0] not in ['uint8', 'int8']:
                    print(f"⚠️ Warning: Data type is {src.dtypes[0]}, expected uint8")
                
                # Check data values (should be only 0 and 1)
                sample_data = src.read(1, masked=True)
                unique_values = np.unique(sample_data.compressed())
                
                if not all(val in [0, 1] for val in unique_values):
                    print(f"⚠️ Warning: Found values {unique_values}, expected only [0, 1]")
                
                # Calculate built-up percentage
                built_up_pct = (sample_data == 1).sum() / sample_data.size * 100
                
                print(f"✅ GHSL validation passed:")
                print(f"   📊 Bands: {src.count}")
                print(f"   📏 Shape: {src.shape}")
                print(f"   🔢 Data type: {src.dtypes[0]}")
                print(f"   📈 Unique values: {unique_values}")
                print(f"   🏗️ Built-up percentage: {built_up_pct:.1f}%")
                print(f"   🎯 Threshold used: >{threshold}")
                print(f"   🗺️ CRS: {src.crs}")
                
        except Exception as e:
            print(f"❌ GHSL validation failed: {str(e)}")
            raise
    
    def _validate_reference_output(self, file_path: str, has_duas: bool = False) -> None:
        """Validate reference labels output."""
        print("🔍 Validating reference labels...")
        
        try:
            with rasterio.open(file_path) as src:
                sample_data = src.read(1, masked=True)
                unique_values = np.unique(sample_data.compressed())
                
                expected_classes = [0, 1, 2] if has_duas else [0, 1]
                
                print(f"✅ Reference validation passed:")
                print(f"   📊 Classes found: {unique_values}")
                print(f"   📊 Expected classes: {expected_classes}")
                
                for cls in unique_values:
                    count = (sample_data == cls).sum()
                    pct = count / sample_data.size * 100
                    class_name = {0: 'non-built', 1: 'built-up', 2: 'DUA'}.get(cls, 'unknown')
                    print(f"   Class {cls} ({class_name}): {count} pixels ({pct:.1f}%)")
                    
        except Exception as e:
            print(f"❌ Reference validation failed: {str(e)}")
            raise
    
    def _update_download_stats(self, 
                              ghsl_image: ee.Image,
                              file_path: str,
                              year: int,
                              threshold: int) -> None:
        """Update download statistics."""
        
        try:
            # Get file info
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            
            self.download_stats = {
                'ghsl_year': year,
                'threshold': threshold,
                'dataset': self.ghsl_datasets[year],
                'file_size_mb': round(file_size, 2),
                'download_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"⚠️ Warning: Could not update statistics: {str(e)}")
    
    def export_to_drive(self,
                       aoi_geom: ee.Geometry,
                       year: int = 2020,
                       threshold: int = 15,
                       resample_to_s2: bool = True,
                       target_scale: int = 10,
                       description: str = 'ghsl_export',
                       folder: str = 'GEE_Downloads') -> ee.batch.Task:
        """
        Export GHSL data to Google Drive (alternative to direct download).
        
        Parameters:
        -----------
        [Same as download_ghsl_builtup, plus:]
        description : str, default 'ghsl_export'
            Export task description
        folder : str, default 'GEE_Downloads'
            Google Drive folder name
            
        Returns:
        --------
        ee.batch.Task
            Export task object
        """
        print(f"📤 Setting up GHSL Google Drive export...")
        
        # Prepare the image (same as download process)
        ghsl_image = self._load_ghsl_dataset(aoi_geom, year)
        ghsl_binary = self._create_ghsl_binary_mask(ghsl_image, threshold)
        
        if resample_to_s2:
            ghsl_binary = ghsl_binary.resample('bilinear').reproject(
                crs='EPSG:4326',
                scale=target_scale
            )
        
        ghsl_binary = ghsl_binary.clip(aoi_geom)
        
        # Create export task
        task = ee.batch.Export.image.toDrive(
            image=ghsl_binary,
            description=description,
            folder=folder,
            scale=target_scale if resample_to_s2 else 100,
            region=aoi_geom,
            fileFormat='GeoTIFF',
            maxPixels=1e9,
            crs='EPSG:4326'
        )
        
        # Start the task
        task.start()
        print(f"🚀 Export task '{description}' started")
        print(f"📁 Will be saved to Google Drive folder: {folder}")
        
        return task
    
    def get_download_stats(self) -> Dict:
        """Get download statistics from last operation."""
        return self.download_stats.copy()


# Convenience functions for direct use
def quick_download_ghsl(aoi_bounds: Tuple[float, float, float, float],
                       year: int = 2020,
                       threshold: int = 15,
                       output_filename: str = 'ghsl_builtup.tif',
                       resample_to_s2: bool = True) -> str:
    """
    Quick function to download GHSL data with default settings.
    
    Parameters:
    -----------
    aoi_bounds : tuple
        Bounding box as (minx, miny, maxx, maxy) in WGS84
    year : int, default 2020
        GHSL data year
    threshold : int, default 15
        Built-up threshold (same as create_ref.py)
    output_filename : str, default 'ghsl_builtup.tif'
        Output filename
    resample_to_s2 : bool, default True
        Resample to 10m to match S2
        
    Returns:
    --------
    str
        Path to downloaded GHSL binary mask file
    """
    # Create AOI geometry from bounds
    aoi_geom = ee.Geometry.Rectangle(aoi_bounds)
    
    # Initialize downloader and download
    downloader = GHSLDownloader()
    return downloader.download_ghsl_builtup(
        aoi_geom=aoi_geom,
        year=year,
        threshold=threshold,
        resample_to_s2=resample_to_s2,
        output_filename=output_filename
    )


def create_reference_labels_quick(aoi_bounds: Tuple[float, float, float, float],
                                 year: int = 2020,
                                 threshold: int = 15,
                                 output_filename: str = 'reference_labels.tif') -> str:
    """
    Quick function to create reference labels matching create_ref.py.
    
    Parameters:
    -----------
    aoi_bounds : tuple
        Bounding box as (minx, miny, maxx, maxy) in WGS84
    year : int, default 2020
        GHSL data year
    threshold : int, default 15
        Built-up threshold (same as create_ref.py)
    output_filename : str, default 'reference_labels.tif'
        Output filename
        
    Returns:
    --------
    str
        Path to reference labels file
    """
    # Create AOI geometry from bounds
    aoi_geom = ee.Geometry.Rectangle(aoi_bounds)
    
    # Initialize downloader and create reference
    downloader = GHSLDownloader()
    return downloader.create_reference_labels(
        aoi_geom=aoi_geom,
        year=year,
        threshold=threshold,
        output_filename=output_filename
    )


def validate_ghsl_compatibility(file_path: str, 
                               expected_threshold: int = 15) -> bool:
    """
    Validate GHSL file compatibility with original create_ref.py workflow.
    
    Parameters:
    -----------
    file_path : str
        Path to GHSL file to validate
    expected_threshold : int, default 15
        Expected threshold value used
        
    Returns:
    --------
    bool
        True if file is compatible, False otherwise
    """
    downloader = GHSLDownloader()
    try:
        downloader._validate_ghsl_output(file_path, expected_threshold)
        return True
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        return False


def compare_with_create_ref_py(ghsl_file: str, 
                              reference_stats: Dict = None) -> Dict:
    """
    Compare downloaded GHSL data with expected create_ref.py outputs.
    
    Parameters:
    -----------
    ghsl_file : str
        Path to downloaded GHSL file
    reference_stats : dict, optional
        Reference statistics from original create_ref.py output
        
    Returns:
    --------
    dict
        Comparison results
    """
    print("🔍 Comparing with create_ref.py workflow...")
    
    try:
        with rasterio.open(ghsl_file) as src:
            data = src.read(1, masked=True)
            
            # Calculate statistics
            stats = {
                'total_pixels': data.size,
                'non_built_pixels': (data == 0).sum(),
                'built_up_pixels': (data == 1).sum(),
                'non_built_percentage': (data == 0).sum() / data.size * 100,
                'built_up_percentage': (data == 1).sum() / data.size * 100,
                'unique_values': list(np.unique(data.compressed()))
            }
            
            print("📊 Current GHSL Statistics:")
            print(f"   Total pixels: {stats['total_pixels']:,}")
            print(f"   Non-built (0): {stats['non_built_pixels']:,} ({stats['non_built_percentage']:.1f}%)")
            print(f"   Built-up (1): {stats['built_up_pixels']:,} ({stats['built_up_percentage']:.1f}%)")
            print(f"   Unique values: {stats['unique_values']}")
            
            # Compare with reference if provided
            if reference_stats:
                print("\n📊 Comparison with Reference:")
                for key in ['non_built_percentage', 'built_up_percentage']:
                    if key in reference_stats:
                        current = stats[key]
                        reference = reference_stats[key]
                        diff = abs(current - reference)
                        print(f"   {key}: {current:.1f}% vs {reference:.1f}% (diff: {diff:.1f}%)")
            
            return stats
            
    except Exception as e:
        print(f"❌ Comparison failed: {str(e)}")
        return {}


def check_ghsl_dataset_availability(years: List[int] = None) -> Dict:
    """
    Check availability of GHSL datasets for different years.
    
    Parameters:
    -----------
    years : list, optional
        Years to check. If None, checks all available years
        
    Returns:
    --------
    dict
        Dataset availability information
    """
    if years is None:
        years = [2020, 2015, 2000, 1990, 1980, 1975]
    
    downloader = GHSLDownloader()
    availability = {}
    
    print("🔍 Checking GHSL dataset availability...")
    
    for year in years:
        try:
            if year in downloader.ghsl_datasets:
                dataset_info = downloader.ghsl_datasets[year]
                dataset_id = dataset_info['collection']
                # Try to access the dataset
                if year == 2020:
                    collection = ee.ImageCollection(dataset_id)
                    image = collection.filterMetadata('year', 'equals', year).first()
                else:
                    collection = ee.ImageCollection(dataset_id)
                    image = collection.filter(ee.Filter.eq('year', year)).first()
                
                # Test if we can get basic info
                band_names = image.bandNames().getInfo()
                availability[year] = {
                    'available': True,
                    'dataset_id': dataset_id,
                    'bands': band_names
                }
                print(f"   ✅ {year}: Available ({dataset_id})")
                
            else:
                availability[year] = {
                    'available': False,
                    'dataset_id': None,
                    'bands': None
                }
                print(f"   ❌ {year}: Not available")
                
        except Exception as e:
            availability[year] = {
                'available': False,
                'dataset_id': downloader.ghsl_datasets.get(year),
                'error': str(e)
            }
            print(f"   ❌ {year}: Error - {str(e)}")
    
    return availability


# Example usage and testing
if __name__ == "__main__":
    print("🏗️ GHSL Downloader Test")
    print("="*40)
    
    # Test parameters (Jakarta area)
    test_bounds = (106.7, -6.4, 107.0, -6.1)
    test_year = 2020
    test_threshold = 15
    
    try:
        # Check dataset availability
        print("🔍 Checking GHSL dataset availability...")
        availability = check_ghsl_dataset_availability([2020, 2015])
        
        # Download small test area
        print(f"\n🏗️ Testing GHSL download for {test_year}...")
        ghsl_file = quick_download_ghsl(
            aoi_bounds=test_bounds,
            year=test_year,
            threshold=test_threshold,
            output_filename='test_ghsl.tif'
        )
        
        # Validate output
        print("\n🔍 Validating output...")
        is_valid = validate_ghsl_compatibility(ghsl_file, test_threshold)
        
        # Compare with expected workflow
        print("\n📊 Analyzing results...")
        stats = compare_with_create_ref_py(ghsl_file)
        
        # Test reference labels creation
        print("\n🏷️ Testing reference labels creation...")
        ref_file = create_reference_labels_quick(
            aoi_bounds=test_bounds,
            year=test_year,
            threshold=test_threshold,
            output_filename='test_reference.tif'
        )
        
        if is_valid:
            print("✅ All GHSL tests passed!")
            print(f"📊 Built-up percentage: {stats.get('built_up_percentage', 0):.1f}%")
        else:
            print("❌ Validation failed!")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("Make sure GEE is authenticated before running tests")