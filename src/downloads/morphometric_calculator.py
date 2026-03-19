import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal, ogr, osr
import os
import time
import logging
from tqdm import tqdm
import warnings
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
import momepy
from shapely.geometry import Point, box
import gc
import fiona
import pickle
from rasterio import features
from rasterio.transform import Affine

warnings.filterwarnings('ignore')

class MorphometricCalculator:
    """
    Memory-efficient tile-based morphometric calculator for large datasets
    Processes 7.5M+ buildings without running out of RAM
    """
    
    def __init__(self, buildings_geojson_path, reference_raster_path, output_dir, 
                 neighborhood_radius=150, tile_size=1000, checkpoint_freq=5):
        """
        Initialize the morphometric calculator with tile-based processing
        
        Args:
            buildings_geojson_path: Path to buildings GeoJSON
            reference_raster_path: Path to reference raster for matching specs
            output_dir: Directory to save morphometric rasters
            neighborhood_radius: Radius in meters for local variance calculations
            tile_size: Size of spatial tiles in pixels (smaller = less memory)
            checkpoint_freq: Save checkpoint every N tiles
        """
        self.buildings_geojson_path = buildings_geojson_path
        self.reference_raster_path = reference_raster_path
        self.output_dir = output_dir
        self.neighborhood_radius = neighborhood_radius
        self.tile_size = tile_size
        self.checkpoint_freq = checkpoint_freq
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize state variables
        self.reference_metadata = None
        self.pixel_size_x = None
        self.pixel_size_y = None
        self.x_0 = None
        self.y_0 = None
        self.delta_x = None
        self.delta_y = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Morphometric registry
        self.morphometric_registry = {
            'orientation_variance': ('orientation', 'variance'),
            'spacing_irregularity': ('spacing', 'cv'),
            'size_heterogeneity': ('area', 'variance')
        }
        
    def _load_reference_metadata(self):
        """Load metadata from reference raster"""
        self.logger.info("📐 Loading reference raster metadata...")
        
        try:
            dataset = gdal.Open(self.reference_raster_path)
            if dataset is None:
                raise ValueError(f"Cannot open reference raster: {self.reference_raster_path}")
            
            self.reference_metadata = {
                'width': dataset.RasterXSize,
                'height': dataset.RasterYSize,
                'geotransform': dataset.GetGeoTransform(),
                'projection': dataset.GetProjection()
            }
            
            # Extract transformation parameters
            geotransform = self.reference_metadata['geotransform']
            self.x_0 = geotransform[0]
            self.y_0 = geotransform[3]
            self.delta_x = geotransform[1]
            self.delta_y = geotransform[5]
            self.pixel_size_x = abs(self.delta_x)
            self.pixel_size_y = abs(self.delta_y)
            
            self.logger.info(f"   📊 Raster: {self.reference_metadata['width']}x{self.reference_metadata['height']} pixels")
            self.logger.info(f"   🔍 Resolution: {self.pixel_size_x}m x {self.pixel_size_y}m")
            
            dataset = None
            
        except Exception as e:
            self.logger.error(f"❌ Error loading reference metadata: {e}")
            raise
            
    def _create_spatial_tiles(self):
        """Create spatial tiles for processing"""
        self.logger.info(f"🗂️  Creating {self.tile_size}x{self.tile_size} pixel tiles...")
        
        tiles = []
        height, width = self.reference_metadata['height'], self.reference_metadata['width']
        
        for i in range(0, height, self.tile_size):
            for j in range(0, width, self.tile_size):
                i_max = min(i + self.tile_size, height)
                j_max = min(j + self.tile_size, width)
                
                # Convert to geographic coordinates
                x_min = self.x_0 + j * self.delta_x
                x_max = self.x_0 + j_max * self.delta_x
                y_max = self.y_0 + i * self.delta_y
                y_min = self.y_0 + i_max * self.delta_y
                
                tile_info = {
                    'tile_id': len(tiles),
                    'i_start': i, 'i_end': i_max,
                    'j_start': j, 'j_end': j_max,
                    'bbox': box(x_min, y_min, x_max, y_max)
                }
                tiles.append(tile_info)
                
        self.logger.info(f"   📋 Created {len(tiles)} tiles")
        return tiles
        
    def _calculate_momepy_attribute(self, buildings_gdf, attribute_type):
        """Calculate Momepy attribute for buildings in a tile"""
        try:
            if len(buildings_gdf) == 0:
                return pd.Series(dtype=float)
                
            if attribute_type == 'orientation':
                return momepy.Orientation(buildings_gdf).series
            elif attribute_type == 'spacing':
                return momepy.NeighborDistance(buildings_gdf).series
            elif attribute_type == 'area':
                return momepy.Area(buildings_gdf).series
            else:
                raise ValueError(f"Unknown attribute type: {attribute_type}")
                
        except Exception as e:
            self.logger.warning(f"⚠️  Error calculating {attribute_type}: {e}")
            return pd.Series([0] * len(buildings_gdf), dtype=float)
            
    def _process_tile(self, tile_info, attribute_type):
        """Process a single tile: load buildings, calculate Momepy, rasterize"""
        tile_array = np.full((tile_info['i_end'] - tile_info['i_start'],
                             tile_info['j_end'] - tile_info['j_start']), 
                            -9999, dtype=np.float32)
        
        try:
            # Load buildings that intersect this tile
            with fiona.open(self.buildings_geojson_path) as src:
                buildings_in_tile = []
                for feature in src.filter(bbox=tile_info['bbox'].bounds):
                    buildings_in_tile.append(feature)
                    
            if len(buildings_in_tile) == 0:
                return tile_array
                
            # Convert to GeoDataFrame
            buildings_gdf = gpd.GeoDataFrame.from_features(buildings_in_tile)
            
            # Calculate Momepy attribute
            attribute_values = self._calculate_momepy_attribute(buildings_gdf, attribute_type)
            buildings_gdf['attribute'] = attribute_values
            
            # Rasterize to tile
            geometries = [(geom, value) for geom, value in 
                         zip(buildings_gdf.geometry, buildings_gdf['attribute'])
                         if value is not None and not np.isnan(value)]
            
            if len(geometries) > 0:
                # Create transform for this tile
                tile_x_min = self.x_0 + tile_info['j_start'] * self.delta_x
                tile_y_max = self.y_0 + tile_info['i_start'] * self.delta_y
                
                transform = Affine(self.delta_x, 0, tile_x_min,
                                 0, self.delta_y, tile_y_max)
                
                tile_array = features.rasterize(
                    geometries,
                    out_shape=tile_array.shape,
                    transform=transform,
                    fill=-9999,
                    dtype=np.float32
                )
                
        except Exception as e:
            self.logger.warning(f"⚠️  Error processing tile {tile_info['tile_id']}: {e}")
            
        return tile_array
        
    def _create_attribute_raster(self, attribute_type, output_path):
        """Create full raster of building attributes using tile-based processing"""
        self.logger.info(f"🗺️  Creating {attribute_type} raster...")
        
        # Initialize full raster
        full_raster = np.full((self.reference_metadata['height'],
                              self.reference_metadata['width']), 
                             -9999, dtype=np.float32)
        
        # Create tiles
        tiles = self._create_spatial_tiles()
        
        # Process each tile
        with tqdm(total=len(tiles), desc=f"Tiles ({attribute_type})") as pbar:
            for tile in tiles:
                tile_array = self._process_tile(tile, attribute_type)
                
                # Insert tile into full raster
                full_raster[tile['i_start']:tile['i_end'],
                           tile['j_start']:tile['j_end']] = tile_array
                
                pbar.update(1)
                
                # Periodic garbage collection
                if tile['tile_id'] % 10 == 0:
                    gc.collect()
                    
        # Save raster
        self._save_raster(full_raster, output_path)
        
        return full_raster
        
    def _save_raster(self, data, output_path):
        """Save array as GeoTIFF"""
        try:
            driver = gdal.GetDriverByName('GTiff')
            output_ds = driver.Create(
                output_path,
                self.reference_metadata['width'],
                self.reference_metadata['height'],
                1,
                gdal.GDT_Float32,
                options=['COMPRESS=LZW', 'TILED=YES']
            )
            
            output_ds.SetGeoTransform(self.reference_metadata['geotransform'])
            output_ds.SetProjection(self.reference_metadata['projection'])
            
            band = output_ds.GetRasterBand(1)
            band.SetNoDataValue(-9999)
            band.WriteArray(data)
            band.FlushCache()
            
            output_ds = None
            
        except Exception as e:
            self.logger.error(f"❌ Error saving raster: {e}")
            raise
            
    def _calculate_local_statistic(self, input_array, stat_type='variance'):
        """Calculate local statistics using moving window"""
        self.logger.info(f"📊 Calculating local {stat_type}...")
        
        # Calculate neighborhood size in pixels
        neighborhood_pixels = int(self.neighborhood_radius / self.pixel_size_x)
        self.logger.info(f"   🎯 Window: {self.neighborhood_radius}m ({neighborhood_pixels} pixels)")
        
        output_array = np.full_like(input_array, -9999, dtype=np.float32)
        
        from scipy.ndimage import generic_filter
        
        def local_variance_func(values):
            valid = values[values != -9999]
            if len(valid) < 2:
                return -9999
            return np.var(valid)
            
        def local_cv_func(values):
            valid = values[values != -9999]
            if len(valid) < 2:
                return -9999
            mean_val = np.mean(valid)
            if mean_val == 0:
                return -9999
            return np.std(valid) / mean_val
            
        if stat_type == 'variance':
            output_array = generic_filter(input_array, local_variance_func,
                                        size=neighborhood_pixels, mode='constant', cval=-9999)
        elif stat_type == 'cv':
            output_array = generic_filter(input_array, local_cv_func,
                                        size=neighborhood_pixels, mode='constant', cval=-9999)
            
        # Normalize to 0-1
        valid_data = output_array[output_array != -9999]
        if len(valid_data) > 0:
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)
            if max_val > min_val:
                output_array[output_array != -9999] = \
                    (output_array[output_array != -9999] - min_val) / (max_val - min_val)
                    
        return output_array
        
    def calculate_morphometric(self, morphometric_name):
        """Calculate a single morphometric"""
        if morphometric_name not in self.morphometric_registry:
            raise ValueError(f"Unknown morphometric: {morphometric_name}")
            
        self.logger.info(f"🚀 Calculating: {morphometric_name}")
        start_time = time.time()
        
        try:
            # Load reference metadata
            if self.reference_metadata is None:
                self._load_reference_metadata()
                
            attribute_type, stat_type = self.morphometric_registry[morphometric_name]
            
            # Step 1: Create attribute raster
            temp_path = os.path.join(self.output_dir, f"temp_{morphometric_name}.tif")
            attribute_raster = self._create_attribute_raster(attribute_type, temp_path)
            
            # Step 2: Calculate local statistics
            self.logger.info(f"🔄 Calculating local {stat_type}...")
            morphometric_raster = self._calculate_local_statistic(attribute_raster, stat_type)
            
            # Step 3: Save final result
            output_path = os.path.join(self.output_dir, f"{morphometric_name}.tif")
            self._save_raster(morphometric_raster, output_path)
            
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            elapsed = time.time() - start_time
            self.logger.info(f"✅ {morphometric_name} completed in {elapsed:.1f}s")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed {morphometric_name}: {e}")
            raise
            
    def calculate_all_morphometrics(self):
        """Calculate all morphometrics"""
        self.logger.info("🎯 Calculating all morphometrics...")
        start_time = time.time()
        
        results = {}
        
        for morphometric_name in self.morphometric_registry.keys():
            try:
                output_path = self.calculate_morphometric(morphometric_name)
                results[morphometric_name] = output_path
                gc.collect()
            except Exception as e:
                self.logger.error(f"❌ Failed {morphometric_name}: {e}")
                results[morphometric_name] = None
                
        elapsed = time.time() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.logger.info("🎉 === COMPLETED ===")
        self.logger.info(f"   ⏱️  Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        self.logger.info(f"   📊 Success: {len([r for r in results.values() if r])}/{len(results)}")
        
        return results
        
    def validate_setup(self):
        """Validate configuration"""
        self.logger.info("🔍 Validating setup...")
        
        if not os.path.exists(self.buildings_geojson_path):
            self.logger.error(f"❌ Buildings not found: {self.buildings_geojson_path}")
            return False
            
        if not os.path.exists(self.reference_raster_path):
            self.logger.error(f"❌ Reference not found: {self.reference_raster_path}")
            return False
            
        self.logger.info("✅ Validation passed")
        return True


def show_morphometric_stats(morphometric_path):
    """Show statistics for morphometric raster"""
    if not os.path.exists(morphometric_path):
        print(f"❌ File not found: {morphometric_path}")
        return
        
    try:
        dataset = gdal.Open(morphometric_path)
        data = dataset.GetRasterBand(1).ReadAsArray()
        
        valid_data = data[data != -9999]
        
        if len(valid_data) > 0:
            print(f"\n📊 {os.path.basename(morphometric_path)}")
            print(f"   📐 Shape: {data.shape}")
            print(f"   📈 Range: {np.min(valid_data):.4f} - {np.max(valid_data):.4f}")
            print(f"   📋 Valid: {len(valid_data):,} / {data.size:,}")
            print(f"   💾 Size: {os.path.getsize(morphometric_path) / (1024**2):.1f} MB")
            
        dataset = None
        
    except Exception as e:
        print(f"❌ Error: {e}")