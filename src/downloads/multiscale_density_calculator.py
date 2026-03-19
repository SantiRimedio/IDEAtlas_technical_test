# multiscale_density_calculator.py - VERSIÓN OPTIMIZADA CON GDAL
"""
Calculadora de densidad multi-escala para detección de asentamientos informales
Usa GDAL para resampleo eficiente sin consumir RAM excesiva
Optimizado para Villa 31 (urbano denso) y asentamientos rurales dispersos
"""

import numpy as np
from osgeo import gdal, ogr, osr
import os
import time
import geopandas as gpd
import fiona
from shapely.geometry import box
import pickle
import logging
from tqdm import tqdm
import gc
import warnings
from scipy.ndimage import uniform_filter

warnings.filterwarnings('ignore')

# Intentar importar entropía (opcional)
try:
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    ENTROPY_AVAILABLE = True
except ImportError:
    ENTROPY_AVAILABLE = False


class MultiScaleDensityCalculator:
    """
    Calculadora de densidad multi-escala optimizada con GDAL
    """
    
    def __init__(self, sentinel_img_path, buildings_geojson_path, output_dir, 
                 resolutions=[10, 30, 50, 100, 200],
                 tile_size=2000, batch_size=2000, checkpoint_freq=10):
        """
        Inicializa el calculador multi-escala
        
        Args:
            sentinel_img_path: Ruta a la imagen Sentinel base
            buildings_geojson_path: Ruta al GeoJSON de edificios
            output_dir: Directorio donde guardar todos los resultados
            resolutions: Lista de resoluciones a generar (en metros)
            tile_size: Tamaño de tiles internos para procesamiento
            batch_size: Número de features a procesar por batch
            checkpoint_freq: Frecuencia de guardado de checkpoints
        """
        self.sentinel_img_path = sentinel_img_path
        self.buildings_geojson_path = buildings_geojson_path
        self.output_dir = output_dir
        self.resolutions = resolutions
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.checkpoint_freq = checkpoint_freq
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Variables de estado
        self.density_array = None
        self.image_metadata = None
        self.total_features = 0
        self.processed_features = 0
        self.band_names = []
        
    def _load_image_metadata(self):
        """Carga metadatos de la imagen Sentinel"""
        self.logger.info("📷 Cargando metadatos de imagen Sentinel...")
        
        dataset = gdal.Open(self.sentinel_img_path)
        if dataset is None:
            raise ValueError(f"No se pudo abrir la imagen: {self.sentinel_img_path}")
        
        self.image_metadata = {
            'width': dataset.RasterXSize,
            'height': dataset.RasterYSize,
            'geotransform': dataset.GetGeoTransform(),
            'projection': dataset.GetProjection()
        }
        
        geoTransform = self.image_metadata['geotransform']
        self.x_0 = geoTransform[0]
        self.y_0 = geoTransform[3] 
        self.delta_x = geoTransform[1]
        self.delta_y = geoTransform[5]
        
        self.logger.info(f"   Imagen: {self.image_metadata['width']}x{self.image_metadata['height']} píxeles")
        self.logger.info(f"   Resolución: {self.delta_x}m x {abs(self.delta_y)}m")
        
        self.density_array = np.zeros((self.image_metadata['height'], 
                                     self.image_metadata['width']), 
                                    dtype=np.float32)
        
        dataset = None
        
    def _get_total_features_count(self):
        """Cuenta el total de features"""
        self.logger.info("🔢 Contando features totales...")
        
        with fiona.open(self.buildings_geojson_path) as src:
            self.total_features = len(src)
            
        self.logger.info(f"   Total features: {self.total_features:,}")
        
    def _create_spatial_tiles(self):
        """Crea tiles espaciales para procesamiento eficiente"""
        self.logger.info("🗺️  Creando tiles espaciales...")
        
        tiles = []
        height, width = self.image_metadata['height'], self.image_metadata['width']
        
        for i in range(0, height, self.tile_size):
            for j in range(0, width, self.tile_size):
                i_max = min(i + self.tile_size, height)
                j_max = min(j + self.tile_size, width)
                
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
                
        self.logger.info(f"   Creados {len(tiles)} tiles")
        return tiles
        
    def _process_features_in_tile(self, tile_info):
        """Procesa features que intersectan con un tile"""
        tile_bbox = tile_info['bbox']
        features_processed = 0
        
        with fiona.open(self.buildings_geojson_path) as src:
            filtered_features = src.filter(bbox=tile_bbox.bounds)
            
            batch_features = []
            for feature in filtered_features:
                batch_features.append(feature)
                
                if len(batch_features) >= self.batch_size:
                    features_processed += self._process_feature_batch(batch_features, tile_info)
                    batch_features = []
                    
            if batch_features:
                features_processed += self._process_feature_batch(batch_features, tile_info)
                
        return features_processed
        
    def _process_feature_batch(self, features_batch, tile_info):
        """Procesa un batch de features"""
        features_count = 0
        
        for feature in features_batch:
            try:
                from shapely.geometry import shape
                geom_shapely = shape(feature['geometry'])
                
                if geom_shapely is not None and not geom_shapely.is_empty:
                    bounds = geom_shapely.bounds
                    
                    i_min = int(np.floor((bounds[0] - self.x_0) / self.delta_x))
                    i_max = int(np.ceil((bounds[2] - self.x_0) / self.delta_x))
                    j_max = int(np.ceil((bounds[1] - self.y_0) / self.delta_y))
                    j_min = int(np.floor((bounds[3] - self.y_0) / self.delta_y))
                    
                    for i in range(i_min-1, i_max+1):
                        for j in range(j_min-1, j_max+1):
                            if i > 0 and i < self.density_array.shape[1]:
                                if j > 0 and j < self.density_array.shape[0]:
                                    if (tile_info['i_start'] <= j < tile_info['i_end'] and 
                                        tile_info['j_start'] <= i < tile_info['j_end']):
                                        self.density_array[j, i] += 1
                    
                    features_count += 1
                    
            except Exception as e:
                self.logger.warning(f"Error procesando feature: {e}")
                continue
                
        return features_count
    
    def _calculate_base_density(self):
        """Calcula la densidad base"""
        start_time = time.time()
        self.logger.info("🏗️  PASO 1/4: Calculando densidad base...")
        
        self._load_image_metadata()
        self._get_total_features_count()
        
        tiles = self._create_spatial_tiles()
        
        self.logger.info("   Procesando tiles...")
        with tqdm(total=len(tiles), desc="   Tiles") as pbar:
            for tile_idx, tile_info in enumerate(tiles):
                features_in_tile = self._process_features_in_tile(tile_info)
                self.processed_features += features_in_tile
                
                pbar.set_postfix({
                    'Features': f"{self.processed_features:,}"
                })
                pbar.update(1)
                
                if tile_idx % 5 == 0:
                    gc.collect()
        
        # Normalizar
        self.logger.info("   Normalizando densidad base...")
        min_val = np.min(self.density_array)
        max_val = np.max(self.density_array)
        
        if max_val > min_val:
            self.density_array = (self.density_array - min_val) / (max_val - min_val)
        
        elapsed = time.time() - start_time
        self.logger.info(f"   ✅ Densidad base completada en {elapsed/60:.1f} min")
        
        # Guardar densidad base inmediatamente
        base_path = os.path.join(self.output_dir, 'density_base.tif')
        self._save_georeferenced_tiff(self.density_array, base_path)
        self.logger.info(f"   💾 Densidad base guardada: density_base.tif")
        
        return self.density_array
    
    def _save_georeferenced_tiff(self, data, filename):
        """Guarda array como GeoTIFF georeferenciado"""
        height, width = data.shape
        driver = gdal.GetDriverByName("GTiff")
        
        outdata = driver.Create(filename, width, height, 1, gdal.GDT_Float32,
                               options=['COMPRESS=LZW', 'TILED=YES'])
        outdata.SetGeoTransform(self.image_metadata['geotransform'])
        outdata.SetProjection(self.image_metadata['projection'])
        outdata.GetRasterBand(1).WriteArray(data)
        outdata.FlushCache()
        outdata = None
        
        gc.collect()
    
    def _process_and_save_multiscale_bands(self):
        """
        Genera bandas multi-escala usando GDAL (eficiente en memoria)
        """
        self.logger.info("📊 PASO 2/4: Generando bandas multi-escala con GDAL...")
        
        base_res = abs(self.delta_x)
        base_path = os.path.join(self.output_dir, 'density_base.tif')
        
        for res in self.resolutions:
            self.logger.info(f"   → Procesando density_{res}m")
            
            if abs(res - base_res) < 1:
                # Es la resolución base
                band_name = f'density_{res}m'
                output_path = os.path.join(self.output_dir, f'{band_name}.tif')
                
                import shutil
                shutil.copy(base_path, output_path)
                
                self.band_names.append(band_name)
                
            else:
                # Usar GDAL para resampleo eficiente
                band_name = f'density_{res}m'
                output_path = os.path.join(self.output_dir, f'{band_name}.tif')
                
                self._create_aggregated_band_gdal(base_path, output_path, band_name, res, base_res)
                
                self.band_names.append(band_name)
                
                gc.collect()
        
        self.logger.info(f"   ✅ {len(self.resolutions)} bandas multi-escala guardadas")
    
    def _create_aggregated_band_gdal(self, input_path, output_path, band_name, target_res, base_res):
        """
        Crea banda agregada usando GDAL (mucho más eficiente que scipy)
        """
        factor = target_res / base_res
        
        # Abrir imagen base
        src_ds = gdal.Open(input_path)
        
        # Calcular nuevas dimensiones para downsample
        new_width = int(src_ds.RasterXSize / factor)
        new_height = int(src_ds.RasterYSize / factor)
        
        # Crear archivo temporal downsampled
        temp_path = output_path.replace('.tif', '_temp.tif')
        
        self.logger.info(f"      Downsampling {band_name}...")
        
        # Downsample usando gdalwarp con AVERAGE resampling
        gdal.Warp(
            temp_path,
            src_ds,
            width=new_width,
            height=new_height,
            resampleAlg=gdal.GRA_Average,
            creationOptions=['COMPRESS=LZW', 'TILED=YES'],
            options=['NUM_THREADS=ALL_CPUS']
        )
        
        src_ds = None
        gc.collect()
        
        self.logger.info(f"      Upsampling {band_name}...")
        
        # Upsample de vuelta usando NEAREST NEIGHBOR (mantiene bloques)
        temp_ds = gdal.Open(temp_path)
        
        gdal.Warp(
            output_path,
            temp_ds,
            width=self.image_metadata['width'],
            height=self.image_metadata['height'],
            resampleAlg=gdal.GRA_NearestNeighbour,
            creationOptions=['COMPRESS=LZW', 'TILED=YES'],
            options=['NUM_THREADS=ALL_CPUS']
        )
        
        temp_ds = None
        gc.collect()
        
        # Eliminar temporal
        try:
            os.remove(temp_path)
        except:
            pass
        
        # Normalizar
        self.logger.info(f"      Normalizando {band_name}...")
        ds = gdal.Open(output_path, gdal.GA_Update)
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val > min_val:
            data = (data - min_val) / (max_val - min_val)
            band.WriteArray(data)
        
        band.FlushCache()
        ds.FlushCache()
        ds = None
        
        del data
        gc.collect()
    
    def _process_and_save_derived_bands(self):
        """
        Procesa y guarda bandas derivadas una por una
        """
        self.logger.info("🔧 PASO 3/4: Calculando bandas derivadas...")
        
        base = self.density_array
        
        # 1. Varianza local
        self.logger.info("   → Procesando variance")
        mean = uniform_filter(base, size=15)
        mean_sq = uniform_filter(base**2, size=15)
        variance = mean_sq - mean**2
        variance = np.clip(variance, 0, None)
        
        # Normalizar y guardar
        min_val = np.min(variance)
        max_val = np.max(variance)
        if max_val > min_val:
            variance = (variance - min_val) / (max_val - min_val)
        
        output_path = os.path.join(self.output_dir, 'variance.tif')
        self._save_georeferenced_tiff(variance, output_path)
        self.band_names.append('variance')
        
        del variance, mean, mean_sq
        gc.collect()
        
        # 2. Gradiente
        self.logger.info("   → Procesando gradient")
        grad_y, grad_x = np.gradient(base)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalizar y guardar
        min_val = np.min(gradient)
        max_val = np.max(gradient)
        if max_val > min_val:
            gradient = (gradient - min_val) / (max_val - min_val)
        
        output_path = os.path.join(self.output_dir, 'gradient.tif')
        self._save_georeferenced_tiff(gradient, output_path)
        self.band_names.append('gradient')
        
        del gradient, grad_x, grad_y
        gc.collect()
        
        # 3. Ratio local/regional
        self.logger.info("   → Procesando ratio_local_regional")
        local = uniform_filter(base, size=10)
        regional = uniform_filter(base, size=50)
        ratio = np.divide(local, regional + 1e-6)
        
        # Normalizar y guardar
        min_val = np.min(ratio)
        max_val = np.max(ratio)
        if max_val > min_val:
            ratio = (ratio - min_val) / (max_val - min_val)
        
        output_path = os.path.join(self.output_dir, 'ratio_local_regional.tif')
        self._save_georeferenced_tiff(ratio, output_path)
        self.band_names.append('ratio_local_regional')
        
        del ratio, local, regional
        gc.collect()
        
        # 4. Entropía (opcional)
        if ENTROPY_AVAILABLE:
            self.logger.info("   → Procesando entropy")
            try:
                base_uint = ((base - base.min()) / (base.max() - base.min() + 1e-6) * 255).astype(np.uint8)
                ent = entropy(base_uint, disk(7))
                ent = ent / np.log(256)
                
                # Normalizar y guardar
                min_val = np.min(ent)
                max_val = np.max(ent)
                if max_val > min_val:
                    ent = (ent - min_val) / (max_val - min_val)
                
                output_path = os.path.join(self.output_dir, 'entropy.tif')
                self._save_georeferenced_tiff(ent, output_path)
                self.band_names.append('entropy')
                
                del ent, base_uint
                gc.collect()
                
            except Exception as e:
                self.logger.warning(f"   ⚠️  No se pudo calcular entropía: {e}")
        else:
            self.logger.warning("   ⚠️  Entropía no disponible (falta skimage.filters.rank)")
        
        derived_count = len([n for n in self.band_names if n.startswith(('variance', 'gradient', 'ratio', 'entropy'))])
        self.logger.info(f"   ✅ {derived_count} bandas derivadas guardadas")
    
    def _create_multiband_stack(self):
        """
        Crea stack multi-banda leyendo archivos individuales
        """
        self.logger.info("📚 PASO 4/4: Creando stack multi-banda...")
        
        h, w = self.image_metadata['height'], self.image_metadata['width']
        n_bands = len(self.band_names)
        
        self.logger.info(f"   Combinando {n_bands} bandas...")
        
        # Crear archivo multi-banda
        stack_path = os.path.join(self.output_dir, 'density_stack_multiband.tif')
        driver = gdal.GetDriverByName("GTiff")
        
        out_ds = driver.Create(stack_path, w, h, n_bands, gdal.GDT_Float32,
                              options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])
        out_ds.SetGeoTransform(self.image_metadata['geotransform'])
        out_ds.SetProjection(self.image_metadata['projection'])
        
        # Copiar cada banda
        for i, band_name in enumerate(tqdm(self.band_names, desc="   Copiando bandas")):
            band_path = os.path.join(self.output_dir, f'{band_name}.tif')
            
            # Leer banda
            src_ds = gdal.Open(band_path)
            data = src_ds.GetRasterBand(1).ReadAsArray()
            src_ds = None
            
            # Escribir en stack
            out_band = out_ds.GetRasterBand(i + 1)
            out_band.WriteArray(data)
            out_band.SetDescription(band_name)
            out_band.FlushCache()
            
            # Liberar
            del data
            out_band = None
            gc.collect()
        
        out_ds.FlushCache()
        out_ds = None
        
        self.logger.info(f"   ✅ Stack multi-banda creado")
        
        return (h, w, n_bands)
    
    def calculate_all(self):
        """
        Método principal optimizado con GDAL
        """
        total_start = time.time()
        
        self.logger.info("="*60)
        self.logger.info("🚀 CÁLCULO MULTI-ESCALA CON GDAL")
        self.logger.info("="*60)
        
        # Paso 1: Densidad base
        self._calculate_base_density()
        
        # Paso 2: Bandas multi-escala con GDAL
        self._process_and_save_multiscale_bands()
        
        # Paso 3: Bandas derivadas
        self._process_and_save_derived_bands()
        
        # Liberar densidad base antes del paso final
        self.logger.info("\n🧹 Liberando memoria...")
        del self.density_array
        self.density_array = None
        gc.collect()
        
        # Paso 4: Crear stack final
        stack_shape = self._create_multiband_stack()
        
        # Estadísticas finales
        total_elapsed = time.time() - total_start
        hours, remainder = divmod(total_elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.logger.info("\n" + "="*60)
        self.logger.info("🎉 PROCESAMIENTO COMPLETADO")
        self.logger.info("="*60)
        self.logger.info(f"⏱️  Tiempo total: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        self.logger.info(f"🏗️  Features procesadas: {self.processed_features:,}")
        self.logger.info(f"📊 Bandas generadas: {len(self.band_names)}")
        self.logger.info(f"📦 Stack shape: {stack_shape}")
        self.logger.info(f"📁 Directorio salida: {self.output_dir}")
        self.logger.info("\n📋 Bandas disponibles:")
        for i, name in enumerate(self.band_names, 1):
            self.logger.info(f"   {i:2d}. {name}")
        
        return stack_shape, self.band_names


def create_multiscale_density_stack(sentinel_img_path, buildings_geojson_path, 
                                    output_dir, resolutions=[10, 30, 50, 100, 200],
                                    tile_size=2000, batch_size=2000):
    """
    Función principal optimizada con GDAL
    """
    calculator = MultiScaleDensityCalculator(
        sentinel_img_path=sentinel_img_path,
        buildings_geojson_path=buildings_geojson_path,
        output_dir=output_dir,
        resolutions=resolutions,
        tile_size=tile_size,
        batch_size=batch_size
    )
    
    return calculator.calculate_all()


def quick_run_multiscale(sentinel_img, buildings_geojson, output_dir,
                         resolutions=[10, 30, 50, 100, 200],
                         tile_size=2000, batch_size=2000):
    """
    Función simplificada optimizada con GDAL
    """
    try:
        print("\n🚀 INICIANDO CÁLCULO MULTI-ESCALA CON GDAL")
        print("="*60)
        
        if not os.path.exists(sentinel_img):
            print(f"❌ No se encuentra imagen Sentinel: {sentinel_img}")
            return None, None
        
        if not os.path.exists(buildings_geojson):
            print(f"❌ No se encuentra GeoJSON: {buildings_geojson}")
            return None, None
        
        stack_shape, band_names = create_multiscale_density_stack(
            sentinel_img_path=sentinel_img,
            buildings_geojson_path=buildings_geojson,
            output_dir=output_dir,
            resolutions=resolutions,
            tile_size=tile_size,
            batch_size=batch_size
        )
        
        print("\n🎉 PROCESAMIENTO COMPLETADO")
        print("="*60)
        print(f"📦 Stack shape: {stack_shape}")
        print(f"📁 Archivos guardados en: {output_dir}")
        print(f"\n🎯 Archivo principal para tu modelo:")
        print(f"   {output_dir}/density_stack_multiband.tif")
        
        return stack_shape, band_names
        
    except Exception as e:
        print(f"\n❌ Error durante procesamiento: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def show_multiscale_stats(output_dir):
    """Muestra estadísticas de las bandas generadas"""
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS DE BANDAS MULTI-ESCALA")
    print("="*60)
    
    stack_path = os.path.join(output_dir, 'density_stack_multiband.tif')
    
    if not os.path.exists(stack_path):
        print(f"❌ No se encuentra el stack: {stack_path}")
        return
    
    try:
        dataset = gdal.Open(stack_path)
        
        print(f"\n📦 Stack multi-banda:")
        print(f"   Dimensiones: {dataset.RasterXSize} x {dataset.RasterYSize}")
        print(f"   Bandas: {dataset.RasterCount}")
        
        print(f"\n📋 Estadísticas por banda:")
        print("-"*60)
        
        for i in range(dataset.RasterCount):
            band = dataset.GetRasterBand(i + 1)
            band_name = band.GetDescription()
            data = band.ReadAsArray()
            
            print(f"\n{i+1:2d}. {band_name}")
            print(f"    Min:    {np.min(data):.4f}")
            print(f"    Max:    {np.max(data):.4f}")
            print(f"    Mean:   {np.mean(data):.4f}")
            print(f"    Std:    {np.std(data):.4f}")
            print(f"    Non-0:  {np.sum(data > 0):,} ({np.sum(data > 0)/data.size*100:.1f}%)")
            
            del data
            gc.collect()
        
        dataset = None
        
    except Exception as e:
        print(f"❌ Error leyendo estadísticas: {e}")


def validate_multiscale_inputs(sentinel_img_path, buildings_geojson_path, output_dir):
    """Valida archivos de entrada"""
    print("🔍 Validando archivos de entrada...")
    
    if not os.path.exists(sentinel_img_path):
        print(f"❌ No se encuentra imagen Sentinel: {sentinel_img_path}")
        return False
    
    if not os.path.exists(buildings_geojson_path):
        print(f"❌ No se encuentra GeoJSON: {buildings_geojson_path}")
        return False
        
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        dataset = gdal.Open(sentinel_img_path)
        if dataset:
            print(f"✅ Imagen Sentinel: {dataset.RasterXSize}x{dataset.RasterYSize} píxeles")
            print(f"✅ Resolución: {abs(dataset.GetGeoTransform()[1])}m")
            dataset = None
        
        file_size_gb = os.path.getsize(buildings_geojson_path) / (1024**3)
        print(f"✅ GeoJSON: {file_size_gb:.2f} GB")
        print(f"✅ Directorio salida: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error validando archivos: {e}")
        return False