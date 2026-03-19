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
warnings.filterwarnings('ignore')

class OptimizedDensityCalculator:
    """
    Calculadora de densidad optimizada para GeoJSONs grandes
    Mantiene exacta compatibilidad con el script original
    """
    
    def __init__(self, sentinel_img_path, buildings_geojson_path, output_path, 
                 tile_size=2000, batch_size=2000, checkpoint_freq=10):
        """
        Inicializa el calculador de densidad optimizado
        
        Args:
            sentinel_img_path: Ruta a la imagen Sentinel base
            buildings_geojson_path: Ruta al GeoJSON de edificios
            output_path: Ruta donde guardar el resultado final
            tile_size: Tamaño de tiles internos para procesamiento
            batch_size: Número de features a procesar por batch
            checkpoint_freq: Frecuencia de guardado de checkpoints
        """
        self.sentinel_img_path = sentinel_img_path
        self.buildings_geojson_path = buildings_geojson_path
        self.output_path = output_path
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.checkpoint_freq = checkpoint_freq
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Variables de estado
        self.density_array = None
        self.image_metadata = None
        self.total_features = 0
        self.processed_features = 0
        
    def _load_image_metadata(self):
        """
        Carga metadatos de la imagen Sentinel sin cargar los datos en memoria
        Mantiene exacta compatibilidad con el método original
        """
        self.logger.info("Cargando metadatos de imagen Sentinel...")
        
        # Abrir imagen (solo metadatos, no datos)
        dataset = gdal.Open(self.sentinel_img_path)
        if dataset is None:
            raise ValueError(f"No se pudo abrir la imagen: {self.sentinel_img_path}")
        
        # Extraer información geoespacial (igual que original)
        self.image_metadata = {
            'width': dataset.RasterXSize,
            'height': dataset.RasterYSize,
            'geotransform': dataset.GetGeoTransform(),
            'projection': dataset.GetProjection()
        }
        
        # Extraer parámetros de transformación (igual que original)
        geoTransform = self.image_metadata['geotransform']
        self.x_0 = geoTransform[0]
        self.y_0 = geoTransform[3] 
        self.delta_x = geoTransform[1]
        self.delta_y = geoTransform[5]
        
        self.logger.info(f"Imagen: {self.image_metadata['width']}x{self.image_metadata['height']} píxeles")
        self.logger.info(f"Resolución: {self.delta_x}m x {abs(self.delta_y)}m")
        
        # Inicializar array de densidad (igual que original)
        self.density_array = np.zeros((self.image_metadata['height'], 
                                     self.image_metadata['width']), 
                                    dtype=np.float32)
        
        dataset = None  # Liberar memoria
        
    def _get_total_features_count(self):
        """
        Cuenta el total de features sin cargar todo en memoria
        """
        self.logger.info("Contando features totales...")
        
        with fiona.open(self.buildings_geojson_path) as src:
            self.total_features = len(src)
            
        self.logger.info(f"Total features encontradas: {self.total_features:,}")
        
    def _create_spatial_tiles(self):
        """
        Crea tiles espaciales para procesamiento eficiente
        """
        self.logger.info("Creando tiles espaciales...")
        
        tiles = []
        height, width = self.image_metadata['height'], self.image_metadata['width']
        
        for i in range(0, height, self.tile_size):
            for j in range(0, width, self.tile_size):
                # Calcular límites del tile en coordenadas de imagen
                i_max = min(i + self.tile_size, height)
                j_max = min(j + self.tile_size, width)
                
                # Convertir a coordenadas geográficas para spatial filter
                x_min = self.x_0 + j * self.delta_x
                x_max = self.x_0 + j_max * self.delta_x
                y_max = self.y_0 + i * self.delta_y  # delta_y es negativo
                y_min = self.y_0 + i_max * self.delta_y
                
                tile_info = {
                    'tile_id': len(tiles),
                    'i_start': i, 'i_end': i_max,
                    'j_start': j, 'j_end': j_max,
                    'bbox': box(x_min, y_min, x_max, y_max)
                }
                tiles.append(tile_info)
                
        self.logger.info(f"Creados {len(tiles)} tiles de procesamiento")
        return tiles
        
    def _process_features_in_tile(self, tile_info):
        """
        Procesa features que intersectan con un tile específico
        Mantiene la misma lógica de cálculo que el script original
        """
        tile_bbox = tile_info['bbox']
        features_processed = 0
        
        # Abrir GeoJSON con spatial filter
        with fiona.open(self.buildings_geojson_path) as src:
            # Filtrar features que intersectan con el tile
            filtered_features = src.filter(bbox=tile_bbox.bounds)
            
            batch_features = []
            for feature in filtered_features:
                batch_features.append(feature)
                
                # Procesar en batches para controlar memoria
                if len(batch_features) >= self.batch_size:
                    features_processed += self._process_feature_batch(batch_features, tile_info)
                    batch_features = []
                    
            # Procesar último batch si queda algo
            if batch_features:
                features_processed += self._process_feature_batch(batch_features, tile_info)
                
        return features_processed
        
    def _process_feature_batch(self, features_batch, tile_info):
        """
        Procesa un batch de features manteniendo exacta compatibilidad
        con la lógica original del script
        """
        features_count = 0
        
        for feature in features_batch:
            try:
                # Crear geometría desde feature (igual que original)
                from shapely.geometry import shape
                geom_shapely = shape(feature['geometry'])
                
                if geom_shapely is not None and not geom_shapely.is_empty:
                    # Obtener envelope (igual que original)
                    bounds = geom_shapely.bounds  # (minx, miny, maxx, maxy)
                    
                    # Calcular índices de píxeles (EXACTAMENTE igual que original)
                    i_min = int(np.floor((bounds[0] - self.x_0) / self.delta_x))
                    i_max = int(np.ceil((bounds[2] - self.x_0) / self.delta_x))
                    j_max = int(np.ceil((bounds[1] - self.y_0) / self.delta_y))
                    j_min = int(np.floor((bounds[3] - self.y_0) / self.delta_y))
                    
                    # Aplicar la misma lógica de incremento que el original
                    for i in range(i_min-1, i_max+1):
                        for j in range(j_min-1, j_max+1):
                            if i > 0 and i < self.density_array.shape[1]:
                                if j > 0 and j < self.density_array.shape[0]:
                                    # Solo procesar si está dentro del tile actual
                                    if (tile_info['i_start'] <= j < tile_info['i_end'] and 
                                        tile_info['j_start'] <= i < tile_info['j_end']):
                                        self.density_array[j, i] += 1
                    
                    features_count += 1
                    
            except Exception as e:
                self.logger.warning(f"Error procesando feature: {e}")
                continue
                
        return features_count
        
    def _save_checkpoint(self, tile_id):
        """
        Guarda checkpoint del progreso actual
        """
        checkpoint_data = {
            'density_array': self.density_array,
            'processed_features': self.processed_features,
            'last_tile': tile_id,
            'image_metadata': self.image_metadata
        }
        
        checkpoint_path = f"{self.output_path}_checkpoint_{tile_id}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        self.logger.info(f"Checkpoint guardado: tile {tile_id}")
        
    def _load_checkpoint(self, checkpoint_path):
        """
        Carga checkpoint previo si existe
        """
        if os.path.exists(checkpoint_path):
            self.logger.info(f"Cargando checkpoint: {checkpoint_path}")
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
                
            self.density_array = checkpoint_data['density_array']
            self.processed_features = checkpoint_data['processed_features']
            self.image_metadata = checkpoint_data['image_metadata']
            
            return checkpoint_data['last_tile']
        return None
        
    def _save_georeferenced_tiff(self, data, filename):
        """
        Guarda array como GeoTIFF georeferenciado
        Mantiene exacta compatibilidad con función original georrefData
        """
        height, width = data.shape
        driver = gdal.GetDriverByName("GTiff")
        
        # Crear dataset con mismas especificaciones que original
        outdata = driver.Create(filename, width, height, 1, gdal.GDT_Float32)
        outdata.SetGeoTransform(self.image_metadata['geotransform'])
        outdata.SetProjection(self.image_metadata['projection'])
        outdata.GetRasterBand(1).WriteArray(data)
        outdata.FlushCache()
        
        # Limpiar referencias (igual que original)
        outdata = None
        
    def calculate_density(self, resume_from_checkpoint=None):
        """
        Método principal para calcular densidad optimizado
        Mantiene exactos los mismos resultados que el script original
        """
        start_time = time.time()
        self.logger.info("Iniciando cálculo de densidad optimizado...")
        
        # Cargar metadatos de imagen
        self._load_image_metadata()
        
        # Contar features totales
        self._get_total_features_count()
        
        # Manejar checkpoint si se especifica
        start_tile = 0
        if resume_from_checkpoint:
            start_tile = self._load_checkpoint(resume_from_checkpoint)
            if start_tile is None:
                self.logger.info("No se encontró checkpoint válido, iniciando desde el principio")
                start_tile = 0
            else:
                self.logger.info(f"Resumiendo desde tile {start_tile}")
        
        # Crear tiles espaciales
        tiles = self._create_spatial_tiles()
        
        # Procesar cada tile
        self.logger.info("Iniciando procesamiento por tiles...")
        
        with tqdm(total=len(tiles), initial=start_tile, desc="Procesando tiles") as pbar:
            for tile_idx in range(start_tile, len(tiles)):
                tile_info = tiles[tile_idx]
                
                # Procesar features en este tile
                features_in_tile = self._process_features_in_tile(tile_info)
                self.processed_features += features_in_tile
                
                # Actualizar progress bar
                pbar.set_postfix({
                    'Features': f"{self.processed_features:,}",
                    'Tile': f"{tile_idx+1}/{len(tiles)}"
                })
                pbar.update(1)
                
                # Guardar checkpoint periódicamente
                if (tile_idx + 1) % self.checkpoint_freq == 0:
                    self._save_checkpoint(tile_idx)
                    
                # Forzar garbage collection para liberar memoria
                if tile_idx % 5 == 0:
                    gc.collect()
                    
        # Aplicar normalización EXACTAMENTE igual que el original
        self.logger.info("Aplicando normalización...")
        min_val = np.min(self.density_array)
        max_val = np.max(self.density_array)
        
        if max_val > min_val:  # Evitar división por cero
            self.density_array = (self.density_array - min_val) / (max_val - min_val)
        else:
            self.logger.warning("Array de densidad tiene valores constantes")
            
        # Guardar resultado final
        self.logger.info(f"Guardando resultado final: {self.output_path}")
        self._save_georeferenced_tiff(self.density_array, self.output_path)
        
        # Estadísticas finales
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.logger.info("=== PROCESAMIENTO COMPLETADO ===")
        self.logger.info(f"Features procesadas: {self.processed_features:,}")
        self.logger.info(f"Tiempo total: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        self.logger.info(f"Densidad mín/máx: {np.min(self.density_array):.4f} / {np.max(self.density_array):.4f}")
        
        # Limpiar checkpoints
        self._cleanup_checkpoints()
        
        return self.density_array
        
    def _cleanup_checkpoints(self):
        """
        Limpia archivos de checkpoint después del procesamiento exitoso
        """
        checkpoint_pattern = f"{self.output_path}_checkpoint_*.pkl"
        import glob
        
        for checkpoint_file in glob.glob(checkpoint_pattern):
            try:
                os.remove(checkpoint_file)
                self.logger.info(f"Checkpoint eliminado: {checkpoint_file}")
            except Exception as e:
                self.logger.warning(f"No se pudo eliminar checkpoint {checkpoint_file}: {e}")


def create_density_optimized(sentinel_img_path, buildings_geojson_path, output_path,
                           tile_size=2000, batch_size=2000, checkpoint_freq=10,
                           resume_from_checkpoint=None):
    """
    Función principal optimizada que replica exactamente los resultados del script original
    pero con manejo eficiente de memoria para GeoJSONs grandes
    
    Args:
        sentinel_img_path: Ruta a imagen Sentinel-2 base
        buildings_geojson_path: Ruta al GeoJSON de edificios (puede ser de varios GB)
        output_path: Ruta donde guardar el GeoTIFF de densidad final
        tile_size: Tamaño de tiles internos (default: 2000 píxeles)
        batch_size: Features por batch (default: 2000)
        checkpoint_freq: Frecuencia de guardado de checkpoints (default: cada 10 tiles)
        resume_from_checkpoint: Ruta a checkpoint para resumir procesamiento
        
    Returns:
        numpy.ndarray: Array de densidad normalizado (0-1)
    """
    
    calculator = OptimizedDensityCalculator(
        sentinel_img_path=sentinel_img_path,
        buildings_geojson_path=buildings_geojson_path,
        output_path=output_path,
        tile_size=tile_size,
        batch_size=batch_size,
        checkpoint_freq=checkpoint_freq
    )
    
    return calculator.calculate_density(resume_from_checkpoint=resume_from_checkpoint)


def validate_inputs(sentinel_img_path, buildings_geojson_path, output_path):
    """
    Valida que los archivos de entrada existan y sean accesibles
    
    Args:
        sentinel_img_path: Ruta a imagen Sentinel
        buildings_geojson_path: Ruta a GeoJSON de edificios  
        output_path: Ruta de salida
        
    Returns:
        bool: True si todos los archivos son válidos
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Validando archivos de entrada...")
    
    # Validar imagen Sentinel
    if not os.path.exists(sentinel_img_path):
        logger.error(f"❌ No se encuentra imagen Sentinel: {sentinel_img_path}")
        return False
    
    # Validar GeoJSON
    if not os.path.exists(buildings_geojson_path):
        logger.error(f"❌ No se encuentra GeoJSON: {buildings_geojson_path}")
        return False
        
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Mostrar información de archivos
    try:
        # Info imagen Sentinel
        dataset = gdal.Open(sentinel_img_path)
        if dataset:
            logger.info(f"✅ Imagen Sentinel: {dataset.RasterXSize}x{dataset.RasterYSize} píxeles")
            logger.info(f"✅ Resolución: {abs(dataset.GetGeoTransform()[1])}m")
            dataset = None
        
        # Info GeoJSON
        file_size_gb = os.path.getsize(buildings_geojson_path) / (1024**3)
        logger.info(f"✅ GeoJSON: {file_size_gb:.2f} GB")
        
        logger.info(f"✅ Salida configurada: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validando archivos: {e}")
        return False


def show_result_stats(output_path):
    """
    Muestra estadísticas del resultado final
    
    Args:
        output_path: Ruta al archivo GeoTIFF de densidad
    """
    if not os.path.exists(output_path):
        print(f"❌ Archivo de salida no encontrado: {output_path}")
        return
        
    try:
        dataset = gdal.Open(output_path)
        if dataset:
            density_data = dataset.GetRasterBand(1).ReadAsArray()
            
            print(f"\n📊 ESTADÍSTICAS DEL RESULTADO:")
            print(f"   - Archivo: {os.path.basename(output_path)}")
            print(f"   - Dimensiones: {density_data.shape}")
            print(f"   - Rango valores: {np.min(density_data):.4f} - {np.max(density_data):.4f}")
            print(f"   - Píxeles con edificaciones: {np.sum(density_data > 0):,}")
            print(f"   - Píxeles totales: {density_data.size:,}")
            print(f"   - Cobertura: {(np.sum(density_data > 0) / density_data.size * 100):.2f}%")
            print(f"   - Tamaño archivo: {os.path.getsize(output_path) / (1024**2):.1f} MB")
            
            dataset = None
            
    except Exception as e:
        print(f"❌ Error leyendo resultado: {e}")


def quick_run(sentinel_img, buildings_geojson, output_tiff, 
              tile_size=2000, batch_size=2000):
    """
    Función simplificada para ejecución rápida desde Jupyter
    
    Args:
        sentinel_img: Ruta a imagen Sentinel
        buildings_geojson: Ruta a GeoJSON de edificios
        output_tiff: Ruta de salida
        tile_size: Tamaño de tiles (default: 2000)
        batch_size: Features por batch (default: 2000)
        
    Returns:
        bool: True si el procesamiento fue exitoso
    """
    try:
        # Validar inputs
        if not validate_inputs(sentinel_img, buildings_geojson, output_tiff):
            return False
            
        print("\n🚀 INICIANDO CÁLCULO DE DENSIDAD OPTIMIZADO")
        print("="*50)
        
        # Ejecutar cálculo
        result = create_density_optimized(
            sentinel_img_path=sentinel_img,
            buildings_geojson_path=buildings_geojson,
            output_path=output_tiff,
            tile_size=tile_size,
            batch_size=batch_size,
            checkpoint_freq=10
        )
        
        print("\n🎉 PROCESAMIENTO COMPLETADO")
        print("="*50)
        
        # Mostrar estadísticas
        show_result_stats(output_tiff)
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante procesamiento: {e}")
        return False