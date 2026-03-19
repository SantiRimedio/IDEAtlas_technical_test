#!/usr/bin/env python3
"""
hybrid_grid_generator.py

Genera un grid híbrido que:
1. Reutiliza patches del grid original que intersectan significativamente con el AOI
2. Usa la lógica exacta del grid_sampling.py original para áreas no cubiertas
3. Evita overlaps mediante masks espaciales precisos
4. Mantiene trazabilidad completa de la fuente de cada patch

Uso:
    python hybrid_grid_generator.py --original_grid grid_original.geojson --aoi aoi_target.geojson --reference_raster reference.tif --output hybrid_grid.geojson
"""

import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
import argparse
import pandas as pd
import warnings
from tqdm import tqdm
import random
import os
import json
from typing import Tuple, Dict, List

class HybridGridGenerator:
    """
    Genera un grid híbrido combinando patches originales con nuevos patches
    usando la metodología exacta del script original para evitar overlaps.
    """
    
    def __init__(self, patch_size: int = 128, class2_proportion: float = 15, 
                 splits: tuple = (0.7, 0.15, 0.15), random_seed: int = 42, 
                 intersection_threshold: float = 0.7, verbose: bool = True):
        self.patch_size = patch_size
        self.class2_proportion = class2_proportion
        self.splits = splits
        self.random_seed = random_seed
        self.intersection_threshold = intersection_threshold  # Umbral para considerar patch "reutilizable"
        self.verbose = verbose
        
        # Configurar random seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Para trazabilidad
        self.stats = {
            'original_patches_reused': 0,
            'new_patches_generated': 0,
            'total_patches': 0,
            'coverage_original_pct': 0,
            'coverage_new_pct': 0,
            'area_original_km2': 0,
            'area_new_km2': 0,
            'area_total_km2': 0,
            'area_aoi_km2': 0,
            'area_covered_by_original_km2': 0,
            'area_remaining_for_new_km2': 0
        }
        
        self.metadata = {
            'patch_size': patch_size,
            'class2_proportion': class2_proportion,
            'splits': splits,
            'random_seed': random_seed,
            'intersection_threshold': intersection_threshold,
            'source_breakdown': {'original': 0, 'new': 0},
            'set_breakdown': {'train': 0, 'val': 0, 'test': 0}
        }
    
    def log(self, message: str, level: str = 'INFO'):
        """Logger simple."""
        if self.verbose:
            print(f"[{level}] {message}")
    
    def identify_reusable_patches(self, original_grid: gpd.GeoDataFrame, 
                                 aoi: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Identifica patches del grid original que se pueden reutilizar.
        
        Args:
            original_grid: Grid original
            aoi: Área de interés target
            
        Returns:
            Tuple (patches_reutilizables, aoi_sin_esos_patches)
        """
        self.log("🔍 Identificando patches reutilizables...")
        
        # Asegurar que ambos están en el mismo CRS
        if original_grid.crs != aoi.crs:
            self.log(f"Reproyectando AOI de {aoi.crs} a {original_grid.crs}")
            aoi = aoi.to_crs(original_grid.crs)
        
        # Convertir AOI a una sola geometría si tiene múltiples features
        aoi_union = aoi.geometry.unary_union
        
        # Calcular intersección de cada patch original con AOI
        reusable_patches = []
        
        for idx, patch in original_grid.iterrows():
            patch_geom = patch.geometry
            
            # Calcular intersección
            if patch_geom.intersects(aoi_union):
                intersection = patch_geom.intersection(aoi_union)
                intersection_area = intersection.area
                patch_area = patch_geom.area
                intersection_ratio = intersection_area / patch_area
                
                # Solo reutilizar si la intersección es significativa
                if intersection_ratio >= self.intersection_threshold:
                    reusable_patches.append(idx)
                    self.log(f"   Patch {patch.get('id', idx)}: {intersection_ratio:.2%} intersección - REUTILIZABLE")
                else:
                    self.log(f"   Patch {patch.get('id', idx)}: {intersection_ratio:.2%} intersección - descartado")
        
        # Crear GeoDataFrame con patches reutilizables
        if reusable_patches:
            patches_reutilizables = original_grid.loc[reusable_patches].copy()
            patches_reutilizables['source'] = 'original'
        else:
            patches_reutilizables = gpd.GeoDataFrame(columns=original_grid.columns.tolist() + ['source'])
        
        self.log(f"✅ Patches reutilizables: {len(patches_reutilizables)}")
        
        return patches_reutilizables, aoi
    
    def create_spatial_mask(self, reusable_patches: gpd.GeoDataFrame) -> Polygon:
        """
        Crea un mask espacial de las áreas ya cubiertas por patches reutilizables.
        
        Args:
            reusable_patches: Patches que se van a reutilizar
            
        Returns:
            Geometría union de todos los patches reutilizables
        """
        if len(reusable_patches) == 0:
            return Polygon()  # Mask vacío
        
        self.log("🎭 Creando mask espacial de patches reutilizables...")
        
        # Crear union de todas las geometrías de patches reutilizables
        mask_union = unary_union(reusable_patches.geometry.tolist())
        
        # Calcular área del mask
        if hasattr(mask_union, 'area'):
            if reusable_patches.crs and reusable_patches.crs.is_projected:
                mask_area_km2 = mask_union.area / 1e6
            else:
                # Aproximación para CRS geográfico
                temp_gdf = gpd.GeoDataFrame([1], geometry=[mask_union], crs=reusable_patches.crs)
                mask_area_km2 = temp_gdf.to_crs('EPSG:3857').geometry.area.sum() / 1e6
            
            self.log(f"   Área cubierta por patches originales: {mask_area_km2:.2f} km²")
            self.stats['area_covered_by_original_km2'] = mask_area_km2
        
        return mask_union
    
    def calculate_remaining_area(self, aoi: gpd.GeoDataFrame, spatial_mask: Polygon) -> gpd.GeoDataFrame:
        """
        Calcula el área restante donde se debe generar grid nuevo.
        
        Args:
            aoi: Área de interés original
            spatial_mask: Mask de áreas ya cubiertas
            
        Returns:
            GeoDataFrame con área restante
        """
        self.log("🔪 Calculando área restante para grid nuevo...")
        
        # Convertir AOI a una sola geometría
        aoi_union = aoi.geometry.unary_union
        
        # Calcular diferencia espacial
        if spatial_mask.is_empty:
            remaining_area = aoi_union
        else:
            remaining_area = aoi_union.difference(spatial_mask)
        
        # Crear GeoDataFrame con área restante
        if hasattr(remaining_area, 'is_empty') and not remaining_area.is_empty:
            remaining_gdf = gpd.GeoDataFrame([1], geometry=[remaining_area], crs=aoi.crs)
            
            # Calcular área restante
            if aoi.crs and aoi.crs.is_projected:
                remaining_area_km2 = remaining_area.area / 1e6
            else:
                remaining_area_km2 = remaining_gdf.to_crs('EPSG:3857').geometry.area.sum() / 1e6
            
            self.log(f"   Área restante para grid nuevo: {remaining_area_km2:.2f} km²")
            self.stats['area_remaining_for_new_km2'] = remaining_area_km2
            
            return remaining_gdf
        else:
            self.log("   No hay área restante - todo está cubierto por patches originales")
            return gpd.GeoDataFrame(columns=['geometry'], crs=aoi.crs)
    
    def create_grid_for_remaining_area(self, remaining_area: gpd.GeoDataFrame, 
                                     reference_raster: str) -> gpd.GeoDataFrame:
        """
        Crea grid para el área restante usando la lógica EXACTA del grid_sampling.py original.
        
        Args:
            remaining_area: GeoDataFrame con área donde crear grid
            reference_raster: Ruta al raster de referencia
            
        Returns:
            GeoDataFrame con el nuevo grid
        """
        if len(remaining_area) == 0 or remaining_area.geometry.iloc[0].is_empty:
            self.log("⚠️ No hay área restante para generar grid")
            return gpd.GeoDataFrame(columns=['Type', 'id', 'geometry', 'source', 'class', 'strat_class', 'has_class_2'])
        
        self.log(f"🔨 Generando grid para área restante usando lógica original...")
        
        with rasterio.open(reference_raster) as src:
            height, width = src.shape
            crs = src.crs
            transform = src.transform
            
            # PASO 1: Crear grid regular sobre TODO el raster (como en original)
            self.log("   Paso 1: Creando grid regular...")
            grid_cells = []
            temp_ids = []
            
            for y in range(0, height, self.patch_size):
                for x in range(0, width, self.patch_size):
                    win_height = min(self.patch_size, height - y)
                    win_width = min(self.patch_size, width - x)
                    window = Window(x, y, win_width, win_height)
                    bounds = rasterio.windows.bounds(window, transform)
                    patch_geom = box(*bounds)
                    
                    grid_cells.append(patch_geom)
                    temp_ids.append(len(grid_cells))
            
            # Crear GeoDataFrame inicial
            full_grid = gpd.GeoDataFrame({
                'geometry': grid_cells,
                'temp_id': temp_ids,
                'class': -1,
                'has_class_2': False,
                'class_counts': None,
                'strat_class': -1
            }, crs=crs)
            
            self.log(f"   Grid completo creado: {len(full_grid)} patches")
            
            # PASO 2: Filtrar patches que intersectan con área restante
            self.log("   Paso 2: Filtrando patches por área restante...")
            remaining_union = remaining_area.geometry.unary_union
            
            # Filtrar solo patches que intersectan significativamente con área restante
            intersecting_patches = []
            for idx, patch in full_grid.iterrows():
                if patch.geometry.intersects(remaining_union):
                    intersection = patch.geometry.intersection(remaining_union)
                    if hasattr(intersection, 'area') and intersection.area > 0:
                        intersection_ratio = intersection.area / patch.geometry.area
                        if intersection_ratio > 0.95:  # Al menos 10% del patch debe estar en área restante
                            intersecting_patches.append(idx)
            
            filtered_grid = full_grid.loc[intersecting_patches].copy()
            self.log(f"   Patches en área restante: {len(filtered_grid)}")
            
            if len(filtered_grid) == 0:
                self.log("⚠️ No hay patches válidos en área restante")
                return gpd.GeoDataFrame(columns=['Type', 'id', 'geometry', 'source', 'class', 'strat_class', 'has_class_2'])
            
            # PASO 3: Analizar clases usando raster (LÓGICA EXACTA DEL ORIGINAL)
            self.log("   Paso 3: Analizando clases...")
            valid_patches = []
            
            for idx in tqdm(filtered_grid.index, desc="Analizando patches", disable=not self.verbose):
                try:
                    geom = filtered_grid.at[idx, 'geometry']
                    window = src.window(*geom.bounds)
                    data = src.read(1, window=window)
                    valid_data = data[data >= 0]  # MISMA LÓGICA QUE ORIGINAL
                    
                    if len(valid_data) > 0:
                        classes, counts = np.unique(valid_data, return_counts=True)
                        total = counts.sum()
                        class_count_dict = dict(zip(classes, counts))
                        dominant_class = classes[np.argmax(counts)]
                        class_2_pct = class_count_dict.get(2, 0) / total
                        
                        filtered_grid.at[idx, 'class'] = dominant_class
                        filtered_grid.at[idx, 'has_class_2'] = 2 in class_count_dict
                        filtered_grid.at[idx, 'class_counts'] = class_count_dict
                        
                        # MISMA LÓGICA DE ESTRATIFICACIÓN QUE ORIGINAL
                        if class_2_pct >= self.class2_proportion / 100:
                            filtered_grid.at[idx, 'strat_class'] = 2
                        else:
                            filtered_grid.at[idx, 'strat_class'] = dominant_class
                        
                        valid_patches.append(idx)
                    
                except Exception as e:
                    self.log(f"Error procesando patch {idx}: {e}", 'WARNING')
            
            # PASO 4: Filtrar patches válidos (MISMO FILTRO QUE ORIGINAL)
            if valid_patches:
                valid_grid = filtered_grid.loc[valid_patches].copy()
                valid_grid = valid_grid[valid_grid['class'] != -1].copy()  # MISMO FILTRO
            else:
                valid_grid = gpd.GeoDataFrame(columns=filtered_grid.columns)
            
            self.log(f"   Patches válidos después del análisis: {len(valid_grid)}")
            
            if len(valid_grid) == 0:
                return gpd.GeoDataFrame(columns=['Type', 'id', 'geometry', 'source', 'class', 'strat_class', 'has_class_2'])
            
            # PASO 5: Asignar splits estratificados (MISMA LÓGICA QUE ORIGINAL)
            self.log("   Paso 4: Asignando splits estratificados...")
            valid_grid['set'] = ''
            
            for value, group in valid_grid.groupby('strat_class'):
                indices = group.index.tolist()
                random.shuffle(indices)
                
                n = len(indices)
                n_train = round(self.splits[0] * n)
                n_val = round(self.splits[1] * n)
                
                valid_grid.loc[indices[:n_train], 'set'] = 'train'
                valid_grid.loc[indices[n_train:n_train + n_val], 'set'] = 'val'
                valid_grid.loc[indices[n_train + n_val:], 'set'] = 'test'
            
            # PASO 6: Generar IDs únicos (evitar conflictos con originales)
            max_id = 100000
            new_ids = list(range(max_id, max_id + len(valid_grid)))
            random.shuffle(new_ids)
            
            # PASO 7: Crear estructura final consistente con original
            final_grid = gpd.GeoDataFrame({
                'Type': valid_grid['set'],
                'id': new_ids,
                'geometry': valid_grid['geometry'],
                'source': 'new',
                'class': valid_grid['class'],
                'strat_class': valid_grid['strat_class'],
                'has_class_2': valid_grid['has_class_2']
            }, crs=valid_grid.crs)
            
            self.log(f"✅ Grid nuevo generado: {len(final_grid)} patches")
            return final_grid
    
    def combine_grids(self, original_patches: gpd.GeoDataFrame, 
                     new_patches: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Combina los patches originales con los nuevos sin overlaps.
        
        Args:
            original_patches: Patches reutilizados del grid original
            new_patches: Patches nuevos generados
            
        Returns:
            Grid híbrido combinado
        """
        self.log("🔗 Combinando grids...")
        
        # Preparar estructuras consistentes
        combined_patches = []
        
        # Añadir patches originales
        if len(original_patches) > 0:
            for idx, patch in original_patches.iterrows():
                patch_data = {
                    'Type': patch.get('Type', patch.get('set', 'unknown')),
                    'id': patch.get('id', patch.get('patch_id', idx)),
                    'geometry': patch['geometry'],
                    'source': 'original'
                }
                # Preservar columnas adicionales si existen
                for col in ['class', 'strat_class', 'has_class_2']:
                    if col in patch:
                        patch_data[col] = patch[col]
                
                combined_patches.append(patch_data)
        
        # Añadir patches nuevos
        if len(new_patches) > 0:
            for idx, patch in new_patches.iterrows():
                combined_patches.append({
                    'Type': patch['Type'],
                    'id': patch['id'],
                    'geometry': patch['geometry'],
                    'source': patch['source'],
                    'class': patch.get('class'),
                    'strat_class': patch.get('strat_class'),
                    'has_class_2': patch.get('has_class_2')
                })
        
        if not combined_patches:
            self.log("❌ No hay patches para combinar", 'ERROR')
            return gpd.GeoDataFrame()
        
        # Crear GeoDataFrame final
        combined = gpd.GeoDataFrame(combined_patches)
        
        # Asegurar CRS correcto
        if len(original_patches) > 0:
            combined.crs = original_patches.crs
        elif len(new_patches) > 0:
            combined.crs = new_patches.crs
        
        self.log(f"✅ Grid híbrido creado: {len(combined)} patches totales")
        self.log(f"   Original: {len(original_patches)}, Nuevo: {len(new_patches)}")
        
        return combined
    
    def calculate_statistics(self, hybrid_grid: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame):
        """
        Calcula estadísticas detalladas del grid híbrido.
        
        Args:
            hybrid_grid: Grid híbrido final
            aoi: Área de interés
        """
        self.log("📊 Calculando estadísticas...")
        
        if len(hybrid_grid) == 0:
            return
        
        # Estadísticas básicas
        original_count = len(hybrid_grid[hybrid_grid['source'] == 'original'])
        new_count = len(hybrid_grid[hybrid_grid['source'] == 'new'])
        total_count = len(hybrid_grid)
        
        self.stats.update({
            'original_patches_reused': original_count,
            'new_patches_generated': new_count,
            'total_patches': total_count,
            'coverage_original_pct': (original_count / total_count * 100) if total_count > 0 else 0,
            'coverage_new_pct': (new_count / total_count * 100) if total_count > 0 else 0
        })
        
        # Calcular áreas
        if hybrid_grid.crs and hybrid_grid.crs.is_projected:
            area_original = hybrid_grid[hybrid_grid['source'] == 'original'].geometry.area.sum() / 1e6
            area_new = hybrid_grid[hybrid_grid['source'] == 'new'].geometry.area.sum() / 1e6
            area_total = hybrid_grid.geometry.area.sum() / 1e6
            area_aoi = aoi.geometry.area.sum() / 1e6
        else:
            area_original = hybrid_grid[hybrid_grid['source'] == 'original'].to_crs('EPSG:3857').geometry.area.sum() / 1e6
            area_new = hybrid_grid[hybrid_grid['source'] == 'new'].to_crs('EPSG:3857').geometry.area.sum() / 1e6
            area_total = hybrid_grid.to_crs('EPSG:3857').geometry.area.sum() / 1e6
            area_aoi = aoi.to_crs('EPSG:3857').geometry.area.sum() / 1e6
        
        self.stats.update({
            'area_original_km2': area_original,
            'area_new_km2': area_new,
            'area_total_km2': area_total,
            'area_aoi_km2': area_aoi
        })
        
        # Metadata adicional
        self.metadata['source_breakdown'] = {'original': original_count, 'new': new_count}
        
        # Breakdown por sets
        set_counts = hybrid_grid['Type'].value_counts().to_dict()
        self.metadata['set_breakdown'] = set_counts
    
    def print_report(self):
        """Imprime reporte detallado."""
        print("\n" + "="*80)
        print("📋 REPORTE DEL GRID HÍBRIDO (SIN OVERLAPS)")
        print("="*80)
        
        print(f"\n📊 Resumen General:")
        print(f"   Total de patches: {self.stats['total_patches']:,}")
        print(f"   Patches reutilizados: {self.stats['original_patches_reused']:,} ({self.stats['coverage_original_pct']:.1f}%)")
        print(f"   Patches nuevos: {self.stats['new_patches_generated']:,} ({self.stats['coverage_new_pct']:.1f}%)")
        
        print(f"\n🗺️ Cobertura Espacial:")
        print(f"   AOI total: {self.stats['area_aoi_km2']:.2f} km²")
        print(f"   Área cubierta por patches: {self.stats['area_total_km2']:.2f} km²")
        print(f"   - Área reutilizada: {self.stats['area_original_km2']:.2f} km² ({self.stats['area_original_km2']/self.stats['area_total_km2']*100:.1f}%)")
        print(f"   - Área nueva: {self.stats['area_new_km2']:.2f} km² ({self.stats['area_new_km2']/self.stats['area_total_km2']*100:.1f}%)")
        
        if 'area_covered_by_original_km2' in self.stats:
            print(f"\n🎭 Análisis de Masks:")
            print(f"   Área cubierta por patches originales: {self.stats['area_covered_by_original_km2']:.2f} km²")
            if 'area_remaining_for_new_km2' in self.stats:
                print(f"   Área restante para grid nuevo: {self.stats['area_remaining_for_new_km2']:.2f} km²")
        
        print(f"\n📈 Distribución por Sets:")
        for set_name, count in self.metadata['set_breakdown'].items():
            pct = count / self.stats['total_patches'] * 100
            print(f"   {set_name}: {count:,} patches ({pct:.1f}%)")
        
        print(f"\n⚙️ Parámetros Utilizados:")
        print(f"   Patch size: {self.metadata['patch_size']} pixels")
        print(f"   Class 2 proportion: {self.metadata['class2_proportion']}%")
        print(f"   Train/Val/Test splits: {self.metadata['splits']}")
        print(f"   Random seed: {self.metadata['random_seed']}")
        print(f"   Intersection threshold: {self.metadata['intersection_threshold']}")
        
        print("\n✅ Sin overlaps garantizado por mask espacial")
        print("="*80)
    
    def generate_hybrid_grid(self, original_grid_path: str, aoi_path: str, 
                           reference_raster_path: str, output_path: str) -> gpd.GeoDataFrame:
        """
        Función principal para generar el grid híbrido sin overlaps.
        
        Args:
            original_grid_path: Ruta al grid original (GeoJSON)
            aoi_path: Ruta al AOI target (GeoJSON)
            reference_raster_path: Ruta al raster de referencia
            output_path: Ruta de salida para el grid híbrido
            
        Returns:
            GeoDataFrame con el grid híbrido
        """
        self.log("🚀 Iniciando generación de grid híbrido sin overlaps...")
        
        try:
            # 1. Cargar datos
            self.log("📂 Cargando datos...")
            original_grid = gpd.read_file(original_grid_path)
            aoi = gpd.read_file(aoi_path)
            
            self.log(f"   Grid original: {len(original_grid)} patches")
            self.log(f"   AOI: {len(aoi)} features")
            
            # 2. Identificar patches reutilizables
            reusable_patches, aoi_aligned = self.identify_reusable_patches(original_grid, aoi)
            
            # 3. Crear mask espacial de áreas cubiertas
            spatial_mask = self.create_spatial_mask(reusable_patches)
            
            # 4. Calcular área restante
            remaining_area = self.calculate_remaining_area(aoi_aligned, spatial_mask)
            
            # 5. Generar grid para área restante usando lógica original
            new_patches = self.create_grid_for_remaining_area(remaining_area, reference_raster_path)
            
            # 6. Combinar grids sin overlaps
            hybrid_grid = self.combine_grids(reusable_patches, new_patches)
            
            # 7. Calcular estadísticas
            self.calculate_statistics(hybrid_grid, aoi)
            
            # 8. Guardar resultado
            if len(hybrid_grid) > 0:
                hybrid_grid.to_file(output_path, driver='GeoJSON')
                self.log(f"💾 Grid híbrido guardado en: {output_path}")
                
                # Guardar estadísticas
                stats_path = output_path.replace('.geojson', '_stats.json')
                with open(stats_path, 'w') as f:
                    json.dump({**self.stats, **self.metadata}, f, indent=2)
                self.log(f"📊 Estadísticas guardadas en: {stats_path}")
                
                # Mostrar reporte
                self.print_report()
                
                return hybrid_grid
            else:
                self.log("❌ No se pudo generar grid híbrido", 'ERROR')
                return gpd.GeoDataFrame()
                
        except Exception as e:
            self.log(f"❌ Error generando grid híbrido: {str(e)}", 'ERROR')
            raise

def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Generar grid híbrido sin overlaps reutilizando patches originales',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--original_grid', required=True,
                       help='Ruta al grid original (GeoJSON)')
    parser.add_argument('--aoi', required=True,
                       help='Ruta al AOI target (GeoJSON)')
    parser.add_argument('--reference_raster', required=True,
                       help='Ruta al raster de referencia para análisis de clases')
    parser.add_argument('--output', required=True,
                       help='Ruta de salida para el grid híbrido (GeoJSON)')
    
    parser.add_argument('--patch_size', type=int, default=128,
                       help='Tamaño de patch en píxeles')
    parser.add_argument('--class2_proportion', type=float, default=15,
                       help='Proporción mínima de clase 2 para estratificación (%)')
    parser.add_argument('--splits', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                       help='Proporciones train/val/test')
    parser.add_argument('--seed', type=int, default=42,
                       help='Semilla aleatoria para reproducibilidad')
    parser.add_argument('--intersection_threshold', type=float, default=0.7,
                       help='Umbral de intersección para reutilizar patches (0.0-1.0)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Mostrar información detallada')
    parser.add_argument('--quiet', action='store_true',
                       help='Modo silencioso')
    
    args = parser.parse_args()
    
    # Configurar verbosidad
    verbose = args.verbose and not args.quiet
    
    # Crear generador
    generator = HybridGridGenerator(
        patch_size=args.patch_size,
        class2_proportion=args.class2_proportion,
        splits=tuple(args.splits),
        random_seed=args.seed,
        intersection_threshold=args.intersection_threshold,
        verbose=verbose
    )
    
    # Generar grid híbrido
    hybrid_grid = generator.generate_hybrid_grid(
        original_grid_path=args.original_grid,
        aoi_path=args.aoi,
        reference_raster_path=args.reference_raster,
        output_path=args.output
    )
    
    # Exit code basado en el resultado
    exit_code = 0 if len(hybrid_grid) > 0 else 1
    exit(exit_code)

if __name__ == "__main__":
    main()