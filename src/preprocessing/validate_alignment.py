"""
validate_alignment.py

Script para validar la alineación espacial de múltiples rasters.
Verifica proyección, resolución, bounds y superposición de grids.

Uso:
    python validate_alignment.py --rasters s2.tif morpho.tif labels.tif --verbose
    python validate_alignment.py --rasters *.tif --output report.txt
"""

import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
import argparse
import os
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import warnings

class RasterAlignmentValidator:
    """
    Clase para validar la alineación espacial entre múltiples rasters.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.report = {
            'rasters': [],
            'alignment_status': 'UNKNOWN',
            'issues': [],
            'recommendations': []
        }
    
    def log(self, message: str, level: str = 'INFO'):
        """Logger simple con niveles."""
        if self.verbose:
            print(f"[{level}] {message}")
    
    def get_raster_info(self, raster_path: str) -> Dict:
        """
        Extrae información completa de un raster.
        
        Args:
            raster_path: Ruta al archivo raster
            
        Returns:
            Dict con información del raster
        """
        try:
            with rasterio.open(raster_path) as src:
                info = {
                    'path': raster_path,
                    'filename': os.path.basename(raster_path),
                    'crs': src.crs.to_string() if src.crs else None,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': str(src.dtypes[0]),
                    'nodata': src.nodata,
                    'pixel_size_x': abs(src.transform[0]),
                    'pixel_size_y': abs(src.transform[4]),
                    'area_m2': None,
                    'size_mb': os.path.getsize(raster_path) / (1024 * 1024)
                }
                
                # Calcular área aproximada en m²
                if src.crs and src.crs.is_projected:
                    width_m = info['width'] * info['pixel_size_x']
                    height_m = info['height'] * info['pixel_size_y']
                    info['area_m2'] = width_m * height_m
                
                return info
                
        except Exception as e:
            self.log(f"Error leyendo {raster_path}: {str(e)}", 'ERROR')
            return None
    
    def check_crs_consistency(self, raster_infos: List[Dict]) -> bool:
        """
        Verifica que todos los rasters tengan la misma proyección.
        
        Args:
            raster_infos: Lista de información de rasters
            
        Returns:
            True si todas las proyecciones coinciden
        """
        crs_list = [info['crs'] for info in raster_infos if info['crs']]
        
        if len(set(crs_list)) == 1:
            self.log(f"✅ CRS consistente: {crs_list[0]}")
            return True
        else:
            self.log("❌ CRS inconsistente encontrado:", 'WARNING')
            for info in raster_infos:
                self.log(f"  - {info['filename']}: {info['crs']}")
            
            self.report['issues'].append({
                'type': 'CRS_MISMATCH',
                'description': 'Sistemas de coordenadas diferentes entre rasters',
                'details': {info['filename']: info['crs'] for info in raster_infos}
            })
            return False
    
    def check_resolution_consistency(self, raster_infos: List[Dict], tolerance: float = 1e-6) -> bool:
        """
        Verifica que todos los rasters tengan la misma resolución espacial.
        
        Args:
            raster_infos: Lista de información de rasters
            tolerance: Tolerancia para diferencias en resolución
            
        Returns:
            True si las resoluciones son consistentes
        """
        resolutions = [(info['pixel_size_x'], info['pixel_size_y']) for info in raster_infos]
        ref_resolution = resolutions[0]
        
        consistent = True
        for i, (res_x, res_y) in enumerate(resolutions):
            if (abs(res_x - ref_resolution[0]) > tolerance or 
                abs(res_y - ref_resolution[1]) > tolerance):
                consistent = False
                self.log(f"❌ Resolución diferente en {raster_infos[i]['filename']}: "
                        f"{res_x:.6f} x {res_y:.6f}", 'WARNING')
        
        if consistent:
            self.log(f"✅ Resolución consistente: {ref_resolution[0]:.6f} x {ref_resolution[1]:.6f}")
        else:
            self.report['issues'].append({
                'type': 'RESOLUTION_MISMATCH',
                'description': 'Diferentes resoluciones espaciales',
                'details': {info['filename']: (info['pixel_size_x'], info['pixel_size_y']) 
                           for info in raster_infos}
            })
        
        return consistent
    
    def check_bounds_overlap(self, raster_infos: List[Dict]) -> Tuple[bool, Dict]:
        """
        Verifica la superposición de bounds entre rasters.
        
        Args:
            raster_infos: Lista de información de rasters
            
        Returns:
            Tuple (hay_superposicion_completa, info_superposicion)
        """
        bounds_list = [info['bounds'] for info in raster_infos]
        
        # Calcular intersección de todos los bounds
        intersection_left = max([bounds.left for bounds in bounds_list])
        intersection_bottom = max([bounds.bottom for bounds in bounds_list])
        intersection_right = min([bounds.right for bounds in bounds_list])
        intersection_top = min([bounds.top for bounds in bounds_list])
        
        # Verificar si hay intersección válida
        has_intersection = (intersection_left < intersection_right and 
                           intersection_bottom < intersection_top)
        
        # Calcular union de todos los bounds
        union_left = min([bounds.left for bounds in bounds_list])
        union_bottom = min([bounds.bottom for bounds in bounds_list])
        union_right = max([bounds.right for bounds in bounds_list])
        union_top = max([bounds.top for bounds in bounds_list])
        
        overlap_info = {
            'has_intersection': has_intersection,
            'intersection_bounds': (intersection_left, intersection_bottom, 
                                  intersection_right, intersection_top) if has_intersection else None,
            'union_bounds': (union_left, union_bottom, union_right, union_top),
            'individual_bounds': {info['filename']: info['bounds'] for info in raster_infos}
        }
        
        if has_intersection:
            # Calcular área de intersección
            intersection_width = intersection_right - intersection_left
            intersection_height = intersection_top - intersection_bottom
            
            self.log(f"✅ Bounds se superponen correctamente")
            self.log(f"   Intersección: {intersection_width:.2f} x {intersection_height:.2f} unidades")
        else:
            self.log("❌ No hay superposición entre todos los rasters", 'WARNING')
            self.report['issues'].append({
                'type': 'NO_BOUNDS_OVERLAP',
                'description': 'Los rasters no se superponen espacialmente',
                'details': overlap_info['individual_bounds']
            })
        
        return has_intersection, overlap_info
    
    def check_grid_alignment(self, raster_infos: List[Dict]) -> bool:
        """
        Verifica que los grids de píxeles estén perfectamente alineados.
        
        Args:
            raster_infos: Lista de información de rasters
            
        Returns:
            True si los grids están alineados
        """
        if len(raster_infos) < 2:
            return True
        
        ref_transform = raster_infos[0]['transform']
        ref_origin_x, ref_origin_y = ref_transform[2], ref_transform[5]
        ref_pixel_x, ref_pixel_y = ref_transform[0], ref_transform[4]
        
        aligned = True
        
        for i, info in enumerate(raster_infos[1:], 1):
            transform = info['transform']
            origin_x, origin_y = transform[2], transform[5]
            pixel_x, pixel_y = transform[0], transform[4]
            
            # Verificar que los orígenes estén alineados en la grilla de referencia
            offset_x = (origin_x - ref_origin_x) % abs(ref_pixel_x)
            offset_y = (origin_y - ref_origin_y) % abs(ref_pixel_y)
            
            # Tolerancia para errores de punto flotante
            tolerance = 1e-9
            
            if offset_x > tolerance and offset_x < (abs(ref_pixel_x) - tolerance):
                aligned = False
                self.log(f"❌ Grid no alineado en X para {info['filename']}: offset = {offset_x}", 'WARNING')
            
            if offset_y > tolerance and offset_y < (abs(ref_pixel_y) - tolerance):
                aligned = False
                self.log(f"❌ Grid no alineado en Y para {info['filename']}: offset = {offset_y}", 'WARNING')
        
        if aligned:
            self.log("✅ Grids de píxeles correctamente alineados")
        else:
            self.report['issues'].append({
                'type': 'GRID_MISALIGNMENT',
                'description': 'Los grids de píxeles no están perfectamente alineados',
                'details': 'Ver logs para detalles específicos'
            })
        
        return aligned
    
    def generate_recommendations(self, raster_infos: List[Dict]):
        """
        Genera recomendaciones basadas en los problemas encontrados.
        
        Args:
            raster_infos: Lista de información de rasters
        """
        recommendations = []
        
        # Analizar tipos de issues para generar recomendaciones
        issue_types = [issue['type'] for issue in self.report['issues']]
        
        if 'CRS_MISMATCH' in issue_types:
            recommendations.append({
                'issue': 'CRS_MISMATCH',
                'recommendation': 'Reproyectar todos los rasters al mismo CRS usando reproject.py',
                'priority': 'HIGH',
                'command_example': 'python reproject.py input.tif 4326 --output_path output.tif'
            })
        
        if 'RESOLUTION_MISMATCH' in issue_types:
            recommendations.append({
                'issue': 'RESOLUTION_MISMATCH', 
                'recommendation': 'Usar align_rasters.py para homogeneizar resolución',
                'priority': 'HIGH',
                'command_example': 'python align_rasters.py --input input.tif --master master.tif --output aligned.tif'
            })
        
        if 'GRID_MISALIGNMENT' in issue_types:
            recommendations.append({
                'issue': 'GRID_MISALIGNMENT',
                'recommendation': 'Usar align_rasters.py para alinear grids perfectamente',
                'priority': 'MEDIUM',
                'command_example': 'python align_rasters.py --input input.tif --master reference.tif --output aligned.tif'
            })
        
        if 'NO_BOUNDS_OVERLAP' in issue_types:
            recommendations.append({
                'issue': 'NO_BOUNDS_OVERLAP',
                'recommendation': 'Verificar que los rasters cubran la misma área geográfica',
                'priority': 'HIGH',
                'command_example': 'Revisar bounds individualmente y usar clip.py si es necesario'
            })
        
        self.report['recommendations'] = recommendations
    
    def print_summary_report(self, raster_infos: List[Dict]):
        """
        Imprime un reporte resumen de la validación.
        
        Args:
            raster_infos: Lista de información de rasters
        """
        print("\n" + "="*80)
        print("🔍 REPORTE DE VALIDACIÓN DE ALINEACIÓN DE RASTERS")
        print("="*80)
        
        print(f"\n📁 Rasters analizados: {len(raster_infos)}")
        for info in raster_infos:
            print(f"   • {info['filename']} ({info['size_mb']:.1f} MB)")
        
        print(f"\n📊 Estado general: {self.report['alignment_status']}")
        
        if self.report['issues']:
            print(f"\n⚠️  Problemas encontrados ({len(self.report['issues'])}):")
            for issue in self.report['issues']:
                print(f"   • {issue['type']}: {issue['description']}")
        else:
            print("\n✅ No se encontraron problemas de alineación")
        
        if self.report['recommendations']:
            print(f"\n💡 Recomendaciones ({len(self.report['recommendations'])}):")
            for rec in self.report['recommendations']:
                print(f"   • {rec['issue']} (Prioridad: {rec['priority']})")
                print(f"     → {rec['recommendation']}")
                print(f"     → Ejemplo: {rec['command_example']}")
        
        print("\n" + "="*80)
    
    def validate_rasters(self, raster_paths: List[str]) -> bool:
        """
        Función principal para validar la alineación de múltiples rasters.
        
        Args:
            raster_paths: Lista de rutas a los rasters
            
        Returns:
            True si todos los rasters están correctamente alineados
        """
        self.log(f"🚀 Iniciando validación de {len(raster_paths)} rasters...")
        
        # 1. Obtener información de cada raster
        raster_infos = []
        for path in raster_paths:
            if not os.path.exists(path):
                self.log(f"❌ Archivo no encontrado: {path}", 'ERROR')
                continue
            
            info = self.get_raster_info(path)
            if info:
                raster_infos.append(info)
                self.report['rasters'].append(info)
        
        if len(raster_infos) < 2:
            self.log("❌ Se necesitan al menos 2 rasters para validar alineación", 'ERROR')
            return False
        
        # 2. Verificar consistencia de CRS
        crs_ok = self.check_crs_consistency(raster_infos)
        
        # 3. Verificar consistencia de resolución
        resolution_ok = self.check_resolution_consistency(raster_infos)
        
        # 4. Verificar superposición de bounds
        bounds_ok, bounds_info = self.check_bounds_overlap(raster_infos)
        
        # 5. Verificar alineación de grid (solo si CRS y resolución son consistentes)
        grid_ok = True
        if crs_ok and resolution_ok:
            grid_ok = self.check_grid_alignment(raster_infos)
        
        # 6. Determinar estado general
        all_aligned = crs_ok and resolution_ok and bounds_ok and grid_ok
        
        if all_aligned:
            self.report['alignment_status'] = 'PERFECT_ALIGNMENT'
            self.log("\n🎉 ¡Todos los rasters están perfectamente alineados!")
        elif crs_ok and resolution_ok and bounds_ok:
            self.report['alignment_status'] = 'MINOR_ISSUES'
            self.log("\n⚠️  Alineación casi perfecta con problemas menores")
        else:
            self.report['alignment_status'] = 'MAJOR_ISSUES'
            self.log("\n❌ Problemas significativos de alineación encontrados")
        
        # 7. Generar recomendaciones
        self.generate_recommendations(raster_infos)
        
        # 8. Mostrar reporte
        self.print_summary_report(raster_infos)
        
        return all_aligned
    
    def save_report(self, output_path: str):
        """
        Guarda el reporte completo en un archivo JSON.
        
        Args:
            output_path: Ruta donde guardar el reporte
        """
        # Serializar objetos que no son JSON-serializables
        report_copy = self.report.copy()
        for raster in report_copy['rasters']:
            if 'transform' in raster:
                raster['transform'] = list(raster['transform'])
            if 'bounds' in raster:
                raster['bounds'] = list(raster['bounds'])
        
        with open(output_path, 'w') as f:
            json.dump(report_copy, f, indent=2)
        
        self.log(f"📝 Reporte guardado en: {output_path}")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Validar alineación espacial de múltiples rasters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--rasters', nargs='+', required=True,
                       help='Rutas a los archivos raster a validar')
    parser.add_argument('--output', type=str,
                       help='Archivo de salida para el reporte JSON (opcional)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Mostrar información detallada')
    parser.add_argument('--quiet', action='store_true',
                       help='Modo silencioso (sobrescribe --verbose)')
    
    args = parser.parse_args()
    
    # Configurar verbosidad
    verbose = args.verbose and not args.quiet
    
    # Crear validador
    validator = RasterAlignmentValidator(verbose=verbose)
    
    # Validar rasters
    success = validator.validate_rasters(args.rasters)
    
    # Guardar reporte si se especifica
    if args.output:
        validator.save_report(args.output)
    
    # Exit code basado en el resultado
    exit_code = 0 if success else 1
    exit(exit_code)


if __name__ == "__main__":
    main()