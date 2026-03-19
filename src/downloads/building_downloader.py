"""
Building Data Downloader for ML Inference Pipeline
=================================================

This module downloads building footprint data from various sources (Microsoft, Google, OpenStreetMap)
for Google Colab environment. Ensures compatibility with the original morphometric pipeline.

Key Features:
- Microsoft Building Footprints (preferred - matches original)
- Google Open Buildings (backup)
- OpenStreetMap buildings (fallback)
- AOI-based filtering and clipping
- CRS handling and reprojection
- Data validation and quality checks
- Integration with existing S2/GHSL pipeline

Author: Adapted for ML inference preprocessing pipeline
Compatible with: Google Colab, original morphometric workflow
Dependencies: geopandas, planetary-computer, pystac-client, requests
"""

import os
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import planetary_computer
    import pystac_client
    import adlfs
    MICROSOFT_AVAILABLE = True
except ImportError:
    MICROSOFT_AVAILABLE = False
    print("⚠️ Warning: planetary-computer not available. Microsoft buildings will be disabled.")

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    print("⚠️ Warning: osmnx not available. OpenStreetMap buildings will be disabled.")


class BuildingDownloader:
    """
    Downloads building footprint data from multiple sources for ML pipeline compatibility.
    """
    
    def __init__(self, validate_outputs: bool = True):
        """
        Initialize building downloader.
        
        Parameters:
        -----------
        validate_outputs : bool, default True
            Whether to validate downloaded building data
        """
        self.validate_outputs = validate_outputs
        self.download_stats = {}
        
        # Expected building properties (for validation)
        self.expected_crs = "EPSG:4326"
        self.min_building_area = 10  # m² minimum building size
        self.max_building_area = 1000000  # m² maximum building size
    
    def download_microsoft_buildings(self,
                                   aoi_geom,
                                   region: str = None,
                                   processing_date: str = "2023-04-25",
                                   output_filename: str = "microsoft_buildings.geojson") -> str:
        """
        Download Microsoft Building Footprints for AOI.
        
        This is the preferred method as it matches the original pipeline's data source.
        
        Parameters:
        -----------
        aoi_geom : various
            Area of interest (bounds tuple, geometry, or file path)
        region : str, optional
            Region name (e.g., 'Argentina', 'Ethiopia'). Auto-detected if None
        processing_date : str, default "2023-04-25"
            Processing date for Microsoft buildings
        output_filename : str, default "microsoft_buildings.geojson"
            Output filename
            
        Returns:
        --------
        str
            Path to downloaded building file
        """
        if not MICROSOFT_AVAILABLE:
            raise ImportError("Microsoft buildings require: pip install planetary-computer pystac-client adlfs")
        
        print("🏢 Downloading Microsoft Building Footprints...")
        
        # Prepare AOI
        aoi_bounds, aoi_poly = self._prepare_aoi(aoi_geom)
        
        # Auto-detect region if not provided
        if region is None:
            region = self._detect_region_from_aoi(aoi_bounds)
            print(f"🗺️ Auto-detected region: {region}")
        
        try:
            # Initialize Planetary Computer client
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace,
            )
            
            # Search for building data
            print(f"🔍 Searching Microsoft buildings for region: {region}")
            search = catalog.search(
                collections=["ms-buildings"],
                intersects=aoi_poly,
                query={
                    "msbuildings:region": {"eq": region},
                    "msbuildings:processing-date": {"eq": processing_date}
                }
            )
            
            items = search.item_collection()
            
            if len(items) == 0:
                raise ValueError(f"No Microsoft building data found for region '{region}' and date '{processing_date}'")
            
            print(f"📦 Found {len(items)} building dataset(s)")
            
            # Get storage options
            storage_options = items[0].assets["data"].extra_fields["table:storage_options"]
            
            # Download and process building data
            parts = []
            for item in items:
                fs = adlfs.AzureBlobFileSystem(**storage_options)
                parts.extend(fs.ls(item.assets["data"].href))
            
            print(f"📥 Loading {len(parts)} data parts...")
            
            # Load all parts and combine
            dfs = []
            for part in parts:
                try:
                    df_part = gpd.read_parquet(f"az://{part}", storage_options=storage_options)
                    dfs.append(df_part)
                except Exception as e:
                    print(f"⚠️ Failed to load part {part}: {str(e)}")
            
            if not dfs:
                raise Exception("Failed to load any building data parts")
            
            # Combine all parts
            buildings_gdf = pd.concat(dfs, ignore_index=True)
            buildings_gdf = gpd.GeoDataFrame(buildings_gdf, geometry="geometry")
            
            # Ensure correct CRS
            if buildings_gdf.crs != self.expected_crs:
                buildings_gdf = buildings_gdf.to_crs(self.expected_crs)
            
            # Clip to AOI bounds
            print("✂️ Clipping buildings to AOI...")
            aoi_box = box(*aoi_bounds)
            buildings_gdf = buildings_gdf[buildings_gdf.intersects(aoi_box)]
            
            # Validate and clean data
            if self.validate_outputs:
                buildings_gdf = self._validate_and_clean_buildings(buildings_gdf)
            
            # Save to file
            buildings_gdf.to_file(output_filename, driver="GeoJSON")
            
            # Update statistics
            self._update_download_stats("Microsoft", len(buildings_gdf), region, output_filename)
            
            print(f"✅ Microsoft buildings downloaded: {len(buildings_gdf):,} buildings")
            print(f"📁 Saved to: {output_filename}")
            
            return output_filename
            
        except Exception as e:
            print(f"❌ Microsoft buildings download failed: {str(e)}")
            raise
    
    def download_google_buildings(self,
                                 aoi_geom,
                                 output_filename: str = "google_buildings.geojson") -> str:
        """
        Download Google Open Buildings data for AOI.
        
        Backup method when Microsoft buildings are not available.
        
        Parameters:
        -----------
        aoi_geom : various
            Area of interest (bounds tuple, geometry, or file path)
        output_filename : str, default "google_buildings.geojson"
            Output filename
            
        Returns:
        --------
        str
            Path to downloaded building file
        """
        print("🏢 Downloading Google Open Buildings...")
        
        # Prepare AOI
        aoi_bounds, aoi_poly = self._prepare_aoi(aoi_geom)
        
        try:
            # Google Open Buildings is distributed via CSV files
            # For demo purposes, this is a simplified implementation
            # In practice, you'd need to identify the correct regional CSV file
            
            print("⚠️ Google Open Buildings requires manual dataset identification")
            print("   Please refer to: https://sites.research.google/open-buildings/")
            print("   This method needs regional CSV file paths to be implemented")
            
            # Placeholder implementation
            # You would need to:
            # 1. Identify the correct regional CSV file for your AOI
            # 2. Download and filter by AOI bounds
            # 3. Convert WKT geometries to GeoDataFrame
            
            raise NotImplementedError("Google Buildings download needs regional CSV file mapping")
            
        except Exception as e:
            print(f"❌ Google buildings download failed: {str(e)}")
            raise
    
    def download_osm_buildings(self,
                              aoi_geom,
                              building_types: List[str] = None,
                              output_filename: str = "osm_buildings.geojson") -> str:
        """
        Download OpenStreetMap building data for AOI.
        
        Fallback method when commercial datasets are not available.
        
        Parameters:
        -----------
        aoi_geom : various
            Area of interest (bounds tuple, geometry, or file path)
        building_types : list, optional
            Building types to include. Default: all buildings
        output_filename : str, default "osm_buildings.geojson"
            Output filename
            
        Returns:
        --------
        str
            Path to downloaded building file
        """
        if not OSMNX_AVAILABLE:
            raise ImportError("OSM buildings require: pip install osmnx")
        
        print("🏢 Downloading OpenStreetMap buildings...")
        
        # Prepare AOI
        aoi_bounds, aoi_poly = self._prepare_aoi(aoi_geom)
        
        try:
            # Default building types
            if building_types is None:
                building_types = ['yes', 'residential', 'commercial', 'industrial', 
                                'retail', 'office', 'house', 'apartments']
            
            print(f"🔍 Searching for building types: {building_types}")
            
            # Create tags for OSM query
            tags = {'building': building_types}
            
            # Download buildings using osmnx
            buildings_gdf = ox.features_from_bbox(
                north=aoi_bounds[3],
                south=aoi_bounds[1], 
                east=aoi_bounds[2],
                west=aoi_bounds[0],
                tags=tags
            )
            
            if buildings_gdf.empty:
                raise ValueError("No OSM buildings found in AOI")
            
            # Keep only polygon geometries
            buildings_gdf = buildings_gdf[buildings_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            
            # Clean up columns (keep essential ones)
            essential_cols = ['osmid', 'building', 'geometry']
            available_cols = [col for col in essential_cols if col in buildings_gdf.columns]
            buildings_gdf = buildings_gdf[available_cols]
            
            # Reset index
            buildings_gdf = buildings_gdf.reset_index(drop=True)
            
            # Ensure correct CRS
            if buildings_gdf.crs != self.expected_crs:
                buildings_gdf = buildings_gdf.to_crs(self.expected_crs)
            
            # Validate and clean data
            if self.validate_outputs:
                buildings_gdf = self._validate_and_clean_buildings(buildings_gdf)
            
            # Save to file
            buildings_gdf.to_file(output_filename, driver="GeoJSON")
            
            # Update statistics
            self._update_download_stats("OpenStreetMap", len(buildings_gdf), "OSM", output_filename)
            
            print(f"✅ OSM buildings downloaded: {len(buildings_gdf):,} buildings")
            print(f"📁 Saved to: {output_filename}")
            
            return output_filename
            
        except Exception as e:
            print(f"❌ OSM buildings download failed: {str(e)}")
            raise
    
    def download_buildings_auto(self,
                               aoi_geom,
                               region: str = None,
                               output_filename: str = "buildings.geojson") -> str:
        """
        Automatically download buildings using best available source.
        
        Tries sources in order: Microsoft → OSM → Google
        
        Parameters:
        -----------
        aoi_geom : various
            Area of interest (bounds tuple, geometry, or file path)
        region : str, optional
            Region name for Microsoft buildings
        output_filename : str, default "buildings.geojson"
            Output filename
            
        Returns:
        --------
        str
            Path to downloaded building file
        """
        print("🏢 Auto-downloading buildings from best available source...")
        
        # Method 1: Try Microsoft Buildings (preferred)
        if MICROSOFT_AVAILABLE:
            try:
                print("🥇 Attempting Microsoft Building Footprints...")
                return self.download_microsoft_buildings(
                    aoi_geom, region, output_filename=output_filename
                )
            except Exception as e:
                print(f"⚠️ Microsoft buildings failed: {str(e)}")
        
        # Method 2: Try OSM Buildings (good coverage)
        if OSMNX_AVAILABLE:
            try:
                print("🥈 Attempting OpenStreetMap buildings...")
                return self.download_osm_buildings(
                    aoi_geom, output_filename=output_filename
                )
            except Exception as e:
                print(f"⚠️ OSM buildings failed: {str(e)}")
        
        # Method 3: Google Buildings (requires manual setup)
        try:
            print("🥉 Attempting Google Open Buildings...")
            return self.download_google_buildings(
                aoi_geom, output_filename=output_filename
            )
        except Exception as e:
            print(f"❌ Google buildings failed: {str(e)}")
        
        raise Exception("All building download methods failed")
    
    def _prepare_aoi(self, aoi_geom) -> Tuple[Tuple[float, float, float, float], Dict]:
        """Prepare AOI geometry and bounds."""
        
        if isinstance(aoi_geom, (tuple, list)) and len(aoi_geom) == 4:
            # Bounding box
            aoi_bounds = tuple(aoi_geom)
            aoi_poly = {
                "type": "Polygon",
                "coordinates": [[
                    [aoi_bounds[0], aoi_bounds[1]],
                    [aoi_bounds[2], aoi_bounds[1]], 
                    [aoi_bounds[2], aoi_bounds[3]],
                    [aoi_bounds[0], aoi_bounds[3]],
                    [aoi_bounds[0], aoi_bounds[1]]
                ]]
            }
        elif isinstance(aoi_geom, str):
            # File path
            aoi_gdf = gpd.read_file(aoi_geom).to_crs(self.expected_crs)
            aoi_bounds = tuple(aoi_gdf.total_bounds)
            aoi_poly = aoi_gdf.iloc[0].geometry.__geo_interface__
        else:
            # Assume geometry object
            if hasattr(aoi_geom, 'bounds'):
                aoi_bounds = aoi_geom.bounds
                aoi_poly = aoi_geom.__geo_interface__
            else:
                raise ValueError("Unsupported AOI geometry type")
        
        return aoi_bounds, aoi_poly
    
    def _detect_region_from_aoi(self, aoi_bounds: Tuple[float, float, float, float]) -> str:
        """Auto-detect region name from AOI bounds."""
        
        # Simple region detection based on bounds
        # This is a basic implementation - you might want to enhance it
        
        minx, miny, maxx, maxy = aoi_bounds
        center_lon = (minx + maxx) / 2
        center_lat = (miny + maxy) / 2
        
        # Basic region mapping (expand as needed)
        if -70 <= center_lon <= -30 and -60 <= center_lat <= 15:
            if center_lat > -40:
                return "Argentina"
            else:
                return "Argentina"  # Southern regions
        elif 30 <= center_lon <= 50 and 0 <= center_lat <= 20:
            return "Ethiopia"
        elif 65 <= center_lon <= 100 and 5 <= center_lat <= 40:
            return "India"
        elif 90 <= center_lon <= 150 and -50 <= center_lat <= 10:
            return "Indonesia"
        elif -20 <= center_lon <= 60 and -40 <= center_lat <= 40:
            return "South Africa"
        else:
            # Default fallback - might need manual specification
            print(f"⚠️ Could not auto-detect region for bounds {aoi_bounds}")
            print("   Please specify region manually for Microsoft buildings")
            return "Global"  # This might not work for Microsoft data
    
    def _validate_and_clean_buildings(self, buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Validate and clean building geometries."""
        
        print("🔍 Validating and cleaning building data...")
        
        initial_count = len(buildings_gdf)
        
        # Remove invalid geometries
        buildings_gdf = buildings_gdf[buildings_gdf.geometry.is_valid]
        
        # Remove empty geometries
        buildings_gdf = buildings_gdf[~buildings_gdf.geometry.is_empty]
        
        # Convert to projected CRS for area calculation
        temp_gdf = buildings_gdf.to_crs(buildings_gdf.estimate_utm_crs())
        areas = temp_gdf.geometry.area
        
        # Filter by area (remove too small/large buildings)
        area_mask = (areas >= self.min_building_area) & (areas <= self.max_building_area)
        buildings_gdf = buildings_gdf[area_mask]
        
        # Remove duplicate geometries
        buildings_gdf = buildings_gdf.drop_duplicates(subset=['geometry'])
        
        # Reset index
        buildings_gdf = buildings_gdf.reset_index(drop=True)
        
        # Add unique ID if not present
        if 'uID' not in buildings_gdf.columns:
            buildings_gdf['uID'] = range(len(buildings_gdf))
        
        final_count = len(buildings_gdf)
        removed_count = initial_count - final_count
        
        print(f"🧹 Cleaning results:")
        print(f"   Initial buildings: {initial_count:,}")
        print(f"   Removed: {removed_count:,}")
        print(f"   Final buildings: {final_count:,}")
        
        return buildings_gdf
    
    def _update_download_stats(self, source: str, count: int, region: str, filename: str):
        """Update download statistics."""
        
        self.download_stats = {
            'source': source,
            'building_count': count,
            'region': region,
            'filename': filename,
            'download_time': pd.Timestamp.now().isoformat()
        }
    
    def get_download_stats(self) -> Dict:
        """Get download statistics."""
        return self.download_stats.copy()


# Convenience functions for direct use
def download_buildings_for_aoi(aoi_input,
                              region: str = None,
                              output_filename: str = "buildings.geojson",
                              source: str = "auto") -> str:
    """
    Convenience function to download buildings for AOI.
    
    Parameters:
    -----------
    aoi_input : various
        AOI specification (bounds, file path, or geometry)
    region : str, optional
        Region name for Microsoft buildings
    output_filename : str, default "buildings.geojson"
        Output filename
    source : str, default "auto"
        Source preference ("auto", "microsoft", "osm", "google")
        
    Returns:
    --------
    str
        Path to downloaded building file
    """
    downloader = BuildingDownloader()
    
    if source == "auto":
        return downloader.download_buildings_auto(aoi_input, region, output_filename)
    elif source == "microsoft":
        return downloader.download_microsoft_buildings(aoi_input, region, output_filename)
    elif source == "osm":
        return downloader.download_osm_buildings(aoi_input, output_filename=output_filename)
    elif source == "google":
        return downloader.download_google_buildings(aoi_input, output_filename)
    else:
        raise ValueError(f"Unknown source: {source}")


def validate_building_data(building_file: str) -> Dict[str, Any]:
    """
    Validate building data file.
    
    Parameters:
    -----------
    building_file : str
        Path to building file
        
    Returns:
    --------
    dict
        Validation results
    """
    try:
        buildings = gpd.read_file(building_file)
        
        validation = {
            'valid': True,
            'building_count': len(buildings),
            'geometry_types': buildings.geometry.type.value_counts().to_dict(),
            'crs': str(buildings.crs),
            'bounds': tuple(buildings.total_bounds),
            'has_unique_id': 'uID' in buildings.columns,
            'warnings': []
        }
        
        # Check for issues
        if len(buildings) == 0:
            validation['valid'] = False
            validation['warnings'].append("No buildings found")
        
        if buildings.crs != "EPSG:4326":
            validation['warnings'].append(f"CRS is {buildings.crs}, expected EPSG:4326")
        
        invalid_geoms = ~buildings.geometry.is_valid
        if invalid_geoms.any():
            validation['warnings'].append(f"{invalid_geoms.sum()} invalid geometries")
        
        return validation
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'building_count': 0
        }


# Example usage and testing
if __name__ == "__main__":
    print("🏢 Building Downloader Test")
    print("="*40)
    
    # Test parameters (Buenos Aires area)
    test_bounds = (-58.5, -34.7, -58.3, -34.5)
    
    try:
        print("🔍 Testing building download...")
        
        # Test auto-download
        building_file = download_buildings_for_aoi(
            aoi_input=test_bounds,
            region="Argentina",
            output_filename="test_buildings.geojson",
            source="auto"
        )
        
        # Validate result
        validation = validate_building_data(building_file)
        
        if validation['valid']:
            print(f"✅ Buildings downloaded successfully:")
            print(f"   Count: {validation['building_count']:,}")
            print(f"   CRS: {validation['crs']}")
            print(f"   Bounds: {validation['bounds']}")
        else:
            print(f"❌ Validation failed: {validation.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("Make sure required packages are installed:")