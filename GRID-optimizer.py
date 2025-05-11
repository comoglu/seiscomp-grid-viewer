#!/usr/bin/python3

import numpy as np
import pandas as pd
from scipy.spatial import KDTree, ConvexHull, cKDTree
from dataclasses import dataclass, field
import argparse
from typing import List, Tuple, Optional, Dict, Set
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import warnings
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, euclidean
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"grid_optimizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("GridOptimizer")

@dataclass
class GridPoint:
    lat: float
    lon: float
    depth: float
    radius: float
    max_station_dist: float
    min_pick_count: int
    quality_score: float = 0.0
    longitude_format: str = "180_180"
    station_count: int = 0
    azimuthal_gap: float = 360.0
    mean_station_dist: float = 0.0
    is_new: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "lat": self.lat,
            "lon": self.lon,
            "depth": self.depth,
            "radius": self.radius,
            "max_station_dist": self.max_station_dist,
            "min_pick_count": self.min_pick_count,
            "quality_score": self.quality_score,
            "station_count": self.station_count,
            "azimuthal_gap": self.azimuthal_gap,
            "mean_station_dist": self.mean_station_dist,
            "is_new": self.is_new
        }
    
    def __hash__(self):
        """Custom hash function for grid point deduplication"""
        # Round to 3 decimal places for latitude and longitude to avoid floating point issues
        return hash((round(self.lat, 3), round(self.lon, 3), round(self.depth, 1)))

class GridOptimizer:
    def __init__(self, longitude_format="180_180"):
        self.base_grid = []
        self.stations = None
        self.events = None
        self.station_kdtree = None
        self.optimal_spacing = None
        self.longitude_format = longitude_format
        self.existing_grid_points = set()  # For deduplication
        self.grid_point_cache = {}  # Cache for station coverage calculations
        
    def normalize_longitude(self, lon: float) -> float:
        """Normalize longitude based on format"""
        if self.longitude_format == "0_360":
            return lon % 360
        return ((lon + 180) % 360) - 180
        
    def calculate_coverage_area(self) -> float:
        """Calculate the convex hull area of station network"""
        if self.stations is None or len(self.stations) < 3:
            return 0.0
            
        try:
            coords = self.stations[['Latitude', 'Longitude']].values
            hull = ConvexHull(coords)
            # Convert to approximate km²
            area_in_degrees = hull.area
            # Approximate conversion: 1 degree ≈ 111 km at the equator
            area_factor = 111 * 111
            return area_in_degrees * area_factor
        except Exception as e:
            logger.warning(f"Could not calculate coverage area: {e}")
            return 0.0
        
    def optimize_grid_spacing(self) -> float:
        """Calculate optimal grid spacing based on station density"""
        if self.stations is None:
            return 1.0
            
        coverage_area = self.calculate_coverage_area()
        if coverage_area < 1:
            return 1.0
            
        station_density = len(self.stations) / coverage_area
        # Adjust scaling factor to get appropriate grid spacing based on network size
        if coverage_area > 10000000:  # Global network (>10M km²)
            base_spacing = 2.0
        elif coverage_area > 1000000:  # Continental network
            base_spacing = 1.0
        elif coverage_area > 100000:  # Regional network
            base_spacing = 0.5
        else:  # Local network
            base_spacing = 0.25
            
        return max(0.25, min(5.0, base_spacing / np.sqrt(station_density)))

    def read_base_grid(self, filename: str):
        """Read and validate the base grid configuration"""
        try:
            MIN_SPACING = 0.01  # 0.01 degrees minimum spacing
            
            # Check if file exists
            if not os.path.isfile(filename):
                logger.error(f"Grid file not found: {filename}")
                return
                
            # Read with more flexible format handling, allowing comments
            data = pd.read_csv(filename, delimiter=r'\s+', header=None, comment='#',
                            names=['lat', 'lon', 'depth', 'radius', 
                                  'max_station_dist', 'min_pick_count'],
                            usecols=range(6))
            
            # Handle additional quality column if it exists
            if len(data.columns) >= 7:
                data = data.iloc[:, :6]  # Use only first 6 columns
                
            # Normalize longitudes
            data['lon'] = data['lon'].apply(self.normalize_longitude)
            
            # Filter invalid values
            data = data[
                (data['lat'].between(-90, 90)) &
                (data['lon'].between(-180, 180) if self.longitude_format == "180_180" else data['lon'].between(0, 360)) &
                (data['depth'] >= 0) &
                (data['radius'] > 0) &
                (data['max_station_dist'] > 0) &
                (data['min_pick_count'] >= 3)
            ]
            
            # Filter close points with improved deduplication
            filtered_data = []
            used_positions = set()
            for _, row in data.iterrows():
                pos = (round(row.lat/MIN_SPACING), round(row.lon/MIN_SPACING), round(row.depth))
                if pos not in used_positions:
                    filtered_data.append(row)
                    used_positions.add(pos)
            
            data = pd.DataFrame(filtered_data)
            
            # Check minimum spacing between grid points
            if len(data) > 1:
                coords = data[['lat', 'lon']].values
                distances = pdist(coords)
                min_spacing = np.min(distances) if len(distances) > 0 else 99999
                if min_spacing < MIN_SPACING:
                    logger.warning(f"Grid points too close (min spacing: {min_spacing:.5f}°)")
            
            # Create grid points
            self.base_grid = []
            for _, row in data.iterrows():
                grid_point = GridPoint(
                    lat=row.lat, 
                    lon=row.lon, 
                    depth=row.depth, 
                    radius=row.radius, 
                    max_station_dist=row.max_station_dist, 
                    min_pick_count=row.min_pick_count,
                    longitude_format=self.longitude_format
                )
                self.base_grid.append(grid_point)
                self.existing_grid_points.add(hash(grid_point))
                
            logger.info(f"Loaded {len(self.base_grid)} grid points from {filename}")
        except Exception as e:
            logger.error(f"Error reading base grid: {e}")
            raise

    def read_stations(self, filename: str):
        """Read and validate station information"""
        try:
            # Check if file exists
            if not os.path.isfile(filename):
                logger.error(f"Station file not found: {filename}")
                return
                
            # Detect file format based on extension and content
            if filename.endswith('.csv'):
                # CSV format
                self.stations = pd.read_csv(filename)
                
                # Look for standard column names
                required_cols = {'Latitude', 'Longitude'}
                if not required_cols.issubset(set(self.stations.columns)):
                    # Try alternative column names
                    rename_map = {}
                    for col in self.stations.columns:
                        if col.lower() in ('lat', 'latitude'):
                            rename_map[col] = 'Latitude'
                        elif col.lower() in ('lon', 'long', 'longitude'):
                            rename_map[col] = 'Longitude'
                        elif col.lower() in ('net', 'network'):
                            rename_map[col] = 'Network'
                        elif col.lower() in ('sta', 'station'):
                            rename_map[col] = 'Station'
                            
                    if rename_map:
                        self.stations = self.stations.rename(columns=rename_map)
            else:
                # Try to parse the SeisComP station format
                try:
                    with open(filename, 'r') as f:
                        first_line = f.readline().strip()
                        # Check if the file has the expected header format
                        if first_line.startswith('#'):
                            header = first_line.lstrip('#').split('|')
                            header = [h.strip() for h in header]
                            
                            self.stations = pd.read_csv(filename, delimiter='|', 
                                                     names=header, skiprows=1, 
                                                     comment='#', skip_blank_lines=True)
                        else:
                            # Try fixed width format
                            f.seek(0)
                            lines = []
                            for line in f:
                                if not line.strip() or line.strip().startswith('#'):
                                    continue
                                parts = line.strip().split()
                                if len(parts) >= 4:  # Minimum: Network, Station, Latitude, Longitude
                                    lines.append(parts)
                                    
                            if lines:
                                # Create DataFrame with appropriate columns
                                self.stations = pd.DataFrame(lines)
                                if len(self.stations.columns) >= 4:
                                    self.stations = self.stations.iloc[:, :4]
                                    self.stations.columns = ['Network', 'Station', 'Latitude', 'Longitude']
                                    self.stations['Latitude'] = pd.to_numeric(self.stations['Latitude'])
                                    self.stations['Longitude'] = pd.to_numeric(self.stations['Longitude'])
                except Exception as e:
                    logger.error(f"Could not parse station file: {e}")
                    raise
            
            # Validate station data
            if self.stations is None or 'Latitude' not in self.stations.columns or 'Longitude' not in self.stations.columns:
                logger.error("Station file must contain Latitude and Longitude columns")
                raise ValueError("Invalid station data format")
                
            # Create Network and Station columns if they don't exist
            if 'Network' not in self.stations.columns:
                self.stations['Network'] = 'NET'
            if 'Station' not in self.stations.columns:
                self.stations['Station'] = self.stations.index.astype(str)
                
            # Convert coordinates to numeric and filter invalid values
            self.stations['Latitude'] = pd.to_numeric(self.stations['Latitude'], errors='coerce')
            self.stations['Longitude'] = pd.to_numeric(self.stations['Longitude'], errors='coerce')
            
            # Drop rows with invalid coordinates
            original_count = len(self.stations)
            self.stations = self.stations.dropna(subset=['Latitude', 'Longitude'])
            self.stations = self.stations[
                (self.stations['Latitude'].between(-90, 90)) &
                (self.stations['Longitude'].between(-180, 180) if self.longitude_format == "180_180" 
                 else self.stations['Longitude'].between(0, 360))
            ]
            
            if len(self.stations) < original_count:
                logger.warning(f"Removed {original_count - len(self.stations)} stations with invalid coordinates")
            
            # Normalize longitudes
            self.stations['Longitude'] = self.stations['Longitude'].apply(self.normalize_longitude)
            
            # Convert coordinates to radians for KDTree
            coords = np.radians(self.stations[['Latitude', 'Longitude']].values)
            
            # Use cKDTree for better performance
            self.station_kdtree = cKDTree(coords)
            
            # Calculate optimal grid spacing
            self.optimal_spacing = self.optimize_grid_spacing()
            
            logger.info(f"Loaded {len(self.stations)} stations from {filename}")
            logger.info(f"Optimal grid spacing: {self.optimal_spacing:.2f}°")
            
            # Calculate network statistics
            if len(self.stations) >= 3:
                coverage_area = self.calculate_coverage_area()
                logger.info(f"Network coverage area: {coverage_area:.2f} km²")
                station_density = len(self.stations) / coverage_area if coverage_area > 0 else 0
                logger.info(f"Station density: {station_density:.6f} stations/km²")
                
        except Exception as e:
            logger.error(f"Error reading stations: {e}")
            raise

    def read_events(self, filename: str):
        """Read and preprocess seismic events"""
        try:
            # Check if file exists
            if not os.path.isfile(filename):
                logger.error(f"Event file not found: {filename}")
                return
                
            # Try different formats
            try:
                # First, try standard CSV format
                self.events = pd.read_csv(filename)
                
                # Look for required columns with different possible names
                required_cols = ['latitude', 'longitude', 'depth']
                rename_map = {}
                
                for col in self.events.columns:
                    if col.lower() in ('lat', 'latitude'):
                        rename_map[col] = 'latitude'
                    elif col.lower() in ('lon', 'long', 'longitude'):
                        rename_map[col] = 'longitude'
                    elif col.lower() in ('depth', 'dep', 'z'):
                        rename_map[col] = 'depth'
                    elif col.lower() in ('mag', 'magnitude'):
                        rename_map[col] = 'magnitude'
                    elif col.lower() in ('time', 'origin_time', 'datetime'):
                        rename_map[col] = 'time'
                
                if rename_map:
                    self.events = self.events.rename(columns=rename_map)
                
                # Check if we have the minimum required columns
                missing_cols = [col for col in required_cols if col not in self.events.columns]
                if missing_cols:
                    logger.warning(f"Event file missing columns: {missing_cols}")
                    raise ValueError("Missing required columns")
                    
            except Exception:
                # If standard CSV fails, try SeisComP event format
                logger.info("Trying to parse as SeisComP event format...")
                with open(filename, 'r') as f:
                    lines = []
                    for line in f:
                        if not line.strip() or line.strip().startswith('#'):
                            continue
                        parts = line.strip().split('|')
                        if len(parts) >= 4:  # Minimum: eventID, time, latitude, longitude
                            lines.append(parts)
                            
                    if lines:
                        self.events = pd.DataFrame(lines)
                        # Map columns based on standard SeisComP event format
                        col_map = {
                            0: 'event_id',
                            1: 'time',
                            2: 'latitude',
                            3: 'longitude',
                            4: 'depth',
                            5: 'magnitude'
                        }
                        
                        self.events = self.events.rename(columns={i: name for i, name in col_map.items() 
                                                              if i < len(self.events.columns)})
                        
                        # Ensure numeric columns
                        for col in ['latitude', 'longitude', 'depth']:
                            if col in self.events.columns:
                                self.events[col] = pd.to_numeric(self.events[col], errors='coerce')
                        
                        if 'magnitude' in self.events.columns:
                            self.events['magnitude'] = pd.to_numeric(self.events['magnitude'], errors='coerce')
            
            # Normalize longitudes
            if 'longitude' in self.events.columns:
                self.events['longitude'] = self.events['longitude'].apply(self.normalize_longitude)
            
            # Convert columns to numeric and filter invalid values
            for col in ['latitude', 'longitude', 'depth']:
                if col in self.events.columns:
                    self.events[col] = pd.to_numeric(self.events[col], errors='coerce')
            
            # Remove events with invalid coordinates
            original_count = len(self.events)
            self.events = self.events.dropna(subset=['latitude', 'longitude', 'depth'])
            self.events = self.events[
                (self.events['latitude'].between(-90, 90)) &
                (self.events['longitude'].between(-180, 180) if self.longitude_format == "180_180" 
                 else self.events['longitude'].between(0, 360)) &
                (self.events['depth'] >= 0)
            ]
            
            if len(self.events) < original_count:
                logger.warning(f"Removed {original_count - len(self.events)} events with invalid coordinates")
            
            logger.info(f"Loaded {len(self.events)} valid events from {filename}")
            
            # Calculate event statistics
            if len(self.events) > 0:
                logger.info(f"Depth range: {self.events['depth'].min():.1f} - {self.events['depth'].max():.1f} km")
                if 'magnitude' in self.events.columns:
                    logger.info(f"Magnitude range: {self.events['magnitude'].min():.1f} - {self.events['magnitude'].max():.1f}")
                
        except Exception as e:
            logger.error(f"Error reading events: {e}")
            raise

    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r

    def calculate_azimuth(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate azimuth between two points in degrees"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        azimuth = np.degrees(np.arctan2(y, x))
        
        # Normalize to 0-360
        return (azimuth + 360) % 360

    def calculate_station_coverage(self, lat: float, lon: float, radius_deg: float) -> Tuple[int, float, float]:
        """Calculate station coverage with improved azimuth calculation"""
        # Check if result is in cache
        cache_key = (round(lat, 5), round(lon, 5), round(radius_deg, 5))
        if cache_key in self.grid_point_cache:
            return self.grid_point_cache[cache_key]
            
        if self.station_kdtree is None or len(self.stations) < 3:
            return 0, 360.0, 0.0
            
        # Convert degrees to radians for the query point
        point = np.radians([lat, lon])
        
        # Convert radius from degrees to radians
        # 1 degree ≈ 111 km at the equator
        radius_rad = np.radians(radius_deg)
        
        # Query the KD-tree for stations within radius
        indices = self.station_kdtree.query_ball_point(point, radius_rad)
        
        if not indices or len(indices) < 3:
            return len(indices) if indices else 0, 360.0, 0.0
            
        station_coords = self.stations.iloc[indices][['Latitude', 'Longitude']].values
        
        # Calculate distances and azimuths more accurately
        distances_km = []
        azimuths = []
        
        for coord in station_coords:
            # Calculate distance using Haversine formula
            dist_km = self.haversine_distance(lat, lon, coord[0], coord[1])
            distances_km.append(dist_km)
            
            # Calculate azimuth
            azimuth = self.calculate_azimuth(lat, lon, coord[0], coord[1])
            azimuths.append(azimuth)
        
        # Calculate azimuthal gap
        sorted_azimuths = np.sort(azimuths)
        # Add the first azimuth again at the end to complete the circle
        sorted_azimuths = np.append(sorted_azimuths, sorted_azimuths[0] + 360)
        # Calculate gaps between consecutive azimuths
        gaps = np.diff(sorted_azimuths)
        max_gap = np.max(gaps)
        
        # Calculate mean distance
        mean_dist_km = np.mean(distances_km)
        
        # Convert mean distance back to degrees for consistency
        mean_dist_deg = mean_dist_km / 111.195
        
        # Cache the result
        result = (len(indices), max_gap, mean_dist_deg)
        self.grid_point_cache[cache_key] = result
        
        return result

    def validate_grid_point(self, point: GridPoint) -> bool:
        """Validate grid point with improved criteria"""
        # Calculate coverage if not already done
        if point.station_count == 0:
            station_count, max_gap, mean_dist = self.calculate_station_coverage(
                point.lat, point.lon, point.max_station_dist)
            point.station_count = station_count
            point.azimuthal_gap = max_gap
            point.mean_station_dist = mean_dist
        
        # Updated validation criteria
        min_stations = max(4, min(point.min_pick_count, 6))
        
        # Adjust max allowed gap based on network type
        if self.optimal_spacing <= 0.5:  # Local/dense network
            max_allowed_gap = 270.0
        elif self.optimal_spacing <= 1.0:  # Regional network
            max_allowed_gap = 300.0
        else:  # Global/sparse network
            max_allowed_gap = 330.0
        
        # Maximum allowed mean distance as a fraction of the maximum station distance
        max_mean_dist = point.max_station_dist * 0.8
        
        return (point.station_count >= min_stations and 
                point.azimuthal_gap <= max_allowed_gap and
                point.mean_station_dist <= max_mean_dist)

    def optimize_grid_point(self, point: GridPoint) -> GridPoint:
        """Optimize grid point with improved metrics"""
        if self.stations is None:
            return point
            
        # Calculate coverage if not already done
        if point.station_count == 0:
            station_count, max_gap, mean_dist = self.calculate_station_coverage(
                point.lat, point.lon, point.max_station_dist)
            point.station_count = station_count
            point.azimuthal_gap = max_gap
            point.mean_station_dist = mean_dist
        else:
            station_count = point.station_count
            max_gap = point.azimuthal_gap
            mean_dist = point.mean_station_dist
            
        if station_count > 0:
            # Adjust pick count based on available stations
            if station_count <= 4:
                point.min_pick_count = 3
            elif station_count <= 8:
                point.min_pick_count = 4
            elif station_count <= 15:
                point.min_pick_count = 5
            elif station_count <= 25:
                point.min_pick_count = 6
            else:
                point.min_pick_count = 8
            
            # Adjust radius based on coverage and azimuthal gap
            if max_gap > 270:
                point.radius = min(15.0, point.radius * 1.5)
            elif max_gap > 220 and mean_dist > point.radius * 0.7:
                point.radius = min(10.0, point.radius * 1.2)
            elif max_gap < 120 and mean_dist < point.radius * 0.5:
                point.radius = max(1.5, point.radius * 0.8)
                
            # Adjust max station distance based on network size
            if self.optimal_spacing <= 0.5:  # Local network
                optimal_max_dist = 10.0
            elif self.optimal_spacing <= 1.0:  # Regional network
                optimal_max_dist = 30.0
            else:  # Global network
                optimal_max_dist = 180.0
                
            # Smoothly adjust max_station_dist toward the optimal value
            point.max_station_dist = 0.7 * point.max_station_dist + 0.3 * optimal_max_dist
                
            # Calculate quality score with weighted components
            azimuth_weight = 0.4
            density_weight = 0.4
            distance_weight = 0.2
            
            azimuth_score = max(0, 1 - max_gap/360)
            
            # Density score calculation: optimize for different network densities
            if self.optimal_spacing <= 0.5:  # Local network
                optimal_station_count = 15
            elif self.optimal_spacing <= 1.0:  # Regional network
                optimal_station_count = 10
            else:  # Global network
                optimal_station_count = 8
                
            density_score = min(1, station_count/optimal_station_count)
            
            # Distance score: penalize grid points with only very distant stations
            distance_score = max(0, 1 - mean_dist/point.max_station_dist)
            
            # Combined score with weights
            point.quality_score = (
                azimuth_weight * azimuth_score +
                density_weight * density_score +
                distance_weight * distance_score
            )
            
            # Apply depth-based adjustment to quality score
            # Prefer standard depths (10, 33, 50, 100 km) with a small boost
            standard_depths = np.array([10, 33, 50, 100])
            depth_diff = np.min(np.abs(point.depth - standard_depths))
            if depth_diff < 1:
                point.quality_score *= 1.05  # 5% boost for standard depths
                
            # Cap quality score at 1.0
            point.quality_score = min(1.0, point.quality_score)
            
        return point

    def create_depth_variations(self, grid_points: List[GridPoint]) -> List[GridPoint]:
        """Create depth variations for high-quality grid points"""
        # Standard depths to consider
        standard_depths = [10, 33, 50, 100, 200, 300]
        
        new_points = []
        existing_hashes = set(hash(p) for p in grid_points)
        
        # Select high-quality points
        for point in grid_points:
            if point.quality_score > 0.7:
                # Only create depth variations for points without existing variations
                existing_depths = {p.depth for p in grid_points 
                                 if abs(p.lat - point.lat) < 0.01 and abs(p.lon - point.lon) < 0.01}
                
                for depth in standard_depths:
                    # Skip if a point already exists at this depth
                    if depth in existing_depths or abs(depth - point.depth) < 1:
                        continue
                        
                    # Create new grid point at different depth
                    new_point = GridPoint(
                        lat=point.lat,
                        lon=point.lon,
                        depth=depth,
                        radius=point.radius,
                        max_station_dist=point.max_station_dist,
                        min_pick_count=point.min_pick_count,
                        quality_score=point.quality_score * 0.95,  # Slightly lower score for variants
                        longitude_format=point.longitude_format,
                        station_count=point.station_count,
                        azimuthal_gap=point.azimuthal_gap,
                        mean_station_dist=point.mean_station_dist,
                        is_new=True
                    )
                    
                    # Only add if it's not a duplicate
                    new_hash = hash(new_point)
                    if new_hash not in existing_hashes:
                        new_points.append(new_point)
                        existing_hashes.add(new_hash)
                        
        return new_points

    def suggest_new_grid_points(self) -> List[GridPoint]:
        """Suggest new grid points based on seismicity and station coverage"""
        if self.events is None or self.stations is None or len(self.events) < 10:
            return []
            
        logger.info("Analyzing seismicity to suggest new grid points...")
        new_points = []
        
        try:
            # Process events with DBSCAN clustering
            coords = self.events[['latitude', 'longitude', 'depth']].values
            # Scale depth for better clustering (give less weight to depth)
            scaled_coords = coords.copy()
            scaled_coords[:, 2] *= 0.1  # Scale depth by factor of 0.1
            
            # Find optimal DBSCAN parameters based on data
            if len(coords) > 1000:
                # For large datasets, use more aggressive clustering
                eps = 0.5
                min_samples = 10
            elif len(coords) > 100:
                # For medium datasets
                eps = 0.4
                min_samples = 5
            else:
                # For small datasets
                eps = 0.3
                min_samples = 3
                
            # Run DBSCAN clustering
            clustering = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='euclidean',
                n_jobs=-1  # Use all available cores
            ).fit(scaled_coords)
            
            cluster_labels = clustering.labels_
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            logger.info(f"Found {n_clusters} event clusters")
            
            # Process each cluster
            existing_hashes = set(hash(p) for p in self.base_grid)
            for label in set(cluster_labels):
                if label == -1:  # Skip noise points
                    continue
                    
                # Get events in this cluster
                cluster_mask = cluster_labels == label
                cluster_events = coords[cluster_mask]
                
                if len(cluster_events) < min_samples:
                    continue
                
                # Calculate cluster statistics
                center = np.median(cluster_events, axis=0)  # Use median for robustness
                
                # Create grid point at cluster center
                point = GridPoint(
                    lat=center[0],
                    lon=center[1],
                    depth=max(1.0, center[2]),  # Ensure depth is at least 1 km
                    radius=5.0,  # Starting radius
                    max_station_dist=180.0,  # Will be adjusted during optimization
                    min_pick_count=6,
                    longitude_format=self.longitude_format,
                    is_new=True
                )
                
                # Optimize the grid point
                point = self.optimize_grid_point(point)
                
                # Check if it's valid and not a duplicate
                if self.validate_grid_point(point) and hash(point) not in existing_hashes:
                    new_points.append(point)
                    existing_hashes.add(hash(point))
                    
                    # For large clusters, also try adding grid points at different depths
                    if len(cluster_events) > 20:
                        # Find depth distribution within cluster
                        depths = cluster_events[:, 2]
                        depth_percentiles = np.percentile(depths, [25, 50, 75])
                        
                        for depth in depth_percentiles:
                            if abs(depth - point.depth) < 5:  # Skip if too close to existing point
                                continue
                                
                            depth_point = GridPoint(
                                lat=point.lat,
                                lon=point.lon,
                                depth=depth,
                                radius=point.radius,
                                max_station_dist=point.max_station_dist,
                                min_pick_count=point.min_pick_count,
                                longitude_format=self.longitude_format,
                                is_new=True
                            )
                            
                            depth_point = self.optimize_grid_point(depth_point)
                            if self.validate_grid_point(depth_point) and hash(depth_point) not in existing_hashes:
                                new_points.append(depth_point)
                                existing_hashes.add(hash(depth_point))
            
            # If there's a good station network but few or no events, suggest grid points
            # based on station distribution
            if len(new_points) < 5 and len(self.stations) > 10:
                logger.info("Adding grid points based on station coverage...")
                
                # Create a grid covering the station network with optimal spacing
                station_lats = self.stations['Latitude'].values
                station_lons = self.stations['Longitude'].values
                
                # Calculate the bounding box with some margin
                margin = 2.0 * self.optimal_spacing
                min_lat = max(-90, np.min(station_lats) - margin)
                max_lat = min(90, np.max(station_lats) + margin)
                min_lon = np.min(station_lons) - margin
                max_lon = np.max(station_lons) + margin
                
                # Adjust lon range for 0-360 format if needed
                if self.longitude_format == "0_360":
                    min_lon = min_lon % 360
                    max_lon = max_lon % 360
                    if min_lon > max_lon:
                        min_lon, max_lon = max_lon, min_lon
                
                # Create grid with optimal spacing
                lat_steps = np.arange(min_lat, max_lat, self.optimal_spacing)
                lon_steps = np.arange(min_lon, max_lon, self.optimal_spacing)
                
                # Standard depths for grid points
                depths = [10, 33, 100]
                
                for lat in lat_steps:
                    for lon in lon_steps:
                        # Skip if too close to existing grid points
                        too_close = False
                        for p in self.base_grid + new_points:
                            if (abs(p.lat - lat) < self.optimal_spacing * 0.5 and 
                                abs(p.lon - lon) < self.optimal_spacing * 0.5):
                                too_close = True
                                break
                        
                        if too_close:
                            continue
                            
                        # Create grid point at 10 km depth (most common default)
                        point = GridPoint(
                            lat=lat,
                            lon=lon,
                            depth=10.0,
                            radius=5.0,
                            max_station_dist=180.0,
                            min_pick_count=6,
                            longitude_format=self.longitude_format,
                            is_new=True
                        )
                        
                        point = self.optimize_grid_point(point)
                        if self.validate_grid_point(point) and hash(point) not in existing_hashes:
                            new_points.append(point)
                            existing_hashes.add(hash(point))
            
            logger.info(f"Generated {len(new_points)} new grid points")
            return new_points
            
        except Exception as e:
            logger.error(f"Error suggesting new grid points: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def write_grid(self, filename: str, include_new: bool = True, min_quality: float = 0.3):
        """Write optimized grid in SeisComP-compatible format"""
        grid_points = list(self.base_grid)
        
        # Add new points if requested
        if include_new:
            try:
                new_points = self.suggest_new_grid_points()
                grid_points.extend(new_points)
                
                # Also add depth variations for high-quality points
                depth_points = self.create_depth_variations(grid_points)
                grid_points.extend(depth_points)
                logger.info(f"Added {len(depth_points)} depth variations")
            except Exception as e:
                logger.error(f"Error adding new points: {e}")
        
        # Optimize all grid points
        optimized_points = []
        for point in grid_points:
            try:
                opt_point = self.optimize_grid_point(point)
                if (self.validate_grid_point(opt_point) and 
                    opt_point.quality_score >= min_quality):
                    optimized_points.append(opt_point)
            except Exception as e:
                logger.error(f"Error optimizing point {point.lat}, {point.lon}: {e}")
        
        # Sort by quality score
        optimized_points.sort(key=lambda x: x.quality_score, reverse=True)
        
        # Write to file - FIXED FORMAT for SeisComP compatibility
        try:
            with open(filename, 'w') as f:
                f.write("# Optimized grid configuration for SeisComP scautoloc\n")
                f.write("# Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("# Format: lat lon depth radius max_dist min_picks\n")
                f.write("# lat: Latitude in degrees\n")
                f.write("# lon: Longitude in degrees\n")
                f.write("# depth: Depth in kilometers\n")
                f.write("# radius: Clustering radius in degrees\n")
                f.write("# max_dist: Maximum station distance in degrees\n")
                f.write("# min_picks: Minimum number of picks required\n")
                f.write("#\n")
                
                for point in optimized_points:
                    # Write only the 6 columns expected by SeisComP
                    f.write(f"{point.lat:.4f} {point.lon:.4f} {point.depth:.1f} "
                        f"{point.radius:.1f} {point.max_station_dist:.1f} "
                        f"{point.min_pick_count}\n")
            
            logger.info(f"Written {len(optimized_points)} optimized grid points to {filename}")
            
            # Write additional metadata file with the quality scores and other metrics
            metadata_file = os.path.splitext(filename)[0] + "_metadata.csv"
            with open(metadata_file, 'w') as f:
                f.write("latitude,longitude,depth,radius,max_station_dist,min_pick_count,quality_score,"
                    "station_count,azimuthal_gap,mean_station_dist,is_new\n")
                
                for point in optimized_points:
                    f.write(f"{point.lat:.4f},{point.lon:.4f},{point.depth:.1f},"
                        f"{point.radius:.1f},{point.max_station_dist:.1f},"
                        f"{point.min_pick_count},{point.quality_score:.3f},"
                        f"{point.station_count},{point.azimuthal_gap:.1f},"
                        f"{point.mean_station_dist:.3f},{1 if point.is_new else 0}\n")
                        
            logger.info(f"Written detailed metadata to {metadata_file}")
            
        except Exception as e:
            logger.error(f"Error writing grid file: {e}")

    def visualize_grid(self, output_file: str = 'grid_visualization.html'):
        """Create interactive visualization with quality metrics"""
        try:
            import folium
            from folium import plugins
            import branca.colormap as cm
        except ImportError:
            logger.error("Visualization requires folium and branca packages. Install with: pip install folium branca")
            return
            
        logger.info("Creating interactive visualization...")
        
        try:
            # Create base map
            if self.stations is not None and len(self.stations) > 0:
                center_lat = np.median(self.stations['Latitude'])
                center_lon = np.median(self.stations['Longitude'])
            elif len(self.base_grid) > 0:
                center_lat = np.median([p.lat for p in self.base_grid])
                center_lon = np.median([p.lon for p in self.base_grid])
            else:
                center_lat, center_lon = 0, 0
                
            m = folium.Map(location=[center_lat, center_lon], 
                          zoom_start=4,
                          tiles='CartoDB positron')
            
            # Add controls
            plugins.Fullscreen().add_to(m)
            folium.plugins.MeasureControl().add_to(m)
            
            # Add layer groups
            grid_group = folium.FeatureGroup(name='Grid Points')
            new_grid_group = folium.FeatureGroup(name='New Grid Points')
            station_group = folium.FeatureGroup(name='Stations')
            event_group = folium.FeatureGroup(name='Events')
            
            # Create colormaps
            quality_colormap = cm.LinearColormap(
                colors=['red', 'orange', 'yellow', 'green'],
                vmin=0, vmax=1,
                caption='Grid Point Quality Score'
            )
            
            # Add stations
            if self.stations is not None:
                for _, row in self.stations.iterrows():
                    popup_html = f"<b>Network:</b> {row.get('Network', 'N/A')}<br>"
                    popup_html += f"<b>Station:</b> {row.get('Station', 'N/A')}<br>"
                    if 'SiteName' in row:
                        popup_html += f"<b>Site:</b> {row['SiteName']}<br>"
                    
                    folium.CircleMarker(
                        location=[row['Latitude'], row['Longitude']],
                        radius=4,
                        color='blue',
                        fill=True,
                        fill_color='blue',
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=f"{row.get('Network', '')}.{row.get('Station', '')}"
                    ).add_to(station_group)
            
            # Process grid points
            optimized_grid = []
            for point in self.base_grid:
                # Optimize the point
                opt_point = self.optimize_grid_point(point)
                if self.validate_grid_point(opt_point):
                    optimized_grid.append(opt_point)
            
            # Add grid points
            for point in optimized_grid:
                color = quality_colormap(point.quality_score)
                
                # Popup content
                popup_html = f"<b>Location:</b> {point.lat:.3f}, {point.lon:.3f}<br>"
                popup_html += f"<b>Depth:</b> {point.depth:.1f} km<br>"
                popup_html += f"<b>Quality Score:</b> {point.quality_score:.3f}<br>"
                popup_html += f"<b>Radius:</b> {point.radius:.1f}°<br>"
                popup_html += f"<b>Max Station Dist:</b> {point.max_station_dist:.1f}°<br>"
                popup_html += f"<b>Min Picks:</b> {point.min_pick_count}<br>"
                popup_html += f"<b>Station Count:</b> {point.station_count}<br>"
                popup_html += f"<b>Azimuthal Gap:</b> {point.azimuthal_gap:.1f}°<br>"
                popup_html += f"<b>Mean Station Dist:</b> {point.mean_station_dist:.1f}°"
                
                # Add grid point marker
                folium.CircleMarker(
                    location=[point.lat, point.lon],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"Grid Point: {point.lat:.3f}, {point.lon:.3f}"
                ).add_to(grid_group)
                
            # Add new grid points
            try:
                new_points = self.suggest_new_grid_points()
                
                for point in new_points:
                    color = quality_colormap(point.quality_score)
                    
                    # Popup content
                    popup_html = f"<b>Location:</b> {point.lat:.3f}, {point.lon:.3f}<br>"
                    popup_html += f"<b>Depth:</b> {point.depth:.1f} km<br>"
                    popup_html += f"<b>Quality Score:</b> {point.quality_score:.3f}<br>"
                    popup_html += f"<b>Radius:</b> {point.radius:.1f}°<br>"
                    popup_html += f"<b>Max Station Dist:</b> {point.max_station_dist:.1f}°<br>"
                    popup_html += f"<b>Min Picks:</b> {point.min_pick_count}<br>"
                    popup_html += f"<b>Station Count:</b> {point.station_count}<br>"
                    popup_html += f"<b>Azimuthal Gap:</b> {point.azimuthal_gap:.1f}°<br>"
                    popup_html += f"<b>Mean Station Dist:</b> {point.mean_station_dist:.1f}°"
                    
                    # Add grid point marker - use star shape for new points
                    folium.CircleMarker(
                        location=[point.lat, point.lon],
                        radius=6,
                        color='black',
                        weight=1,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=f"New Grid Point: {point.lat:.3f}, {point.lon:.3f}"
                    ).add_to(new_grid_group)
            except Exception as e:
                logger.error(f"Error adding new grid points to visualization: {e}")
            
            # Add colormap to the map
            m.add_child(quality_colormap)
            
            # Add events with depth coloring
            if self.events is not None and len(self.events) > 0:
                depth_min = max(0, self.events['depth'].min())
                depth_max = min(700, self.events['depth'].max())
                
                depth_colormap = cm.LinearColormap(
                    colors=['green', 'yellow', 'orange', 'red', 'purple'],
                    vmin=depth_min,
                    vmax=depth_max,
                    caption='Event Depth (km)'
                )
                m.add_child(depth_colormap)
                
                # Add a subset of events for performance (max 2000)
                max_events = 2000
                if len(self.events) > max_events:
                    logger.info(f"Sampling {max_events} events for visualization")
                    events_sample = self.events.sample(max_events, random_state=42)
                else:
                    events_sample = self.events
                
                for _, row in events_sample.iterrows():
                    try:
                        depth = row['depth']
                        color = depth_colormap(depth)
                        
                        # Create popup content
                        popup_html = f"<b>Depth:</b> {depth:.1f} km<br>"
                        if 'magnitude' in row:
                            popup_html += f"<b>Magnitude:</b> {row['magnitude']:.1f}<br>"
                        if 'time' in row:
                            popup_html += f"<b>Time:</b> {row['time']}<br>"
                        
                        folium.CircleMarker(
                            location=[row['latitude'], row['longitude']],
                            radius=3,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.7,
                            popup=folium.Popup(popup_html, max_width=300),
                            tooltip=f"Event: {depth:.1f} km"
                        ).add_to(event_group)
                    except Exception as e:
                        # Skip problematic events
                        continue
            
            # Add all layers and controls
            grid_group.add_to(m)
            new_grid_group.add_to(m)
            station_group.add_to(m)
            event_group.add_to(m)
            folium.LayerControl().add_to(m)
            
            m.save(output_file)
            logger.info(f"Visualization saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def create_matplotlib_visualization(self, output_file: str = 'grid_map.png'):
        """Create a static visualization using matplotlib"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot stations
            if self.stations is not None and len(self.stations) > 0:
                ax.scatter(
                    self.stations['Longitude'], 
                    self.stations['Latitude'],
                    marker='^', 
                    color='blue', 
                    s=20, 
                    alpha=0.7,
                    label='Stations'
                )
            
            # Plot grid points with quality coloring
            if self.base_grid:
                qualities = [p.quality_score for p in self.base_grid]
                lats = [p.lat for p in self.base_grid]
                lons = [p.lon for p in self.base_grid]
                
                scatter = ax.scatter(
                    lons,
                    lats,
                    c=qualities,
                    cmap='RdYlGn',
                    marker='o',
                    s=30,
                    alpha=0.8,
                    label='Grid Points',
                    vmin=0,
                    vmax=1
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Quality Score')
            
            # Plot events if available
            if self.events is not None and len(self.events) > 0:
                # Sample events for clarity
                max_events = 1000
                if len(self.events) > max_events:
                    events_sample = self.events.sample(max_events, random_state=42)
                else:
                    events_sample = self.events
                    
                ax.scatter(
                    events_sample['longitude'],
                    events_sample['latitude'],
                    c=events_sample['depth'],
                    cmap='viridis',
                    marker='.',
                    s=10,
                    alpha=0.5,
                    label='Events'
                )
            
            # Set plot labels and title
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('Grid Optimization Results')
            ax.legend()
            
            # Add gridlines
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Save figure
            plt.tight_layout()
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Static visualization saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating matplotlib visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def generate_report(self, output_file: str = "grid_optimization_report.txt"):
        """Generate a detailed report on the grid optimization"""
        try:
            with open(output_file, 'w') as f:
                f.write("====================================\n")
                f.write("SeisComP Grid Optimization Report\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("====================================\n\n")
                
                # Network statistics
                f.write("Network Statistics:\n")
                f.write("------------------\n")
                if self.stations is not None:
                    f.write(f"Total stations: {len(self.stations)}\n")
                    coverage_area = self.calculate_coverage_area()
                    f.write(f"Coverage area: {coverage_area:.2f} km²\n")
                    f.write(f"Station density: {len(self.stations)/coverage_area if coverage_area > 0 else 0:.8f} stations/km²\n")
                    f.write(f"Optimal grid spacing: {self.optimal_spacing:.2f}°\n")
                else:
                    f.write("No station data available\n")
                    
                f.write("\n")
                
                # Grid statistics
                f.write("Grid Statistics:\n")
                f.write("---------------\n")
                if self.base_grid:
                    f.write(f"Original grid points: {len(self.base_grid)}\n")
                    
                    # Calculate statistics on optimized points
                    optimized_points = []
                    for point in self.base_grid:
                        opt_point = self.optimize_grid_point(point)
                        if self.validate_grid_point(opt_point):
                            optimized_points.append(opt_point)
                    
                    f.write(f"Valid optimized points: {len(optimized_points)}\n")
                    
                    if optimized_points:
                        qualities = [p.quality_score for p in optimized_points]
                        f.write(f"Quality score range: {min(qualities):.3f} - {max(qualities):.3f}\n")
                        f.write(f"Average quality score: {np.mean(qualities):.3f}\n")
                        f.write(f"Median quality score: {np.median(qualities):.3f}\n")
                        
                        # Statistics on station coverage
                        station_counts = [p.station_count for p in optimized_points]
                        f.write(f"Station count range: {min(station_counts)} - {max(station_counts)}\n")
                        f.write(f"Average stations per grid point: {np.mean(station_counts):.1f}\n")
                        
                        # Statistics on azimuthal gaps
                        gaps = [p.azimuthal_gap for p in optimized_points]
                        f.write(f"Azimuthal gap range: {min(gaps):.1f}° - {max(gaps):.1f}°\n")
                        f.write(f"Average azimuthal gap: {np.mean(gaps):.1f}°\n")
                else:
                    f.write("No grid data available\n")
                    
                f.write("\n")
                
                # Event statistics
                f.write("Event Statistics:\n")
                f.write("----------------\n")
                if self.events is not None:
                    f.write(f"Total events: {len(self.events)}\n")
                    if 'depth' in self.events.columns:
                        f.write(f"Depth range: {self.events['depth'].min():.1f} - {self.events['depth'].max():.1f} km\n")
                        f.write(f"Average depth: {self.events['depth'].mean():.1f} km\n")
                    if 'magnitude' in self.events.columns:
                        f.write(f"Magnitude range: {self.events['magnitude'].min():.1f} - {self.events['magnitude'].max():.1f}\n")
                else:
                    f.write("No event data available\n")
                    
                f.write("\n")
                
                # Recommendations
                f.write("Recommendations:\n")
                f.write("----------------\n")
                
                if self.stations is not None and len(self.stations) > 0:
                    station_density = len(self.stations) / self.calculate_coverage_area() if self.calculate_coverage_area() > 0 else 0
                    
                    if station_density < 0.00001:  # Very sparse network
                        f.write("- Network is very sparse. Consider using larger grid spacing.\n")
                        f.write("- Increase 'radius' parameter for grid points.\n")
                        f.write("- Decrease 'min_pick_count' for sparse areas.\n")
                    elif station_density < 0.0001:  # Sparse network
                        f.write("- Network has moderate density. Default grid parameters should work well.\n")
                        f.write("- Consider adding more depth layers for better depth resolution.\n")
                    else:  # Dense network
                        f.write("- Network is dense. Use smaller grid spacing for better resolution.\n")
                        f.write("- Decrease 'radius' parameter for dense areas.\n")
                        f.write("- Increase 'min_pick_count' for better stability.\n")
                
                if self.events is not None and len(self.events) > 0:
                    f.write("- Grid points have been optimized based on historical seismicity.\n")
                    f.write("- Consider adding more grid points in areas with recurring seismicity.\n")
                
                f.write("\n")
                f.write("Quality Score Interpretation:\n")
                f.write("- 0.0-0.3: Poor coverage, may produce unreliable locations\n")
                f.write("- 0.3-0.6: Moderate coverage, acceptable for many applications\n")
                f.write("- 0.6-0.8: Good coverage, should produce reliable locations\n")
                f.write("- 0.8-1.0: Excellent coverage, optimal for high-precision locations\n")
                
            logger.info(f"Report written to {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Optimize seismic grid configuration for SeisComP scautoloc')
    parser.add_argument('base_grid', help='Base grid configuration file')
    parser.add_argument('station_file', help='Station information file')
    parser.add_argument('--events', help='Seismic events file (optional)')
    parser.add_argument('--output', default='optimized_grid.txt',
                       help='Output grid file (default: optimized_grid.txt)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create interactive visualization')
    parser.add_argument('--static-plot', action='store_true',
                       help='Create static plot with matplotlib')
    parser.add_argument('--longitude-format', choices=['180_180', '0_360'],
                       default='180_180', help='Longitude format (default: 180_180)')
    parser.add_argument('--min-quality', type=float, default=0.3,
                        help='Minimum quality score for grid points (default: 0.3)')
    parser.add_argument('--report', action='store_true',
                        help='Generate detailed optimization report')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        logger.info(f"Grid optimization started")
        logger.info(f"Base grid: {args.base_grid}")
        logger.info(f"Station file: {args.station_file}")
        if args.events:
            logger.info(f"Events file: {args.events}")
        
        # Create optimizer
        optimizer = GridOptimizer(longitude_format=args.longitude_format)
        
        # Load data
        logger.info("Loading base grid...")
        optimizer.read_base_grid(args.base_grid)
        
        logger.info("Loading station data...")
        optimizer.read_stations(args.station_file)
        
        if args.events:
            logger.info("Loading event data...")
            optimizer.read_events(args.events)
        
        # Optimize grid
        logger.info("Optimizing grid...")
        optimizer.write_grid(args.output, include_new=True, min_quality=args.min_quality)
        
        # Generate visualizations if requested
        if args.visualize:
            logger.info("Creating interactive visualization...")
            optimizer.visualize_grid(output_file=os.path.splitext(args.output)[0] + "_map.html")
            
        if args.static_plot:
            logger.info("Creating static plot...")
            optimizer.create_matplotlib_visualization(output_file=os.path.splitext(args.output)[0] + "_map.png")
            
        if args.report:
            logger.info("Generating report...")
            optimizer.generate_report(output_file=os.path.splitext(args.output)[0] + "_report.txt")
        
        logger.info("Grid optimization completed successfully")
        
    except Exception as e:
        logger.error(f"Error during grid optimization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
        
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())