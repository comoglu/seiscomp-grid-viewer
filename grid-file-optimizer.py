#!/usr/bin/python3 

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
from dataclasses import dataclass
import argparse
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import logging
import json
from pathlib import Path
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('grid_optimization.log')
    ]
)

@dataclass
class GridPoint:
    lat: float
    lon: float
    depth: float
    radius: float
    max_station_dist: float
    min_pick_count: int
    
    # Add validation metrics
    station_coverage: float = 0.0  # Percentage of area covered by stations
    azimuthal_gap: float = 360.0   # Largest azimuthal gap between stations
    theoretical_uncertainty: float = 0.0  # Estimated location uncertainty

class GridMetrics:
    """Class to store and calculate grid quality metrics"""
    def __init__(self):
        self.mean_station_coverage: float = 0.0
        self.mean_azimuthal_gap: float = 0.0
        self.mean_uncertainty: float = 0.0
        self.coverage_uniformity: float = 0.0
        self.point_density: float = 0.0
        
    def to_dict(self) -> Dict:
        return {
            'mean_station_coverage': self.mean_station_coverage,
            'mean_azimuthal_gap': self.mean_azimuthal_gap,
            'mean_uncertainty': self.mean_uncertainty,
            'coverage_uniformity': self.coverage_uniformity,
            'point_density': self.point_density
        }

def convert_longitude(lon: float, to_format: str) -> float:
    """Convert longitude between 0-360 and -180-180 formats"""
    if to_format == '0_360':
        return lon % 360
    elif to_format == '180_180':
        lon = lon % 360
        if lon > 180:
            lon -= 360
        return lon
    else:
        raise ValueError("Invalid longitude format. Use '0_360' or '180_180'")

class GridOptimizer:
    def __init__(self, longitude_format: str = '0_360'):
        self.base_grid = []
        self.stations = None
        self.events = None
        self.station_kdtree = None
        self.metrics = GridMetrics()
        if longitude_format not in ['0_360', '180_180']:
            raise ValueError("Invalid longitude format. Use '0_360' or '180_180'")
        self.longitude_format = longitude_format
        self.longitude_format = longitude_format

    def read_base_grid(self, filename: str):
        """Read the base grid configuration"""
        try:
            logging.info(f"Reading base grid from {filename}")
            data = pd.read_csv(filename, delimiter=r'\s+', header=None,
                             names=['lat', 'lon', 'depth', 'radius', 
                                   'max_station_dist', 'min_pick_count'])
                                   
            # Convert longitudes to desired format
            data['lon'] = data['lon'].apply(lambda x: convert_longitude(x, self.longitude_format))
            
            self.base_grid = [
                GridPoint(row.lat, row.lon, row.depth, row.radius, 
                         row.max_station_dist, row.min_pick_count)
                for _, row in data.iterrows()
            ]
            logging.info(f"Loaded {len(self.base_grid)} grid points from base configuration")
            logging.info(f"Using longitude format: {self.longitude_format}")
        except Exception as e:
            logging.error(f"Error reading base grid file: {e}")
            raise

    def read_stations(self, filename: str):
        """Read station information from file"""
        try:
            logging.info(f"Reading station information from {filename}")
            # Read the first line to get the header
            with open(filename, 'r') as f:
                header_line = f.readline().strip()
                if header_line.startswith('#'):
                    header_line = header_line[1:]
                columns = [col.strip() for col in header_line.split('|')]
            
            # Read the data with the extracted column names
            self.stations = pd.read_csv(filename, delimiter='|', 
                                      names=columns, skiprows=1, 
                                      comment='#', skip_blank_lines=True)
                                      
            # Convert longitudes to desired format
            self.stations['Longitude'] = self.stations['Longitude'].apply(
                lambda x: convert_longitude(float(x), self.longitude_format)
            )
            
            # Create KDTree for efficient distance calculations
            # Convert lat/lon to radians for KDTree
            coords = np.radians(self.stations[['Latitude', 'Longitude']].values.astype(float))
            self.station_kdtree = KDTree(coords)
            
            logging.info(f"Loaded {len(self.stations)} stations")
            logging.info(f"Network coverage: {len(set(self.stations['Network']))} networks")
        except Exception as e:
            logging.error(f"Error reading station file: {e}")
            raise

    def read_events(self, filename: str):
        """Read seismic events data"""
        try:
            logging.info(f"Reading seismic events from {filename}")
            self.events = pd.read_csv(filename)
            
            # Convert longitudes to desired format
            self.events['longitude'] = self.events['longitude'].apply(
                lambda x: convert_longitude(x, self.longitude_format)
            )
            
            event_stats = {
                'count': len(self.events),
                'depth_range': f"{self.events['depth'].min():.1f} - {self.events['depth'].max():.1f} km",
                'magnitude_range': f"{self.events['magnitude'].min():.1f} - {self.events['magnitude'].max():.1f}"
            }
            
            logging.info("Event statistics:")
            for key, value in event_stats.items():
                logging.info(f"  - {key}: {value}")
                
        except Exception as e:
            logging.error(f"Error reading events file: {e}")
            raise

    def calculate_station_density(self, lat: float, lon: float, radius_deg: float) -> int:
        """Calculate number of stations within radius of a point"""
        if self.station_kdtree is None:
            return 0
        
        # Convert to radians for spherical distance calculation
        point = np.radians([lat, lon])
        radius_rad = np.radians(radius_deg)
        
        # Query KDTree for stations within radius
        indices = self.station_kdtree.query_ball_point(point, radius_rad)
        return len(indices)

    def calculate_azimuthal_gap(self, lat: float, lon: float, radius_deg: float) -> float:
        """Calculate the largest azimuthal gap between stations"""
        if self.station_kdtree is None:
            return 360.0
            
        point = np.radians([lat, lon])
        radius_rad = np.radians(radius_deg)
        
        # Find stations within radius
        indices = self.station_kdtree.query_ball_point(point, radius_rad)
        if len(indices) < 2:
            return 360.0
            
        # Calculate azimuths to all stations
        stations = self.stations.iloc[indices]
        azimuths = []
        for _, station in stations.iterrows():
            azimuth = np.degrees(np.arctan2(
                station['Longitude'] - lon,
                station['Latitude'] - lat
            )) % 360
            azimuths.append(azimuth)
            
        # Sort azimuths and find largest gap
        azimuths.sort()
        gaps = np.diff(azimuths)
        max_gap = max(gaps)
        
        # Check gap crossing 360/0
        gap_crossing_zero = 360 - azimuths[-1] + azimuths[0]
        max_gap = max(max_gap, gap_crossing_zero)
        
        return max_gap

    def estimate_location_uncertainty(self, point: GridPoint) -> float:
        """Estimate theoretical location uncertainty based on station geometry"""
        if self.station_kdtree is None:
            return float('inf')
            
        # Find stations within maximum distance
        indices = self.station_kdtree.query_ball_point(
            np.radians([point.lat, point.lon]),
            np.radians(point.max_station_dist)
        )
        
        if len(indices) < 4:  # Minimum stations needed for location
            return float('inf')
            
        stations = self.stations.iloc[indices]
        
        # Calculate station distribution matrix
        station_coords = stations[['Latitude', 'Longitude']].values
        relative_positions = station_coords - np.array([point.lat, point.lon])
        
        # Compute condition number of the design matrix
        design_matrix = np.column_stack([
            relative_positions,
            np.ones(len(relative_positions))
        ])
        
        try:
            condition_number = np.linalg.cond(design_matrix)
            # Scale condition number to meaningful uncertainty value (km)
            uncertainty = np.log10(condition_number) * point.depth / 10
            return min(uncertainty, 100.0)  # Cap at 100 km
        except np.linalg.LinAlgError:
            return float('inf')

    def optimize_grid_point(self, point: GridPoint, iteration: int) -> GridPoint:
        """Optimize grid point parameters based on station distribution and seismicity"""
        logging.info(f"Optimizing grid point at ({point.lat:.2f}, {point.lon:.2f}, {point.depth:.1f}km) - Iteration {iteration}")
        
        if self.stations is None:
            return point
            
        # Calculate station density and coverage metrics
        station_count = self.calculate_station_density(point.lat, point.lon, point.radius)
        azimuthal_gap = self.calculate_azimuthal_gap(point.lat, point.lon, point.radius)
        uncertainty = self.estimate_location_uncertainty(point)
        
        logging.info(f"  Initial metrics:")
        logging.info(f"  - Station count: {station_count}")
        logging.info(f"  - Azimuthal gap: {azimuthal_gap:.1f}°")
        logging.info(f"  - Location uncertainty: {uncertainty:.1f} km")
        
        # Adjust parameters based on metrics
        original_params = {
            'radius': point.radius,
            'min_pick_count': point.min_pick_count,
            'max_station_dist': point.max_station_dist
        }
        
        # Adjust radius based on azimuthal gap
        if azimuthal_gap > 180:
            point.radius = min(8.0, point.radius * 1.2)
        elif azimuthal_gap < 90 and station_count > 15:
            point.radius = max(2.0, point.radius * 0.9)
            
        # Adjust minimum pick count based on station density
        if station_count > 0:
            ideal_pick_count = min(max(4, station_count // 3), 12)
            point.min_pick_count = ideal_pick_count
            
        # Adjust maximum station distance based on uncertainty
        if uncertainty > 20 and point.max_station_dist < 150:
            point.max_station_dist = min(180.0, point.max_station_dist * 1.2)
        elif uncertainty < 5 and point.max_station_dist > 100:
            point.max_station_dist = max(90.0, point.max_station_dist * 0.9)
            
        # Store optimization metrics
        point.station_coverage = station_count / max(1, np.pi * (point.radius ** 2))
        point.azimuthal_gap = azimuthal_gap
        point.theoretical_uncertainty = uncertainty
        
        # Log parameter changes
        logging.info("  Parameter adjustments:")
        for param in ['radius', 'min_pick_count', 'max_station_dist']:
            old_val = original_params[param]
            new_val = getattr(point, param)
            if abs(old_val - new_val) > 1e-6:
                logging.info(f"  - {param}: {old_val:.1f} -> {new_val:.1f}")
                
        return point

    def suggest_new_grid_points(self) -> List[GridPoint]:
        """Suggest new grid points based on seismicity clusters and station gaps"""
        if self.events is None or self.stations is None:
            return []
        
        logging.info("Analyzing seismicity patterns for new grid points...")
        new_points = []
        
        # Find seismicity clusters
        if len(self.events) > 0:
            from sklearn.cluster import DBSCAN
            
            # Prepare coordinates for clustering
            coords = self.events[['latitude', 'longitude', 'depth']].values
            # Scale depth to have less weight in clustering
            coords[:, 2] = coords[:, 2] * 0.1
            
            # Perform clustering
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(coords)
            
            # For each significant cluster, suggest a grid point
            unique_labels = set(clustering.labels_)
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                    
                cluster_points = coords[clustering.labels_ == label]
                center = np.mean(cluster_points, axis=0)
                
                # Unscale depth
                center[2] = center[2] * 10
                
                # Create new grid point
                new_point = GridPoint(
                    lat=center[0],
                    lon=center[1],
                    depth=center[2],
                    radius=5.0,  # Default radius
                    max_station_dist=180.0,
                    min_pick_count=6
                )
                
                # Optimize parameters for the new point
                new_point = self.optimize_grid_point(new_point, 0)
                new_points.append(new_point)
                
                logging.info(f"Added new grid point at ({new_point.lat:.2f}, {new_point.lon:.2f}, "
                           f"{new_point.depth:.1f}km) based on seismicity cluster")
        
        return new_points

    def calculate_grid_metrics(self) -> GridMetrics:
        """Calculate overall grid quality metrics"""
        if not self.base_grid:
            return self.metrics
            
        coverages = [p.station_coverage for p in self.base_grid]
        gaps = [p.azimuthal_gap for p in self.base_grid]
        uncertainties = [p.theoretical_uncertainty for p in self.base_grid]
        
        self.metrics.mean_station_coverage = np.mean(coverages)
        self.metrics.mean_azimuthal_gap = np.mean(gaps)
        self.metrics.mean_uncertainty = np.mean(uncertainties)
        self.metrics.coverage_uniformity = 1 - np.std(coverages) / np.mean(coverages)
        
        # Calculate point density using KDE
        coords = np.array([[p.lat, p.lon] for p in self.base_grid])
        if len(coords) > 1:
            kde = gaussian_kde(coords.T)
            self.metrics.point_density = kde.evaluate(coords.T).mean()
            
        return self.metrics

    def visualize_grid(self, output_file: str = 'grid_visualization.html'):
        """Create an interactive visualization of the grid points, stations, and events"""
        import folium
        from folium import plugins
        import branca.colormap as cm

        logging.info("Creating interactive visualization...")

        # Create base map centered on the data
        center_lat = 0
        center_lon = 0
        
        if self.stations is not None:
            center_lat = self.stations['Latitude'].mean()
            center_lon = self.stations['Longitude'].mean()
        elif self.base_grid:
            center_lat = np.mean([p.lat for p in self.base_grid])
            center_lon = np.mean([p.lon for p in self.base_grid])

        m = folium.Map(location=[center_lat, center_lon], 
                      zoom_start=4,
                      tiles='CartoDB positron')
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Add scale bar
        folium.plugins.MeasureControl().add_to(m)

        # Create feature groups for different elements
        grid_group = folium.FeatureGroup(name='Grid Points')
        station_group = folium.FeatureGroup(name='Stations')
        event_group = folium.FeatureGroup(name='Events')
        uncertainty_group = folium.FeatureGroup(name='Uncertainty Areas')

        # Add stations if available
        if self.stations is not None:
            logging.info("Adding stations to visualization...")
            for _, row in self.stations.iterrows():
                folium.RegularPolygonMarker(
                    location=[row['Latitude'], row['Longitude']],
                    number_of_sides=3,
                    radius=6,
                    color='blue',
                    fill=True,
                    popup=f"Network: {row['Network']}<br>"
                          f"Station: {row['Station']}<br>"
                          f"Site: {row['SiteName']}"
                ).add_to(station_group)

        # Add grid points with uncertainty areas
        logging.info("Adding grid points to visualization...")
        for point in self.base_grid:
            # Create circle marker for grid point
            folium.CircleMarker(
                location=[point.lat, point.lon],
                radius=5,
                color='red',
                fill=True,
                popup=f"Depth: {point.depth} km<br>"
                      f"Radius: {point.radius}°<br>"
                      f"Max Station Dist: {point.max_station_dist}°<br>"
                      f"Min Pick Count: {point.min_pick_count}<br>"
                      f"Azimuthal Gap: {point.azimuthal_gap:.1f}°<br>"
                      f"Uncertainty: {point.theoretical_uncertainty:.1f} km"
            ).add_to(grid_group)
            
            # Add circle showing the sensitivity radius
            folium.Circle(
                location=[point.lat, point.lon],
                radius=point.radius * 111000,  # Convert degrees to meters (approx)
                color='red',
                weight=1,
                fill=False,
                opacity=0.3
            ).add_to(grid_group)
            
            # Add uncertainty area
            if point.theoretical_uncertainty < 100:
                folium.Circle(
                    location=[point.lat, point.lon],
                    radius=point.theoretical_uncertainty * 1000,  # Convert km to meters
                    color='purple',
                    weight=1,
                    fill=True,
                    fillOpacity=0.1
                ).add_to(uncertainty_group)

        # Add events if available
        if self.events is not None:
            logging.info("Adding events to visualization...")
            # Create colormap for event depths
            depth_min = self.events['depth'].min()
            depth_max = self.events['depth'].max()
            colormap = cm.LinearColormap(
                colors=['green', 'yellow', 'red'],
                vmin=depth_min,
                vmax=depth_max,
                caption='Event Depth (km)'
            )
            m.add_child(colormap)

            for _, row in self.events.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=4,
                    color=colormap(row['depth']),
                    fill=True,
                    popup=f"Depth: {row['depth']} km<br>"
                          f"Magnitude: {row['magnitude']}<br>"
                          f"Time: {row['time']}"
                ).add_to(event_group)

        # Add all feature groups to map
        grid_group.add_to(m)
        station_group.add_to(m)
        event_group.add_to(m)
        uncertainty_group.add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save the map
        m.save(output_file)
        logging.info(f"Interactive visualization saved to {output_file}")

    def write_grid(self, filename: str, include_new: bool = True):
        """Write the optimized grid configuration to file"""
        grid_points = self.base_grid.copy()
        
        if include_new:
            new_points = self.suggest_new_grid_points()
            grid_points.extend(new_points)
            
        # Calculate final metrics for logging
        metrics = self.calculate_grid_metrics()
        
        # Write grid points in original format
        with open(filename, 'w') as f:
            for point in grid_points:
                # Format exactly as specified: lat lon depth radius max_station_dist min_pick_count
                f.write(f"{point.lat:7.2f} {point.lon:7.2f} {point.depth:4.1f} "
                       f"{point.radius:4.1f} {point.max_station_dist:5.1f} "
                       f"{point.min_pick_count}\n")
        
        # Write detailed metrics to separate file for analysis
        metrics_file = Path(filename).with_suffix('.metrics.json')
        with open(metrics_file, 'w') as f:
            metrics_data = {
                'grid_metrics': metrics.to_dict(),
                'point_details': [{
                    'lat': p.lat,
                    'lon': p.lon,
                    'depth': p.depth,
                    'station_coverage': p.station_coverage,
                    'azimuthal_gap': p.azimuthal_gap,
                    'theoretical_uncertainty': p.theoretical_uncertainty
                } for p in grid_points]
            }
            json.dump(metrics_data, f, indent=2)
            
        logging.info(f"Written {len(grid_points)} grid points to {filename}")
        logging.info(f"Detailed metrics saved to {metrics_file}")
        logging.info("\nGrid optimization summary:")
        for metric, value in metrics.to_dict().items():
            logging.info(f"  - {metric}: {value:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Optimize seismic grid configuration')
    parser.add_argument('base_grid', help='Base grid configuration file')
    parser.add_argument('station_file', help='Station information file')
    parser.add_argument('--events', help='Seismic events file (optional)')
    parser.add_argument('--output', default='optimized_grid.txt', 
                       help='Output grid file (default: optimized_grid.txt)')
    parser.add_argument('--visualize', action='store_true', 
                       help='Create visualization plot')
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of optimization iterations (default: 3)')
    parser.add_argument('--min-stations', type=int, default=4,
                       help='Minimum number of stations required for location (default: 4)')
    parser.add_argument('--longitude-format', choices=['0_360', '180_180'],
                       default='0_360',
                       help='Longitude format for input and output (0_360 or 180_180, default: 0_360)')
    
    args = parser.parse_args()
    
    optimizer = GridOptimizer(longitude_format=args.longitude_format)
    
    # Load input files
    optimizer.read_base_grid(args.base_grid)
    optimizer.read_stations(args.station_file)
    
    if args.events:
        optimizer.read_events(args.events)
    
    # Perform multiple iterations of optimization
    for iteration in range(args.iterations):
        logging.info(f"\nStarting optimization iteration {iteration + 1}/{args.iterations}")
        optimizer.base_grid = [
            optimizer.optimize_grid_point(point, iteration + 1) 
            for point in optimizer.base_grid
        ]
        
        # Calculate and log intermediate metrics
        metrics = optimizer.calculate_grid_metrics()
        logging.info(f"\nIteration {iteration + 1} metrics:")
        for metric, value in metrics.to_dict().items():
            logging.info(f"  - {metric}: {value:.3f}")
    
    # Write optimized grid
    optimizer.write_grid(args.output)
    
    # Create visualization if requested
    if args.visualize:
        optimizer.visualize_grid()

if __name__ == '__main__':
    main()