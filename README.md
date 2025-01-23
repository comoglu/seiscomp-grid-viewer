# Seismic Grid Optimizer

A Python tool for optimizing seismic monitoring grid configurations based on station distribution, seismicity patterns, and network geometry. This tool is designed to work with SeisComP grid configuration files.

## Overview

The Seismic Grid Optimizer helps optimize grid point configurations for seismic monitoring networks by:
- Analyzing station distribution and network geometry
- Evaluating azimuthal gaps and station coverage
- Estimating theoretical location uncertainties
- Suggesting grid point adjustments based on network capabilities
- Providing visualization of the network configuration

## Installation

### Prerequisites
```bash
python >= 3.8
numpy
pandas
scipy
scikit-learn
folium
matplotlib
branca
```

### Installation
```bash
# Clone the repository
git clone https://github.com/comoglu/seiscomp-grid-viewer.git

# Install required packages
pip install numpy pandas scipy scikit-learn folium matplotlib branca
```

## Usage

Basic usage:
```bash
python grid-optimizer.py base_grid.txt stations.txt \
    --events events.csv \
    --output optimized_grid.txt \
    --visualize \
    --iterations 3 \
    --longitude-format 0_360
```

### Command Line Arguments
- `base_grid`: Input grid configuration file (required)
- `stations`: Station information file (required)
- `--events`: Seismic events file (optional)
- `--output`: Output grid file (default: optimized_grid.txt)
- `--visualize`: Create interactive visualization
- `--iterations`: Number of optimization iterations (default: 3)
- `--min-stations`: Minimum stations required for location (default: 4)
- `--longitude-format`: Longitude format (0_360 or 180_180, default: 0_360)

## Input File Formats

### Grid Configuration File
The grid configuration file follows the SeisComP format with 6 columns:
```
# lat lon depth radius max_station_dist min_pick_count
-10.00 105.00 20.0 5.0 180.0 8
```

Fields:
- `lat`: Latitude in degrees (-90 to 90)
- `lon`: Longitude in degrees (format specified by --longitude-format)
- `depth`: Depth in kilometers
- `radius`: Sensitivity radius in degrees
- `max_station_dist`: Maximum allowed station distance in degrees
- `min_pick_count`: Minimum required pick count

### Station File
Standard SeisComP station file format with pipe-separated values:
```
#Network|Station|Latitude|Longitude|Elevation|SiteName
AU|ARMA|-30.4214|151.6302|372|Armidale, NSW
```

### Events File (Optional)
CSV file containing historical seismicity:
```csv
time,latitude,longitude,depth,magnitude
2024-01-01T00:00:00,-35.5,142.3,10.5,3.2
```

## Optimization Process

The tool optimizes grid points through several steps:

1. **Station Coverage Analysis**
   - Uses KDTree for efficient spatial queries
   - Calculates station density within grid point radius
   - Evaluates azimuthal gaps in station distribution

2. **Grid Point Parameter Adjustment**
   - Adapts radius based on station coverage
   - Adjusts minimum pick count requirements
   - Optimizes maximum station distance

3. **Network Capability Assessment**
   - Evaluates theoretical location uncertainties
   - Considers station geometry effects
   - Analyzes seismicity patterns (if events file provided)

## Output Files

### 1. Optimized Grid File
Maintains the same format as input with optimized parameters:
```
-10.00 105.00 15.0 4.5 160.0 6
```

### 2. Metrics File (JSON)
Contains optimization metrics and statistics:
```json
{
  "grid_metrics": {
    "mean_station_coverage": 0.856,
    "mean_azimuthal_gap": 125.3,
    "mean_uncertainty": 8.4,
    "coverage_uniformity": 0.923,
    "point_density": 0.045
  }
}
```

### 3. Log File
Detailed optimization process log:
```
2025-01-23 10:00:00 [INFO] Starting optimization iteration 1/3
2025-01-23 10:00:00 [INFO] Optimizing grid point at (45.32, -122.67, 10.0km)
2025-01-23 10:00:00 [INFO]   - Station count: 15
2025-01-23 10:00:00 [INFO]   - Azimuthal gap: 120.5°
```

## Visualization

The tool generates an interactive HTML visualization showing:
- Grid points with their sensitivity radii
- Station locations and networks
- Seismicity patterns (if events file provided)
- Network coverage analysis
- Interactive controls for layer visibility

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues on the GitHub repository.

## License

See the LICENSE file in the repository.

## Repository

This tool is part of the seiscomp-grid-viewer project:
https://github.com/comoglu/seiscomp-grid-viewer

For issues, feature requests, or contributions, please use the GitHub issue tracker.
