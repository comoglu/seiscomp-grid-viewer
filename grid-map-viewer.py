import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QFileDialog,
                           QLabel, QComboBox, QScrollArea, QCheckBox,
                           QGroupBox, QFrame)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt
import folium
import pandas as pd
import tempfile
from datetime import datetime

class LayerControl(QWidget):
    def __init__(self, name, color, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.checkbox = QCheckBox(name)
        self.checkbox.setChecked(True)
        layout.addWidget(self.checkbox)
        
        # Add color indicator
        color_indicator = QFrame()
        color_indicator.setStyleSheet(f"background-color: {color};")
        color_indicator.setFixedSize(16, 16)
        layout.addWidget(color_indicator)
        
        layout.addStretch()
        self.setLayout(layout)

class GridMapViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grid Point Map Viewer")
        self.setGeometry(100, 100, 1400, 800)
        
        # Store loaded data
        self.grid_data = {}
        self.station_data = {}
        self.colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen']
        
        self.init_ui()
        
    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create left control panel
        left_panel = QWidget()
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        # Grid Files Section
        grid_group = QGroupBox("Grid Files")
        grid_layout = QVBoxLayout()
        
        load_grid_button = QPushButton("Load Grid File")
        load_grid_button.clicked.connect(self.load_grid_file)
        grid_layout.addWidget(load_grid_button)
        
        self.grid_layers = QWidget()
        self.grid_layers_layout = QVBoxLayout(self.grid_layers)
        grid_layout.addWidget(self.grid_layers)
        
        grid_group.setLayout(grid_layout)
        left_layout.addWidget(grid_group)
        
        # Station Files Section
        station_group = QGroupBox("Station Files")
        station_layout = QVBoxLayout()
        
        load_station_button = QPushButton("Load Station File")
        load_station_button.clicked.connect(self.load_station_file)
        station_layout.addWidget(load_station_button)
        
        self.station_layers = QWidget()
        self.station_layers_layout = QVBoxLayout(self.station_layers)
        station_layout.addWidget(self.station_layers)
        
        station_group.setLayout(station_layout)
        left_layout.addWidget(station_group)
        
        # Filters Section
        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout()
        
        # Depth filter
        self.depth_filter = QComboBox()
        self.depth_filter.addItems(["All Depths", "0-100 km", "100-300 km", ">300 km"])
        self.depth_filter.currentTextChanged.connect(self.update_map)
        filter_layout.addWidget(QLabel("Depth Filter:"))
        filter_layout.addWidget(self.depth_filter)
        
        # Radius filter
        self.radius_filter = QComboBox()
        self.radius_filter.addItems(["All Radii", "<5°", "5-10°", ">10°"])
        self.radius_filter.currentTextChanged.connect(self.update_map)
        filter_layout.addWidget(QLabel("Radius Filter:"))
        filter_layout.addWidget(self.radius_filter)
        
        filter_group.setLayout(filter_layout)
        left_layout.addWidget(filter_group)
        
        # Add stretch to push everything up
        left_layout.addStretch()
        
        # Add left panel to main layout
        main_layout.addWidget(left_panel)
        
        # Create map view
        self.web_view = QWebEngineView()
        main_layout.addWidget(self.web_view)
        
        # Set stretch factor to make map larger
        main_layout.setStretchFactor(self.web_view, 4)
        
        # Initialize empty map
        self.create_empty_map()
        
    def create_empty_map(self):
        """Create an empty map centered at (0, 0)"""
        m = folium.Map(location=[0, 0], zoom_start=2)
        
        # Save map to temporary file and display it
        temp_file = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        m.save(temp_file.name)
        self.web_view.setUrl(QUrl.fromLocalFile(temp_file.name))
        
    def load_grid_file(self):
        """Load a grid configuration file"""
        filename, _ = QFileDialog.getOpenFileName(self, "Open Grid File", "", 
                                                "Text Files (*.txt *.dat);;All Files (*)")
        
        if filename:
            try:
                data = pd.read_csv(filename, delimiter=r'\s+', header=None,
                                 names=['lat', 'lon', 'depth', 'radius', 'max_station_dist', 'min_pick_count'])
                
                # Generate unique name if file is already loaded
                basename = os.path.basename(filename)
                name = basename
                counter = 1
                while name in self.grid_data:
                    name = f"{basename}_{counter}"
                    counter += 1
                
                # Store data with a unique color
                color_idx = len(self.grid_data) % len(self.colors)
                
                self.grid_data[name] = {
                    'data': data,
                    'color': self.colors[color_idx],
                    'visible': True,
                    'filename': filename  # Store original filename
                }
                
                # Add layer control
                layer_control = LayerControl(name, self.colors[color_idx])
                layer_control.checkbox.stateChanged.connect(self.update_map)
                self.grid_layers_layout.addWidget(layer_control)
                
                self.update_map()
                
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Failed to load grid file: {str(e)}")

    def load_station_file(self):
        """Load a station file"""
        filename, _ = QFileDialog.getOpenFileName(self, "Open Station File", "", 
                                                "Text Files (*.txt *.dat);;All Files (*)")
        
        if filename:
            try:
                # Read the first line to get the header
                with open(filename, 'r') as f:
                    header_line = f.readline().strip()
                    if header_line.startswith('#'):
                        header_line = header_line[1:]  # Remove the '#'
                    columns = [col.strip() for col in header_line.split('|')]
                
                # Read the data with the extracted column names
                data = pd.read_csv(filename, delimiter='|', names=columns, 
                                 skiprows=1, skip_blank_lines=True)
                
                # Store data with a unique color
                color_idx = len(self.station_data) % len(self.colors)
                basename = os.path.basename(filename)
                
                self.station_data[basename] = {
                    'data': data,
                    'color': self.colors[color_idx],
                    'visible': True
                }
                
                # Add layer control
                layer_control = LayerControl(basename, self.colors[color_idx])
                layer_control.checkbox.stateChanged.connect(self.update_map)
                self.station_layers_layout.addWidget(layer_control)
                
                self.update_map()
                
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Failed to load station file: {str(e)}")
    
    def get_visible_layers(self):
        """Get the currently visible grid and station layers"""
        grid_layers = {}
        station_layers = {}
        
        # Check grid layers
        for i in range(self.grid_layers_layout.count()):
            widget = self.grid_layers_layout.itemAt(i).widget()
            if isinstance(widget, LayerControl):
                filename = widget.checkbox.text()
                if filename in self.grid_data and widget.checkbox.isChecked():
                    grid_layers[filename] = self.grid_data[filename]
        
        # Check station layers
        for i in range(self.station_layers_layout.count()):
            widget = self.station_layers_layout.itemAt(i).widget()
            if isinstance(widget, LayerControl):
                filename = widget.checkbox.text()
                if filename in self.station_data and widget.checkbox.isChecked():
                    station_layers[filename] = self.station_data[filename]
        
        return grid_layers, station_layers
    
    def filter_data(self, data):
        """Apply depth and radius filters to the grid data"""
        depth_filter = self.depth_filter.currentText()
        radius_filter = self.radius_filter.currentText()
        
        # Apply depth filter
        if depth_filter == "0-100 km":
            data = data[data['depth'] <= 100]
        elif depth_filter == "100-300 km":
            data = data[(data['depth'] > 100) & (data['depth'] <= 300)]
        elif depth_filter == ">300 km":
            data = data[data['depth'] > 300]
            
        # Apply radius filter
        if radius_filter == "<5°":
            data = data[data['radius'] < 5]
        elif radius_filter == "5-10°":
            data = data[(data['radius'] >= 5) & (data['radius'] <= 10)]
        elif radius_filter == ">10°":
            data = data[data['radius'] > 10]
            
        return data
    
    def update_map(self):
        """Update the map with current grid and station data"""
        # Get visible layers
        grid_layers, station_layers = self.get_visible_layers()
        
        if not (grid_layers or station_layers):
            return
            
        # Create new map
        m = folium.Map(location=[0, 0], zoom_start=2)
        
        # Add grid points
        for filename, grid_info in grid_layers.items():
            data = self.filter_data(grid_info['data'])
            color = grid_info['color']
            
            for _, row in data.iterrows():
                # Create circle marker
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=5,
                    color=color,
                    fill=True,
                    popup=f"Depth: {row['depth']} km<br>"
                          f"Radius: {row['radius']}°<br>"
                          f"Max Station Dist: {row['max_station_dist']}°<br>"
                          f"Min Pick Count: {row['min_pick_count']}"
                ).add_to(m)
                
                # Add circle showing the sensitivity radius
                folium.Circle(
                    location=[row['lat'], row['lon']],
                    radius=row['radius'] * 111000,  # Convert degrees to meters (approx)
                    color=color,
                    weight=1,
                    fill=False,
                    opacity=0.3
                ).add_to(m)
        
        # Add stations
        for filename, station_info in station_layers.items():
            data = station_info['data']
            color = station_info['color']
            
            for _, row in data.iterrows():
                # Create station marker (triangle)
                folium.RegularPolygonMarker(
                    location=[row['Latitude'], row['Longitude']],
                    number_of_sides=3,
                    radius=6,
                    color=color,
                    fill=True,
                    popup=f"Network: {row['Network']}<br>"
                          f"Station: {row['Station']}<br>"
                          f"Site: {row['SiteName']}<br>"
                          f"Elevation: {row['Elevation']} m<br>"
                          f"Start: {row['StartTime']}<br>"
                          f"End: {row['EndTime'] if row['EndTime'] else 'Present'}"
                ).add_to(m)
        
        # Save and display updated map
        temp_file = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        m.save(temp_file.name)
        self.web_view.setUrl(QUrl.fromLocalFile(temp_file.name))

def main():
    app = QApplication(sys.argv)
    viewer = GridMapViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
