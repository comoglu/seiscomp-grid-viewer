#!/usr/bin/python3

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QFileDialog,
                           QLabel, QComboBox, QScrollArea, QCheckBox,
                           QGroupBox, QFrame, QTabWidget, QTextEdit,
                           QSpinBox, QDoubleSpinBox, QMessageBox,
                           QSplitter, QStyle)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt
import folium
import pandas as pd
import tempfile
from datetime import datetime
import folium.plugins

class ColorSchemes:
    """Color schemes for different data types"""
    GRID_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    STATION_COLORS = ['#33a02c', '#1f78b4', '#e31a1c', '#6a3d9a', '#b15928', '#ffff99']
    
    @staticmethod
    def get_grid_color(index):
        return ColorSchemes.GRID_COLORS[index % len(ColorSchemes.GRID_COLORS)]
    
    @staticmethod
    def get_station_color(index):
        return ColorSchemes.STATION_COLORS[index % len(ColorSchemes.STATION_COLORS)]

class LayerControl(QWidget):
    """Widget for controlling layer visibility"""
    def __init__(self, name, color, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.checkbox = QCheckBox(name)
        self.checkbox.setChecked(True)
        layout.addWidget(self.checkbox)
        
        # Color indicator
        color_indicator = QFrame()
        color_indicator.setStyleSheet(f"background-color: {color};")
        color_indicator.setFixedSize(16, 16)
        layout.addWidget(color_indicator)
        
        # Info button
        info_button = QPushButton()
        info_button.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        info_button.setFixedSize(16, 16)
        info_button.clicked.connect(lambda: self.show_info(name))
        layout.addWidget(info_button)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def show_info(self, name):
        """Show information about the layer"""
        info = self.parent().parent().parent().get_layer_info(name)
        QMessageBox.information(self, f"Layer Info - {name}", info)

class StationConfigDialog(QMainWindow):
    """Dialog for editing station configuration"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Station Configuration")
        self.setGeometry(200, 200, 600, 400)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Config editor
        self.editor = QTextEdit()
        self.editor.setPlaceholderText(
            "Enter station configuration here...\n"
            "Format: Network Station UsageFlag MaxDistance\n"
            "Example:\n"
            "* * 1 90\n"
            "GE * 1 180\n"
            "GE HLG 1 10\n"
            "TE RGN 0 10"
        )
        layout.addWidget(self.editor)
        
        # Buttons
        button_layout = QHBoxLayout()
        load_button = QPushButton("Load Config")
        save_button = QPushButton("Save Config")
        apply_button = QPushButton("Apply")
        
        load_button.clicked.connect(self.load_config)
        save_button.clicked.connect(self.save_config)
        apply_button.clicked.connect(self.apply_config)
        
        button_layout.addWidget(load_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(apply_button)
        layout.addLayout(button_layout)

    def load_config(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Station Config", "", 
            "Config Files (*.conf);;All Files (*)"
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    self.editor.setText(f.read())
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load config: {str(e)}")

    def save_config(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Station Config", "", 
            "Config Files (*.conf);;All Files (*)"
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.editor.toPlainText())
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save config: {str(e)}")

    def apply_config(self):
        config_text = self.editor.toPlainText()
        self.parent().apply_station_config(config_text)
        self.close()

class MapSettingsWidget(QWidget):
    """Widget for map display settings"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Map Type
        map_group = QGroupBox("Map Settings")
        map_layout = QVBoxLayout()
        
        self.map_type = QComboBox()
        self.map_type.addItems([
            "OpenStreetMap",
            "Stamen Terrain",
            "CartoDB positron",
            "CartoDB dark_matter"
        ])
        map_layout.addWidget(QLabel("Map Type:"))
        map_layout.addWidget(self.map_type)
        
        # Circle opacity
        self.circle_opacity = QDoubleSpinBox()
        self.circle_opacity.setRange(0.1, 1.0)
        self.circle_opacity.setSingleStep(0.1)
        self.circle_opacity.setValue(0.3)
        map_layout.addWidget(QLabel("Circle Opacity:"))
        map_layout.addWidget(self.circle_opacity)
        
        # Station marker size
        self.station_size = QSpinBox()
        self.station_size.setRange(4, 12)
        self.station_size.setValue(6)
        map_layout.addWidget(QLabel("Station Marker Size:"))
        map_layout.addWidget(self.station_size)
        
        map_group.setLayout(map_layout)
        layout.addWidget(map_group)
        
        # Add statistics group
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        self.stats_label = QLabel("No data loaded")
        stats_layout.addWidget(self.stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()

class GridMapViewer(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grid Point Map Viewer")
        self.setGeometry(100, 100, 1400, 800)
        
        # Store loaded data
        self.grid_data = {}
        self.station_data = {}
        self.station_config = {}
        
        self.init_ui()
        
    def init_ui(self):
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(main_splitter)
        
        # Left panel with tabs
        left_panel = QTabWidget()
        left_panel.setMaximumWidth(400)
        
        # Layers tab
        layers_tab = QWidget()
        layers_layout = QVBoxLayout(layers_tab)
        
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
        layers_layout.addWidget(grid_group)
        
        # Station Files Section
        station_group = QGroupBox("Station Files")
        station_layout = QVBoxLayout()
        
        station_buttons = QHBoxLayout()
        load_station_button = QPushButton("Load Station File")
        load_station_button.clicked.connect(self.load_station_file)
        config_station_button = QPushButton("Station Config")
        config_station_button.clicked.connect(self.open_station_config)
        
        station_buttons.addWidget(load_station_button)
        station_buttons.addWidget(config_station_button)
        station_layout.addLayout(station_buttons)
        
        self.station_layers = QWidget()
        self.station_layers_layout = QVBoxLayout(self.station_layers)
        station_layout.addWidget(self.station_layers)
        
        station_group.setLayout(station_layout)
        layers_layout.addWidget(station_group)
        
        # Filters Section
        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout()
        
        self.depth_filter = QComboBox()
        self.depth_filter.addItems(["All Depths", "0-100 km", "100-300 km", ">300 km"])
        self.depth_filter.currentTextChanged.connect(self.update_map)
        filter_layout.addWidget(QLabel("Depth Filter:"))
        filter_layout.addWidget(self.depth_filter)
        
        self.radius_filter = QComboBox()
        self.radius_filter.addItems(["All Radii", "<5°", "5-10°", ">10°"])
        self.radius_filter.currentTextChanged.connect(self.update_map)
        filter_layout.addWidget(QLabel("Radius Filter:"))
        filter_layout.addWidget(self.radius_filter)
        
        filter_group.setLayout(filter_layout)
        layers_layout.addWidget(filter_group)
        
        layers_layout.addStretch()
        
        # Settings tab
        settings_tab = MapSettingsWidget()
        settings_tab.map_type.currentTextChanged.connect(self.update_map)
        settings_tab.circle_opacity.valueChanged.connect(self.update_map)
        settings_tab.station_size.valueChanged.connect(self.update_map)
        
        # Add tabs
        left_panel.addTab(layers_tab, "Layers")
        left_panel.addTab(settings_tab, "Settings")
        
        # Add panels to splitter
        main_splitter.addWidget(left_panel)
        
        # Create map view
        self.web_view = QWebEngineView()
        main_splitter.addWidget(self.web_view)
        
        # Set stretch factor
        main_splitter.setStretchFactor(1, 1)
        
        # Initialize empty map
        self.create_empty_map()
    
    def get_layer_info(self, name):
        """Get information about a layer"""
        if name in self.grid_data:
            data = self.grid_data[name]
            return (f"Grid File: {data['filename']}\n"
                   f"Points: {len(data['data'])}\n"
                   f"Depth range: {data['data']['depth'].min():.1f} - {data['data']['depth'].max():.1f} km\n"
                   f"Radius range: {data['data']['radius'].min():.1f} - {data['data']['radius'].max():.1f}°")
        elif name in self.station_data:
            data = self.station_data[name]
            return (f"Station File: {data['filename']}\n"
                   f"Stations: {len(data['data'])}\n"
                   f"Networks: {', '.join(data['data']['Network'].unique())}")
        return "No information available"

    def open_station_config(self):
        dialog = StationConfigDialog(self)
        dialog.show()
        
    def apply_station_config(self, config_text):
        """Apply station configuration from text"""
        self.station_config = {}
        for line in config_text.strip().split('\n'):
            if line.strip() and not line.startswith('#'):
                try:
                    network, station, usage, max_dist = line.strip().split()
                    self.station_config[(network, station)] = {
                        'usage': int(usage),
                        'max_distance': float(max_dist)
                    }
                except Exception as e:
                    QMessageBox.warning(self, "Warning", f"Invalid config line: {line}")
        
        self.update_map()

    def get_station_config(self, network, station):
        """Get station configuration, handling wildcards"""
        # Try exact match
        if (network, station) in self.station_config:
            return self.station_config[(network, station)]
        
        # Try network wildcard
        if ('*', station) in self.station_config:
            return self.station_config[('*', station)]
        
        # Try station wildcard
        if (network, '*') in self.station_config:
            return self.station_config[(network, '*')]
        
        # Try full wildcard
        if ('*', '*') in self.station_config:
            return self.station_config[('*', '*')]
        
        # Default values if no config found
        return {'usage': 1, 'max_distance': 180}

    def create_empty_map(self):
        """Create an empty map centered at (0, 0)"""
        m = folium.Map(location=[0, 0], zoom_start=2)
        
        temp_file = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        m.save(temp_file.name)
        self.web_view.setUrl(QUrl.fromLocalFile(temp_file.name))
        
    def load_grid_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Grid File", "", 
            "Text Files (*.txt *.dat);;All Files (*)"
        )
        
        if filename:
            try:
                data = pd.read_csv(
                    filename, 
                    delimiter=r'\s+', 
                    header=None,
                    names=['lat', 'lon', 'depth', 'radius', 'max_station_dist', 'min_pick_count']
                )
                
                # Generate unique name
                basename = os.path.basename(filename)
                name = basename
                counter = 1
                while name in self.grid_data:
                    name = f"{basename}_{counter}"
                    counter += 1
                
                # Store data with a unique color
                color_idx = len(self.grid_data)
                
                self.grid_data[name] = {
                    'data': data,
                    'color': ColorSchemes.get_grid_color(color_idx),
                    'visible': True,
                    'filename': filename
                }
                
                # Add layer control
                layer_control = LayerControl(name, ColorSchemes.get_grid_color(color_idx))
                layer_control.checkbox.stateChanged.connect(self.update_map)
                self.grid_layers_layout.addWidget(layer_control)
                
                self.update_map()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load grid file: {str(e)}")

    def load_station_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Station File", "", 
            "Text Files (*.txt *.dat);;All Files (*)"
        )
        
        if filename:
            try:
                # Read the first line to get the header
                with open(filename, 'r') as f:
                    header_line = f.readline().strip()
                    if header_line.startswith('#'):
                        header_line = header_line[1:]
                    columns = [col.strip() for col in header_line.split('|')]
                
                # Read the data with the extracted column names
                data = pd.read_csv(filename, delimiter='|', names=columns, 
                                 skiprows=1, skip_blank_lines=True)
                
                # Generate unique name
                basename = os.path.basename(filename)
                name = basename
                counter = 1
                while name in self.station_data:
                    name = f"{basename}_{counter}"
                    counter += 1
                
                # Store data with a unique color
                color_idx = len(self.station_data)
                
                self.station_data[name] = {
                    'data': data,
                    'color': ColorSchemes.get_station_color(color_idx),
                    'visible': True,
                    'filename': filename
                }
                
                # Add layer control
                layer_control = LayerControl(name, ColorSchemes.get_station_color(color_idx))
                layer_control.checkbox.stateChanged.connect(self.update_map)
                self.station_layers_layout.addWidget(layer_control)
                
                self.update_map()
                
            except Exception as e:
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

    def create_map_legend(self, m):
        """Create a legend for the map"""
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; 
                    border:2px solid grey; z-index:9999; 
                    background-color:white;
                    padding:10px;
                    font-size:14px;
                    border-radius: 5px;">
        <p style="margin-bottom:5px"><strong>Legend</strong></p>
        """
        
        # Add grid points to legend
        if self.grid_data:
            legend_html += "<p style='margin:0'><strong>Grid Points:</strong></p>"
            for name, info in self.grid_data.items():
                if info['visible']:
                    color = info['color']
                    legend_html += f"""
                    <p style='margin:0'>
                        <span style='color:{color};'>●</span> {name}
                    </p>
                    """
        
        # Add stations to legend
        if self.station_data:
            legend_html += "<p style='margin:0'><strong>Stations:</strong></p>"
            for name, info in self.station_data.items():
                if info['visible']:
                    color = info['color']
                    legend_html += f"""
                    <p style='margin:0'>
                        <span style='color:{color};'>▲</span> {name}
                    </p>
                    """
        
        legend_html += "</div>"
        
        # Add the legend to the map
        m.get_root().html.add_child(folium.Element(legend_html))

    def update_statistics(self):
        """Update statistics in the settings panel"""
        settings_tab = self.findChild(MapSettingsWidget)
        if settings_tab:
            grid_layers, station_layers = self.get_visible_layers()
            
            stats_text = []
            if grid_layers:
                total_grid_points = sum(len(self.filter_data(info['data'])) 
                                      for info in grid_layers.values())
                stats_text.append(f"Grid Points: {total_grid_points}")
            
            if station_layers:
                total_stations = sum(len(info['data']) for info in station_layers.values())
                enabled_stations = sum(
                    sum(1 for _, row in info['data'].iterrows()
                        if self.get_station_config(row['Network'], row['Station'])['usage'] == 1)
                    for info in station_layers.values()
                )
                stats_text.append(f"Stations: {enabled_stations} enabled / {total_stations} total")
            
            if not stats_text:
                stats_text = ["No data loaded"]
            
            settings_tab.stats_label.setText("\n".join(stats_text))
    
    def update_map(self):
        """Update the map with current grid and station data"""
        # Get visible layers
        grid_layers, station_layers = self.get_visible_layers()
        
        if not (grid_layers or station_layers):
            return
            
        # Get map settings from the settings tab
        settings_tab = self.findChild(MapSettingsWidget)
        map_type = settings_tab.map_type.currentText()
        circle_opacity = settings_tab.circle_opacity.value()
        station_size = settings_tab.station_size.value()
        
        # Create new map with selected tile layer
        if map_type == "Stamen Terrain":
            m = folium.Map(location=[0, 0], zoom_start=2, tiles='Stamen Terrain')
        elif map_type == "CartoDB positron":
            m = folium.Map(location=[0, 0], zoom_start=2, tiles='CartoDB positron')
        elif map_type == "CartoDB dark_matter":
            m = folium.Map(location=[0, 0], zoom_start=2, tiles='CartoDB dark_matter')
        else:
            m = folium.Map(location=[0, 0], zoom_start=2)  # Default OpenStreetMap
        
        # Add grid points
        for filename, grid_info in grid_layers.items():
            data = self.filter_data(grid_info['data'])
            color = grid_info['color']
            
            # Create a feature group for this grid file
            grid_group = folium.FeatureGroup(name=f"Grid: {filename}")
            
            for _, row in data.iterrows():
                # Create circle marker for the grid point
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=5,
                    color=color,
                    fill=True,
                    popup=f"<strong>Grid Point</strong><br>"
                          f"Depth: {row['depth']} km<br>"
                          f"Radius: {row['radius']}°<br>"
                          f"Max Station Dist: {row['max_station_dist']}°<br>"
                          f"Min Pick Count: {row['min_pick_count']}"
                ).add_to(grid_group)
                
                # Add circle showing the sensitivity radius
                folium.Circle(
                    location=[row['lat'], row['lon']],
                    radius=row['radius'] * 111000,  # Convert degrees to meters (approx)
                    color=color,
                    weight=1,
                    fill=False,
                    opacity=circle_opacity
                ).add_to(grid_group)
            
            grid_group.add_to(m)
        
        # Add stations
        for filename, station_info in station_layers.items():
            data = station_info['data']
            color = station_info['color']
            
            # Create a feature group for this station file
            station_group = folium.FeatureGroup(name=f"Stations: {filename}")
            
            for _, row in data.iterrows():
                # Get station configuration
                config = self.get_station_config(row['Network'], row['Station'])
                
                # Determine marker color based on usage flag
                marker_color = color if config['usage'] == 1 else 'gray'
                
                # Create station marker (triangle)
                folium.RegularPolygonMarker(
                    location=[row['Latitude'], row['Longitude']],
                    number_of_sides=3,
                    radius=station_size,
                    color=marker_color,
                    fill=True,
                    popup=f"<strong>Station Information</strong><br>"
                          f"Network: {row['Network']}<br>"
                          f"Station: {row['Station']}<br>"
                          f"Site: {row['SiteName']}<br>"
                          f"Elevation: {row['Elevation']} m<br>"
                          f"Start: {row['StartTime']}<br>"
                          f"End: {row['EndTime'] if row['EndTime'] else 'Present'}<br>"
                          f"Usage: {'Enabled' if config['usage'] == 1 else 'Disabled'}<br>"
                          f"Max Distance: {config['max_distance']}°"
                ).add_to(station_group)
                
                # Add circle showing maximum nucleation distance if enabled
                if config['usage'] == 1:
                    folium.Circle(
                        location=[row['Latitude'], row['Longitude']],
                        radius=config['max_distance'] * 111000,  # Convert degrees to meters
                        color=marker_color,
                        weight=1,
                        fill=False,
                        opacity=circle_opacity * 0.5,
                        dash_array='5, 10'
                    ).add_to(station_group)
            
            station_group.add_to(m)
        
        # Add map controls
        folium.plugins.MeasureControl(position='topright').add_to(m)
        folium.plugins.Fullscreen(position='topright').add_to(m)
        
        # Add legend
        self.create_map_legend(m)
        
        # Update statistics
        self.update_statistics()
        
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
