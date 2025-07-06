# 🌪️ Tropical Cloud Cluster Monitor Dashboard

A modern, interactive Streamlit dashboard for real-time monitoring and visualization of tropical cloud cluster detection and tracking systems using INSAT-3D satellite data.

## 🚀 Features

### 📊 Real-time Monitoring
- **Live System Metrics**: CPU, memory, and disk usage monitoring
- **Processing Status**: Real-time status of data processing pipeline
- **Auto-refresh**: Automatic data updates every 60 seconds

### 🗺️ Interactive Visualizations
- **Interactive Maps**: Folium-based maps with multiple tile layers
- **3D Track Visualization**: Plotly 3D plots showing cluster evolution
- **Timeline Analysis**: Gantt-style timeline of cluster events
- **Risk Assessment**: Color-coded risk levels and alerts

### 📈 Advanced Analytics
- **Cyclogenesis Risk Scoring**: AI-powered risk assessment
- **Cluster Metrics**: Comprehensive cluster property analysis
- **Track Analysis**: Multi-dimensional track visualization
- **Historical Comparison**: Track comparison and trend analysis

### 🎨 Modern UI/UX
- **Dark Theme**: Professional dark mode interface
- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Controls**: Real-time filtering and selection
- **Alert System**: Pulsing alerts for high-risk clusters

## 📋 Requirements

### Python Version
- Python 3.7 or higher

### Required Packages
```
streamlit>=1.28.0
streamlit-folium>=0.13.0
plotly>=5.15.0
pandas>=1.5.0
numpy>=1.21.0
pyyaml>=6.0
psutil>=5.9.0
folium>=0.14.0
matplotlib>=3.5.0
```

## 🛠️ Installation

### 1. Install Dependencies
```bash
# Install dashboard-specific requirements
pip install -r dashboard_requirements.txt

# Or install individually
pip install streamlit streamlit-folium plotly pandas numpy pyyaml psutil folium matplotlib
```

### 2. Verify Installation
```bash
# Run the test suite
python test_dashboard.py
```

### 3. Start the Dashboard
```bash
# Start the Streamlit dashboard
streamlit run dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## 📁 File Structure

```
tropical_cloud_project/
├── dashboard.py                 # Main dashboard application
├── dashboard_requirements.txt   # Dashboard dependencies
├── test_dashboard.py           # Dashboard test suite
├── DASHBOARD_README.md         # This file
├── data/
│   └── insat_real/            # Satellite data directory
├── outputs/
│   └── tracks/
│       └── final_tracks.csv   # Processed tracking data
└── real_data_config.yaml      # Configuration file
```

## 🎯 Usage Guide

### 1. Dashboard Overview
The dashboard is divided into several sections:

- **Header**: System title and status indicators
- **Metrics Row**: Real-time system performance metrics
- **Main Content**: Interactive maps and visualizations
- **Sidebar**: Controls and recent activity
- **3D Visualization**: Advanced track analysis
- **Data Table**: Detailed cluster metrics

### 2. Interactive Features

#### Map Controls
- **Track Selection**: Choose specific tracks to highlight
- **Layer Toggle**: Switch between different map layers
- **Zoom Controls**: Interactive zoom and pan
- **Fullscreen Mode**: Expand map for detailed viewing

#### 3D Visualization
- **Track Selection**: Choose tracks for 3D analysis
- **Interactive Rotation**: Rotate and zoom 3D plots
- **Risk Highlighting**: High-risk clusters marked in red
- **Property Mapping**: Color-coded by temperature and size

#### Data Filtering
- **Risk Level Filtering**: Filter by cyclogenesis risk
- **Time Range Selection**: Select specific time periods
- **Geographic Filtering**: Filter by latitude/longitude ranges

### 3. Alert System
The dashboard includes an intelligent alert system:

- **High Risk Alerts**: Pulsing red alerts for clusters with >70% risk
- **Medium Risk Warnings**: Orange warnings for 50-70% risk
- **System Alerts**: Processing status and error notifications

## 🔧 Configuration

### Configuration File (`real_data_config.yaml`)
```yaml
data:
  input_dir: 'data/insat_real'
  output_dir: 'outputs'
  thresholds:
    irbt: 220
    min_area: 50000
```

### Environment Variables
```bash
# Optional: Set custom paths
export DATA_DIR="path/to/satellite/data"
export OUTPUT_DIR="path/to/outputs"
```

## 📊 Data Requirements

### Input Data Format
The dashboard expects processed tracking data in CSV format with the following columns:

- `track_id`: Unique identifier for each track
- `timestamp`: Timestamp of observation
- `center_lat`, `center_lon`: Geographic coordinates
- `min_tb`, `mean_tb`: Brightness temperature values
- `area_km2`: Cluster area in square kilometers
- `cyclogenesis_risk`: Risk score (0-1)

### Data Sources
- **INSAT-3D Satellite Data**: NetCDF, HDF4, HDF5 formats
- **Processed Tracks**: CSV files from detection/tracking pipeline
- **Configuration**: YAML configuration files

## 🚨 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Solution: Install missing dependencies
pip install -r dashboard_requirements.txt
```

#### 2. Data Loading Errors
```bash
# Check file paths and permissions
ls -la outputs/tracks/final_tracks.csv
```

#### 3. Map Display Issues
```bash
# Ensure internet connection for tile layers
# Check if folium is properly installed
pip install folium streamlit-folium
```

#### 4. Performance Issues
```bash
# Monitor system resources
# Consider reducing data size or increasing refresh interval
```

### Debug Mode
```bash
# Run with debug information
streamlit run dashboard.py --logger.level debug
```

## 🔄 Integration with Processing Pipeline

The dashboard integrates seamlessly with the tropical cloud cluster processing pipeline:

1. **Data Processing**: Use `real_data_processor.py` to process satellite data
2. **Detection**: Run `detection.py` to identify cloud clusters
3. **Tracking**: Execute `tracking.py` to track clusters over time
4. **Visualization**: Launch dashboard to monitor results

### Pipeline Integration Example
```bash
# Process new data
python real_data_processor.py

# Start dashboard
streamlit run dashboard.py
```

## 📈 Performance Optimization

### For Large Datasets
- **Data Sampling**: Use data sampling for large track files
- **Caching**: Enable Streamlit caching for expensive computations
- **Lazy Loading**: Load data on demand rather than all at once

### For Real-time Monitoring
- **Reduced Refresh Rate**: Increase refresh interval for better performance
- **Selective Updates**: Update only changed components
- **Background Processing**: Process data in background threads

## 🔮 Future Enhancements

### Planned Features
- **Machine Learning Integration**: Advanced risk prediction models
- **Real-time Alerts**: Email/SMS notifications for high-risk events
- **Export Functionality**: PDF reports and data export
- **Multi-user Support**: User authentication and role-based access
- **Mobile App**: Native mobile application
- **API Integration**: REST API for external systems

### Customization Options
- **Theme Customization**: Custom color schemes and layouts
- **Widget Configuration**: Configurable dashboard widgets
- **Data Source Integration**: Support for additional data sources
- **Plugin System**: Extensible plugin architecture

## 📞 Support

### Getting Help
1. **Check Documentation**: Review this README and code comments
2. **Run Tests**: Execute `python test_dashboard.py` for diagnostics
3. **Check Logs**: Review Streamlit logs for error messages
4. **Verify Dependencies**: Ensure all required packages are installed

### Reporting Issues
When reporting issues, please include:
- Python version
- Operating system
- Error messages
- Steps to reproduce
- Expected vs actual behavior

## 📄 License

This dashboard is part of the Tropical Cloud Cluster Detection and Tracking System.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

**🌪️ Happy Monitoring!** 🚀 