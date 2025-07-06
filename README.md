# 🌪️ Tropical Cloud Cluster Detection & Monitoring System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive AI/ML-based system for detecting and tracking Tropical Cloud Clusters (TCCs) using INSAT-3D satellite data. This system provides real-time monitoring, advanced analytics, and interactive visualization capabilities.

## 🚀 Features

- **24 Tracks with Persistent Evolution**: Monitor 24 distinct tropical cloud cluster tracks
- **Real-time Monitoring**: Live tracking with status indicators and alert systems
- **Interactive Dashboard**: 6-tab Streamlit interface with comprehensive analytics
- **Multi-format Reports**: Generate PDF, CSV, JSON, and NetCDF reports
- **3D Visualization**: Advanced 3D cluster evolution analysis
- **Quality Metrics**: Comprehensive data quality assessment
- **Alert System**: Real-time alerts for high-risk clusters

## 📊 System Overview

### Detection Capabilities
- **576 Total Clusters**: 24 tracks × 24 timesteps
- **Quality Metrics**: Edge confidence, data coverage, consistency index
- **Development Stages**: Formation, Development, Mature, Decay
- **Spatial Tracking**: Persistent track assignment based on proximity

### Dashboard Features
- **🌎 World View**: Global cluster map with real-time track table
- **📊 Real-time Monitoring**: Comprehensive track monitoring dashboard
- **📈 Cluster Analytics**: Statistical analysis and visualizations
- **🌀 Track Explorer**: Individual track analysis and forecasting
- **📊 Reports**: Multi-format report generation
- **⚙️ 3D Analysis**: 3D cluster evolution visualization

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/tropical-cloud-cluster-system.git
   cd tropical-cloud-cluster-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the system**
   ```bash
   python launch_enhanced_system.py
   ```

## 📁 Project Structure

```
tropical_cloud_project/
├── detection.py                   # Enhanced detection system
├── app_enhanced_v2.py            # Enhanced dashboard v2.0
├── launch_enhanced_system.py     # System launcher
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── ENHANCED_SYSTEM_README.md     # Detailed system documentation
├── outputs/                      # Generated data and maps
│   ├── cloud_clusters.csv        # Main data file (576 clusters)
│   ├── cloud_clusters.nc         # NetCDF format
│   ├── tcc_detection_t{0-23}.png # Individual timestep maps
│   └── tcc_detection.png         # Combined overview map
└── .gitignore                    # Git ignore file
```

## 🎯 Quick Start

### Option 1: Complete System
```bash
python launch_enhanced_system.py
```

### Option 2: Individual Components
```bash
# Run detection only
python detection.py

# Run dashboard only
streamlit run app_enhanced_v2.py --server.port 8518
```

## 📈 Data Structure

The system generates clusters with comprehensive properties:

| Property | Description | Range |
|----------|-------------|-------|
| `track_id` | Unique track identifier | track_00 to track_23 |
| `timestep` | Time step (0-23) | 0-23 |
| `center_lat/lon` | Cluster center coordinates | lat: 10-19°, lon: 70-89° |
| `area_km2` | Cluster area in km² | 2k-50k km² |
| `cloud_top_height_km` | Cloud top height | 10-16 km |
| `convective_intensity` | Convective intensity | 0.5-2.0 |
| `precipitation` | Precipitation rate | 0.1-50 mm/h |
| `quality_score` | Data quality score | 0.85-0.99 |
| `development_stage` | Cluster development stage | Formation/Development/Mature/Decay |

## 🎨 Dashboard Features

### Real-time Monitoring
- **Status Indicators**: 🟢 Active, 🟡 Weak, 🔴 Inactive
- **Alert Levels**: 🔴 High, 🟡 Medium, 🟢 Low
- **Trend Indicators**: 📈 Intensifying, 📉 Weakening, ➡️ Stable
- **Advanced Filtering**: By status, alert level, development stage
- **Search Functionality**: Find tracks by ID

### Analytics & Visualization
- **Interactive Maps**: Global cluster distribution
- **Statistical Analysis**: Intensity distribution, precipitation vs. size
- **Time Series**: Development timeline, intensity evolution
- **3D Visualization**: Cluster evolution in 3D space
- **Quality Assessment**: Comprehensive quality metrics

### Reporting
- **Multi-format Export**: PDF, CSV, JSON, NetCDF
- **Customizable Sections**: Summary, Alerts, Track Analysis
- **Professional Formatting**: Timestamped reports
- **Downloadable Content**: All data and visualizations

## 🔧 Configuration

### Detection Parameters
```python
NUM_TRACKS = 24          # Number of tracks
NUM_TIMESTEPS = 24       # Number of timesteps per track
TIME_INTERVAL = 60       # Minutes between timesteps
```

### Dashboard Configuration
- **Port**: Configurable via `--server.port` argument
- **Alert Thresholds**: Adjustable via sidebar
- **Data Sources**: Automatic CSV/NetCDF detection

## 🚨 Alert System

The system provides comprehensive alerting:

- **High-intensity Clusters**: Configurable threshold-based alerts
- **Status-based Monitoring**: Active, Weak, Inactive status tracking
- **Trend Analysis**: Intensifying, Weakening, Stable trend detection
- **Real-time Updates**: Live status and alert updates
- **Export Capabilities**: Alert data export in multiple formats

## 📊 Quality Metrics

Comprehensive quality assessment including:

- **Quality Score**: Overall data quality (0.85-0.99)
- **Data Coverage**: Percentage of valid data (0.9-1.0)
- **Edge Confidence**: Boundary detection confidence (0.8-0.95)
- **Consistency Index**: Temporal consistency (0.75-0.95)
- **Signal-to-Noise**: Signal-to-noise ratio (0.7-0.95)
- **Temporal Stability**: Temporal stability (0.8-0.98)

## 🛠️ Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install streamlit pandas numpy plotly pydeck xarray matplotlib seaborn reportlab
   ```

2. **Port Already in Use**
   ```bash
   streamlit run app_enhanced_v2.py --server.port 8524
   ```

3. **Data Not Found**
   - Ensure detection has been run first
   - Check file paths in `outputs/` directory
   - Verify CSV file contains 576 clusters

4. **Plotly Errors**
   - Updated colorbar configuration
   - Fixed deprecated property issues
   - Enhanced 3D visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- INSAT-3D satellite data processing
- Streamlit for interactive dashboard framework
- Plotly for advanced visualizations
- Scientific community for tropical meteorology research

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the [ENHANCED_SYSTEM_README.md](ENHANCED_SYSTEM_README.md) for detailed documentation

---

**Made with ❤️ for tropical meteorology research and monitoring** 