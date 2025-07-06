# 🌪️ Enhanced Tropical Cloud Cluster System v2.0

## Overview

This enhanced system provides comprehensive tropical cloud cluster detection, tracking, and analysis with:

- **24 tracks** with persistent cluster evolution
- **24 timesteps** of realistic cluster data per track
- **Real-time track monitoring** with comprehensive status tables
- **Advanced tracking** with multiple cluster properties and quality metrics
- **Interactive 3D visualization** using Plotly
- **Real-time monitoring** with alert systems and status indicators
- **Multi-format report generation** (PDF, CSV, JSON, NetCDF)
- **Modern Streamlit dashboard** with 6 comprehensive tabs

## 🚀 Quick Start

### 1. Run the Complete System

```bash
python launch_enhanced_system.py
```

This will:
- Generate 24 tracks with 24 timesteps each (576 total clusters)
- Create individual and combined maps
- Launch the dashboard on port 8518

### 2. Run Components Separately

#### Detection Only
```bash
python detection.py
```

#### Dashboard Only
```bash
streamlit run app_enhanced_v2.py --server.port 8518
```

## 📊 System Components

### 1. Enhanced Detection (`detection.py`)

**Features:**
- **24 tracks** (track_00 to track_23) with persistent evolution
- **24 timesteps** with 1-hour intervals
- Realistic cluster evolution and tracking with quality metrics
- Advanced properties: windspeed, precipitation, development stages
- Individual timestep maps + combined overview map
- **Quality metrics**: edge confidence, data coverage, consistency index
- **Track assignment** based on spatial proximity and persistence

**Output:**
- `outputs/cloud_clusters.csv` - Main data file (576 clusters)
- `outputs/cloud_clusters.nc` - NetCDF format
- `outputs/tcc_detection_t{0-23}.png` - Individual timestep maps
- `outputs/tcc_detection.png` - Combined overview map

### 2. Enhanced Dashboard (`app_enhanced_v2.py`)

**Features:**
- **6 Main Tabs:**
  - 🌎 World View - Global cluster map with real-time track table
  - 📊 Real-time Monitoring - Comprehensive track monitoring dashboard
  - 📈 Cluster Analytics - Statistical analysis and visualizations
  - 🌀 Track Explorer - Individual track analysis and forecasting
  - 📊 Reports - Multi-format report generation
  - ⚙️ 3D Analysis - 3D cluster evolution visualization

**Interactive Elements:**
- Track selection dropdown
- Time range slider
- Alert threshold configuration
- Real-time metrics display
- Interactive maps and charts
- **Real-time track status table** with filtering and search

## 📈 Data Structure

The enhanced system generates clusters with these properties:

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
| `edge_confidence` | Boundary detection confidence | 0.8-0.95 |
| `data_coverage` | Data coverage percentage | 0.9-1.0 |
| `consistency_index` | Temporal consistency | 0.75-0.95 |
| `signal_to_noise` | Signal-to-noise ratio | 0.7-0.95 |
| `temporal_stability` | Temporal stability | 0.8-0.98 |

## 🎨 Dashboard Features

### World View Tab
- **Global cluster map** with interactive markers
- **Real-time track status table** below the map
- **Status indicators**: 🟢 Active, 🟡 Weak, 🔴 Inactive
- **Alert levels**: 🔴 High, 🟡 Medium, 🟢 Low
- **Quick metrics**: Active Tracks, High Alert, Mature Tracks
- **Clean interface** without attribution text

### Real-time Monitoring Tab
- **Comprehensive track monitoring** with 24 tracks
- **Advanced filtering** by status, alert level, and development stage
- **Search functionality** by track ID
- **Real-time alerts** for high-risk tracks
- **Export capabilities** (CSV download)
- **Trend indicators**: 📈 Intensifying, 📉 Weakening, ➡️ Stable

### Cluster Analytics Tab
- **Intensity distribution** histograms
- **Precipitation vs. Size** scatter plots
- **Development timeline** area charts
- **Quality metrics** box plots

### Track Explorer Tab
- **Individual track analysis** with detailed metrics
- **5-step forecasting** for wind speed
- **Track path visualization** on interactive maps
- **Intensity evolution** time series

### Reports Tab
- **Multi-format reports**: PDF, CSV, JSON, NetCDF
- **Customizable sections**: Summary, Alerts, Track Analysis, Quality Metrics
- **Professional formatting** with timestamps
- **Downloadable reports** with comprehensive data

### 3D Analysis Tab
- **3D cluster evolution** visualization
- **Time-based clustering** with color coding
- **Interactive 3D plots** with hover information
- **Enhanced colorbar** with proper positioning

## 🔧 Configuration

### Detection Parameters
```python
NUM_TRACKS = 24  # Number of tracks (track_00 to track_23)
NUM_TIMESTEPS = 24  # Number of timesteps per track
TIME_INTERVAL = 60  # 60 minutes between timesteps
```

### Dashboard Configuration
- **Port**: Configurable via `--server.port` argument
- **Data Sources**: Automatically detects CSV/NetCDF files
- **Alert Thresholds**: Adjustable via sidebar
- **Visualization Options**: Multiple chart types and layouts

## 📁 File Structure

```
tropical_cloud_project/
├── detection.py                   # Enhanced detection system
├── app_enhanced_v2.py            # Enhanced dashboard v2.0
├── launch_enhanced_system.py     # System launcher
├── outputs/                      # Generated data and maps
│   ├── cloud_clusters.csv        # Main data file (576 clusters)
│   ├── cloud_clusters.nc         # NetCDF format
│   ├── tcc_detection_t{0-23}.png # Individual timestep maps
│   └── tcc_detection.png         # Combined overview map
└── ENHANCED_SYSTEM_README.md     # This file
```

## 🚨 Alert System

The system includes a comprehensive alert system:

- **High-intensity clusters** (configurable threshold)
- **Real-time monitoring** with visual indicators
- **Status-based alerts**: Active, Weak, Inactive
- **Alert levels**: High, Medium, Low
- **Trend monitoring**: Intensifying, Weakening, Stable
- **Alert tables** with detailed cluster information
- **Multi-format reports** with alert summaries

## 📊 Quality Metrics

The enhanced system tracks multiple quality indicators:

- **Quality Score**: Overall data quality (0.85-0.99)
- **Data Coverage**: Percentage of valid data (0.9-1.0)
- **Edge Confidence**: Boundary detection confidence (0.8-0.95)
- **Consistency Index**: Temporal consistency (0.75-0.95)
- **Signal-to-Noise**: Signal-to-noise ratio (0.7-0.95)
- **Temporal Stability**: Temporal stability (0.8-0.98)

## 🔄 Workflow

1. **Data Generation**: Run detection to create 24 tracks with 24 timesteps each
2. **Visualization**: Generate maps for each timestep and combined view
3. **Real-time Monitoring**: Use dashboard for interactive exploration and monitoring
4. **Alert Management**: Monitor high-risk tracks and set up alerts
5. **Reporting**: Generate multi-format reports with custom sections
6. **Analysis**: Use 3D visualization for advanced analysis

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
   - Verify CSV file contains 576 clusters (24 tracks × 24 timesteps)

4. **Plotly Errors**
   - Updated colorbar configuration to remove deprecated properties
   - Fixed `titleside` property issues
   - Enhanced 3D visualization with proper positioning

## 🆕 Recent Updates (v2.0)

### Major Improvements:
- **24 tracks** instead of 8 (576 total clusters)
- **Real-time track monitoring table** in World View tab
- **Enhanced status indicators** with emojis and colors
- **Advanced filtering** and search capabilities
- **Multi-format report generation**
- **Improved 3D visualization** with proper colorbar
- **Clean interface** without attribution text
- **Comprehensive alert system** with trend monitoring

### Technical Fixes:
- Fixed Plotly colorbar configuration errors
- Resolved pandas FutureWarning for groupby operations
- Enhanced coordinate validation and cleaning
- Improved data mapping and column handling
- Fixed tab structure and navigation

## 🎯 Use Cases

1. **Research**: Academic tropical meteorology research
2. **Operational**: Real-time monitoring and forecasting
3. **Education**: Teaching tropical meteorology concepts
4. **Analysis**: Historical cluster pattern analysis

## 🔮 Future Enhancements

- **Real satellite data** integration
- **Machine learning** forecasting models
- **Multi-satellite** data fusion
- **Advanced visualization** options
- **API endpoints** for external access

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the data structure documentation
3. Verify all dependencies are installed
4. Check file permissions and paths

---

**🌪️ Enhanced Tropical Cloud Cluster System v2.0**
*Advanced monitoring and analysis for tropical meteorology* 