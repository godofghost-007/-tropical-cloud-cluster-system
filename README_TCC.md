# Tropical Cloud Cluster (TCC) Detection and Tracking System

A comprehensive AI/ML-based solution for identifying and tracking Tropical Cloud Clusters (TCCs) using INSAT-3D IRBRT satellite data.

## 🌪️ Features

### Core Capabilities
- **Multi-format Data Support**: NetCDF, HDF4, HDF5 file formats
- **Advanced Detection**: Convective region identification using IRBT thresholds
- **Intelligent Tracking**: Multi-temporal cluster tracking with gap handling
- **Risk Assessment**: ML-based cyclogenesis risk prediction
- **Visualization**: Interactive maps and time series plots
- **Real-time Dashboard**: Streamlit-based monitoring interface

### AI/ML Components
- **Isolation Forest**: Anomaly detection for risk assessment
- **Feature Engineering**: Multi-dimensional cluster characterization
- **Predictive Modeling**: Cyclogenesis risk forecasting
- **Pattern Recognition**: Cluster development analysis

## 📋 Requirements

### System Requirements
- Python 3.8+
- 8GB+ RAM (for large datasets)
- 10GB+ disk space

### Dependencies
Install all required packages:
```bash
pip install -r requirements_tcc.txt
```

## 🚀 Quick Start

### 1. Setup
```bash
# Clone or download the project
cd tropical_cloud_project

# Install dependencies
pip install -r requirements_tcc.txt

# Create data directory
mkdir -p data/insat_real
```

### 2. Add Data
Place your INSAT-3D data files in `data/insat_real/`:
- Supported formats: `.nc`, `.nc4`, `.hdf`, `.h4`, `.h5`
- Files should contain brightness temperature data
- Timestamp extraction from filenames (e.g., `INSAT3D_IRBRT_202307061200.nc`)

### 3. Run Processing
```bash
# Process data files
python run_tcc_processor.py

# Or run directly
python tcc_processor.py
```

### 4. View Results
```bash
# Launch dashboard
streamlit run dashboard.py
```

## 📊 Configuration

Edit `tcc_config.yaml` to customize parameters:

```yaml
data:
  input_dir: "data/insat_real"
  output_dir: "outputs"
  region:
    lat_range: [5, 25]    # Bay of Bengal region
    lon_range: [60, 90]
  thresholds:
    irbt: 220            # Brightness temperature threshold
    min_area: 34800      # Minimum cluster area (km²)
    min_radius: 111      # Minimum radius (km)
  tracking:
    search_radii:        # Search radii for different time gaps
      3: 450            # 3-hour gap: 450km
      6: 550            # 6-hour gap: 550km
      9: 600            # 9-hour gap: 600km
      12: 650           # 12-hour gap: 650km
    max_gap: 12         # Maximum gap to continue tracking
    independence_dist: 1200  # Distance for cluster independence
```

## 🔧 Architecture

### Core Classes

#### TCCDetector
- **Data Loading**: Multi-format satellite data ingestion
- **Preprocessing**: Region extraction and coordinate handling
- **Cluster Detection**: Connected component analysis
- **Characterization**: Physical properties calculation

#### TCCTracker
- **Multi-temporal Tracking**: Time-based cluster association
- **Gap Handling**: Robust tracking across missing timesteps
- **Independence Management**: Distance-based cluster separation

#### RiskAnalyzer
- **Feature Engineering**: Multi-dimensional cluster features
- **ML Model**: Isolation Forest for anomaly detection
- **Risk Scoring**: Normalized cyclogenesis risk (0-1)

#### TCCProcessor
- **Pipeline Orchestration**: End-to-end processing workflow
- **Visualization**: Geographic plots with risk indicators
- **Output Management**: Results and track data export

## 📈 Output Files

### Generated Files
```
outputs/
├── tcc_results.csv          # Complete processing results
├── tracks/
│   └── final_tracks.csv     # Track data for dashboard
├── visualizations/          # Geographic plots
│   └── tcc_YYYYMMDDHHMM.png
├── reports/                 # Analysis reports
└── processing_summary.txt   # Processing statistics
```

### Data Schema
| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Observation time |
| track_id | int | Unique track identifier |
| center_lat | float | Cluster center latitude |
| center_lon | float | Cluster center longitude |
| area_km2 | float | Cluster area in km² |
| min_tb | float | Minimum brightness temperature |
| mean_tb | float | Mean brightness temperature |
| max_radius | float | Maximum cluster radius |
| cloud_top_height_km | float | Cloud top height |
| cyclogenesis_risk | float | ML-predicted risk (0-1) |

## 🎯 Key Algorithms

### 1. Convective Detection
```python
# IRBT threshold-based detection
convective_mask = tb_data < threshold  # Default: 220K
```

### 2. Cluster Identification
```python
# Connected component analysis
labeled = label(convective_mask, connectivity=2)
regions = regionprops(labeled, intensity_image=tb_data)
```

### 3. Risk Assessment
```python
# Feature engineering
features = [min_tb, max_height, mean_radius, pixel_count, intensity_proxy]
risk_score = isolation_forest.predict(features)
```

### 4. Multi-temporal Tracking
```python
# Distance-based association with time-varying search radii
distance = haversine(last_position, new_position)
if distance < search_radius[time_gap]:
    associate_clusters()
```

## 🔍 Advanced Features

### Forecasting System
- **Linear Extrapolation**: Position and intensity forecasting
- **Risk Projection**: Cyclogenesis risk evolution
- **Visualization**: Forecast tracks on maps

### Alert Detection
- **High-Risk Clusters**: Real-time risk monitoring
- **Rapid Intensification**: Sudden development detection
- **Operational Alerts**: Actionable notifications

### Data Quality
- **Missing Data Handling**: Robust processing with incomplete data
- **Format Flexibility**: Multiple satellite data formats
- **Error Recovery**: Graceful failure handling

## 📊 Dashboard Features

### Modern Dashboard
- **Real-time Monitoring**: Live system status
- **Interactive Maps**: Folium-based geographic visualization
- **3D Visualization**: Plotly 3D track plots
- **Risk Analysis**: Real-time risk assessment

### Classic View
- **Time Series Analysis**: Multi-axis property plots
- **Development Patterns**: Cluster evolution analysis
- **Correlation Analysis**: Feature relationship heatmaps
- **Report Generation**: PDF export capabilities

## 🛠️ Troubleshooting

### Common Issues

#### 1. No Data Files Found
```bash
# Check data directory
ls data/insat_real/

# Ensure supported formats
# .nc, .nc4, .hdf, .h4, .h5
```

#### 2. Import Errors
```bash
# Install missing dependencies
pip install -r requirements_tcc.txt

# For HDF4 issues on Windows
python setup_hdf4.py
```

#### 3. Memory Issues
```yaml
# Reduce processing region in config
region:
  lat_range: [10, 20]  # Smaller area
  lon_range: [70, 85]
```

#### 4. No Clusters Detected
```yaml
# Adjust thresholds in config
thresholds:
  irbt: 230        # Higher threshold
  min_area: 20000  # Smaller minimum area
```

## 📚 References

### Scientific Background
- **Tropical Cloud Clusters**: Convective organization in tropics
- **Cyclogenesis**: Tropical cyclone formation processes
- **INSAT-3D**: Indian geostationary satellite system

### Technical Methods
- **Connected Component Analysis**: Image processing technique
- **Haversine Distance**: Great-circle distance calculation
- **Isolation Forest**: Anomaly detection algorithm

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd tropical_cloud_project

# Install development dependencies
pip install -r requirements_tcc.txt
pip install pytest black flake8

# Run tests
python test_missing_data_handling.py
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions
- Include error handling

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **INSAT-3D Data**: Indian Space Research Organisation (ISRO)
- **Scientific Community**: Tropical meteorology researchers
- **Open Source**: Python scientific computing ecosystem

---

**For support and questions, please check the logs in the `logs/` directory or create an issue in the project repository.** 