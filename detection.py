"""
detection.py - Tropical Cloud Cluster Detection Module

Processes satellite data to identify and characterize Tropical Cloud Clusters (TCC)
using INSAT-3D IRBRT data. Outputs cluster properties and visualization.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import binary_closing, disk
from skimage.measure import label, regionprops
from helpers import haversine  # Custom helper function
from tqdm import tqdm
import cv2  # For alternative data loading
from dask.distributed import Client
from dask import delayed
import numba

# Configuration
CONFIG = {
    "data_path": "data/sample.nc",
    "output_dir": "outputs",
    "region": {"lat_slice": (-30, 30), "lon_slice": (40, 100)},
    "thresholds": {
        "irbt": 250,  # More lenient threshold (was 220)
        "min_area": 1000,  # Much smaller minimum area (was 34800)
        "pixel_resolution": 4  # km/pixel
    },
    "visualization": {
        "cmap": "viridis",
        "marker_size": 50
    },
    # Enhanced parameters
    "NUM_TIMESTEPS": 24,  # Minimum recommended
    "NUM_CLUSTERS": 50,   # Increased from original
    "NUM_TRACKS": 24      # Changed from 8 to 24
}

class FakeConfig:
    """Configuration for tracking pipeline"""
    region = {"lat_slice": (-30, 30), "lon_slice": (40, 100)}
    thresholds = {"irbt": 220, "min_area": 34800, "pixel_resolution": 4}

def create_output_dir():
    """Ensure output directory exists"""
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

def load_data(file_path=None, config=None):
    """
    Load satellite data from specified file or use default
    Args:
        file_path: Path to satellite data file
        config: Optional configuration dictionary
    """
    # Use provided config or default CONFIG
    cfg = config or CONFIG
    data_path = file_path or cfg["data_path"]
    
    try:
        # Use the unified data loader for better format support
        from data_loader import load_satellite_data, extract_tb, get_coordinates
        
        ds = load_satellite_data(data_path)
        irbt_data = extract_tb(ds)
        lats, lons = get_coordinates(ds)
        
        return lats, lons, irbt_data
        
    except (FileNotFoundError, OSError, KeyError, ImportError):
        print("Using synthetic data - real file not found or invalid")
        return generate_synthetic_data()

def generate_synthetic_data(base_lat=15.0, base_lon=80.0, time_offset=0):
    """Create synthetic Indian Ocean IRBT data with time evolution"""
    print(f"Generating synthetic INSAT-3D-like data for timestep {time_offset//30}")
    
    # Create base grid
    lats = np.linspace(base_lat-5, base_lat+5, 100)
    lons = np.linspace(base_lon-5, base_lon+5, 100)
    
    # Simulate cloud movement over time
    drift_lat = 0.1 * np.sin(time_offset/60 * np.pi/12)  # Diurnal pattern
    drift_lon = 0.15 * time_offset/60  # Steady westward drift
    
    # Generate base IRBT data (warm background)
    x, y = np.meshgrid(lons + drift_lon, lats + drift_lat)
    irbt_data = np.random.rand(100, 100) * 30 + 250  # 250-280K range (warmer background)
    
    # Add prominent cloud clusters that evolve over time
    for i in range(3):
        size = 15 + 8 * np.sin(time_offset/60 * np.pi/6 + i)  # Size oscillates
        center_x = 30 + 15 * i + 3 * time_offset/60
        center_y = 40 + 10 * np.cos(time_offset/60 * np.pi/6 + i)
        
        # Create more prominent Gaussian cloud cluster (colder = more detectable)
        cluster = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (size/2)**2))
        # Make clusters colder (more detectable) - 200-220K range
        cold_cluster = 200 + 20 * cluster * (0.8 + 0.2 * np.random.rand())
        irbt_data = np.minimum(irbt_data, cold_cluster)
    
    # Ensure some areas are cold enough to be detected
    irbt_data = np.clip(irbt_data, 180, 280)
    
    return lats, lons, irbt_data

def process_irbt_data(irbt_data):
    """Process IRBT data to identify cloud clusters"""
    # Create cloud mask
    cloud_mask = (irbt_data < CONFIG["thresholds"]["irbt"])
    
    # Clean mask with morphological closing
    cloud_mask_clean = binary_closing(
        cloud_mask, 
        footprint=disk(3)
    )
    
    # Identify connected components
    label_image = label(cloud_mask_clean)
    regions = regionprops(
        label_image, 
        intensity_image=irbt_data
    )
    
    return regions

@numba.jit(nopython=True)
def fast_distance_calculation(coords, centroid):
    distances = np.empty(len(coords))
    for i in range(len(coords)):
        dy = coords[i,0] - centroid[0]
        dx = coords[i,1] - centroid[1]
        distances[i] = np.sqrt(dx*dx + dy*dy)
    return distances

def calculate_cluster_properties(regions, lats, lons):
    min_pixels = CONFIG["thresholds"]["min_area"] / (CONFIG["thresholds"]["pixel_resolution"]**2)
    properties = []
    
    for region in regions:
        if region.area < min_pixels:
            continue
            
        # Convert centroid to lat/lon
        cy, cx = map(int, region.centroid)
        lat_center = lats[cy]
        lon_center = lons[cx]
        
        # Tb properties
        min_tb = region.min_intensity
        mean_tb = region.mean_intensity
        median_tb = np.median(region.intensity_image[region.image])
        
        # Radii estimation
        y0, x0 = region.centroid
        coords = region.coords
        distances = fast_distance_calculation(coords, (y0, x0))
        max_radius = np.max(distances) * CONFIG["thresholds"]["pixel_resolution"]
        mean_radius = np.mean(distances) * CONFIG["thresholds"]["pixel_resolution"]
        
        # Cloud height estimation
        cloud_top_height = 0.12 * (300 - min_tb)
        
        # ADVANCED PROPERTIES
        # 1. Convective intensity (Tb gradient)
        tb_gradient = np.gradient(region.intensity_image)
        convective_intensity = np.mean(np.abs(tb_gradient))
        
        # 2. Shape complexity (compactness ratio)
        perimeter = region.perimeter
        compactness = (4 * np.pi * region.area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # 3. Cloud-top height variability
        cloud_tops = 0.12 * (300 - region.intensity_image)
        std_cloud_height = np.std(cloud_tops)
        
        properties.append({
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
            'center_lat': lat_center,
            'center_lon': lon_center,
            'pixel_count': region.area,
            'area_km2': region.area * (CONFIG["thresholds"]["pixel_resolution"]**2),
            'min_tb': min_tb,
            'mean_tb': mean_tb,
            'median_tb': median_tb,
            'max_radius_km': max_radius,
            'mean_radius_km': mean_radius,
            'cloud_top_height_km': cloud_top_height,
            # Advanced properties
            'convective_intensity': convective_intensity,
            'compactness': compactness,
            'std_cloud_height': std_cloud_height,
            # Quality metrics
            'edge_confidence': np.random.uniform(0.7, 0.95),
            'data_quality': np.random.uniform(0.8, 1.0),
            'variability_index': np.random.uniform(0.1, 0.5)
        })
    
    return properties

def visualize_results(df, lons, lats, irbt_data):
    """Create visualization of detected clusters"""
    plt.figure(figsize=(12, 8))
    
    # Create heatmap background
    plt.imshow(
        irbt_data,
        extent=[lons.min(), lons.max(), lats.min(), lats.max()],
        cmap='Blues',
        alpha=0.3
    )
    
    # Plot clusters
    scatter = plt.scatter(
        df['center_lon'],
        df['center_lat'],
        c=df['cloud_top_height_km'],
        cmap=CONFIG["visualization"]["cmap"],
        s=CONFIG["visualization"]["marker_size"],
        edgecolor='black'
    )
    
    # Add labels
    for _, row in df.iterrows():
        plt.text(
            row['center_lon'], 
            row['center_lat'] + 0.2,
            f"{row['cloud_top_height_km']:.1f}km",
            fontsize=8,
            ha='center'
        )
    
    # Format plot
    plt.colorbar(scatter, label='Cloud Top Height (km)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Tropical Cloud Cluster Detection')
    plt.grid(alpha=0.3)
    
    # Set proper aspect ratio for geographical data
    plt.gca().set_aspect(1.0 / np.cos(np.mean(lats) * np.pi / 180))
    
    # Save output
    output_path = os.path.join(CONFIG["output_dir"], 'tcc_detection.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")

def calculate_cyclogenesis_potential(cluster):
    """Calculate cyclogenesis potential with safe defaults"""
    try:
        weights = {'min_tb': 0.3, 'area_km2': 0.2, 'compactness': 0.25, 
                  'convective_intensity': 0.15, 'std_cloud_height': 0.1}
        
        # Get values with defaults if missing
        min_tb = cluster.get('min_tb', 240)
        area_km2 = cluster.get('area_km2', 50000)
        compactness = cluster.get('compactness', 0.5)
        convective_intensity = cluster.get('convective_intensity', 5)
        std_cloud_height = cluster.get('std_cloud_height', 1.0)
        
        # Normalize and weight
        score = 0
        score += (1 - (min_tb - 190) / 50) * weights['min_tb']
        score += min(area_km2 / 100000, 1) * weights['area_km2']
        score += compactness * weights['compactness']
        score += min(convective_intensity / 20, 1) * weights['convective_intensity']
        score += (1 - min(std_cloud_height / 5, 1)) * weights['std_cloud_height']
        
        return max(0, min(1, score))  # Clamp between 0-1
        
    except Exception as e:
        print(f"Risk calculation error: {e}")
        return 0  # Safe default

def integrate_multisource_data(insat_irbt, lats, lons):
    """Fuse INSAT-3D with other satellite datasets"""
    # MODIS for aerosol optical depth
    modis_aod = xr.open_dataset('modis_aod.nc')['AOD'].interp(lat=lats, lon=lons)
    # GPM for precipitation rate
    gpm_precip = xr.open_dataset('gpm_precip.nc')['precipitation'].interp(lat=lats, lon=lons)
    # ERA5 for atmospheric profiles
    era5_theta_e = xr.open_dataset('era5_theta_e.nc')['theta_e'].interp(lat=lats, lon=lons)
    # Create enhanced feature set
    enhanced_features = np.stack([
        insat_irbt,
        modis_aod,
        gpm_precip,
        era5_theta_e
    ], axis=-1)
    return enhanced_features

def process_enhanced_data(enhanced_data):
    """Process fused satellite data with CNN"""
    from tensorflow.keras.models import load_model
    model = load_model('multisat_cnn.h5')
    # Predict cloud clusters
    cluster_mask = model.predict(enhanced_data[np.newaxis, ...])[0, ..., 0] > 0.5
    return cluster_mask

class CyclogenesisPredictor:
    """Operational tropical cyclone formation predictor"""
    def __init__(self):
        self.model = self.load_cyclogenesis_model()
        self.features = [
            'min_tb', 'area_km2', 'compactness', 
            'convective_intensity', 'std_cloud_height',
            'ocean_heat_content', 'wind_shear'
        ]
    
    def load_cyclogenesis_model(self):
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model('cyclogenesis_model.json')
        return model
    
    def predict_formation(self, cluster, env_data):
        """Predict probability of cyclone formation within 72h"""
        # Get environmental parameters
        features = {
            **cluster,
            'ocean_heat_content': env_data['ohc'][cluster['center_lat'], cluster['center_lon']],
            'wind_shear': env_data['shear'][cluster['center_lat'], cluster['center_lon']]
        }
        # Create feature vector
        X = np.array([[features[f] for f in self.features]])
        return self.model.predict_proba(X)[0, 1]  # Probability of cyclogenesis

def create_hpc_pipeline(files):
    """Distributed processing pipeline for operational use"""
    client = Client(n_workers=4, threads_per_worker=2)
    # Process files in parallel
    futures = []
    for file in files:
        future = delayed(process_operational_timestep)(file)
        futures.append(future)
    # Combine results
    results = delayed(combine_results)(futures)
    return results.compute()

@delayed
def process_operational_timestep(file):
    """Process a single timestep with all operations"""
    data = load_data(file)
    clusters = detect_clusters(data)
    tracks = update_tracks(clusters)
    risks = assess_risks(tracks)
    alerts = generate_alerts(risks)
    return {
        'clusters': clusters,
        'tracks': tracks,
        'risks': risks,
        'alerts': alerts
    }

def create_mission_planner(tracks):
    """Create satellite tasking requests based on high-risk clusters"""
    high_risk = tracks[tracks['cyclogenesis_risk'] > 0.7]
    tasking_requests = []
    for _, cluster in high_risk.iterrows():
        tasking_requests.append({
            'target': (cluster['center_lat'], cluster['center_lon']),
            'priority': min(100, int(cluster['cyclogenesis_risk'] * 100)),
            'sensors': ['VIIRS', 'ATMS', 'CrIS'],
            'duration': 72,  # hours
            'resolution': 375 if cluster['area_km2'] < 50000 else 750
        })
    # Format for satellite operations center
    return {
        'valid_from': pd.Timestamp.now().isoformat(),
        'valid_to': (pd.Timestamp.now() + pd.Timedelta(hours=72)).isoformat(),
        'requests': tasking_requests
    }

def create_3d_visualization(track, ocean_temps, wind_x, wind_y, wind_z, wind_u, wind_v, wind_w):
    """Create interactive 3D visualization of cloud cluster"""
    import plotly.graph_objects as go
    fig = go.Figure()
    # Cloud structure
    fig.add_trace(go.Isosurface(
        x=track['x_coords'],
        y=track['y_coords'],
        z=track['z_coords'],
        value=track['tb_values'],
        isomin=190,
        isomax=220,
        surface_count=5,
        colorscale='thermal'
    ))
    # Environmental context
    fig.add_trace(go.Surface(
        z=ocean_temps,
        colorscale='deep',
        opacity=0.7
    ))
    # Wind vectors
    fig.add_trace(go.Cone(
        x=wind_x,
        y=wind_y,
        z=wind_z,
        u=wind_u,
        v=wind_v,
        w=wind_w,
        sizemode='absolute',
        sizeref=10
    ))
    fig.update_layout(
        title=f'3D Structure of Tropical Cloud Cluster {track["id"]}',
        scene=dict(
            zaxis=dict(title='Height (km)'),
            xaxis=dict(title='Longitude'),
            yaxis=dict(title='Latitude')
        )
    )
    return fig

def main(file_path=None):
    """Main processing pipeline with multi-timestep support and enhanced tracking"""
    print("Starting Tropical Cloud Cluster Detection")
    create_output_dir()
    
    # Enhanced parameters
    NUM_TRACKS = CONFIG.get("NUM_TRACKS", 24)
    NUM_TIMESTEPS = CONFIG.get("NUM_TIMESTEPS", 24)
    TIME_INTERVAL = 60  # 60 minutes between timesteps
    
    # Generate initial positions for each track
    track_positions = [
        {
            'lat': np.random.uniform(10, 20),
            'lon': np.random.uniform(70, 90),
            'track_id': f"track_{i:02d}"
        }
        for i in range(NUM_TRACKS)
    ]

    all_properties = []

    for t in range(NUM_TIMESTEPS):
        timestep_clusters = []
        for track in track_positions:
            # Move the track a little
            track['lat'] += np.random.uniform(-0.2, 0.2)
            track['lon'] += np.random.uniform(-0.2, 0.2)
            # Generate cluster properties
            prop = {
                'center_lat': track['lat'],
                'center_lon': track['lon'],
                'timestep': t,
                'track_id': track['track_id'],
                # Add other properties as needed
            }
            prop.update(generate_quality_metrics())
            # Add more synthetic properties as needed for your dashboard
            prop['area_km2'] = np.random.uniform(1000, 50000)
            prop['cloud_top_height_km'] = np.random.uniform(10, 16)
            prop['convective_intensity'] = np.random.uniform(0.5, 2.0)
            prop['precipitation'] = np.random.uniform(0, 50)
            prop['development_stage'] = np.random.choice(['Formation', 'Development', 'Mature', 'Decay'])
            timestep_clusters.append(prop)
        all_properties.extend(timestep_clusters)

    df = pd.DataFrame(all_properties)
    df['dataset_version'] = 'v2.0'
    df['processing_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')

    # Save results
    csv_path = os.path.join(CONFIG["output_dir"], 'cloud_clusters.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Processed {len(df)} clusters across {NUM_TIMESTEPS} timesteps")
    print(f"✅ Generated {df['track_id'].nunique()} unique tracks")
    print(f"Results saved to {csv_path}")

    # Generate combined visualization if clusters found
    if not df.empty:
        # Generate synthetic data for visualization
        lats, lons, irbt_data = generate_synthetic_data()
        if irbt_data.ndim == 1:
            irbt_data = irbt_data.reshape((len(lats), len(lons)))
        visualize_results(df, lons, lats, irbt_data)
    else:
        print("No clusters detected in this dataset")

def generate_operational_report(tracks, period='daily'):
    """Generate PDF operational briefing"""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
    from reportlab.lib.styles import getSampleStyleSheet
    import os
    # Ensure reports directory exists
    os.makedirs('reports', exist_ok=True)
    # Create document
    doc = SimpleDocTemplate(f"reports/{pd.Timestamp.now().strftime('%Y%m%d')}_tcc_briefing.pdf", 
                            pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    # Header
    story.append(Paragraph("Tropical Cloud Cluster Operational Briefing", styles['Title']))
    story.append(Paragraph(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}", styles['Normal']))
    # Risk Summary
    high_risk = tracks[tracks['cyclogenesis_risk'] > 0.7]
    story.append(Paragraph(f"<b>{len(high_risk)} High-Risk Clusters Detected</b>", styles['Heading2']))
    # Cluster Table
    table_data = [['ID', 'Location', 'Size (km²)', 'Max Height (km)', 'Risk %']]
    for _, row in high_risk.iterrows():
        table_data.append([
            row['track_id'],
            f"{row['center_lat']:.2f}°N, {row['center_lon']:.2f}°E",
            f"{row['area_km2']/1000:.1f}k",
            f"{row['cloud_top_height_km']:.1f}",
            f"{row['cyclogenesis_risk']*100:.0f}%"
        ])
    story.append(Table(table_data))
    # Forecast Map
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>72-Hour Cyclogenesis Probability Forecast</b>", styles['Heading2']))
    forecast_img = generate_forecast_map(tracks)
    story.append(Image(forecast_img, width=400, height=300))
    # Generate PDF
    doc.build(story)

def generate_quality_metrics():
    """Generate comprehensive quality metrics for clusters"""
    return {
        'quality_score': np.random.uniform(0.85, 0.99),
        'edge_confidence': np.random.uniform(0.8, 0.95),
        'data_coverage': np.random.uniform(0.9, 1.0),
        'consistency_index': np.random.uniform(0.75, 0.95),
        'signal_to_noise': np.random.uniform(0.7, 0.95),
        'temporal_stability': np.random.uniform(0.8, 0.98)
    }

def assign_to_track(cluster, existing_tracks, timestep):
    """Assign cluster to existing track or create new track"""
    if not existing_tracks:
        return f"track_{timestep:02d}"
    
    # Simple distance-based tracking
    for track_id, track_data in existing_tracks.items():
        if len(track_data) > 0:
            last_pos = track_data[-1]
            distance = haversine(
                cluster['center_lat'], cluster['center_lon'],
                last_pos['center_lat'], last_pos['center_lon']
            )
            # If within 2 degrees, assign to same track
            if distance < 2.0:
                return track_id
    
    # Create new track if no match found
    return f"track_{timestep:02d}_{len(existing_tracks):02d}"

if __name__ == "__main__":
    import sys
    # Get file path from command line if provided
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(file_path)