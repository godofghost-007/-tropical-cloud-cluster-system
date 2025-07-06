import numpy as np
import pandas as pd
import xarray as xr
import h5py
# Handle pyhdf import gracefully for Windows compatibility
try:
    from pyhdf.SD import SD, SDC
    HDF4_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pyhdf not available ({e}). HDF4 files will not be supported.")
    HDF4_AVAILABLE = False
    SD = None
    SDC = None

import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta
import os
import yaml
import logging
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
import joblib
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import IsolationForest

# Constants (configurable via YAML)
CONFIG = {
    "data": {
        "input_dir": "data/insat_real",
        "output_dir": "outputs",
        "region": {
            "lat_range": [5, 25],
            "lon_range": [60, 90]
        },
        "thresholds": {
            "irbt": 220,
            "min_area": 34800,  # 34,800 km²
            "min_radius": 111   # 1° ≈ 111 km
        },
        "tracking": {
            "search_radii": {3: 450, 6: 550, 9: 600, 12: 650},
            "max_gap": 12,  # hours
            "independence_dist": 1200  # km
        },
        "resolution_km": 4.0
    },
    "convection": {
        "tb_to_height": {
            "a": 12.0,
            "b": -0.02,
            "c": 200
        }
    }
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tcc_detection.log"),
        logging.StreamHandler()
    ]
)

class TCCDetector:
    def __init__(self, config):
        self.config = config
        self.resolution_km = config["data"]["resolution_km"]
        self.min_pixels = int(config["data"]["thresholds"]["min_area"] / (self.resolution_km ** 2))
        
    def load_insat_data(self, file_path):
        """Load INSAT-3D data from various formats"""
        try:
            if file_path.endswith(('.nc', '.nc4')):
                ds = xr.open_dataset(file_path)
                return ds
            elif file_path.endswith('.h5'):
                with h5py.File(file_path, 'r') as f:
                    return xr.open_dataset(xr.backends.H5netcdfStore(f))
            elif file_path.endswith(('.hdf', '.h4')):
                if not HDF4_AVAILABLE:
                    logging.warning(f"HDF4 support not available. Skipping {file_path}")
                    return None
                hdf = SD(file_path, SDC.READ)
                ds = xr.Dataset()
                for var in hdf.datasets().keys():
                    data = hdf.select(var).get()
                    ds[var] = xr.DataArray(data)
                return ds
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            logging.error(f"Error loading {file_path}: {str(e)}")
            return None
    
    def preprocess_data(self, ds):
        """Preprocess INSAT data and extract region of interest"""
        if ds is None:
            return None, None, None
            
        # Extract brightness temperature
        tb_candidates = ['IRBRT', 'brightness_temperature', 'BT', 'temp_11um', 'brightness_temp']
        tb_data = None
        for var in tb_candidates:
            if var in ds:
                tb_data = ds[var].values
                break
        
        if tb_data is None:
            # Try to find any temperature-like variable
            for var in ds.data_vars:
                if 'temp' in var.lower() or 'brightness' in var.lower():
                    tb_data = ds[var].values
                    break
        
        if tb_data is None:
            raise KeyError("Brightness temperature field not found")
        
        # Extract coordinates
        lat_candidates = ['latitude', 'lat', 'y']
        lon_candidates = ['longitude', 'lon', 'x']
        
        lats = None
        lons = None
        
        for var in lat_candidates:
            if var in ds:
                lats = ds[var].values
                break
                
        for var in lon_candidates:
            if var in ds:
                lons = ds[var].values
                break
        
        if lats is None or lons is None:
            # Use dimension coordinates
            dims = list(ds.dims.keys())
            if len(dims) >= 2:
                lats = ds[dims[0]].values
                lons = ds[dims[1]].values
        
        if lats is None or lons is None:
            raise KeyError("Latitude/longitude coordinates not found")
        
        # Crop to region of interest
        lat_min, lat_max = self.config["data"]["region"]["lat_range"]
        lon_min, lon_max = self.config["data"]["region"]["lon_range"]
        
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        
        tb_data = tb_data[lat_mask, :][:, lon_mask]
        lats = lats[lat_mask]
        lons = lons[lon_mask]
        
        return tb_data, lats, lons
    
    def detect_convective_regions(self, tb_data):
        """Identify convective regions using IRBT threshold"""
        if tb_data is None:
            return None
        threshold = self.config["data"]["thresholds"]["irbt"]
        convective_mask = tb_data < threshold
        return convective_mask
    
    def calculate_cloud_height(self, tb):
        """Convert brightness temperature to cloud-top height"""
        a = self.config["convection"]["tb_to_height"]["a"]
        b = self.config["convection"]["tb_to_height"]["b"]
        c = self.config["convection"]["tb_to_height"]["c"]
        return a + b * (tb - c)
    
    def identify_clusters(self, convective_mask, tb_data, lats, lons):
        """Identify and characterize TCC candidates"""
        if convective_mask is None or tb_data is None:
            return []
            
        # Label connected regions
        labeled = label(convective_mask, connectivity=2)
        regions = regionprops(labeled, intensity_image=tb_data)
        
        clusters = []
        for region in regions:
            # Skip small regions
            if region.area < self.min_pixels:
                continue
                
            # Extract region properties
            min_row, min_col, max_row, max_col = region.bbox
            region_lats = lats[min_row:max_row]
            region_lons = lons[min_col:max_col]
            region_tb = region.intensity_image
            
            # Find coldest pixel (convective center)
            min_tb_idx = np.argmin(region_tb)
            min_tb_coords = np.unravel_index(min_tb_idx, region_tb.shape)
            center_lat = region_lats[min_tb_coords[0]]
            center_lon = region_lons[min_tb_coords[1]]
            min_tb = region_tb[min_tb_coords]
            
            # Calculate radii properties
            edge_mask = binary_dilation(region.image) ^ region.image
            edge_distances = []
            
            for i in range(region.image.shape[0]):
                for j in range(region.image.shape[1]):
                    if edge_mask[i, j]:
                        # Calculate great-circle distance
                        lat1, lon1 = region_lats[i], region_lons[j]
                        dist = self.haversine(center_lon, center_lat, lon1, lat1)
                        edge_distances.append(dist)
            
            if not edge_distances:
                continue
                
            max_radius = max(edge_distances)
            min_radius = min(edge_distances)
            mean_radius = np.mean(edge_distances)
            
            # Check size requirement
            if max_radius < self.config["data"]["thresholds"]["min_radius"]:
                continue
                
            # Calculate Tb statistics
            tb_values = region_tb[region.image]
            mean_tb = np.mean(tb_values)
            median_tb = np.median(tb_values)
            std_tb = np.std(tb_values)
            
            # Calculate cloud heights
            heights = self.calculate_cloud_height(tb_values)
            max_height = np.max(heights)
            mean_height = np.mean(heights)
            
            # Calculate area in km²
            area_km2 = region.area * (self.resolution_km ** 2)
            
            clusters.append({
                "center_lat": center_lat,
                "center_lon": center_lon,
                "pixel_count": region.area,
                "area_km2": area_km2,
                "min_tb": min_tb,
                "mean_tb": mean_tb,
                "median_tb": median_tb,
                "std_tb": std_tb,
                "max_radius": max_radius,
                "min_radius": min_radius,
                "mean_radius": mean_radius,
                "max_height": max_height,
                "mean_height": mean_height,
                "cloud_top_height_km": max_height,
                "mask": region.image,
                "bbox": region.bbox
            })
        
        return clusters
    
    def handle_independence(self, clusters):
        """Ensure cluster independence based on distance criteria"""
        if not clusters:
            return []
        
        # Sort clusters by size (largest first)
        clusters.sort(key=lambda x: x["pixel_count"], reverse=True)
        independent_clusters = []
        removed_indices = set()
        
        # Create distance matrix
        positions = np.array([[c["center_lat"], c["center_lon"]] for c in clusters])
        dist_matrix = cdist(positions, positions, metric=self.haversine_vectorized)
        
        max_dist = self.config["data"]["tracking"]["independence_dist"]
        
        for i, cluster in enumerate(clusters):
            if i in removed_indices:
                continue
                
            independent_clusters.append(cluster)
            
            # Find dependent clusters
            for j in range(i + 1, len(clusters)):
                if dist_matrix[i, j] <= max_dist:
                    removed_indices.add(j)
        
        return independent_clusters
    
    def haversine(self, lon1, lat1, lon2, lat2):
        """Calculate great-circle distance between two points"""
        # Convert degrees to radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in km
        r = 6371
        return c * r
    
    def haversine_vectorized(self, pos1, pos2):
        """Vectorized haversine distance calculation"""
        lat1, lon1 = np.radians(pos1)
        lat2, lon2 = np.radians(pos2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 6371 * c

class TCCTracker:
    def __init__(self, config):
        self.config = config
        self.tracks = {}  # track_id: {history: list, last_update: datetime}
        self.next_track_id = 1
        self.search_radii = config["data"]["tracking"]["search_radii"]
        self.max_gap = config["data"]["tracking"]["max_gap"]
        
    def update_tracks(self, timestamp, clusters):
        """Update existing tracks and create new ones"""
        active_tracks = {}
        matched_clusters = set()
        
        # Update existing tracks
        for track_id, track_data in self.tracks.items():
            if (timestamp - track_data["last_update"]).total_seconds() / 3600 > self.max_gap:
                continue  # Skip expired tracks
                
            last_cluster = track_data["history"][-1]
            hours_gap = (timestamp - track_data["last_update"]).total_seconds() / 3600
            
            # Find appropriate search radius
            search_radius = self.search_radii.get(
                min(self.search_radii.keys(), key=lambda x: abs(x - hours_gap)),
                self.search_radii[max(self.search_radii.keys())]
            )
            
            best_match = None
            min_distance = float('inf')
            
            for i, cluster in enumerate(clusters):
                if i in matched_clusters:
                    continue
                    
                distance = self.haversine(
                    last_cluster["center_lon"], last_cluster["center_lat"],
                    cluster["center_lon"], cluster["center_lat"]
                )
                
                if distance < search_radius and distance < min_distance:
                    best_match = i
                    min_distance = distance
            
            if best_match is not None:
                # Update track
                track_data["history"].append(clusters[best_match])
                track_data["last_update"] = timestamp
                active_tracks[track_id] = track_data
                matched_clusters.add(best_match)
        
        # Create new tracks for unmatched clusters
        for i, cluster in enumerate(clusters):
            if i not in matched_clusters:
                track_id = self.next_track_id
                self.next_track_id += 1
                
                active_tracks[track_id] = {
                    "history": [cluster],
                    "last_update": timestamp,
                    "start_time": timestamp
                }
        
        self.tracks = active_tracks
        return active_tracks
    
    def haversine(self, lon1, lat1, lon2, lat2):
        """Calculate great-circle distance between two points"""
        # (Same implementation as in TCCDetector)
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c

class RiskAnalyzer:
    def __init__(self, config):
        self.config = config
        self.model = self.train_model()
        
    def train_model(self):
        """Train cyclogenesis risk prediction model"""
        # Create synthetic training data based on typical cluster properties
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic training features
        min_tb = np.random.uniform(180, 250, n_samples)  # Brightness temperature
        max_height = np.random.uniform(8, 18, n_samples)  # Cloud top height
        mean_radius = np.random.uniform(50, 200, n_samples)  # Cluster radius
        pixel_count = np.random.uniform(1000, 10000, n_samples)  # Cluster size
        intensity_proxy = np.random.uniform(0.1, 2.0, n_samples)  # Intensity
        
        # Combine features
        X_train = np.column_stack([
            min_tb,
            max_height,
            mean_radius,
            pixel_count / 1000,
            intensity_proxy
        ])
        
        # Train the model
        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        model.fit(X_train)
        
        return model
    
    def calculate_risk(self, cluster):
        """Calculate cyclogenesis risk for a cluster"""
        try:
            # Feature engineering
            features = np.array([
                cluster.get("min_tb", 220),
                cluster.get("max_height", 10),
                cluster.get("mean_radius", 100),
                cluster.get("pixel_count", 1000) / 1000,
                (220 - cluster.get("min_tb", 220)) / max(cluster.get("mean_radius", 100), 1)  # Intensity proxy
            ]).reshape(1, -1)
            
            # Predict anomaly score (higher score = higher risk)
            risk_score = -self.model.decision_function(features)[0]
            
            # Normalize to 0-1 range
            risk_score = (risk_score - 0.2) / 0.8  # Adjust based on model behavior
            return max(0, min(1, risk_score))
        except Exception as e:
            logging.warning(f"Risk calculation failed: {str(e)}")
            return 0.1  # Default low risk

class TCCProcessor:
    def __init__(self, config):
        self.config = config
        self.detector = TCCDetector(config)
        self.tracker = TCCTracker(config)
        self.risk_analyzer = RiskAnalyzer(config)
        self.output_dir = config["data"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        
    def process_files(self, file_paths):
        """Process a sequence of INSAT data files"""
        all_results = []
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            try:
                timestamp = self.extract_timestamp(file_path)
                ds = self.detector.load_insat_data(file_path)
                
                if ds is None:
                    continue
                    
                tb_data, lats, lons = self.detector.preprocess_data(ds)
                
                if tb_data is None:
                    continue
                
                # Detect convective regions
                convective_mask = self.detector.detect_convective_regions(tb_data)
                
                # Identify and characterize clusters
                clusters = self.detector.identify_clusters(convective_mask, tb_data, lats, lons)
                independent_clusters = self.detector.handle_independence(clusters)
                
                # Update tracks
                tracks = self.tracker.update_tracks(timestamp, independent_clusters)
                
                # Calculate risk and prepare results
                for track_id, track_data in tracks.items():
                    latest_cluster = track_data["history"][-1]
                    
                    # Calculate cyclogenesis risk
                    risk = self.risk_analyzer.calculate_risk(latest_cluster)
                    
                    # Prepare result entry
                    result = {
                        "timestamp": timestamp,
                        "datetime": timestamp,
                        "track_id": track_id,
                        "center_lat": latest_cluster["center_lat"],
                        "center_lon": latest_cluster["center_lon"],
                        "pixel_count": latest_cluster["pixel_count"],
                        "area_km2": latest_cluster.get("area_km2", 0),
                        "min_tb": latest_cluster["min_tb"],
                        "mean_tb": latest_cluster["mean_tb"],
                        "median_tb": latest_cluster["median_tb"],
                        "std_tb": latest_cluster["std_tb"],
                        "max_radius": latest_cluster["max_radius"],
                        "min_radius": latest_cluster["min_radius"],
                        "mean_radius": latest_cluster["mean_radius"],
                        "max_height": latest_cluster["max_height"],
                        "mean_height": latest_cluster["mean_height"],
                        "cloud_top_height_km": latest_cluster.get("cloud_top_height_km", 0),
                        "cyclogenesis_risk": risk,
                        "duration_hours": (timestamp - track_data["start_time"]).total_seconds() / 3600
                    }
                    all_results.append(result)
                
                # Generate visualization
                if len(independent_clusters) > 0:
                    self.generate_visualization(tb_data, lats, lons, independent_clusters, timestamp)
                    
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")
        
        # Save results
        if all_results:
            results_df = pd.DataFrame(all_results)
            output_path = os.path.join(self.output_dir, "tcc_results.csv")
            results_df.to_csv(output_path, index=False)
            logging.info(f"Saved results to {output_path}")
            
            # Save tracks for dashboard
            tracks_path = os.path.join(self.output_dir, "tracks", "final_tracks.csv")
            os.makedirs(os.path.dirname(tracks_path), exist_ok=True)
            results_df.to_csv(tracks_path, index=False)
            logging.info(f"Saved track data to {tracks_path}")
            
            return results_df
        else:
            logging.warning("No results generated")
            return pd.DataFrame()
    
    def extract_timestamp(self, file_path):
        """Extract timestamp from filename (implementation varies by data format)"""
        try:
            # Example: INSAT3D_IRBRT_202307061200.nc
            filename = os.path.basename(file_path)
            
            # Try different patterns
            patterns = [
                "%Y%m%d%H%M",  # 202307061200
                "%Y%m%d_%H%M",  # 20230706_1200
                "%Y-%m-%d_%H%M",  # 2023-07-06_1200
                "%Y%m%d",  # 20230706
            ]
            
            for pattern in patterns:
                try:
                    # Extract date string from filename
                    parts = filename.split("_")
                    for part in parts:
                        if len(part) >= 8 and part.isdigit():
                            if pattern == "%Y%m%d%H%M" and len(part) == 12:
                                return datetime.strptime(part, pattern)
                            elif pattern == "%Y%m%d_%H%M" and len(part) == 8:
                                # Try to find time part
                                for p in parts:
                                    if len(p) == 4 and p.isdigit():
                                        return datetime.strptime(part + "_" + p, pattern)
                            elif pattern == "%Y%m%d" and len(part) == 8:
                                return datetime.strptime(part, pattern)
                except:
                    continue
            
            # If no pattern matches, use file modification time
            return datetime.fromtimestamp(os.path.getmtime(file_path))
            
        except Exception as e:
            logging.warning(f"Could not extract timestamp from {file_path}: {str(e)}")
            return datetime.now()
    
    def generate_visualization(self, tb_data, lats, lons, clusters, timestamp):
        """Generate visualization for current time step"""
        try:
            plt.figure(figsize=(12, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            # Plot brightness temperature
            plt.contourf(lons, lats, tb_data, levels=20, cmap='viridis', transform=ccrs.PlateCarree())
            plt.colorbar(label='Brightness Temperature (K)')
            
            # Add geographic features
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.OCEAN)
            ax.add_feature(cfeature.LAND)
            ax.gridlines(draw_labels=True)
            
            # Plot clusters
            for cluster in clusters:
                center_lon, center_lat = cluster["center_lon"], cluster["center_lat"]
                plt.plot(center_lon, center_lat, 'ro', markersize=3, transform=ccrs.PlateCarree())
                
                # Add risk indicator
                risk = self.risk_analyzer.calculate_risk(cluster)
                color = 'red' if risk > 0.7 else 'orange' if risk > 0.5 else 'yellow'
                circle = plt.Circle(
                    (center_lon, center_lat), 
                    cluster["mean_radius"]/111,  # Convert km to degrees
                    color=color, alpha=0.3, transform=ccrs.PlateCarree()
                )
                ax.add_patch(circle)
                
                # Add text
                plt.text(
                    center_lon, center_lat, 
                    f"Risk: {risk:.1%}\nMinT: {cluster['min_tb']:.1f}K",
                    fontsize=8, color='white', transform=ccrs.PlateCarree()
                )
            
            plt.title(f"Tropical Cloud Clusters - {timestamp.strftime('%Y-%m-%d %H:%M')}")
            
            # Save visualization
            vis_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, f"tcc_{timestamp.strftime('%Y%m%d%H%M')}.png")
            plt.savefig(vis_path, bbox_inches='tight', dpi=150)
            plt.close()
            logging.info(f"Saved visualization to {vis_path}")
            
        except Exception as e:
            logging.error(f"Visualization generation failed: {str(e)}")

# Main execution
if __name__ == "__main__":
    # Load configuration
    config_file = "tcc_config.yaml"
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = CONFIG
        # Save default config
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    # Get list of data files
    input_dir = config["data"]["input_dir"]
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        logging.warning(f"Input directory {input_dir} created. Please add data files.")
        exit(1)
        
    file_paths = sorted([
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.lower().endswith(('.nc', '.hdf', '.h5', '.nc4', '.h4'))
    ])
    
    if not file_paths:
        logging.error("No data files found in input directory")
        exit(1)
    
    # Process files
    processor = TCCProcessor(config)
    results = processor.process_files(file_paths)
    
    # Generate final report
    logging.info("Processing complete. Generating final report...")
    if not results.empty:
        print(results.describe())
        print(f"\nTotal tracks detected: {results['track_id'].nunique()}")
        print(f"Total observations: {len(results)}")
        print(f"Date range: {results['datetime'].min()} to {results['datetime'].max()}")
    else:
        print("No results generated") 