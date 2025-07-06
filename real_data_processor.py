"""
real_data_processor.py - INSAT-3D Data Processing Pipeline
Enhanced with parallel processing and robust error handling
"""

import yaml
import xarray as xr
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from detection import main as detect_clusters
from tracking import track_clusters
from data_loader import load_satellite_data, extract_metadata, get_file_format_info, get_data_quality_score
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('processing.log')
    ]
)

# Add HDF4 DLL directory to PATH (Windows fix)
sys.path.append(os.path.join(sys.prefix, "Library", "bin"))
os.environ['PATH'] = f"{os.path.join(sys.prefix, 'Library', 'bin')};{os.environ['PATH']}"

def load_config(config_path='real_data_config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Create default config if file doesn't exist
        default_config = {
            'data': {
                'input_dir': 'data/insat_real',
                'output_dir': 'outputs',
                'thresholds': {
                    'irbt': 240,  # K
                    'min_area': 100  # pixels
                }
            },
            'processing': {
                'parallel_workers': 'auto',
                'chunk_size': 1000
            }
        }
        
        # Save default config
        config_dir = os.path.dirname(config_path)
        if config_dir:  # Only create directory if path is not empty
            os.makedirs(config_dir, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        logging.info(f"Created default config: {config_path}")
        return default_config

def verify_file(file_path):
    """Verify file integrity and format"""
    try:
        if file_path.endswith(('.nc', '.nc4')):
            with xr.open_dataset(file_path) as ds:
                return bool(ds.data_vars)
        elif file_path.endswith('.h5'):
            import h5py
            with h5py.File(file_path, 'r') as f:
                return bool(f.keys())
        elif file_path.endswith(('.hdf', '.h4')):
            try:
                from pyhdf.SD import SD, SDC
                hdf = SD(file_path, SDC.READ)
                return bool(hdf.datasets().keys())
            except ImportError:
                logging.warning("HDF4 support not available")
                return False
        return False
    except Exception as e:
        logging.warning(f"File verification failed for {file_path}: {str(e)}")
        return False

def extract_metadata(ds):
    """Extract critical metadata from various formats"""
    meta = {}
    
    # NetCDF attributes
    if hasattr(ds, 'attrs'):
        meta.update(ds.attrs)
    
    # HDF4 special handling
    if 'HDF4' in str(type(ds)):
        meta['sensor'] = ds.attrs.get('Sensor_Name', 'Unknown')
        meta['time'] = datetime.utcfromtimestamp(ds.attrs.get('Start_Time', 0))
    
    # Common satellite metadata
    meta['resolution'] = ds.attrs.get('spatial_resolution', '4km')
    meta['format_type'] = ds.attrs.get('source_format', 'unknown')
    meta['format_version'] = ds.attrs.get('version', '1.0')
    
    return meta

def process_single_file(file_path, config):
    """Process a single satellite data file with enhanced error handling"""
    start_time = datetime.now()
    
    try:
        logging.info(f"Processing {os.path.basename(file_path)}")
        
        # Verify file integrity
        if not verify_file(file_path):
            logging.error(f"File verification failed: {file_path}")
            return None
        
        # Load data using unified loader
        ds = load_satellite_data(file_path)
        
        # Extract metadata
        metadata = extract_metadata(ds)
        
        # Preprocess and save
        processed_path = os.path.join(config['data']['output_dir'], f"proc_{os.path.basename(file_path)}")
        ds.to_netcdf(processed_path)
        
        # Detect clusters
        detect_clusters(processed_path)
        
        # Collect results with metadata
        csv_path = "outputs/cloud_clusters.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['source_file'] = os.path.basename(file_path)
            df['file_format'] = metadata.get('format_type', 'unknown')
            df['data_version'] = metadata.get('format_version', '1.0')
            df['processing_time'] = (datetime.now() - start_time).total_seconds()
            df['data_quality'] = 1.0  # Can be enhanced with quality metrics
            
            # Add metadata columns
            for key, value in metadata.items():
                if key not in df.columns:
                    df[f'meta_{key}'] = value
            
            logging.info(f"Detected {len(df)} clusters in {os.path.basename(file_path)}")
            return df
        else:
            logging.warning(f"No detection output found for {file_path}")
            return None
            
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return None

def process_file(file_path):
    """Process a single satellite file"""
    try:
        logging.info(f"Processing {os.path.basename(file_path)}")
        ds = load_satellite_data(file_path)
        
        # Extract brightness temperature data
        tb_data = None
        for candidate in ['IRBRT', 'brightness_temperature', 'Tb', 'temp_11um']:
            if candidate in ds:
                tb_data = ds[candidate].values
                break
        
        if tb_data is None:
            raise ValueError("Brightness temperature field not found in dataset")
        
        # Apply thresholds from config
        config = load_config()
        tb_threshold = config['data']['thresholds']['irbt']
        min_area = config['data']['thresholds']['min_area']
        
        # Create cold cloud mask
        cold_mask = tb_data < tb_threshold
        
        # Find connected components (cloud clusters)
        from scipy.ndimage import label
        structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
        labeled_array, num_features = label(cold_mask, structure=structure)
        
        logging.info(f"Found {num_features} potential cloud clusters in {os.path.basename(file_path)}")
        
        # Process each cluster
        clusters_found = 0
        for cluster_id in range(1, num_features + 1):
            cluster_mask = labeled_array == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            # Skip small clusters
            if cluster_size < min_area:
                continue
                
            # Extract cluster properties
            cluster_tb = tb_data[cluster_mask]
            min_tb = np.min(cluster_tb)
            mean_tb = np.mean(cluster_tb)
            
            # Find cluster centroid
            y_idx, x_idx = np.where(cluster_mask)
            center_y = np.mean(y_idx)
            center_x = np.mean(x_idx)
            
            # Convert pixel coordinates to lat/lon
            lat, lon = None, None
            if 'latitude' in ds and 'longitude' in ds:
                try:
                    # Handle 2D coordinate arrays
                    if len(ds.latitude.dims) == 2:
                        lat = ds.latitude.values[int(center_y), int(center_x)]
                        lon = ds.longitude.values[int(center_y), int(center_x)]
                    else:
                        # Handle 1D coordinate arrays
                        lat = ds.latitude.values[int(center_y)]
                        lon = ds.longitude.values[int(center_x)]
                except:
                    lat, lon = None, None
            elif 'lat' in ds and 'lon' in ds:
                try:
                    # Handle 2D coordinate arrays
                    if len(ds.lat.dims) == 2:
                        lat = ds.lat.values[int(center_y), int(center_x)]
                        lon = ds.lon.values[int(center_y), int(center_x)]
                    else:
                        # Handle 1D coordinate arrays
                        lat = ds.lat.values[int(center_y)]
                        lon = ds.lon.values[int(center_x)]
                except:
                    lat, lon = None, None
            
            clusters_found += 1
            logging.info(f"Cluster {cluster_id}: Center at ({lat:.2f}, {lon:.2f}), Min Tb: {min_tb:.1f}K, Size: {cluster_size} pixels")
        
        logging.info(f"Processed {clusters_found} valid clusters from {os.path.basename(file_path)}")
        return True
    
    except Exception as e:
        logging.error(f"Failed to process {file_path}: {str(e)}")
        return False

def process_real_data():
    """Main processing function"""
    config = load_config()
    input_dir = config['data']['input_dir']
    
    # Create input directory if it doesn't exist
    os.makedirs(input_dir, exist_ok=True)
    
    # Get supported files with case-insensitive check
    supported_ext = ('.nc', '.nc4', '.h5', '.hdf', '.h4')
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
             if os.path.splitext(f)[1].lower() in supported_ext]
    
    logging.info(f"Found {len(files)} files to process")
    
    if not files:
        logging.warning("No supported files found in directory!")
        logging.info(f"Please place satellite data files in: {input_dir}")
        return
    
    # Calculate workers (min of files count and CPU cores)
    max_workers = min(os.cpu_count(), len(files)) or 1  # Ensure at least 1 worker
    logging.info(f"Using {max_workers} parallel workers")
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_file, files))
        
        # Count successful results
        successful = sum(1 for r in results if r)
        logging.info(f"Successfully processed {successful}/{len(files)} files")
        
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise

def process_real_data_sequential(config=None):
    """Sequential processing version for debugging"""
    if config is None:
        config = load_config()
    
    input_dir = config['data']['input_dir']
    output_dir = config['data']['output_dir']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get supported files
    supported_formats = tuple(config['data']['formats'])
    files = [f for f in os.listdir(input_dir) 
             if f.lower().endswith(supported_formats)]
    
    logging.info(f"Processing {len(files)} files sequentially")
    
    if len(files) == 0:
        logging.warning("No files found to process")
        return
    
    all_detections = []
    for file in files:
        file_path = os.path.join(input_dir, file)
        result = process_single_file(file_path, config)
        if result is not None:
            all_detections.append(result)
    
    # Combine results
    if all_detections:
        combined = pd.concat(all_detections, ignore_index=True)
        combined.to_csv("outputs/all_detections.csv", index=False)
        logging.info(f"Combined {len(combined)} total detections")
        
        # Run tracking
        logging.info("Running tracking on all detections...")
        try:
            track_clusters()
            logging.info("Tracking completed successfully")
        except Exception as e:
            logging.error(f"Tracking failed: {str(e)}")
    else:
        logging.warning("No detections found in any file")
    
    logging.info("Sequential processing complete")

if __name__ == "__main__":
    # Use parallel processing by default, sequential for debugging
    if len(sys.argv) > 1 and sys.argv[1] == '--sequential':
        process_real_data_sequential()
    else:
        process_real_data() 