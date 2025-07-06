"""
enhanced_processor.py - Enhanced Satellite Data Processing
Uses unified data loader for HDF/NetCDF support and advanced convective metrics
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import binary_closing, disk
from skimage.measure import label, regionprops
from data_loader import load_satellite_data, extract_tb, tb_to_height, get_coordinates
from helpers import haversine
import warnings

class EnhancedProcessor:
    """Enhanced processor for satellite data with multi-format support"""
    
    def __init__(self, config=None):
        self.config = config or {
            "thresholds": {
                "tb_cold": 220,  # Kelvin - cold cloud threshold
                "tb_very_cold": 200,  # Kelvin - very cold cloud threshold
                "min_area": 34800,  # km² minimum cluster area
                "pixel_resolution": 4  # km/pixel
            },
            "output_dir": "outputs/enhanced",
            "visualization": {
                "cmap": "viridis",
                "marker_size": 50
            }
        }
        
        os.makedirs(self.config["output_dir"], exist_ok=True)
    
    def process_file(self, file_path, tb_threshold=None):
        """
        Process a single satellite data file
        
        Args:
            file_path: Path to satellite data file
            tb_threshold: Optional custom brightness temperature threshold
            
        Returns:
            dict: Processing results and metrics
        """
        print(f"Processing: {os.path.basename(file_path)}")
        
        try:
            # Load data using unified loader
            ds = load_satellite_data(file_path)
            tb = extract_tb(ds)
            lats, lons = get_coordinates(ds)
            
            # Calculate cloud heights
            heights = tb_to_height(tb)
            
            # Use provided threshold or default
            threshold = tb_threshold or self.config["thresholds"]["tb_cold"]
            
            # Calculate convective metrics
            metrics = self._calculate_convective_metrics(tb, heights, threshold)
            
            # Detect cloud clusters
            clusters = self._detect_clusters(tb, lats, lons, threshold)
            
            # Enhanced cluster properties
            enhanced_clusters = self._calculate_enhanced_properties(clusters, tb, heights, lats, lons)
            
            results = {
                'file_path': file_path,
                'metrics': metrics,
                'clusters': enhanced_clusters,
                'dataset_info': {
                    'shape': tb.shape,
                    'format': ds.attrs.get('source_format', 'unknown'),
                    'variables': list(ds.data_vars.keys())
                }
            }
            
            # Save results
            self._save_results(results, file_path)
            
            return results
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
    
    def _calculate_convective_metrics(self, tb, heights, threshold):
        """Calculate comprehensive convective metrics"""
        # Create masks
        cold_mask = tb < threshold
        very_cold_mask = tb < self.config["thresholds"]["tb_very_cold"]
        
        # Basic statistics
        cold_pixels = tb[cold_mask]
        very_cold_pixels = tb[very_cold_mask]
        
        metrics = {
            'min_tb': np.min(cold_pixels) if len(cold_pixels) > 0 else np.nan,
            'mean_tb': np.mean(cold_pixels) if len(cold_pixels) > 0 else np.nan,
            'std_tb': np.std(cold_pixels) if len(cold_pixels) > 0 else np.nan,
            'pixel_count_cold': np.sum(cold_mask),
            'pixel_count_very_cold': np.sum(very_cold_mask),
            'coverage_fraction_cold': np.sum(cold_mask) / tb.size,
            'coverage_fraction_very_cold': np.sum(very_cold_mask) / tb.size,
            'mean_height_cold': np.mean(heights[cold_mask]) if np.sum(cold_mask) > 0 else np.nan,
            'max_height': np.nanmax(heights),
            'height_std': np.nanstd(heights)
        }
        
        # Advanced metrics
        if len(cold_pixels) > 0:
            # Convective intensity (gradient-based)
            tb_gradient = np.gradient(tb)
            convective_intensity = np.mean(np.abs(tb_gradient[cold_mask]))
            metrics['convective_intensity'] = convective_intensity
            
            # Temperature gradient at cold cloud boundaries
            from scipy.ndimage import binary_erosion
            cold_boundary = cold_mask & ~binary_erosion(cold_mask)
            if np.sum(cold_boundary) > 0:
                boundary_gradient = np.mean(np.abs(tb_gradient[cold_boundary]))
                metrics['boundary_gradient'] = boundary_gradient
        
        return metrics
    
    def _detect_clusters(self, tb, lats, lons, threshold):
        """Detect cloud clusters using morphological operations"""
        # Create cloud mask
        cloud_mask = (tb < threshold)
        
        # Clean mask with morphological closing
        cloud_mask_clean = binary_closing(cloud_mask, footprint=disk(3))
        
        # Identify connected components
        label_image = label(cloud_mask_clean)
        regions = regionprops(label_image, intensity_image=tb)
        
        return regions
    
    def _calculate_enhanced_properties(self, regions, tb, heights, lats, lons):
        """Calculate enhanced properties for detected clusters"""
        min_pixels = self.config["thresholds"]["min_area"] / (self.config["thresholds"]["pixel_resolution"]**2)
        properties = []
        
        for region in regions:
            if region.area < min_pixels:
                continue
            
            # Basic properties
            cy, cx = map(int, region.centroid)
            lat_center = lats[cy] if cy < len(lats) else lats[-1]
            lon_center = lons[cx] if cx < len(lons) else lons[-1]
            
            # Temperature properties
            min_tb = region.min_intensity
            mean_tb = region.mean_intensity
            median_tb = np.median(region.intensity_image[region.image])
            
            # Height properties
            region_heights = heights[region.coords[:, 0], region.coords[:, 1]]
            mean_height = np.nanmean(region_heights)
            max_height = np.nanmax(region_heights)
            height_std = np.nanstd(region_heights)
            
            # Size and shape properties
            area_km2 = region.area * (self.config["thresholds"]["pixel_resolution"]**2)
            perimeter = region.perimeter * self.config["thresholds"]["pixel_resolution"]
            compactness = (4 * np.pi * region.area) / (perimeter ** 2) if perimeter > 0 else 0
            
            # Advanced properties
            # Convective intensity (temperature gradient)
            tb_gradient = np.gradient(region.intensity_image)
            convective_intensity = np.mean(np.abs(tb_gradient))
            
            # Cloud-top height variability
            cloud_tops = tb_to_height(region.intensity_image)
            std_cloud_height = np.nanstd(cloud_tops)
            
            # Cyclogenesis potential indicators
            cyclogenesis_score = self._calculate_cyclogenesis_potential(
                min_tb, mean_height, area_km2, convective_intensity
            )
            
            properties.append({
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                'center_lat': lat_center,
                'center_lon': lon_center,
                'pixel_count': region.area,
                'area_km2': area_km2,
                'perimeter_km': perimeter,
                'compactness': compactness,
                'min_tb': min_tb,
                'mean_tb': mean_tb,
                'median_tb': median_tb,
                'mean_height_km': mean_height,
                'max_height_km': max_height,
                'height_std_km': height_std,
                'convective_intensity': convective_intensity,
                'std_cloud_height_km': std_cloud_height,
                'cyclogenesis_score': cyclogenesis_score
            })
        
        return properties
    
    def _calculate_cyclogenesis_potential(self, min_tb, mean_height, area_km2, convective_intensity):
        """Calculate cyclogenesis potential score"""
        # Simple scoring system (0-100)
        score = 0
        
        # Temperature factor (colder = higher score)
        if min_tb < 200:
            score += 30
        elif min_tb < 220:
            score += 20
        elif min_tb < 240:
            score += 10
        
        # Height factor (higher = higher score)
        if mean_height > 10:
            score += 25
        elif mean_height > 8:
            score += 15
        elif mean_height > 6:
            score += 10
        
        # Size factor (larger = higher score)
        if area_km2 > 100000:  # 100k km²
            score += 25
        elif area_km2 > 50000:  # 50k km²
            score += 15
        elif area_km2 > 25000:  # 25k km²
            score += 10
        
        # Convective intensity factor
        if convective_intensity > 5:
            score += 20
        elif convective_intensity > 3:
            score += 10
        
        return min(score, 100)
    
    def _save_results(self, results, file_path):
        """Save processing results"""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Save cluster properties
        if results['clusters']:
            df = pd.DataFrame(results['clusters'])
            output_path = os.path.join(self.config["output_dir"], f"{base_name}_clusters.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved {len(results['clusters'])} clusters to {output_path}")
        
        # Save metrics
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_path = os.path.join(self.config["output_dir"], f"{base_name}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics to {metrics_path}")
    
    def process_directory(self, input_dir, output_dir=None, force=False):
        """Process all files in a directory"""
        if output_dir:
            self.config["output_dir"] = output_dir
            os.makedirs(output_dir, exist_ok=True)
        
        files = [f for f in os.listdir(input_dir) 
                if f.endswith(('.nc', '.nc4', '.h5', '.hdf', '.h4'))]
        
        all_results = []
        for file in files:
            file_path = os.path.join(input_dir, file)
            
            # Check if already processed
            base_name = os.path.splitext(file)[0]
            output_file = os.path.join(self.config["output_dir"], f"{base_name}_clusters.csv")
            
            if not force and os.path.exists(output_file):
                print(f"Skipping {file} (already processed)")
                continue
            
            result = self.process_file(file_path)
            if result:
                all_results.append(result)
        
        return all_results

def main():
    """Example usage"""
    processor = EnhancedProcessor()
    
    # Process a single file
    # result = processor.process_file("data/sample.nc")
    
    # Process a directory
    results = processor.process_directory("data/time_series", force=True)
    
    print(f"Processed {len(results)} files")

if __name__ == "__main__":
    main() 