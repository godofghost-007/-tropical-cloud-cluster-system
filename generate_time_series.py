"""
generate_time_series.py - Synthetic INSAT-3D Data Generation
"""

import numpy as np
import xarray as xr
import os
import argparse

def generate_time_series(num_files=6, output_dir="data/time_series", overwrite=False):
    """Create synthetic INSAT-3D-like data"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_files} timesteps...")
    
    # Create base grid
    lats = np.linspace(-30, 30, 600)
    lons = np.linspace(40, 100, 600)
    
    # Create moving cyclone-like cluster
    start_lat = 12  # Starting latitude
    start_lon = 88  # Starting longitude
    lat_step = 0.6  # Degrees per timestep
    lon_step = -1.2  # Degrees per timestep
    
    for t in range(num_files):
        file_name = f"timestep_{t:02d}.nc"
        file_path = os.path.join(output_dir, file_name)
        
        # Skip existing files if not overwriting
        if not overwrite and os.path.exists(file_path):
            print(f"Skipping existing file: {file_name}")
            continue
            
        print(f"Generating timestep {t}: {file_name}")
        
        # Base temperature field
        irbt = np.random.rand(600, 600) * 50 + 230  # 230-280K range
        
        # Calculate cyclone position
        cyclone_lat = start_lat + t * lat_step
        cyclone_lon = start_lon + t * lon_step
        
        # Convert to pixel coordinates
        lat_pixel = int((cyclone_lat - (-30)) / 60 * 600)
        lon_pixel = int((cyclone_lon - 40) / 60 * 600)
        
        # Create cyclone cluster (circular)
        size = 80
        temp = 185  # Very cold tops
        for y in range(max(0, lat_pixel-size), min(600, lat_pixel+size)):
            for x in range(max(0, lon_pixel-size), min(600, lon_pixel+size)):
                # Check if inside circle
                dy = (y - lat_pixel) / size
                dx = (x - lon_pixel) / size
                if dx*dx + dy*dy < 1:
                    irbt[y, x] = temp + np.random.rand() * 5
        
        # Create dataset
        ds = xr.Dataset(
            {'irbt': (['lat', 'lon'], irbt)},
            coords={'lat': lats, 'lon': lons}
        )
        
        # Save to file
        ds.to_netcdf(file_path)
    
    print("Synthetic data generation complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    args = parser.parse_args()
    
    generate_time_series(overwrite=args.overwrite) 