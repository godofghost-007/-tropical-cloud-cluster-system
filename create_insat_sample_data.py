#!/usr/bin/env python3
"""
Create sample INSAT-3D data files for testing TCC processor
Generates realistic synthetic satellite data in NetCDF format
"""

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
import os

def create_insat_sample_data():
    """Create sample INSAT-3D IRBRT data files"""
    
    # Create data directory if it doesn't exist
    data_dir = "data/insat_real"
    os.makedirs(data_dir, exist_ok=True)
    
    print("🌪️ Creating Sample INSAT-3D Data Files")
    print("=" * 50)
    
    # Define region (Bay of Bengal) - smaller grid for testing
    lat_range = np.arange(5, 25.1, 0.5)  # 5°N to 25°N, 0.5° resolution
    lon_range = np.arange(60, 90.1, 0.5)  # 60°E to 90°E, 0.5° resolution
    
    # Create time series (6-hour intervals for 2 days)
    start_time = datetime(2024, 7, 1, 0, 0)  # July 1, 2024
    time_steps = []
    for i in range(8):  # 2 days * 4 times per day
        time_steps.append(start_time + timedelta(hours=i*6))
    
    print(f"📅 Creating data for {len(time_steps)} time steps")
    print(f"🌍 Region: {lat_range[0]:.1f}°N to {lat_range[-1]:.1f}°N, {lon_range[0]:.1f}°E to {lon_range[-1]:.1f}°E")
    print(f"📊 Grid size: {len(lat_range)} x {len(lon_range)}")
    
    for i, timestamp in enumerate(time_steps):
        # Create filename with timestamp
        filename = f"INSAT3D_IRBRT_{timestamp.strftime('%Y%m%d%H%M')}.nc"
        filepath = os.path.join(data_dir, filename)
        
        # Create synthetic brightness temperature data
        # Base temperature field (warmer in tropics)
        lat_grid, lon_grid = np.meshgrid(lat_range, lon_range, indexing='ij')
        base_temp = 280 + 10 * np.cos(np.radians(lat_grid))
        
        # Add some convective regions (colder areas)
        convective_regions = []
        
        # Region 1: Active convection near 15°N, 75°E
        if i < 6:  # Active for first 1.5 days
            lat_center, lon_center = 15, 75
            lat_idx = np.argmin(np.abs(lat_range - lat_center))
            lon_idx = np.argmin(np.abs(lon_range - lon_center))
            
            # Create cold convective region
            for lat_offset in range(-4, 5):
                for lon_offset in range(-4, 5):
                    if 0 <= lat_idx + lat_offset < len(lat_range) and 0 <= lon_idx + lon_offset < len(lon_range):
                        distance = np.sqrt(lat_offset**2 + lon_offset**2)
                        if distance <= 3:
                            temp_reduction = 30 * np.exp(-distance/1.5)
                            convective_regions.append((lat_idx + lat_offset, lon_idx + lon_offset, temp_reduction))
        
        # Region 2: Developing convection near 12°N, 82°E
        if i >= 3:  # Starts developing on day 1
            lat_center, lon_center = 12, 82
            lat_idx = np.argmin(np.abs(lat_range - lat_center))
            lon_idx = np.argmin(np.abs(lon_range - lon_center))
            
            # Create smaller convective region
            for lat_offset in range(-3, 4):
                for lon_offset in range(-3, 4):
                    if 0 <= lat_idx + lat_offset < len(lat_range) and 0 <= lon_idx + lon_offset < len(lon_range):
                        distance = np.sqrt(lat_offset**2 + lon_offset**2)
                        if distance <= 2:
                            temp_reduction = 25 * np.exp(-distance/1.2) * (i - 2) / 3  # Gradually intensifying
                            convective_regions.append((lat_idx + lat_offset, lon_idx + lon_offset, temp_reduction))
        
        # Apply convective cooling
        tb_data = base_temp.copy()
        for lat_idx, lon_idx, temp_reduction in convective_regions:
            tb_data[lat_idx, lon_idx] -= temp_reduction
        
        # Add some noise
        noise = np.random.normal(0, 2, tb_data.shape)
        tb_data += noise
        
        # Ensure realistic temperature range (180-320K)
        tb_data = np.clip(tb_data, 180, 320)
        
        # Create xarray Dataset
        ds = xr.Dataset(
            {
                'IRBRT': xr.DataArray(
                    tb_data,
                    dims=['latitude', 'longitude'],
                    attrs={
                        'units': 'K',
                        'long_name': 'Brightness Temperature',
                        'standard_name': 'brightness_temperature'
                    }
                )
            },
            coords={
                'latitude': xr.DataArray(
                    lat_range,
                    dims=['latitude'],
                    attrs={'units': 'degrees_north', 'long_name': 'Latitude'}
                ),
                'longitude': xr.DataArray(
                    lon_range,
                    dims=['longitude'],
                    attrs={'units': 'degrees_east', 'long_name': 'Longitude'}
                )
            },
            attrs={
                'title': 'INSAT-3D IRBRT Data',
                'institution': 'ISRO',
                'source': 'INSAT-3D Satellite',
                'history': f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                'Conventions': 'CF-1.8'
            }
        )
        
        # Save to NetCDF file
        ds.to_netcdf(filepath, format='NETCDF4')
        
        print(f"✅ Created: {filename} ({tb_data.shape[0]}x{tb_data.shape[1]} grid)")
        
        # Print some statistics
        min_temp = np.min(tb_data)
        max_temp = np.max(tb_data)
        convective_pixels = np.sum(tb_data < 220)  # Convective threshold
        print(f"   Temperature range: {min_temp:.1f}K - {max_temp:.1f}K")
        print(f"   Convective pixels: {convective_pixels}")
    
    print(f"\n🎉 Successfully created {len(time_steps)} sample files in {data_dir}/")
    print("\n📋 File Information:")
    print("   - Format: NetCDF4")
    print("   - Variable: IRBRT (Brightness Temperature)")
    print("   - Units: Kelvin (K)")
    print("   - Region: Bay of Bengal (5-25°N, 60-90°E)")
    print("   - Resolution: 0.5° x 0.5°")
    print("   - Time interval: 6 hours")
    
    return data_dir

def create_metadata_file():
    """Create a metadata file describing the sample data"""
    
    metadata = """
# Sample INSAT-3D Data Description

## Overview
This directory contains synthetic INSAT-3D IRBRT (Infrared Brightness Temperature) data files
created for testing the Tropical Cloud Cluster (TCC) detection and tracking system.

## File Format
- **Format**: NetCDF4
- **Naming Convention**: INSAT3D_IRBRT_YYYYMMDDHHMM.nc
- **Example**: INSAT3D_IRBRT_202407010000.nc

## Data Structure
- **Variable**: IRBRT (Brightness Temperature)
- **Units**: Kelvin (K)
- **Dimensions**: [latitude, longitude]
- **Coordinate System**: Geographic (lat/lon)

## Geographic Coverage
- **Latitude**: 5°N to 25°N (Bay of Bengal region)
- **Longitude**: 60°E to 90°E
- **Resolution**: 0.5° x 0.5° (~55 km)
- **Grid Size**: 41 x 61 pixels

## Temporal Coverage
- **Start Date**: July 1, 2024, 00:00 UTC
- **End Date**: July 2, 2024, 18:00 UTC
- **Interval**: 6 hours
- **Total Files**: 8

## Synthetic Features
The data includes realistic convective regions:
1. **Primary Convection**: Near 15°N, 75°E (active for first 1.5 days)
2. **Secondary Convection**: Near 12°N, 82°E (developing on day 1)

## Usage
These files can be processed by the TCC processor:
```bash
python run_tcc_processor.py
```

## Real Data Replacement
Replace these files with actual INSAT-3D data when available.
Ensure real data follows the same naming convention and format.
"""
    
    metadata_file = "data/insat_real/README.md"
    with open(metadata_file, 'w') as f:
        f.write(metadata)
    
    print(f"📄 Created metadata file: {metadata_file}")

def main():
    """Main function to create sample data"""
    
    print("🚀 INSAT-3D Sample Data Generator")
    print("=" * 60)
    
    try:
        # Create sample data files
        data_dir = create_insat_sample_data()
        
        # Create metadata file
        create_metadata_file()
        
        print("\n" + "=" * 60)
        print("✅ Sample data creation completed successfully!")
        print("\n🎯 Next Steps:")
        print("   1. Run TCC processor: python run_tcc_processor.py")
        print("   2. Launch dashboard: streamlit run dashboard.py")
        print("   3. Replace with real INSAT-3D data when available")
        
    except Exception as e:
        print(f"❌ Error creating sample data: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 