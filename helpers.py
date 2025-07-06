"""
helpers.py - Geospatial Utilities for Tropical Cloud Cluster Project
"""

import numpy as np
from math import radians, sin, cos, sqrt, atan2
import xarray as xr

def haversine(lon1, lat1, lon2, lat2, units='km'):
    """
    Calculate great-circle distance between two points on Earth.
    """
    # Validate input coordinates
    if not (-180 <= lon1 <= 180) or not (-180 <= lon2 <= 180):
        raise ValueError("Longitude must be between -180 and 180 degrees")
    if not (-90 <= lat1 <= 90) or not (-90 <= lat2 <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees")
    
    # Convert to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Earth radius in km
    r = 6371
    
    # Convert to requested units
    if units == 'miles':
        return r * c * 0.621371
    return r * c

def haversine_vector(points1, points2, units='km'):
    """
    Vectorized haversine distance for multiple points.
    """
    # Convert to radians
    points1 = np.radians(points1)
    points2 = np.radians(points2)
    
    # Extract coordinates
    lon1 = points1[:, 0]
    lat1 = points1[:, 1]
    lon2 = points2[:, 0]
    lat2 = points2[:, 1]
    
    # Calculate differences
    dlon = lon2[:, None] - lon1
    dlat = lat2[:, None] - lat1
    
    # Haversine formula (vectorized)
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2[:, None]) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Earth radius and unit conversion
    r = 6371  # km
    distances = r * c
    if units == 'miles':
        distances *= 0.621371
        
    return distances

# FIXED VERSION OF km_to_degrees
def km_to_degrees(km, latitude=0):
    """
    Convert kilometers to approximate degrees at given latitude.
    """
    # Earth's radius at equator (km)
    r_equator = 6378.137
    # Earth's radius at poles (km)
    r_polar = 6356.752
    
    # Calculate radius at given latitude (FIXED PARENTHESIS)
    lat_rad = radians(latitude)
    r = sqrt(
        ((r_equator**2 * cos(lat_rad))**2 + 
         (r_polar**2 * sin(lat_rad))**2) 
        / 
        ((r_equator * cos(lat_rad))**2 + 
         (r_polar * sin(lat_rad))**2)
    )
    
    # Circumference at latitude
    circumference = 2 * np.pi * r
    km_per_degree = circumference / 360
    
    return km / km_per_degree

def load_data(file_path=None):
    """
    Load satellite data from NetCDF file or generate synthetic data
    Returns: (lats, lons, irbt_data)
    """
    try:
        # Use provided file_path if given, else use CONFIG["data_path"]
        data_path = file_path if file_path is not None else CONFIG["data_path"]
        ds = xr.open_dataset(data_path)
        irbt = ds['irbt']
        return (
            irbt.lat.values,
            irbt.lon.values,
            irbt.values
        )
    except (FileNotFoundError, OSError, KeyError):
        print("Using synthetic data - real file not found or invalid")
        return generate_synthetic_data()

if __name__ == "__main__":
    # Test cases
    print("Testing haversine function:")
    print(f"New York to LA: {haversine(-74, 40, -118, 34, units='miles'):.1f} miles")
    
    print("\nTesting vectorized version:")
    ny = np.array([[-74, 40]])
    la = np.array([[-118, 34]])
    print(f"Vectorized distance: {haversine_vector(ny, la)[0][0]:.1f} km")
    
    print("\nTesting km to degrees conversion:")
    print(f"100 km at equator: {km_to_degrees(100):.2f}°")
    print(f"100 km at 45°: {km_to_degrees(100, 45):.2f}°")

    # Add this check before calling visualize_results
    if irbt_data.ndim == 1:
        irbt_data = irbt_data.reshape((len(lats), len(lons)))

    print("irbt_data shape:", irbt_data.shape)
    print("lons shape:", lons.shape)
    print("lats shape:", lats.shape)