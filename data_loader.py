"""
data_loader.py - Unified Satellite Data Loader
Handles HDF4, HDF5, and NetCDF formats for satellite data
"""

import os
import numpy as np
import xarray as xr
import h5py
import warnings
from datetime import datetime

# Explicitly check for pyhdf availability
try:
    from pyhdf.SD import SD, SDC
    HAS_HDF4 = True
except ImportError:
    HAS_HDF4 = False
    warnings.warn("pyhdf not available. HDF4 files will not be supported.")

def load_satellite_data(file_path):
    """
    Load satellite data from HDF4/HDF5/NetCDF formats
    
    Args:
        file_path: Path to satellite data file
        
    Returns:
        xarray.Dataset: Loaded dataset with coordinates and attributes
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in ('.nc', '.nc4'):
        return xr.open_dataset(file_path)  # NetCDF
    
    elif file_ext == '.h5':
        with h5py.File(file_path, 'r') as f:
            return xr.open_dataset(xr.backends.H5netcdfStore(f))  # HDF5
    
    elif file_ext in ('.hdf', '.h4'):
        if not HAS_HDF4:
            raise ImportError("HDF4 support requires pyhdf. Please install it.")
        
        hdf = SD(file_path, SDC.READ)
        ds = xr.Dataset()
        
        # Handle global attributes
        for attr_name in hdf.attributes():
            ds.attrs[attr_name] = hdf.attr(attr_name).get()
        
        # Handle variables
        for var_name in list(hdf.datasets().keys()):
            var = hdf.select(var_name)
            data = var.get()
            
            # Apply scaling if available
            if hasattr(var, 'scale_factor'):
                data = data * var.scale_factor
            if hasattr(var, 'add_offset'):
                data = data + var.add_offset
            
            # Add dimension names if available
            dim_names = []
            for i, dim in enumerate(var.dimensions()):
                dim_names.append(dim.name if hasattr(dim, 'name') else f"dim_{i}")
            
            ds[var_name] = xr.DataArray(data, dims=dim_names)
            
            # Add variable attributes
            var_attrs = {}
            for attr_name in dir(var):
                if not attr_name.startswith('_') and attr_name not in ['get', 'select']:
                    var_attrs[attr_name] = getattr(var, attr_name)
            ds[var_name].attrs = var_attrs
            
        return ds
    
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def extract_brightness_temperature(ds):
    """Extract brightness temperature data from various satellite formats"""
    # Common variable names for brightness temperature
    tb_vars = ['brightness_temperature', 'tb', 'brightness_temp', 'temperature', 'tbb', 'IRBRT', 'Tb', 'temp_11um']
    
    for var_name in tb_vars:
        if var_name in ds.data_vars:
            return ds[var_name]
    
    # If no standard name found, return the first variable
    if len(ds.data_vars) > 0:
        return list(ds.data_vars.values())[0]
    
    raise ValueError("No brightness temperature data found in dataset")

def convert_to_cloud_height(tb_data):
    """Convert brightness temperature to cloud top height"""
    # Simple conversion based on typical IR brightness temperature
    # This is a simplified model - real conversion depends on atmospheric conditions
    cloud_height = 12.0 - (tb_data - 200) / 10.0  # km
    return cloud_height.clip(min=0, max=20)  # Limit to reasonable range

def get_coordinates(ds):
    """Extract coordinate information from dataset"""
    coords = {}
    
    # Common coordinate names
    coord_names = {
        'latitude': ['lat', 'latitude', 'y'],
        'longitude': ['lon', 'longitude', 'x'],
        'time': ['time', 'timestamp', 'date']
    }
    
    for coord_type, possible_names in coord_names.items():
        for name in possible_names:
            if name in ds.coords:
                coords[coord_type] = ds[name]
                break
    
    return coords

def get_data_quality_score(ds):
    """Calculate data quality score based on various metrics"""
    score = 1.0
    
    # Check for missing data
    if hasattr(ds, 'attrs') and 'missing_value' in ds.attrs:
        missing_ratio = (ds == ds.attrs['missing_value']).sum() / ds.size
        score -= missing_ratio * 0.5
    
    # Check for valid range
    if hasattr(ds, 'attrs') and 'valid_range' in ds.attrs:
        valid_range = ds.attrs['valid_range']
        if len(valid_range) == 2:
            valid_ratio = ((ds >= valid_range[0]) & (ds <= valid_range[1])).sum() / ds.size
            score *= valid_ratio
    
    return max(0.0, min(1.0, score))

def get_file_format_info(file_path):
    """Get format information for a file"""
    if file_path.endswith(('.nc', '.nc4')):
        return {'format': 'NetCDF', 'version': '4'}
    elif file_path.endswith('.h5'):
        return {'format': 'HDF5', 'version': '5'}
    elif file_path.endswith(('.hdf', '.h4')):
        return {'format': 'HDF4', 'version': '4'}
    else:
        return {'format': 'Unknown', 'version': 'Unknown'}

def extract_tb(ds):
    """Extract brightness temperature from dataset"""
    # Try common variable names
    tb_vars = ['IRBRT', 'brightness_temperature', 'Tb', 'temp_11um', 'tb']
    
    for var_name in tb_vars:
        if var_name in ds.data_vars:
            return ds[var_name].values
    
    # If no standard name found, return the first variable
    if len(ds.data_vars) > 0:
        return list(ds.data_vars.values())[0].values
    
    raise ValueError("No brightness temperature data found")

def tb_to_height(tb_data):
    """Convert brightness temperature to cloud top height"""
    # Simple conversion: colder = higher clouds
    height = 12.0 - (tb_data - 200) / 10.0  # km
    return np.clip(height, 0, 20)  # Limit to reasonable range

def load_data_chunked(file_path, chunks=None):
    """
    Load satellite data with chunking for large files
    
    Args:
        file_path: Path to satellite data file
        chunks: Dictionary of chunk sizes for each dimension
        
    Returns:
        xarray.Dataset: Loaded dataset
    """
    if chunks is None:
        chunks = {'lat': 100, 'lon': 100}
    
    return xr.open_dataset(file_path, chunks=chunks)

def get_file_info(file_path):
    """
    Get basic information about a satellite data file
    
    Args:
        file_path: Path to satellite data file
        
    Returns:
        dict: File information
    """
    info = {
        'file_path': file_path,
        'file_size': os.path.getsize(file_path),
        'format': os.path.splitext(file_path)[1].lower()
    }
    
    try:
        ds = load_satellite_data(file_path)
        info['variables'] = list(ds.data_vars.keys())
        info['coordinates'] = list(ds.coords.keys())
        info['attributes'] = dict(ds.attrs)
        
        # Try to get data shape
        if ds.data_vars:
            first_var = list(ds.data_vars.keys())[0]
            info['data_shape'] = ds[first_var].shape
        
    except Exception as e:
        info['error'] = str(e)
    
    return info

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

def is_hdf4_supported():
    """Check if HDF4 support is available"""
    return HAS_HDF4 