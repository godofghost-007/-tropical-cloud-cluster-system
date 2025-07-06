# test_data_loader.py
import os
import sys
import numpy as np
from data_loader import load_satellite_data

def create_test_file(file_path):
    """Create a test NetCDF file"""
    import xarray as xr
    
    data = np.random.rand(100, 100) * 50 + 200  # 200-250K brightness temp
    ds = xr.Dataset(
        data_vars={
            'IRBRT': (['lat', 'lon'], data),
        },
        coords={
            'lat': np.linspace(5, 25, 100),
            'lon': np.linspace(60, 90, 100),
        }
    )
    ds.to_netcdf(file_path)
    print(f"Created test file: {file_path}")

if __name__ == "__main__":
    # Create test directory
    test_dir = "data/insat_real"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create test file
    test_path = os.path.join(test_dir, "test_file.nc")
    create_test_file(test_path)
    
    # Test loading
    print("\nTesting data loader...")
    try:
        ds = load_satellite_data(test_path)
        print("✅ Data loaded successfully!")
        print(f"Variables: {list(ds.data_vars)}")
        print(f"Dimensions: {dict(ds.dims)}")
        print(f"Data shape: {ds['IRBRT'].shape}")
        print(f"Temperature range: {ds['IRBRT'].min().values:.1f}K - {ds['IRBRT'].max().values:.1f}K")
    except Exception as e:
        print(f"❌ Load failed: {str(e)}")
        sys.exit(1)
    
    print("\n✅ Test completed successfully!")
    print("\nNext: Run the full processor")
    print("python real_data_processor.py") 