# setup_hdf4.py
import os
import sys
import platform
from pathlib import Path

def fix_hdf4_dll():
    """Resolve HDF4 DLL issues on Windows"""
    try:
        from pyhdf.SD import SD, SDC
        print("✅ HDF4 libraries are working properly!")
        return True
    except OSError as e:
        if "mfhdf" in str(e) or "hdf" in str(e):
            print("⚠️ HDF4 DLL issue detected. Attempting fix...")
            
            # Find the DLL directory in site-packages
            lib_dir = Path(sys.prefix) / "Library" / "bin"
            
            if not lib_dir.exists():
                print(f"❌ Library directory not found: {lib_dir}")
                return False
            
            # Add to PATH
            os.environ['PATH'] = f"{lib_dir}{os.pathsep}{os.environ['PATH']}"
            sys.path.append(str(lib_dir))
            
            print(f"✅ Added to PATH: {lib_dir}")
            return True
        return False
    except ImportError:
        print("❌ pyhdf module not installed. Please install with:")
        print("pip install pyhdf")
        return False

def check_hdf4_installation():
    """Check if HDF4 is properly installed"""
    try:
        import pyhdf
        print("pyhdf is installed")
        return True
    except ImportError:
        print("pyhdf not installed. Installing...")
        return False

def install_hdf4():
    """Install pyhdf with proper dependencies"""
    import subprocess
    
    print("Installing pyhdf...")
    try:
        # First try to uninstall if exists
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "pyhdf", "-y"], 
                      capture_output=True)
        
        # Install with no cache
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "pyhdf"],
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("pyhdf installed successfully!")
            return True
        else:
            print(f"Installation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Installation error: {str(e)}")
        return False

def verify_hdf4_functionality():
    """Test HDF4 functionality with a simple operation"""
    try:
        from pyhdf.SD import SD, SDC
        print("HDF4 import successful!")
        
        # Try to create a simple dataset
        test_file = "test_hdf4.hdf"
        hdf = SD(test_file, SDC.WRITE | SDC.CREATE)
        
        # Create a simple dataset
        data = [[1, 2, 3], [4, 5, 6]]
        dataset = hdf.create("test_data", SDC.INT8, (2, 3))
        dataset.set(data)
        
        # Close and reopen to test reading
        hdf.end()
        
        hdf_read = SD(test_file, SDC.READ)
        dataset_read = hdf_read.select("test_data")
        data_read = dataset_read.get()
        
        hdf_read.end()
        
        # Clean up
        os.remove(test_file)
        
        print("HDF4 read/write test successful!")
        return True
        
    except Exception as e:
        print(f"HDF4 functionality test failed: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("="*50)
    print(f"Running HDF4 Diagnostic on {platform.system()} {platform.release()}")
    print("="*50)
    
    # Check if pyhdf is installed
    if not check_hdf4_installation():
        if not install_hdf4():
            print("Failed to install pyhdf. Please install manually:")
            print("pip install --no-cache-dir pyhdf")
            return False
    
    # Try to fix DLL issues
    if not fix_hdf4_dll():
        print("No DLL issues detected or fix failed")
    
    # Verify functionality
    if verify_hdf4_functionality():
        print("✅ HDF4 setup completed successfully!")
        
        # Test with sample file if available
        test_file = Path("test.hdf")
        if test_file.exists():
            print(f"Testing with {test_file}")
            try:
                from pyhdf.SD import SD, SDC
                hdf = SD(str(test_file), SDC.READ)
                print(f"File contains {len(hdf.datasets())} datasets")
            except Exception as e:
                print(f"Test file reading failed: {str(e)}")
        else:
            print("ℹ️ Create a 'test.hdf' file for full verification")
        
        return True
    else:
        print("❌ HDF4 setup failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nTroubleshooting tips:")
        print("1. Try reinstalling pyhdf: pip uninstall pyhdf && pip install --no-cache-dir pyhdf")
        print("2. Install HDF4 libraries manually from: https://www.hdfgroup.org/downloads/")
        print("3. Add HDF4 bin directory to your PATH environment variable")
        print("4. On Windows, you may need to install Visual C++ Redistributable")
        sys.exit(1)
    
    print("\nNext steps:")
    print("1. Place satellite data files in data/insat_real/")
    print("2. Run: python real_data_processor.py") 