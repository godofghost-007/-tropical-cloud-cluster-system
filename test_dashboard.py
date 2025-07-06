#!/usr/bin/env python3
"""
Test script for the Streamlit dashboard
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✓ streamlit imported successfully")
    except ImportError as e:
        print(f"✗ streamlit import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✓ plotly imported successfully")
    except ImportError as e:
        print(f"✗ plotly import failed: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✓ pandas and numpy imported successfully")
    except ImportError as e:
        print(f"✗ pandas/numpy import failed: {e}")
        return False
    
    try:
        import yaml
        print("✓ pyyaml imported successfully")
    except ImportError as e:
        print(f"✗ pyyaml import failed: {e}")
        return False
    
    try:
        import psutil
        print("✓ psutil imported successfully")
    except ImportError as e:
        print(f"✗ psutil import failed: {e}")
        return False
    
    try:
        import folium
        from streamlit_folium import folium_static
        print("✓ folium and streamlit-folium imported successfully")
    except ImportError as e:
        print(f"✗ folium import failed: {e}")
        return False
    
    return True

def test_dashboard_class():
    """Test if the Dashboard class can be instantiated"""
    print("\nTesting Dashboard class...")
    
    try:
        from dashboard import Dashboard
        dashboard = Dashboard()
        print("✓ Dashboard class instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Dashboard class instantiation failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\nTesting data loading...")
    
    try:
        from dashboard import Dashboard
        dashboard = Dashboard()
        
        # Test tracks loading
        tracks_df = dashboard.load_tracks()
        print(f"✓ Tracks loaded: {len(tracks_df)} rows")
        
        # Test config loading
        config = dashboard.load_config()
        print("✓ Config loaded successfully")
        
        # Test system stats
        stats = dashboard.get_system_stats()
        print(f"✓ System stats: CPU={stats['cpu']}%, Memory={stats['memory']}%")
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Dashboard Test Suite ===\n")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install missing dependencies.")
        return False
    
    # Test dashboard class
    if not test_dashboard_class():
        print("\n❌ Dashboard class test failed.")
        return False
    
    # Test data loading
    if not test_data_loading():
        print("\n❌ Data loading test failed.")
        return False
    
    print("\n✅ All tests passed! Dashboard is ready to run.")
    print("\nTo start the dashboard, run:")
    print("streamlit run dashboard.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 