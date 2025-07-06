#!/usr/bin/env python3
"""
Dashboard Launcher Script
Provides an easy way to start the Tropical Cloud Cluster Monitor Dashboard
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit', 'plotly', 'pandas', 'numpy', 
        'pyyaml', 'psutil', 'folium', 'streamlit_folium'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("Please run: pip install -r dashboard_requirements.txt")
            return False
    
    return True

def check_data_files():
    """Check if required data files exist"""
    required_paths = [
        'outputs/tracks/final_tracks.csv',
        'data/insat_real',
        'real_data_config.yaml'
    ]
    
    missing_paths = []
    
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print("⚠️  Warning: Some data files are missing:")
        for path in missing_paths:
            print(f"   - {path}")
        print("\nThe dashboard will still run but may show limited data.")
        print("Consider running the processing pipeline first.")
    
    return True

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'outputs/tracks',
        'data/insat_real',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def start_dashboard():
    """Start the Streamlit dashboard"""
    print("🚀 Starting Tropical Cloud Cluster Monitor Dashboard...")
    print("📊 Dashboard will open in your default web browser")
    print("🌐 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'dashboard.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🌪️ Tropical Cloud Cluster Monitor Dashboard Launcher")
    print("=" * 50)
    
    # Check and install dependencies
    if not check_dependencies():
        return False
    
    # Create necessary directories
    create_directories()
    
    # Check data files
    check_data_files()
    
    # Start dashboard
    return start_dashboard()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 