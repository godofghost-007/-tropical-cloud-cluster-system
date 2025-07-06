#!/usr/bin/env python3
"""
Enhanced Tropical Cloud Cluster System Launcher
Runs detection and dashboard with proper configuration
"""

import os
import sys
import subprocess
import time
import argparse

def run_detection():
    """Run the enhanced detection system"""
    print("🚀 Starting Enhanced Detection System...")
    
    try:
        # Run enhanced detection
        result = subprocess.run([
            sys.executable, "detection_enhanced.py",
            "--synthetic", "--full", "--output-dir", "outputs"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Detection completed successfully!")
            print(result.stdout)
            return True
        else:
            print("❌ Detection failed!")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running detection: {e}")
        return False

def run_dashboard(port=8505):
    """Run the enhanced dashboard"""
    print(f"🌐 Starting Enhanced Dashboard on port {port}...")
    
    try:
        # Run dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app_enhanced_v2.py",
            "--server.port", str(port),
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'pydeck',
        'xarray', 'matplotlib', 'seaborn', 'reportlab'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies are installed")
    return True

def main():
    parser = argparse.ArgumentParser(description='Enhanced Tropical Cloud Cluster System')
    parser.add_argument('--detection-only', action='store_true', help='Run only detection')
    parser.add_argument('--dashboard-only', action='store_true', help='Run only dashboard')
    parser.add_argument('--port', type=int, default=8505, help='Dashboard port')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency check')
    
    args = parser.parse_args()
    
    print("🌪️ Enhanced Tropical Cloud Cluster System")
    print("=" * 50)
    
    # Check dependencies
    if not args.skip_deps and not check_dependencies():
        return
    
    # Run detection if needed
    if not args.dashboard_only:
        if not run_detection():
            print("❌ Detection failed. Exiting.")
            return
    
    # Run dashboard if needed
    if not args.detection_only:
        print(f"\n🌐 Dashboard will be available at: http://localhost:{args.port}")
        print("Press Ctrl+C to stop the dashboard")
        run_dashboard(args.port)

if __name__ == "__main__":
    main() 