"""
batch_processor.py - Batch Processing for Tropical Cloud Clusters
Processes a directory of INSAT-3D files to detect and track cloud clusters
"""

import argparse
import os
import sys
import glob
import pandas as pd
from detection import load_data, process_irbt_data, calculate_cluster_properties, FakeConfig
import time

def process_directory(input_dir, output_dir="outputs/batch", force=False):
    """
    Process all NetCDF files in a directory
    Args:
        input_dir: Directory with INSAT-3D files (*.nc)
        output_dir: Where to save processed results
        force: Whether to reprocess all files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all NetCDF files
    files = sorted(glob.glob(os.path.join(input_dir, "*.nc")))
    if not files:
        print(f"No NetCDF files found in {input_dir}")
        return
    
    print(f"Found {len(files)} files to process\n")
    
    all_detections = []
    
    for i, file in enumerate(files):
        print(f"Processing file {i+1}/{len(files)}: {os.path.basename(file)}")
        
        # Check if output already exists
        output_file = os.path.join(output_dir, f"timestep_{i}.csv")
        if not force and os.path.exists(output_file):
            print(f"Skipping processed file: {file} (output exists)")
            continue
            
        start_time = time.time()
        
        try:
            # Load and process data
            lats, lons, irbt_data = load_data(config=FakeConfig(), file_path=file)
            regions = process_irbt_data(irbt_data)
            properties = calculate_cluster_properties(regions, lats, lons)
            
            if properties:
                # Add timestep information
                for prop in properties:
                    prop['timestep'] = i
                    prop['source_file'] = os.path.basename(file)
                
                # Save individual timestep results
                df = pd.DataFrame(properties)
                df.to_csv(output_file, index=False)
                
                all_detections.append(df)
                print(f"Detected {len(properties)} clusters in this file")
            else:
                print("No clusters detected in this file")
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
            
        processing_time = time.time() - start_time
        print(f"Processing time: {processing_time:.2f} seconds\n")
    
    # Combine all detections
    if all_detections:
        combined = pd.concat(all_detections)
        combined_path = os.path.join(output_dir, 'all_detections.csv')
        combined.to_csv(combined_path, index=False)
        print(f"Saved combined detections to {combined_path}")
        
        # Create tracked_clusters.csv for compatibility
        tracked_path = os.path.join('outputs', 'tracks', 'tracked_clusters.csv')
        combined.to_csv(tracked_path, index=False)
        print(f"Created compatibility file: {tracked_path}")
    else:
        print("No clusters detected in any file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('--force', action='store_true', help='Reprocess all files')
    args = parser.parse_args()
    process_directory(args.input_dir, force=args.force) 