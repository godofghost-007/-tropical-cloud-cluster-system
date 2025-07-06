#!/usr/bin/env python3
"""
fix_data_columns.py - Auto-fix common column issues
"""

import pandas as pd
import os
import glob

def fix_timestep_column(df):
    """Add timestep column if missing"""
    if 'timestep' not in df.columns:
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
            df['timestep'] = range(len(df))
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            df['timestep'] = range(len(df))
        else:
            df['timestep'] = range(len(df))
        print(f"Added timestep column")
    return df

def fix_track_id_column(df):
    """Add track_id column if missing"""
    track_id_variants = ['track_id', 'trackid', 'track', 'cluster_id']
    if not any(col in df.columns for col in track_id_variants):
        df['track_id'] = range(1, len(df)+1)
        print(f"Added track_id column")
    return df

def fix_missing_columns(df):
    """Add default values for missing critical columns"""
    defaults = {
        'center_lat': 0.0,
        'center_lon': 0.0,
        'area_km2': 1000.0,
        'cloud_top_height_km': 10.0,
        'min_tb': 200.0,
        'cyclogenesis_risk': 0.0
    }
    
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val
            print(f"Added {col} column with default value {default_val}")
    
    return df

def process_file(file_path):
    """Process a single file and fix column issues"""
    print(f"Processing: {os.path.basename(file_path)}")
    
    try:
        df = pd.read_csv(file_path)
        original_columns = list(df.columns)
        
        # Apply fixes
        df = fix_timestep_column(df)
        df = fix_track_id_column(df)
        df = fix_missing_columns(df)
        
        # Save fixed file
        backup_path = file_path.replace('.csv', '_backup.csv')
        os.rename(file_path, backup_path)
        df.to_csv(file_path, index=False)
        
        print(f"Fixed and saved. Backup created: {os.path.basename(backup_path)}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    """Fix all data files"""
    print("Auto-fixing column issues in data files...")
    
    # Process files in outputs/tracks
    tracks_dir = "outputs/tracks"
    if os.path.exists(tracks_dir):
        for file_path in glob.glob(os.path.join(tracks_dir, "*.csv")):
            process_file(file_path)
    
    # Process files in outputs
    outputs_dir = "outputs"
    for file_path in glob.glob(os.path.join(outputs_dir, "*.csv")):
        if not file_path.startswith(os.path.join(outputs_dir, "tracks")):
            process_file(file_path)
    
    print("Column fixing complete!")

if __name__ == "__main__":
    main()
