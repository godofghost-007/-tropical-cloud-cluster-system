#!/usr/bin/env python3
"""
validate_timesteps.py - Validate timestep data in processed files
"""

import pandas as pd
import os

def validate_timesteps():
    """Validate timestep data in processed files"""
    print("🔍 Validating Timestep Data")
    print("=" * 50)
    
    # Check tracked_clusters.csv
    tracks_file = 'outputs/tracks/tracked_clusters.csv'
    if os.path.exists(tracks_file):
        print(f"\n📄 {tracks_file}:")
        df = pd.read_csv(tracks_file)
        
        if 'timestep' in df.columns:
            print(f"✅ Timestep column found")
            print(f"📊 Timestep Statistics:")
            print(df['timestep'].describe())
            print(f"🔢 Unique timesteps: {sorted(df['timestep'].unique())}")
            print(f"📈 Total records: {len(df)}")
            
            # Check for gaps in timesteps
            timesteps = sorted(df['timestep'].unique())
            expected_range = list(range(min(timesteps), max(timesteps) + 1))
            missing_timesteps = set(expected_range) - set(timesteps)
            
            if missing_timesteps:
                print(f"⚠️  Missing timesteps: {sorted(missing_timesteps)}")
            else:
                print(f"✅ All timesteps present (no gaps)")
                
        else:
            print(f"❌ Timestep column missing")
    
    # Check final_tracks.csv
    final_file = 'outputs/tracks/final_tracks.csv'
    if os.path.exists(final_file):
        print(f"\n📄 {final_file}:")
        df = pd.read_csv(final_file)
        
        if 'timestep' in df.columns:
            print(f"✅ Timestep column found")
            print(f"📊 Timestep Statistics:")
            print(df['timestep'].describe())
            print(f"🔢 Unique timesteps: {sorted(df['timestep'].unique())}")
            print(f"📈 Total records: {len(df)}")
        else:
            print(f"❌ Timestep column missing")
    
    # Check individual timestep files
    print(f"\n📁 Individual timestep files:")
    timestep_files = [f for f in os.listdir('outputs/tracks') if f.startswith('timestep_') and f.endswith('.csv')]
    timestep_files.sort()
    
    for file in timestep_files:
        file_path = os.path.join('outputs/tracks', file)
        df = pd.read_csv(file_path)
        
        if 'timestep' in df.columns:
            timestep_value = df['timestep'].iloc[0]
            print(f"✅ {file}: timestep {timestep_value}, {len(df)} records")
        else:
            print(f"❌ {file}: timestep column missing")
    
    print("\n" + "=" * 50)
    print("✅ Validation Complete!")

if __name__ == "__main__":
    validate_timesteps() 