#!/usr/bin/env python3
"""
diagnose_data_columns.py - Data Column Diagnostics
Checks for missing columns in all data files and provides recommendations
"""

import os
import pandas as pd
import glob

def check_data_files():
    """Check all data files for column consistency"""
    print("🔍 Diagnosing Data Column Issues")
    print("=" * 50)
    
    # Define expected columns for different file types
    expected_columns = {
        'tracked_clusters': [
            'timestamp', 'center_lat', 'center_lon', 'pixel_count', 'area_km2',
            'min_tb', 'mean_tb', 'median_tb', 'max_radius_km', 'mean_radius_km',
            'cloud_top_height_km', 'convective_intensity', 'compactness',
            'std_cloud_height', 'timestep', 'source_file'
        ],
        'final_tracks': [
            'timestamp', 'datetime', 'track_id', 'center_lat', 'center_lon',
            'pixel_count', 'area_km2', 'min_tb', 'mean_tb', 'median_tb',
            'std_tb', 'max_radius', 'min_radius', 'mean_radius', 'max_height',
            'mean_height', 'cloud_top_height_km', 'cyclogenesis_risk', 'duration_hours'
        ],
        'cloud_clusters': [
            'center_lat', 'center_lon', 'area_km2', 'min_tb', 'cloud_top_height_km'
        ]
    }
    
    # Check files in outputs/tracks directory
    tracks_dir = "outputs/tracks"
    if os.path.exists(tracks_dir):
        print(f"\n📁 Checking {tracks_dir}/")
        for file_path in glob.glob(os.path.join(tracks_dir, "*.csv")):
            filename = os.path.basename(file_path)
            print(f"\n📄 {filename}:")
            
            try:
                df = pd.read_csv(file_path)
                print(f"   ✅ Loaded successfully ({len(df)} rows, {len(df.columns)} columns)")
                print(f"   📊 Available columns: {list(df.columns)}")
                
                # Check for missing timestep column
                if 'timestep' not in df.columns:
                    print(f"   ⚠️  MISSING: 'timestep' column")
                    print(f"   💡 Available time columns: {[col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]}")
                    
                    # Suggest fix
                    if 'timestamp' in df.columns:
                        print(f"   🔧 FIX: Can create timestep from 'timestamp' column")
                    elif 'datetime' in df.columns:
                        print(f"   🔧 FIX: Can create timestep from 'datetime' column")
                    else:
                        print(f"   🔧 FIX: Need to create default timestep column")
                
                # Check for other critical missing columns
                critical_columns = ['center_lat', 'center_lon', 'area_km2']
                missing_critical = [col for col in critical_columns if col not in df.columns]
                if missing_critical:
                    print(f"   ❌ CRITICAL MISSING: {missing_critical}")
                
                # Check for track_id column
                track_id_variants = ['track_id', 'trackid', 'track', 'cluster_id']
                track_id_found = any(col in df.columns for col in track_id_variants)
                if not track_id_found:
                    print(f"   ⚠️  MISSING: Track ID column (expected one of: {track_id_variants})")
                
            except Exception as e:
                print(f"   ❌ Error loading file: {str(e)}")
    
    # Check files in outputs directory
    outputs_dir = "outputs"
    print(f"\n📁 Checking {outputs_dir}/")
    for file_path in glob.glob(os.path.join(outputs_dir, "*.csv")):
        filename = os.path.basename(file_path)
        if filename not in ['tracked_clusters.csv', 'final_tracks.csv']:  # Already checked
            print(f"\n📄 {filename}:")
            
            try:
                df = pd.read_csv(file_path)
                print(f"   ✅ Loaded successfully ({len(df)} rows, {len(df.columns)} columns)")
                print(f"   📊 Available columns: {list(df.columns)}")
                
                # Check for timestep column
                if 'timestep' not in df.columns:
                    print(f"   ⚠️  MISSING: 'timestep' column")
                
            except Exception as e:
                print(f"   ❌ Error loading file: {str(e)}")
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY OF ISSUES AND RECOMMENDATIONS:")
    print("=" * 50)
    
    # Provide recommendations
    print("\n🔧 RECOMMENDED FIXES:")
    print("1. For missing 'timestep' column:")
    print("   - If 'timestamp' exists: df['timestep'] = range(len(df))")
    print("   - If 'datetime' exists: df['timestep'] = range(len(df))")
    print("   - Otherwise: df['timestep'] = range(len(df))")
    
    print("\n2. For missing track_id column:")
    print("   - Create: df['track_id'] = range(1, len(df)+1)")
    
    print("\n3. For missing critical columns:")
    print("   - Reprocess data with updated detection.py")
    print("   - Check data source format and processing pipeline")
    
    print("\n4. Data validation:")
    print("   - Add column validation before processing")
    print("   - Use fallback values for missing columns")
    print("   - Implement graceful degradation in UI")

def create_fix_script():
    """Create a script to automatically fix common column issues"""
    fix_script = '''#!/usr/bin/env python3
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
'''
    
    with open('fix_data_columns.py', 'w', encoding='utf-8') as f:
        f.write(fix_script)
    
    print("\nCreated fix_data_columns.py script")
    print("   Run: python fix_data_columns.py")
    print("   This will automatically fix common column issues")

if __name__ == "__main__":
    check_data_files()
    create_fix_script() 