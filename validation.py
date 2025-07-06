"""
validation.py - Validation Against Historical Cyclone Events
Combines timestep files and compares detected clusters with historical cyclone tracks
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from helpers import haversine  # Ensure this is in helpers.py
import streamlit as st

# Configuration
TRACKS_DIR = "outputs/tracks"
HISTORICAL_DATA = "data/historical_cyclones.csv"
VALIDATION_OUTPUT = "validation_report.png"

def load_combined_tracks():
    """Load and combine all timestep CSV files"""
    track_files = glob.glob(os.path.join(TRACKS_DIR, "timestep_*.csv"))
    if not track_files:
        raise FileNotFoundError(f"No track files found in {TRACKS_DIR}")
    
    print(f"Found {len(track_files)} track files. Combining...")
    
    all_dfs = []
    for file in track_files:
        try:
            df = pd.read_csv(file)
            # Extract timestep from filename
            timestep = int(os.path.basename(file).split('_')[1].split('.')[0])
            df['timestep'] = timestep
            all_dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_dfs:
        raise ValueError("No valid track files could be loaded")
    
    return pd.concat(all_dfs, ignore_index=True)

def load_historical_cyclones():
    """Load historical cyclone data with error handling"""
    try:
        return pd.read_csv(HISTORICAL_DATA)
    except FileNotFoundError:
        print("Historical data not found. Using sample cyclone data.")
        return pd.DataFrame({
            'name': ['Biparjoy', 'Mocha', 'Tauktae'],
            'lat': [20.5, 18.2, 15.8],
            'lon': [68.3, 92.5, 72.1],
            'date': ['2023-06-15', '2023-05-14', '2021-05-17']
        })

def match_cyclones(clusters, cyclones, time_window=24, distance_threshold=300):
    """
    Match detected clusters to historical cyclones
    time_window: hours (each timestep = 30 min, so 48 timesteps = 24 hours)
    distance_threshold: km
    """
    clusters['matched_cyclone'] = None
    clusters['match_distance'] = np.nan

    # Ensure timestamp is datetime
    if 'timestamp' in clusters.columns:
        clusters['timestamp'] = pd.to_datetime(clusters['timestamp'])

    # Convert cyclone date to timestamp
    cyclones['timestamp'] = pd.to_datetime(cyclones['date'])

    for _, cyclone in cyclones.iterrows():
        # Find clusters in time window
        cyclone_time = cyclone['timestamp']
        time_diff = (clusters['timestamp'] - cyclone_time).dt.total_seconds() / 3600
        time_match = np.abs(time_diff) <= time_window

        # Find clusters in spatial window
        for idx, cluster in clusters[time_match].iterrows():
            distance = haversine(
                cluster['center_lon'], cluster['center_lat'],
                cyclone['lon'], cyclone['lat']
            )

            if distance <= distance_threshold:
                # Update with better match if found
                if clusters.at[idx, 'match_distance'] > distance or pd.isna(clusters.at[idx, 'match_distance']):
                    clusters.at[idx, 'matched_cyclone'] = cyclone['name']
                    clusters.at[idx, 'match_distance'] = distance

    return clusters

def generate_validation_report(clusters, cyclones):
    """Create validation visualization"""
    plt.figure(figsize=(12, 10))
    
    # Plot all detected clusters
    plt.scatter(clusters['center_lon'], clusters['center_lat'], 
               s=50, c='blue', alpha=0.3, label='Detected Clusters')
    
    # Plot matched clusters
    matched = clusters.dropna(subset=['matched_cyclone'])
    if not matched.empty:
        plt.scatter(matched['center_lon'], matched['center_lat'], 
                   s=100, c='red', edgecolor='black', label='Matched Cyclones')
        
        # Add labels for matched cyclones
        for _, row in matched.iterrows():
            plt.text(row['center_lon'], row['center_lat'] + 0.3, 
                     f"{row['matched_cyclone']}\n{row['match_distance']:.0f}km", 
                     fontsize=9, ha='center')
    
    # Plot historical cyclone positions
    plt.scatter(cyclones['lon'], cyclones['lat'], 
               s=200, marker='*', c='gold', edgecolor='black', label='Historical Cyclones')
    
    # Add cyclone labels
    for _, row in cyclones.iterrows():
        plt.text(row['lon'], row['lat'] + 0.5, row['name'], 
                 fontsize=10, ha='center', fontweight='bold')
    
    # Configure plot
    plt.title('Tropical Cloud Cluster Detection Validation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xlim(40, 100)
    plt.ylim(-30, 30)
    
    # Save output
    plt.savefig(VALIDATION_OUTPUT, dpi=150, bbox_inches='tight')
    print(f"Validation report saved to {VALIDATION_OUTPUT}")
    
    # Calculate metrics
    if not matched.empty:
        precision = len(matched) / len(clusters)
        recall = len(matched) / len(cyclones)
        mean_distance = matched['match_distance'].mean()
        
        print("\nValidation Metrics:")
        print(f"- Precision: {precision:.2%} (matched clusters/total clusters)")
        print(f"- Recall: {recall:.2%} (matched cyclones/total cyclones)")
        print(f"- Mean Distance Error: {mean_distance:.1f} km")
    else:
        print("No matches found between clusters and historical cyclones")

def main():
    """Main validation workflow"""
    try:
        # Step 1: Load and combine cluster tracks
        clusters = load_combined_tracks()
        
        # Step 2: Load historical cyclone data
        cyclones = load_historical_cyclones()

        # Add synthetic timestamps if missing
        if 'timestamp' not in clusters.columns:
            # Create synthetic timestamps aligned with today
            base_date = pd.Timestamp.now().normalize() + pd.Timedelta(hours=12)
            clusters['timestamp'] = base_date + pd.to_timedelta(clusters['timestep'] * 30, 'm')

        # Add debug output
        print(f"Historical cyclones:\n{cyclones}")
        print(f"Sample clusters:\n{clusters[['center_lat', 'center_lon', 'timestamp']].head()}")

        # Step 3: Match clusters to cyclones
        clusters = match_cyclones(clusters, cyclones)
        
        # Step 4: Generate validation report
        generate_validation_report(clusters, cyclones)
        
        # Optional: Save matched data
        clusters.to_csv('outputs/validated_clusters.csv', index=False)
        
    except Exception as e:
        print(f"Validation failed: {str(e)}")
        # Create error visualization
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "Validation Failed\n" + str(e), 
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.savefig(VALIDATION_OUTPUT)
        print(f"Error visualization saved to {VALIDATION_OUTPUT}")

    # Add Streamlit report
    st.subheader("Data Source Info")
    st.metric("Data Format", f"{metadata.get('source_format', 'unknown')}")
    st.metric("Processing Time", f"{proc_time:.2f} sec/file")
    with st.expander("Technical Metadata"):
        st.json(metadata)

if __name__ == "__main__":
    main() 