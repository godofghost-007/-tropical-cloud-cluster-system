# tracking.py
import os
import pandas as pd
import numpy as np
from detection import load_data, process_irbt_data, calculate_cluster_properties, FakeConfig
from helpers import haversine_vector
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

# Configuration
TIME_SERIES_DIR = "data/time_series"
OUTPUT_DIR = "outputs/tracks"
MAX_DISTANCE = 400  # km (maximum movement between 30-min frames)

def process_timestep(file_path):
    """Process a single timestep file"""
    lats, lons, irbt_data = load_data(file_path=file_path)
    regions = process_irbt_data(irbt_data)
    properties = calculate_cluster_properties(regions, lats, lons)
    return pd.DataFrame(properties)

def track_clusters():
    """Main tracking function"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check for combined data
    combined_path = os.path.join(OUTPUT_DIR, 'all_detections.csv')
    if not os.path.exists(combined_path):
        # Try alternative location
        combined_path = 'outputs/all_detections.csv'
        if not os.path.exists(combined_path):
            print("Error: No combined detections found. Run batch_processor.py first.")
            return
    print(f"Loading combined detections from {combined_path}")
    df = pd.read_csv(combined_path)
    # Ensure required columns exist
    if 'timestep' not in df.columns:
        print("Warning: 'timestep' column missing. Creating sequential timesteps.")
        df['timestep'] = np.arange(len(df))
    all_dfs = [df[df['timestep'] == t].copy() for t in sorted(df['timestep'].unique())]
    # Simple tracking algorithm
    next_id = 1
    for i in range(1, len(all_dfs)):
        prev_df = all_dfs[i-1]
        curr_df = all_dfs[i]
        if len(prev_df) == 0 or len(curr_df) == 0:
            print(f"Skipping timestep {i} - no clusters to match")
            continue
        prev_points = prev_df[['center_lon', 'center_lat']].values
        curr_points = curr_df[['center_lon', 'center_lat']].values
        distances = haversine_vector(prev_points, curr_points)
        row_ind, col_ind = linear_sum_assignment(distances)
        for r, c in zip(row_ind, col_ind):
            if distances[r, c] < MAX_DISTANCE:
                prev_id = prev_df.iloc[r]['track_id'] if 'track_id' in prev_df.columns else -1
                if prev_id == -1:
                    prev_id = next_id
                    next_id += 1
                    prev_df.at[prev_df.index[r], 'track_id'] = prev_id
                curr_df.at[curr_df.index[c], 'track_id'] = prev_id
    combined = pd.concat(all_dfs)
    output_csv = os.path.join(OUTPUT_DIR, 'final_tracks.csv')
    combined.to_csv(output_csv, index=False)
    print(f"Saved tracked clusters to {output_csv}")
    return combined

def forecast_tracks(track_df, hours=24):
    """Forecast cluster movement using linear regression"""
    forecasts = []
    
    for track_id in track_df['track_id'].unique():
        track = track_df[track_df['track_id'] == track_id].sort_values('timestep')
        if len(track) < 3:
            continue
        
        # Prepare data
        X = track[['timestep']].values
        y_lon = track['center_lon'].values
        y_lat = track['center_lat'].values
        
        # Train models
        model_lon = LinearRegression().fit(X, y_lon)
        model_lat = LinearRegression().fit(X, y_lat)
        
        # Forecast future positions
        future_steps = np.array([[track['timestep'].max() + i] for i in range(1, int(hours/0.5)+1)])
        pred_lon = model_lon.predict(future_steps)
        pred_lat = model_lat.predict(future_steps)
        
        for i, (lon, lat) in enumerate(zip(pred_lon, pred_lat)):
            forecasts.append({
                'track_id': track_id,
                'timestep': track['timestep'].max() + i + 1,
                'center_lon': lon,
                'center_lat': lat,
                'forecast': True,
                'hours_ahead': (i+1)*0.5
            })
            
    return pd.DataFrame(forecasts)

def generate_alerts(track_df):
    """Generate operational alerts based on track evolution"""
    alerts = []
    for track_id in track_df['track_id'].unique():
        track = track_df[track_df['track_id'] == track_id].sort_values('timestep')
        if len(track) < 4:
            continue
        # Alert 1: Rapid intensification
        last_heights = track['cloud_top_height_km'].tail(3)
        if last_heights.diff().mean() > 1.0:  # >1km/step increase
            alerts.append({
                'track_id': track_id,
                'type': 'Rapid Intensification',
                'severity': 'High',
                'message': f"Track {track_id} shows rapid convective deepening (+{last_heights.diff().mean():.1f}km/step)"
            })
        # Alert 2: Landfall prediction
        if track['center_lat'].iloc[-1] > 15 and track['center_lon'].diff().mean() < -0.5:
            alerts.append({
                'track_id': track_id,
                'type': 'Possible Landfall',
                'severity': 'Medium',
                'message': f"Track {track_id} moving west-northwest toward land"
            })
    return alerts

def archive_to_hdf5(df, filename="outputs/cluster_archive.h5"):
    """Archive cluster data to HDF5 for efficient storage"""
    with pd.HDFStore(filename, 'w') as store:
        store.put('clusters', df)
        store.get_storer('clusters').attrs.metadata = {
            'description': 'Tropical Cloud Cluster Archive',
            'creation_date': pd.Timestamp.now().isoformat(),
            'data_source': 'INSAT-3D IRBRT'
        }
    print(f"Archived {len(df)} records to {filename}")

if __name__ == "__main__":
    track_clusters()
