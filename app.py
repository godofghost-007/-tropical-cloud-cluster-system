import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from detection import calculate_cyclogenesis_potential
from tracking import forecast_tracks, generate_alerts

# Set page configuration
st.set_page_config(
    page_title="Tropical Cloud Cluster Tracker",
    page_icon="🌪️",
    layout="wide"
)

# Load tracked data
@st.cache_data
def load_tracked_data():
    return pd.read_csv('outputs/tracks/tracked_clusters.csv')

df = load_tracked_data()

# CYCLOGENESIS RISK ASSESSMENT
st.sidebar.header("Risk Analysis")
risk_threshold = st.sidebar.slider("Cyclogenesis Risk Threshold", 0.0, 1.0, 0.65)

# Check if required columns exist
required_columns = ['min_tb', 'area_km2', 'compactness', 'convective_intensity', 'std_cloud_height']
if all(col in df.columns for col in required_columns):
    df['cyclogenesis_risk'] = df.apply(calculate_cyclogenesis_potential, axis=1)
    high_risk = df[df['cyclogenesis_risk'] > risk_threshold]
    
    # Display risk clusters
    if not high_risk.empty:
        st.warning(f"🚨 {len(high_risk)} high-risk clusters detected!")
        st.dataframe(high_risk[['track_id', 'timestep', 'cyclogenesis_risk']])
        
        # Plot risk evolution
        fig_risk = plt.figure(figsize=(10, 4))
        for track_id in high_risk['track_id'].unique():
            track = df[df['track_id'] == track_id]
            plt.plot(track['timestep'], track['cyclogenesis_risk'], 
                     marker='o', label=f"Track {track_id}")
        
        plt.axhline(y=risk_threshold, color='r', linestyle='--')
        plt.xlabel("Timestep")
        plt.ylabel("Cyclogenesis Risk")
        plt.title("High-Risk Cluster Evolution")
        plt.legend()
        st.pyplot(fig_risk)
    else:
        st.success("✅ No high-risk clusters detected")
else:
    st.warning("⚠️ Advanced properties not available. Reprocess data with updated detection.py")

# FORECAST MODULE
st.sidebar.header("Forecasting")
forecast_hours = st.sidebar.slider("Forecast Hours", 6, 72, 24)
if st.sidebar.button("Generate Forecast"):
    forecast_df = forecast_tracks(df, forecast_hours)
    combined_df = pd.concat([df, forecast_df])
    
    # Visualize forecast
    fig_forecast = plt.figure(figsize=(10, 8))
    ax = fig_forecast.add_subplot(111)
    ax.set_xlim(40, 100)
    ax.set_ylim(-30, 30)
    
    for track_id in combined_df['track_id'].unique():
        track = combined_df[combined_df['track_id'] == track_id]
        actual = track[track.get('forecast', False) == False]
        forecast = track[track.get('forecast', False) == True]
        
        # Plot actual track
        ax.plot(actual['center_lon'], actual['center_lat'], 
                'o-', markersize=4, label=f"Track {track_id}")
        
        # Plot forecast
        if not forecast.empty:
            ax.plot(forecast['center_lon'], forecast['center_lat'], 
                    'x--', color='red', alpha=0.7)
            ax.text(forecast['center_lon'].iloc[-1], 
                    forecast['center_lat'].iloc[-1] + 0.5,
                    f"+{forecast['hours_ahead'].iloc[-1]}h",
                    fontsize=8, color='red')
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Cluster Tracks with {forecast_hours}h Forecast")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig_forecast)
    st.session_state.forecast_df = forecast_df

# Title and description
st.title('🌪️ Tropical Cloud Cluster Tracker')
st.markdown("""
Visualize the movement and evolution of tropical cloud clusters detected in INSAT-3D satellite data.
""")

# Sidebar controls
st.sidebar.header("Controls")

# Check if timestep column exists, if not create it from available data
if 'timestep' not in df.columns:
    # Try to create timestep from other available columns
    if 'datetime' in df.columns:
        # Convert datetime to timestep numbers
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        df['timestep'] = range(len(df))
    elif 'timestamp' in df.columns:
        # Convert timestamp to timestep numbers
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df['timestep'] = range(len(df))
    else:
        # Create default timestep column
        df['timestep'] = range(len(df))
        st.warning("⚠️ No time information found. Using default timesteps.")

timesteps = sorted(df['timestep'].unique())
selected_timestep = st.sidebar.slider(
    'Select Timestep', 
    min_value=min(timesteps), 
    max_value=max(timesteps), 
    value=min(timesteps)
)

# Filter data for selected timestep
df_timestep = df[df['timestep'] == selected_timestep]

# Main columns
col1, col2 = st.columns([2, 1])

with col1:
    # Create map visualization
    st.subheader(f"Cluster Map - Timestep {selected_timestep}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set geographic boundaries
    ax.set_xlim(40, 100)
    ax.set_ylim(-30, 30)
    
    # Plot clusters
    for _, row in df_timestep.iterrows():
        color = plt.cm.viridis(row['cloud_top_height_km'] / 20)
        size = row['area_km2'] / 500
        ax.scatter(row['center_lon'], row['center_lat'], 
                   color=color, s=size, alpha=0.8)
        ax.text(row['center_lon'], row['center_lat'] + 0.5, 
                f"ID:{row['track_id']}", fontsize=8, ha='center')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    ax.set_title('Cloud Cluster Positions')
    
    st.pyplot(fig)

with col2:
    # Track selector
    st.subheader("Track Explorer")
    
    if not df_timestep.empty:
        selected_track = st.selectbox(
            "Select a Track", 
            df_timestep['track_id'].unique()
        )
        
        # Get full track history
        track_df = df[df['track_id'] == selected_track].sort_values('timestep')
        
        if not track_df.empty:
            st.metric("Current Cloud Top Height", 
                      f"{track_df.iloc[-1]['cloud_top_height_km']:.1f} km")
            
            # Show track properties
            st.write("**Track Properties**")
            st.dataframe(track_df[['timestep', 'area_km2', 'min_tb', 'cloud_top_height_km']])
            
            # Create evolution plot
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(track_df['timestep'], track_df['cloud_top_height_km'], 
                     marker='o', label='Cloud Top Height')
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('Height (km)')
            ax2.set_title(f'Track {selected_track} Evolution')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            st.pyplot(fig2)
        else:
            st.warning("No data available for selected track")
    else:
        st.info("No clusters detected in this timestep")

# Show raw data
st.subheader("Raw Data")
st.dataframe(df_timestep)

# Operational Alerts
st.sidebar.header("Operational Alerts")
if st.sidebar.button("Check for Alerts"):
    alerts = generate_alerts(df)
    if alerts:
        for alert in alerts:
            st.error(f"⚠️ **{alert['type']}** ({alert['severity']}): {alert['message']}")
    else:
        st.info("🟢 No critical alerts at this time")
