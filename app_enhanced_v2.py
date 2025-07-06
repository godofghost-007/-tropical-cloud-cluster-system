#!/usr/bin/env python3
"""
Enhanced Tropical Cloud Cluster Dashboard v2.0
Advanced monitoring and analysis with 3D visualization and reporting
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime
import xarray as xr
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import glob

# Page Configuration
st.set_page_config(
    page_title="🌪️ Tropical Cloud Cluster Monitor",
    layout="wide",
    page_icon="🌪️",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1d2e 100%);
        color: #f0f2f6;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0d47a1 0%, #2196f3 50%, #64b5f6 100%);
        padding: 2rem 1rem;
        border-radius: 0 0 20px 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #2d3748 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
        border-color: rgba(33, 150, 243, 0.3);
    }
    
    .alert-card {
        background: linear-gradient(135deg, #f44336 0%, #b71c1c 100%);
        animation: pulse 2s infinite;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 2px solid rgba(255,255,255,0.2);
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(244, 67, 54, 0); }
        100% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0); }
    }
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    """Load and preprocess the cluster data"""
    try:
        # Check if the data file exists
        if not os.path.exists('outputs/cloud_clusters.csv'):
            st.warning("Data file not found. Generating sample data for demonstration.")
            return generate_sample_data()
        
        # Load the CSV data
        df = pd.read_csv('outputs/cloud_clusters.csv')
        
        # Check if DataFrame is empty
        if df.empty:
            st.warning("Data file is empty. Generating sample data for demonstration.")
            return generate_sample_data()
        
        # Map the actual column names to expected names
        column_mapping = {
            'center_lat': 'centroid_lat',
            'center_lon': 'centroid_lon',
            'convective_intensity': 'max_windspeed',  # Use convective intensity as proxy for windspeed
            'area_km2': 'area_km2',
            'timestep': 'timestep',
            'quality_score': 'quality_score',
            'edge_confidence': 'edge_confidence',
            'data_quality': 'data_quality',
            'data_coverage': 'data_coverage'
        }
        
        # Rename columns that exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        # Use existing track_id if available, otherwise create from timestep
        if 'track_id' not in df.columns:
            df['track_id'] = df['timestep'].astype(str)
        
        # Add missing columns with reasonable defaults
        if 'max_windspeed' not in df.columns:
            # Convert convective_intensity to windspeed (km/h)
            df['max_windspeed'] = df['convective_intensity'] * 1000 + 20  # Scale and offset
        
        if 'precipitation' not in df.columns:
            # Generate precipitation based on convective intensity
            df['precipitation'] = df['convective_intensity'] * 100 + np.random.uniform(0, 10, len(df))
        
        if 'development_stage' not in df.columns:
            # Assign development stages based on timestep
            stages = ['Formation', 'Development', 'Mature', 'Decay']
            df['development_stage'] = pd.cut(
                df['timestep'], 
                bins=4, 
                labels=stages
            )
        
        if 'mean_tb' not in df.columns:
            # Use cloud_top_height_km as proxy for mean_tb
            df['mean_tb'] = 280 - (df['cloud_top_height_km'] * 5)  # Inverse relationship
        
        # Data validation and cleaning
        df = clean_coordinate_data(df)
        
        # Add intensity categories
        df['intensity_category'] = pd.cut(
            df['max_windspeed'], 
            bins=[0, 30, 60, 90, 120, float('inf')],
            labels=['Weak', 'Moderate', 'Strong', 'Very Strong', 'Extreme']
        )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Generating sample data for demonstration.")
        return generate_sample_data()

def generate_sample_data():
    """Generate sample data for demonstration when real data is not available"""
    np.random.seed(42)
    
    # Generate sample data
    n_clusters = 50
    n_tracks = 8
    
    data = []
    for track_id in range(1, n_tracks + 1):
        # Generate track path
        start_lat = np.random.uniform(-30, 30)
        start_lon = np.random.uniform(-180, 180)
        
        for timestep in range(1, 25):  # 24 timesteps
            # Simulate movement
            lat = start_lat + np.random.normal(0, 0.5) * timestep
            lon = start_lon + np.random.normal(0, 0.5) * timestep
            
            # Ensure coordinates are valid
            lat = np.clip(lat, -90, 90)
            lon = np.clip(lon, -180, 180)
            
            # Generate cluster properties
            max_windspeed = np.random.uniform(20, 100)
            area_km2 = np.random.uniform(1000, 50000)
            precipitation = np.random.uniform(0, 50)
            quality_score = np.random.uniform(0.7, 1.0)
            mean_irbt = np.random.uniform(200, 280)
            
            # Development stage
            stages = ['Formation', 'Development', 'Mature', 'Decay']
            development_stage = np.random.choice(stages, p=[0.2, 0.3, 0.3, 0.2])
            
            data.append({
                'track_id': track_id,
                'timestep': timestep,
                'centroid_lat': lat,
                'centroid_lon': lon,
                'max_windspeed': max_windspeed,
                'area_km2': area_km2,
                'precipitation': precipitation,
                'quality_score': quality_score,
                'mean_irbt': mean_irbt,
                'development_stage': development_stage
            })
    
    df = pd.DataFrame(data)
    
    # Add intensity categories
    df['intensity_category'] = pd.cut(
        df['max_windspeed'], 
        bins=[0, 30, 60, 90, 120, float('inf')],
        labels=['Weak', 'Moderate', 'Strong', 'Very Strong', 'Extreme']
    )
    
    return df

def clean_coordinate_data(df):
    """Clean and validate coordinate data"""
    if df.empty:
        return df
    
    # Ensure we have the right column names
    if 'centroid_lat' not in df.columns and 'center_lat' in df.columns:
        df['centroid_lat'] = df['center_lat']
        df['centroid_lon'] = df['center_lon']
    
    # Validate latitude values (must be between -90 and 90)
    if 'centroid_lat' in df.columns:
        invalid_lat = (df['centroid_lat'] < -90) | (df['centroid_lat'] > 90)
        if invalid_lat.any():
            st.warning(f"Found {invalid_lat.sum()} invalid latitude values. Filtering them out.")
            df = df[~invalid_lat]
    
    # Validate longitude values (must be between -180 and 180)
    if 'centroid_lon' in df.columns:
        invalid_lon = (df['centroid_lon'] < -180) | (df['centroid_lon'] > 180)
        if invalid_lon.any():
            st.warning(f"Found {invalid_lon.sum()} invalid longitude values. Filtering them out.")
            df = df[~invalid_lon]
    
    # Remove rows with NaN coordinates
    if 'centroid_lat' in df.columns and 'centroid_lon' in df.columns:
        nan_coords = df['centroid_lat'].isna() | df['centroid_lon'].isna()
        if nan_coords.any():
            st.warning(f"Found {nan_coords.sum()} rows with NaN coordinates. Filtering them out.")
            df = df[~nan_coords]
    
    # Ensure numeric types
    if 'centroid_lat' in df.columns:
        df['centroid_lat'] = pd.to_numeric(df['centroid_lat'], errors='coerce')
    if 'centroid_lon' in df.columns:
        df['centroid_lon'] = pd.to_numeric(df['centroid_lon'], errors='coerce')
    
    return df

# Load data
df = load_data()

# Header
st.markdown("""
<div class="main-header">
    <h1>🌪️ Advanced Tropical Cloud Cluster Monitor</h1>
    <p>Real-time Monitoring and Forecasting System v2.0</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("🌍 Global Controls")

# Track selection
track_options = ['All'] + sorted(df['track_id'].unique().tolist())
selected_track = st.sidebar.selectbox(
    "Select Track:",
    options=track_options
)

# Enhanced time slider with robust handling
timesteps = sorted(df['timestep'].unique())
if len(timesteps) == 0:
    st.sidebar.error("No timesteps available")
    st.stop()
elif len(timesteps) == 1:
    min_ts, max_ts = 0, 24  # Fallback range
    st.sidebar.warning("Limited timestep data. Using extended range.")
else:
    min_ts, max_ts = min(timesteps), max(timesteps)

time_range = st.sidebar.slider(
    "Select Time Range:",
    min_value=min_ts,
    max_value=max_ts,
    value=(min_ts, max_ts)
)

# Alert threshold
alert_threshold = st.sidebar.slider(
    "Alert Threshold (Max Windspeed km/h):",
    min_value=30,
    max_value=100,
    value=50
)

# Apply Filters
filtered_df = df[
    (df['timestep'] >= time_range[0]) & 
    (df['timestep'] <= time_range[1])
]

if selected_track != 'All':
    filtered_df = filtered_df[filtered_df['track_id'] == selected_track]

# Alert System
alert_df = filtered_df[filtered_df['max_windspeed'] >= alert_threshold]
if not alert_df.empty:
    st.sidebar.markdown(f"""
    <div class="alert-card">
        <h3>⚠️ ALERT: {len(alert_df)} clusters with winds > {alert_threshold} km/h!</h3>
    </div>
    """, unsafe_allow_html=True)

# Overview Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div style="text-align: center;">
            <h3>Total Clusters</h3>
            <h2 style="color: #2196f3;">{len(filtered_df)}</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div style="text-align: center;">
            <h3>Active Tracks</h3>
            <h2 style="color: #4caf50;">{filtered_df['track_id'].nunique()}</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div style="text-align: center;">
            <h3>High Intensity</h3>
            <h2 style="color: #ff9800;">{len(alert_df)}</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_precip = filtered_df['precipitation'].mean() if 'precipitation' in filtered_df.columns else 0
    st.markdown(f"""
    <div class="metric-card">
        <div style="text-align: center;">
            <h3>Avg Precipitation</h3>
            <h2 style="color: #9c27b0;">{avg_precip:.1f} mm/h</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌎 World View", "📊 Real-time Monitoring", "📈 Cluster Analytics", 
    "🌀 Track Explorer", "📊 Reports", "⚙️ 3D Analysis"
])

# World View
with tab1:
    st.header("🌍 Global Cluster Overview")
    
    # Create map with enhanced features
    fig = px.scatter_map(
        filtered_df,
        lat='centroid_lat',
        lon='centroid_lon',
        hover_name='track_id',
        zoom=2,
        color='max_windspeed',
        size='area_km2',
        color_continuous_scale=px.colors.sequential.Viridis,
        hover_data=['timestep', 'development_stage', 'quality_score'],
        title="Tropical Cloud Clusters - Global Distribution"
    )
    
    # Add heatmap layer for high-intensity areas
    high_intensity = filtered_df[filtered_df['max_windspeed'] > alert_threshold]
    if not high_intensity.empty:
        fig.add_trace(px.density_mapbox(
            high_intensity,
            lat='centroid_lat',
            lon='centroid_lon',
            z='max_windspeed',
            radius=20,
            opacity=0.6,
            colorscale='Reds'
        ).data[0])
    
    fig.update_layout(
        mapbox_style="open-street-map",
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        mapbox=dict(
            center=dict(lat=0, lon=0),
            zoom=2
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Tracks", filtered_df['track_id'].nunique())
    with col2:
        st.metric("High Intensity", len(alert_df))
    with col3:
        st.metric("Avg Windspeed", f"{filtered_df['max_windspeed'].mean():.1f} km/h")

    # Real-time Track Table
    st.subheader("📊 Real-time Track Status")
    
    if not filtered_df.empty:
        # Create track summary for the table
        track_summary = filtered_df.groupby('track_id').agg({
            'timestep': ['max', 'count'],
            'centroid_lat': 'last',
            'centroid_lon': 'last',
            'max_windspeed': ['max', 'last'],
            'area_km2': ['max', 'last'],
            'quality_score': 'mean',
            'development_stage': 'last',
            'precipitation': 'max'
        }).round(2)
        
        # Flatten column names
        track_summary.columns = [
            'Latest_Timestep', 'Duration', 'Current_Lat', 'Current_Lon',
            'Peak_Windspeed', 'Current_Windspeed', 'Peak_Area', 'Current_Area',
            'Avg_Quality', 'Current_Stage', 'Max_Precipitation'
        ]
        
        # Add status indicators
        track_summary['Status'] = track_summary.apply(
            lambda row: '🟢 Active' if row['Current_Windspeed'] > 30 else '🟡 Weak' if row['Current_Windspeed'] > 15 else '🔴 Inactive', axis=1
        )
        
        # Add alert level
        track_summary['Alert'] = track_summary.apply(
            lambda row: '🔴 High' if row['Current_Windspeed'] > 70 else '🟡 Medium' if row['Current_Windspeed'] > 50 else '🟢 Low', axis=1
        )
        
        # Reset index to make track_id a column
        track_summary = track_summary.reset_index()
        
        # Select key columns for the summary table
        summary_columns = ['track_id', 'Status', 'Alert', 'Current_Windspeed', 'Current_Area', 'Current_Stage', 'Duration']
        available_summary_columns = [col for col in summary_columns if col in track_summary.columns]
        track_summary_display = track_summary[available_summary_columns]
        
        # Format the data for display
        display_data = track_summary_display.copy()
        
        # Format windspeed
        if 'Current_Windspeed' in display_data.columns:
            display_data['Current_Windspeed'] = display_data['Current_Windspeed'].apply(lambda x: f"{x:.1f} km/h")
        
        # Format area
        if 'Current_Area' in display_data.columns:
            display_data['Current_Area'] = display_data['Current_Area'].apply(lambda x: f"{x/1000:.1f}k km²")
        
        # Display the table
        st.dataframe(
            display_data,
            use_container_width=True,
            height=300
        )
        
        # Add quick filters
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'Status' in track_summary_display.columns:
                active_tracks = len(track_summary_display[track_summary_display['Status'] == '🟢 Active'])
                st.metric("🟢 Active Tracks", active_tracks)
        
        with col2:
            if 'Alert' in track_summary_display.columns:
                high_alert = len(track_summary_display[track_summary_display['Alert'] == '🔴 High'])
                st.metric("🔴 High Alert", high_alert)
        
        with col3:
            if 'Current_Stage' in track_summary_display.columns:
                mature_tracks = len(track_summary_display[track_summary_display['Current_Stage'] == 'Mature'])
                st.metric("🌪️ Mature Tracks", mature_tracks)
    else:
        st.warning("No track data available")

# Real-time Track Monitoring Table
with tab2:
    st.header("📊 Real-time Track Monitoring")
    
    # Create comprehensive track monitoring table
    if not filtered_df.empty:
        # Group by track_id to get latest status for each track
        track_summary = filtered_df.groupby('track_id').agg({
            'timestep': ['max', 'count'],
            'centroid_lat': 'last',
            'centroid_lon': 'last',
            'max_windspeed': ['max', 'last'],
            'area_km2': ['max', 'last'],
            'quality_score': 'mean',
            'development_stage': 'last',
            'precipitation': 'max'
        }).round(2)
        
        # Flatten column names
        track_summary.columns = [
            'Latest_Timestep', 'Duration', 'Current_Lat', 'Current_Lon',
            'Peak_Windspeed', 'Current_Windspeed', 'Peak_Area', 'Current_Area',
            'Avg_Quality', 'Current_Stage', 'Max_Precipitation'
        ]
        
        # Add status indicators
        track_summary['Status'] = track_summary.apply(
            lambda row: '🟢 Active' if row['Current_Windspeed'] > 30 else '🟡 Weak' if row['Current_Windspeed'] > 15 else '🔴 Inactive', axis=1
        )
        
        # Add alert level
        track_summary['Alert_Level'] = track_summary.apply(
            lambda row: '🔴 High' if row['Current_Windspeed'] > 70 else '🟡 Medium' if row['Current_Windspeed'] > 50 else '🟢 Low', axis=1
        )
        
        # Add trend indicators
        track_summary['Trend'] = track_summary.apply(
            lambda row: '📈 Intensifying' if row['Current_Windspeed'] > row['Peak_Windspeed'] * 0.9 else '📉 Weakening' if row['Current_Windspeed'] < row['Peak_Windspeed'] * 0.7 else '➡️ Stable', axis=1
        )
        
        # Reset index to make track_id a column
        track_summary = track_summary.reset_index()
        
        # Reorder columns for better display
        display_columns = [
            'track_id', 'Status', 'Alert_Level', 'Trend', 'Current_Lat', 'Current_Lon',
            'Current_Windspeed', 'Peak_Windspeed', 'Current_Area', 'Peak_Area',
            'Current_Stage', 'Duration', 'Avg_Quality', 'Max_Precipitation'
        ]
        
        # Filter columns that exist
        available_columns = [col for col in display_columns if col in track_summary.columns]
        track_summary_display = track_summary[available_columns]
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.multiselect(
                "Filter by Status:",
                options=track_summary_display['Status'].unique(),
                default=track_summary_display['Status'].unique()
            )
        
        with col2:
            alert_filter = st.multiselect(
                "Filter by Alert Level:",
                options=track_summary_display['Alert_Level'].unique(),
                default=track_summary_display['Alert_Level'].unique()
            )
        
        with col3:
            stage_filter = st.multiselect(
                "Filter by Development Stage:",
                options=track_summary_display['Current_Stage'].unique() if 'Current_Stage' in track_summary_display.columns else [],
                default=track_summary_display['Current_Stage'].unique() if 'Current_Stage' in track_summary_display.columns else []
            )
        
        # Apply filters
        filtered_summary = track_summary_display[
            (track_summary_display['Status'].isin(status_filter)) &
            (track_summary_display['Alert_Level'].isin(alert_filter))
        ]
        
        if 'Current_Stage' in filtered_summary.columns and stage_filter:
            filtered_summary = filtered_summary[filtered_summary['Current_Stage'].isin(stage_filter)]
        
        # Display the table with enhanced styling
        st.subheader(f"Track Monitoring Dashboard ({len(filtered_summary)} tracks)")
        
        # Add search functionality
        search_term = st.text_input("🔍 Search tracks by ID:", "")
        if search_term:
            filtered_summary = filtered_summary[filtered_summary['track_id'].str.contains(search_term, case=False)]
        
        # Display table with better formatting
        if not filtered_summary.empty:
            # Format the data for display
            display_data = filtered_summary.copy()
            
            # Format coordinates
            if 'Current_Lat' in display_data.columns:
                display_data['Current_Lat'] = display_data['Current_Lat'].apply(lambda x: f"{x:.2f}°")
            if 'Current_Lon' in display_data.columns:
                display_data['Current_Lon'] = display_data['Current_Lon'].apply(lambda x: f"{x:.2f}°")
            
            # Format windspeed
            if 'Current_Windspeed' in display_data.columns:
                display_data['Current_Windspeed'] = display_data['Current_Windspeed'].apply(lambda x: f"{x:.1f} km/h")
            if 'Peak_Windspeed' in display_data.columns:
                display_data['Peak_Windspeed'] = display_data['Peak_Windspeed'].apply(lambda x: f"{x:.1f} km/h")
            
            # Format area
            if 'Current_Area' in display_data.columns:
                display_data['Current_Area'] = display_data['Current_Area'].apply(lambda x: f"{x/1000:.1f}k km²")
            if 'Peak_Area' in display_data.columns:
                display_data['Peak_Area'] = display_data['Peak_Area'].apply(lambda x: f"{x/1000:.1f}k km²")
            
            # Format quality
            if 'Avg_Quality' in display_data.columns:
                display_data['Avg_Quality'] = display_data['Avg_Quality'].apply(lambda x: f"{x:.2f}")
            
            # Format precipitation
            if 'Max_Precipitation' in display_data.columns:
                display_data['Max_Precipitation'] = display_data['Max_Precipitation'].apply(lambda x: f"{x:.1f} mm/h")
            
            # Display the table
            st.dataframe(
                display_data,
                use_container_width=True,
                height=400
            )
            
            # Add export functionality
            col1, col2 = st.columns(2)
            with col1:
                csv_data = filtered_summary.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    file_name=f"track_monitoring_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Add summary statistics
                st.metric("High Alert Tracks", len(filtered_summary[filtered_summary['Alert_Level'] == '🔴 High']))
            
            # Add real-time alerts section
            st.subheader("🚨 Real-time Alerts")
            
            # Get high alert tracks
            high_alert_tracks = filtered_summary[filtered_summary['Alert_Level'] == '🔴 High']
            
            if not high_alert_tracks.empty:
                for _, track in high_alert_tracks.iterrows():
                    with st.expander(f"🚨 {track['track_id']} - High Alert"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Windspeed", track['Current_Windspeed'])
                        with col2:
                            st.metric("Peak Windspeed", track['Peak_Windspeed'])
                        with col3:
                            st.metric("Duration", f"{track['Duration']} timesteps")
                        
                        st.info(f"Track {track['track_id']} is currently at high alert level with windspeed of {track['Current_Windspeed']}")
            else:
                st.success("✅ No high alert tracks currently active")
        
        else:
            st.warning("No tracks match the current filters")
    
    else:
        st.warning("No data available for track monitoring")

# Cluster Analytics
with tab3:
    st.header("Cluster Property Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Intensity Distribution")
        fig = px.histogram(
            filtered_df, 
            x='max_windspeed', 
            nbins=20,
            color='intensity_category',
            labels={'max_windspeed': 'Maximum Windspeed (km/h)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Precipitation vs. Size")
        fig = px.scatter(
            filtered_df,
            x='area_km2',
            y='precipitation',
            size='max_windspeed',
            color='intensity_category',
            hover_name='track_id'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Development Timeline")
        if 'development_stage' in filtered_df.columns:
            timeline_df = filtered_df.groupby(['timestep', 'development_stage'], observed=True).size().reset_index(name='count')
            fig = px.area(
                timeline_df,
                x='timestep',
                y='count',
                color='development_stage',
                labels={'timestep': 'Timestep', 'count': 'Cluster Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Development stage data not available")
        
        st.subheader("Quality Metrics")
        fig = go.Figure()
        if 'quality_score' in filtered_df.columns:
            fig.add_trace(go.Box(y=filtered_df['quality_score'], name='Quality Score'))
        if 'mean_irbt' in filtered_df.columns:
            fig.add_trace(go.Box(y=filtered_df['mean_irbt'], name='Mean IRBT'))
        st.plotly_chart(fig, use_container_width=True)

# Track Explorer
with tab4:
    st.header("Track Analysis Explorer")
    
    if selected_track == 'All':
        st.info("Select a specific track from the sidebar to enable detailed analysis")
    else:
        track_df = filtered_df[filtered_df['track_id'] == selected_track]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"Track {selected_track} Profile")
            st.metric("Maximum Intensity", f"{track_df['max_windspeed'].max()} km/h")
            if 'precipitation' in track_df.columns:
                st.metric("Peak Precipitation", f"{track_df['precipitation'].max()} mm/h")
            st.metric("Duration", f"{len(track_df)} timesteps")
            if 'development_stage' in track_df.columns:
                st.metric("Current Stage", track_df['development_stage'].iloc[-1])
            
            st.subheader("Track Forecast")
            # Simplified forecast model
            current_wind = track_df['max_windspeed'].iloc[-1]
            forecast = [
                current_wind * 1.1,
                current_wind * 1.25,
                current_wind * 1.15,
                current_wind * 1.0,
                current_wind * 0.9
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, 6)),
                y=forecast,
                mode='lines+markers',
                name='Forecast'
            ))
            fig.update_layout(
                title="5-Step Windspeed Forecast",
                xaxis_title="Timesteps Ahead",
                yaxis_title="Windspeed (km/h)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Track Path")
            
            # Validate coordinates for track visualization
            valid_track_coords = (
                (track_df['centroid_lat'] >= -90) & (track_df['centroid_lat'] <= 90) &
                (track_df['centroid_lon'] >= -180) & (track_df['centroid_lon'] <= 180) &
                track_df['centroid_lat'].notna() & track_df['centroid_lon'].notna()
            )
            
            if not valid_track_coords.all():
                invalid_count = (~valid_track_coords).sum()
                st.warning(f"Filtering out {invalid_count} invalid coordinates for track visualization")
                track_df_clean = track_df[valid_track_coords]
            else:
                track_df_clean = track_df
            
            if not track_df_clean.empty:
                fig = px.scatter_map(
                    track_df_clean,
                    lat='centroid_lat',
                    lon='centroid_lon',
                    hover_name='timestep',
                    zoom=3,
                    color='max_windspeed',
                    size='area_km2',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                fig.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig, use_container_width=True)
                
                # Add line connection between points using scattermap instead of scattermapbox
                if len(track_df_clean) > 1:
                    fig_line = go.Figure()
                    fig_line.add_trace(go.Scattermap(
                        lat=track_df_clean['centroid_lat'],
                        lon=track_df_clean['centroid_lon'],
                        mode='lines',
                        line=dict(width=3, color='red'),
                        name='Track Path',
                        showlegend=False
                    ))
                    fig_line.update_layout(
                        mapbox=dict(
                            style="open-street-map",
                            center=dict(lat=track_df_clean['centroid_lat'].mean(), lon=track_df_clean['centroid_lon'].mean()),
                            zoom=3
                        ),
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=400
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.error("No valid coordinates available for track visualization")
            
            # Add intensity evolution chart
            st.subheader("Intensity Evolution")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=track_df['timestep'],
                y=track_df['max_windspeed'],
                mode='lines+markers',
                name='Windspeed'
            ))
            if 'precipitation' in track_df.columns:
                fig.add_trace(go.Scatter(
                    x=track_df['timestep'],
                    y=track_df['precipitation'],
                    mode='lines',
                    name='Precipitation',
                    yaxis='y2'
                ))
            fig.update_layout(
                title="Track Development",
                xaxis_title="Timestep",
                yaxis_title="Max Windspeed (km/h)",
                yaxis2=dict(
                    title="Precipitation (mm/h)",
                    overlaying='y',
                    side='right'
                )
            )
            st.plotly_chart(fig, use_container_width=True)

# Report Generation
with tab5:
    st.header("Report Generation")
    
    report_name = st.text_input("Report Name", "Cloud_Analysis_Report")
    
    # Enhanced report options
    col1, col2 = st.columns(2)
    
    with col1:
        include_sections = st.multiselect(
            "Include Sections:",
            ["Summary", "Alert Clusters", "Track Analysis", "Quality Metrics", "3D Analysis"],
            ["Summary", "Alert Clusters"]
        )
    
    with col2:
        report_types = st.multiselect(
            "Output Formats:",
            ["PDF", "CSV", "JSON", "NetCDF"],
            default=["PDF"]
        )
    
    if st.button("Generate Reports"):
        with st.spinner("Generating reports..."):
            # Generate PDF Report
            if "PDF" in report_types:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                        c = canvas.Canvas(tmpfile.name, pagesize=letter)
                        
                        # Title
                        c.setFont("Helvetica-Bold", 16)
                        c.drawString(100, 750, f"Tropical Cloud Cluster Analysis Report")
                        c.setFont("Helvetica", 12)
                        c.drawString(100, 730, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                        c.drawString(100, 710, f"Time Range: {time_range[0]} to {time_range[1]}")
                        
                        # Summary
                        if "Summary" in include_sections:
                            c.setFont("Helvetica-Bold", 14)
                            c.drawString(100, 680, "Summary Statistics")
                            c.setFont("Helvetica", 12)
                            c.drawString(120, 660, f"Total Clusters: {len(filtered_df)}")
                            c.drawString(120, 640, f"Active Tracks: {filtered_df['track_id'].nunique()}")
                            c.drawString(120, 620, f"High Intensity Clusters: {len(alert_df)}")
                            if 'precipitation' in filtered_df.columns:
                                c.drawString(120, 600, f"Average Precipitation: {filtered_df['precipitation'].mean():.1f} mm/h")
                            if 'quality_score' in filtered_df.columns:
                                c.drawString(120, 580, f"Average Quality Score: {filtered_df['quality_score'].mean():.2f}")
                        
                        # Alerts
                        if "Alert Clusters" in include_sections and not alert_df.empty:
                            c.showPage()
                            c.setFont("Helvetica-Bold", 14)
                            c.drawString(100, 750, "High Intensity Clusters")
                            c.setFont("Helvetica", 10)
                            
                            y_pos = 730
                            headers = ["Track ID", "Timestep", "Lat", "Lon", "Windspeed"]
                            col_positions = [100, 180, 250, 320, 390]
                            
                            # Table headers
                            for i, header in enumerate(headers):
                                c.drawString(col_positions[i], y_pos, header)
                            
                            # Table rows
                            for _, row in alert_df.iterrows():
                                y_pos -= 20
                                c.drawString(col_positions[0], y_pos, str(row['track_id']))
                                c.drawString(col_positions[1], y_pos, str(row['timestep']))
                                c.drawString(col_positions[2], y_pos, f"{row['centroid_lat']:.2f}")
                                c.drawString(col_positions[3], y_pos, f"{row['centroid_lon']:.2f}")
                                c.drawString(col_positions[4], y_pos, f"{row['max_windspeed']:.1f}")
                        
                        c.save()
                        
                        with open(tmpfile.name, "rb") as f:
                            st.success("PDF report generated successfully!")
                            st.download_button(
                                "Download PDF Report",
                                f,
                                file_name=f"{report_name}.pdf",
                                mime="application/pdf"
                            )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
            
            # Generate CSV Report
            if "CSV" in report_types:
                try:
                    csv_data = filtered_df.to_csv(index=False)
                    st.success("CSV report generated successfully!")
                    st.download_button(
                        "Download CSV Report",
                        csv_data,
                        file_name=f"{report_name}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error generating CSV: {str(e)}")
            
            # Generate JSON Report
            if "JSON" in report_types:
                try:
                    json_data = filtered_df.to_json(orient='records', indent=2)
                    st.success("JSON report generated successfully!")
                    st.download_button(
                        "Download JSON Report",
                        json_data,
                        file_name=f"{report_name}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Error generating JSON: {str(e)}")
            
            # Generate NetCDF Report
            if "NetCDF" in report_types:
                try:
                    import xarray as xr
                    
                    # Create xarray dataset
                    ds = xr.Dataset.from_dataframe(filtered_df)
                    
                    # Add metadata
                    ds.attrs['title'] = f"Tropical Cloud Cluster Analysis - {report_name}"
                    ds.attrs['creation_date'] = datetime.now().isoformat()
                    ds.attrs['description'] = "Tropical cloud cluster detection and analysis results"
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmpfile:
                        ds.to_netcdf(tmpfile.name)
                        
                        with open(tmpfile.name, "rb") as f:
                            st.success("NetCDF report generated successfully!")
                            st.download_button(
                                "Download NetCDF Report",
                                f,
                                file_name=f"{report_name}.nc",
                                mime="application/octet-stream"
                            )
                except Exception as e:
                    st.error(f"Error generating NetCDF: {str(e)}")
                    st.info("NetCDF generation requires xarray library. Install with: pip install xarray")

# 3D Analysis
with tab6:
    st.header("3D Cluster Visualization")
    
    if filtered_df.empty:
        st.warning("No data available for 3D visualization")
    else:
        # Ensure we have the required columns
        required_cols = ['centroid_lon', 'centroid_lat', 'timestep', 'max_windspeed', 'area_km2']
        missing_cols = [col for col in required_cols if col not in filtered_df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns for 3D visualization: {missing_cols}")
        else:
            # Sample data if too large
            if len(filtered_df) > 1000:
                st.warning("Too many points for 3D visualization. Applying sampling.")
                sample_df = filtered_df.sample(1000)
            else:
                sample_df = filtered_df
            
            # Create enhanced 3D scatter plot
            fig = go.Figure(data=[go.Scatter3d(
                x=sample_df['centroid_lon'],
                y=sample_df['centroid_lat'],
                z=sample_df['timestep'],
                mode='markers',
                marker=dict(
                    size=sample_df['area_km2']/50,  # Adjusted size scaling
                    sizemode='diameter',
                    color=sample_df['max_windspeed'],
                    colorscale='Rainbow',  # Enhanced colorscale
                    opacity=0.9,
                    colorbar=dict(
                        title=dict(text='Windspeed (km/h)'),
                        thickness=20,
                        len=0.8,
                        x=1.1
                    ),
                    showscale=True
                ),
                hovertext=sample_df.apply(
                    lambda r: f"Track {r['track_id']}<br>Winds: {r['max_windspeed']:.1f} km/h<br>Area: {r['area_km2']:.0f} km²<br>Quality: {r.get('quality_score', 0):.2f}",
                    axis=1
                ),
                hovertemplate='<b>%{hovertext}</b><extra></extra>'
            )])
            
            fig.update_layout(
                scene=dict(
                    xaxis_title='Longitude',
                    yaxis_title='Latitude',
                    zaxis_title='Timestep',
                    zaxis_type="log",  # Log scale for timestep
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5),
                        center=dict(x=0, y=0, z=0)
                    ),
                    aspectmode='manual',
                    aspectratio=dict(x=2, y=1, z=1)
                ),
                height=700,  # Increased height
                title="3D Cluster Evolution - Enhanced View",
                template='plotly_dark',
                showlegend=False,
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add enhanced summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Points", len(sample_df))
            with col2:
                st.metric("Avg Windspeed", f"{sample_df['max_windspeed'].mean():.1f} km/h")
            with col3:
                st.metric("Avg Area", f"{sample_df['area_km2'].mean():.0f} km²")
            with col4:
                if 'quality_score' in sample_df.columns:
                    st.metric("Avg Quality", f"{sample_df['quality_score'].mean():.2f}")
                else:
                    st.metric("Time Range", f"{sample_df['timestep'].min()}-{sample_df['timestep'].max()}")
            
            # Add quality distribution chart
            if 'quality_score' in sample_df.columns:
                st.subheader("Data Quality Distribution")
                fig_quality = px.histogram(
                    sample_df,
                    x='quality_score',
                    nbins=20,
                    color_discrete_sequence=['#00ff88'],
                    labels={'quality_score': 'Quality Score', 'count': 'Frequency'}
                )
                fig_quality.update_layout(
                    title="Distribution of Data Quality Scores",
                    xaxis_title="Quality Score",
                    yaxis_title="Frequency",
                    template='plotly_dark'
                )
                st.plotly_chart(fig_quality, use_container_width=True)

# Footer
st.divider()
st.caption("🌪️ Tropical Cloud Monitoring System | Version 2.0 | Real-time Analysis | Enhanced Dashboard") 