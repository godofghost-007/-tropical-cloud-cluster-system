import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from detection import calculate_cyclogenesis_potential
from tracking import forecast_tracks, generate_alerts
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Enhanced Tropical Cloud Cluster Tracker",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .quality-high { color: #28a745; }
    .quality-medium { color: #ffc107; }
    .quality-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Load tracked data with error handling
@st.cache_data
def load_tracked_data():
    try:
        # Try multiple possible file locations in order of preference
        possible_files = [
            'outputs/tracks/final_tracks.csv',
            'outputs/tracks/tracked_clusters.csv',
            'outputs/all_detections.csv',
            'outputs/cloud_clusters.csv'
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                st.success(f"✅ Loaded data from {file_path}")
                return df
        
        st.error("❌ No tracked data found. Please run the processing pipeline first.")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()

# Load data
df = load_tracked_data()

# Enhanced sidebar with data source information
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data Source Information")

if not df.empty:
    # Data format information
    if 'file_format' in df.columns:
        source = df['file_format'].iloc[0]
        version = df.get('data_version', ['1.0']).iloc[0] if 'data_version' in df.columns else '1.0'
        st.sidebar.metric("Data Format", f"{source} (v{version})")
    else:
        st.sidebar.info("Format: Standard NetCDF")
    
    # Data quality metrics
    if 'data_quality' in df.columns:
        quality_score = df['data_quality'].mean() * 100
        st.sidebar.metric("Data Quality Score", f"{quality_score:.1f}%")
        
        # Quality indicator
        if quality_score >= 80:
            quality_class = "quality-high"
            quality_icon = "🟢"
        elif quality_score >= 60:
            quality_class = "quality-medium"
            quality_icon = "🟡"
        else:
            quality_class = "quality-low"
            quality_icon = "🔴"
        
        st.sidebar.markdown(f'<p class="{quality_class}">{quality_icon} Quality: {"High" if quality_score >= 80 else "Medium" if quality_score >= 60 else "Low"}</p>', 
                           unsafe_allow_html=True)
        
        # Quality progress bar
        st.sidebar.progress(quality_score / 100, text="Data Quality")
    else:
        st.sidebar.info("Quality metrics not available")
    
    # Processing statistics
    if 'processing_time' in df.columns:
        avg_time = df['processing_time'].mean()
        st.sidebar.metric("Avg Processing Time", f"{avg_time:.2f}s")
    
    # Data coverage
    total_clusters = len(df)
    # Handle different possible track ID column names
    track_id_col = None
    for col in ['track_id', 'trackid', 'track', 'cluster_id']:
        if col in df.columns:
            track_id_col = col
            break
    
    unique_tracks = df[track_id_col].nunique() if track_id_col else 0
    st.sidebar.metric("Total Clusters", total_clusters)
    st.sidebar.metric("Unique Tracks", unique_tracks)
    
    # Metadata summary
    meta_cols = [col for col in df.columns if col.startswith('meta_')]
    if meta_cols:
        with st.sidebar.expander("📋 Technical Metadata"):
            for col in meta_cols[:5]:  # Show first 5 metadata columns
                if col in df.columns:
                    unique_vals = df[col].unique()
                    if len(unique_vals) == 1:
                        st.write(f"**{col[5:]}**: {unique_vals[0]}")
                    else:
                        st.write(f"**{col[5:]}**: {len(unique_vals)} unique values")

else:
    st.sidebar.warning("⚠️ No data available")

# Main dashboard
st.title('🌪️ Enhanced Tropical Cloud Cluster Tracker')
st.markdown("""
Advanced visualization and analysis of tropical cloud clusters with multi-format satellite data support.
""")

# CYCLOGENESIS RISK ASSESSMENT
st.sidebar.markdown("---")
st.sidebar.header("🎯 Risk Analysis")
risk_threshold = st.sidebar.slider("Cyclogenesis Risk Threshold", 0.0, 1.0, 0.65)

# Check if required columns exist
required_columns = ['min_tb', 'area_km2', 'compactness', 'convective_intensity', 'std_cloud_height']
if all(col in df.columns for col in required_columns):
    df['cyclogenesis_risk'] = df.apply(calculate_cyclogenesis_potential, axis=1)
    high_risk = df[df['cyclogenesis_risk'] > risk_threshold]
    
    # Display risk clusters
    if not high_risk.empty:
        st.warning(f"🚨 {len(high_risk)} high-risk clusters detected!")
        
        # Enhanced risk visualization with Plotly
        fig_risk = go.Figure()
        
        # Get track ID column name
        track_id_col = None
        for col in ['track_id', 'trackid', 'track', 'cluster_id']:
            if col in high_risk.columns:
                track_id_col = col
                break
        
        if track_id_col:
            for track_id in high_risk[track_id_col].unique():
                track = df[df[track_id_col] == track_id]
            fig_risk.add_trace(go.Scatter(
                x=track['timestep'],
                y=track['cyclogenesis_risk'],
                mode='lines+markers',
                name=f"Track {track_id}",
                line=dict(width=2),
                marker=dict(size=6)
            ))
        
        fig_risk.add_hline(y=risk_threshold, line_dash="dash", line_color="red",
                          annotation_text=f"Threshold ({risk_threshold})")
        
        fig_risk.update_layout(
            title="High-Risk Cluster Evolution",
            xaxis_title="Timestep",
            yaxis_title="Cyclogenesis Risk",
            height=400
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Risk summary table
        if track_id_col:
            risk_summary = high_risk.groupby(track_id_col).agg({
                'cyclogenesis_risk': ['max', 'mean'],
                'timestep': 'count'
            }).round(3)
            risk_summary.columns = ['Max Risk', 'Avg Risk', 'Timesteps']
            st.dataframe(risk_summary)
        
    else:
        st.success("✅ No high-risk clusters detected")
else:
    st.warning("⚠️ Advanced properties not available. Reprocess data with updated detection.py")

# FORECAST MODULE
st.sidebar.markdown("---")
st.sidebar.header("🔮 Forecasting")
forecast_hours = st.sidebar.slider("Forecast Hours", 6, 72, 24)
if st.sidebar.button("Generate Forecast"):
    try:
        forecast_df = forecast_tracks(df, forecast_hours)
        combined_df = pd.concat([df, forecast_df])
        
        # Enhanced forecast visualization with Plotly
        fig_forecast = go.Figure()
        
        # Get track ID column name
        track_id_col = None
        for col in ['track_id', 'trackid', 'track', 'cluster_id']:
            if col in combined_df.columns:
                track_id_col = col
                break
        
        if track_id_col:
            for track_id in combined_df[track_id_col].unique():
                track = combined_df[combined_df[track_id_col] == track_id]
                actual = track[track.get('forecast', False) == False]
                forecast = track[track.get('forecast', False) == True]
            
            # Plot actual track
            fig_forecast.add_trace(go.Scatter(
                x=actual['center_lon'],
                y=actual['center_lat'],
                mode='lines+markers',
                name=f"Track {track_id} (Actual)",
                line=dict(width=2),
                marker=dict(size=4)
            ))
            
            # Plot forecast
            if not forecast.empty:
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['center_lon'],
                    y=forecast['center_lat'],
                    mode='lines+markers',
                    name=f"Track {track_id} (Forecast)",
                    line=dict(dash='dash', width=1),
                    marker=dict(symbol='x', size=3),
                    showlegend=False
                ))
        
        fig_forecast.update_layout(
            title=f"Cluster Tracks with {forecast_hours}h Forecast",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            xaxis=dict(range=[40, 100]),
            yaxis=dict(range=[-30, 30]),
            height=600
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.session_state.forecast_df = forecast_df
        
    except Exception as e:
        st.error(f"Forecast generation failed: {str(e)}")

# Main visualization section
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Controls")

if not df.empty:
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

    if len(timesteps) == 0:
        st.warning("No timesteps available. Please reprocess your data.")
        st.stop()
    elif len(timesteps) == 1:
        st.warning(f"Only one timestep available: {timesteps[0]}. Please reprocess your data for more timesteps.")
        selected_timestep = timesteps[0]
    else:
        min_ts = min(timesteps)
        max_ts = max(timesteps)
        selected_timestep = st.sidebar.slider(
            "Select Timestep:",
            min_value=min_ts,
            max_value=max_ts,
            value=min_ts
        )
    
    # Filter data for selected timestep
    df_timestep = df[df['timestep'] == selected_timestep]
    
    # Main columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Enhanced map visualization with Plotly
        st.subheader(f"🗺️ Cluster Map - Timestep {selected_timestep}")
        
        if not df_timestep.empty:
            # Create scatter plot with size and color mapping
            fig_map = px.scatter(
                df_timestep,
                x='center_lon',
                y='center_lat',
                size='area_km2',
                color='cloud_top_height_km',
                hover_data=['track_id' if 'track_id' in df_timestep.columns else 'cluster_id', 'min_tb', 'area_km2'],
                title='Cloud Cluster Positions',
                color_continuous_scale='viridis',
                size_max=20
            )
            
            fig_map.update_layout(
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                xaxis=dict(range=[40, 100]),
                yaxis=dict(range=[-30, 30]),
                height=500
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No clusters detected in this timestep")
    
    with col2:
        # Enhanced track explorer
        st.subheader("🔍 Track Explorer")
        
        if not df_timestep.empty:
            # Get track ID column name
            track_id_col = None
            for col in ['track_id', 'trackid', 'track', 'cluster_id']:
                if col in df_timestep.columns:
                    track_id_col = col
                    break
            
            if track_id_col:
                selected_track = st.selectbox(
                    "Select a Track", 
                    df_timestep[track_id_col].unique()
                )
                
                # Get full track history
                track_df = df[df[track_id_col] == selected_track].sort_values('timestep')
                
                if not track_df.empty:
                    # Current metrics
                    current = track_df.iloc[-1]
                    col_metric1, col_metric2 = st.columns(2)
                    
                    with col_metric1:
                        st.metric("Cloud Top Height", 
                                  f"{current['cloud_top_height_km']:.1f} km")
                    
                    with col_metric2:
                        st.metric("Area", f"{current['area_km2']:.0f} km²")
                    
                    # Track evolution plot
                    fig_evolution = go.Figure()
                    
                    fig_evolution.add_trace(go.Scatter(
                        x=track_df['timestep'],
                        y=track_df['cloud_top_height_km'],
                        mode='lines+markers',
                        name='Cloud Top Height',
                        line=dict(width=2)
                    ))
                    
                    fig_evolution.update_layout(
                        title=f'Track {selected_track} Evolution',
                        xaxis_title='Timestep',
                        yaxis_title='Height (km)',
                        height=300
                    )
                    
                    st.plotly_chart(fig_evolution, use_container_width=True)
                    
                    # Track properties table
                    st.write("**Track Properties**")
                    display_cols = ['timestep', 'area_km2', 'min_tb', 'cloud_top_height_km']
                    available_cols = [col for col in display_cols if col in track_df.columns]
                    st.dataframe(track_df[available_cols].tail(5))
                    
                else:
                    st.warning("No data available for selected track")
            else:
                st.warning("No track ID column found in data")
        else:
            st.info("No clusters detected in this timestep")

# Data summary section
st.markdown("---")
st.subheader("📈 Data Summary")

if not df.empty:
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    
    with col_sum1:
        st.metric("Total Clusters", len(df))
    
    with col_sum2:
        # Get track ID column name
        track_id_col = None
        for col in ['track_id', 'trackid', 'track', 'cluster_id']:
            if col in df.columns:
                track_id_col = col
                break
        
        unique_tracks = df[track_id_col].nunique() if track_id_col else 0
        st.metric("Unique Tracks", unique_tracks)
    
    with col_sum3:
        if 'cloud_top_height_km' in df.columns:
            avg_height = df['cloud_top_height_km'].mean()
            st.metric("Avg Cloud Height", f"{avg_height:.1f} km")
        else:
            st.metric("Avg Cloud Height", "N/A")
    
    with col_sum4:
        if 'area_km2' in df.columns:
            avg_area = df['area_km2'].mean()
            st.metric("Avg Area", f"{avg_area:.0f} km²")
        else:
            st.metric("Avg Area", "N/A")
    
    # Show raw data with pagination
    st.subheader("📋 Raw Data")
    if len(df_timestep) > 10:
        st.info(f"Showing {len(df_timestep)} clusters. Use pagination below.")
    
    st.dataframe(df_timestep, use_container_width=True)

# Operational Alerts
st.sidebar.markdown("---")
st.sidebar.header("🚨 Operational Alerts")
if st.sidebar.button("Check for Alerts"):
    try:
        alerts = generate_alerts(df)
        if alerts:
            for alert in alerts:
                if alert['severity'] == 'High':
                    st.sidebar.error(f"🔴 **{alert['type']}**: {alert['message']}")
                elif alert['severity'] == 'Medium':
                    st.sidebar.warning(f"🟡 **{alert['type']}**: {alert['message']}")
                else:
                    st.sidebar.info(f"🔵 **{alert['type']}**: {alert['message']}")
        else:
            st.sidebar.success("🟢 No critical alerts at this time")
    except Exception as e:
        st.sidebar.error(f"Alert generation failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Enhanced Tropical Cloud Cluster Tracker | Multi-format Satellite Data Support</p>
    <p>Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True) 