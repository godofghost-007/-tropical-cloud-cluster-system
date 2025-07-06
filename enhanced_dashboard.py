#!/usr/bin/env python3
"""
Enhanced Tropical Cloud Cluster Dashboard
Combines best UX features with 24 timesteps and interactive maps
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import time
import yaml
import psutil
import folium
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from streamlit_folium import folium_static
from folium.plugins import MiniMap, Fullscreen, MarkerCluster
import glob

# Set page configuration
st.set_page_config(
    page_title="🌪️ Tropical Cloud Cluster Monitor",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Enhanced styling */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1d2e 100%);
        color: #f0f2f6;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0d47a1 0%, #2196f3 50%, #64b5f6 100%);
        padding: 2rem 1rem;
        border-radius: 0 0 20px 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling */
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
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2196f3, #64b5f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Alert styling */
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
    
    /* Enhanced Toggle styling */
    .toggle-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
        padding: 2rem;
        background: linear-gradient(135deg, #1e2130 0%, #2d3748 100%);
        border-radius: 25px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .toggle-btn {
        padding: 1.5rem 3rem;
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
        border: 2px solid transparent;
        color: #d1d5db;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        font-weight: 600;
        font-size: 1.2rem;
        position: relative;
        overflow: hidden;
        min-width: 180px;
    }
    
    .toggle-btn:hover {
        background: linear-gradient(135deg, #4b5563 0%, #6b7280 100%);
        color: #ffffff;
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(33, 150, 243, 0.4);
    }
    
    .toggle-btn.active {
        background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
        color: white;
        font-weight: bold;
        border-color: #64b5f6;
        box-shadow: 0 16px 45px rgba(33, 150, 243, 0.5);
        transform: translateY(-4px);
    }
    
    /* Map container styling */
    .map-container {
        background: linear-gradient(135deg, #1e2130 0%, #2d3748 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Timeline styling */
    .timeline-container {
        background: linear-gradient(135deg, #1e2130 0%, #2d3748 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: #4caf50;
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
    }
    
    .status-warning {
        background-color: #ff9800;
        box-shadow: 0 0 10px rgba(255, 152, 0, 0.5);
    }
    
    .status-danger {
        background-color: #f44336;
        box-shadow: 0 0 10px rgba(244, 67, 54, 0.5);
    }
</style>
""", unsafe_allow_html=True)

class EnhancedDashboard:
    def __init__(self):
        self.data_file = 'outputs/cloud_clusters.csv'
        self.maps_dir = 'outputs'
        self.df = self.load_data()
        self.last_updated = datetime.now()
        
    def load_data(self):
        """Load cluster data with error handling"""
        try:
            if os.path.exists(self.data_file):
                df = pd.read_csv(self.data_file)
                st.success(f"✅ Loaded {len(df)} clusters from {len(df['timestep'].unique())} timesteps")
                return df
            else:
                st.error(f"❌ Data file not found: {self.data_file}")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return pd.DataFrame()
    
    def get_system_stats(self):
        """Get system performance metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available': memory.available / (1024**3),  # GB
            'disk_percent': disk.percent,
            'disk_free': disk.free / (1024**3)  # GB
        }
    
    def create_interactive_map(self, selected_timestep=None):
        """Create interactive Folium map with cluster data"""
        if self.df.empty:
            return None
            
        # Filter data for selected timestep
        if selected_timestep is not None:
            df_map = self.df[self.df['timestep'] == selected_timestep]
        else:
            df_map = self.df
            
        if df_map.empty:
            return None
            
        # Create base map centered on data
        center_lat = df_map['center_lat'].mean()
        center_lon = df_map['center_lon'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='CartoDB dark_matter',
            control_scale=True
        )
        
        # Add cluster markers
        for _, row in df_map.iterrows():
            # Color based on cloud height
            height = row.get('cloud_top_height_km', 0)
            if height > 12:
                color = 'red'
            elif height > 8:
                color = 'orange'
            else:
                color = 'blue'
                
            # Popup content
            popup_content = f"""
            <div style="width: 200px;">
                <h4>🌪️ Cluster Details</h4>
                <p><strong>Location:</strong> {row['center_lat']:.2f}°N, {row['center_lon']:.2f}°E</p>
                <p><strong>Area:</strong> {row.get('area_km2', 0):.0f} km²</p>
                <p><strong>Cloud Height:</strong> {height:.1f} km</p>
                <p><strong>Min Temp:</strong> {row.get('min_tb', 0):.1f} K</p>
                <p><strong>Quality:</strong> {row.get('quality_score', 0):.2f}</p>
            </div>
            """
            
            folium.CircleMarker(
                location=[row['center_lat'], row['center_lon']],
                radius=row.get('area_km2', 1000) / 1000,  # Scale radius by area
                popup=folium.Popup(popup_content, max_width=300),
                color=color,
                fill=True,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
        
        # Add plugins
        folium.plugins.MiniMap().add_to(m)
        folium.plugins.Fullscreen().add_to(m)
        folium.plugins.MarkerCluster().add_to(m)
        
        return m
    
    def create_timeline_plot(self):
        """Create timeline visualization of cluster development"""
        if self.df.empty:
            return None
            
        fig = go.Figure()
        
        # Plot cluster count over time
        timestep_counts = self.df.groupby('timestep').size()
        
        fig.add_trace(go.Scatter(
            x=timestep_counts.index,
            y=timestep_counts.values,
            mode='lines+markers',
            name='Cluster Count',
            line=dict(color='#2196f3', width=3),
            marker=dict(size=8, color='#2196f3')
        ))
        
        # Add area trend
        if 'area_km2' in self.df.columns:
            area_trend = self.df.groupby('timestep')['area_km2'].mean()
            fig.add_trace(go.Scatter(
                x=area_trend.index,
                y=area_trend.values,
                mode='lines+markers',
                name='Avg Area (km²)',
                yaxis='y2',
                line=dict(color='#4caf50', width=2),
                marker=dict(size=6, color='#4caf50')
            ))
        
        fig.update_layout(
            title='🌪️ Cluster Development Timeline',
            xaxis_title='Timestep',
            yaxis_title='Number of Clusters',
            yaxis2=dict(title='Average Area (km²)', overlaying='y', side='right'),
            template='plotly_dark',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_3d_visualization(self):
        """Create 3D scatter plot of clusters"""
        if self.df.empty:
            return None
            
        fig = go.Figure(data=[go.Scatter3d(
            x=self.df['center_lon'],
            y=self.df['center_lat'],
            z=self.df.get('cloud_top_height_km', 0),
            mode='markers',
            marker=dict(
                size=self.df.get('area_km2', 1000) / 1000,
                color=self.df.get('min_tb', 240),
                colorscale='Viridis',
                opacity=0.8
            ),
            text=[f"Cluster {i}<br>Height: {h:.1f}km<br>Area: {a:.0f}km²" 
                  for i, (h, a) in enumerate(zip(self.df.get('cloud_top_height_km', 0), 
                                               self.df.get('area_km2', 0)))],
            hovertemplate='<b>%{text}</b><extra></extra>'
        )])
        
        fig.update_layout(
            title='🌪️ 3D Cluster Visualization',
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Cloud Height (km)'
            ),
            template='plotly_dark',
            height=600
        )
        
        return fig
    
    def display_timestep_maps(self):
        """Display timestep maps in a grid"""
        if self.df.empty:
            return
            
        # Get available map files
        map_files = glob.glob(os.path.join(self.maps_dir, 'tcc_detection_t*.png'))
        map_files.sort()
        
        if not map_files:
            st.warning("No timestep maps found")
            return
            
        st.subheader("🗺️ Timestep Maps")
        
        # Create columns for map display
        cols = st.columns(4)
        
        for i, map_file in enumerate(map_files[:12]):  # Show first 12 maps
            with cols[i % 4]:
                timestep = os.path.basename(map_file).split('_t')[1].split('.')[0]
                st.image(map_file, caption=f"Timestep {timestep}", use_column_width=True)
                
                # Add cluster info for this timestep
                timestep_data = self.df[self.df['timestep'] == int(timestep)]
                if not timestep_data.empty:
                    st.caption(f"Clusters: {len(timestep_data)}")
    
    def render_metrics(self):
        """Render key metrics"""
        if self.df.empty:
            return
            
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Total Clusters</div>
                <div class="metric-value">{}</div>
            </div>
            """.format(len(self.df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Timesteps</div>
                <div class="metric-value">{}</div>
            </div>
            """.format(len(self.df['timestep'].unique())), unsafe_allow_html=True)
        
        with col3:
            avg_area = self.df.get('area_km2', 0).mean()
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Avg Area (km²)</div>
                <div class="metric-value">{:.0f}</div>
            </div>
            """.format(avg_area), unsafe_allow_html=True)
        
        with col4:
            avg_height = self.df.get('cloud_top_height_km', 0).mean()
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Avg Height (km)</div>
                <div class="metric-value">{:.1f}</div>
            </div>
            """.format(avg_height), unsafe_allow_html=True)
    
    def render_alerts(self):
        """Render alerts for high-risk clusters"""
        if self.df.empty:
            return
            
        # Check for high-risk conditions
        high_risk = self.df[
            (self.df.get('cloud_top_height_km', 0) > 12) |
            (self.df.get('area_km2', 0) > 50000)
        ]
        
        if not high_risk.empty:
            st.markdown("""
            <div class="alert-card">
                <h3>⚠️ High-Risk Clusters Detected</h3>
                <p>Found {} clusters with elevated risk factors</p>
            </div>
            """.format(len(high_risk)), unsafe_allow_html=True)
    
    def render(self):
        """Main render function"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🌪️ Tropical Cloud Cluster Monitor</h1>
            <p>Advanced monitoring and analysis of tropical cloud clusters with 24-hour timestep data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar controls
        st.sidebar.title("🎛️ Controls")
        
        # Timestep selection
        if not self.df.empty:
            timesteps = sorted(self.df['timestep'].unique())
            selected_timestep = st.sidebar.selectbox(
                "Select Timestep",
                timesteps,
                index=0
            )
            
            st.sidebar.write(f"📊 {len(timesteps)} timesteps available")
        else:
            selected_timestep = None
        
        # View mode toggle
        st.markdown("""
        <div class="toggle-container">
            <button class="toggle-btn active" onclick="setViewMode('overview')">Overview</button>
            <button class="toggle-btn" onclick="setViewMode('analysis')">Analysis</button>
            <button class="toggle-btn" onclick="setViewMode('maps')">Maps</button>
        </div>
        """, unsafe_allow_html=True)
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Analysis", "🗺️ Maps"])
        
        with tab1:
            # Metrics
            self.render_metrics()
            
            # Alerts
            self.render_alerts()
            
            # Interactive map
            st.subheader("🗺️ Interactive Map")
            map_obj = self.create_interactive_map(selected_timestep)
            if map_obj:
                folium_static(map_obj, width=800, height=500)
            else:
                st.warning("No data available for map")
            
            # Timeline
            st.subheader("📈 Development Timeline")
            timeline_fig = self.create_timeline_plot()
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)
        
        with tab2:
            # 3D Visualization
            st.subheader("🌪️ 3D Cluster Visualization")
            fig_3d = self.create_3d_visualization()
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)
            
            # Data table
            st.subheader("📋 Cluster Data")
            if not self.df.empty:
                st.dataframe(self.df, use_container_width=True)
        
        with tab3:
            # Timestep maps
            self.display_timestep_maps()
            
            # Combined map
            st.subheader("🗺️ Combined Overview Map")
            combined_map_path = os.path.join(self.maps_dir, 'tcc_detection.png')
            if os.path.exists(combined_map_path):
                st.image(combined_map_path, use_container_width=True)
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"Last updated: {self.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        with col2:
            stats = self.get_system_stats()
            st.caption(f"CPU: {stats['cpu']:.1f}% | Memory: {stats['memory_percent']:.1f}%")
        with col3:
            st.caption("🌪️ Enhanced Dashboard v2.0")

# Main execution
if __name__ == "__main__":
    dashboard = EnhancedDashboard()
    dashboard.render() 