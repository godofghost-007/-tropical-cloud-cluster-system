# dashboard.py (fixed version)
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
from folium.plugins import MiniMap, Fullscreen

# Set page configuration
st.set_page_config(
    page_title="Tropical Cloud Cluster Monitor",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main styling */
    .stApp {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    
    /* Classic mode styling */
    .classic-mode {
        background-color: #f0f2f6;
        color: #333;
        padding: 20px;
        border-radius: 10px;
    }
    
    /* Header styling */
    .header {
        background: linear-gradient(135deg, #0d47a1 0%, #2196f3 100%);
        padding: 2rem 1rem;
        border-radius: 0 0 10px 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Card styling */
    .card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #2196f3 0%, #0d47a1 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    /* Alert styling */
    .alert-card {
        background: linear-gradient(135deg, #f44336 0%, #b71c1c 100%);
        animation: pulse 2s infinite;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(244, 67, 54, 0); }
        100% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0); }
    }
    
    /* Metric styling */
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background-color: #1e2130;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.7;
    }
    
    /* Enhanced Toggle styling */
    .toggle-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1e2130 0%, #2d3748 100%);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .toggle-btn {
        padding: 1.2rem 2.5rem;
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
        border: 2px solid transparent;
        color: #d1d5db;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        font-weight: 600;
        font-size: 1.1rem;
        position: relative;
        overflow: hidden;
        min-width: 160px;
    }
    
    .toggle-btn:hover {
        background: linear-gradient(135deg, #4b5563 0%, #6b7280 100%);
        color: #ffffff;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(33, 150, 243, 0.4);
    }
    
    .toggle-btn.active {
        background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
        color: white;
        font-weight: bold;
        border-color: #64b5f6;
        box-shadow: 0 12px 35px rgba(33, 150, 243, 0.5);
        transform: translateY(-4px);
    }
    
    .toggle-btn:first-child {
        border-radius: 15px 0 0 15px;
        border-right: 1px solid #4b5563;
    }
    
    .toggle-btn:last-child {
        border-radius: 0 15px 15px 0;
        border-left: 1px solid #4b5563;
    }
    
    .toggle-btn.active:first-child {
        border-right: 1px solid #64b5f6;
    }
    
    .toggle-btn.active:last-child {
        border-left: 1px solid #64b5f6;
    }
    
    /* Toggle indicator */
    .toggle-indicator {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.2) 50%, transparent 70%);
        transform: translateX(-100%);
        transition: transform 0.6s;
    }
    
    .toggle-btn:hover .toggle-indicator {
        transform: translateX(100%);
    }
    
    /* View mode labels */
    .view-mode-label {
        text-align: center;
        margin-bottom: 1.5rem;
        font-size: 1.3rem;
        font-weight: 700;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .view-mode-label.active {
        color: #2196f3;
        text-shadow: 0 0 15px rgba(33, 150, 243, 0.4);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 15px rgba(33, 150, 243, 0.4); }
        to { text-shadow: 0 0 25px rgba(33, 150, 243, 0.6), 0 0 35px rgba(33, 150, 243, 0.4); }
    }
    
    /* Classic mode specific */
    .classic-graph {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: status-pulse 2s infinite;
    }
    
    .status-active {
        background-color: #10b981;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
    }
    
    @keyframes status-pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

class Dashboard:
    def __init__(self):
        self.tracks_file = 'outputs/tracks/final_tracks.csv'
        self.config_file = 'real_data_config.yaml'
        self.log_file = 'processing.log'
        self.data_dir = 'data/insat_real'
        self.output_dir = 'outputs'
        self.tracks_df = self.load_tracks()
        self.config = self.load_config()
        self.last_updated = datetime.now()
        self.required_columns = ['track_id', 'datetime', 'center_lat', 'center_lon']
        
        # Critical column validation
        REQUIRED_COLUMNS = ['timestep', 'center_lat', 'center_lon', 'track_id']
        missing = [col for col in REQUIRED_COLUMNS if col not in self.tracks_df.columns]
        if missing:
            st.error(f"⚠️ Missing critical columns: {', '.join(missing)}")
            st.error("Execute this command to fix: `python detection.py --reprocess --full`")
            st.stop()
        
    def safe_get(self, row, col, default=np.nan):
        """Safely get value from DataFrame row with enhanced error handling"""
        try:
            if col in row.index:
                value = row[col]
                # Handle various data types safely
                if pd.isna(value) or value is None:
                    return default
                return value
            return default
        except (KeyError, AttributeError, TypeError):
            return default
        
    def has_column(self, col):
        """Check if column exists and has non-null data"""
        return col in self.tracks_df.columns and not self.tracks_df[col].isna().all()
        
    def load_tracks(self):
        """Load tracking data with robust error handling"""
        try:
            if os.path.exists(self.tracks_file):
                df = pd.read_csv(self.tracks_file)
                
                # Basic preprocessing
                if 'timestamp' in df.columns:
                    df['datetime'] = pd.to_datetime(df['timestamp'])
                    
                # Create essential columns if missing
                if 'center_lat' not in df.columns:
                    df['center_lat'] = np.nan
                if 'center_lon' not in df.columns:
                    df['center_lon'] = np.nan
                if 'track_id' not in df.columns:
                    df['track_id'] = range(1, len(df)+1)
                
                # Calculate movement vectors if possible
                if 'track_id' in df.columns and 'datetime' in df.columns:
                    df = df.sort_values(['track_id', 'datetime'])
                    df['dx'] = df.groupby('track_id')['center_lon'].diff()
                    df['dy'] = df.groupby('track_id')['center_lat'].diff()
                    df['dt'] = df.groupby('track_id')['datetime'].diff().dt.total_seconds() / 3600
                    
                    # Only calculate if we have valid differences
                    if 'dx' in df.columns and 'dy' in df.columns and 'dt' in df.columns:
                        df['speed_kmh'] = np.sqrt(df['dx']**2 + df['dy']**2) * 111 / df['dt']
                        df['direction_deg'] = np.degrees(np.arctan2(df['dy'], df['dx']))
                
                # Add risk column if missing
                if 'cyclogenesis_risk' not in df.columns:
                    df['cyclogenesis_risk'] = 0.0
                    
                return df
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading tracks: {str(e)}")
            return pd.DataFrame()
    
    def load_config(self):
        """Load configuration with defaults"""
        defaults = {
            'data': {
                'input_dir': 'data/insat_real',
                'output_dir': 'outputs',
                'thresholds': {
                    'irbt': 220,
                    'min_area': 50000
                }
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return yaml.safe_load(f)
            return defaults
        except Exception as e:
            st.error(f"Error loading config: {str(e)}")
            return defaults
    
    def get_recent_files(self, count=5):
        """Get recently processed files"""
        try:
            files = []
            if os.path.exists(self.data_dir):
                for f in os.listdir(self.data_dir):
                    if f.lower().endswith(('.nc', '.hdf', '.h5')):
                        files.append({
                            'name': f,
                            'path': os.path.join(self.data_dir, f),
                            'size': os.path.getsize(os.path.join(self.data_dir, f)) // 1024,
                            'modified': datetime.fromtimestamp(os.path.getmtime(os.path.join(self.data_dir, f)))
                        })
                # Sort by modification time
                files.sort(key=lambda x: x['modified'], reverse=True)
                return files[:count]
            return []
        except Exception as e:
            st.error(f"Error listing files: {str(e)}")
            return []
    
    def get_system_stats(self):
        """Get system resource utilization"""
        return {
            'cpu': psutil.cpu_percent(),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent,
            'process_time': datetime.now() - self.last_updated
        }
    
    def create_map(self, selected_track=None):
        """Create interactive map visualization with safe column access"""
        if self.tracks_df.empty:
            return None
            
        # Create base map
        mean_lat = self.tracks_df['center_lat'].mean()
        mean_lon = self.tracks_df['center_lon'].mean()
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=4, tiles='CartoDB dark_matter')
        
        # Add tile layers
        folium.TileLayer('OpenStreetMap').add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite Imagery'
        ).add_to(m)
        
        # Add tracks with safe column access
        for track_id, group in self.tracks_df.groupby('track_id'):
            if selected_track and selected_track != track_id:
                continue
                
            # Safe risk calculation
            max_risk = group['cyclogenesis_risk'].max() if 'cyclogenesis_risk' in group.columns else 0.0
            color = 'red' if max_risk > 0.7 else 'orange' if max_risk > 0.5 else 'blue'
                
            # Create polyline
            folium.PolyLine(
                locations=group[['center_lat', 'center_lon']].values,
                color=color,
                weight=3,
                opacity=0.7,
                popup=f'Track {track_id}'
            ).add_to(m)
            
            # Add markers
            for idx, row in group.iterrows():
                # Safely get values with fallbacks
                min_tb = self.safe_get(row, 'min_tb', 0)
                area = self.safe_get(row, 'area_km2', 0)
                risk = self.safe_get(row, 'cyclogenesis_risk', 0)
                dt = self.safe_get(row, 'datetime', 'N/A')
                
                folium.CircleMarker(
                    location=[row['center_lat'], row['center_lon']],
                    radius=3 + (risk * 7),
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=folium.Popup(
                        f"<b>Track {track_id}</b><br>"
                        f"Time: {dt}<br>"
                        f"Min Tb: {min_tb}K<br>"
                        f"Area: {area:.1f} km²<br>"
                        f"Risk: <b>{risk*100:.1f}%</b>",
                        max_width=250
                    )
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        minimap = MiniMap(toggle_display=True)
        m.add_child(minimap)
        Fullscreen(position="topright").add_to(m)
        
        return m
    
    def create_3d_track(self, track_id):
        """Create 3D visualization with safe column access"""
        track = self.tracks_df[self.tracks_df['track_id'] == track_id]
        if track.empty:
            return None
            
        fig = go.Figure()
        
        # Safely get values with fallbacks
        lon = track['center_lon'] if 'center_lon' in track.columns else np.zeros(len(track))
        lat = track['center_lat'] if 'center_lat' in track.columns else np.zeros(len(track))
        height = track['cloud_top_height_km'] if 'cloud_top_height_km' in track.columns else np.zeros(len(track))
        min_tb = track['min_tb'] if 'min_tb' in track.columns else np.full(len(track), 220)
        area = track['area_km2'] if 'area_km2' in track.columns else np.full(len(track), 50000)
        risk = track['cyclogenesis_risk'] if 'cyclogenesis_risk' in track.columns else np.zeros(len(track))
        dt = track['datetime'] if 'datetime' in track.columns else [f"Time {i}" for i in range(len(track))]
        
        # Add track line
        fig.add_trace(go.Scatter3d(
            x=lon,
            y=lat,
            z=height,
            mode='lines',
            line=dict(width=8, color='#2196F3'),
            name='Track Path'
        ))
        
        # Add markers
        fig.add_trace(go.Scatter3d(
            x=lon,
            y=lat,
            z=height,
            mode='markers',
            marker=dict(
                size=area/2000,
                color=min_tb,
                colorscale='Viridis',
                cmin=180,
                cmax=250,
                showscale=True,
                opacity=0.8,
                colorbar=dict(title='Min Tb (K)')
            ),
            text=[f"Time: {t}<br>Min Tb: {tb}K<br>Area: {a:.0f} km²" 
                  for t, tb, a in zip(dt, min_tb, area)],
            name='Cluster Positions'
        ))
        
        # Add risk markers if available
        if 'cyclogenesis_risk' in track.columns:
            high_risk = track[track['cyclogenesis_risk'] > 0.7]
            if not high_risk.empty:
                fig.add_trace(go.Scatter3d(
                    x=high_risk['center_lon'],
                    y=high_risk['center_lat'],
                    z=high_risk['cloud_top_height_km'],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='diamond'),
                    name='High Risk (>70%)'
                ))
        
        # FIXED: Correct scene properties
        fig.update_layout(
            title=f'3D Visualization of Track {track_id}',
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Cloud Top Height (km)',
                bgcolor='#0e1117',
                xaxis=dict(
                    gridcolor='rgba(100, 100, 100, 0.2)',
                    backgroundcolor='rgba(0, 0, 0, 0)'
                ),
                yaxis=dict(
                    gridcolor='rgba(100, 100, 100, 0.2)',
                    backgroundcolor='rgba(0, 0, 0, 0)'
                ),
                zaxis=dict(
                    gridcolor='rgba(100, 100, 100, 0.2)',
                    backgroundcolor='rgba(0, 0, 0, 0)'
                ),
            ),
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            font=dict(color='#f0f2f6'),
            margin=dict(l=0, r=0, b=0, t=30),
            height=600
        )
        
        return fig
    
    def create_timeline(self):
        """Create timeline with safe column access"""
        if self.tracks_df.empty or not all(col in self.tracks_df.columns for col in ['track_id', 'datetime', 'cyclogenesis_risk']):
            return None
            
        # Prepare data
        timeline_data = []
        for track_id, group in self.tracks_df.groupby('track_id'):
            start_time = group['datetime'].min()
            end_time = group['datetime'].max()
            max_risk = group['cyclogenesis_risk'].max()
            timeline_data.append({
                'Track': track_id,
                'Start': start_time,
                'Finish': end_time,
                'Duration': (end_time - start_time).total_seconds() / 3600,
                'Max Risk': max_risk,
                'Risk Category': 'High' if max_risk > 0.7 else 'Medium' if max_risk > 0.5 else 'Low'
            })
        
        df = pd.DataFrame(timeline_data)
        
        # Create figure
        fig = px.timeline(
            df, 
            x_start="Start", 
            x_end="Finish", 
            y="Track",
            color="Risk Category",
            color_discrete_map={
                'High': '#f44336',
                'Medium': '#ff9800',
                'Low': '#4caf50'
            },
            hover_data=['Duration', 'Max Risk'],
            title='Cluster Timeline'
        )
        
        fig.update_layout(
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            font=dict(color='#f0f2f6'),
            height=500
        )
        
        return fig
    
    def plot_track_properties(self, track_id):
        """Classic view: Plot time series with safe column access"""
        track = self.tracks_df[self.tracks_df['track_id'] == track_id]
        if track.empty:
            return None
            
        # Create figure
        fig = go.Figure()
        
        # Add traces only if columns exist
        if 'min_tb' in track.columns:
            fig.add_trace(go.Scatter(
                x=track['datetime'], y=track['min_tb'], 
                mode='lines+markers', name='Min Tb', line=dict(color='blue')))
        
        if 'mean_tb' in track.columns:
            fig.add_trace(go.Scatter(
                x=track['datetime'], y=track['mean_tb'], 
                mode='lines', name='Mean Tb', line=dict(color='lightblue')))
        
        if 'area_km2' in track.columns:
            fig.add_trace(go.Scatter(
                x=track['datetime'], y=track['area_km2'], 
                mode='lines+markers', name='Area (km²)', line=dict(color='green'),
                yaxis='y2'))
        
        if 'cloud_top_height_km' in track.columns:
            fig.add_trace(go.Scatter(
                x=track['datetime'], y=track['cloud_top_height_km'], 
                mode='lines', name='Cloud Height (km)', line=dict(color='purple'),
                yaxis='y3'))
        
        if 'speed_kmh' in track.columns:
            fig.add_trace(go.Scatter(
                x=track['datetime'], y=track['speed_kmh'], 
                mode='lines+markers', name='Speed (km/h)', line=dict(color='red'),
                yaxis='y4'))
        
        # Update layout with multiple y-axes
        fig.update_layout(
            title=f'Properties of Track {track_id}',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Temperature (K)', side='left', position=0.0),
            yaxis2=dict(title='Area (km²)', overlaying='y', side='right', position=0.15,
                       showgrid=False) if 'area_km2' in track.columns else {},
            yaxis3=dict(title='Height (km)', overlaying='y', side='right', position=0.3,
                       showgrid=False) if 'cloud_top_height_km' in track.columns else {},
            yaxis4=dict(title='Speed (km/h)', overlaying='y', side='right', position=0.45,
                       showgrid=False) if 'speed_kmh' in track.columns else {},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        # Risk indicators if available
        if 'cyclogenesis_risk' in track.columns:
            for idx, row in track.iterrows():
                if row['cyclogenesis_risk'] > 0.7:
                    fig.add_vline(x=row['datetime'], line=dict(color="red", width=2, dash="dot"))
        
        return fig
    
    def analyze_cluster_development(self, track_id):
        """Classic view: Analyze development with safe column access"""
        track = self.tracks_df[self.tracks_df['track_id'] == track_id]
        if track.empty:
            return None, None
            
        # Calculate development metrics with fallbacks
        start_time = track['datetime'].min() if 'datetime' in track.columns else None
        end_time = track['datetime'].max() if 'datetime' in track.columns else None
        
        duration_h = (end_time - start_time).total_seconds() / 3600 if start_time and end_time else 0
        area_growth = (track['area_km2'].iloc[-1] / track['area_km2'].iloc[0] 
                      if 'area_km2' in track.columns and len(track) > 1 else 1)
        min_tb_change = (track['min_tb'].iloc[0] - track['min_tb'].iloc[-1] 
                         if 'min_tb' in track.columns and len(track) > 1 else 0)
        max_risk = track['cyclogenesis_risk'].max() if 'cyclogenesis_risk' in track.columns else 0
        
        # Classify development pattern
        pattern = "Unknown"
        if max_risk > 0.7 and min_tb_change > 15 and area_growth > 2:
            pattern = "Rapid Intensification"
        elif max_risk > 0.5 and min_tb_change > 10:
            pattern = "Moderate Intensification"
        elif area_growth > 1.5:
            pattern = "Expansion Dominated"
        elif area_growth > 0:
            pattern = "Stable or Weakening"
        
        # Create summary table
        summary = pd.DataFrame({
            'Metric': ['Duration (hours)', 'Area Growth', 'Min Tb Change', 'Max Risk', 'Development Pattern'],
            'Value': [f"{duration_h:.1f}", f"{area_growth:.1f}x", f"{min_tb_change:.1f}K", 
                      f"{max_risk*100:.1f}%", pattern]
        })
        
        # Plot correlation matrix
        corr_data = None
        possible_cols = ['min_tb', 'mean_tb', 'area_km2', 'cloud_top_height_km', 
                         'speed_kmh', 'cyclogenesis_risk']
        available_cols = [col for col in possible_cols if col in track.columns]
        
        if len(available_cols) > 1:
            corr_data = track[available_cols].corr()
        
        return summary, corr_data
    
    def validate_data_quality(self):
        """Validate data quality and return status report"""
        if self.tracks_df.empty:
            return {
                'status': 'empty',
                'message': 'No track data available',
                'quality_score': 0.0
            }
        
        # Check essential columns
        essential_cols = ['track_id', 'datetime', 'center_lat', 'center_lon']
        missing_cols = [col for col in essential_cols if col not in self.tracks_df.columns]
        
        # Check data completeness
        completeness_scores = {}
        for col in self.tracks_df.columns:
            if col in self.tracks_df.columns:
                non_null_ratio = 1 - (self.tracks_df[col].isna().sum() / len(self.tracks_df))
                completeness_scores[col] = non_null_ratio
        
        # Calculate overall quality score
        if missing_cols:
            quality_score = 0.3  # Low score if essential columns missing
        else:
            avg_completeness = np.mean(list(completeness_scores.values()))
            quality_score = min(1.0, avg_completeness * 1.2)  # Boost score slightly
        
        return {
            'status': 'valid' if not missing_cols else 'partial',
            'message': f"Missing columns: {missing_cols}" if missing_cols else "Data quality good",
            'quality_score': quality_score,
            'completeness': completeness_scores,
            'total_tracks': self.tracks_df['track_id'].nunique() if 'track_id' in self.tracks_df.columns else 0,
            'total_records': len(self.tracks_df)
        }
    
    def generate_forecast(self, track_id, hours=24):
        # Get recent track data
        track = self.tracks_df[self.tracks_df['track_id'] == track_id]
        if track.empty or 'datetime' not in track.columns:
            return pd.DataFrame()
        recent = track.sort_values('datetime').tail(3)
        if recent.empty:
            return pd.DataFrame()
        
        forecast_points = []
        last_point = recent.iloc[-1]
        for i in range(1, hours + 1):
            forecast_time = last_point['datetime'] + timedelta(hours=i)
            forecast_points.append({
                'track_id': track_id,
                'datetime': forecast_time,
                'center_lat': last_point.get('center_lat', 0) + i * 0.1,
                'center_lon': last_point.get('center_lon', 0) + i * 0.2,
                'min_tb': max(180, last_point.get('min_tb', 200) - i * 0.5),
                'area_km2': last_point.get('area_km2', 50000) * (1 + i * 0.05),
                'cyclogenesis_risk': min(0.99, last_point.get('cyclogenesis_risk', 0.1) * (1 + i * 0.1))
            })
        return pd.DataFrame(forecast_points)
    
    def check_alerts(self, current_data, forecast_data):
        alerts = []
        risk_threshold = getattr(self, 'risk_threshold', 0.7)
        # Current high-risk clusters
        if 'cyclogenesis_risk' in current_data.columns:
            high_risk = current_data[current_data['cyclogenesis_risk'] > risk_threshold]
            for _, row in high_risk.iterrows():
                alerts.append({
                    'type': 'CURRENT_HIGH_RISK',
                    'track_id': row.get('track_id', 'N/A'),
                    'risk': row.get('cyclogenesis_risk', 0),
                    'time': row.get('datetime', 'N/A'),
                    'location': f"{row.get('center_lat', 0):.4f}, {row.get('center_lon', 0):.4f}",
                    'message': f"High risk cluster detected: Track {row.get('track_id', 'N/A')}"
                })
        # ... (other alert types, e.g., forecasted high risk, rapid intensification) ...
        return alerts
    
    def render_classic_view(self):
        """Render the classic analytical view"""
        with st.container():
            st.markdown("<div class='classic-mode'>", unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="color: #1e40af; margin-bottom: 0.5rem;">🔬 Classic Track Explorer</h1>
                <p style="color: #6b7280; font-style: italic;">Advanced analytical interface for detailed cluster analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            if self.tracks_df.empty:
                st.warning("No track data available. Process some data first.")
                st.markdown("</div>", unsafe_allow_html=True)
                return
            
            # Data Quality Details Expander
            with st.expander("📊 Data Quality Report", expanded=False):
                data_quality = self.validate_data_quality()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Quality Score", f"{data_quality['quality_score']*100:.1f}%")
                with col2:
                    st.metric("Total Tracks", data_quality['total_tracks'])
                with col3:
                    st.metric("Total Records", data_quality['total_records'])
                
                if data_quality['completeness']:
                    st.markdown("### Column Completeness")
                    completeness_df = pd.DataFrame([
                        {'Column': col, 'Completeness': f"{score*100:.1f}%"}
                        for col, score in data_quality['completeness'].items()
                    ])
                    st.dataframe(completeness_df, use_container_width=True)
                
                if data_quality['status'] == 'partial':
                    st.warning(f"⚠️ Data Quality Issues: {data_quality['message']}")
                else:
                    st.success("✅ Data quality is good")
                
            # Track selection
            track_id = st.selectbox("Select Track", options=self.tracks_df['track_id'].unique())
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Track properties plot
                st.markdown("### Cluster Properties Over Time")
                fig = self.plot_track_properties(track_id)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not generate properties plot")
                
                # Development analysis
                st.markdown("### Development Analysis")
                summary, corr_data = self.analyze_cluster_development(track_id)
                
                if summary is not None:
                    st.table(summary.style.set_properties(**{'background-color': '#f0f0f0',
                                                           'color': 'black'}))
                
            with col2:
                # Map visualization
                st.markdown("### Track Location")
                cluster_map = self.create_map(track_id)
                if cluster_map:
                    folium_static(cluster_map, width=400, height=300)
                else:
                    st.info("No map data available")
                
                # Correlation heatmap
                st.markdown("### Property Correlations")
                if corr_data is not None:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax)
                    st.pyplot(fig)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def render_modern_dashboard(self):
        """Render modern dashboard with safe column access"""
        # Header
        st.markdown("""
        <div class="header">
            <h1 style="color: white; margin: 0;">🌪️ Tropical Cloud Cluster Monitor</h1>
            <p style="color: rgba(255,255,255,0.8); margin: 0;">Real-time detection and tracking system</p>
        </div>
        """, unsafe_allow_html=True)
        
        # System status row
        col1, col2, col3, col4, col5 = st.columns(5)
        stats = self.get_system_stats()
        data_quality = self.validate_data_quality()
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Processing Status</div>
                <div class="metric-value">
                    <span class="status-indicator status-active"></span>
                    Active
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CPU Usage</div>
                <div class="metric-value">{stats['cpu']}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Memory Usage</div>
                <div class="metric-value">{stats['memory']}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Data Quality</div>
                <div class="metric-value">{data_quality['quality_score']*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Last Updated</div>
                <div class="metric-value">{datetime.now().strftime('%H:%M:%S')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Main columns
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            # Map visualization
            st.markdown("### Cluster Tracking Map")
            
            # Get available track IDs safely
            track_ids = ['All'] 
            if 'track_id' in self.tracks_df.columns:
                track_ids.extend(self.tracks_df['track_id'].unique())
                
            selected_track = st.selectbox(
                "Select a track to highlight:",
                options=track_ids,
                index=0
            )
            
            # Handle 'All', NaN, and valid integers safely
            if selected_track == 'All' or (isinstance(selected_track, float) and np.isnan(selected_track)):
                track_id = None
            else:
                track_id = int(selected_track)
            cluster_map = self.create_map(track_id)
            if cluster_map:
                folium_static(cluster_map, width=800, height=500)
            else:
                st.info("No track data available. Process some data to see visualizations.")
            
            # Timeline visualization
            st.markdown("### Cluster Timeline")
            timeline = self.create_timeline()
            if timeline:
                st.plotly_chart(timeline, use_container_width=True)
            else:
                st.info("No timeline data available")
        
        with col_right:
            # Recent activity card
            st.markdown("### Recent Activity")
            recent_files = self.get_recent_files(3)
            
            if recent_files:
                for file in recent_files:
                    with st.container():
                        st.markdown(f"""
                        <div class="card">
                            <div style="display: flex; justify-content: space-between;">
                                <b>{file['name']}</b>
                                <span>{file['size']} KB</span>
                            </div>
                            <div style="color: #aaa; font-size: 0.9rem; margin-top: 0.5rem;">
                                Processed: {file['modified'].strftime('%Y-%m-%d %H:%M')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No recent files processed")
            
            # Alerts card (only if risk column exists)
            st.markdown("### Risk Alerts")
            if not self.tracks_df.empty and 'cyclogenesis_risk' in self.tracks_df.columns:
                high_risk = self.tracks_df[self.tracks_df['cyclogenesis_risk'] > 0.7]
                
                if not high_risk.empty:
                    for idx, row in high_risk.head(3).iterrows():
                        st.markdown(f"""
                        <div class="alert-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <b>🚨 High Risk Cluster</b>
                                    <div>Track ID: {row['track_id']}</div>
                                </div>
                                <div style="font-size: 1.5rem; font-weight: bold;">
                                    {row['cyclogenesis_risk']*100:.0f}%
                                </div>
                            </div>
                            <div style="margin-top: 0.5rem;">
                                Location: {row['lat_lon'] if 'lat_lon' in row else 'Unknown'}<br>
                                Min Temp: {row['min_tb'] if 'min_tb' in row else 'N/A'}K<br>
                                Time: {row['datetime'] if 'datetime' in row else 'N/A'}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No high-risk clusters detected")
            else:
                st.info("Risk data not available")
            
            # Processing controls
            st.markdown("### Processing Controls")
            if st.button("🔄 Process New Data", use_container_width=True):
                with st.spinner("Processing data..."):
                    try:
                        # For demonstration - in real system call your processor
                        st.success("Data processing feature would run here")
                        time.sleep(2)
                        self.tracks_df = self.load_tracks()
                        st.success("Data processed successfully!")
                    except Exception as e:
                        st.error(f"Processing failed: {str(e)}")
            
            if st.button("📊 Generate Report", use_container_width=True):
                st.info("Report generation feature coming soon!")
        
        # 3D Visualization section (only if required columns exist)
        st.markdown("### 3D Track Visualization")
        if not self.tracks_df.empty and 'track_id' in self.tracks_df.columns:
            selected_3d_track = st.selectbox(
                "Select a track for 3D visualization:",
                options=self.tracks_df['track_id'].unique()
            )
            
            fig_3d = self.create_3d_track(selected_3d_track)
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning(f"No data available for track {selected_3d_track}")
        else:
            st.info("No track data available for 3D visualization")
        
        # Cluster metrics table with safe columns
        st.markdown("### Cluster Metrics")
        if not self.tracks_df.empty:
            # Select columns to display
            possible_cols = ['track_id', 'datetime', 'center_lat', 'center_lon', 
                           'min_tb', 'mean_tb', 'area_km2', 'cloud_top_height_km', 
                           'cyclogenesis_risk']
            available_cols = [col for col in possible_cols if col in self.tracks_df.columns]
            
            # Format data
            if available_cols:
                display_df = self.tracks_df[available_cols].copy()
                if 'datetime' in display_df.columns:
                    display_df.loc[:, 'datetime'] = display_df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
                
                # Display with styling
                st.dataframe(
                    display_df,
                    height=400
                )
            else:
                st.info("No cluster metrics available")
        else:
            st.info("No track data available")
    
    def render(self):
        """Render the dashboard with view toggle and data checks"""
        # Enhanced data status and quality reporting
        data_quality = self.validate_data_quality()
        
        if data_quality['status'] == 'empty':
            st.warning("No track data available. Please process data first.")
        elif data_quality['status'] == 'partial':
            st.warning(f"⚠️ Data Quality Issue: {data_quality['message']}")
            st.info(f"Quality Score: {data_quality['quality_score']*100:.1f}% | "
                   f"Tracks: {data_quality['total_tracks']} | "
                   f"Records: {data_quality['total_records']}")
        else:
            st.success(f"✅ Data Quality: {data_quality['quality_score']*100:.1f}% | "
                      f"Tracks: {data_quality['total_tracks']} | "
                      f"Records: {data_quality['total_records']}")
        
        # Enhanced view toggle with visual indicators
        st.markdown("<div class='toggle-container'>", unsafe_allow_html=True)
        
        # View mode label
        current_mode = st.session_state.get('view_mode', 'modern')
        mode_label = "🔬 CLASSIC ANALYTICAL VIEW" if current_mode == "classic" else "🌪️ MODERN DASHBOARD"
        st.markdown(f"<div class='view-mode-label active'>{mode_label}</div>", unsafe_allow_html=True)
        
        # Toggle buttons with icons
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col2:
            classic_btn = st.button("🔬 Classic View", key="classic_btn", use_container_width=True)
        with col4:
            modern_btn = st.button("🌪️ Modern Dashboard", key="modern_btn", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Determine view with enhanced state management
        if 'view_mode' not in st.session_state:
            st.session_state.view_mode = "modern"
        
        if classic_btn:
            st.session_state.view_mode = "classic"
            st.success("Switched to Classic Analytical View")
        elif modern_btn:
            st.session_state.view_mode = "modern"
            st.success("Switched to Modern Dashboard View")
        
        # Status indicator
        status_col1, status_col2, status_col3 = st.columns([1, 2, 1])
        with status_col2:
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <span class="status-indicator status-active"></span>
                <strong>Current Mode:</strong> {st.session_state.view_mode.title()}
            </div>
            """, unsafe_allow_html=True)
        
        # Render appropriate view
        if st.session_state.view_mode == "classic":
            self.render_classic_view()
        else:
            self.render_modern_dashboard()
            
        # Enhanced auto-refresh with better styling
        st.markdown("---")
        refresh_col1, refresh_col2, refresh_col3 = st.columns([1, 2, 1])
        with refresh_col2:
            st_autorefresh = st.empty()
            if st_autorefresh.button("🔄 Refresh Data", key="refresh_button", use_container_width=True):
                with st.spinner("Refreshing data..."):
                    self.tracks_df = self.load_tracks()
                    st.success("Data refreshed successfully!")
                    time.sleep(1)
                    st.rerun()
        
        st.caption("💡 Data refreshes automatically every 60 seconds. Click above to refresh manually.")
        time.sleep(60)
        st.rerun()

    def get_selected_timestep(self):
        """Robust timestep slider logic for Streamlit UI"""
        if 'timestep' not in self.tracks_df.columns:
            st.warning("No 'timestep' column found in data. Using default range.")
            timesteps = []
        else:
            timesteps = sorted(self.tracks_df['timestep'].unique())

        # Add data validation and debug output
        st.sidebar.write(f"{len(timesteps)} timesteps loaded")
        if len(timesteps) < 2:
            st.info("Insufficient timesteps for full analysis. Please reprocess data or check filters.")

        if len(timesteps) == 0:
            st.warning("No timesteps available. Using default range.")
            min_ts, max_ts = 0, 10  # Default fallback values
        elif len(timesteps) == 1:
            st.warning(f"Only one timestep available: {timesteps[0]}")
            min_ts = timesteps[0]
            max_ts = timesteps[0] + 1  # Add buffer to create a valid range
        else:
            min_ts = min(timesteps)
            max_ts = max(timesteps)

        # If you want a single-value slider when only one timestep exists:
        if min_ts == max_ts:
            selected_timestep = st.sidebar.slider(
                "Select Timestep:",
                min_value=min_ts,
                max_value=max_ts + 1,
                value=min_ts
            )
        else:
            selected_timestep = st.sidebar.slider(
                "Select Timestep:",
                min_value=min_ts,
                max_value=max_ts,
                value=min_ts
            )
        return selected_timestep

# Run the dashboard
if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.render() 