# track_explorer.py
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import TimestampedGeoJson, HeatMap
import ipywidgets as widgets
from IPython.display import display, HTML
import seaborn as sns
from datetime import datetime, timedelta
from ipywidgets import interact, interactive, fixed, interact_manual

# Enhanced visualization settings
plt.style.use('ggplot')
sns.set_style('whitegrid')
pd.options.plotting.backend = 'plotly'

class TrackExplorer:
    def __init__(self, tracks_file='outputs/tracks/final_tracks.csv'):
        self.tracks_df = self.load_tracks(tracks_file)
        self.cluster_properties = {}
        
    def load_tracks(self, file_path):
        """Load and preprocess tracking data"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Track file not found: {file_path}")
            
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['datetime'].dt.date
            df['time'] = df['datetime'].dt.time
            
        # Calculate movement vectors
        if 'track_id' in df.columns and 'datetime' in df.columns:
            df = df.sort_values(['track_id', 'datetime'])
            df['dx'] = df.groupby('track_id')['center_lon'].diff()
            df['dy'] = df.groupby('track_id')['center_lat'].diff()
            df['dt'] = df.groupby('track_id')['datetime'].diff().dt.total_seconds() / 3600
            df['speed_kmh'] = np.sqrt(df['dx']**2 + df['dy']**2) * 111 / df['dt']  # Approx 111km per degree
            df['direction_deg'] = np.degrees(np.arctan2(df['dy'], df['dx']))
            
            # Handle NaN values from diff operation
            df['speed_kmh'] = df['speed_kmh'].fillna(0)
            df['direction_deg'] = df['direction_deg'].fillna(0)
            
        # Calculate cyclogenesis risk if not present
        if 'cyclogenesis_risk' not in df.columns:
            df['cyclogenesis_risk'] = self.calculate_cyclogenesis_risk(df)
            
        # Calculate cloud top height if not present
        if 'cloud_top_height_km' not in df.columns:
            df['cloud_top_height_km'] = self.calculate_cloud_height(df)
            
        return df
    
    def calculate_cyclogenesis_risk(self, df):
        """Calculate cyclogenesis risk based on cluster properties"""
        risk = np.zeros(len(df))
        
        # Temperature-based risk (colder = higher risk)
        if 'min_tb' in df.columns:
            temp_risk = (250 - df['min_tb']) / 70  # Normalize to 0-1
            temp_risk = np.clip(temp_risk, 0, 1)
            risk += temp_risk * 0.4
            
        # Size-based risk (larger = higher risk)
        if 'area_km2' in df.columns:
            size_risk = np.log10(df['area_km2']) / 4  # Normalize to 0-1
            size_risk = np.clip(size_risk, 0, 1)
            risk += size_risk * 0.3
            
        # Duration-based risk (longer = higher risk)
        if 'track_id' in df.columns:
            track_durations = df.groupby('track_id').size()
            duration_risk = track_durations.reindex(df['track_id']).values / 10
            duration_risk = np.clip(duration_risk, 0, 1)
            risk += duration_risk * 0.3
            
        return np.clip(risk, 0, 1)
    
    def calculate_cloud_height(self, df):
        """Calculate cloud top height from brightness temperature"""
        if 'min_tb' in df.columns:
            # Simple conversion: colder = higher clouds
            height = 12.0 - (df['min_tb'] - 200) / 10.0  # km
            return np.clip(height, 0, 20)  # Limit to reasonable range
        return np.full(len(df), 8.0)  # Default height
    
    def create_interactive_map(self, output_file='outputs/tracks/track_map.html'):
        """Create an interactive Folium map with all tracks"""
        if self.tracks_df.empty:
            print("No track data available")
            return
            
        # Create base map centered on mean position
        mean_lat = self.tracks_df['center_lat'].mean()
        mean_lon = self.tracks_df['center_lon'].mean()
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=5)
        
        # Add tile layers
        folium.TileLayer('OpenStreetMap').add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite Imagery'
        ).add_to(m)
        
        # Create feature groups
        track_group = folium.FeatureGroup(name='Cluster Tracks')
        heatmap_group = folium.FeatureGroup(name='Intensity Heatmap', show=False)
        marker_group = folium.FeatureGroup(name='Cluster Markers', show=False)
        
        # Add tracks with color coding by risk
        for track_id, group in self.tracks_df.groupby('track_id'):
            # Determine track color based on max risk
            max_risk = group['cyclogenesis_risk'].max()
            if max_risk > 0.7:
                color = 'red'
            elif max_risk > 0.5:
                color = 'orange'
            else:
                color = 'blue'
                
            # Create polyline for track
            folium.PolyLine(
                locations=group[['center_lat', 'center_lon']].values,
                color=color,
                weight=3,
                opacity=0.7,
                popup=f'Track {track_id} (Max Risk: {max_risk*100:.1f}%)'
            ).add_to(track_group)
            
            # Add markers for each position
            for idx, row in group.iterrows():
                folium.CircleMarker(
                    location=[row['center_lat'], row['center_lon']],
                    radius=row['area_km2']/5000,
                    color=color,
                    fill=True,
                    fill_opacity=0.5,
                    popup=f"ID: {track_id}<br>Time: {row['datetime']}<br>Min Tb: {row['min_tb']}K<br>Risk: {row['cyclogenesis_risk']*100:.1f}%"
                ).add_to(marker_group)
        
        # Add heatmap
        heat_data = [[row['center_lat'], row['center_lon'], row['min_tb']] 
                    for idx, row in self.tracks_df.iterrows()]
        HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(heatmap_group)
        
        # Add layer control
        track_group.add_to(m)
        heatmap_group.add_to(m)
        marker_group.add_to(m)
        folium.LayerControl().add_to(m)
        
        # Add minimap
        try:
            from folium.plugins import MiniMap
            minimap = MiniMap()
            m.add_child(minimap)
        except ImportError:
            pass
        
        # Add fullscreen control
        try:
            from folium.plugins import Fullscreen
            Fullscreen().add_to(m)
        except ImportError:
            pass
        
        # Save and display
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        m.save(output_file)
        print(f"Interactive map saved to {output_file}")
        return m
    
    def plot_3d_track(self, track_id):
        """Create 3D visualization of a track"""
        track = self.tracks_df[self.tracks_df['track_id'] == track_id]
        if track.empty:
            print(f"Track {track_id} not found")
            return
            
        fig = go.Figure()
        
        # Add track line
        fig.add_trace(go.Scatter3d(
            x=track['center_lon'],
            y=track['center_lat'],
            z=track['cloud_top_height_km'],
            mode='lines',
            line=dict(width=4, color='blue'),
            name='Track Path'
        ))
        
        # Add markers for each position
        fig.add_trace(go.Scatter3d(
            x=track['center_lon'],
            y=track['center_lat'],
            z=track['cloud_top_height_km'],
            mode='markers',
            marker=dict(
                size=track['area_km2']/2000,
                color=track['min_tb'],
                colorscale='Viridis',
                cmin=180,
                cmax=250,
                showscale=True,
                opacity=0.8),
            text=[f"Time: {t}<br>Min Tb: {tb}K<br>Height: {h}km" 
                  for t, tb, h in zip(track['datetime'], track['min_tb'], track['cloud_top_height_km'])],
            name='Cluster Positions'
        ))
        
        # Add risk markers
        high_risk = track[track['cyclogenesis_risk'] > 0.7]
        if not high_risk.empty:
            fig.add_trace(go.Scatter3d(
                x=high_risk['center_lon'],
                y=high_risk['center_lat'],
                z=high_risk['cloud_top_height_km'],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='diamond'),
                name='High Risk (>70%)'
            ))
        
        fig.update_layout(
            title=f'3D Visualization of Track {track_id}',
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Cloud Top Height (km)'),
            margin=dict(l=0, r=0, b=0, t=30),
            height=700
        )
        
        return fig
    
    def plot_track_properties(self, track_id):
        """Plot time series of cluster properties for a track"""
        track = self.tracks_df[self.tracks_df['track_id'] == track_id]
        if track.empty:
            print(f"Track {track_id} not found")
            return
            
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           subplot_titles=('Temperature Properties', 'Size and Height', 'Movement'))
        
        # Temperature properties
        fig.add_trace(go.Scatter(
            x=track['datetime'], y=track['min_tb'], 
            mode='lines+markers', name='Min Tb', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(go.Scatter(
            x=track['datetime'], y=track['mean_tb'], 
            mode='lines', name='Mean Tb', line=dict(color='lightblue')),
            row=1, col=1
        )
        
        # Size and height
        fig.add_trace(go.Scatter(
            x=track['datetime'], y=track['area_km2'], 
            mode='lines+markers', name='Area (km²)', line=dict(color='green')),
            row=2, col=1
        )
        fig.add_trace(go.Scatter(
            x=track['datetime'], y=track['cloud_top_height_km'], 
            mode='lines', name='Cloud Height (km)', line=dict(color='purple'), yaxis='y2'),
            row=2, col=1
        )
        fig.update_yaxes(title_text="Area (km²)", row=2, col=1)
        fig.update_yaxes(title_text="Height (km)", secondary_y=True, row=2, col=1)
        
        # Movement
        fig.add_trace(go.Scatter(
            x=track['datetime'], y=track['speed_kmh'], 
            mode='lines+markers', name='Speed (km/h)', line=dict(color='red')),
            row=3, col=1
        )
        fig.add_trace(go.Bar(
            x=track['datetime'], y=track['direction_deg'], 
            name='Direction (°)', marker=dict(color='orange')),
            row=3, col=1
        )
        
        # Risk indicators
        for idx, row in track.iterrows():
            if row['cyclogenesis_risk'] > 0.7:
                fig.add_vline(x=row['datetime'], line=dict(color="red", width=2, dash="dot"), row="all")
        
        fig.update_layout(
            title=f'Properties of Track {track_id}',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def analyze_cluster_development(self, track_id):
        """Analyze development patterns of a cluster track"""
        track = self.tracks_df[self.tracks_df['track_id'] == track_id]
        if track.empty:
            print(f"Track {track_id} not found")
            return
            
        # Calculate development metrics
        duration_h = (track['datetime'].max() - track['datetime'].min()).total_seconds() / 3600
        area_growth = track['area_km2'].iloc[-1] / track['area_km2'].iloc[0]
        min_tb_change = track['min_tb'].iloc[0] - track['min_tb'].iloc[-1]
        max_risk = track['cyclogenesis_risk'].max()
        
        # Classify development pattern
        if max_risk > 0.7 and min_tb_change > 15 and area_growth > 2:
            pattern = "Rapid Intensification"
        elif max_risk > 0.5 and min_tb_change > 10:
            pattern = "Moderate Intensification"
        elif area_growth > 1.5:
            pattern = "Expansion Dominated"
        else:
            pattern = "Stable or Weakening"
        
        # Create summary table
        summary = pd.DataFrame({
            'Metric': ['Duration (hours)', 'Area Growth', 'Min Tb Change', 'Max Risk', 'Development Pattern'],
            'Value': [f"{duration_h:.1f}", f"{area_growth:.1f}x", f"{min_tb_change:.1f}K", 
                      f"{max_risk*100:.1f}%", pattern]
        })
        
        # Plot correlation matrix
        corr_data = track[['min_tb', 'mean_tb', 'area_km2', 'cloud_top_height_km', 
                          'speed_kmh', 'cyclogenesis_risk']].corr()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Summary table
        ax1.axis('off')
        table = ax1.table(
            cellText=summary.values,
            colLabels=summary.columns,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        ax1.set_title(f'Track {track_id} Development Summary')
        
        # Correlation heatmap
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax2)
        ax2.set_title('Property Correlations')
        
        plt.tight_layout()
        return fig
    
    def create_web_dashboard(self):
        """Create an interactive web dashboard for track exploration"""
        if self.tracks_df.empty:
            print("No track data available")
            return
            
        # Create widgets
        track_ids = sorted(self.tracks_df['track_id'].unique())
        track_selector = widgets.Dropdown(
            options=track_ids,
            description='Track ID:',
            value=track_ids[0] if track_ids else None
        )
        
        date_range = widgets.DateRangeSlider(
            value=(self.tracks_df['datetime'].min().date(), self.tracks_df['datetime'].max().date()),
            min=self.tracks_df['datetime'].min().date(),
            max=self.tracks_df['datetime'].max().date(),
            description='Date Range:',
            layout={'width': '500px'}
        )
        
        risk_threshold = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1,
            step=0.05,
            description='Risk Threshold:',
            readout_format='.0%'
        )
        
        # Create output areas
        map_output = widgets.Output()
        plot_output = widgets.Output()
        analysis_output = widgets.Output()
        
        # Update function
        def update_dashboard(track_id, start_date, end_date, threshold):
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.max.time())
            
            # Filter data
            filtered = self.tracks_df[
                (self.tracks_df['track_id'] == track_id) & 
                (self.tracks_df['datetime'] >= start_dt) & 
                (self.tracks_df['datetime'] <= end_dt)
            ]
            
            with plot_output:
                plot_output.clear_output()
                if not filtered.empty:
                    fig = self.plot_track_properties(track_id)
                    fig.show()
                else:
                    print("No data in selected date range")
            
            with analysis_output:
                analysis_output.clear_output()
                if not filtered.empty:
                    fig = self.analyze_cluster_development(track_id)
                    plt.show()
                else:
                    print("No data in selected date range")
        
        # Create interactive widget
        interact(
            update_dashboard,
            track_id=track_selector,
            start_date=date_range.observe(lambda change: change['new'][0], names='value'),
            end_date=date_range.observe(lambda change: change['new'][1], names='value'),
            threshold=risk_threshold
        )
        
        # Display dashboard
        display(widgets.VBox([
            widgets.HBox([track_selector, date_range, risk_threshold]),
            widgets.HBox([plot_output, analysis_output])
        ]))
    
    def export_report(self, track_id, output_dir='reports'):
        """Generate a comprehensive PDF report for a track"""
        try:
            from fpdf import FPDF
            from PIL import Image
            import tempfile
        except ImportError:
            print("fpdf and PIL required for PDF generation. Install with: pip install fpdf2 pillow")
            return None
            
        os.makedirs(output_dir, exist_ok=True)
        track = self.tracks_df[self.tracks_df['track_id'] == track_id]
        if track.empty:
            print(f"Track {track_id} not found")
            return
            
        # Create PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Add title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"Tropical Cloud Cluster Track Report: ID {track_id}", ln=True, align='C')
        pdf.ln(10)
        
        # Add summary
        pdf.set_font("Arial", '', 12)
        start_time = track['datetime'].min()
        end_time = track['datetime'].max()
        duration = end_time - start_time
        max_risk = track['cyclogenesis_risk'].max()
        
        summary = f"""
        Track Duration: {duration}
        Start Time: {start_time}
        End Time: {end_time}
        Maximum Cyclogenesis Risk: {max_risk*100:.1f}%
        Distance Traveled: {track['speed_kmh'].sum() * duration.total_seconds()/3600:.1f} km
        """
        
        pdf.multi_cell(0, 8, summary)
        pdf.ln(10)
        
        # Save and add plots
        with tempfile.TemporaryDirectory() as tmpdir:
            # Properties plot
            fig = self.plot_track_properties(track_id)
            prop_path = os.path.join(tmpdir, 'properties.png')
            fig.write_image(prop_path)
            
            # 3D plot
            fig_3d = self.plot_3d_track(track_id)
            path_3d = os.path.join(tmpdir, '3d_track.png')
            fig_3d.write_image(path_3d)
            
            # Analysis plot
            fig_analysis = self.analyze_cluster_development(track_id)
            analysis_path = os.path.join(tmpdir, 'analysis.png')
            fig_analysis.savefig(analysis_path)
            
            # Add properties plot
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Cluster Properties Over Time", ln=True)
            pdf.image(prop_path, x=10, w=190)
            pdf.ln(5)
            
            # Add 3D track
            pdf.cell(0, 10, "3D Visualization", ln=True)
            pdf.image(path_3d, x=10, w=190)
            pdf.ln(5)
            
            # Add analysis
            pdf.cell(0, 10, "Development Analysis", ln=True)
            pdf.image(analysis_path, x=10, w=190)
        
        # Save PDF
        report_path = os.path.join(output_dir, f'track_{track_id}_report.pdf')
        pdf.output(report_path)
        print(f"Report saved to {report_path}")
        return report_path
    
    def get_track_summary(self):
        """Get summary statistics for all tracks"""
        if self.tracks_df.empty:
            return pd.DataFrame()
            
        summary = self.tracks_df.groupby('track_id').agg({
            'datetime': ['min', 'max', 'count'],
            'min_tb': ['min', 'mean'],
            'area_km2': ['min', 'max', 'mean'],
            'cyclogenesis_risk': ['max', 'mean'],
            'speed_kmh': ['mean', 'max'],
            'cloud_top_height_km': ['mean', 'max']
        }).round(2)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        
        # Calculate duration
        summary['duration_hours'] = (summary['datetime_max'] - summary['datetime_min']).dt.total_seconds() / 3600
        
        return summary
    
    def plot_track_comparison(self, track_ids=None):
        """Compare multiple tracks side by side"""
        if track_ids is None:
            track_ids = self.tracks_df['track_id'].unique()[:5]  # First 5 tracks
            
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=('Min Temperature', 'Area', 'Risk', 'Speed'))
        
        colors = px.colors.qualitative.Set1
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks_df[self.tracks_df['track_id'] == track_id]
            if track.empty:
                continue
                
            color = colors[i % len(colors)]
            
            # Min temperature
            fig.add_trace(go.Scatter(
                x=track['datetime'], y=track['min_tb'],
                mode='lines+markers', name=f'Track {track_id}',
                line=dict(color=color)), row=1, col=1)
            
            # Area
            fig.add_trace(go.Scatter(
                x=track['datetime'], y=track['area_km2'],
                mode='lines+markers', name=f'Track {track_id}',
                line=dict(color=color), showlegend=False), row=1, col=2)
            
            # Risk
            fig.add_trace(go.Scatter(
                x=track['datetime'], y=track['cyclogenesis_risk'],
                mode='lines+markers', name=f'Track {track_id}',
                line=dict(color=color), showlegend=False), row=2, col=1)
            
            # Speed
            fig.add_trace(go.Scatter(
                x=track['datetime'], y=track['speed_kmh'],
                mode='lines+markers', name=f'Track {track_id}',
                line=dict(color=color), showlegend=False), row=2, col=2)
        
        fig.update_layout(height=800, title_text="Track Comparison")
        return fig


if __name__ == "__main__":
    try:
        explorer = TrackExplorer()
        
        # Create interactive map
        explorer.create_interactive_map('outputs/tracks/track_map.html')
        
        # Example usage for a specific track
        if not explorer.tracks_df.empty:
            track_id = explorer.tracks_df['track_id'].iloc[0]
            
            print(f"Analyzing Track {track_id}")
            
            # Display 3D visualization
            fig_3d = explorer.plot_3d_track(track_id)
            fig_3d.show()
            
            # Display properties plot
            fig_props = explorer.plot_track_properties(track_id)
            fig_props.show()
            
            # Generate development analysis
            explorer.analyze_cluster_development(track_id)
            plt.show()
            
            # Show track summary
            summary = explorer.get_track_summary()
            print("\nTrack Summary:")
            print(summary)
            
            # Export report
            explorer.export_report(track_id)
            
        else:
            print("No track data available. Please run the tracking pipeline first.")
            
    except Exception as e:
        print(f"Error initializing Track Explorer: {str(e)}")
        print("Make sure you have the required dependencies installed:")
        print("pip install folium plotly seaborn ipywidgets") 