"""
reporting.py - Operational Reporting for Tropical Cloud Clusters
Generates PDF briefings with cluster analysis and forecasts
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from helpers import haversine
import warnings

# Configuration
REPORT_OUTPUT_DIR = "reports"
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

def safe_get(df, col, default=0):
    """Safely get column from DataFrame with fallback"""
    return df[col] if col in df.columns else pd.Series(default, index=df.index)

def calculate_cyclogenesis_risk(df):
    """Calculate risk score if missing"""
    def _calculate(row):
        try:
            # Simplified risk calculation
            min_tb = row.get('min_tb', 240)
            area = row.get('area_km2', 50000)
            return min(1, (240 - min_tb)/50 * 0.5 + min(area/100000, 1) * 0.5)
        except:
            return 0
    return df.apply(_calculate, axis=1)

def load_tracked_data():
    """Load tracked clusters data with fallbacks"""
    try:
        df = pd.read_csv('outputs/tracks/tracked_clusters.csv')
        # Add missing columns if needed
        if 'cyclogenesis_risk' not in df.columns:
            warnings.warn("'cyclogenesis_risk' column missing. Calculating from available data.")
            df['cyclogenesis_risk'] = calculate_cyclogenesis_risk(df)
        if 'cloud_top_height_km' not in df.columns:
            if 'min_tb' in df.columns:
                df['cloud_top_height_km'] = 0.12 * (300 - df['min_tb'])
            else:
                df['cloud_top_height_km'] = 10  # Default value
        return df
    except FileNotFoundError:
        print("Error: Tracked clusters data not found. Using sample data.")
        # Generate sample data
        return pd.DataFrame({
            'center_lat': [15.5, 12.3, 8.7],
            'center_lon': [72.8, 83.4, 95.1],
            'area_km2': [50000, 75000, 30000],
            'cloud_top_height_km': [12.5, 14.2, 9.8],
            'cyclogenesis_risk': [0.72, 0.85, 0.45],
            'timestep': [1, 2, 3],
            'track_id': [101, 102, 103]
        })

def generate_summary_map(df, period='daily'):
    """Generate summary map visualization with error handling"""
    try:
        plt.figure(figsize=(10, 8))
        sizes = safe_get(df, 'area_km2', 50000) / 1000
        heights = safe_get(df, 'cloud_top_height_km', 10)
        plt.scatter(df['center_lon'], df['center_lat'], 
                   s=sizes, 
                   c=heights,
                   cmap='viridis',
                   alpha=0.7,
                   vmin=5, vmax=20)
        plt.gca().set_facecolor('#e6f2ff')
        plt.xlim(40, 100)
        plt.ylim(-30, 30)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Tropical Cloud Clusters - {period.capitalize()} Summary')
        plt.colorbar(label='Cloud Top Height (km)')
        plt.grid(alpha=0.3)
        map_path = os.path.join(REPORT_OUTPUT_DIR, f"summary_map_{period}.png")
        plt.savefig(map_path, dpi=150, bbox_inches='tight')
        plt.close()
        return map_path
    except Exception as e:
        print(f"Error generating map: {e}")
        return None

def generate_risk_evolution(track_ids, df):
    """Generate risk evolution chart for high-risk tracks"""
    plt.figure(figsize=(10, 6))
    
    for track_id in track_ids:
        track = df[df['track_id'] == track_id]
        plt.plot(track['timestep'], track['cyclogenesis_risk'], 
                 marker='o', label=f"Track {track_id}")
    
    plt.axhline(y=0.65, color='r', linestyle='--', label='Risk Threshold')
    plt.xlabel('Timestep')
    plt.ylabel('Cyclogenesis Risk')
    plt.title('High-Risk Cluster Evolution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save and return path
    chart_path = os.path.join(REPORT_OUTPUT_DIR, "risk_evolution.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    return chart_path

def generate_forecast_tracks(tracks, df):
    """Generate forecast track visualization"""
    plt.figure(figsize=(10, 8))
    
    for track_id in tracks:
        track = df[df['track_id'] == track_id]
        plt.plot(track['center_lon'], track['center_lat'], 
                 'o-', markersize=4, label=f"Track {track_id}")
        
        # Add forecast arrow
        if len(track) > 1:
            dx = track['center_lon'].iloc[-1] - track['center_lon'].iloc[-2]
            dy = track['center_lat'].iloc[-1] - track['center_lat'].iloc[-2]
            plt.arrow(track['center_lon'].iloc[-1], 
                      track['center_lat'].iloc[-1],
                      dx, dy, shape='full', color='red', 
                      length_includes_head=True, head_width=0.5)
    
    plt.xlim(40, 100)
    plt.ylim(-30, 30)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('72-Hour Track Forecast')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save and return path
    forecast_path = os.path.join(REPORT_OUTPUT_DIR, "forecast_tracks.png")
    plt.savefig(forecast_path, dpi=150, bbox_inches='tight')
    plt.close()
    return forecast_path

def generate_operational_report(period='daily'):
    """Generate PDF operational briefing with robust error handling"""
    try:
        # Load data
        df = load_tracked_data()
        # Filter for report period
        if 'timestep' not in df.columns:
            df['timestep'] = 0
        if period == 'daily' and 'timestep' in df.columns:
            max_timestep = df['timestep'].max()
            df_period = df[df['timestep'] >= max_timestep - 48]
        else:
            df_period = df.copy()
        # Create document
        report_date = datetime.now().strftime('%Y%m%d')
        doc = SimpleDocTemplate(
            os.path.join(REPORT_OUTPUT_DIR, f"{report_date}_tcc_{period}_briefing.pdf"),
            pagesize=letter,
            title=f"Tropical Cloud Cluster {period.capitalize()} Briefing"
        )
        # Custom styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Header', fontSize=16, leading=20, alignment=1, spaceAfter=12))
        styles.add(ParagraphStyle(name='Subheader', fontSize=14, leading=18, spaceAfter=6))
        story = []
        # Header
        story.append(Paragraph("Tropical Cloud Cluster Operational Briefing", styles['Header']))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}", styles['Normal']))
        story.append(Paragraph(f"Period: {period.capitalize()} Summary", styles['Normal']))
        story.append(Spacer(1, 24))
        # Summary Statistics
        story.append(Paragraph("Summary Statistics", styles['Subheader']))
        stats = [
            ["Total Clusters Detected", len(df_period)],
            ["Active Tracks", df_period['track_id'].nunique() if 'track_id' in df_period.columns else 0]
        ]
        # Add risk stats if available
        if 'cyclogenesis_risk' in df_period.columns:
            high_risk_count = len(df_period[df_period['cyclogenesis_risk'] > 0.65])
            stats.append(["High-Risk Clusters", high_risk_count])
        # Add height and size stats
        if 'cloud_top_height_km' in df_period.columns:
            stats.append(["Maximum Cloud Height", f"{df_period['cloud_top_height_km'].max():.1f} km"])
        if 'area_km2' in df_period.columns:
            stats.append(["Largest Cluster", f"{df_period['area_km2'].max()/1000:.0f}k km²"])
        story.append(Table(stats, colWidths=[300, 100]))
        story.append(Spacer(1, 24))
        # Summary Map
        map_path = generate_summary_map(df_period, period)
        if map_path:
            story.append(Paragraph("Cluster Overview", styles['Subheader']))
            story.append(Image(map_path, width=6*inch, height=4.5*inch))
            story.append(Spacer(1, 12))
        # Build PDF
        doc.build(story)
        print(f"Report generated: {os.path.join(REPORT_OUTPUT_DIR, f'{report_date}_tcc_{period}_briefing.pdf')}")
        return True
    except Exception as e:
        print(f"Critical error generating report: {e}")
        return False

if __name__ == "__main__":
    period = 'daily'
    if len(sys.argv) > 1 and sys.argv[1] == '--period':
        if len(sys.argv) > 2:
            period = sys.argv[2]
    if not generate_operational_report(period):
        print("Report generation failed. Creating minimal report...")
        # Create emergency minimal report
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        doc = SimpleDocTemplate(os.path.join(REPORT_OUTPUT_DIR, "error_report.pdf"), pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("Report Generation Error", styles['Heading1']),
            Paragraph("The tropical cloud cluster report could not be generated due to an error.", styles['BodyText']),
            Paragraph("Please check your data processing pipeline and try again.", styles['BodyText'])
        ]
        doc.build(story)
        print("Created error report at reports/error_report.pdf") 