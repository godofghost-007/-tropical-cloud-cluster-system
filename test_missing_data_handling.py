#!/usr/bin/env python3
"""
Test script for missing data handling in the dashboard
Tests various scenarios with incomplete or malformed data
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append('.')

def create_test_data_scenarios():
    """Create various test data scenarios with missing data"""
    
    # Scenario 1: Complete data
    complete_data = pd.DataFrame({
        'track_id': [1, 1, 1, 2, 2, 2],
        'datetime': pd.date_range('2024-01-01', periods=6, freq='H'),
        'center_lat': [10.0, 10.1, 10.2, 15.0, 15.1, 15.2],
        'center_lon': [80.0, 80.1, 80.2, 85.0, 85.1, 85.2],
        'min_tb': [200, 195, 190, 210, 205, 200],
        'mean_tb': [220, 215, 210, 230, 225, 220],
        'area_km2': [50000, 55000, 60000, 45000, 50000, 55000],
        'cloud_top_height_km': [12, 13, 14, 11, 12, 13],
        'cyclogenesis_risk': [0.3, 0.5, 0.8, 0.2, 0.4, 0.6]
    })
    
    # Scenario 2: Missing essential columns
    missing_essential = pd.DataFrame({
        'min_tb': [200, 195, 190],
        'mean_tb': [220, 215, 210],
        'area_km2': [50000, 55000, 60000]
    })
    
    # Scenario 3: Partial data with NaN values
    partial_data = pd.DataFrame({
        'track_id': [1, 1, 1, 2, 2, 2],
        'datetime': pd.date_range('2024-01-01', periods=6, freq='H'),
        'center_lat': [10.0, np.nan, 10.2, 15.0, 15.1, np.nan],
        'center_lon': [80.0, 80.1, np.nan, 85.0, np.nan, 85.2],
        'min_tb': [200, 195, 190, np.nan, 205, 200],
        'cyclogenesis_risk': [0.3, np.nan, 0.8, 0.2, 0.4, np.nan]
    })
    
    # Scenario 4: Empty dataframe
    empty_data = pd.DataFrame()
    
    # Scenario 5: Data with wrong types
    wrong_types = pd.DataFrame({
        'track_id': ['a', 'b', 'c'],  # Should be numeric
        'datetime': ['2024-01-01', '2024-01-02', '2024-01-03'],  # Should be datetime
        'center_lat': ['10.0', '10.1', '10.2'],  # Should be float
        'cyclogenesis_risk': ['high', 'medium', 'low']  # Should be float
    })
    
    return {
        'complete': complete_data,
        'missing_essential': missing_essential,
        'partial': partial_data,
        'empty': empty_data,
        'wrong_types': wrong_types
    }

def test_dashboard_handling():
    """Test dashboard handling of various data scenarios"""
    
    print("🧪 Testing Dashboard Missing Data Handling")
    print("=" * 50)
    
    # Import dashboard
    try:
        from dashboard import Dashboard
        print("✅ Dashboard imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import dashboard: {e}")
        return
    
    # Create test scenarios
    scenarios = create_test_data_scenarios()
    
    # Test each scenario
    for scenario_name, test_data in scenarios.items():
        print(f"\n📊 Testing Scenario: {scenario_name.upper()}")
        print("-" * 30)
        
        # Create dashboard instance
        dashboard = Dashboard()
        
        # Manually set the test data
        dashboard.tracks_df = test_data.copy()
        
        # Test data quality validation
        try:
            quality_report = dashboard.validate_data_quality()
            print(f"✅ Quality validation: {quality_report['status']}")
            print(f"   Quality Score: {quality_report['quality_score']*100:.1f}%")
            print(f"   Message: {quality_report['message']}")
            print(f"   Total Tracks: {quality_report['total_tracks']}")
            print(f"   Total Records: {quality_report['total_records']}")
        except Exception as e:
            print(f"❌ Quality validation failed: {e}")
        
        # Test safe_get method
        try:
            if not test_data.empty:
                test_row = test_data.iloc[0]
                safe_value = dashboard.safe_get(test_row, 'min_tb', default=999)
                print(f"✅ Safe get test: {safe_value}")
        except Exception as e:
            print(f"❌ Safe get test failed: {e}")
        
        # Test has_column method
        try:
            has_track_id = dashboard.has_column('track_id')
            print(f"✅ Has track_id column: {has_track_id}")
        except Exception as e:
            print(f"❌ Has column test failed: {e}")
        
        # Test visualization methods (should not crash)
        try:
            if not test_data.empty and 'track_id' in test_data.columns:
                timeline = dashboard.create_timeline()
                print(f"✅ Timeline creation: {'Success' if timeline else 'No data'}")
            else:
                print("⏭️  Timeline creation: Skipped (no track_id)")
        except Exception as e:
            print(f"❌ Timeline creation failed: {e}")
        
        try:
            if not test_data.empty and 'track_id' in test_data.columns:
                map_viz = dashboard.create_map()
                print(f"✅ Map creation: {'Success' if map_viz else 'No data'}")
            else:
                print("⏭️  Map creation: Skipped (no track_id)")
        except Exception as e:
            print(f"❌ Map creation failed: {e}")

def test_load_tracks_method():
    """Test the load_tracks method with various file scenarios"""
    
    print("\n🔄 Testing load_tracks Method")
    print("=" * 30)
    
    try:
        from dashboard import Dashboard
        dashboard = Dashboard()
        
        # Test with non-existent file
        print("\n📁 Testing with non-existent file:")
        result = dashboard.load_tracks()
        print(f"   Result: {'Empty DataFrame' if result.empty else 'Has data'}")
        print(f"   Expected: Empty DataFrame")
        
        # Test with malformed CSV (create a test file)
        test_csv_content = """track_id,datetime,center_lat,center_lon,min_tb
1,2024-01-01 00:00:00,10.0,80.0,200
2,invalid_date,10.1,80.1,195
3,2024-01-01 02:00:00,10.2,80.2,190"""
        
        with open('test_tracks.csv', 'w') as f:
            f.write(test_csv_content)
        
        # Temporarily change the tracks file
        original_file = dashboard.tracks_file
        dashboard.tracks_file = 'test_tracks.csv'
        
        print("\n📁 Testing with malformed CSV:")
        result = dashboard.load_tracks()
        print(f"   Result: {'Empty DataFrame' if result.empty else 'Has data'}")
        print(f"   Columns: {list(result.columns)}")
        
        # Clean up
        dashboard.tracks_file = original_file
        if os.path.exists('test_tracks.csv'):
            os.remove('test_tracks.csv')
            
    except Exception as e:
        print(f"❌ load_tracks test failed: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting Missing Data Handling Tests")
    print("=" * 60)
    
    # Test dashboard handling
    test_dashboard_handling()
    
    # Test load_tracks method
    test_load_tracks_method()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("\n📋 Summary:")
    print("   - Dashboard should handle missing columns gracefully")
    print("   - Safe value retrieval should work with fallbacks")
    print("   - Visualizations should not crash with incomplete data")
    print("   - Data quality reporting should provide meaningful feedback")
    print("   - Empty or malformed data should be handled safely")

if __name__ == "__main__":
    main() 