
# Sample INSAT-3D Data Description

## Overview
This directory contains synthetic INSAT-3D IRBRT (Infrared Brightness Temperature) data files
created for testing the Tropical Cloud Cluster (TCC) detection and tracking system.

## File Format
- **Format**: NetCDF4
- **Naming Convention**: INSAT3D_IRBRT_YYYYMMDDHHMM.nc
- **Example**: INSAT3D_IRBRT_202407010000.nc

## Data Structure
- **Variable**: IRBRT (Brightness Temperature)
- **Units**: Kelvin (K)
- **Dimensions**: [latitude, longitude]
- **Coordinate System**: Geographic (lat/lon)

## Geographic Coverage
- **Latitude**: 5°N to 25°N (Bay of Bengal region)
- **Longitude**: 60°E to 90°E
- **Resolution**: 0.5° x 0.5° (~55 km)
- **Grid Size**: 41 x 61 pixels

## Temporal Coverage
- **Start Date**: July 1, 2024, 00:00 UTC
- **End Date**: July 2, 2024, 18:00 UTC
- **Interval**: 6 hours
- **Total Files**: 8

## Synthetic Features
The data includes realistic convective regions:
1. **Primary Convection**: Near 15°N, 75°E (active for first 1.5 days)
2. **Secondary Convection**: Near 12°N, 82°E (developing on day 1)

## Usage
These files can be processed by the TCC processor:
```bash
python run_tcc_processor.py
```

## Real Data Replacement
Replace these files with actual INSAT-3D data when available.
Ensure real data follows the same naming convention and format.
