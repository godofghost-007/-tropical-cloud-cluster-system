# 🚀 Quick Start Guide

Get the Tropical Cloud Cluster Monitor Dashboard running in 3 simple steps!

## Step 1: Install Dependencies

```bash
# Install all required packages
pip install -r dashboard_requirements.txt
```

## Step 2: Test Installation

```bash
# Run the test suite to verify everything works
python test_dashboard.py
```

## Step 3: Launch Dashboard

### Option A: Using Python (All Platforms)
```bash
python launch_dashboard.py
```

### Option B: Using Streamlit Directly
```bash
streamlit run dashboard.py
```

### Option C: Using Batch File (Windows)
```bash
# Double-click launch_dashboard.bat
# Or run from command line:
launch_dashboard.bat
```

## 🎯 What You'll See

1. **Dashboard opens** in your default web browser at `http://localhost:8501`
2. **Real-time metrics** showing system performance
3. **Interactive map** with cluster tracks (if data available)
4. **3D visualizations** for detailed analysis
5. **Risk alerts** for high-risk clusters

## 📊 Sample Data

If you don't have processed data yet, the dashboard will show:
- System metrics and status
- Empty visualizations with helpful messages
- Processing controls to run the pipeline

## 🔧 Troubleshooting

### Dashboard won't start?
```bash
# Check Python version (needs 3.7+)
python --version

# Reinstall dependencies
pip install --upgrade -r dashboard_requirements.txt
```

### No data showing?
```bash
# Run the processing pipeline first
python real_data_processor.py
```

### Port already in use?
```bash
# Use a different port
streamlit run dashboard.py --server.port 8502
```

## 📞 Need Help?

- 📖 Read the full documentation: `DASHBOARD_README.md`
- 🧪 Run diagnostics: `python test_dashboard.py`
- 🔍 Check logs in the Streamlit interface

---

**🌪️ Ready to monitor tropical cloud clusters!** 🚀 