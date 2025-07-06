@echo off
echo 🌪️ Tropical Cloud Cluster Monitor Dashboard
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

REM Check if dashboard files exist
if not exist "dashboard.py" (
    echo ❌ dashboard.py not found
    echo Please ensure you're in the correct directory
    pause
    exit /b 1
)

REM Install dependencies if needed
echo 📦 Checking dependencies...
python -c "import streamlit, plotly, pandas, numpy, yaml, psutil, folium, streamlit_folium" >nul 2>&1
if errorlevel 1 (
    echo Installing missing dependencies...
    pip install -r dashboard_requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo ✅ Dependencies ready
echo.

REM Start the dashboard
echo 🚀 Starting dashboard...
echo 📊 Dashboard will open in your default web browser
echo 🌐 URL: http://localhost:8501
echo ⏹️  Press Ctrl+C to stop the dashboard
echo.

streamlit run dashboard.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false

echo.
echo 🛑 Dashboard stopped
pause 