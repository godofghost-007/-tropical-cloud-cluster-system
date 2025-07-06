import streamlit as st
import psutil
import time
import pandas as pd

st.title("System Monitoring Dashboard")

# System metrics
col1, col2, col3 = st.columns(3)
col1.metric("CPU Usage", f"{psutil.cpu_percent()}%")
col2.metric("Memory Usage", f"{psutil.virtual_memory().percent}%")
col3.metric("Disk Space", f"{psutil.disk_usage('/').percent}%")

# Processing statistics
if st.button("Refresh Data"):
    try:
        df = pd.read_csv('processing_stats.csv')
        st.line_chart(df.set_index('timestamp'))
    except:
        st.warning("No statistics available")

# Alert log
st.subheader("Latest Alerts")
st.json([
    {"time": "2023-07-15 12:30", "cluster_id": 105, "message": "Rapid intensification detected"},
    {"time": "2023-07-15 11:45", "cluster_id": 102, "message": "Landfall predicted in 24h"}
]) 