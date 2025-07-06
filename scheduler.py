import schedule
import time
import subprocess
from datetime import datetime

def process_new_data():
    print(f"{datetime.now()}: Processing new data...")
    subprocess.run(["python", "real_data_processor.py"])
    subprocess.run(["python", "reporting.py", "--period", "daily"])

# Schedule to run every 30 minutes
schedule.every(30).minutes.do(process_new_data)

print("Scheduler started. Processing will run every 30 minutes.")
while True:
    schedule.run_pending()
    time.sleep(60) 