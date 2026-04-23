import pandas as pd
import requests
import time

# Load data
df = pd.read_csv("../telco_preprocessing/telco_processed.csv")
X = df.drop("Churn", axis=1)
sample = X.iloc[:5]
payload = {
    "dataframe_split": {
        "columns": list(sample.columns),
        "data": sample.values.tolist()
    }
}

# Tembak ke Exporter (Port 8000)
url = "http://127.0.0.1:8000/predict"
headers = {"Content-Type": "application/json"}

print("Memompa data ke API...")
# loop panjang agar data di Grafana terlihat bergerak
for i in range(100): 
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Request {i+1} | Status: {response.status_code}")
        time.sleep(2) # Jeda 2 detik
    except Exception as e:
        print(f"Error: {e}")