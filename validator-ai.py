import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
import requests

# Telegram settings
TELEGRAM_BOT_TOKEN = 'your-bot-token-here'
TELEGRAM_USER_ID = 'your-user-id-here'

THRESHOLD = 0.7  # prediction threshold

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_USER_ID,
        "text": message
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")

def collect_metrics():
    metrics = requests.get('http://localhost:9090/api/v1/query?query=node_load1').json()
    load1 = float(metrics['data']['result'][0]['value'][1])

    metrics = requests.get('http://localhost:9090/api/v1/query?query=node_memory_MemAvailable_bytes').json()
    mem_available = float(metrics['data']['result'][0]['value'][1])

    metrics = requests.get('http://localhost:9090/api/v1/query?query=node_disk_io_time_seconds_total').json()
    disk_io = float(metrics['data']['result'][0]['value'][1])

    return [load1, mem_available, disk_io]

def fake_training_data():
    import numpy as np
    X = np.random.rand(500, 3)
    y = (X[:,0] > 0.7).astype(int)
    return X, y

model = RandomForestClassifier()
X, y = fake_training_data()
model.fit(X, y)

while True:
    metrics = collect_metrics()
    prediction = model.predict_proba([metrics])[0][1]  # risk score

    print(f"Predicted block miss probability: {prediction:.2f}")

    if prediction > THRESHOLD:
        alert_message = f"⚠️ High risk of block miss! Predicted probability: {prediction:.2f}"
        print(alert_message)
        send_telegram_alert(alert_message)

    time.sleep(60)
