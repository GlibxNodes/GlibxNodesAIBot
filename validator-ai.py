#!/usr/bin/env python3
import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator
from prometheus_api_client import PrometheusConnect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SafeModel(BaseEstimator):
    """Fallback model that provides safe predictions"""
    def fit(self, X, y):
        return self
    
    def predict_proba(self, X):
        return np.array([[0.5, 0.5]] * len(X))

class SystemMonitor:
    def __init__(self):
        # Configuration
        self.prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.threshold = 0.85
        
        # Healthy baseline thresholds
        self.cpu_threshold = 5.0
        self.mem_threshold = 1.0
        
        # Initialize components
        self.prom = PrometheusConnect(url=self.prometheus_url)
        self.models_dir = self._get_writable_dir()
        self.model_path = os.path.join(self.models_dir, 'system_monitor_model.pkl')
        self.model = self._init_model()
        
        self.mitigation_history = []
        self.feature_names = None
        self.training_data = []
        
        # Timing settings
        self.mitigation_cooldown = 3600
        self.alert_cooldown = 600
        self.last_alert_time = 0
        self.retrain_interval = 86400
        self.last_retrain = time.time()
        
        # Available metrics
        self.base_metrics = [
            'node_load1',
            'node_memory_MemAvailable_bytes',
            'node_disk_io_time_seconds_total',
            'node_network_receive_errs_total'
        ]
        
        self._initial_training()
        print("✅ Initialized System Monitor")
        for metric in self.base_metrics:
            print(f"   - {metric}")

    def _get_writable_dir(self):
        """Find a writable directory for storing models"""
        candidates = [
            os.path.join(os.path.expanduser("~"), ".system_monitor"),
            "/var/lib/system_monitor",
            "models"
        ]
        
        for path in candidates:
            try:
                os.makedirs(path, exist_ok=True)
                if os.access(path, os.W_OK):
                    print(f"📁 Using storage directory: {path}")
                    return path
            except Exception as e:
                continue
                
        print("⚠️ Could not find writable directory, using temporary storage")
        return os.path.join(tempfile.gettempdir(), "system_monitor")

    def _init_model(self):
        """Initialize model with proper error handling"""
        try:
            if os.path.exists(self.model_path):
                model, self.feature_names = joblib.load(self.model_path)
                if hasattr(model, 'predict_proba'):
                    print("✅ Loaded trained model from disk")
                    return model
        except Exception as e:
            print(f"⚠️ Model loading failed: {e}")
        
        print("🆕 Initializing new safe model")
        return GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

    def _initial_training(self):
        """Train model with synthetic data that includes both classes"""
        try:
            if not hasattr(self.model, 'fit'):
                return
            
            dummy_data = self.collect_metrics()
            dummy_features = self.preprocess_data(dummy_data)
            self.feature_names = list(dummy_features.columns)
            num_features = len(self.feature_names)
            
            # Generate balanced synthetic data (50% normal, 50% anomalies)
            X_normal = np.random.rand(50, num_features) * 0.7  # Normal state
            X_anomaly = 0.7 + np.random.rand(50, num_features) * 0.3  # Anomalous state
            X = np.vstack([X_normal, X_anomaly])
            y = np.array([0]*50 + [1]*50)  # Labels
            
            self.model.fit(X, y)
            print(f"🔧 Initialized model with {num_features} features")
            
            temp_path = self.model_path + ".tmp"
            joblib.dump((self.model, self.feature_names), temp_path)
            os.replace(temp_path, self.model_path)
            
        except Exception as e:
            print(f"⚠️ Initial training failed: {e}")
            self.model = SafeModel()

    def collect_metrics(self):
        """Collect system metrics safely"""
        metrics = {'timestamp': datetime.now().isoformat()}
        
        for metric in self.base_metrics:
            try:
                result = self.prom.get_current_metric_value(metric)
                if result:
                    metrics[metric] = float(result[0]['value'][1])
                else:
                    metrics[metric] = 0.0
                    print(f"⚠️ No data for {metric}")
            except Exception as e:
                metrics[metric] = 0.0
                print(f"⚠️ Failed to query {metric}: {str(e)}")
        
        return pd.DataFrame([metrics])

    def preprocess_data(self, df):
        """Prepare data for model prediction with consistent features"""
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create rolling features
        for window in [5, 15, 60]:
            df[f'load_ma_{window}'] = df['node_load1'].rolling(window, min_periods=1).mean()
            df[f'mem_ma_{window}'] = df['node_memory_MemAvailable_bytes'].rolling(window, min_periods=1).mean()
        
        # Temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour/24)
        
        # Interaction terms
        df['cpu_mem_ratio'] = df['node_load1'] / (df['node_memory_MemAvailable_bytes'] / 1e9 + 1e-6)
        
        # Handle missing values
        df = df.ffill().bfill().fillna(0)
        
        # Ensure consistent features
        if self.feature_names is not None:
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0.0
            df = df[self.feature_names]
        
        return df.drop(columns=['timestamp'], errors='ignore')

    def predict_risk(self, features):
        """Safe prediction with feature validation"""
        try:
            features_array = features.values
            
            if not hasattr(self.model, 'predict_proba'):
                return {'probability': 0.5, 'risk_level': 'medium'}
            
            if features_array.shape[1] != self.model.n_features_in_:
                print(f"⚠️ Feature mismatch: Got {features_array.shape[1]}, expected {self.model.n_features_in_}")
                return {'probability': 0.5, 'risk_level': 'medium'}
                
            proba = self.model.predict_proba(features_array)[0][1]
            return {
                'probability': proba,
                'risk_level': 'high' if proba > self.threshold else 'medium' if proba > 0.5 else 'low'
            }
        except Exception as e:
            print(f"⚠️ Prediction failed: {str(e)}")
            return {'probability': 0.5, 'risk_level': 'medium'}

    def analyze_anomaly(self, features):
        """Detect actual system anomalies"""
        analysis = {
            'primary_issue': None,
            'confidence': 0.0,
            'metrics': {
                'cpu_load': features['node_load1'].values[0],
                'mem_available_gb': features['node_memory_MemAvailable_bytes'].values[0] / 1e9,
                'disk_io': features['node_disk_io_time_seconds_total'].values[0],
                'network_errors': features['node_network_receive_errs_total'].values[0]
            }
        }
        
        cpu_load = analysis['metrics']['cpu_load']
        mem_avail = analysis['metrics']['mem_available_gb']
        
        if cpu_load > self.cpu_threshold:
            analysis['primary_issue'] = f"High CPU load ({cpu_load:.1f})"
            analysis['confidence'] = min(90 + (cpu_load-self.cpu_threshold)*10, 99)
        
        if mem_avail < self.mem_threshold:
            issue = f"Low memory available ({mem_avail:.1f}GB)"
            if analysis['primary_issue']:
                analysis['secondary_issue'] = issue
            else:
                analysis['primary_issue'] = issue
                analysis['confidence'] = max(analysis['confidence'], 85.0)
        
        if analysis['primary_issue'] is None and self.model.__class__.__name__ != 'SafeModel':
            print("⚠️ Model prediction doesn't match actual system state")
            self._retrain_model()
        
        return analysis

    def should_alert(self, prediction, analysis):
        """Determine if alert should be sent"""
        return prediction['risk_level'] == 'high' and analysis['primary_issue'] is not None

    def safe_mitigation(self, features):
        """Non-disruptive system optimizations"""
        actions = []
        
        if self.mitigation_history and time.time() - self.mitigation_history[-1]['time'] < self.mitigation_cooldown:
            return actions
        
        try:
            mem_avail = features['node_memory_MemAvailable_bytes'].values[0]
            if mem_avail < 1e9:
                os.system("sync && echo 3 > /proc/sys/vm/drop_caches")
                actions.append("Cleared page cache")
            
            cpu_load = features['node_load1'].values[0]
            if cpu_load > 5.0:
                os.system("pkill -f low_priority_process")
                actions.append("Terminated low priority processes")
            
            if actions:
                self.mitigation_history.append({
                    'time': time.time(),
                    'actions': actions,
                    'metrics': features.to_dict()
                })
        
        except Exception as e:
            print(f"⚠️ Mitigation failed: {str(e)}")
        
        return actions

    def generate_recommendations(self, analysis):
        """Generate recommendations based on analysis"""
        recs = []
        
        if 'CPU' in (analysis.get('primary_issue') or ''):
            recs.extend([
                "Check for CPU-intensive processes (top/htop)",
                "Consider upgrading CPU or optimizing workloads"
            ])
        
        if 'memory' in (analysis.get('primary_issue') or '').lower():
            recs.extend([
                "Review memory usage (free -h)",
                "Check for memory leaks in applications"
            ])
        
        return recs or ["Check system logs for errors"]

    def send_alert(self, message, metadata=None):
        """Send alert with rate limiting"""
        try:
            current_time = time.time()
            
            if current_time - self.last_alert_time < self.alert_cooldown:
                print("🛑 Alert cooldown active")
                return
            
            alert_text = f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n🚨 {message}"
            
            if metadata:
                alert_text += "\n\n🔍 Details:\n" + json.dumps(metadata, indent=2)
            
            if self.telegram_token and self.telegram_chat_id:
                import requests
                requests.post(
                    f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                    json={
                        'chat_id': self.telegram_chat_id,
                        'text': alert_text,
                        'parse_mode': 'Markdown'
                    }
                )
            
            print(f"📢 {alert_text}")
            self.last_alert_time = current_time
            
        except Exception as e:
            print(f"⚠️ Alert failed: {str(e)}")

    def _retrain_model(self):
        """Retrain model with collected data"""
        if len(self.training_data) < 100:
            print("⚠️ Not enough data for retraining")
            return
            
        try:
            df = pd.concat(self.training_data[-100:])
            features = self.preprocess_data(df)
            
            y = np.where(
                (features['node_load1'] > self.cpu_threshold) | 
                (features['node_memory_MemAvailable_bytes']/1e9 < self.mem_threshold),
                1, 0
            )
            
            if len(np.unique(y)) >= 2:
                self.model.fit(features.values, y)
                temp_path = self.model_path + ".tmp"
                joblib.dump((self.model, self.feature_names), temp_path)
                os.replace(temp_path, self.model_path)
                print("✅ Model retrained successfully")
            else:
                print("⚠️ Not enough class diversity for retraining")
                
        except Exception as e:
            print(f"⚠️ Retraining failed: {str(e)}")

    def monitor_loop(self):
        """Main monitoring loop"""
        print("🚀 Starting monitoring loop (60s intervals)")
        
        while True:
            try:
                raw_data = self.collect_metrics()
                self.training_data.append(raw_data)
                features = self.preprocess_data(raw_data)
                prediction = self.predict_risk(features)
                analysis = self.analyze_anomaly(features)
                
                if time.time() - self.last_retrain > self.retrain_interval:
                    self._retrain_model()
                    self.last_retrain = time.time()
                
                print(f"📊 Status: {prediction['risk_level']} ({prediction['probability']:.2f}) | "
                      f"CPU: {analysis['metrics']['cpu_load']:.1f} | "
                      f"Mem: {analysis['metrics']['mem_available_gb']:.1f}GB")
                
                if self.should_alert(prediction, analysis):
                    recommendations = self.generate_recommendations(analysis)
                    actions = self.safe_mitigation(features)
                    
                    self.send_alert(
                        "⚠️ System anomaly detected",
                        {
                            'analysis': analysis,
                            'actions_taken': actions,
                            'recommendations': recommendations,
                            'prediction_confidence': prediction['probability']
                        }
                    )
                
                time.sleep(60)
                
            except Exception as e:
                print(f"❌ Monitor error: {str(e)}")
                time.sleep(300)

if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.monitor_loop()