#!/usr/bin/env python3
import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime
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
        # Always return medium risk (50%) if not properly trained
        return np.array([[0.5, 0.5]] * len(X))

class SystemMonitor:
    def __init__(self):
        # Configuration
        self.prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.threshold = 0.7
        
        # Initialize components
        self.prom = PrometheusConnect(url=self.prometheus_url)
        
        # Set model paths with proper permissions handling
        self.models_dir = self._get_writable_dir()
        self.model_path = os.path.join(self.models_dir, 'system_monitor_model.pkl')
        self.model = self._init_model()
        
        self.mitigation_history = []
        self.feature_names = None  # Track expected feature names
        
        # Timing settings
        self.mitigation_cooldown = 3600  # 1 hour
        self.alert_cooldown = 600        # 10 minutes
        self.last_alert_time = 0
        
        # Available metrics
        self.base_metrics = [
            'node_load1',
            'node_memory_MemAvailable_bytes',
            'node_disk_io_time_seconds_total',
            'node_network_receive_errs_total'
        ]
        
        # Initial training
        self._initial_training()
        
        print("✅ Initialized System Monitor with metrics:")
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
        """Train model with synthetic data matching exact feature count"""
        try:
            if not hasattr(self.model, 'fit'):
                return
            
            # First collect real metrics to determine feature count
            dummy_data = self.collect_metrics()
            dummy_features = self.preprocess_data(dummy_data)
            self.feature_names = list(dummy_features.columns)
            num_features = len(self.feature_names)
            
            # Generate synthetic data matching exact feature count
            X = np.random.rand(100, num_features)
            y = (X[:, 0] > 0.7).astype(int)
            
            self.model.fit(X, y)
            print(f"🔧 Initialized model with {num_features} features")
            
            # Save with atomic write to prevent corruption
            temp_path = self.model_path + ".tmp"
            joblib.dump((self.model, self.feature_names), temp_path)
            os.replace(temp_path, self.model_path)
            
        except Exception as e:
            print(f"⚠️ Initial training failed: {e}")
            print("📣 Run as root or fix directory permissions:")
            print(f"  sudo mkdir -p {self.models_dir}")
            print(f"  sudo chown $USER {self.models_dir}")
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
            df[f'load_ma_{window}'] = (
                df['node_load1']
                .rolling(window, min_periods=1)
                .mean()
            )
            df[f'mem_ma_{window}'] = (
                df['node_memory_MemAvailable_bytes']
                .rolling(window, min_periods=1)
                .mean()
            )
        
        # Temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour/24)
        
        # Interaction terms
        df['cpu_mem_ratio'] = (
            df['node_load1'] / 
            (df['node_memory_MemAvailable_bytes'] / 1e9 + 1e-6)  # Normalize
        )
        
        # Handle missing values
        df = df.ffill().bfill().fillna(0)
        
        # Ensure consistent features if we have a trained model
        if self.feature_names is not None:
            # Add any missing features with default values
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0.0
            # Select only the expected features in the right order
            df = df[self.feature_names]
        
        return df.drop(columns=['timestamp'], errors='ignore')
    
    def predict_risk(self, features):
        """Safe prediction with feature validation"""
        try:
            # Convert to numpy array to avoid feature name warnings
            features_array = features.values
            
            if not hasattr(self.model, 'predict_proba'):
                return {'probability': 0.5, 'risk_level': 'medium'}
            
            # Verify feature count matches
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
        """Basic anomaly analysis"""
        analysis = {
            'primary_issue': None,
            'confidence': 0.0
        }
        
        # CPU analysis
        cpu_load = features['node_load1'].values[0]
        if cpu_load > 5.0:
            analysis['primary_issue'] = f"High CPU load ({cpu_load:.1f})"
            analysis['confidence'] = min(90 + (cpu_load-5)*10, 99)
        
        # Memory analysis
        mem_avail = features['node_memory_MemAvailable_bytes'].values[0] / 1e9  # Convert to GB
        if mem_avail < 1.0:
            issue = f"Low memory available ({mem_avail:.1f}GB)"
            if analysis['primary_issue']:
                analysis['secondary_issue'] = issue
            else:
                analysis['primary_issue'] = issue
                analysis['confidence'] = max(analysis['confidence'], 85.0)
        
        return analysis
    
    def safe_mitigation(self, features):
        """Non-disruptive system optimizations"""
        actions = []
        
        # Check cooldown
        if self.mitigation_history:
            last_action = self.mitigation_history[-1]
            if time.time() - last_action['time'] < self.mitigation_cooldown:
                return actions
        
        try:
            # Memory optimization
            mem_avail = features['node_memory_MemAvailable_bytes'].values[0]
            if mem_avail < 1e9:  # < 1GB
                os.system("sync && echo 3 > /proc/sys/vm/drop_caches")
                actions.append("Cleared page cache")
            
            # CPU optimization
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
            
            # Rate limiting
            if current_time - self.last_alert_time < self.alert_cooldown:
                print("🛑 Alert cooldown active")
                return
            
            # Format message
            alert_text = f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            alert_text += f"🚨 {message}"
            
            if metadata:
                alert_text += "\n\n🔍 Details:\n"
                alert_text += json.dumps(metadata, indent=2)
            
            # Telegram notification
            if self.telegram_token and self.telegram_chat_id:
                import requests
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                payload = {
                    'chat_id': self.telegram_chat_id,
                    'text': alert_text,
                    'parse_mode': 'Markdown'
                }
                requests.post(url, json=payload)
            
            # Console fallback
            print(f"📢 {alert_text}")
            
            # Record alert
            self.last_alert_time = current_time
            
        except Exception as e:
            print(f"⚠️ Alert failed: {str(e)}")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        print("🚀 Starting monitoring loop (60s intervals)")
        
        while True:
            try:
                # Data collection
                raw_data = self.collect_metrics()
                features = self.preprocess_data(raw_data)
                
                # Prediction
                prediction = self.predict_risk(features)
                print(f"📊 System status: {prediction['risk_level']} ({prediction['probability']:.2f})")
                
                # High risk handling
                if prediction['risk_level'] == 'high':
                    analysis = self.analyze_anomaly(features)
                    recommendations = self.generate_recommendations(analysis)
                    actions = self.safe_mitigation(features)
                    
                    self.send_alert(
                        "High system load detected",
                        {
                            'analysis': analysis,
                            'actions_taken': actions,
                            'recommendations': recommendations,
                            'current_metrics': {
                                'cpu_load': features['node_load1'].values[0],
                                'mem_available_gb': features['node_memory_MemAvailable_bytes'].values[0] / 1e9,
                                'disk_io': features['node_disk_io_time_seconds_total'].values[0],
                                'network_errors': features['node_network_receive_errs_total'].values[0]
                            }
                        }
                    )
                
                time.sleep(60)
                
            except Exception as e:
                print(f"❌ Monitor error: {str(e)}")
                time.sleep(300)

if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.monitor_loop()