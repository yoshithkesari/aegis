"""
Stream Replayer - Replays data streams for testing and monitoring
"""

import pandas as pd
import numpy as np
from typing import Iterator, Optional, Dict, Any
from datetime import datetime, timedelta


class StreamReplayer:
    """Replays data streams with configurable batch size and timing"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        batch_size: int = 100,
        delay_ms: int = 0
    ):
        self.data = data
        self.batch_size = batch_size
        self.delay_ms = delay_ms
        self.current_index = 0
        self.total_rows = len(data)
    
    def replay(self) -> Iterator[pd.DataFrame]:
        """Replay the stream in batches"""
        while self.current_index < self.total_rows:
            end_index = min(self.current_index + self.batch_size, self.total_rows)
            batch = self.data.iloc[self.current_index:end_index].copy()
            
            self.current_index = end_index
            yield batch
    
    def replay_with_timestamps(self) -> Iterator[Dict[str, Any]]:
        """Replay with timestamps for realistic streaming"""
        current_time = datetime.utcnow()
        
        for batch in self.replay():
            yield {
                "data": batch,
                "timestamp": current_time.isoformat(),
                "batch_size": len(batch)
            }
            
            if self.delay_ms > 0:
                import time
                time.sleep(self.delay_ms / 1000.0)
            
            current_time += timedelta(seconds=1)
    
    def reset(self):
        """Reset the replayer to the beginning"""
        self.current_index = 0
    
    def get_progress(self) -> float:
        """Get replay progress (0-1)"""
        return self.current_index / self.total_rows if self.total_rows > 0 else 0.0


class PredictionLogger:
    """Logs predictions to a store for monitoring"""
    
    def __init__(self):
        self.predictions: list = []
    
    def log(
        self,
        features: Dict[str, Any],
        prediction: float,
        probability: float,
        actual: Optional[float] = None,
        model_version: str = "champion"
    ):
        """Log a prediction"""
        self.predictions.append({
            "timestamp": datetime.utcnow().isoformat(),
            "features": features,
            "prediction": prediction,
            "probability": probability,
            "actual": actual,
            "model_version": model_version
        })
    
    def get_recent(self, n: int = 1000) -> list:
        """Get recent predictions"""
        return self.predictions[-n:]
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get predictions as DataFrame"""
        return pd.DataFrame(self.predictions)
    
    def clear(self):
        """Clear all predictions"""
        self.predictions = []


class MetricsStore:
    """Stores metrics for monitoring and evaluation"""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {
            "timestamp": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "auc": [],
            "drift_detected": [],
            "model_version": []
        }
    
    def record(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        auc: float,
        drift_detected: bool,
        model_version: str = "champion"
    ):
        """Record metrics"""
        self.metrics["timestamp"].append(datetime.utcnow().isoformat())
        self.metrics["accuracy"].append(accuracy)
        self.metrics["precision"].append(precision)
        self.metrics["recall"].append(recall)
        self.metrics["f1"].append(f1)
        self.metrics["auc"].append(auc)
        self.metrics["drift_detected"].append(drift_detected)
        self.metrics["model_version"].append(model_version)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get metrics as DataFrame"""
        return pd.DataFrame(self.metrics)
    
    def get_latest(self) -> Dict[str, Any]:
        """Get latest metrics"""
        if not self.metrics["timestamp"]:
            return {}
        
        return {
            "timestamp": self.metrics["timestamp"][-1],
            "accuracy": self.metrics["accuracy"][-1],
            "precision": self.metrics["precision"][-1],
            "recall": self.metrics["recall"][-1],
            "f1": self.metrics["f1"][-1],
            "auc": self.metrics["auc"][-1],
            "drift_detected": self.metrics["drift_detected"][-1],
            "model_version": self.metrics["model_version"][-1]
        }
    
    def clear(self):
        """Clear all metrics"""
        for key in self.metrics:
            self.metrics[key] = []
