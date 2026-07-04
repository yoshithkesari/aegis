"""
API Usage Example for Agentic Model Monitoring

This example demonstrates how to interact with the REST API
"""

import requests
import json
import time
from typing import Dict, Any


class MonitoringAPIClient:
    """Client for interacting with the Monitoring API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def register_model(
        self,
        model_id: str,
        model_name: str,
        model_type: str,
        model_path: str,
        baseline_metrics: Dict[str, Any],
        feature_names: list,
        target_name: str
    ) -> Dict[str, Any]:
        """Register a model for monitoring"""
        url = f"{self.base_url}/api/v1/models"
        payload = {
            "model_id": model_id,
            "model_name": model_name,
            "model_type": model_type,
            "model_path": model_path,
            "baseline_metrics": baseline_metrics,
            "feature_names": feature_names,
            "target_name": target_name
        }
        response = requests.post(url, json=payload)
        return response.json()
    
    def log_prediction(
        self,
        model_id: str,
        features: Dict[str, Any],
        prediction: Any,
        actual_value: Any = None,
        prediction_probability: float = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Log a prediction"""
        url = f"{self.base_url}/api/v1/predictions"
        payload = {
            "model_id": model_id,
            "features": features,
            "prediction": prediction,
            "actual_value": actual_value,
            "prediction_probability": prediction_probability,
            "metadata": metadata
        }
        response = requests.post(url, json=payload)
        return response.json()
    
    def detect_drift(
        self,
        model_id: str,
        reference_data: list,
        current_data: list
    ) -> Dict[str, Any]:
        """Detect data drift"""
        url = f"{self.base_url}/api/v1/drift/detect"
        payload = {
            "model_id": model_id,
            "reference_data": reference_data,
            "current_data": current_data
        }
        response = requests.post(url, json=payload)
        return response.json()
    
    def trigger_investigation(
        self,
        model_id: str,
        trigger_event: str = "drift_detected"
    ) -> Dict[str, Any]:
        """Trigger autonomous investigation"""
        url = f"{self.base_url}/api/v1/investigate/{model_id}"
        params = {"trigger_event": trigger_event}
        response = requests.post(url, params=params)
        return response.json()
    
    def get_investigations(self, model_id: str, limit: int = 50) -> Dict[str, Any]:
        """Get investigation reports"""
        url = f"{self.base_url}/api/v1/investigation/{model_id}"
        params = {"limit": limit}
        response = requests.get(url, params=params)
        return response.json()
    
    def trigger_retraining(
        self,
        model_id: str,
        trigger_reason: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """Trigger model retraining"""
        url = f"{self.base_url}/api/v1/retrain/{model_id}"
        payload = {
            "trigger_reason": trigger_reason,
            "force": force
        }
        response = requests.post(url, json=payload)
        return response.json()
    
    def verify_deployment(
        self,
        model_id: str,
        model_version: str,
        model_path: str,
        validation_data: list = None
    ) -> Dict[str, Any]:
        """Verify model deployment"""
        url = f"{self.base_url}/api/v1/verify"
        payload = {
            "model_id": model_id,
            "model_version": model_version,
            "model_path": model_path,
            "validation_data": validation_data
        }
        response = requests.post(url, json=payload)
        return response.json()
    
    def get_model_statistics(self, model_id: str) -> Dict[str, Any]:
        """Get model statistics"""
        url = f"{self.base_url}/api/v1/stats/{model_id}"
        response = requests.get(url)
        return response.json()


def main():
    """Demonstrate API usage"""
    
    print("=" * 60)
    print("Agentic Model Monitoring - API Usage Example")
    print("=" * 60)
    
    # Initialize client
    client = MonitoringAPIClient()
    
    # Step 1: Register a model
    print("\n1. Registering model via API...")
    response = client.register_model(
        model_id="churn_prediction_v1",
        model_name="Customer Churn Prediction",
        model_type="classification",
        model_path="models/churn_model.pkl",
        baseline_metrics={
            "accuracy": 0.89,
            "precision": 0.85,
            "recall": 0.82,
            "f1": 0.83
        },
        feature_names=["age", "tenure", "monthly_charges", "total_charges"],
        target_name="churn"
    )
    print(f"✓ {response}")
    
    # Step 2: Log predictions
    print("\n2. Logging predictions via API...")
    for i in range(5):
        response = client.log_prediction(
            model_id="churn_prediction_v1",
            features={
                "age": 35 + i,
                "tenure": 12 + i,
                "monthly_charges": 50.0 + i * 10,
                "total_charges": 600.0 + i * 100
            },
            prediction=0,
            actual_value=0,
            prediction_probability=0.75
        )
        print(f"  Logged prediction {i+1}: {response['log_id']}")
    
    # Step 3: Detect drift
    print("\n3. Detecting drift via API...")
    import numpy as np
    
    reference_data = [
        {"age": float(np.random.normal(40, 10)),
         "tenure": float(np.random.normal(24, 12)),
         "monthly_charges": float(np.random.normal(70, 20)),
         "total_charges": float(np.random.normal(1500, 500))}
        for _ in range(100)
    ]
    
    current_data = [
        {"age": float(np.random.normal(45, 12)),
         "tenure": float(np.random.normal(20, 10)),
         "monthly_charges": float(np.random.normal(80, 25)),
         "total_charges": float(np.random.normal(1600, 600))}
        for _ in range(100)
    ]
    
    response = client.detect_drift(
        model_id="churn_prediction_v1",
        reference_data=reference_data,
        current_data=current_data
    )
    print(f"✓ Drift detected: {response['drift_detected']}")
    print(f"  Severity: {response['severity']}")
    print(f"  Summary: {response['summary']}")
    
    # Step 4: Trigger investigation
    if response['drift_detected']:
        print("\n4. Triggering investigation via API...")
        response = client.trigger_investigation(
            model_id="churn_prediction_v1",
            trigger_event="data_drift_detected"
        )
        print(f"✓ {response}")
        
        # Wait for investigation to complete
        print("  Waiting for investigation to complete...")
        time.sleep(5)
        
        # Get investigation results
        print("\n5. Getting investigation results...")
        response = client.get_investigations("churn_prediction_v1")
        print(f"  Total investigations: {response['count']}")
        if response['investigations']:
            latest = response['investigations'][0]
            print(f"  Latest investigation:")
            print(f"    Summary: {latest['investigation_summary']}")
            print(f"    Confidence: {latest['confidence_score']:.2f}")
    
    # Step 6: Get statistics
    print("\n6. Getting model statistics...")
    response = client.get_model_statistics("churn_prediction_v1")
    print(f"✓ Statistics: {json.dumps(response, indent=2)}")
    
    print("\n" + "=" * 60)
    print("API example completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Note: This requires the API server to be running
    # Start it with: python -m api.main
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n⚠ Error: Could not connect to API server.")
        print("Please start the API server first:")
        print("  python -m api.main")
