"""
Basic Usage Example for Agentic Model Monitoring

This example demonstrates how to:
1. Register a model for monitoring
2. Log predictions
3. Detect drift
4. Trigger investigation
5. Handle retraining
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from storage.database import MonitoringDatabase, ModelMetadata
from drift_detection.data_drift import DataDriftDetector
from drift_detection.concept_drift import ConceptDriftDetector
from agent.investigation_agent import InvestigationAgent
from retraining.retraining_pipeline import RetrainingPipeline, RetrainingConfig


def main():
    """Demonstrate basic usage of the monitoring system"""
    
    # Initialize components
    db = MonitoringDatabase()
    data_drift_detector = DataDriftDetector()
    concept_drift_detector = ConceptDriftDetector()
    investigation_agent = InvestigationAgent()
    retraining_pipeline = RetrainingPipeline(RetrainingConfig(), db)
    
    print("=" * 60)
    print("Agentic Model Monitoring - Basic Usage Example")
    print("=" * 60)
    
    # Step 1: Register a model
    print("\n1. Registering a model for monitoring...")
    model_metadata = ModelMetadata(
        model_id="fraud_detection_v1",
        model_name="Fraud Detection Model",
        model_type="classification",
        model_path="models/fraud_model.pkl",
        created_at=datetime.utcnow().isoformat(),
        last_updated=datetime.utcnow().isoformat(),
        is_active=True,
        baseline_metrics={
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88,
            "f1": 0.90
        },
        feature_names=["transaction_amount", "merchant_category", "time_of_day", "location"],
        target_name="is_fraud"
    )
    
    db.register_model(model_metadata)
    print(f"✓ Model registered: {model_metadata.model_id}")
    
    # Step 2: Generate and log some predictions
    print("\n2. Logging predictions...")
    from storage.database import PredictionLog
    
    for i in range(10):
        prediction_log = PredictionLog(
            id=None,
            model_id="fraud_detection_v1",
            timestamp=datetime.utcnow().isoformat(),
            features={
                "transaction_amount": np.random.uniform(10, 1000),
                "merchant_category": np.random.choice(["retail", "food", "travel"]),
                "time_of_day": np.random.uniform(0, 24),
                "location": np.random.choice(["US", "UK", "CA"])
            },
            prediction=np.random.choice([0, 1]),
            actual_value=np.random.choice([0, 1]),
            prediction_probability=np.random.uniform(0.5, 0.99),
            metadata={"source": "api"}
        )
        db.log_prediction(prediction_log)
    
    print(f"✓ Logged 10 predictions")
    
    # Step 3: Detect data drift
    print("\n3. Detecting data drift...")
    
    # Create reference data (simulated training data)
    reference_data = pd.DataFrame({
        "transaction_amount": np.random.normal(100, 50, 1000),
        "merchant_category": np.random.choice(["retail", "food", "travel"], 1000),
        "time_of_day": np.random.uniform(0, 24, 1000),
        "location": np.random.choice(["US", "UK", "CA"], 1000)
    })
    
    # Create current data (simulated with some drift)
    current_data = pd.DataFrame({
        "transaction_amount": np.random.normal(150, 60, 1000),  # Shifted mean
        "merchant_category": np.random.choice(["retail", "food", "travel", "online"], 1000),  # New category
        "time_of_day": np.random.uniform(0, 24, 1000),
        "location": np.random.choice(["US", "UK", "CA", "AU"], 1000)  # New location
    })
    
    drift_report = data_drift_detector.detect_drift(
        reference_data=reference_data,
        current_data=current_data,
        model_id="fraud_detection_v1"
    )
    
    print(f"✓ Data drift detected: {drift_report.overall_drift_detected}")
    print(f"  Severity: {drift_report.overall_severity.value}")
    print(f"  Summary: {drift_report.summary}")
    
    # Step 4: Trigger investigation if drift detected
    if drift_report.overall_drift_detected:
        print("\n4. Triggering autonomous investigation...")
        
        from agent.investigation_agent import InvestigationContext
        
        context = InvestigationContext(
            model_id="fraud_detection_v1",
            model_type="classification",
            trigger_event="data_drift_detected",
            data_drift_report={
                "overall_drift_detected": drift_report.overall_drift_detected,
                "overall_severity": drift_report.overall_severity.value,
                "summary": drift_report.summary,
                "feature_results": [
                    {
                        "feature_name": r.feature_name,
                        "drift_detected": r.drift_detected,
                        "severity": r.severity.value
                    }
                    for r in drift_report.feature_results
                ]
            },
            concept_drift_report=None,
            performance_metrics={},
            baseline_metrics=model_metadata.baseline_metrics,
            recent_predictions=[],
            feature_importance=None,
            historical_context=None
        )
        
        investigation_result = investigation_agent.investigate(context)
        
        print(f"✓ Investigation completed")
        print(f"  Summary: {investigation_result.investigation_summary}")
        print(f"  Severity: {investigation_result.severity_assessment}")
        print(f"  Confidence: {investigation_result.confidence_score:.2f}")
        print(f"  Recommended actions:")
        for action in investigation_result.recommended_actions[:3]:
            print(f"    - {action}")
    
    # Step 5: Check if retraining is needed
    print("\n5. Checking retraining requirements...")
    
    should_retrain, reason = retraining_pipeline.should_retrain(
        model_id="fraud_detection_v1",
        drift_result={
            "drift_detected": drift_report.overall_drift_detected,
            "overall_severity": drift_report.overall_severity.value
        }
    )
    
    print(f"  Should retrain: {should_retrain}")
    print(f"  Reason: {reason}")
    
    # Step 6: Get model statistics
    print("\n6. Getting model statistics...")
    stats = db.get_model_statistics("fraud_detection_v1")
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Drift detections: {stats['drift_detections']}")
    print(f"  Investigations: {stats['investigations']}")
    print(f"  Retrainings: {stats['retrainings']}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
