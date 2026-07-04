"""
Advanced Usage Example for Agentic Model Monitoring

This example demonstrates advanced features:
- Custom model training functions
- Scheduled retraining
- Canary deployments
- Rollback management
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from storage.database import MonitoringDatabase, ModelMetadata
from drift_detection.data_drift import DataDriftDetector
from drift_detection.concept_drift import ConceptDriftDetector
from agent.investigation_agent import InvestigationAgent
from retraining.retraining_pipeline import (
    RetrainingPipeline, 
    RetrainingConfig,
    RetrainingScheduler,
    IncrementalRetrainer
)
from deployment.verification import (
    DeploymentVerifier,
    VerificationConfig,
    CanaryDeployment,
    RollbackManager
)


def custom_model_trainer(train_data: dict) -> object:
    """Custom model training function"""
    from sklearn.ensemble import RandomForestClassifier
    
    X_train = train_data['X']
    y_train = train_data['y']
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def custom_model_loader(model_path: str) -> object:
    """Custom model loader function"""
    import joblib
    return joblib.load(model_path)


def main():
    """Demonstrate advanced usage"""
    
    print("=" * 60)
    print("Agentic Model Monitoring - Advanced Usage Example")
    print("=" * 60)
    
    # Initialize components with custom functions
    db = MonitoringDatabase()
    config = RetrainingConfig(
        max_iterations=10,
        early_stop_patience=5,
        validation_split=0.3
    )
    
    retraining_pipeline = RetrainingPipeline(
        config=config,
        storage_client=db,
        model_trainer=custom_model_trainer,
        model_loader=custom_model_loader
    )
    
    scheduler = RetrainingScheduler(retraining_pipeline)
    incremental_retrainer = IncrementalRetrainer(config)
    deployment_verifier = DeploymentVerifier(VerificationConfig(), db, custom_model_loader)
    canary_deployment = CanaryDeployment(db)
    rollback_manager = RollbackManager(db)
    
    # Step 1: Register model with advanced configuration
    print("\n1. Registering model with advanced configuration...")
    model_metadata = ModelMetadata(
        model_id="credit_scoring_v2",
        model_name="Credit Scoring Model v2",
        model_type="classification",
        model_path="models/credit_model_v2.pkl",
        created_at=datetime.utcnow().isoformat(),
        last_updated=datetime.utcnow().isoformat(),
        is_active=True,
        baseline_metrics={
            "accuracy": 0.92,
            "precision": 0.90,
            "recall": 0.85,
            "f1": 0.87,
            "auc_roc": 0.93
        },
        feature_names=["income", "debt_ratio", "credit_history", "employment_length"],
        target_name="default_risk"
    )
    
    db.register_model(model_metadata)
    print(f"✓ Model registered: {model_metadata.model_id}")
    
    # Step 2: Setup scheduled retraining
    print("\n2. Setting up scheduled retraining...")
    scheduler.schedule_retraining(
        model_id="credit_scoring_v2",
        schedule_type="performance_based",
        performance_threshold=0.05  # Retrain if performance degrades by 5%
    )
    print("✓ Scheduled retraining configured")
    
    # Step 3: Setup incremental learning
    print("\n3. Setting up incremental learning...")
    for i in range(100):
        incremental_retrainer.add_sample(
            model_id="credit_scoring_v2",
            features={
                "income": float(np.random.normal(50000, 15000)),
                "debt_ratio": float(np.random.uniform(0.1, 0.8)),
                "credit_history": float(np.random.randint(300, 850)),
                "employment_length": float(np.random.randint(1, 20))
            },
            target=int(np.random.choice([0, 1], p=[0.9, 0.1]))
        )
    print(f"✓ Added 100 samples to incremental buffer")
    
    # Check if incremental training should be triggered
    should_train = incremental_retrainer.should_incremental_train("credit_scoring_v2")
    print(f"  Should trigger incremental training: {should_train}")
    
    # Step 4: Setup canary deployment
    print("\n4. Setting up canary deployment...")
    canary_deployment.setup_canary(
        model_id="credit_scoring_v2",
        new_model_version="models/credit_model_v3.pkl",
        canary_percentage=0.1,  # 10% traffic to new version
        duration_hours=24
    )
    print("✓ Canary deployment configured")
    
    # Simulate traffic routing
    print("\n  Simulating traffic routing...")
    for i in range(10):
        canary_model = canary_deployment.get_canary_model("credit_scoring_v2")
        if canary_model:
            print(f"  Request {i+1}: Routed to canary model {canary_model}")
        else:
            print(f"  Request {i+1}: Routed to production model")
    
    # Step 5: Create rollback checkpoint
    print("\n5. Creating rollback checkpoint...")
    checkpoint_id = rollback_manager.create_checkpoint(
        model_id="credit_scoring_v2",
        model_path="models/credit_model_v2.pkl"
    )
    print(f"✓ Checkpoint created: {checkpoint_id}")
    
    # Step 6: Simulate performance check and scheduled retraining
    print("\n6. Simulating performance check for scheduled retraining...")
    current_metrics = {
        "accuracy": 0.87,  # Degraded from 0.92
        "precision": 0.85,
        "recall": 0.80,
        "f1": 0.82
    }
    
    triggered = scheduler.check_and_trigger("credit_scoring_v2", current_metrics)
    print(f"  Retraining triggered: {triggered}")
    
    # Step 7: Verify deployment
    print("\n7. Verifying deployment...")
    
    # Create validation data
    validation_data = pd.DataFrame({
        "income": np.random.normal(50000, 15000, 100),
        "debt_ratio": np.random.uniform(0.1, 0.8, 100),
        "credit_history": np.random.randint(300, 850, 100),
        "employment_length": np.random.randint(1, 20, 100),
        "default_risk": np.random.choice([0, 1], 100, p=[0.9, 0.1])
    })
    
    verification_result = deployment_verifier.verify_deployment(
        model_id="credit_scoring_v2",
        model_version="credit_model_v2",
        model_path="models/credit_model_v2.pkl",
        validation_data=validation_data
    )
    
    print(f"✓ Verification passed: {verification_result.verification_passed}")
    print(f"  Summary: {verification_result.summary}")
    
    if not verification_result.verification_passed:
        print("\n8. Rolling back due to failed verification...")
        rollback_success = rollback_manager.rollback("credit_scoring_v2")
        print(f"  Rollback successful: {rollback_success}")
    
    # Step 9: Promote canary if successful
    print("\n9. Promoting canary deployment...")
    canary_status = canary_deployment.get_canary_status("credit_scoring_v2")
    if canary_status and canary_status['status'] == 'active':
        # In a real scenario, you'd check canary performance first
        print("  (In production, would check canary performance before promoting)")
        # canary_deployment.promote_canary("credit_scoring_v2")
        print("  Canary promotion skipped in demo")
    
    # Step 10: Get rollback history
    print("\n10. Getting rollback history...")
    history = rollback_manager.get_rollback_history("credit_scoring_v2")
    print(f"  Total checkpoints: {len(history)}")
    for checkpoint in history:
        print(f"    - {checkpoint['checkpoint_id']}: {checkpoint['timestamp']}")
    
    print("\n" + "=" * 60)
    print("Advanced example completed successfully!")
    print("=" * 60)
    print("\nKey features demonstrated:")
    print("  ✓ Custom model training and loading functions")
    print("  ✓ Scheduled performance-based retraining")
    print("  ✓ Incremental learning with data buffering")
    print("  ✓ Canary deployment with traffic splitting")
    print("  ✓ Rollback management with checkpoints")
    print("  ✓ Deployment verification")
    print("=" * 60)


if __name__ == "__main__":
    main()
