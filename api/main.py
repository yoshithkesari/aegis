"""
FastAPI REST API for Model Monitoring System

Provides endpoints for:
- Model registration and management
- Prediction logging
- Drift detection queries
- Investigation reports
- Retraining triggers
- Deployment verification
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.database import (
    MonitoringDatabase,
    ModelMetadata,
    PredictionLog,
    DriftDetectionResult,
    InvestigationReport,
    RetrainingRecord
)
from drift_detection.data_drift import DataDriftDetector
from drift_detection.concept_drift import ConceptDriftDetector
from agent.investigation_agent import InvestigationAgent, InvestigationContext
from retraining.retraining_pipeline import RetrainingPipeline, RetrainingConfig
from deployment.verification import DeploymentVerifier, VerificationConfig


app = FastAPI(
    title="Agentic Model Monitoring API",
    description="Autonomous AI operations engineer for model monitoring and drift detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (in production, use dependency injection)
db = MonitoringDatabase()
data_drift_detector = DataDriftDetector()
concept_drift_detector = ConceptDriftDetector()
investigation_agent = InvestigationAgent()
retraining_pipeline = RetrainingPipeline(RetrainingConfig(), db)
deployment_verifier = DeploymentVerifier(VerificationConfig(), db)


# Pydantic models for API
class ModelRegistration(BaseModel):
    model_id: str
    model_name: str
    model_type: str
    model_path: str
    baseline_metrics: Dict[str, float]
    feature_names: List[str]
    target_name: str


class PredictionRequest(BaseModel):
    model_id: str
    features: Dict[str, Any]
    prediction: Any
    actual_value: Optional[Any] = None
    prediction_probability: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class DriftDetectionRequest(BaseModel):
    model_id: str
    reference_data: List[Dict[str, Any]]
    current_data: List[Dict[str, Any]]


class RetrainingRequest(BaseModel):
    model_id: str
    trigger_reason: str
    force: bool = False


class VerificationRequest(BaseModel):
    model_id: str
    model_version: str
    model_path: str
    validation_data: Optional[List[Dict[str, Any]]] = None


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "agentic-model-monitoring"
    }


# Model management endpoints
@app.post("/api/v1/models")
async def register_model(model: ModelRegistration):
    """Register a new model for monitoring"""
    metadata = ModelMetadata(
        model_id=model.model_id,
        model_name=model.model_name,
        model_type=model.model_type,
        model_path=model.model_path,
        created_at=datetime.utcnow().isoformat(),
        last_updated=datetime.utcnow().isoformat(),
        is_active=True,
        baseline_metrics=model.baseline_metrics,
        feature_names=model.feature_names,
        target_name=model.target_name
    )
    
    success = db.register_model(metadata)
    
    if not success:
        raise HTTPException(status_code=400, detail="Model already exists")
    
    return {"status": "success", "model_id": model.model_id}


@app.get("/api/v1/models")
async def list_models(active_only: bool = True):
    """List all monitored models"""
    models = db.list_models(active_only=active_only)
    return {
        "models": [
            {
                "model_id": m.model_id,
                "model_name": m.model_name,
                "model_type": m.model_type,
                "created_at": m.created_at,
                "is_active": m.is_active
            }
            for m in models
        ]
    }


@app.get("/api/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get model details"""
    model = db.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "model_id": model.model_id,
        "model_name": model.model_name,
        "model_type": model.model_type,
        "model_path": model.model_path,
        "created_at": model.created_at,
        "last_updated": model.last_updated,
        "is_active": model.is_active,
        "baseline_metrics": model.baseline_metrics,
        "feature_names": model.feature_names,
        "target_name": model.target_name
    }


@app.delete("/api/v1/models/{model_id}")
async def deactivate_model(model_id: str):
    """Deactivate a model from monitoring"""
    success = db.update_model(model_id, is_active=False)
    
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {"status": "success", "message": f"Model {model_id} deactivated"}


# Prediction logging endpoints
@app.post("/api/v1/predictions")
async def log_prediction(prediction: PredictionRequest):
    """Log a prediction for monitoring"""
    log = PredictionLog(
        id=None,
        model_id=prediction.model_id,
        timestamp=datetime.utcnow().isoformat(),
        features=prediction.features,
        prediction=prediction.prediction,
        actual_value=prediction.actual_value,
        prediction_probability=prediction.prediction_probability,
        metadata=prediction.metadata
    )
    
    log_id = db.log_prediction(log)
    
    return {"status": "success", "log_id": log_id}


@app.get("/api/v1/predictions/{model_id}")
async def get_predictions(
    model_id: str,
    limit: int = 1000,
    offset: int = 0,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    """Get predictions for a model"""
    predictions = db.get_predictions(
        model_id,
        limit=limit,
        offset=offset,
        start_time=start_time,
        end_time=end_time
    )
    
    return {
        "model_id": model_id,
        "count": len(predictions),
        "predictions": [
            {
                "id": p.id,
                "timestamp": p.timestamp,
                "features": p.features,
                "prediction": p.prediction,
                "actual_value": p.actual_value,
                "prediction_probability": p.prediction_probability
            }
            for p in predictions
        ]
    }


# Drift detection endpoints
@app.post("/api/v1/drift/detect")
async def detect_drift(request: DriftDetectionRequest):
    """Detect data drift between reference and current data"""
    import pandas as pd
    
    reference_df = pd.DataFrame(request.reference_data)
    current_df = pd.DataFrame(request.current_data)
    
    # Detect data drift
    drift_report = data_drift_detector.detect_drift(
        reference_data=reference_df,
        current_data=current_df,
        model_id=request.model_id
    )
    
    # Save to database
    drift_result = DriftDetectionResult(
        id=None,
        model_id=request.model_id,
        timestamp=drift_report.timestamp,
        drift_type="data_drift",
        drift_detected=drift_report.overall_drift_detected,
        severity=drift_report.overall_severity.value,
        metrics={
            "feature_results": [
                {
                    "feature_name": r.feature_name,
                    "drift_detected": r.drift_detected,
                    "severity": r.severity.value,
                    "statistic": r.statistic,
                    "p_value": r.p_value
                }
                for r in drift_report.feature_results
            ]
        },
        summary=drift_report.summary,
        recommendations=drift_report.recommendations
    )
    
    db.save_drift_result(drift_result)
    
    return {
        "model_id": request.model_id,
        "drift_detected": drift_report.overall_drift_detected,
        "severity": drift_report.overall_severity.value,
        "summary": drift_report.summary,
        "recommendations": drift_report.recommendations,
        "feature_results": [
            {
                "feature_name": r.feature_name,
                "drift_detected": r.drift_detected,
                "severity": r.severity.value,
                "statistic": r.statistic,
                "p_value": r.p_value,
                "description": r.description
            }
            for r in drift_report.feature_results
        ]
    }


@app.get("/api/v1/drift/{model_id}")
async def get_drift_history(model_id: str, limit: int = 100):
    """Get drift detection history for a model"""
    drift_results = db.get_drift_results(model_id, limit=limit)
    
    return {
        "model_id": model_id,
        "count": len(drift_results),
        "results": [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "drift_type": r.drift_type,
                "drift_detected": r.drift_detected,
                "severity": r.severity,
                "summary": r.summary,
                "recommendations": r.recommendations
            }
            for r in drift_results
        ]
    }


# Investigation endpoints
@app.post("/api/v1/investigate/{model_id}")
async def trigger_investigation(
    model_id: str,
    background_tasks: BackgroundTasks,
    trigger_event: str = "drift_detected"
):
    """Trigger autonomous investigation of a model"""
    
    def run_investigation():
        # Get recent drift results
        drift_results = db.get_drift_results(model_id, limit=5)
        
        # Get model metadata
        model = db.get_model(model_id)
        
        # Build investigation context
        context = InvestigationContext(
            model_id=model_id,
            model_type=model.model_type if model else "unknown",
            trigger_event=trigger_event,
            data_drift_report=drift_results[0].__dict__ if drift_results else None,
            concept_drift_report=None,
            performance_metrics={},
            baseline_metrics=model.baseline_metrics if model else {},
            recent_predictions=[],
            feature_importance=None,
            historical_context=None
        )
        
        # Run investigation
        result = investigation_agent.investigate(context)
        
        # Save to database
        report = InvestigationReport(
            id=None,
            model_id=model_id,
            timestamp=result.timestamp,
            trigger_event=trigger_event,
            investigation_summary=result.investigation_summary,
            root_cause_analysis=result.root_cause_analysis,
            recommended_actions=result.recommended_actions,
            confidence_score=result.confidence_score,
            llm_reasoning=result.llm_reasoning
        )
        
        db.save_investigation(report)
    
    background_tasks.add_task(run_investigation)
    
    return {"status": "investigation_started", "model_id": model_id}


@app.get("/api/v1/investigation/{model_id}")
async def get_investigations(model_id: str, limit: int = 50):
    """Get investigation reports for a model"""
    investigations = db.get_investigations(model_id, limit=limit)
    
    return {
        "model_id": model_id,
        "count": len(investigations),
        "investigations": [
            {
                "id": i.id,
                "timestamp": i.timestamp,
                "trigger_event": i.trigger_event,
                "investigation_summary": i.investigation_summary,
                "root_cause_analysis": i.root_cause_analysis,
                "recommended_actions": i.recommended_actions,
                "confidence_score": i.confidence_score
            }
            for i in investigations
        ]
    }


# Retraining endpoints
@app.post("/api/v1/retrain/{model_id}")
async def trigger_retraining(
    model_id: str,
    request: RetrainingRequest,
    background_tasks: BackgroundTasks
):
    """Trigger model retraining"""
    
    def run_retraining():
        retraining_pipeline.trigger_retraining(
            model_id=model_id,
            trigger_reason=request.trigger_reason
        )
    
    background_tasks.add_task(run_retraining)
    
    return {"status": "retraining_started", "model_id": model_id}


@app.get("/api/v1/retraining/{model_id}")
async def get_retraining_history(model_id: str, limit: int = 50):
    """Get retraining history for a model"""
    records = db.get_retraining_history(model_id, limit=limit)
    
    return {
        "model_id": model_id,
        "count": len(records),
        "records": [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "trigger_reason": r.trigger_reason,
                "old_version": r.old_version,
                "new_version": r.new_version,
                "status": r.status,
                "duration_seconds": r.duration_seconds,
                "training_metrics": r.training_metrics,
                "validation_metrics": r.validation_metrics
            }
            for r in records
        ]
    }


# Deployment verification endpoints
@app.post("/api/v1/verify")
async def verify_deployment(request: VerificationRequest):
    """Verify a model deployment"""
    import pandas as pd
    
    validation_data = None
    if request.validation_data:
        validation_data = pd.DataFrame(request.validation_data)
    
    result = deployment_verifier.verify_deployment(
        model_id=request.model_id,
        model_version=request.model_version,
        model_path=request.model_path,
        validation_data=validation_data
    )
    
    return {
        "model_id": result.model_id,
        "model_version": result.model_version,
        "verification_passed": result.verification_passed,
        "performance_metrics": result.performance_metrics,
        "consistency_score": result.consistency_score,
        "latency_ms": result.latency_ms,
        "error_rate": result.error_rate,
        "summary": result.summary,
        "recommendations": result.recommendations,
        "checks": result.checks
    }


# Statistics endpoints
@app.get("/api/v1/stats/{model_id}")
async def get_model_statistics(model_id: str):
    """Get statistics for a model"""
    stats = db.get_model_statistics(model_id)
    
    return {
        "model_id": model_id,
        **stats
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
