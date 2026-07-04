"""
Core Monitoring Engine

Orchestrates the entire monitoring workflow:
- Continuous monitoring of models
- Drift detection scheduling
- Investigation triggering
- Retraining coordination
- Deployment verification
"""

import time
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
import logging

from storage.database import MonitoringDatabase, ModelMetadata
from drift_detection.data_drift import DataDriftDetector
from drift_detection.concept_drift import ConceptDriftDetector
from agent.investigation_agent import InvestigationAgent, InvestigationContext
from retraining.retraining_pipeline import RetrainingPipeline, RetrainingConfig
from deployment.verification import DeploymentVerifier, VerificationConfig


@dataclass
class MonitoringConfig:
    """Configuration for the monitoring engine"""
    check_interval_seconds: int = 300  # 5 minutes
    enable_data_drift: bool = True
    enable_concept_drift: bool = True
    enable_auto_investigation: bool = True
    enable_auto_retraining: bool = False  # Disabled by default for safety
    drift_threshold: float = 0.05
    performance_threshold: float = 0.1


class ModelMonitor:
    """Main monitoring engine for a single model"""
    
    def __init__(
        self,
        model_id: str,
        db: MonitoringDatabase,
        config: MonitoringConfig,
        data_drift_detector: DataDriftDetector,
        concept_drift_detector: ConceptDriftDetector,
        investigation_agent: InvestigationAgent,
        retraining_pipeline: RetrainingPipeline,
        deployment_verifier: DeploymentVerifier
    ):
        self.model_id = model_id
        self.db = db
        self.config = config
        self.data_drift_detector = data_drift_detector
        self.concept_drift_detector = concept_drift_detector
        self.investigation_agent = investigation_agent
        self.retraining_pipeline = retraining_pipeline
        self.deployment_verifier = deployment_verifier
        
        self.logger = logging.getLogger(f"ModelMonitor.{model_id}")
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start monitoring this model"""
        if self.is_running:
            self.logger.warning(f"Monitoring already running for {self.model_id}")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"Started monitoring for {self.model_id}")
    
    def stop(self):
        """Stop monitoring this model"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        self.logger.info(f"Stopped monitoring for {self.model_id}")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                self._run_monitoring_cycle()
            except Exception as e:
                self.logger.error(f"Error in monitoring cycle: {e}")
            
            # Wait for next cycle
            time.sleep(self.config.check_interval_seconds)
    
    def _run_monitoring_cycle(self):
        """Run a single monitoring cycle"""
        self.logger.debug(f"Running monitoring cycle for {self.model_id}")
        
        # Get model metadata
        model = self.db.get_model(self.model_id)
        if not model or not model.is_active:
            self.logger.warning(f"Model {self.model_id} not found or inactive")
            return
        
        # Get recent predictions
        predictions_data = self.db.get_predictions_for_analysis(
            self.model_id,
            sample_size=1000
        )
        
        if not predictions_data['features']:
            self.logger.debug(f"No predictions to analyze for {self.model_id}")
            return
        
        # Convert to DataFrame
        import pandas as pd
        current_data = pd.DataFrame(predictions_data['features'])
        
        # Get reference data (baseline)
        reference_data = self._get_reference_data(model)
        
        if reference_data is None:
            self.logger.warning(f"No reference data available for {self.model_id}")
            return
        
        # Detect data drift
        if self.config.enable_data_drift:
            self._check_data_drift(reference_data, current_data, model)
        
        # Detect concept drift
        if self.config.enable_concept_drift and predictions_data['actuals']:
            self._check_concept_drift(predictions_data, model)
    
    def _get_reference_data(self, model: ModelMetadata):
        """Get reference data for drift detection"""
        # In a real implementation, this would load from a file or database
        # For now, return None to indicate it needs to be set up
        return None
    
    def _check_data_drift(self, reference_data, current_data, model: ModelMetadata):
        """Check for data drift"""
        try:
            drift_report = self.data_drift_detector.detect_drift(
                reference_data=reference_data,
                current_data=current_data,
                model_id=self.model_id
            )
            
            # Save to database
            from storage.database import DriftDetectionResult
            drift_result = DriftDetectionResult(
                id=None,
                model_id=self.model_id,
                timestamp=drift_report.timestamp,
                drift_type="data_drift",
                drift_detected=drift_report.overall_drift_detected,
                severity=drift_report.overall_severity.value,
                metrics={
                    "feature_results": [
                        {
                            "feature_name": r.feature_name,
                            "drift_detected": r.drift_detected,
                            "severity": r.severity.value
                        }
                        for r in drift_report.feature_results
                    ]
                },
                summary=drift_report.summary,
                recommendations=drift_report.recommendations
            )
            self.db.save_drift_result(drift_result)
            
            # Log drift detection
            if drift_report.overall_drift_detected:
                self.logger.warning(
                    f"Data drift detected for {self.model_id}: "
                    f"{drift_report.overall_severity.value} - {drift_report.summary}"
                )
                
                # Trigger investigation if enabled
                if self.config.enable_auto_investigation:
                    self._trigger_investigation("data_drift_detected", drift_report)
                
                # Trigger retraining if enabled and drift is severe
                if self.config.enable_auto_retraining:
                    if drift_report.overall_severity.value in ["high", "critical"]:
                        self._trigger_retraining(
                            f"Data drift detected ({drift_report.overall_severity.value})",
                            drift_report
                        )
            
        except Exception as e:
            self.logger.error(f"Error checking data drift: {e}")
    
    def _check_concept_drift(self, predictions_data: Dict, model: ModelMetadata):
        """Check for concept drift"""
        try:
            import numpy as np
            
            predictions = np.array(predictions_data['predictions'])
            actuals = np.array(predictions_data['actuals'])
            
            # Set baseline if not set
            if self.model_id not in self.concept_drift_detector.baseline_performance:
                self.concept_drift_detector.set_baseline(
                    self.model_id,
                    model.baseline_metrics
                )
            
            # Detect concept drift
            drift_result = self.concept_drift_detector.detect_drift(
                model_id=self.model_id,
                predictions=predictions,
                actuals=actuals
            )
            
            # Log concept drift detection
            if drift_result.drift_detected:
                self.logger.warning(
                    f"Concept drift detected for {self.model_id}: "
                    f"{drift_result.drift_type} - {drift_result.description}"
                )
                
                # Trigger investigation if enabled
                if self.config.enable_auto_investigation:
                    self._trigger_investigation("concept_drift_detected", drift_result)
                
                # Trigger retraining if enabled
                if self.config.enable_auto_retraining:
                    if drift_result.severity in ["high", "critical"]:
                        self._trigger_retraining(
                            f"Concept drift detected ({drift_result.severity})",
                            drift_result
                        )
            
        except Exception as e:
            self.logger.error(f"Error checking concept drift: {e}")
    
    def _trigger_investigation(self, trigger_event: str, drift_data):
        """Trigger autonomous investigation"""
        try:
            context = InvestigationContext(
                model_id=self.model_id,
                model_type="unknown",
                trigger_event=trigger_event,
                data_drift_report=drift_data.__dict__ if hasattr(drift_data, '__dict__') else drift_data,
                concept_drift_report=None,
                performance_metrics={},
                baseline_metrics={},
                recent_predictions=[],
                feature_importance=None,
                historical_context=None
            )
            
            result = self.investigation_agent.investigate(context)
            
            # Save to database
            from storage.database import InvestigationReport
            report = InvestigationReport(
                id=None,
                model_id=self.model_id,
                timestamp=result.timestamp,
                trigger_event=trigger_event,
                investigation_summary=result.investigation_summary,
                root_cause_analysis=result.root_cause_analysis,
                recommended_actions=result.recommended_actions,
                confidence_score=result.confidence_score,
                llm_reasoning=result.llm_reasoning
            )
            self.db.save_investigation(report)
            
            self.logger.info(f"Investigation completed for {self.model_id}: {result.investigation_summary}")
            
        except Exception as e:
            self.logger.error(f"Error triggering investigation: {e}")
    
    def _trigger_retraining(self, reason: str, drift_data):
        """Trigger model retraining"""
        try:
            should_retrain, trigger_reason = self.retraining_pipeline.should_retrain(
                self.model_id,
                drift_data.__dict__ if hasattr(drift_data, '__dict__') else drift_data
            )
            
            if should_retrain:
                job_id = self.retraining_pipeline.trigger_retraining(
                    model_id=self.model_id,
                    trigger_reason=reason
                )
                self.logger.info(f"Retraining triggered for {self.model_id}: job_id={job_id}")
            
        except Exception as e:
            self.logger.error(f"Error triggering retraining: {e}")


class MonitoringOrchestrator:
    """Orchestrates monitoring for multiple models"""
    
    def __init__(
        self,
        db: MonitoringDatabase,
        config: MonitoringConfig = None
    ):
        self.db = db
        self.config = config or MonitoringConfig()
        
        # Initialize components
        self.data_drift_detector = DataDriftDetector()
        self.concept_drift_detector = ConceptDriftDetector()
        self.investigation_agent = InvestigationAgent()
        self.retraining_pipeline = RetrainingPipeline(RetrainingConfig(), db)
        self.deployment_verifier = DeploymentVerifier(VerificationConfig(), db)
        
        # Model monitors
        self.monitors: Dict[str, ModelMonitor] = {}
        
        self.logger = logging.getLogger("MonitoringOrchestrator")
    
    def register_model(self, model_id: str):
        """Register a model for monitoring"""
        if model_id in self.monitors:
            self.logger.warning(f"Model {model_id} already registered")
            return
        
        monitor = ModelMonitor(
            model_id=model_id,
            db=self.db,
            config=self.config,
            data_drift_detector=self.data_drift_detector,
            concept_drift_detector=self.concept_drift_detector,
            investigation_agent=self.investigation_agent,
            retraining_pipeline=self.retraining_pipeline,
            deployment_verifier=self.deployment_verifier
        )
        
        self.monitors[model_id] = monitor
        monitor.start()
        self.logger.info(f"Registered model {model_id} for monitoring")
    
    def unregister_model(self, model_id: str):
        """Unregister a model from monitoring"""
        if model_id in self.monitors:
            self.monitors[model_id].stop()
            del self.monitors[model_id]
            self.logger.info(f"Unregistered model {model_id} from monitoring")
    
    def start_all(self):
        """Start monitoring for all active models"""
        models = self.db.list_models(active_only=True)
        for model in models:
            self.register_model(model.model_id)
        
        self.logger.info(f"Started monitoring for {len(models)} models")
    
    def stop_all(self):
        """Stop monitoring for all models"""
        for model_id in list(self.monitors.keys()):
            self.unregister_model(model_id)
        
        self.logger.info("Stopped all monitoring")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all monitors"""
        return {
            "total_models": len(self.monitors),
            "running_models": sum(1 for m in self.monitors.values() if m.is_running),
            "models": [
                {
                    "model_id": model_id,
                    "is_running": monitor.is_running
                }
                for model_id, monitor in self.monitors.items()
            ]
        }
