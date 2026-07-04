"""
Automatic Retraining Pipeline

Handles:
- Triggering retraining based on drift detection
- Managing retraining jobs
- Tracking retraining progress
- Validating retrained models
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class RetrainingConfig:
    """Configuration for retraining"""
    max_iterations: int = 5
    early_stop_patience: int = 3
    validation_split: float = 0.2
    min_improvement: float = 0.01
    max_training_time: int = 3600  # seconds
    data_path: str = "data/"
    model_save_path: str = "models/"


@dataclass
class RetrainingJob:
    """Represents a retraining job"""
    job_id: str
    model_id: str
    trigger_reason: str
    status: str  # pending, running, completed, failed
    start_time: str
    end_time: Optional[str]
    duration_seconds: Optional[float]
    old_model_path: str
    new_model_path: str
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    improvement: float
    error_message: Optional[str]


class RetrainingPipeline:
    """Automatic retraining pipeline"""
    
    def __init__(
        self,
        config: RetrainingConfig,
        storage_client: Any,
        model_loader: Optional[Callable] = None,
        model_trainer: Optional[Callable] = None
    ):
        self.config = config
        self.storage = storage_client
        self.model_loader = model_loader
        self.model_trainer = model_trainer
        
        # Ensure directories exist
        Path(config.data_path).mkdir(parents=True, exist_ok=True)
        Path(config.model_save_path).mkdir(parents=True, exist_ok=True)
        
        # Active jobs
        self.active_jobs: Dict[str, RetrainingJob] = {}
    
    def should_retrain(
        self,
        model_id: str,
        drift_result: Dict[str, Any],
        investigation_result: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, str]:
        """
        Determine if retraining should be triggered
        
        Returns:
            (should_retrain, reason)
        """
        reasons = []
        
        # Check drift severity
        severity = drift_result.get("overall_severity", "none")
        if severity in ["high", "critical"]:
            reasons.append(f"High severity drift detected ({severity})")
        
        # Check if drift is detected
        if drift_result.get("drift_detected", False):
            reasons.append("Drift detected in model performance")
        
        # Check investigation recommendations
        if investigation_result:
            recommended_actions = investigation_result.get("recommended_actions", [])
            for action in recommended_actions:
                if "retrain" in action.lower():
                    reasons.append("Investigation recommends retraining")
                    break
        
        # Check severity assessment
        if investigation_result:
            severity_assessment = investigation_result.get("severity_assessment", "low")
            if severity_assessment in ["high", "critical"]:
                reasons.append(f"Investigation severity: {severity_assessment}")
        
        should_retrain = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "No retraining trigger conditions met"
        
        return should_retrain, reason
    
    def trigger_retraining(
        self,
        model_id: str,
        trigger_reason: str,
        investigation_result: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Trigger retraining for a model
        
        Returns:
            job_id for tracking the retraining job
        """
        # Get model metadata
        model_metadata = self.storage.get_model(model_id)
        if not model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        # Generate job ID
        job_id = f"{model_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Create job
        job = RetrainingJob(
            job_id=job_id,
            model_id=model_id,
            trigger_reason=trigger_reason,
            status="pending",
            start_time=datetime.utcnow().isoformat(),
            end_time=None,
            duration_seconds=None,
            old_model_path=model_metadata.model_path,
            new_model_path="",
            training_metrics={},
            validation_metrics={},
            improvement=0.0,
            error_message=None
        )
        
        self.active_jobs[job_id] = job
        
        # Start retraining in background
        self._execute_retraining(job, investigation_result)
        
        return job_id
    
    def _execute_retraining(
        self,
        job: RetrainingJob,
        investigation_result: Optional[Dict[str, Any]] = None
    ):
        """Execute retraining job"""
        job.status = "running"
        
        try:
            start_time = time.time()
            
            # Load training data
            training_data = self._load_training_data(job.model_id)
            
            # Load old model for comparison
            old_model = self._load_model(job.old_model_path)
            old_metrics = self._evaluate_model(old_model, training_data['validation'])
            
            # Train new model
            if self.model_trainer:
                new_model = self.model_trainer(training_data['train'])
            else:
                new_model = self._default_trainer(training_data['train'], job.model_id)
            
            # Evaluate new model
            new_metrics = self._evaluate_model(new_model, training_data['validation'])
            
            # Calculate improvement
            improvement = self._calculate_improvement(old_metrics, new_metrics)
            
            # Save new model
            new_model_path = self._save_model(new_model, job.model_id, job.job_id)
            job.new_model_path = new_model_path
            
            # Update job with results
            job.training_metrics = new_metrics
            job.validation_metrics = new_metrics
            job.improvement = improvement
            job.status = "completed"
            
            # Update storage
            self._update_model_after_retraining(job.model_id, new_model_path, new_metrics)
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
        
        finally:
            job.end_time = datetime.utcnow().isoformat()
            job.duration_seconds = time.time() - start_time
            
            # Save to storage
            self._save_retraining_record(job)
    
    def _load_training_data(self, model_id: str) -> Dict[str, Any]:
        """Load training data for retraining"""
        # Get model metadata to understand data structure
        model_metadata = self.storage.get_model(model_id)
        
        # Try to load from file
        data_path = os.path.join(self.config.data_path, f"{model_id}_training.csv")
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            # Fallback: use recent predictions from storage
            predictions_data = self.storage.get_predictions_for_analysis(
                model_id, 
                sample_size=10000
            )
            
            if not predictions_data['features']:
                raise ValueError("No training data available")
            
            # Convert to DataFrame
            df = pd.DataFrame(predictions_data['features'])
            
            # Add target if available
            if predictions_data['actuals']:
                df[model_metadata.target_name] = predictions_data['actuals']
        
        # Split features and target
        target_name = model_metadata.target_name
        feature_names = model_metadata.feature_names
        
        X = df[feature_names]
        y = df[target_name]
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.config.validation_split,
            random_state=42
        )
        
        return {
            'train': {'X': X_train, 'y': y_train},
            'validation': {'X': X_val, 'y': y_val},
            'feature_names': feature_names,
            'target_name': target_name
        }
    
    def _load_model(self, model_path: str) -> Any:
        """Load model from disk"""
        if self.model_loader:
            return self.model_loader(model_path)
        else:
            return joblib.load(model_path)
    
    def _default_trainer(self, train_data: Dict[str, Any], model_id: str) -> Any:
        """Default model trainer using scikit-learn"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available for default training")
        
        X_train = train_data['X']
        y_train = train_data['y']
        
        # Determine if classification or regression
        if y_train.dtype == 'object' or len(y_train.unique()) < 20:
            # Classification
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        else:
            # Regression
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        
        model.fit(X_train, y_train)
        return model
    
    def _evaluate_model(self, model: Any, validation_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model on validation data"""
        X_val = validation_data['X']
        y_val = validation_data['y']
        
        predictions = model.predict(X_val)
        
        metrics = {}
        
        # Determine task type
        if y_val.dtype == 'object' or len(y_val.unique()) < 20:
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y_val, predictions)
            try:
                metrics['f1'] = f1_score(y_val, predictions, average='weighted')
            except:
                pass
        else:
            # Regression metrics
            metrics['mse'] = mean_squared_error(y_val, predictions)
            metrics['mae'] = np.mean(np.abs(predictions - y_val))
            metrics['r2'] = r2_score(y_val, predictions)
        
        return metrics
    
    def _calculate_improvement(
        self,
        old_metrics: Dict[str, float],
        new_metrics: Dict[str, float]
    ) -> float:
        """Calculate improvement between old and new metrics"""
        improvements = []
        
        for key in old_metrics:
            if key in new_metrics:
                old_val = old_metrics[key]
                new_val = new_metrics[key]
                
                # For metrics where higher is better (accuracy, r2, f1)
                if key in ['accuracy', 'r2', 'f1']:
                    if old_val > 0:
                        improvements.append((new_val - old_val) / old_val)
                
                # For metrics where lower is better (mse, mae)
                elif key in ['mse', 'mae']:
                    if old_val > 0:
                        improvements.append((old_val - new_val) / old_val)
        
        return np.mean(improvements) if improvements else 0.0
    
    def _save_model(self, model: Any, model_id: str, job_id: str) -> str:
        """Save trained model to disk"""
        filename = f"{model_id}_{job_id}.pkl"
        filepath = os.path.join(self.config.model_save_path, filename)
        
        joblib.dump(model, filepath)
        return filepath
    
    def _update_model_after_retraining(
        self,
        model_id: str,
        new_model_path: str,
        new_metrics: Dict[str, float]
    ):
        """Update model metadata after successful retraining"""
        self.storage.update_model(
            model_id,
            model_path=new_model_path,
            last_updated=datetime.utcnow().isoformat(),
            baseline_metrics=new_metrics
        )
    
    def _save_retraining_record(self, job: RetrainingJob):
        """Save retraining record to storage"""
        from storage.database import RetrainingRecord
        
        record = RetrainingRecord(
            id=None,
            model_id=job.model_id,
            timestamp=job.start_time,
            trigger_reason=job.trigger_reason,
            old_version=job.old_model_path,
            new_version=job.new_model_path,
            training_metrics=job.training_metrics,
            validation_metrics=job.validation_metrics,
            status=job.status,
            duration_seconds=job.duration_seconds,
            data_used={}
        )
        
        self.storage.save_retraining_record(record)
    
    def get_job_status(self, job_id: str) -> Optional[RetrainingJob]:
        """Get status of a retraining job"""
        return self.active_jobs.get(job_id)
    
    def list_jobs(self, model_id: Optional[str] = None) -> List[RetrainingJob]:
        """List retraining jobs"""
        jobs = list(self.active_jobs.values())
        
        if model_id:
            jobs = [j for j in jobs if j.model_id == model_id]
        
        return jobs


class IncrementalRetrainer:
    """Handles incremental/online learning for models that support it"""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.buffer_size = 1000
        self.data_buffers: Dict[str, List[Dict]] = {}
    
    def add_sample(self, model_id: str, features: Dict[str, Any], target: Any):
        """Add a new sample to the buffer for incremental learning"""
        if model_id not in self.data_buffers:
            self.data_buffers[model_id] = []
        
        self.data_buffers[model_id].append({
            'features': features,
            'target': target,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Trim buffer if too large
        if len(self.data_buffers[model_id]) > self.buffer_size:
            self.data_buffers[model_id] = self.data_buffers[model_id][-self.buffer_size:]
    
    def should_incremental_train(self, model_id: str, threshold: int = 500) -> bool:
        """Check if enough samples accumulated for incremental training"""
        return len(self.data_buffers.get(model_id, [])) >= threshold
    
    def get_incremental_data(self, model_id: str) -> List[Dict]:
        """Get buffered data for incremental training"""
        return self.data_buffers.get(model_id, [])
    
    def clear_buffer(self, model_id: str):
        """Clear the buffer after training"""
        self.data_buffers[model_id] = []


class RetrainingScheduler:
    """Schedules and manages retraining jobs"""
    
    def __init__(self, pipeline: RetrainingPipeline):
        self.pipeline = pipeline
        self.schedule: Dict[str, Dict] = {}
    
    def schedule_retraining(
        self,
        model_id: str,
        schedule_type: str = "manual",
        interval_hours: Optional[int] = None,
        performance_threshold: Optional[float] = None
    ):
        """Schedule retraining for a model"""
        self.schedule[model_id] = {
            'type': schedule_type,
            'interval_hours': interval_hours,
            'performance_threshold': performance_threshold,
            'last_retraining': None,
            'next_retraining': None
        }
    
    def check_and_trigger(self, model_id: str, current_metrics: Dict[str, float]) -> bool:
        """Check if scheduled retraining should be triggered"""
        if model_id not in self.schedule:
            return False
        
        schedule_config = self.schedule[model_id]
        
        # Check performance-based schedule
        if schedule_config['performance_threshold']:
            # Compare with baseline
            model_metadata = self.pipeline.storage.get_model(model_id)
            if model_metadata:
                baseline = model_metadata.baseline_metrics
                degradation = self._calculate_degradation(baseline, current_metrics)
                
                if degradation > schedule_config['performance_threshold']:
                    self.pipeline.trigger_retraining(
                        model_id,
                        f"Performance degradation {degradation:.2f} exceeded threshold {schedule_config['performance_threshold']}"
                    )
                    return True
        
        # Check time-based schedule
        if schedule_config['interval_hours']:
            if schedule_config['last_retraining']:
                last_time = datetime.fromisoformat(schedule_config['last_retraining'])
                next_time = last_time + timedelta(hours=schedule_config['interval_hours'])
                
                if datetime.utcnow() >= next_time:
                    self.pipeline.trigger_retraining(
                        model_id,
                        f"Scheduled retraining (interval: {schedule_config['interval_hours']} hours)"
                    )
                    return True
        
        return False
    
    def _calculate_degradation(
        self,
        baseline: Dict[str, float],
        current: Dict[str, float]
    ) -> float:
        """Calculate performance degradation"""
        degradations = []
        
        for key in baseline:
            if key in current:
                if key in ['accuracy', 'r2', 'f1']:
                    degradations.append(baseline[key] - current[key])
                else:
                    degradations.append(current[key] - baseline[key])
        
        return sum(degradations) / len(degradations) if degradations else 0.0
