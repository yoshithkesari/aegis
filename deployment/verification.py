"""
Deployment Verification System

Validates that newly deployed models:
- Perform correctly on validation data
- Meet performance thresholds
- Have consistent predictions
- Are properly integrated
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import joblib


@dataclass
class VerificationConfig:
    """Configuration for deployment verification"""
    sample_size: int = 1000
    timeout_seconds: int = 300
    performance_threshold: float = 0.05
    consistency_threshold: float = 0.95
    latency_threshold_ms: float = 100.0
    error_rate_threshold: float = 0.01


@dataclass
class VerificationResult:
    """Result of deployment verification"""
    model_id: str
    model_version: str
    timestamp: str
    verification_passed: bool
    performance_metrics: Dict[str, float]
    consistency_score: float
    latency_ms: float
    error_rate: float
    checks: List[Dict[str, Any]]
    summary: str
    recommendations: List[str]


class DeploymentVerifier:
    """Verifies model deployments"""
    
    def __init__(
        self,
        config: VerificationConfig,
        storage_client: Any,
        model_loader: Optional[Callable] = None
    ):
        self.config = config
        self.storage = storage_client
        self.model_loader = model_loader
    
    def verify_deployment(
        self,
        model_id: str,
        model_version: str,
        model_path: str,
        validation_data: Optional[pd.DataFrame] = None
    ) -> VerificationResult:
        """
        Verify that a model deployment is successful
        
        Args:
            model_id: Model identifier
            model_version: Version of the model
            model_path: Path to the model file
            validation_data: Optional validation data for testing
            
        Returns:
            VerificationResult with detailed verification status
        """
        checks = []
        
        # Check 1: Model can be loaded
        load_check = self._check_model_loading(model_path)
        checks.append(load_check)
        
        if not load_check['passed']:
            return self._create_failure_result(
                model_id, model_version, checks, "Model loading failed"
            )
        
        # Load the model
        model = self._load_model(model_path)
        
        # Check 2: Model can make predictions
        prediction_check = self._check_model_prediction(model, validation_data)
        checks.append(prediction_check)
        
        if not prediction_check['passed']:
            return self._create_failure_result(
                model_id, model_version, checks, "Model prediction failed"
            )
        
        # Check 3: Performance meets threshold
        performance_check = self._check_performance(model, validation_data, model_id)
        checks.append(performance_check)
        
        # Check 4: Prediction consistency
        consistency_check = self._check_prediction_consistency(model, validation_data)
        checks.append(consistency_check)
        
        # Check 5: Latency
        latency_check = self._check_latency(model, validation_data)
        checks.append(latency_check)
        
        # Check 6: Error rate
        error_rate_check = self._check_error_rate(model, validation_data)
        checks.append(error_rate_check)
        
        # Overall verdict
        all_passed = all(check['passed'] for check in checks)
        
        # Extract metrics
        performance_metrics = performance_check.get('metrics', {})
        consistency_score = consistency_check.get('score', 0.0)
        latency_ms = latency_check.get('latency_ms', 0.0)
        error_rate = error_rate_check.get('error_rate', 0.0)
        
        # Generate summary and recommendations
        summary = self._generate_summary(checks, all_passed)
        recommendations = self._generate_recommendations(checks)
        
        return VerificationResult(
            model_id=model_id,
            model_version=model_version,
            timestamp=datetime.utcnow().isoformat(),
            verification_passed=all_passed,
            performance_metrics=performance_metrics,
            consistency_score=consistency_score,
            latency_ms=latency_ms,
            error_rate=error_rate,
            checks=checks,
            summary=summary,
            recommendations=recommendations
        )
    
    def _check_model_loading(self, model_path: str) -> Dict[str, Any]:
        """Check if model can be loaded"""
        try:
            if not Path(model_path).exists():
                return {
                    'name': 'model_loading',
                    'passed': False,
                    'message': f"Model file not found at {model_path}"
                }
            
            model = self._load_model(model_path)
            
            return {
                'name': 'model_loading',
                'passed': True,
                'message': 'Model loaded successfully'
            }
        except Exception as e:
            return {
                'name': 'model_loading',
                'passed': False,
                'message': f"Failed to load model: {str(e)}"
            }
    
    def _check_model_prediction(
        self,
        model: Any,
        validation_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Check if model can make predictions"""
        try:
            if validation_data is None or len(validation_data) == 0:
                return {
                    'name': 'model_prediction',
                    'passed': False,
                    'message': 'No validation data provided'
                }
            
            # Get feature columns (exclude target if present)
            model_metadata = self.storage.get_model(validation_data.columns[0] if hasattr(self.storage, 'get_model') else None)
            
            # Try to make a prediction
            sample = validation_data.iloc[:1]
            prediction = model.predict(sample)
            
            return {
                'name': 'model_prediction',
                'passed': True,
                'message': 'Model can make predictions',
                'sample_prediction': prediction[0] if len(prediction) > 0 else None
            }
        except Exception as e:
            return {
                'name': 'model_prediction',
                'passed': False,
                'message': f"Prediction failed: {str(e)}"
            }
    
    def _check_performance(
        self,
        model: Any,
        validation_data: Optional[pd.DataFrame],
        model_id: str
    ) -> Dict[str, Any]:
        """Check if model performance meets threshold"""
        try:
            if validation_data is None or len(validation_data) == 0:
                return {
                    'name': 'performance',
                    'passed': False,
                    'message': 'No validation data for performance check'
                }
            
            # Get baseline metrics
            model_metadata = self.storage.get_model(model_id)
            baseline_metrics = model_metadata.baseline_metrics if model_metadata else {}
            
            # Sample data for performance check
            sample_size = min(len(validation_data), self.config.sample_size)
            sample = validation_data.iloc[:sample_size]
            
            # Separate features and target
            if model_metadata:
                feature_cols = model_metadata.feature_names
                target_col = model_metadata.target_name
                
                X = sample[feature_cols]
                y = sample[target_col]
                
                # Make predictions
                predictions = model.predict(X)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y, predictions)
                
                # Compare with baseline
                performance_degradation = 0.0
                if baseline_metrics:
                    performance_degradation = self._calculate_degradation(
                        baseline_metrics, metrics
                    )
                
                passed = performance_degradation <= self.config.performance_threshold
                
                return {
                    'name': 'performance',
                    'passed': passed,
                    'message': f"Performance degradation: {performance_degradation:.4f}",
                    'metrics': metrics,
                    'baseline_metrics': baseline_metrics,
                    'degradation': performance_degradation
                }
            else:
                return {
                    'name': 'performance',
                    'passed': True,
                    'message': 'No baseline available, skipping performance check',
                    'metrics': {}
                }
        except Exception as e:
            return {
                'name': 'performance',
                'passed': False,
                'message': f"Performance check failed: {str(e)}"
            }
    
    def _check_prediction_consistency(
        self,
        model: Any,
        validation_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Check if model predictions are consistent"""
        try:
            if validation_data is None or len(validation_data) < 10:
                return {
                    'name': 'consistency',
                    'passed': True,
                    'message': 'Insufficient data for consistency check',
                    'score': 1.0
                }
            
            # Sample data
            sample = validation_data.iloc[:min(100, len(validation_data))]
            
            # Make predictions multiple times
            predictions_1 = model.predict(sample)
            predictions_2 = model.predict(sample)
            
            # Calculate consistency
            if len(predictions_1) == len(predictions_2):
                consistency = np.mean(predictions_1 == predictions_2)
                
                # For regression, use correlation
                if predictions_1.dtype == float:
                    consistency = np.corrcoef(predictions_1, predictions_2)[0, 1]
                
                passed = consistency >= self.config.consistency_threshold
                
                return {
                    'name': 'consistency',
                    'passed': passed,
                    'message': f"Consistency score: {consistency:.4f}",
                    'score': consistency
                }
            else:
                return {
                    'name': 'consistency',
                    'passed': False,
                    'message': 'Prediction length mismatch',
                    'score': 0.0
                }
        except Exception as e:
            return {
                'name': 'consistency',
                'passed': False,
                'message': f"Consistency check failed: {str(e)}",
                'score': 0.0
            }
    
    def _check_latency(
        self,
        model: Any,
        validation_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Check if model prediction latency is acceptable"""
        try:
            if validation_data is None or len(validation_data) == 0:
                return {
                    'name': 'latency',
                    'passed': True,
                    'message': 'No data for latency check',
                    'latency_ms': 0.0
                }
            
            sample = validation_data.iloc[:min(100, len(validation_data))]
            
            # Measure latency
            start_time = datetime.utcnow()
            predictions = model.predict(sample)
            end_time = datetime.utcnow()
            
            latency_ms = (end_time - start_time).total_seconds() * 1000 / len(sample)
            
            passed = latency_ms <= self.config.latency_threshold_ms
            
            return {
                'name': 'latency',
                'passed': passed,
                'message': f"Average latency: {latency_ms:.2f}ms",
                'latency_ms': latency_ms
            }
        except Exception as e:
            return {
                'name': 'latency',
                'passed': False,
                'message': f"Latency check failed: {str(e)}",
                'latency_ms': float('inf')
            }
    
    def _check_error_rate(
        self,
        model: Any,
        validation_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Check if model error rate is acceptable"""
        try:
            if validation_data is None or len(validation_data) == 0:
                return {
                    'name': 'error_rate',
                    'passed': True,
                    'message': 'No data for error rate check',
                    'error_rate': 0.0
                }
            
            sample = validation_data.iloc[:min(100, len(validation_data))]
            
            # Count errors
            errors = 0
            total = len(sample)
            
            for i in range(total):
                try:
                    model.predict(sample.iloc[i:i+1])
                except:
                    errors += 1
            
            error_rate = errors / total if total > 0 else 0.0
            passed = error_rate <= self.config.error_rate_threshold
            
            return {
                'name': 'error_rate',
                'passed': passed,
                'message': f"Error rate: {error_rate:.4f}",
                'error_rate': error_rate
            }
        except Exception as e:
            return {
                'name': 'error_rate',
                'passed': False,
                'message': f"Error rate check failed: {str(e)}",
                'error_rate': 1.0
            }
    
    def _load_model(self, model_path: str) -> Any:
        """Load model from disk"""
        if self.model_loader:
            return self.model_loader(model_path)
        else:
            return joblib.load(model_path)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
        metrics = {}
        
        # Determine task type
        if y_true.dtype == 'object' or len(np.unique(y_true)) < 20:
            # Classification
            metrics['accuracy'] = np.mean(y_true == y_pred)
        else:
            # Regression
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
            metrics['r2'] = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        
        return metrics
    
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
    
    def _create_failure_result(
        self,
        model_id: str,
        model_version: str,
        checks: List[Dict[str, Any]],
        reason: str
    ) -> VerificationResult:
        """Create a failure verification result"""
        return VerificationResult(
            model_id=model_id,
            model_version=model_version,
            timestamp=datetime.utcnow().isoformat(),
            verification_passed=False,
            performance_metrics={},
            consistency_score=0.0,
            latency_ms=float('inf'),
            error_rate=1.0,
            checks=checks,
            summary=f"Deployment verification failed: {reason}",
            recommendations=["Review model file", "Check model compatibility", "Verify model training"]
        )
    
    def _generate_summary(self, checks: List[Dict[str, Any]], all_passed: bool) -> str:
        """Generate verification summary"""
        if all_passed:
            return "All verification checks passed. Deployment is successful."
        
        failed_checks = [c for c in checks if not c['passed']]
        return f"Deployment verification failed. Failed checks: {', '.join([c['name'] for c in failed_checks])}"
    
    def _generate_recommendations(self, checks: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on failed checks"""
        recommendations = []
        
        for check in checks:
            if not check['passed']:
                if check['name'] == 'model_loading':
                    recommendations.append("Verify model file exists and is not corrupted")
                    recommendations.append("Check model file permissions")
                elif check['name'] == 'model_prediction':
                    recommendations.append("Verify model input format matches expected features")
                    recommendations.append("Check model compatibility with current environment")
                elif check['name'] == 'performance':
                    recommendations.append("Consider retraining with more recent data")
                    recommendations.append("Review feature engineering pipeline")
                elif check['name'] == 'consistency':
                    recommendations.append("Check for randomness in model predictions")
                    recommendations.append("Verify model is deterministic if required")
                elif check['name'] == 'latency':
                    recommendations.append("Consider model optimization or quantization")
                    recommendations.append("Review infrastructure resources")
                elif check['name'] == 'error_rate':
                    recommendations.append("Review model error handling")
                    recommendations.append("Check input data quality")
        
        return recommendations


class CanaryDeployment:
    """Manages canary deployments for gradual rollout"""
    
    def __init__(self, storage_client: Any):
        self.storage = storage_client
        self.canary_configs: Dict[str, Dict] = {}
    
    def setup_canary(
        self,
        model_id: str,
        new_model_version: str,
        canary_percentage: float = 0.1,
        duration_hours: int = 24
    ):
        """Setup canary deployment configuration"""
        self.canary_configs[model_id] = {
            'new_model_version': new_model_version,
            'canary_percentage': canary_percentage,
            'start_time': datetime.utcnow().isoformat(),
            'end_time': (datetime.utcnow() + timedelta(hours=duration_hours)).isoformat(),
            'traffic_split': canary_percentage,
            'status': 'active'
        }
    
    def get_canary_model(self, model_id: str) -> Optional[str]:
        """Get which model version to use for a request"""
        config = self.canary_configs.get(model_id)
        
        if not config or config['status'] != 'active':
            return None
        
        # Check if canary period has ended
        if datetime.utcnow() >= datetime.fromisoformat(config['end_time']):
            config['status'] = 'completed'
            return None
        
        # Randomly assign to canary based on percentage
        import random
        if random.random() < config['canary_percentage']:
            return config['new_model_version']
        
        return None
    
    def update_canary_percentage(self, model_id: str, new_percentage: float):
        """Update canary traffic percentage"""
        if model_id in self.canary_configs:
            self.canary_configs[model_id]['canary_percentage'] = new_percentage
    
    def get_canary_status(self, model_id: str) -> Optional[Dict]:
        """Get canary deployment status"""
        return self.canary_configs.get(model_id)
    
    def promote_canary(self, model_id: str):
        """Promote canary model to full production"""
        if model_id in self.canary_configs:
            config = self.canary_configs[model_id]
            # Update model to use new version
            self.storage.update_model(
                model_id,
                model_path=config['new_model_version']
            )
            config['status'] = 'promoted'


class RollbackManager:
    """Manages model rollbacks when deployments fail"""
    
    def __init__(self, storage_client: Any):
        self.storage = storage_client
        self.rollback_history: Dict[str, List[Dict]] = {}
    
    def create_checkpoint(self, model_id: str, model_path: str) -> str:
        """Create a checkpoint before deployment"""
        checkpoint_id = f"{model_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        if model_id not in self.rollback_history:
            self.rollback_history[model_id] = []
        
        self.rollback_history[model_id].append({
            'checkpoint_id': checkpoint_id,
            'model_path': model_path,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return checkpoint_id
    
    def rollback(self, model_id: str, checkpoint_id: Optional[str] = None) -> bool:
        """Rollback to a previous model version"""
        if model_id not in self.rollback_history:
            return False
        
        history = self.rollback_history[model_id]
        
        if checkpoint_id:
            # Rollback to specific checkpoint
            checkpoint = next((c for c in history if c['checkpoint_id'] == checkpoint_id), None)
            if checkpoint:
                self.storage.update_model(model_id, model_path=checkpoint['model_path'])
                return True
        else:
            # Rollback to most recent checkpoint
            if history:
                most_recent = history[-1]
                self.storage.update_model(model_id, model_path=most_recent['model_path'])
                return True
        
        return False
    
    def get_rollback_history(self, model_id: str) -> List[Dict]:
        """Get rollback history for a model"""
        return self.rollback_history.get(model_id, [])
