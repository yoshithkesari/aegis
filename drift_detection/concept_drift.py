"""
Concept Drift Detection Module

Detects changes in the relationship between input features and target variables:
- Performance-based drift detection
- Prediction distribution changes
- Error rate monitoring
- Statistical process control
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class ConceptDriftType(Enum):
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    INCREMENTAL = "incremental"
    RECURRING = "recurring"


@dataclass
class ConceptDriftResult:
    """Result of concept drift detection"""
    model_id: str
    drift_detected: bool
    drift_type: Optional[ConceptDriftType]
    severity: str
    performance_degradation: float
    error_rate_increase: float
    confidence: float
    timestamp: str
    metrics: Dict[str, float]
    description: str


class ConceptDriftDetector:
    """Detects concept drift using performance monitoring and statistical methods"""
    
    def __init__(
        self,
        performance_threshold: float = 0.1,
        error_rate_threshold: float = 0.05,
        window_size: int = 100,
        min_samples: int = 50
    ):
        self.performance_threshold = performance_threshold
        self.error_rate_threshold = error_rate_threshold
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Historical performance tracking
        self.performance_history: Dict[str, List[Dict]] = {}
        self.baseline_performance: Dict[str, Dict] = {}
    
    def set_baseline(
        self,
        model_id: str,
        metrics: Dict[str, float],
        sample_size: int = 1000
    ):
        """Set baseline performance for a model"""
        self.baseline_performance[model_id] = {
            'metrics': metrics,
            'sample_size': sample_size,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.performance_history[model_id] = []
    
    def detect_drift(
        self,
        model_id: str,
        predictions: np.ndarray,
        actuals: np.ndarray,
        features: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> ConceptDriftResult:
        """
        Detect concept drift by analyzing performance changes
        
        Args:
            model_id: Identifier for the model
            predictions: Model predictions
            actuals: Ground truth values
            features: Input features (optional, for feature importance analysis)
            metrics: Computed performance metrics (optional, will be computed if not provided)
            
        Returns:
            ConceptDriftResult with detailed drift analysis
        """
        if model_id not in self.baseline_performance:
            raise ValueError(f"No baseline set for model {model_id}. Call set_baseline first.")
        
        # Compute metrics if not provided
        if metrics is None:
            metrics = self._compute_metrics(predictions, actuals)
        
        # Calculate performance degradation
        baseline_metrics = self.baseline_performance[model_id]['metrics']
        performance_degradation = self._calculate_degradation(
            baseline_metrics, metrics
        )
        
        # Calculate error rate
        error_rate = self._calculate_error_rate(predictions, actuals)
        baseline_error_rate = self._calculate_error_rate(
            np.array(baseline_metrics.get('predictions', [])),
            np.array(baseline_metrics.get('actuals', []))
        ) if 'predictions' in baseline_metrics else 0.0
        
        error_rate_increase = error_rate - baseline_error_rate
        
        # Store in history
        self.performance_history[model_id].append({
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics,
            'error_rate': error_rate,
            'performance_degradation': performance_degradation
        })
        
        # Keep only recent history
        if len(self.performance_history[model_id]) > self.window_size:
            self.performance_history[model_id] = self.performance_history[model_id][-self.window_size:]
        
        # Detect drift using multiple methods
        drift_detected, drift_type, severity, confidence = self._analyze_drift(
            model_id, performance_degradation, error_rate_increase
        )
        
        # Generate description
        description = self._generate_description(
            drift_detected, drift_type, severity, performance_degradation, error_rate_increase
        )
        
        return ConceptDriftResult(
            model_id=model_id,
            drift_detected=drift_detected,
            drift_type=drift_type,
            severity=severity,
            performance_degradation=performance_degradation,
            error_rate_increase=error_rate_increase,
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat(),
            metrics=metrics,
            description=description
        )
    
    def _compute_metrics(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
        """Compute standard performance metrics"""
        metrics = {}
        
        # Accuracy (for classification)
        if len(predictions.shape) == 1 or predictions.shape[1] == 1:
            pred_classes = (predictions > 0.5).astype(int) if predictions.dtype == float else predictions
            metrics['accuracy'] = np.mean(pred_classes == actuals)
        
        # MSE (for regression)
        metrics['mse'] = np.mean((predictions - actuals) ** 2)
        
        # MAE (for regression)
        metrics['mae'] = np.mean(np.abs(predictions - actuals))
        
        # R-squared (for regression)
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Store raw predictions and actuals for error rate calculation
        metrics['predictions'] = predictions.tolist() if len(predictions) < 1000 else predictions[:1000].tolist()
        metrics['actuals'] = actuals.tolist() if len(actuals) < 1000 else actuals[:1000].tolist()
        
        return metrics
    
    def _calculate_degradation(
        self,
        baseline: Dict[str, float],
        current: Dict[str, float]
    ) -> float:
        """Calculate overall performance degradation"""
        degradations = []
        
        # For accuracy (higher is better)
        if 'accuracy' in baseline and 'accuracy' in current:
            degradations.append(baseline['accuracy'] - current['accuracy'])
        
        # For MSE/MAE (lower is better)
        for metric in ['mse', 'mae']:
            if metric in baseline and metric in current:
                if baseline[metric] > 0:
                    degradations.append((current[metric] - baseline[metric]) / baseline[metric])
        
        # For R-squared (higher is better)
        if 'r2' in baseline and 'r2' in current:
            degradations.append(baseline['r2'] - current['r2'])
        
        return np.mean(degradations) if degradations else 0.0
    
    def _calculate_error_rate(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate error rate"""
        if len(predictions.shape) == 1 or predictions.shape[1] == 1:
            pred_classes = (predictions > 0.5).astype(int) if predictions.dtype == float else predictions
            return 1.0 - np.mean(pred_classes == actuals)
        else:
            # For multi-class or regression, use relative error
            return np.mean(np.abs(predictions - actuals) / (np.abs(actuals) + 1e-10))
    
    def _analyze_drift(
        self,
        model_id: str,
        performance_degradation: float,
        error_rate_increase: float
    ) -> Tuple[bool, Optional[ConceptDriftType], str, float]:
        """Analyze drift using statistical process control"""
        history = self.performance_history[model_id]
        
        if len(history) < self.min_samples:
            return False, None, "insufficient_data", 0.0
        
        # Extract recent performance degradations
        recent_degradations = [h['performance_degradation'] for h in history[-self.min_samples:]]
        recent_error_rates = [h['error_rate'] for h in history[-self.min_samples:]]
        
        # Statistical tests
        drift_detected = False
        drift_type = None
        severity = "none"
        confidence = 0.0
        
        # Test 1: Performance degradation exceeds threshold
        if performance_degradation > self.performance_threshold:
            drift_detected = True
            confidence = min(0.95, 0.5 + performance_degradation * 2)
        
        # Test 2: Error rate increase exceeds threshold
        if error_rate_increase > self.error_rate_threshold:
            drift_detected = True
            confidence = max(confidence, min(0.95, 0.5 + error_rate_increase * 10))
        
        # Test 3: Statistical significance of change (t-test)
        if len(recent_degradations) >= 10:
            early_degradations = recent_degradations[:len(recent_degradations)//2]
            late_degradations = recent_degradations[len(recent_degradations)//2:]
            
            if len(early_degradations) > 0 and len(late_degradations) > 0:
                t_stat, p_value = stats.ttest_ind(early_degradations, late_degradations)
                if p_value < 0.05:
                    drift_detected = True
                    confidence = max(confidence, 1.0 - p_value)
        
        # Determine drift type
        if drift_detected:
            drift_type = self._classify_drift_type(recent_degradations)
            severity = self._classify_severity(performance_degradation, error_rate_increase)
        
        return drift_detected, drift_type, severity, confidence
    
    def _classify_drift_type(self, degradations: List[float]) -> ConceptDriftType:
        """Classify the type of concept drift"""
        if len(degradations) < 5:
            return ConceptDriftType.SUDDEN
        
        # Calculate trend
        recent = degradations[-5:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        # Calculate variance
        variance = np.var(degradations)
        
        if variance > 0.1:
            return ConceptDriftType.RECURRING
        elif abs(trend) > 0.01:
            return ConceptDriftType.GRADUAL
        elif degradations[-1] > 0.2:
            return ConceptDriftType.SUDDEN
        else:
            return ConceptDriftType.INCREMENTAL
    
    def _classify_severity(self, performance_degradation: float, error_rate_increase: float) -> str:
        """Classify drift severity"""
        if performance_degradation > 0.3 or error_rate_increase > 0.2:
            return "critical"
        elif performance_degradation > 0.2 or error_rate_increase > 0.1:
            return "high"
        elif performance_degradation > 0.1 or error_rate_increase > 0.05:
            return "medium"
        else:
            return "low"
    
    def _generate_description(
        self,
        drift_detected: bool,
        drift_type: Optional[ConceptDriftType],
        severity: str,
        performance_degradation: float,
        error_rate_increase: float
    ) -> str:
        """Generate human-readable description"""
        if not drift_detected:
            return f"No concept drift detected. Performance degradation: {performance_degradation:.4f}, Error rate increase: {error_rate_increase:.4f}"
        
        type_str = drift_type.value if drift_type else "unknown"
        return (
            f"Concept drift detected ({type_str}, {severity} severity). "
            f"Performance degradation: {performance_degradation:.4f}, "
            f"Error rate increase: {error_rate_increase:.4f}"
        )
    
    def get_performance_trend(self, model_id: str) -> Dict:
        """Get performance trend analysis for a model"""
        if model_id not in self.performance_history:
            return {}
        
        history = self.performance_history[model_id]
        
        if len(history) < 2:
            return {'history': history}
        
        # Calculate trends
        degradations = [h['performance_degradation'] for h in history]
        error_rates = [h['error_rate'] for h in history]
        
        return {
            'history': history,
            'trend': {
                'degradation_trend': np.polyfit(range(len(degradations)), degradations, 1)[0],
                'error_rate_trend': np.polyfit(range(len(error_rates)), error_rates, 1)[0],
                'avg_degradation': np.mean(degradations),
                'avg_error_rate': np.mean(error_rates),
                'max_degradation': max(degradations),
                'max_error_rate': max(error_rates)
            }
        }


class PredictionDistributionMonitor:
    """Monitor changes in prediction distributions"""
    
    def __init__(self, bins: int = 50, threshold: float = 0.1):
        self.bins = bins
        self.threshold = threshold
        self.reference_distribution: Optional[Dict] = None
    
    def set_reference(self, predictions: np.ndarray):
        """Set reference prediction distribution"""
        hist, bin_edges = np.histogram(predictions, bins=self.bins)
        self.reference_distribution = {
            'hist': hist,
            'bin_edges': bin_edges,
            'total': len(predictions)
        }
    
    def detect_distribution_shift(self, predictions: np.ndarray) -> Dict:
        """Detect shift in prediction distribution"""
        if self.reference_distribution is None:
            raise ValueError("Reference distribution not set. Call set_reference first.")
        
        # Calculate current distribution
        hist, _ = np.histogram(predictions, bins=self.reference_distribution['bin_edges'])
        
        # Normalize
        ref_hist = self.reference_distribution['hist'] / self.reference_distribution['total']
        curr_hist = hist / len(predictions)
        
        # Calculate KL divergence
        kl_divergence = self._calculate_kl_divergence(ref_hist, curr_hist)
        
        # Calculate Earth Mover's Distance (Wasserstein)
        from scipy.stats import wasserstein_distance
        emd = wasserstein_distance(
            self.reference_distribution['bin_edges'][:-1],
            self.reference_distribution['bin_edges'][:-1],
            ref_hist,
            curr_hist
        )
        
        shift_detected = kl_divergence > self.threshold or emd > self.threshold
        
        return {
            'shift_detected': shift_detected,
            'kl_divergence': kl_divergence,
            'earth_mover_distance': emd,
            'threshold': self.threshold,
            'reference_hist': ref_hist.tolist(),
            'current_hist': curr_hist.tolist()
        }
    
    def _calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Kullback-Leibler divergence"""
        # Add small epsilon to avoid division by zero
        p = p + 1e-10
        q = q + 1e-10
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        return np.sum(p * np.log(p / q))
