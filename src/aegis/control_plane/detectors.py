"""
Drift Detectors - Evidently for batch drift, River for streaming drift
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DriftSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftDetectionResult:
    """Result of drift detection"""
    drift_detected: bool
    severity: DriftSeverity
    drift_type: str
    affected_features: List[str]
    metrics: Dict[str, float]
    summary: str


class EvidentlyDetector:
    """Batch drift detection using Evidently"""
    
    def __init__(self):
        self.reference_data: Optional[pd.DataFrame] = None
    
    def set_reference(self, data: pd.DataFrame):
        """Set reference data for drift comparison"""
        self.reference_data = data
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.05
    ) -> DriftDetectionResult:
        """Detect drift using statistical tests"""
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        affected_features = []
        feature_drift_scores = {}
        
        # Check each feature for drift
        for column in self.reference_data.columns:
            if column not in current_data.columns:
                continue
            
            ref_col = self.reference_data[column].dropna()
            curr_col = current_data[column].dropna()
            
            if len(ref_col) == 0 or len(curr_col) == 0:
                continue
            
            # Numerical feature - use KS test
            if pd.api.types.is_numeric_dtype(ref_col):
                from scipy.stats import ks_2samp
                statistic, p_value = ks_2samp(ref_col, curr_col)
                
                if p_value < threshold:
                    affected_features.append(column)
                    feature_drift_scores[column] = p_value
            
            # Categorical feature - use chi-squared
            else:
                try:
                    from scipy.stats import chi2_contingency
                    
                    # Create contingency table
                    all_categories = set(ref_col.unique()).union(set(curr_col.unique()))
                    ref_counts = [ref_col.value_counts().get(cat, 0) for cat in all_categories]
                    curr_counts = [curr_col.value_counts().get(cat, 0) for cat in all_categories]
                    
                    contingency = np.array([ref_counts, curr_counts])
                    chi2, p_value, _, _ = chi2_contingency(contingency)
                    
                    if p_value < threshold:
                        affected_features.append(column)
                        feature_drift_scores[column] = p_value
                except:
                    pass
        
        # Determine overall severity
        drift_detected = len(affected_features) > 0
        severity = self._calculate_severity(affected_features, threshold)
        
        return DriftDetectionResult(
            drift_detected=drift_detected,
            severity=severity,
            drift_type="data_drift",
            affected_features=affected_features,
            metrics=feature_drift_scores,
            summary=self._generate_summary(drift_detected, affected_features, severity)
        )
    
    def _calculate_severity(
        self,
        affected_features: List[str],
        threshold: float
    ) -> DriftSeverity:
        """Calculate drift severity based on number of affected features"""
        n_affected = len(affected_features)
        
        if n_affected == 0:
            return DriftSeverity.NONE
        elif n_affected == 1:
            return DriftSeverity.LOW
        elif n_affected <= 3:
            return DriftSeverity.MEDIUM
        elif n_affected <= 5:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
    
    def _generate_summary(
        self,
        drift_detected: bool,
        affected_features: List[str],
        severity: DriftSeverity
    ) -> str:
        """Generate human-readable summary"""
        if not drift_detected:
            return "No drift detected"
        
        return f"Drift detected in {len(affected_features)} features: {', '.join(affected_features[:5])}. Severity: {severity.value}"


class RiverDetector:
    """Streaming drift detection using River"""
    
    def __init__(self):
        self.detectors: Dict[str, any] = {}
        self.drift_count: Dict[str, int] = {}
    
    def add_detector(self, feature: str, detector_type: str = "ADWIN"):
        """Add a streaming detector for a feature"""
        try:
            from river import drift
            
            if detector_type == "ADWIN":
                self.detectors[feature] = drift.ADWIN()
            elif detector_type == "DDM":
                self.detectors[feature] = drift.DDM()
            elif detector_type == "PageHinkley":
                self.detectors[feature] = drift.PageHinkley()
            else:
                self.detectors[feature] = drift.ADWIN()
            
            self.drift_count[feature] = 0
        except ImportError:
            print("River not installed, using fallback")
            self.detectors[feature] = None
    
    def learn_one(self, feature: str, value: float) -> bool:
        """Update detector with a single value"""
        if feature not in self.detectors:
            return False
        
        detector = self.detectors[feature]
        if detector is None:
            return False
        
        detector.learn_one(value)
        
        # Check for drift
        if detector.drift_detected:
            self.drift_count[feature] += 1
            return True
        
        return False
    
    def get_drift_count(self, feature: str) -> int:
        """Get number of drifts detected for a feature"""
        return self.drift_count.get(feature, 0)
    
    def reset(self, feature: str):
        """Reset detector for a feature"""
        if feature in self.detectors:
            self.drift_count[feature] = 0


class ConceptDriftDetector:
    """Detects concept drift via performance monitoring"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history: List[float] = []
        self.baseline_performance: Optional[float] = None
    
    def set_baseline(self, performance: float):
        """Set baseline performance"""
        self.baseline_performance = performance
    
    def update(self, performance: float) -> Optional[DriftDetectionResult]:
        """Update with new performance metric"""
        self.performance_history.append(performance)
        
        # Keep only recent history
        if len(self.performance_history) > self.window_size:
            self.performance_history = self.performance_history[-self.window_size:]
        
        # Check for concept drift
        if len(self.performance_history) < self.window_size:
            return None
        
        if self.baseline_performance is None:
            return None
        
        # Calculate degradation
        recent_avg = np.mean(self.performance_history[-20:])
        degradation = self.baseline_performance - recent_avg
        
        if degradation > 0.1:  # 10% degradation threshold
            return DriftDetectionResult(
                drift_detected=True,
                severity=DriftSeverity.HIGH if degradation > 0.2 else DriftSeverity.MEDIUM,
                drift_type="concept_drift",
                affected_features=[],
                metrics={"degradation": degradation, "recent_avg": recent_avg},
                summary=f"Concept drift detected: performance degraded by {degradation:.2%}"
            )
        
        return None


class DetectorSuite:
    """Suite of detectors for comprehensive drift monitoring"""
    
    def __init__(self):
        self.evidently_detector = EvidentlyDetector()
        self.river_detector = RiverDetector()
        self.concept_drift_detector = ConceptDriftDetector()
    
    def detect_batch_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        threshold: float = 0.05
    ) -> DriftDetectionResult:
        """Detect batch data drift"""
        self.evidently_detector.set_reference(reference_data)
        return self.evidently_detector.detect_drift(current_data, threshold=threshold)
    
    def detect_concept_drift(self, performance: float) -> Optional[DriftDetectionResult]:
        """Detect concept drift"""
        return self.concept_drift_detector.update(performance)
    
    def setup_streaming_detectors(self, features: List[str]):
        """Setup streaming detectors for features"""
        for feature in features:
            self.river_detector.add_detector(feature)
