"""
Data Drift Detection Module

Detects statistical changes in input data distributions using multiple methods:
- Kolmogorov-Smirnov test for numerical features
- Chi-squared test for categorical features
- Population Stability Index (PSI)
- Wasserstein distance
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DriftSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of drift detection for a single feature"""
    feature_name: str
    drift_detected: bool
    severity: DriftSeverity
    p_value: Optional[float]
    statistic: float
    method: str
    threshold: float
    description: str


@dataclass
class DataDriftReport:
    """Complete data drift report"""
    model_id: str
    timestamp: str
    overall_drift_detected: bool
    overall_severity: DriftSeverity
    feature_results: List[DriftResult]
    summary: str
    recommendations: List[str]


class DataDriftDetector:
    """Detects data drift using multiple statistical methods"""
    
    def __init__(
        self,
        ks_threshold: float = 0.05,
        chi2_threshold: float = 0.05,
        psi_threshold: float = 0.1,
        wasserstein_threshold: float = 0.1
    ):
        self.ks_threshold = ks_threshold
        self.chi2_threshold = chi2_threshold
        self.psi_threshold = psi_threshold
        self.wasserstein_threshold = wasserstein_threshold
    
    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        model_id: str
    ) -> DataDriftReport:
        """
        Detect data drift between reference and current datasets
        
        Args:
            reference_data: Baseline data (training or validation set)
            current_data: New incoming data
            model_id: Identifier for the model being monitored
            
        Returns:
            DataDriftReport with detailed drift analysis
        """
        from datetime import datetime
        
        feature_results = []
        
        # Analyze each feature
        for column in reference_data.columns:
            if column not in current_data.columns:
                continue
                
            ref_col = reference_data[column].dropna()
            curr_col = current_data[column].dropna()
            
            # Determine feature type
            if self._is_numerical(ref_col):
                result = self._detect_numerical_drift(column, ref_col, curr_col)
            else:
                result = self._detect_categorical_drift(column, ref_col, curr_col)
            
            feature_results.append(result)
        
        # Calculate overall drift
        drift_count = sum(1 for r in feature_results if r.drift_detected)
        severity_counts = {s: 0 for s in DriftSeverity}
        for r in feature_results:
            severity_counts[r.severity] += 1
        
        overall_drift = drift_count > 0
        overall_severity = self._calculate_overall_severity(severity_counts)
        
        # Generate summary and recommendations
        summary = self._generate_summary(feature_results, overall_drift)
        recommendations = self._generate_recommendations(feature_results)
        
        return DataDriftReport(
            model_id=model_id,
            timestamp=datetime.utcnow().isoformat(),
            overall_drift_detected=overall_drift,
            overall_severity=overall_severity,
            feature_results=feature_results,
            summary=summary,
            recommendations=recommendations
        )
    
    def _is_numerical(self, series: pd.Series) -> bool:
        """Determine if a series is numerical"""
        return pd.api.types.is_numeric_dtype(series)
    
    def _detect_numerical_drift(
        self,
        feature_name: str,
        reference: pd.Series,
        current: pd.Series
    ) -> DriftResult:
        """Detect drift in numerical features using multiple methods"""
        results = []
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_p_value = stats.ks_2samp(reference, current)
        ks_drift = ks_p_value < self.ks_threshold
        results.append({
            'method': 'Kolmogorov-Smirnov',
            'drift': ks_drift,
            'p_value': ks_p_value,
            'statistic': ks_statistic,
            'threshold': self.ks_threshold
        })
        
        # Wasserstein distance
        wasserstein_dist = stats.wasserstein_distance(reference, current)
        # Normalize by standard deviation
        normalized_wasserstein = wasserstein_dist / (reference.std() + 1e-10)
        wasserstein_drift = normalized_wasserstein > self.wasserstein_threshold
        results.append({
            'method': 'Wasserstein Distance',
            'drift': wasserstein_drift,
            'p_value': None,
            'statistic': normalized_wasserstein,
            'threshold': self.wasserstein_threshold
        })
        
        # PSI (Population Stability Index)
        psi = self._calculate_psi(reference, current)
        psi_drift = psi > self.psi_threshold
        results.append({
            'method': 'PSI',
            'drift': psi_drift,
            'p_value': None,
            'statistic': psi,
            'threshold': self.psi_threshold
        })
        
        # Combine results (use most severe detection)
        drift_detected = any(r['drift'] for r in results)
        severity = self._determine_severity(results)
        
        # Use KS as primary statistic for reporting
        primary_result = results[0]
        
        return DriftResult(
            feature_name=feature_name,
            drift_detected=drift_detected,
            severity=severity,
            p_value=primary_result['p_value'],
            statistic=primary_result['statistic'],
            method=primary_result['method'],
            threshold=primary_result['threshold'],
            description=self._generate_numerical_description(results)
        )
    
    def _detect_categorical_drift(
        self,
        feature_name: str,
        reference: pd.Series,
        current: pd.Series
    ) -> DriftResult:
        """Detect drift in categorical features using chi-squared test"""
        # Create contingency table
        all_categories = set(reference.unique()).union(set(current.unique()))
        
        ref_counts = [reference.cat.get_categories() if hasattr(reference, 'cat') else reference]
        curr_counts = [current.cat.get_categories() if hasattr(current, 'cat') else current]
        
        # Build contingency table
        contingency = pd.DataFrame({
            'reference': [reference.value_counts().get(cat, 0) for cat in all_categories],
            'current': [current.value_counts().get(cat, 0) for cat in all_categories]
        })
        
        # Chi-squared test
        try:
            chi2_stat, chi2_p_value, _, _ = stats.chi2_contingency(contingency.values)
            chi2_drift = chi2_p_value < self.chi2_threshold
        except:
            # Fallback if chi-squared fails
            chi2_stat = 0.0
            chi2_p_value = 1.0
            chi2_drift = False
        
        # Calculate PSI for categorical
        psi = self._calculate_psi_categorical(reference, current)
        psi_drift = psi > self.psi_threshold
        
        drift_detected = chi2_drift or psi_drift
        severity = self._determine_categorical_severity(chi2_p_value, psi)
        
        return DriftResult(
            feature_name=feature_name,
            drift_detected=drift_detected,
            severity=severity,
            p_value=chi2_p_value,
            statistic=chi2_stat,
            method='Chi-Squared',
            threshold=self.chi2_threshold,
            description=self._generate_categorical_description(chi2_p_value, psi)
        )
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        # Create bins based on reference distribution
        _, bin_edges = pd.qcut(reference, q=bins, retbins=True, duplicates='drop')
        
        # Calculate percentages in each bin
        ref_percents = pd.cut(reference, bins=bin_edges, include_lowest=True).value_counts(normalize=True)
        curr_percents = pd.cut(current, bins=bin_edges, include_lowest=True).value_counts(normalize=True)
        
        # Align bins
        all_bins = bin_edges[:-1]
        psi = 0.0
        
        for i, bin_edge in enumerate(all_bins):
            ref_p = ref_percents.get(i, 1e-10)
            curr_p = curr_percents.get(i, 1e-10)
            
            # Avoid division by zero
            if ref_p > 0 and curr_p > 0:
                psi += (curr_p - ref_p) * np.log(curr_p / ref_p)
        
        return psi
    
    def _calculate_psi_categorical(self, reference: pd.Series, current: pd.Series) -> float:
        """Calculate PSI for categorical features"""
        ref_counts = reference.value_counts(normalize=True)
        curr_counts = current.value_counts(normalize=True)
        
        psi = 0.0
        all_categories = set(ref_counts.index).union(set(curr_counts.index))
        
        for category in all_categories:
            ref_p = ref_counts.get(category, 1e-10)
            curr_p = curr_counts.get(category, 1e-10)
            
            if ref_p > 0 and curr_p > 0:
                psi += (curr_p - ref_p) * np.log(curr_p / ref_p)
        
        return psi
    
    def _determine_severity(self, results: List[Dict]) -> DriftSeverity:
        """Determine drift severity based on multiple test results"""
        drift_count = sum(1 for r in results if r['drift'])
        
        if drift_count == 0:
            return DriftSeverity.NONE
        elif drift_count == 1:
            return DriftSeverity.LOW
        elif drift_count == 2:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.HIGH
    
    def _determine_categorical_severity(self, chi2_p: float, psi: float) -> DriftSeverity:
        """Determine severity for categorical features"""
        if chi2_p > self.chi2_threshold and psi <= self.psi_threshold:
            return DriftSeverity.NONE
        elif psi < 0.2:
            return DriftSeverity.LOW
        elif psi < 0.5:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.HIGH
    
    def _calculate_overall_severity(self, severity_counts: Dict[DriftSeverity, int]) -> DriftSeverity:
        """Calculate overall drift severity from feature-level results"""
        if severity_counts[DriftSeverity.CRITICAL] > 0:
            return DriftSeverity.CRITICAL
        elif severity_counts[DriftSeverity.HIGH] > 0:
            return DriftSeverity.HIGH
        elif severity_counts[DriftSeverity.MEDIUM] >= 2:
            return DriftSeverity.HIGH
        elif severity_counts[DriftSeverity.MEDIUM] > 0:
            return DriftSeverity.MEDIUM
        elif severity_counts[DriftSeverity.LOW] >= 3:
            return DriftSeverity.MEDIUM
        elif severity_counts[DriftSeverity.LOW] > 0:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE
    
    def _generate_numerical_description(self, results: List[Dict]) -> str:
        """Generate description for numerical drift"""
        descriptions = []
        for r in results:
            if r['p_value']:
                descriptions.append(
                    f"{r['method']}: p-value={r['p_value']:.4f} "
                    f"(threshold={r['threshold']})"
                )
            else:
                descriptions.append(
                    f"{r['method']}: statistic={r['statistic']:.4f} "
                    f"(threshold={r['threshold']})"
                )
        return "; ".join(descriptions)
    
    def _generate_categorical_description(self, chi2_p: float, psi: float) -> str:
        """Generate description for categorical drift"""
        return f"Chi-Squared p-value={chi2_p:.4f}; PSI={psi:.4f}"
    
    def _generate_summary(self, feature_results: List[DriftResult], overall_drift: bool) -> str:
        """Generate summary of drift analysis"""
        total_features = len(feature_results)
        drifted_features = [r for r in feature_results if r.drift_detected]
        num_drifted = len(drifted_features)
        
        if not overall_drift:
            return f"No significant data drift detected across {total_features} features."
        
        high_severity = [r for r in drifted_features if r.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]]
        medium_severity = [r for r in drifted_features if r.severity == DriftSeverity.MEDIUM]
        low_severity = [r for r in drifted_features if r.severity == DriftSeverity.LOW]
        
        summary_parts = [
            f"Data drift detected in {num_drifted}/{total_features} features."
        ]
        
        if high_severity:
            summary_parts.append(f"High severity drift in: {', '.join([r.feature_name for r in high_severity])}")
        if medium_severity:
            summary_parts.append(f"Medium severity drift in: {', '.join([r.feature_name for r in medium_severity])}")
        if low_severity:
            summary_parts.append(f"Low severity drift in: {', '.join([r.feature_name for r in low_severity])}")
        
        return " ".join(summary_parts)
    
    def _generate_recommendations(self, feature_results: List[DriftResult]) -> List[str]:
        """Generate actionable recommendations based on drift results"""
        recommendations = []
        drifted_features = [r for r in feature_results if r.drift_detected]
        
        if not drifted_features:
            recommendations.append("Continue regular monitoring schedule.")
            return recommendations
        
        high_severity = [r for r in drifted_features if r.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]]
        medium_severity = [r for r in drifted_features if r.severity == DriftSeverity.MEDIUM]
        
        if high_severity:
            recommendations.append(
                f"URGENT: High severity drift detected in {len(high_severity)} features. "
                "Immediate investigation required."
            )
            recommendations.append("Consider triggering model retraining pipeline.")
        
        if medium_severity:
            recommendations.append(
                f"Medium severity drift in {len(medium_severity)} features. "
                "Schedule investigation within 24 hours."
            )
        
        recommendations.append("Review data pipeline for changes in data sources or collection methods.")
        recommendations.append("Check for seasonal patterns or external events that may explain drift.")
        
        return recommendations
