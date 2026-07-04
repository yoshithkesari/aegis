"""
Investigation Tools - Read-only tools for LLM to call during investigation

These tools are READ-ONLY - the LLM can investigate but cannot deploy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result from a tool call"""
    tool_name: str
    success: bool
    data: Any
    error: Optional[str] = None


class SlicePerformance:
    """Analyze performance by segment/feature value"""
    
    def __init__(self, predictions_df: pd.DataFrame):
        self.df = predictions_df
    
    def by_feature(self, feature: str, target: str = "is_fraud") -> ToolResult:
        """Get performance metrics sliced by a feature"""
        try:
            if feature not in self.df.columns:
                return ToolResult(
                    tool_name="slice_performance",
                    success=False,
                    data=None,
                    error=f"Feature {feature} not found"
                )
            
            results = []
            for value in self.df[feature].unique():
                subset = self.df[self.df[feature] == value]
                
                if len(subset) < 10:
                    continue
                
                accuracy = (subset['prediction'] == subset[target]).mean()
                
                results.append({
                    "feature": feature,
                    "value": str(value),
                    "count": len(subset),
                    "accuracy": float(accuracy)
                })
            
            return ToolResult(
                tool_name="slice_performance",
                success=True,
                data=results
            )
        except Exception as e:
            return ToolResult(
                tool_name="slice_performance",
                success=False,
                data=None,
                error=str(e)
            )


class AttributeDrift:
    """Per-feature drift contribution using SHAP + PSI"""
    
    def __init__(self, reference_df: pd.DataFrame, current_df: pd.DataFrame):
        self.reference = reference_df
        self.current = current_df
    
    def calculate_psi(self, feature: str, bins: int = 10) -> float:
        """Calculate Population Stability Index for a feature"""
        try:
            ref_col = self.reference[feature].dropna()
            curr_col = self.current[feature].dropna()
            
            if pd.api.types.is_numeric_dtype(ref_col):
                # Numerical - use quantile bins
                _, bin_edges = pd.qcut(ref_col, q=bins, retbins=True, duplicates='drop')
                ref_percents = pd.cut(ref_col, bins=bin_edges, include_lowest=True).value_counts(normalize=True)
                curr_percents = pd.cut(curr_col, bins=bin_edges, include_lowest=True).value_counts(normalize=True)
            else:
                # Categorical - use unique values
                all_values = set(ref_col.unique()).union(set(curr_col.unique()))
                ref_percents = ref_col.value_counts(normalize=True)
                curr_percents = curr_col.value_counts(normalize=True)
            
            # Calculate PSI
            psi = 0.0
            for idx in ref_percents.index:
                ref_p = ref_percents.get(idx, 1e-10)
                curr_p = curr_percents.get(idx, 1e-10)
                psi += (curr_p - ref_p) * np.log(curr_p / ref_p + 1e-10)
            
            return float(psi)
        except:
            return 0.0
    
    def per_feature_drift(self) -> ToolResult:
        """Calculate drift contribution for all features"""
        try:
            results = []
            
            for feature in self.reference.columns:
                if feature not in self.current.columns:
                    continue
                
                psi = self.calculate_psi(feature)
                
                results.append({
                    "feature": feature,
                    "psi": psi,
                    "severity": "high" if psi > 0.2 else "medium" if psi > 0.1 else "low"
                })
            
            # Sort by PSI descending
            results.sort(key=lambda x: x["psi"], reverse=True)
            
            return ToolResult(
                tool_name="attribute_drift",
                success=True,
                data=results
            )
        except Exception as e:
            return ToolResult(
                tool_name="attribute_drift",
                success=False,
                data=None,
                error=str(e)
            )


class QueryDeployLog:
    """Query recent model/config/pipeline deploys"""
    
    def __init__(self):
        # In production, this would query a real deploy log
        self.deploy_log = [
            {
                "timestamp": "2024-01-15T10:00:00",
                "model_version": "v1.2",
                "change_type": "model_retrain",
                "description": "Retrained on 30 days of data"
            },
            {
                "timestamp": "2024-01-10T14:30:00",
                "model_version": "v1.1",
                "change_type": "config_update",
                "description": "Updated feature preprocessing"
            }
        ]
    
    def recent_deploys(self, limit: int = 10) -> ToolResult:
        """Get recent deployments"""
        try:
            return ToolResult(
                tool_name="query_deploy_log",
                success=True,
                data=self.deploy_log[:limit]
            )
        except Exception as e:
            return ToolResult(
                tool_name="query_deploy_log",
                success=False,
                data=None,
                error=str(e)
            )


class DiffSchema:
    """Compare dtype / null-rate / cardinality changes"""
    
    def __init__(self, reference_df: pd.DataFrame, current_df: pd.DataFrame):
        self.reference = reference_df
        self.current = current_df
    
    def compare(self) -> ToolResult:
        """Compare schemas between reference and current"""
        try:
            results = []
            
            for col in self.reference.columns:
                if col not in self.current.columns:
                    results.append({
                        "feature": col,
                        "change_type": "column_missing",
                        "reference_dtype": str(self.reference[col].dtype),
                        "current_dtype": None,
                        "reference_null_rate": self.reference[col].isna().mean(),
                        "current_null_rate": None
                    })
                    continue
                
                ref_dtype = str(self.reference[col].dtype)
                curr_dtype = str(self.current[col].dtype)
                ref_null_rate = self.reference[col].isna().mean()
                curr_null_rate = self.current[col].isna().mean()
                
                ref_cardinality = self.reference[col].nunique()
                curr_cardinality = self.current[col].nunique()
                
                change_type = "none"
                if ref_dtype != curr_dtype:
                    change_type = "dtype_change"
                elif abs(ref_null_rate - curr_null_rate) > 0.05:
                    change_type = "null_rate_change"
                elif curr_cardinality > ref_cardinality * 1.5:
                    change_type = "cardinality_increase"
                
                results.append({
                    "feature": col,
                    "change_type": change_type,
                    "reference_dtype": ref_dtype,
                    "current_dtype": curr_dtype,
                    "reference_null_rate": float(ref_null_rate),
                    "current_null_rate": float(curr_null_rate),
                    "reference_cardinality": ref_cardinality,
                    "current_cardinality": curr_cardinality
                })
            
            # Check for new columns
            for col in self.current.columns:
                if col not in self.reference.columns:
                    results.append({
                        "feature": col,
                        "change_type": "new_column",
                        "reference_dtype": None,
                        "current_dtype": str(self.current[col].dtype),
                        "reference_null_rate": None,
                        "current_null_rate": float(self.current[col].isna().mean())
                    })
            
            return ToolResult(
                tool_name="diff_schema",
                success=True,
                data=results
            )
        except Exception as e:
            return ToolResult(
                tool_name="diff_schema",
                success=False,
                data=None,
                error=str(e)
            )


class EstimatePerfWithoutLabels:
    """Estimate performance without ground truth (NannyML CBPE placeholder)"""
    
    def __init__(self):
        pass
    
    def estimate(self, predictions_df: pd.DataFrame) -> ToolResult:
        """Estimate performance using CBPE (placeholder)"""
        try:
            # In production, this would use NannyML CBPE
            # For demo, we provide a placeholder implementation
            
            if 'prediction' in predictions_df.columns:
                # Simple heuristic based on prediction distribution
                pred_mean = predictions_df['prediction'].mean()
                pred_std = predictions_df['prediction'].std()
                
                # Estimate accuracy based on prediction confidence
                estimated_accuracy = max(0.5, min(0.95, 1.0 - pred_std))
                
                return ToolResult(
                    tool_name="estimate_perf_without_labels",
                    success=True,
                    data={
                        "estimated_accuracy": float(estimated_accuracy),
                        "method": "heuristic_placeholder",
                        "confidence": "low"
                    }
                )
            else:
                return ToolResult(
                    tool_name="estimate_perf_without_labels",
                    success=False,
                    data=None,
                    error="No predictions column found"
                )
        except Exception as e:
            return ToolResult(
                tool_name="estimate_perf_without_labels",
                success=False,
                data=None,
                error=str(e)
            )


class InvestigationToolkit:
    """Toolkit containing all investigation tools"""
    
    def __init__(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        predictions_df: Optional[pd.DataFrame] = None
    ):
        self.slice_performance = SlicePerformance(predictions_df) if predictions_df is not None else None
        self.attribute_drift = AttributeDrift(reference_df, current_df)
        self.query_deploy_log = QueryDeployLog()
        self.diff_schema = DiffSchema(reference_df, current_df)
        self.estimate_perf = EstimatePerfWithoutLabels()
    
    def get_tool_descriptions(self) -> List[Dict[str, str]]:
        """Get descriptions of available tools for the LLM"""
        return [
            {
                "name": "slice_performance",
                "description": "Analyze performance metrics sliced by feature values to identify segment-specific issues"
            },
            {
                "name": "attribute_drift",
                "description": "Calculate per-feature drift contribution using PSI to identify which features are drifting most"
            },
            {
                "name": "query_deploy_log",
                "description": "Query recent model, config, and pipeline deployments to identify recent changes"
            },
            {
                "name": "diff_schema",
                "description": "Compare dtype, null-rate, and cardinality changes between reference and current data"
            },
            {
                "name": "estimate_perf_without_labels",
                "description": "Estimate model performance without ground truth labels using CBPE"
            }
        ]
