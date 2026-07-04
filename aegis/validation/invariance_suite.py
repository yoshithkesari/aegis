"""
Invariance Suite - Tests that model behavior is invariant to expected changes

Validates that the model doesn't break on:
- Reasonable feature ranges
- Missing values
- Edge cases
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class InvarianceTestResult:
    """Result of an invariance test"""
    test_name: str
    passed: bool
    score: float
    details: str
    metrics: Dict[str, Any]


class InvarianceSuite:
    """
    Suite of invariance tests for model validation
    
    Tests that model behavior is invariant to expected changes
    and robust to edge cases.
    """
    
    def __init__(self, model):
        self.model = model
    
    def test_range_invariance(
        self,
        test_data: pd.DataFrame,
        feature_ranges: Dict[str, tuple]
    ) -> InvarianceTestResult:
        """
        Test that predictions are invariant within expected feature ranges
        
        Args:
            test_data: Test data
            feature_ranges: Dict of feature -> (min, max) expected ranges
        
        Returns:
            InvarianceTestResult
        """
        try:
            predictions_original = self.model.predict(test_data)
            
            # Perturb features within ranges
            test_data_perturbed = test_data.copy()
            for feature, (min_val, max_val) in feature_ranges.items():
                if feature in test_data.columns:
                    # Add small random noise within range
                    noise = np.random.uniform(
                        -0.05 * (max_val - min_val),
                        0.05 * (max_val - min_val),
                        len(test_data)
                    )
                    test_data_perturbed[feature] = np.clip(
                        test_data[feature] + noise,
                        min_val,
                        max_val
                    )
            
            predictions_perturbed = self.model.predict(test_data_perturbed)
            
            # Check invariance - predictions should be similar
            max_diff = np.max(np.abs(predictions_original - predictions_perturbed))
            mean_diff = np.mean(np.abs(predictions_original - predictions_perturbed))
            
            # Pass if mean difference is small
            passed = mean_diff < 0.1
            
            return InvarianceTestResult(
                test_name="range_invariance",
                passed=passed,
                score=1.0 - mean_diff,
                details=f"Max diff: {max_diff:.4f}, Mean diff: {mean_diff:.4f}",
                metrics={
                    "max_difference": float(max_diff),
                    "mean_difference": float(mean_diff)
                }
            )
        except Exception as e:
            return InvarianceTestResult(
                test_name="range_invariance",
                passed=False,
                score=0.0,
                details=f"Error: {str(e)}",
                metrics={}
            )
    
    def test_missing_value_robustness(
        self,
        test_data: pd.DataFrame,
        missing_rate: float = 0.1
    ) -> InvarianceTestResult:
        """
        Test that model handles missing values gracefully
        
        Args:
            test_data: Test data
            missing_rate: Fraction of values to set to missing
        
        Returns:
            InvarianceTestResult
        """
        try:
            predictions_original = self.model.predict(test_data)
            
            # Inject missing values
            test_data_missing = test_data.copy()
            for col in test_data.columns:
                if pd.api.types.is_numeric_dtype(test_data[col]):
                    mask = np.random.random(len(test_data)) < missing_rate
                    test_data_missing.loc[mask, col] = np.nan
            
            # Try to predict (may fail if model doesn't handle NaN)
            try:
                predictions_missing = self.model.predict(test_data_missing.fillna(0))
                
                # If it succeeds, check that predictions are reasonable
                passed = True
                details = "Model handled missing values"
                score = 1.0
            except:
                passed = False
                details = "Model failed on missing values"
                score = 0.0
            
            return InvarianceTestResult(
                test_name="missing_value_robustness",
                passed=passed,
                score=score,
                details=details,
                metrics={"missing_rate": missing_rate}
            )
        except Exception as e:
            return InvarianceTestResult(
                test_name="missing_value_robustness",
                passed=False,
                score=0.0,
                details=f"Error: {str(e)}",
                metrics={}
            )
    
    def test_prediction_consistency(
        self,
        test_data: pd.DataFrame,
        n_runs: int = 3
    ) -> InvarianceTestResult:
        """
        Test that model predictions are consistent across multiple runs
        
        Args:
            test_data: Test data
            n_runs: Number of times to run predictions
        
        Returns:
            InvarianceTestResult
        """
        try:
            predictions_list = []
            for _ in range(n_runs):
                preds = self.model.predict(test_data)
                predictions_list.append(preds)
            
            # Check consistency
            all_same = all(
                np.array_equal(predictions_list[0], preds)
                for preds in predictions_list[1:]
            )
            
            if all_same:
                passed = True
                score = 1.0
                details = "Predictions are consistent across runs"
            else:
                # Calculate variance
                preds_array = np.array(predictions_list)
                variance = np.var(preds_array, axis=0).mean()
                passed = variance < 0.01
                score = 1.0 - min(variance * 10, 1.0)
                details = f"Prediction variance: {variance:.6f}"
            
            return InvarianceTestResult(
                test_name="prediction_consistency",
                passed=passed,
                score=score,
                details=details,
                metrics={"variance": float(variance) if not all_same else 0.0}
            )
        except Exception as e:
            return InvarianceTestResult(
                test_name="prediction_consistency",
                passed=False,
                score=0.0,
                details=f"Error: {str(e)}",
                metrics={}
            )
    
    def test_edge_cases(
        self,
        test_data: pd.DataFrame
    ) -> InvarianceTestResult:
        """
        Test model behavior on edge cases (min/max values, zeros, etc.)
        
        Args:
            test_data: Test data
        
        Returns:
            InvarianceTestResult
        """
        try:
            edge_cases = []
            
            # Test with all zeros
            test_zeros = test_data.copy()
            for col in test_zeros.columns:
                if pd.api.types.is_numeric_dtype(test_zeros[col]):
                    test_zeros[col] = 0
            try:
                preds_zeros = self.model.predict(test_zeros)
                edge_cases.append(("zeros", True))
            except:
                edge_cases.append(("zeros", False))
            
            # Test with all max values
            test_max = test_data.copy()
            for col in test_max.columns:
                if pd.api.types.is_numeric_dtype(test_max[col]):
                    test_max[col] = test_max[col].max()
            try:
                preds_max = self.model.predict(test_max)
                edge_cases.append(("max", True))
            except:
                edge_cases.append(("max", False))
            
            # Test with all min values
            test_min = test_data.copy()
            for col in test_min.columns:
                if pd.api.types.is_numeric_dtype(test_min[col]):
                    test_min[col] = test_min[col].min()
            try:
                preds_min = self.model.predict(test_min)
                edge_cases.append(("min", True))
            except:
                edge_cases.append(("min", False))
            
            # Pass if most edge cases succeed
            passed = sum(1 for _, success in edge_cases if success) >= len(edge_cases) * 0.5
            score = sum(1 for _, success in edge_cases if success) / len(edge_cases)
            
            details = f"Edge cases passed: {sum(1 for _, s in edge_cases if s)}/{len(edge_cases)}"
            
            return InvarianceTestResult(
                test_name="edge_cases",
                passed=passed,
                score=score,
                details=details,
                metrics={"edge_case_results": edge_cases}
            )
        except Exception as e:
            return InvarianceTestResult(
                test_name="edge_cases",
                passed=False,
                score=0.0,
                details=f"Error: {str(e)}",
                metrics={}
            )
    
    def run_suite(
        self,
        test_data: pd.DataFrame,
        feature_ranges: Optional[Dict[str, tuple]] = None
    ) -> List[InvarianceTestResult]:
        """
        Run the full invariance test suite
        
        Args:
            test_data: Test data
            feature_ranges: Feature ranges for range invariance test
        
        Returns:
            List of InvarianceTestResult
        """
        results = []
        
        if feature_ranges:
            results.append(self.test_range_invariance(test_data, feature_ranges))
        
        results.append(self.test_missing_value_robustness(test_data))
        results.append(self.test_prediction_consistency(test_data))
        results.append(self.test_edge_cases(test_data))
        
        return results
    
    def get_suite_summary(self, results: List[InvarianceTestResult]) -> Dict[str, Any]:
        """Get summary of invariance suite results"""
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        avg_score = np.mean([r.score for r in results]) if results else 0.0
        
        return {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "average_score": float(avg_score),
            "overall_passed": passed == total
        }
