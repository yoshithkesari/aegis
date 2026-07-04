"""
Drift Injection Scenarios for Eval Harness

Injects known drifts with ground truth for objective scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass


class DriftType(Enum):
    COVARIATE_SHIFT = "covariate_shift"
    CONCEPT_DRIFT_SEGMENT = "concept_drift_segment"
    SCHEMA_BREAK = "schema_break"
    UPSTREAM_BUG = "upstream_bug"
    NO_DRIFT = "no_drift"


@dataclass
class DriftScenario:
    """A drift injection scenario with ground truth"""
    scenario_id: str
    drift_type: DriftType
    description: str
    injection_row: int
    ground_truth_cause: str
    affected_features: List[str]
    severity: str


class FraudDataGenerator:
    """Generates synthetic fraud detection data for testing"""
    
    def __init__(self, n_samples: int = 10000):
        self.n_samples = n_samples
        self.merchant_categories = ["retail", "food", "travel", "online", "services"]
        self.locations = ["US", "UK", "CA", "AU", "DE"]
    
    def generate_baseline(self) -> pd.DataFrame:
        """Generate baseline (non-drifted) data"""
        np.random.seed(42)
        
        data = {
            'transaction_amount': np.random.lognormal(4.5, 0.8, self.n_samples),
            'merchant_category': np.random.choice(self.merchant_categories, self.n_samples),
            'time_of_day': np.random.uniform(0, 24, self.n_samples),
            'location': np.random.choice(self.locations, self.n_samples),
            'customer_age': np.random.normal(45, 15, self.n_samples),
            'transaction_count_7d': np.random.poisson(5, self.n_samples),
            'is_fraud': self._generate_fraud_labels()
        }
        
        df = pd.DataFrame(data)
        
        # Clip negative values
        df['customer_age'] = df['customer_age'].clip(18, 90)
        df['transaction_count_7d'] = df['transaction_count_7d'].clip(0, 50)
        
        return df
    
    def _generate_fraud_labels(self) -> np.ndarray:
        """Generate fraud labels based on feature patterns"""
        # Base fraud rate ~2%
        fraud = np.random.random(self.n_samples) < 0.02
        return fraud.astype(int)
    
    def inject_covariate_shift(
        self,
        df: pd.DataFrame,
        injection_row: int,
        feature: str = "transaction_amount",
        shift_factor: float = 1.5
    ) -> Tuple[pd.DataFrame, DriftScenario]:
        """Inject covariate shift - feature distribution changes"""
        df_drifted = df.copy()
        
        # Apply shift from injection row onwards
        if feature == "transaction_amount":
            df_drifted.loc[injection_row:, feature] *= shift_factor
        elif feature == "customer_age":
            df_drifted.loc[injection_row:, feature] += 10
        elif feature == "transaction_count_7d":
            df_drifted.loc[injection_row:, feature] = (
                df_drifted.loc[injection_row:, feature] * 1.3
            ).clip(0, 50)
        
        scenario = DriftScenario(
            scenario_id=f"covariate_shift_{feature}",
            drift_type=DriftType.COVARIATE_SHIFT,
            description=f"Covariate shift in {feature}",
            injection_row=injection_row,
            ground_truth_cause=f"Distribution shift in {feature}",
            affected_features=[feature],
            severity="medium"
        )
        
        return df_drifted, scenario
    
    def inject_concept_drift_segment(
        self,
        df: pd.DataFrame,
        injection_row: int,
        segment_feature: str = "merchant_category",
        segment_value: str = "online"
    ) -> Tuple[pd.DataFrame, DriftScenario]:
        """Inject concept drift - relationship changes for a segment"""
        df_drifted = df.copy()
        
        # For the specified segment, increase fraud rate after injection
        mask = (df_drifted[segment_feature] == segment_value) & (df_drifted.index >= injection_row)
        df_drifted.loc[mask, 'is_fraud'] = np.random.random(mask.sum()) < 0.15  # 15% fraud rate
        
        scenario = DriftScenario(
            scenario_id=f"concept_drift_{segment_feature}_{segment_value}",
            drift_type=DriftType.CONCEPT_DRIFT_SEGMENT,
            description=f"Concept drift in {segment_feature}={segment_value} segment",
            injection_row=injection_row,
            ground_truth_cause=f"Fraud pattern changed for {segment_value} merchants",
            affected_features=[segment_feature],
            severity="high"
        )
        
        return df_drifted, scenario
    
    def inject_schema_break(
        self,
        df: pd.DataFrame,
        injection_row: int,
        feature: str = "merchant_category"
    ) -> Tuple[pd.DataFrame, DriftScenario]:
        """Inject schema break - new category appears"""
        df_drifted = df.copy()
        
        # Add new category from injection row onwards
        new_categories = {
            "merchant_category": "crypto",
            "location": "JP",
            "time_of_day": None  # Null injection
        }
        
        if feature in new_categories:
            new_value = new_categories[feature]
            if new_value is None:
                # Inject nulls
                null_mask = (df_drifted.index >= injection_row) & (np.random.random(len(df_drifted)) < 0.1)
                df_drifted.loc[null_mask, feature] = None
            else:
                # Inject new category
                mask = (df_drifted.index >= injection_row) & (np.random.random(len(df_drifted)) < 0.15)
                df_drifted.loc[mask, feature] = new_value
        
        scenario = DriftScenario(
            scenario_id=f"schema_break_{feature}",
            drift_type=DriftType.SCHEMA_BREAK,
            description=f"Schema break in {feature}",
            injection_row=injection_row,
            ground_truth_cause=f"New value introduced in {feature}",
            affected_features=[feature],
            severity="high"
        )
        
        return df_drifted, scenario
    
    def inject_upstream_bug(
        self,
        df: pd.DataFrame,
        injection_row: int,
        feature: str = "transaction_amount"
    ) -> Tuple[pd.DataFrame, DriftScenario]:
        """Inject upstream bug - feature becomes constant or corrupted"""
        df_drifted = df.copy()
        
        # Make feature constant after injection (simulating sensor failure)
        if feature == "transaction_amount":
            constant_value = df_drifted.loc[injection_row-1, feature]
            df_drifted.loc[injection_row:, feature] = constant_value
        elif feature == "time_of_day":
            df_drifted.loc[injection_row:, feature] = 0.0  # All times become midnight
        
        scenario = DriftScenario(
            scenario_id=f"upstream_bug_{feature}",
            drift_type=DriftType.UPSTREAM_BUG,
            description=f"Upstream bug causing {feature} to become constant",
            injection_row=injection_row,
            ground_truth_cause=f"{feature} stuck at constant value due to upstream failure",
            affected_features=[feature],
            severity="critical"
        )
        
        return df_drifted, scenario


class ScenarioRunner:
    """Runs drift scenarios and returns data with ground truth"""
    
    def __init__(self, n_samples: int = 10000):
        self.generator = FraudDataGenerator(n_samples)
        self.scenarios: List[DriftScenario] = []
    
    def run_all_scenarios(self) -> Dict[str, Tuple[pd.DataFrame, DriftScenario]]:
        """Run all drift injection scenarios"""
        baseline = self.generator.generate_baseline()
        injection_row = int(len(baseline) * 0.7)  # Inject at 70% of data
        
        results = {
            "baseline": (baseline, DriftScenario(
                scenario_id="baseline",
                drift_type=DriftType.NO_DRIFT,
                description="No drift - baseline data",
                injection_row=-1,
                ground_truth_cause="None",
                affected_features=[],
                severity="none"
            ))
        }
        
        # Covariate shift scenarios
        for feature in ["transaction_amount", "customer_age", "transaction_count_7d"]:
            df, scenario = self.generator.inject_covariate_shift(
                baseline.copy(), injection_row, feature
            )
            results[scenario.scenario_id] = (df, scenario)
            self.scenarios.append(scenario)
        
        # Concept drift scenarios
        for merchant in ["online", "travel"]:
            df, scenario = self.generator.inject_concept_drift_segment(
                baseline.copy(), injection_row, "merchant_category", merchant
            )
            results[scenario.scenario_id] = (df, scenario)
            self.scenarios.append(scenario)
        
        # Schema break scenarios
        for feature in ["merchant_category", "location"]:
            df, scenario = self.generator.inject_schema_break(
                baseline.copy(), injection_row, feature
            )
            results[scenario.scenario_id] = (df, scenario)
            self.scenarios.append(scenario)
        
        # Upstream bug scenarios
        for feature in ["transaction_amount", "time_of_day"]:
            df, scenario = self.generator.inject_upstream_bug(
                baseline.copy(), injection_row, feature
            )
            results[scenario.scenario_id] = (df, scenario)
            self.scenarios.append(scenario)
        
        return results
    
    def get_scenario(self, scenario_id: str) -> Optional[Tuple[pd.DataFrame, DriftScenario]]:
        """Get a specific scenario"""
        all_scenarios = self.run_all_scenarios()
        return all_scenarios.get(scenario_id)
    
    def list_scenarios(self) -> List[DriftScenario]:
        """List all available scenarios"""
        if not self.scenarios:
            self.run_all_scenarios()
        return self.scenarios


if __name__ == "__main__":
    # Test scenario generation
    runner = ScenarioRunner(n_samples=5000)
    scenarios = runner.run_all_scenarios()
    
    print(f"Generated {len(scenarios)} scenarios")
    for scenario_id, (df, scenario) in scenarios.items():
        print(f"\n{scenario_id}:")
        print(f"  Type: {scenario.drift_type.value}")
        print(f"  Description: {scenario.description}")
        print(f"  Ground truth: {scenario.ground_truth_cause}")
        print(f"  Severity: {scenario.severity}")
