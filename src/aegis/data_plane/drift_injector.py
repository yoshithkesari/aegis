"""
Drift Injector - Injects drift into data streams for testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from enum import Enum


class InjectionMode(Enum):
    """Types of drift injection"""
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    RECURRING = "recurring"


class DriftInjector:
    """Injects drift into data streams"""
    
    def __init__(self):
        self.active_injections: Dict[str, Dict] = {}
    
    def inject_mean_shift(
        self,
        data: pd.DataFrame,
        feature: str,
        shift_amount: float,
        start_row: int,
        mode: InjectionMode = InjectionMode.SUDDEN
    ) -> pd.DataFrame:
        """Inject mean shift into a feature"""
        result = data.copy()
        
        if mode == InjectionMode.SUDDEN:
            result.loc[start_row:, feature] += shift_amount
        elif mode == InjectionMode.GRADUAL:
            # Gradual shift over 1000 rows
            n_gradual = min(1000, len(result) - start_row)
            shifts = np.linspace(0, shift_amount, n_gradual)
            result.loc[start_row:start_row + n_gradual - 1, feature] += shifts
        
        return result
    
    def inject_variance_change(
        self,
        data: pd.DataFrame,
        feature: str,
        scale_factor: float,
        start_row: int
    ) -> pd.DataFrame:
        """Inject variance change into a feature"""
        result = data.copy()
        
        original_mean = result.loc[:start_row, feature].mean()
        original_std = result.loc[:start_row, feature].std()
        
        # Change variance after start_row
        new_std = original_std * scale_factor
        n_rows = len(result) - start_row
        
        result.loc[start_row:, feature] = np.random.normal(
            original_mean,
            new_std,
            n_rows
        )
        
        return result
    
    def inject_category_shift(
        self,
        data: pd.DataFrame,
        feature: str,
        new_category: str,
        start_row: int,
        probability: float = 0.2
    ) -> pd.DataFrame:
        """Inject new category into categorical feature"""
        result = data.copy()
        
        n_rows = len(result) - start_row
        mask = np.random.random(n_rows) < probability
        indices = result.index[start_row:][mask]
        result.loc[indices, feature] = new_category
        
        return result
    
    def inject_nulls(
        self,
        data: pd.DataFrame,
        feature: str,
        start_row: int,
        null_rate: float = 0.1
    ) -> pd.DataFrame:
        """Inject nulls into a feature"""
        result = data.copy()
        
        n_rows = len(result) - start_row
        mask = np.random.random(n_rows) < null_rate
        indices = result.index[start_row:][mask]
        result.loc[indices, feature] = None
        
        return result
    
    def inject_constant(
        self,
        data: pd.DataFrame,
        feature: str,
        start_row: int
    ) -> pd.DataFrame:
        """Inject constant value (upstream bug simulation)"""
        result = data.copy()
        
        constant_value = result.loc[start_row - 1, feature]
        result.loc[start_row:, feature] = constant_value
        
        return result
    
    def inject_correlation_break(
        self,
        data: pd.DataFrame,
        feature1: str,
        feature2: str,
        start_row: int
    ) -> pd.DataFrame:
        """Break correlation between two features"""
        result = data.copy()
        
        # Shuffle feature2 after start_row to break correlation
        feature2_values = result.loc[start_row:, feature2].values
        np.random.shuffle(feature2_values)
        result.loc[start_row:, feature2] = feature2_values
        
        return result


class StreamReplayer:
    """Replays data streams with drift injection"""
    
    def __init__(self, data: pd.DataFrame, batch_size: int = 100):
        self.data = data
        self.batch_size = batch_size
        self.current_row = 0
        self.injector = DriftInjector()
        self.injection_points: List[Dict] = []
    
    def add_injection(
        self,
        row: int,
        injection_fn: Callable,
        **kwargs
    ):
        """Add an injection at a specific row"""
        self.injection_points.append({
            'row': row,
            'fn': injection_fn,
            'kwargs': kwargs
        })
        self.injection_points.sort(key=lambda x: x['row'])
    
    def replay(self) -> pd.DataFrame:
        """Replay the stream with injections"""
        result = self.data.copy()
        
        for injection in self.injection_points:
            row = injection['row']
            if row < len(result):
                result = injection['fn'](result, start_row=row, **injection['kwargs'])
        
        return result
    
    def stream_batches(self) -> pd.DataFrame:
        """Stream data in batches"""
        while self.current_row < len(self.data):
            end_row = min(self.current_row + self.batch_size, len(self.data))
            batch = self.data.iloc[self.current_row:end_row].copy()
            
            # Apply injections if we've passed their points
            for injection in self.injection_points:
                if self.current_row <= injection['row'] < end_row:
                    batch = injection['fn'](batch, start_row=injection['row'] - self.current_row, **injection['kwargs'])
            
            self.current_row = end_row
            yield batch
    
    def reset(self):
        """Reset the stream"""
        self.current_row = 0
