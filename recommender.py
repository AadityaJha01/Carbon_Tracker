"""
Model recommendation system for optimal carbon-efficient training
"""

import pandas as pd
from typing import Dict, Optional, List, Tuple


class ModelRecommender:
    """
    Recommends optimal model and configuration based on target accuracy or constraints.
    """
    
    def __init__(self, leaderboard_df: pd.DataFrame):
        """
        Initialize recommender with leaderboard data.
        
        Args:
            leaderboard_df: DataFrame with training run data
        """
        self.df = leaderboard_df.copy()
    
    def recommend_by_accuracy(
        self,
        target_accuracy: float,
        tolerance: float = 2.0
    ) -> Optional[Dict]:
        """
        Recommend model configuration to achieve target accuracy with minimum CO₂.
        
        Args:
            target_accuracy: Target accuracy percentage
            tolerance: Acceptable accuracy range (±tolerance)
        
        Returns:
            Dictionary with recommended model and configuration
        """
        if self.df.empty:
            return None
        
        # Filter runs within accuracy range
        valid_runs = self.df[
            (self.df['accuracy'] >= target_accuracy - tolerance) &
            (self.df['accuracy'] <= target_accuracy + tolerance)
        ]
        
        if valid_runs.empty:
            # Find closest match
            self.df['acc_diff'] = abs(self.df['accuracy'] - target_accuracy)
            closest = self.df.nsmallest(1, 'acc_diff')
            valid_runs = closest
        
        # Sort by CO₂ emissions (lowest first)
        best_run = valid_runs.nsmallest(1, 'co2_g')
        
        if best_run.empty:
            return None
        
        run = best_run.iloc[0]
        
        recommendation = {
            'model': run['model'],
            'expected_accuracy': run['accuracy'],
            'expected_co2_g': run['co2_g'],
            'expected_energy_kwh': run['energy_kwh'],
            'expected_time_sec': run['training_time_sec'],
            'recommended_epochs': run['epochs'],
            'recommended_batch_size': run['batch_size'],
            'use_fp16': run.get('fp16', False),
            'use_early_stop': run.get('early_stop', False),
            'efficiency_score': run.get('accuracy_per_kwh', 0),
            'reason': f"Lowest CO₂ ({run['co2_g']:.2f}g) while achieving ~{run['accuracy']:.1f}% accuracy"
        }
        
        return recommendation
    
    def recommend_by_co2_budget(
        self,
        max_co2_g: float
    ) -> Optional[Dict]:
        """
        Recommend model configuration that maximizes accuracy within CO₂ budget.
        
        Args:
            max_co2_g: Maximum CO₂ emissions allowed (grams)
        
        Returns:
            Dictionary with recommended model and configuration
        """
        if self.df.empty:
            return None
        
        # Filter runs within CO₂ budget
        valid_runs = self.df[self.df['co2_g'] <= max_co2_g]
        
        if valid_runs.empty:
            return None
        
        # Sort by accuracy (highest first)
        best_run = valid_runs.nlargest(1, 'accuracy')
        run = best_run.iloc[0]
        
        recommendation = {
            'model': run['model'],
            'expected_accuracy': run['accuracy'],
            'expected_co2_g': run['co2_g'],
            'expected_energy_kwh': run['energy_kwh'],
            'expected_time_sec': run['training_time_sec'],
            'recommended_epochs': run['epochs'],
            'recommended_batch_size': run['batch_size'],
            'use_fp16': run.get('fp16', False),
            'use_early_stop': run.get('early_stop', False),
            'efficiency_score': run.get('accuracy_per_kwh', 0),
            'reason': f"Highest accuracy ({run['accuracy']:.1f}%) within CO₂ budget ({run['co2_g']:.2f}g)"
        }
        
        return recommendation
    
    def recommend_by_time_budget(
        self,
        max_time_hours: float
    ) -> Optional[Dict]:
        """
        Recommend model configuration that maximizes accuracy within time budget.
        
        Args:
            max_time_hours: Maximum training time allowed (hours)
        
        Returns:
            Dictionary with recommended model and configuration
        """
        if self.df.empty:
            return None
        
        max_time_sec = max_time_hours * 3600
        
        # Filter runs within time budget
        valid_runs = self.df[self.df['training_time_sec'] <= max_time_sec]
        
        if valid_runs.empty:
            return None
        
        # Sort by accuracy (highest first)
        best_run = valid_runs.nlargest(1, 'accuracy')
        run = best_run.iloc[0]
        
        recommendation = {
            'model': run['model'],
            'expected_accuracy': run['accuracy'],
            'expected_co2_g': run['co2_g'],
            'expected_energy_kwh': run['energy_kwh'],
            'expected_time_sec': run['training_time_sec'],
            'recommended_epochs': run['epochs'],
            'recommended_batch_size': run['batch_size'],
            'use_fp16': run.get('fp16', False),
            'use_early_stop': run.get('early_stop', False),
            'efficiency_score': run.get('accuracy_per_kwh', 0),
            'reason': f"Highest accuracy ({run['accuracy']:.1f}%) within time budget ({max_time_hours:.1f}h)"
        }
        
        return recommendation
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all models by average metrics.
        
        Returns:
            DataFrame with aggregated metrics per model
        """
        if self.df.empty:
            return pd.DataFrame()
        
        comparison = self.df.groupby('model').agg({
            'accuracy': ['mean', 'std', 'max'],
            'co2_g': ['mean', 'std', 'min'],
            'energy_kwh': ['mean', 'std', 'min'],
            'training_time_sec': ['mean', 'std', 'min'],
            'accuracy_per_kwh': ['mean', 'max']
        }).round(2)
        
        return comparison
    
    def get_best_model(self, metric: str = 'accuracy_per_kwh') -> Optional[Dict]:
        """
        Get the best model by specified metric.
        
        Args:
            metric: Metric to optimize (default: 'accuracy_per_kwh')
        
        Returns:
            Dictionary with best model information
        """
        if self.df.empty:
            return None
        
        best_run = self.df.nlargest(1, metric)
        run = best_run.iloc[0]
        
        return {
            'model': run['model'],
            'metric': metric,
            'value': run[metric],
            'accuracy': run['accuracy'],
            'co2_g': run['co2_g'],
            'energy_kwh': run['energy_kwh'],
            'training_time_sec': run['training_time_sec']
        }

