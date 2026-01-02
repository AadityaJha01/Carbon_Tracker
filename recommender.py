"""
Model recommendation system for optimal carbon-efficient training
"""

import pandas as pd
import numpy as np
from typing import Any


def _native(x: Any):
    """Convert numpy/pandas scalar types to native Python types for JSON.

    Leaves lists and dicts intact; converts numpy arrays to lists.
    """
    # pandas timestamp
    try:
        import pandas as _pd
        if isinstance(x, _pd.Timestamp):
            return x.isoformat()
    except Exception:
        pass

    # numpy scalar
    if isinstance(x, np.generic):
        return x.item()

    # numpy array or pandas Series/Index
    if isinstance(x, (np.ndarray, list, tuple)):
        return list(x)

    # pandas Series or Index
    try:
        if hasattr(x, 'tolist') and not isinstance(x, (str, bytes)):
            return x.tolist()
    except Exception:
        pass

    # fallback
    return x
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
            'model': _native(run['model']),
            'expected_accuracy': float(_native(run['accuracy'])),
            'expected_co2_g': float(_native(run['co2_g'])),
            'expected_energy_kwh': float(_native(run['energy_kwh'])),
            'expected_time_sec': float(_native(run['training_time_sec'])),
            'recommended_epochs': int(_native(run['epochs'])),
            'recommended_batch_size': int(_native(run['batch_size'])),
            'use_fp16': bool(_native(run.get('fp16', False))),
            'use_early_stop': bool(_native(run.get('early_stop', False))),
            'efficiency_score': float(_native(run.get('accuracy_per_kwh', 0))),
            'reason': f"Lowest CO₂ ({float(_native(run['co2_g'])):.2f}g) while achieving ~{float(_native(run['accuracy'])):.1f}% accuracy"
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
            'model': _native(run['model']),
            'expected_accuracy': float(_native(run['accuracy'])),
            'expected_co2_g': float(_native(run['co2_g'])),
            'expected_energy_kwh': float(_native(run['energy_kwh'])),
            'expected_time_sec': float(_native(run['training_time_sec'])),
            'recommended_epochs': int(_native(run['epochs'])),
            'recommended_batch_size': int(_native(run['batch_size'])),
            'use_fp16': bool(_native(run.get('fp16', False))),
            'use_early_stop': bool(_native(run.get('early_stop', False))),
            'efficiency_score': float(_native(run.get('accuracy_per_kwh', 0))),
            'reason': f"Highest accuracy ({float(_native(run['accuracy'])):.1f}%) within CO₂ budget ({float(_native(run['co2_g'])):.2f}g)"
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
            'model': _native(run['model']),
            'expected_accuracy': float(_native(run['accuracy'])),
            'expected_co2_g': float(_native(run['co2_g'])),
            'expected_energy_kwh': float(_native(run['energy_kwh'])),
            'expected_time_sec': float(_native(run['training_time_sec'])),
            'recommended_epochs': int(_native(run['epochs'])),
            'recommended_batch_size': int(_native(run['batch_size'])),
            'use_fp16': bool(_native(run.get('fp16', False))),
            'use_early_stop': bool(_native(run.get('early_stop', False))),
            'efficiency_score': float(_native(run.get('accuracy_per_kwh', 0))),
            'reason': f"Highest accuracy ({float(_native(run['accuracy'])):.1f}%) within time budget ({max_time_hours:.1f}h)"
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
            'model': _native(run['model']),
            'metric': metric,
            'value': float(_native(run[metric])),
            'accuracy': float(_native(run['accuracy'])),
            'co2_g': float(_native(run['co2_g'])),
            'energy_kwh': float(_native(run['energy_kwh'])),
            'training_time_sec': float(_native(run['training_time_sec']))
        }

        def predict_optimal_epochs(
            self,
            model: str,
            dataset: Optional[str] = None,
            threshold_gain_pct: float = 0.1,
            min_runs: int = 3
        ) -> Optional[Dict]:
            """
            Predict an optimal number of epochs for `model` so that further epochs
            yield less than `threshold_gain_pct` absolute accuracy gain per epoch.

            Uses a simple asymptotic model: accuracy(E) = A * (1 - exp(-k * E)).
            Fits k (and uses A as slightly-above-max observed accuracy) using
            least-squares on historical final accuracies vs epochs. If insufficient
            data is available, returns a heuristic based on average epochs.

            Args:
                model: Model name to analyze
                dataset: Optional dataset filter
                threshold_gain_pct: Absolute accuracy gain threshold per epoch (in percentage points)
                min_runs: Minimum historical runs required to perform fit

            Returns:
                Dict with keys `predicted_epochs`, `model`, `reason`, `fits` (diagnostic)
            """
            df = self.df
            if df.empty:
                return None

            sel = df[df['model'].str.lower() == model.lower()]
            if dataset:
                sel = sel[sel['dataset'].str.lower() == dataset.lower()] if 'dataset' in sel.columns else sel

            if len(sel) < 1:
                return None

            # If insufficient runs for robust fitting, return simple heuristic
            if len(sel) < min_runs:
                avg_epochs = int(round(sel['epochs'].mean())) if 'epochs' in sel.columns else None
                reason = 'Insufficient historical runs for curve fit; returning average epochs'
                return {
                    'model': model,
                    'dataset': dataset,
                    'predicted_epochs': avg_epochs,
                    'reason': reason,
                    'historical_runs': len(sel)
                }

            # Prepare arrays
            epochs = sel['epochs'].astype(float).values
            acc = sel['accuracy'].astype(float).values

            # Use A as slightly above observed max accuracy but not above 100
            max_acc = float(np.max(acc))
            A = min(100.0, max_acc * 1.05)

            # Avoid division by zero or log of non-positive numbers
            acc_frac = np.clip(acc / A, 0.0, 0.999)
            y = np.log(1.0 - acc_frac)

            # Fit k in y = -k * epochs  (no intercept)
            denom = np.sum(epochs ** 2)
            if denom == 0:
                return None
            k = -np.sum(y * epochs) / denom

            if k <= 0 or not np.isfinite(k):
                return None

            # Solve for epoch E where marginal gain < threshold_gain_pct
            threshold = float(threshold_gain_pct)
            # derivative dAcc/dE = A * k * exp(-kE) -> set equal to threshold
            val = (threshold / (A * k))
            if val <= 0:
                return None
            import math
            E_opt = -math.log(val) / k

            # Cap prediction to a reasonable max (e.g., 3x max observed epochs)
            max_cap = max(epochs) * 3.0
            predicted_epochs = int(min(math.ceil(E_opt), math.ceil(max_cap)))

            reason = (
                f"Fitted asymptotic model A={A:.2f}, k={k:.4f}. "
                f"Predicted epochs until per-epoch gain < {threshold_gain_pct}%: {predicted_epochs}"
            )

            return {
                'model': model,
                'dataset': dataset,
                'predicted_epochs': predicted_epochs,
                'reason': reason,
                'fits': {
                    'A': A,
                    'k': float(k),
                    'max_observed_epochs': int(max(epochs)),
                    'historical_runs': len(sel)
                }
            }

