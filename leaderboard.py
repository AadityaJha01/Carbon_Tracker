"""
Leaderboard generator for comparing model efficiency
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from datetime import datetime


class Leaderboard:
    """
    Generate and maintain leaderboard of model runs ranked by efficiency.
    """
    
    def __init__(self, csv_path: str = "./results/leaderboard.csv"):
        """
        Initialize leaderboard.
        
        Args:
            csv_path: Path to CSV file storing leaderboard data
        """
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Initialize or load existing leaderboard
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
        else:
            self.df = pd.DataFrame()
    
    def add_run(
        self,
        model_name: str,
        accuracy: float,
        energy_kwh: float,
        co2_g: float,
        training_time_sec: float,
        epochs: int,
        batch_size: int,
        fp16: bool,
        early_stop: bool,
        **kwargs
    ):
        """
        Add a training run to the leaderboard.
        
        Args:
            model_name: Name of the model
            accuracy: Final validation accuracy
            energy_kwh: Energy consumed in kWh
            co2_g: CO₂ emitted in grams
            training_time_sec: Training time in seconds
            epochs: Number of epochs trained
            batch_size: Batch size used
            fp16: Whether FP16 was used
            early_stop: Whether early stopping was used
            **kwargs: Additional metadata
        """
        run_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': model_name,
            'accuracy': accuracy,
            'energy_kwh': energy_kwh,
            'co2_g': co2_g,
            'training_time_sec': training_time_sec,
            'epochs': epochs,
            'batch_size': batch_size,
            'fp16': fp16,
            'early_stop': early_stop,
            **kwargs
        }
        
        # Ensure numeric types
        try:
            energy_kwh = float(energy_kwh)
        except Exception:
            energy_kwh = 0.0
        try:
            co2_g = float(co2_g)
        except Exception:
            co2_g = 0.0
        try:
            training_time_sec = float(training_time_sec)
        except Exception:
            training_time_sec = 0.0

        # Calculate efficiency metrics. If energy or CO2 are too small, mark as None
        EPS = 1e-6
        run_data['accuracy_per_kwh'] = (accuracy / energy_kwh) if energy_kwh > EPS else None
        run_data['accuracy_per_co2'] = (accuracy / co2_g) if co2_g > EPS else None
        run_data['accuracy_per_hour'] = (accuracy / (training_time_sec / 3600.0)) if training_time_sec > EPS else None
        
        # Add to dataframe
        new_row = pd.DataFrame([run_data])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        # Save to CSV
        self.save()
    
    def get_leaderboard(
        self,
        sort_by: str = 'accuracy_per_kwh',
        ascending: bool = False,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get leaderboard sorted by specified metric.
        
        Args:
            sort_by: Column to sort by (default: 'accuracy_per_kwh')
            ascending: Sort order
            top_n: Return only top N results
        
        Returns:
            Sorted dataframe
        """
        if self.df.empty:
            return pd.DataFrame()
        
        sorted_df = self.df.sort_values(by=sort_by, ascending=ascending)
        
        if top_n:
            sorted_df = sorted_df.head(top_n)
        
        return sorted_df
    
    def save(self):
        """Save leaderboard to CSV"""
        self.df.to_csv(self.csv_path, index=False)
    
    def plot_comparison(
        self,
        output_path: Optional[str] = None,
        figsize: tuple = (15, 10)
    ):
        """
        Generate comparison plots.
        
        Args:
            output_path: Path to save plot (default: ./results/leaderboard_plot.png)
            figsize: Figure size
        """
        if self.df.empty:
            print("No data to plot")
            return
        
        if output_path is None:
            output_path = os.path.join(os.path.dirname(self.csv_path), "leaderboard_plot.png")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Accuracy vs CO₂
        ax1 = axes[0, 0]
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            ax1.scatter(model_data['co2_g'], model_data['accuracy'], 
                       label=model, alpha=0.7, s=100)
        ax1.set_xlabel('CO₂ Emissions (g)')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy vs CO₂ Emissions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy vs Energy
        ax2 = axes[0, 1]
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            ax2.scatter(model_data['energy_kwh'], model_data['accuracy'],
                       label=model, alpha=0.7, s=100)
        ax2.set_xlabel('Energy (kWh)')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy vs Energy Consumption')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Accuracy per kWh (efficiency)
        ax3 = axes[1, 0]
        efficiency_df = self.df.groupby('model')['accuracy_per_kwh'].mean().sort_values(ascending=False)
        efficiency_df.plot(kind='bar', ax=ax3, color='green', alpha=0.7)
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Accuracy per kWh')
        ax3.set_title('Model Efficiency (Accuracy/kWh)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Training time vs CO₂
        ax4 = axes[1, 1]
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            ax4.scatter(model_data['training_time_sec'] / 3600, model_data['co2_g'],
                       label=model, alpha=0.7, s=100)
        ax4.set_xlabel('Training Time (hours)')
        ax4.set_ylabel('CO₂ Emissions (g)')
        ax4.set_title('Training Time vs CO₂ Emissions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Leaderboard plot saved to {output_path}")
        plt.close()
    
    def print_table(self, sort_by: str = 'accuracy_per_kwh', top_n: int = 10):
        """Print formatted leaderboard table"""
        leaderboard = self.get_leaderboard(sort_by=sort_by, top_n=top_n)
        
        if leaderboard.empty:
            print("No runs in leaderboard yet")
            return
        
        print("\n" + "="*100)
        print("LEADERBOARD - Ranked by Efficiency (Accuracy/kWh)")
        print("="*100)
        
        # Select columns to display
        display_cols = ['model', 'accuracy', 'energy_kwh', 'co2_g', 
                       'training_time_sec', 'epochs', 'accuracy_per_kwh']
        
        # Format for display
        display_df = leaderboard[display_cols].copy()
        display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.2f}%")
        display_df['energy_kwh'] = display_df['energy_kwh'].apply(lambda x: f"{x:.4f}")
        display_df['co2_g'] = display_df['co2_g'].apply(lambda x: f"{x:.2f}")
        def fmt_time(x):
            try:
                x = float(x)
            except Exception:
                return "-"
            if x < 60:
                return f"{int(x)} s"
            if x < 3600:
                return f"{x/60:.1f} min"
            return f"{int(x/3600)} h {(x%3600)/60:.0f} m"

        display_df['training_time_sec'] = display_df['training_time_sec'].apply(fmt_time)

        def fmt_eff(x):
            if x is None or (isinstance(x, float) and (pd.isna(x))):
                return "N/A"
            try:
                return f"{float(x):.1f}"
            except Exception:
                return str(x)

        display_df['accuracy_per_kwh'] = display_df['accuracy_per_kwh'].apply(fmt_eff)
        
        print(display_df.to_string(index=False))
        print("="*100 + "\n")

