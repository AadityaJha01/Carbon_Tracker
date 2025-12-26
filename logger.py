"""
Experiment logging utilities for MLflow and Weights & Biases
"""

import os
import json
import pandas as pd
from typing import Dict, Optional
from datetime import datetime


class ExperimentLogger:
    """
    Unified logger for MLflow and W&B integration.
    """
    
    def __init__(
        self,
        experiment_name: str = "carbon-ml-training",
        use_mlflow: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        log_dir: str = "./results"
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            use_mlflow: Whether to use MLflow
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            log_dir: Directory for local logs
        """
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize MLflow
        self.mlflow_run = None
        if use_mlflow:
            try:
                import mlflow
                mlflow.set_experiment(experiment_name)
                self.mlflow_run = mlflow.start_run()
            except ImportError:
                print("Warning: MLflow not installed, skipping MLflow logging")
                self.use_mlflow = False
        
        # Initialize W&B
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project or experiment_name)
                self.wandb_run = wandb.run
            except ImportError:
                print("Warning: W&B not installed, skipping W&B logging")
                self.use_wandb = False
        
        # Local log storage
        self.metrics_history = []
    
    def log_params(self, params: Dict):
        """Log hyperparameters"""
        if self.use_mlflow:
            import mlflow
            mlflow.log_params(params)
        
        if self.use_wandb:
            import wandb
            wandb.config.update(params)
    
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics"""
        if self.use_mlflow:
            import mlflow
            mlflow.log_metrics(metrics, step=step)
        
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)
        
        # Store locally
        metrics_with_step = {**metrics, 'step': step or len(self.metrics_history)}
        self.metrics_history.append(metrics_with_step)
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model artifact"""
        if self.use_mlflow:
            import mlflow.pytorch
            mlflow.pytorch.log_model(model_path, model_name)
        
        if self.use_wandb:
            import wandb
            wandb.log_model(model_path, name=model_name)
    
    def log_artifact(self, file_path: str):
        """Log file artifact"""
        if self.use_mlflow:
            import mlflow
            mlflow.log_artifact(file_path)
        
        if self.use_wandb:
            import wandb
            wandb.log_artifact(file_path)
    
    def end_run(self):
        """End the logging run"""
        if self.use_mlflow and self.mlflow_run:
            import mlflow
            mlflow.end_run()
        
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()
        
        # Save local metrics to CSV
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            csv_path = os.path.join(self.log_dir, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Metrics saved to {csv_path}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.end_run()

