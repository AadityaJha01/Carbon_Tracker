"""
Training optimizations for carbon-aware ML training
"""

import torch
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Callable
from contextlib import contextmanager
import numpy as np


class FP16Trainer:
    """
    Mixed precision training wrapper for FP16 optimization.
    Reduces memory usage and can speed up training on modern GPUs.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize FP16 trainer.
        
        Args:
            enabled: Whether to use mixed precision training
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = GradScaler() if self.enabled else None
    
    def autocast_context(self):
        """Get autocast context for forward pass"""
        if self.enabled:
            return autocast()
        # Return a no-op context manager when FP16 is disabled
        @contextmanager
        def noop_context():
            yield
        return noop_context()
    
    def scale_loss(self, loss):
        """Scale loss for backward pass"""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer):
        """Step optimizer with scaled gradients"""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def is_enabled(self):
        """Check if FP16 is enabled"""
        return self.enabled


class EarlyStopping:
    """
    Early stopping callback to stop training when validation accuracy plateaus.
    Reduces unnecessary training epochs and saves energy.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        target_acc: Optional[float] = None,
        mode: str = 'max'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            target_acc: Target accuracy to stop at (optional)
            mode: 'max' for accuracy (higher is better), 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.target_acc = target_acc
        self.mode = mode
        
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score (accuracy or loss)
            epoch: Current epoch number
        
        Returns:
            True if training should stop, False otherwise
        """
        # Check if target accuracy reached
        if self.target_acc is not None and score >= self.target_acc:
            self.early_stop = True
            return True
        
        # Check for improvement
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best"""
        if self.mode == 'max':
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta
    
    def get_best_score(self) -> Optional[float]:
        """Get best score seen so far"""
        return self.best_score
    
    def get_best_epoch(self) -> int:
        """Get epoch with best score"""
        return self.best_epoch

