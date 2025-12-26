"""
Demo trainer for fast simulated training runs used for demos and UI testing.

This avoids heavy PyTorch training and creates realistic-looking progress updates
and a leaderboard entry at the end.
"""
import time
import random
import os
from datetime import datetime
from typing import Callable, Dict, Optional

from leaderboard import Leaderboard


class DemoTrainer:
    def __init__(self, config: Dict, progress_callback: Optional[Callable] = None):
        self.config = config
        self.progress_callback = progress_callback
        self.epochs = int(config.get('epochs', 10))
        self.model = config.get('model', 'demo_model')
        self.batch_size = int(config.get('batch_size', 32))

    def train(self) -> Dict:
        start_time = datetime.now()
        best_acc = 0.0
        best_epoch = 0

        # Simulate realistic per-epoch duration (seconds). For demo, use 8-20s per epoch
        per_epoch_seconds = int(self.config.get('demo_epoch_seconds', random.randint(8, 20)))
        for epoch in range(self.epochs):
            # Simulate some work
            time.sleep(per_epoch_seconds)

            # Create fake metrics that improve slowly
            train_loss = max(0.01, 2.0 / (epoch + 1) + random.random() * 0.1)
            val_loss = train_loss + random.random() * 0.05
            train_acc = min(100.0, 40 + epoch * (40.0 / max(1, self.epochs)) + random.random() * 2)
            val_acc = train_acc - random.random() * 3

            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }

            # report progress
            if self.progress_callback:
                try:
                    self.progress_callback(epoch, self.epochs, metrics)
                except Exception:
                    pass

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch

        end_time = datetime.now()
        training_time_sec = (end_time - start_time).total_seconds()

        # Simulate energy consumption based on device and duration
        device = (self.config.get('device') or 'cuda').lower()
        # Approximate power draw (Watts): CPU lighter, CUDA/GPU heavier
        if 'cpu' in device:
            power_watts = random.randint(50, 120)
        else:
            # GPU: moderate power for demo
            power_watts = random.randint(150, 300)

        # energy in kWh = power (W) * time (s) / 3600 / 1000
        energy_kwh = round((power_watts * training_time_sec) / 3600.0 / 1000.0, 6)
        # Emission factor (g CO2 per kWh) - demo uses a reasonable average
        emission_factor = 475
        co2_g = round(energy_kwh * emission_factor, 2)

        # Add leaderboard entry
        results_dir = self.config.get('log_dir', './results')
        os.makedirs(results_dir, exist_ok=True)
        leaderboard_path = os.path.join(results_dir, 'leaderboard.csv')

        lb = Leaderboard(csv_path=leaderboard_path)
        lb.add_run(
            model_name=self.model,
            accuracy=round(best_acc, 2),
            energy_kwh=energy_kwh,
            co2_g=co2_g,
            training_time_sec=training_time_sec,
            epochs=best_epoch + 1,
            batch_size=self.batch_size,
            fp16=self.config.get('fp16', False),
            early_stop=self.config.get('early_stop', False)
        )

        return {
            'best_accuracy': best_acc,
            'best_epoch': best_epoch + 1,
            'training_time_sec': training_time_sec,
            'energy_kwh': energy_kwh,
            'co2_g': co2_g
        }
