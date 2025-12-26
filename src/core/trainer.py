"""
Core training functionality extracted from train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, Callable

# Import from parent modules
import sys
project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, os.path.abspath(project_root))

from models import SimpleCNN, get_resnet18, get_mobilenet_v2
from tracker import CarbonTracker
from optimizers import FP16Trainer, EarlyStopping
from logger import ExperimentLogger
from leaderboard import Leaderboard


def get_model(model_name: str, num_classes: int = 10):
    """Factory function to get model"""
    model_name = model_name.lower()
    
    if model_name == 'cnn':
        return SimpleCNN(num_classes=num_classes)
    elif model_name == 'resnet18':
        return get_resnet18(num_classes=num_classes)
    elif model_name == 'mobilenet_v2':
        return get_mobilenet_v2(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_dataset(dataset_name: str, batch_size: int, num_workers: int = 4):
    """Load and prepare dataset"""
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        return train_loader, test_loader, 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def train_epoch(model, train_loader, criterion, optimizer, device, fp16_trainer, progress_callback: Optional[Callable] = None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with fp16_trainer.autocast_context():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        fp16_trainer.scale_loss(loss).backward()
        fp16_trainer.step_optimizer(optimizer)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if progress_callback:
            progress_callback(batch_idx, len(train_loader), loss.item())
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device, fp16_trainer):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with fp16_trainer.autocast_context():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


class Trainer:
    """Main training class for carbon-aware ML training"""
    
    def __init__(self, config: Dict, progress_callback: Optional[Callable] = None):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
            progress_callback: Optional callback function(epoch, total_epochs, metrics)
        """
        self.config = config
        self.progress_callback = progress_callback
        
        # Set random seed
        torch.manual_seed(config.get('seed', 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.get('seed', 42))
        
        # Setup device
        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        
        # Load dataset
        self.train_loader, self.test_loader, self.num_classes = get_dataset(
            config['dataset'],
            config['batch_size'],
            config.get('num_workers', 4)
        )
        
        # Load model
        self.model = get_model(config['model'], self.num_classes)
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        if config.get('optimizer', 'adam') == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 0.0001)
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                momentum=config.get('momentum', 0.9),
                weight_decay=config.get('weight_decay', 0.0001)
            )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizations
        self.fp16_trainer = FP16Trainer(enabled=config.get('fp16', False))
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stop_patience', 10),
            target_acc=config.get('target_acc')
        ) if config.get('early_stop', False) else None
        
        # Setup logging
        experiment_name = f"{config['model']}_{config['dataset']}"
        self.logger = ExperimentLogger(
            experiment_name=experiment_name,
            use_mlflow=config.get('use_mlflow', False),
            use_wandb=config.get('use_wandb', False),
            log_dir=config.get('log_dir', './results')
        )
        
        # Log hyperparameters
        self.logger.log_params({
            'model': config['model'],
            'dataset': config['dataset'],
            'epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'optimizer': config.get('optimizer', 'adam'),
            'fp16': config.get('fp16', False),
            'early_stop': config.get('early_stop', False),
            'device': str(self.device)
        })
        
        # Setup carbon tracker
        self.tracker = CarbonTracker(
            output_dir=config.get('log_dir', './results'),
            region=config.get('region')
        )
        
        self.best_acc = 0.0
        self.best_epoch = 0
        self.training_start_time = None
        self.training_time = 0.0
        self.emissions_data = {}
    
    def train(self) -> Dict:
        """
        Run training loop.
        
        Returns:
            Dictionary with training results
        """
        self.training_start_time = datetime.now()
        self.tracker.start()
        
        try:
            for epoch in range(self.config['epochs']):
                # Train
                train_loss, train_acc = train_epoch(
                    self.model, self.train_loader, self.criterion, 
                    self.optimizer, self.device, self.fp16_trainer
                )
                
                # Validate
                val_loss, val_acc = validate(
                    self.model, self.test_loader, self.criterion, 
                    self.device, self.fp16_trainer
                )
                
                # Log metrics
                metrics = {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
                self.logger.log_metrics(metrics, step=epoch)
                
                # Call progress callback
                if self.progress_callback:
                    self.progress_callback(epoch, self.config['epochs'], metrics)
                
                # Save best model
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self.best_epoch = epoch
                    model_path = os.path.join(self.config.get('log_dir', './results'), 'best_model.pt')
                    torch.save(self.model.state_dict(), model_path)
                
                # Early stopping
                if self.early_stopping:
                    if self.early_stopping(val_acc, epoch):
                        break
        
        finally:
            # Stop tracking
            self.emissions_data = self.tracker.stop()
            training_end_time = datetime.now()
            self.training_time = (training_end_time - self.training_start_time).total_seconds()
        
        # Update leaderboard
        leaderboard = Leaderboard(
            csv_path=os.path.join(self.config.get('log_dir', './results'), 'leaderboard.csv')
        )
        leaderboard.add_run(
            model_name=self.config['model'],
            accuracy=self.best_acc,
            energy_kwh=self.emissions_data.get('energy_consumed_kwh', 0),
            co2_g=self.emissions_data.get('emissions_gCO2eq', 0),
            training_time_sec=self.training_time,
            epochs=self.best_epoch + 1,
            batch_size=self.config['batch_size'],
            fp16=self.config.get('fp16', False),
            early_stop=self.config.get('early_stop', False)
        )
        
        # Generate plots
        leaderboard.plot_comparison()
        
        # End logging
        self.logger.end_run()
        
        return {
            'best_accuracy': self.best_acc,
            'best_epoch': self.best_epoch + 1,
            'training_time_sec': self.training_time,
            'energy_kwh': self.emissions_data.get('energy_consumed_kwh', 0),
            'co2_g': self.emissions_data.get('emissions_gCO2eq', 0),
            'emissions_data': self.emissions_data
        }

