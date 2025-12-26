"""
Main training script for Carbon-Aware ML Training Pipeline
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import json
from datetime import datetime

from models import SimpleCNN, get_resnet18, get_mobilenet_v2
from tracker import CarbonTracker
from optimizers import FP16Trainer, EarlyStopping
from logger import ExperimentLogger
from leaderboard import Leaderboard


def load_config(config_path: str = None, args: argparse.Namespace = None):
    """Load configuration from YAML file and override with CLI args"""
    config = {}
    
    # Load base config
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override with CLI args
    if args:
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
    
    return config


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


def train_epoch(model, train_loader, criterion, optimizer, device, fp16_trainer):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
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


def main():
    parser = argparse.ArgumentParser(description='Carbon-Aware ML Training Pipeline')
    
    # Model & Dataset
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['cnn', 'resnet18', 'mobilenet_v2'],
                       help='Model to train')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset to use')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    
    # Optimizations
    parser.add_argument('--fp16', action='store_true', help='Use FP16 mixed precision')
    parser.add_argument('--early_stop', action='store_true', help='Use early stopping')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--target_acc', type=float, default=None,
                       help='Target accuracy for early stopping')
    
    # Device & Config
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--region', type=str, default=None,
                       help='Region code for carbon tracking (e.g., IN-TN)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./results', help='Log directory')
    parser.add_argument('--use_mlflow', action='store_true', help='Use MLflow')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config, args)
    
    # Set random seed
    torch.manual_seed(config.get('seed', 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.get('seed', 42))
    
    # Setup device
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset: {config['dataset']}")
    train_loader, test_loader, num_classes = get_dataset(
        config['dataset'],
        config['batch_size'],
        config.get('num_workers', 4)
    )
    
    # Load model
    print(f"Loading model: {config['model']}")
    model = get_model(config['model'], num_classes)
    model = model.to(device)
    
    # Setup optimizer
    if config.get('optimizer', 'adam') == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0001)
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 0.0001)
        )
    
    criterion = nn.CrossEntropyLoss()
    
    # Setup optimizations
    fp16_trainer = FP16Trainer(enabled=config.get('fp16', False))
    early_stopping = EarlyStopping(
        patience=config.get('early_stop_patience', 10),
        target_acc=config.get('target_acc')
    ) if config.get('early_stop', False) else None
    
    # Setup logging
    experiment_name = f"{config['model']}_{config['dataset']}"
    logger = ExperimentLogger(
        experiment_name=experiment_name,
        use_mlflow=config.get('use_mlflow', False),
        use_wandb=config.get('use_wandb', False),
        log_dir=config.get('log_dir', './results')
    )
    
    # Log hyperparameters
    logger.log_params({
        'model': config['model'],
        'dataset': config['dataset'],
        'epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'learning_rate': config['learning_rate'],
        'optimizer': config.get('optimizer', 'adam'),
        'fp16': config.get('fp16', False),
        'early_stop': config.get('early_stop', False),
        'device': str(device)
    })
    
    # Setup carbon tracker
    tracker = CarbonTracker(
        output_dir=config.get('log_dir', './results'),
        region=config.get('region')
    )
    
    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    print(f"FP16: {fp16_trainer.is_enabled()}, Early Stop: {early_stopping is not None}\n")
    
    best_acc = 0.0
    best_epoch = 0
    training_start_time = datetime.now()
    
    tracker.start()
    
    try:
        for epoch in range(config['epochs']):
            print(f"\nEpoch {epoch+1}/{config['epochs']}")
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, fp16_trainer
            )
            
            # Validate
            val_loss, val_acc = validate(
                model, test_loader, criterion, device, fp16_trainer
            )
            
            # Log metrics
            logger.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, step=epoch)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                model_path = os.path.join(config.get('log_dir', './results'), 'best_model.pt')
                torch.save(model.state_dict(), model_path)
            
            # Early stopping
            if early_stopping:
                if early_stopping(val_acc, epoch):
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best accuracy: {best_acc:.2f}% at epoch {best_epoch+1}")
                    break
    
    finally:
        # Stop tracking
        emissions_data = tracker.stop()
        training_end_time = datetime.now()
        training_time = (training_end_time - training_start_time).total_seconds()
    
    # Final results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Accuracy: {best_acc:.2f}% (Epoch {best_epoch+1})")
    print(f"Training Time: {training_time/60:.2f} minutes")
    print(f"Energy Consumed: {emissions_data.get('energy_consumed_kwh', 0):.4f} kWh")
    print(f"CO2 Emitted: {emissions_data.get('emissions_gCO2eq', 0):.2f} g")
    print("="*60)
    
    # Update leaderboard
    leaderboard = Leaderboard(
        csv_path=os.path.join(config.get('log_dir', './results'), 'leaderboard.csv')
    )
    leaderboard.add_run(
        model_name=config['model'],
        accuracy=best_acc,
        energy_kwh=emissions_data.get('energy_consumed_kwh', 0),
        co2_g=emissions_data.get('emissions_gCO2eq', 0),
        training_time_sec=training_time,
        epochs=best_epoch + 1,
        batch_size=config['batch_size'],
        fp16=config.get('fp16', False),
        early_stop=config.get('early_stop', False)
    )
    
    # Print leaderboard
    leaderboard.print_table()
    
    # Generate plots
    leaderboard.plot_comparison()
    
    # End logging
    logger.end_run()
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()

