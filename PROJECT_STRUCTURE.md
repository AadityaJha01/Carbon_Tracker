# Project Structure

This document describes the reorganized project structure.

## Directory Layout

```
Project/
├── src/                    # Core source code
│   ├── core/              # Core training modules
│   │   ├── __init__.py
│   │   ├── trainer.py     # Main training class
│   │   └── config_loader.py
│   └── models/            # Model definitions
│       ├── __init__.py
│       ├── cnn.py
│       ├── resnet.py
│       └── mobilenet.py
│
├── web/                    # Web application
│   ├── app.py             # Flask application
│   ├── run.py             # Run script
│   ├── templates/         # HTML templates
│   │   └── index.html
│   ├── static/            # Static files (CSS, JS)
│   │   ├── style.css
│   │   └── script.js
│   └── README.md
│
├── scripts/               # Utility scripts
│   └── move_to_src.py
│
├── tests/                 # Test files
│
├── configs/               # Configuration files
│   └── base.yaml
│
├── data/                  # Dataset storage
│   └── cifar-10-batches-py/
│
├── results/               # Training results
│   ├── leaderboard.csv
│   ├── leaderboard_plot.png
│   ├── best_model.pt
│   └── emissions.csv
│
├── train.py               # CLI training script (backward compatible)
├── tracker.py             # Carbon tracking
├── logger.py              # Experiment logging
├── optimizers.py          # Training optimizations
├── leaderboard.py         # Leaderboard management
├── recommender.py         # Model recommendations
├── requirements.txt       # Python dependencies
└── README.md              # Main README
```

## Key Components

### Core Training (`src/core/`)
- **trainer.py**: Main `Trainer` class that handles the training loop
- **config_loader.py**: Configuration loading utilities

### Models (`src/models/`)
- Model definitions: SimpleCNN, ResNet18, MobileNetV2
- Factory functions for model creation

### Web Application (`web/`)
- **app.py**: Flask application with REST API endpoints
- **templates/**: HTML templates for the frontend
- **static/**: CSS and JavaScript for the UI

### Root Level Files
- **train.py**: CLI interface for training (maintains backward compatibility)
- **tracker.py**: Carbon emissions tracking using CodeCarbon
- **logger.py**: Experiment logging (MLflow, W&B)
- **optimizers.py**: FP16 training and early stopping
- **leaderboard.py**: Leaderboard management and visualization
- **recommender.py**: Model recommendation system

## Usage

### Command Line (Original)
```bash
python train.py --model resnet18 --dataset cifar10 --epochs 50 --region IN-TN
```

### Web Interface (New)
```bash
cd web
python run.py
# Open http://localhost:5000
```

### Programmatic (New)
```python
from src.core.trainer import Trainer
from src.core.config_loader import load_config

config = load_config('configs/base.yaml')
trainer = Trainer(config)
results = trainer.train()
```

## Migration Notes

The project maintains backward compatibility:
- All original CLI scripts still work
- Original imports still work (files remain in root)
- New structure is optional but recommended

