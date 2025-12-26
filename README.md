# Carbon-Aware ML Training Pipeline

A system that trains ML models while tracking energy usage & CO₂ emissions, optimizing training to reduce footprint, and recommending greener model choices.

## 🎯 Project Goal

Build a pipeline that makes ML training aware of its environmental cost and helps reduce it.

## ✨ Features

- ✅ **Energy & CO₂ Tracking**: Real-time monitoring using CodeCarbon
- ✅ **Model Comparison**: Leaderboard ranking models by accuracy/kWh
- ✅ **Optimizations**: FP16 mixed precision, early stopping
- ✅ **Recommendations**: Suggest optimal model/config for target accuracy
- ✅ **Comprehensive Logging**: MLflow/W&B integration
- ✅ **Visualizations**: Accuracy vs CO₂, time vs energy plots
- ✅ **Web Dashboard**: Beautiful web interface for training and monitoring (NEW!)

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

## 🚀 Quick Start

### Step 1: Verify Setup

```bash
# Check if everything is installed correctly
python verify_setup.py
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 3: Quick Test Run

```bash
# Run a quick 2-epoch test to verify everything works
python train.py --model cnn --dataset cifar10 --epochs 2 --batch_size 32
```

### Step 4: Start Training

**Option A: Command Line Interface**
```bash
# Basic training
python train.py --model resnet18 --dataset cifar10 --epochs 50 --batch_size 64

# With optimizations
python train.py --model resnet18 --dataset cifar10 --epochs 50 --batch_size 64 --fp16 --early_stop
```

**Option B: Web Dashboard (Recommended)**
```bash
# Start the web interface
cd web
python run.py

# Or use the startup script
# Windows:
start_web.bat
# Linux/Mac:
./start_web.sh
```

Then open your browser to `http://localhost:5000` for an interactive training experience!

### 📖 Detailed Instructions

For complete step-by-step guide, see **[GETTING_STARTED.md](GETTING_STARTED.md)**

For web dashboard documentation, see **[web/README.md](web/README.md)**

## 📁 Project Structure

```
Project/
│
├── src/               # Core source code
│   ├── core/         # Training core modules
│   └── models/       # Model definitions
│
├── web/               # Web dashboard (NEW!)
│   ├── app.py        # Flask application
│   ├── templates/    # HTML templates
│   └── static/       # CSS/JS files
│
├── data/              # Dataset storage
├── configs/           # Configuration files
├── results/           # Output logs, models, plots
│
├── train.py           # CLI training script
├── tracker.py         # Carbon tracking wrapper
├── optimizers.py      # Training optimizations
├── logger.py          # Experiment logging
├── leaderboard.py     # Model comparison & ranking
├── recommender.py     # Model recommendations
│
└── README.md
```

For detailed structure, see **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**

## 📊 Outputs

### Per Run
- Trained model (`.pt` file)
- Final accuracy & loss
- Training time
- Energy used (kWh)
- CO₂ emitted (gCO₂eq)
- Epoch-wise logs

### Aggregated
- Leaderboard (CSV + plots)
- Accuracy vs CO₂ visualizations
- Optimal epoch recommendations
- Best model/config suggestions

## 🎓 Use Cases

- **Students**: Training models on Colab/laptops
- **Researchers**: Benchmarking model efficiency
- **Developers**: Optimizing GPU resource usage

## 📈 Success Criteria

- Track energy & CO₂ for 4-6 models
- FP16/early stopping reduce CO₂ by ≥20%
- Clear leaderboard showing trade-offs
- Actionable recommendations

## 🔧 Configuration

Edit `configs/base.yaml` or pass CLI arguments:

```yaml
model: resnet18
dataset: cifar10
epochs: 100
batch_size: 64
learning_rate: 0.01
fp16: true
early_stop: true
device: cuda
region: IN-TN
```

## 📝 License

MIT

## 🤝 Contributing

This is a research project for carbon-aware ML training.

