# Getting Started Guide - Carbon-Aware ML Training Pipeline

## 📋 Prerequisites

Before starting, ensure you have:
- **Python 3.8 or higher** installed
- **pip** package manager
- **Git** (optional, for version control)
- **CUDA** (optional, for GPU training - will fall back to CPU if not available)

---

## 🚀 Step-by-Step Setup

### Step 1: Verify Python Installation

Open your terminal/command prompt and check Python version:

```bash
python --version
# Should show Python 3.8.x or higher

# Or try:
python3 --version
```

If Python is not installed, download from [python.org](https://www.python.org/downloads/)

---

### Step 2: Navigate to Project Directory

```bash
cd "C:\Users\aadit\Desktop\Collage Material\Major Project\Project"
```

Or if you're already in the project folder, verify you're in the right place:

```bash
# Windows PowerShell
Get-Location

# Should show: C:\Users\aadit\Desktop\Collage Material\Major Project\Project
```

---

### Step 3: Create Virtual Environment (Recommended)

Creating a virtual environment isolates project dependencies:

**Windows PowerShell:**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt when activated.

---

### Step 4: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch & Torchvision
- CodeCarbon (for carbon tracking)
- MLflow (for experiment tracking)
- Pandas, NumPy, Matplotlib, etc.

**Note:** PyTorch installation may take a few minutes. If you have CUDA, PyTorch will automatically detect it.

---

### Step 5: Verify Installation

Check if key packages are installed:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from codecarbon import EmissionsTracker; print('CodeCarbon installed')"
```

Expected output:
```
PyTorch version: 2.x.x
CUDA available: True/False
CodeCarbon installed
```

---

### Step 6: Test with Quick Training Run

Run a quick test to verify everything works:

```bash
python train.py --model cnn --dataset cifar10 --epochs 2 --batch_size 32
```

This will:
- Download CIFAR-10 dataset (first time only)
- Train a simple CNN for 2 epochs
- Track energy and CO₂
- Save results

**Expected time:** 2-5 minutes (depending on hardware)

---

## 🎯 Training Your First Model

### Basic Training Command

Train a simple CNN model:

```bash
python train.py --model cnn --dataset cifar10 --epochs 10 --batch_size 64
```

**What happens:**
1. Downloads CIFAR-10 dataset (if not already downloaded)
2. Initializes the CNN model
3. Starts carbon tracking
4. Trains for 10 epochs
5. Validates after each epoch
6. Saves best model and logs results

---

### Training with Optimizations

**With FP16 (Mixed Precision):**
```bash
python train.py --model resnet18 --dataset cifar10 --epochs 50 --batch_size 128 --fp16
```

**With Early Stopping:**
```bash
python train.py --model mobilenet_v2 --dataset cifar10 --epochs 100 --batch_size 64 --early_stop --target_acc 85
```

**With Both Optimizations:**
```bash
python train.py --model mobilenet_v2 --dataset cifar10 --epochs 100 --batch_size 64 --fp16 --early_stop --target_acc 85
```

---

### Training with Carbon Tracking by Region

Specify your region for accurate carbon calculations:

```bash
python train.py --model resnet18 --dataset cifar10 --epochs 50 --region IN-TN
```

Common region codes:
- `IN-TN` - Tamil Nadu, India
- `US-CA` - California, USA
- `GB` - United Kingdom
- `DE` - Germany

---

## 📊 Understanding the Output

### During Training

You'll see output like:
```
Using device: cuda
Loading dataset: cifar10
Loading model: resnet18

Starting training for 50 epochs...
FP16: True, Early Stop: False

Epoch 1/50
Training: 100%|████████| 782/782 [01:23<00:00,  9.34it/s]
Validating: 100%|████████| 157/157 [00:08<00:00, 18.45it/s]
Train Loss: 1.2345, Train Acc: 45.67%
Val Loss: 1.1234, Val Acc: 52.34%
```

### After Training

```
============================================================
TRAINING COMPLETE
============================================================
Best Accuracy: 87.45% (Epoch 42)
Training Time: 45.23 minutes
Energy Consumed: 0.1234 kWh
CO₂ Emitted: 56.78 g
============================================================

LEADERBOARD - Ranked by Efficiency (Accuracy/kWh)
====================================================================================================
model      accuracy  energy_kwh  co2_g    training_time_sec  epochs  accuracy_per_kwh
resnet18   87.45%    0.1234      56.78    2713.8 min         42      708.5
====================================================================================================
```

---

## 📁 Output Files

After training, check the `results/` folder:

```
results/
├── leaderboard.csv          # All training runs comparison
├── leaderboard_plot.png     # Visualization plots
├── best_model.pt            # Best model weights
├── metrics_YYYYMMDD_HHMMSS.csv  # Detailed epoch metrics
└── emissions_*.csv          # CodeCarbon emissions data
```

---

## 🔄 Complete Training Workflow

### Step 1: Train Multiple Models

Train different models to compare:

```bash
# Simple CNN (baseline)
python train.py --model cnn --dataset cifar10 --epochs 50 --batch_size 64

# ResNet18 (deeper model)
python train.py --model resnet18 --dataset cifar10 --epochs 50 --batch_size 128

# MobileNetV2 (efficient model)
python train.py --model mobilenet_v2 --dataset cifar10 --epochs 50 --batch_size 128 --fp16
```

### Step 2: Compare Results

View the leaderboard:

```bash
# The leaderboard is automatically updated and displayed after each run
# Or check the CSV file:
# Open results/leaderboard.csv in Excel or any CSV viewer
```

### Step 3: Analyze Plots

Check the visualization:

```bash
# Open results/leaderboard_plot.png
# Shows:
# - Accuracy vs CO₂ Emissions
# - Accuracy vs Energy Consumption
# - Model Efficiency (Accuracy/kWh)
# - Training Time vs CO₂
```

---

## 🎓 Example Training Scenarios

### Scenario 1: Quick Test (5 minutes)

```bash
python train.py --model cnn --dataset cifar10 --epochs 5 --batch_size 64
```

### Scenario 2: Full Training (30-60 minutes)

```bash
python train.py --model resnet18 --dataset cifar10 --epochs 100 --batch_size 128 --fp16
```

### Scenario 3: Optimized Training (20-40 minutes)

```bash
python train.py --model mobilenet_v2 --dataset cifar10 --epochs 100 --batch_size 128 --fp16 --early_stop --target_acc 85
```

### Scenario 4: Carbon-Efficient Training

```bash
# Train with early stopping to minimize unnecessary epochs
python train.py --model mobilenet_v2 --dataset cifar10 --epochs 100 --batch_size 128 --fp16 --early_stop --target_acc 80 --region IN-TN
```

---

## ⚙️ Using Configuration File

Instead of CLI arguments, you can use the config file:

1. Edit `configs/base.yaml`:
```yaml
model: resnet18
dataset: cifar10
epochs: 50
batch_size: 128
fp16: true
early_stop: true
target_acc: 85
```

2. Run with config:
```bash
python train.py --config configs/base.yaml
```

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size
```bash
python train.py --model resnet18 --batch_size 32  # Instead of 128
```

### Issue: "ModuleNotFoundError"

**Solution:** Install missing package
```bash
pip install <package_name>
# Or reinstall all:
pip install -r requirements.txt
```

### Issue: "Dataset download fails"

**Solution:** Manual download or check internet connection
- CIFAR-10 will auto-download on first run
- Ensure stable internet connection

### Issue: "CodeCarbon tracking fails"

**Solution:** This is non-critical, training will continue
- Check internet for region detection
- Or specify region manually: `--region IN-TN`

---

## 📈 Next Steps After First Training

1. **Compare Models:** Train all three models and compare in leaderboard
2. **Experiment with Optimizations:** Try FP16 and early stopping separately
3. **Analyze Results:** Check which model is most carbon-efficient
4. **Generate Report:** Document your findings
5. **Try Recommendations:** Use the recommender to find optimal configs

---

## 💡 Tips for Best Results

1. **Start Small:** Begin with 5-10 epochs to test setup
2. **Use FP16:** Saves memory and can speed up training on modern GPUs
3. **Early Stopping:** Prevents overfitting and saves energy
4. **Batch Size:** Larger batches = faster training but more memory
5. **Region Code:** Specify your region for accurate carbon calculations

---

## 📞 Quick Reference

### Common Commands

```bash
# Basic training
python train.py --model <model_name> --dataset cifar10 --epochs <n>

# With optimizations
python train.py --model <model_name> --fp16 --early_stop

# Check results
# View: results/leaderboard.csv
# Plot: results/leaderboard_plot.png
```

### Available Models
- `cnn` - Simple CNN (fastest, baseline)
- `resnet18` - ResNet18 (good accuracy)
- `mobilenet_v2` - MobileNetV2 (most efficient)

---

## ✅ Checklist

Before your first training:
- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Quick test run successful (2 epochs)
- [ ] Understand basic command structure

Ready to train:
- [ ] Choose a model (cnn, resnet18, mobilenet_v2)
- [ ] Decide on epochs (start with 10-20)
- [ ] Set batch size (64-128 for most GPUs)
- [ ] Run training command
- [ ] Check results in `results/` folder

---

**Happy Training! 🚀**

For more details, see:
- `README.md` - Project overview
- `PROJECT_DEFINITION.md` - Complete project specification

