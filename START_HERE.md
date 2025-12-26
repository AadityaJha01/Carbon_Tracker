# 🚀 START HERE - Quick Setup Guide

## ⚡ 5-Minute Quick Start

### Step 1: Verify Your Setup (30 seconds)
```bash
python verify_setup.py
```

This checks if everything is ready. You'll see what's missing.

---

### Step 2: Install Dependencies (2-3 minutes)
```bash
pip install -r requirements.txt
```

**What this installs:**
- PyTorch (ML framework)
- CodeCarbon (carbon tracking)
- MLflow (experiment tracking)
- Other required packages

**Note:** This may take 2-5 minutes depending on your internet speed.

---

### Step 3: Quick Test Run (2-5 minutes)
```bash
python train.py --model cnn --dataset cifar10 --epochs 2 --batch_size 32
```

**What happens:**
- Downloads CIFAR-10 dataset (first time only, ~170 MB)
- Trains a simple CNN for 2 epochs
- Tracks energy and CO₂ emissions
- Saves results to `results/` folder

**Expected output:**
```
Using device: cpu
Loading dataset: cifar10
Loading model: cnn

Starting training for 2 epochs...
...
TRAINING COMPLETE
Best Accuracy: XX.XX%
Energy Consumed: X.XXXX kWh
CO₂ Emitted: XX.XX g
```

---

### Step 4: Train Your First Real Model (10-30 minutes)

**Option A: Simple CNN (Fastest)**
```bash
python train.py --model cnn --dataset cifar10 --epochs 20 --batch_size 64
```

**Option B: ResNet18 (Better Accuracy)**
```bash
python train.py --model resnet18 --dataset cifar10 --epochs 50 --batch_size 128
```

**Option C: MobileNetV2 with Optimizations (Most Efficient)**
```bash
python train.py --model mobilenet_v2 --dataset cifar10 --epochs 50 --batch_size 128 --fp16 --early_stop
```

---

## 📊 Check Your Results

After training completes, check:

1. **Leaderboard** - `results/leaderboard.csv`
   - Compare all your training runs
   - See which model is most carbon-efficient

2. **Visualizations** - `results/leaderboard_plot.png`
   - Accuracy vs CO₂ emissions
   - Energy consumption comparisons
   - Model efficiency rankings

3. **Saved Model** - `results/best_model.pt`
   - Best model weights from training

---

## 🎯 What to Do Next

### Compare Multiple Models

Train different models and compare:

```bash
# 1. Train CNN
python train.py --model cnn --dataset cifar10 --epochs 30 --batch_size 64

# 2. Train ResNet18
python train.py --model resnet18 --dataset cifar10 --epochs 30 --batch_size 128

# 3. Train MobileNetV2
python train.py --model mobilenet_v2 --dataset cifar10 --epochs 30 --batch_size 128 --fp16
```

Then check `results/leaderboard.csv` to see which is most efficient!

---

## 📚 Need More Help?

- **Detailed Guide:** See `GETTING_STARTED.md`
- **Project Overview:** See `README.md`
- **Complete Specs:** See `PROJECT_DEFINITION.md`
- **Quick Reference:** See `QUICK_START.txt`

---

## 🐛 Common Issues

### "CUDA out of memory"
**Fix:** Reduce batch size
```bash
python train.py --model resnet18 --batch_size 32  # Instead of 128
```

### "ModuleNotFoundError"
**Fix:** Install missing package
```bash
pip install <package_name>
# Or reinstall all:
pip install -r requirements.txt
```

### "Training is slow"
**Fix:** Use FP16 (if you have a GPU)
```bash
python train.py --model resnet18 --fp16
```

---

## ✅ Success Checklist

- [ ] Ran `python verify_setup.py` - All checks passed
- [ ] Installed dependencies: `pip install -r requirements.txt`
- [ ] Ran quick test: `python train.py --model cnn --epochs 2`
- [ ] Trained first model successfully
- [ ] Checked results in `results/` folder
- [ ] Ready to train more models!

---

## 💡 Pro Tips

1. **Start Small:** Begin with 5-10 epochs to test
2. **Use FP16:** Saves memory and speeds up training (GPU only)
3. **Early Stopping:** Prevents overfitting and saves energy
4. **Batch Size:** Larger = faster but needs more memory
5. **Region Code:** Add `--region IN-TN` for accurate carbon tracking

---

**You're all set! Happy training! 🎉**

For questions or issues, check the detailed guides in the project files.

