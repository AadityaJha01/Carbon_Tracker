# Carbon-Aware ML Training Pipeline - Project Definition

## 📋 1. Project Overview

**Project Name:** Carbon-Aware ML Training Pipeline (Self-Aware Model)

**Goal:** Build a system that trains ML models while:
- Tracking energy usage & CO₂ emissions
- Optimizing training to reduce footprint
- Recommending greener model choices

**One-liner:** A pipeline that makes ML training aware of its environmental cost and helps reduce it.

---

## 👥 2. Target Users

**Primary Users:**
- Students training ML models on Colab/laptops
- Researchers benchmarking models
- Developers with limited GPU resources

**Skill Level:** Basic ML training knowledge (PyTorch), not necessarily expert in energy systems

---

## 📥 3. System Inputs

### User Inputs

| Input | Example | Description |
|-------|---------|-------------|
| Model choice | `resnet18`, `cnn`, `mobilenet_v2` | Architecture to train |
| Dataset | `cifar10` | Dataset for training |
| Epochs | `100` | Number of training epochs |
| Batch size | `64` | Batch size for training |
| Learning rate | `0.01` | Learning rate for optimizer |
| Optimizations | `--fp16`, `--early_stop` | Training optimizations |
| Device | `cuda` / `cpu` | Hardware device |
| Region (optional) | `IN-TN` | Location for carbon tracking |
| Target accuracy (optional) | `85%` | Early stopping target |
| Time budget (optional) | `30 minutes` | Maximum training time |

### CLI Example

```bash
python train.py \
  --model resnet18 \
  --dataset cifar10 \
  --epochs 100 \
  --batch_size 64 \
  --fp16 \
  --early_stop \
  --target_acc 85 \
  --region IN-TN
```

---

## ⚙️ 4. System Internals

### Runtime Process

1. **Load Dataset** - Download/prepare CIFAR-10
2. **Initialize Model** - Load selected architecture
3. **Start Carbon Tracker** - Begin energy/CO₂ monitoring
4. **Train Model** - Epoch loop with:
   - Forward pass (with FP16 if enabled)
   - Backward pass
   - Optimizer step
   - Validation
5. **Log Metrics** - Track:
   - Accuracy & loss
   - Time per epoch
   - Energy (Wh)
   - CO₂ (g)
6. **Apply Optimizations**:
   - FP16 mixed precision
   - Early stopping
7. **Stop Tracker** - End monitoring
8. **Save Results** - Model, logs, plots
9. **Update Leaderboard** - Add run to comparison
10. **Generate Recommendations** - Suggest optimal configs

---

## 📤 5. System Outputs

### Per Run Outputs

| Output | Description | Format |
|--------|-------------|--------|
| Trained model | Saved model weights | `.pt` file |
| Accuracy | Final/best validation accuracy | Percentage |
| Training time | Total training duration | Seconds/minutes |
| Energy used | Total energy consumption | kWh |
| CO₂ emitted | Total carbon emissions | gCO₂eq |
| Epoch-wise logs | Metrics per epoch | CSV/JSON |
| Curves | Training curves | PNG plots |

### Aggregated Outputs

| Output | Description | Format |
|--------|-------------|--------|
| Leaderboard | Ranked model comparison | CSV + table |
| CSV/JSON logs | All runs data | CSV/JSON files |
| Plots | Visualizations | PNG files |
| Optimal epoch | Best acc/energy trade-off | Recommendation |
| Recommendations | Best model/config suggestions | Text/JSON |

### Example Leaderboard Entry

```
Model: mobilenet_v2 | Acc: 86.3% | Energy: 0.42 kWh | CO₂: 190 g | Acc/kWh: 205
```

---

## 🎯 6. Final Product Deliverables

### Core Deliverables ✅

- [x] **Carbon-aware training pipeline** - Python project with CLI
- [x] **Leaderboard generator** - CSV + simple table/plot
- [x] **Optimization engine** - FP16 + early stopping
- [x] **Recommender (basic)** - Suggest model & epochs for target accuracy
- [x] **Documentation** - How to run, results, analysis
- [ ] **Results report** - Tables + graphs of experiments (to be generated)
- [ ] **Demo notebook** - Example runs on CIFAR-10 (to be created)

### Optional (Future)

- [ ] Simple Streamlit dashboard
- [ ] Web UI

---

## ✅ 7. Functional Requirements

- ✅ Train ML models (CNN, ResNet18, MobileNetV2)
- ✅ Track CPU/GPU energy consumption
- ✅ Estimate CO₂ emissions
- ✅ Log metrics automatically (MLflow/W&B)
- ✅ Compare multiple runs
- ✅ Rank models by carbon efficiency
- ✅ Apply FP16 optimization
- ✅ Support early stopping
- ✅ Run on Colab/laptop
- ✅ Export results (CSV/plots)

---

## 🎨 8. Non-Functional Requirements

- **Reproducible** - Fixed random seeds
- **Lightweight** - Runs on free GPUs (Colab)
- **Modular** - Easy to add new models
- **Transparent** - Clear logging and metrics
- **Open-source** - Only open-source stack

---

## 🏆 9. Success Criteria

Project is successful if:

1. ✅ Can show energy & CO₂ numbers for at least 4-6 models
2. ✅ FP16 / early stopping reduce CO₂ by ≥20%
3. ✅ Leaderboard clearly shows trade-offs
4. ✅ Can recommend: *"Use MobileNet + 40 epochs instead of ResNet + 100 epochs for similar accuracy at half the CO₂."*

---

## 🚀 10. Initial Scope (MVP - Phase 1)

### Included ✅

- Image classification only
- CIFAR-10 dataset
- Models: Simple CNN, ResNet18, MobileNetV2
- Optimizations: FP16 + early stopping
- Output: CSV + plots

### Future Phases

- NLP models
- Transformers
- Power capping
- Web UI
- More datasets (ImageNet, etc.)

---

## 📁 11. Project Structure

```
carbonml/
│
├── data/              # Dataset storage (auto-downloaded)
├── models/            # Model definitions
│   ├── __init__.py
│   ├── cnn.py         # Simple CNN
│   ├── resnet.py      # ResNet18
│   └── mobilenet.py   # MobileNetV2
│
├── train.py           # Main training script
├── tracker.py         # Carbon tracking wrapper
├── optimizers.py      # Training optimizations (FP16, EarlyStop)
├── logger.py          # Experiment logging (MLflow/W&B)
├── leaderboard.py     # Model comparison & ranking
├── recommender.py     # Model recommendations
│
├── configs/           # Configuration files
│   └── base.yaml      # Base configuration
│
├── results/           # Output logs, models, plots
│   ├── leaderboard.csv
│   └── leaderboard_plot.png
│
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
├── PROJECT_DEFINITION.md  # This file
└── .gitignore         # Git ignore rules
```

---

## 🔧 12. Technology Stack

- **ML Framework:** PyTorch 2.0+
- **Carbon Tracking:** CodeCarbon
- **Experiment Tracking:** MLflow, Weights & Biases (optional)
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Configuration:** YAML, argparse

---

## 📊 13. Key Metrics Tracked

1. **Performance Metrics:**
   - Training/validation accuracy
   - Training/validation loss
   - Best epoch

2. **Efficiency Metrics:**
   - Energy consumed (kWh)
   - CO₂ emissions (gCO₂eq)
   - Training time (seconds)
   - Accuracy per kWh
   - Accuracy per CO₂

3. **Optimization Metrics:**
   - FP16 speedup
   - Early stopping savings
   - Epoch reduction

---

## 🎓 14. Usage Examples

### Basic Training

```bash
# Train ResNet18 on CIFAR-10
python train.py --model resnet18 --dataset cifar10 --epochs 50 --batch_size 128
```

### With Optimizations

```bash
# Train with FP16 and early stopping
python train.py \
  --model mobilenet_v2 \
  --dataset cifar10 \
  --epochs 100 \
  --batch_size 64 \
  --fp16 \
  --early_stop \
  --target_acc 85
```

### With Carbon Tracking

```bash
# Train with region-specific carbon tracking
python train.py \
  --model cnn \
  --dataset cifar10 \
  --epochs 50 \
  --region IN-TN
```

---

## 📝 15. Next Steps

1. **Install dependencies:** `pip install -r requirements.txt`
2. **Run first training:** `python train.py --model cnn --epochs 5`
3. **Compare models:** Run multiple models and check leaderboard
4. **Analyze results:** Review plots and recommendations
5. **Generate report:** Document findings and carbon savings

---

## 📚 References

- CodeCarbon: https://github.com/mlco2/codecarbon
- PyTorch: https://pytorch.org/
- MLflow: https://mlflow.org/

---

**Last Updated:** December 2025

