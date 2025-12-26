"""
Test all imports to ensure no errors
"""

print("Testing all imports...")
print("="*60)

try:
    import torch
    print("[OK] torch")
except ImportError as e:
    print(f"[FAIL] torch: {e}")

try:
    import torchvision
    print("[OK] torchvision")
except ImportError as e:
    print(f"[FAIL] torchvision: {e}")

try:
    from codecarbon import EmissionsTracker
    print("[OK] codecarbon")
except ImportError as e:
    print(f"[FAIL] codecarbon: {e}")

try:
    import mlflow
    print("[OK] mlflow")
except ImportError as e:
    print(f"[FAIL] mlflow: {e}")

try:
    import pandas
    print("[OK] pandas")
except ImportError as e:
    print(f"[FAIL] pandas: {e}")

try:
    import numpy
    print("[OK] numpy")
except ImportError as e:
    print(f"[FAIL] numpy: {e}")

try:
    import matplotlib
    print("[OK] matplotlib")
except ImportError as e:
    print(f"[FAIL] matplotlib: {e}")

try:
    import seaborn
    print("[OK] seaborn")
except ImportError as e:
    print(f"[FAIL] seaborn: {e}")

try:
    import yaml
    print("[OK] yaml")
except ImportError as e:
    print(f"[FAIL] yaml: {e}")

try:
    import tqdm
    print("[OK] tqdm")
except ImportError as e:
    print(f"[FAIL] tqdm: {e}")

try:
    import sklearn
    print("[OK] scikit-learn")
except ImportError as e:
    print(f"[FAIL] scikit-learn: {e}")

# Test project imports
print("\nTesting project modules...")
print("="*60)

try:
    from models import SimpleCNN, get_resnet18, get_mobilenet_v2
    print("[OK] models")
except ImportError as e:
    print(f"[FAIL] models: {e}")

try:
    from tracker import CarbonTracker
    print("[OK] tracker")
except ImportError as e:
    print(f"[FAIL] tracker: {e}")

try:
    from optimizers import FP16Trainer, EarlyStopping
    print("[OK] optimizers")
except ImportError as e:
    print(f"[FAIL] optimizers: {e}")

try:
    from logger import ExperimentLogger
    print("[OK] logger")
except ImportError as e:
    print(f"[FAIL] logger: {e}")

try:
    from leaderboard import Leaderboard
    print("[OK] leaderboard")
except ImportError as e:
    print(f"[FAIL] leaderboard: {e}")

try:
    from recommender import ModelRecommender
    print("[OK] recommender")
except ImportError as e:
    print(f"[FAIL] recommender: {e}")

print("\n" + "="*60)
print("Import test complete!")

