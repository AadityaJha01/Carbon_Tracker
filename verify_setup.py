"""
Quick verification script to check if the project is set up correctly
"""

import sys

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"[OK] Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"[FAIL] Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"[OK] {package_name} - Installed")
        return True
    except ImportError:
        print(f"[FAIL] {package_name} - NOT installed")
        return False

def check_torch():
    """Check PyTorch installation and CUDA"""
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__} - Installed")
        
        if torch.cuda.is_available():
            print(f"   [OK] CUDA available - GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"   [WARN] CUDA not available - Will use CPU")
        return True
    except ImportError:
        print(f"[FAIL] PyTorch - NOT installed")
        return False

def check_project_structure():
    """Check if project files exist"""
    import os
    
    required_files = [
        'train.py',
        'tracker.py',
        'optimizers.py',
        'logger.py',
        'leaderboard.py',
        'recommender.py',
        'models/__init__.py',
        'models/cnn.py',
        'models/resnet.py',
        'models/mobilenet.py',
        'configs/base.yaml',
        'requirements.txt'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file} - Exists")
        else:
            print(f"[FAIL] {file} - Missing")
            all_exist = False
    
    return all_exist

def main():
    print("="*60)
    print("Carbon-Aware ML Training Pipeline - Setup Verification")
    print("="*60)
    print()
    
    all_ok = True
    
    print("1. Checking Python version...")
    all_ok &= check_python_version()
    print()
    
    print("2. Checking project structure...")
    all_ok &= check_project_structure()
    print()
    
    print("3. Checking required packages...")
    all_ok &= check_package("torch")
    all_ok &= check_torch()
    all_ok &= check_package("torchvision")
    all_ok &= check_package("codecarbon")
    all_ok &= check_package("mlflow")
    all_ok &= check_package("pandas")
    all_ok &= check_package("numpy")
    all_ok &= check_package("matplotlib")
    all_ok &= check_package("yaml", "yaml")
    all_ok &= check_package("tqdm")
    print()
    
    print("="*60)
    if all_ok:
        print("[SUCCESS] All checks passed! You're ready to start training.")
        print()
        print("Next steps:")
        print("1. Run a quick test: python train.py --model cnn --epochs 2 --batch_size 32")
        print("2. See GETTING_STARTED.md for detailed instructions")
    else:
        print("[ERROR] Some checks failed. Please install missing packages:")
        print("   pip install -r requirements.txt")
    print("="*60)

if __name__ == "__main__":
    main()

