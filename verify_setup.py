#!/usr/bin/env python3
"""
Verification script to check if the project setup is correct.
Run this after setting up the project to verify everything works.
"""

import sys
import os
from pathlib import Path
import importlib.util

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} missing: {filepath}")
        return False

def check_import(module_name, description):
    """Check if a module can be imported."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            print(f"✅ {description} available")
            return True
        else:
            print(f"❌ {description} not available")
            return False
    except ImportError:
        print(f"❌ {description} import failed")
        return False

def check_model_loading():
    """Check if the model can be loaded."""
    try:
        sys.path.append('api')
        from main import ImageClassifier
        classifier = ImageClassifier()
        model_info = classifier.get_model_info()
        
        if model_info['trained_model_available']:
            print(f"✅ Model loaded successfully: {model_info['model_name']}")
            print(f"   Classes: {model_info['num_classes']}")
            print(f"   Device: {model_info['device']}")
            return True
        else:
            print("❌ Model not available")
            return False
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("🔍 CIFAR-10 Project Setup Verification")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 0
    
    # Check essential files
    essential_files = [
        ("models/model.pth", "Trained model file"),
        ("models/class_names.json", "Class names file"),
        ("api/main.py", "API server"),
        ("frontend/index.html", "Web interface"),
        ("requirements.txt", "Requirements file"),
        ("notebooks/cifar10_analysis.ipynb", "Analysis notebook")
    ]
    
    print("\n📁 File Structure Check:")
    for filepath, description in essential_files:
        if check_file_exists(filepath, description):
            checks_passed += 1
        total_checks += 1
    
    # Check Python dependencies
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "Torchvision"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("jupyter", "Jupyter"),
        ("ipykernel", "IPython Kernel")
    ]
    
    print("\n📦 Dependencies Check:")
    for module, description in dependencies:
        if check_import(module, description):
            checks_passed += 1
        total_checks += 1
    
    # Check model loading
    print("\n🤖 Model Loading Check:")
    if check_model_loading():
        checks_passed += 1
    total_checks += 1
    
    # Check virtual environment
    print("\n🔧 Environment Check:")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
        checks_passed += 1
    else:
        print("⚠️  Not running in virtual environment (recommended but not required)")
    total_checks += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Verification Summary: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 All checks passed! Project is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Start the API server: python api/main.py")
        print("2. Open browser to: http://localhost:8000")
        print("3. Try Jupyter Lab: jupyter lab")
        return True
    else:
        print("❌ Some checks failed. Please review the setup.")
        print("\n🔧 Try running the setup script:")
        print("- Windows: setup.bat")
        print("- Unix/Linux/macOS: ./setup.sh")
        print("- Cross-platform: python setup.py")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)