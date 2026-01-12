#!/usr/bin/env python3
"""
Setup script for CIFAR-10 3-Class Image Classification Project
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def setup_project():
    """Set up the project with virtual environment and dependencies."""
    print("🚀 Setting up CIFAR-10 3-Class Image Classification Project")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python version: {sys.version}")
    
    # Create virtual environment
    venv_path = Path(".venv")
    if not venv_path.exists():
        if not run_command(f"{sys.executable} -m venv .venv", "Creating virtual environment"):
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_script = ".venv\\Scripts\\activate"
        pip_path = ".venv\\Scripts\\pip"
        python_path = ".venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        activate_script = ".venv/bin/activate"
        pip_path = ".venv/bin/pip"
        python_path = ".venv/bin/python"
    
    # Upgrade pip in virtual environment
    if not run_command(f"{python_path} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{pip_path} install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Install Jupyter kernel for this virtual environment
    if not run_command(f"{python_path} -m ipykernel install --user --name=cifar10-classifier --display-name='CIFAR-10 Classifier'", "Installing Jupyter kernel"):
        print("⚠️  Jupyter kernel installation failed, but continuing...")
    else:
        print("✅ Jupyter kernel installed for virtual environment")
    
    # Create necessary directories
    directories = ["models", "cifar10_data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print("\n🎉 Project setup completed successfully!")
    print("=" * 60)
    print("📋 Next steps:")
    print(f"1. Activate virtual environment:")
    if os.name == 'nt':
        print(f"   {activate_script}")
    else:
        print(f"   source {activate_script}")
    print("2. Train the 3-class model:")
    print("   python training/cifar10_train_3class.py")
    print("3. Start Jupyter Lab (optional):")
    print("   jupyter lab")
    print("4. Start the API server:")
    print("   python api/main.py")
    print("5. Open your browser to:")
    print("   http://localhost:8000")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = setup_project()
    sys.exit(0 if success else 1)