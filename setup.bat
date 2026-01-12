@echo off
REM Windows setup script for CIFAR-10 3-Class Image Classification Project

echo 🚀 Setting up CIFAR-10 3-Class Image Classification Project
echo ============================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python is installed

REM Create virtual environment
if not exist ".venv" (
    echo 🔧 Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment and install dependencies
echo 🔧 Installing dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Install Jupyter kernel for this virtual environment
echo 🔧 Installing Jupyter kernel...
python -m ipykernel install --user --name=cifar10-classifier --display-name="CIFAR-10 Classifier"
if errorlevel 1 (
    echo ⚠️  Jupyter kernel installation failed, but continuing...
) else (
    echo ✅ Jupyter kernel installed for virtual environment
)

REM Create necessary directories
if not exist "models" mkdir models
if not exist "cifar10_data" mkdir cifar10_data

echo.
echo 🎉 Project setup completed successfully!
echo ============================================================
echo 📋 Next steps:
echo 1. Activate virtual environment: .venv\Scripts\activate.bat
echo 2. Train the 3-class model: python training/cifar10_train_3class.py
echo 3. Start Jupyter Lab (optional): jupyter lab
echo 4. Start the API server: python api/main.py
echo 5. Open your browser to: http://localhost:8000
echo ============================================================
pause