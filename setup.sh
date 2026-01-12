#!/bin/bash
# Unix/Linux/macOS setup script for CIFAR-10 3-Class Image Classification Project

echo "🚀 Setting up CIFAR-10 3-Class Image Classification Project"
echo "============================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8+ from your package manager or https://python.org"
    exit 1
fi

echo "✅ Python 3 is installed: $(python3 --version)"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
echo "🔧 Installing dependencies..."
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Install Jupyter kernel for this virtual environment
echo "🔧 Installing Jupyter kernel..."
python -m ipykernel install --user --name=cifar10-classifier --display-name="CIFAR-10 Classifier"
if [ $? -eq 0 ]; then
    echo "✅ Jupyter kernel installed for virtual environment"
else
    echo "⚠️  Jupyter kernel installation failed, but continuing..."
fi

# Create necessary directories
mkdir -p models cifar10_data

echo ""
echo "🎉 Project setup completed successfully!"
echo "============================================================"
echo "📋 Next steps:"
echo "1. Activate virtual environment: source .venv/bin/activate"
echo "2. Train the 3-class model: python training/cifar10_train_3class.py"
echo "3. Start Jupyter Lab (optional): jupyter lab"
echo "4. Start the API server: python api/main.py"
echo "5. Open your browser to: http://localhost:8000"
echo "============================================================"