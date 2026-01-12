# 🤖 CIFAR-10 Image Classification Web Application

A production-ready machine learning system that classifies images using a ResNet18 model trained on CIFAR-10 data. The project includes a trained model and can be easily set up with proper virtual environment isolation.

## 🎯 Overview

This application:
- **Includes a pre-trained model** ready for immediate use
- Uses ResNet18 with transfer learning from ImageNet
- Serves predictions through a FastAPI backend
- Provides a modern web interface for real-time classification
- Supports proper virtual environment setup for dependency isolation

## 📊 Classification Classes

The model classifies images into 10 CIFAR-10 categories:

| Class | Icon | Description |
|-------|------|-------------|
| **Airplane** | ✈️ | Aircraft, planes |
| **Automobile** | 🚗 | Cars, vehicles, automobiles |
| **Bird** | 🐦 | Flying birds, avian species |
| **Cat** | 🐱 | Domestic cats |
| **Deer** | 🦌 | Wild deer |
| **Dog** | 🐕 | Domestic dogs |
| **Frog** | 🐸 | Amphibians, frogs |
| **Horse** | 🐴 | Horses |
| **Ship** | 🚢 | Boats, ships, watercraft |
| **Truck** | 🚛 | Large vehicles, trucks |

## 📁 Project Structure

```
cifar10-classifier/
├── models/                 # Pre-trained models (included)
│   ├── model.pth          # PyTorch model weights ✅
│   ├── class_names.json   # Class mapping ✅
│   └── training_curves.png # Training visualization
├── notebooks/              # Jupyter notebooks for analysis
│   └── cifar10_analysis.ipynb # Model analysis and testing
├── api/                   # FastAPI backend
│   └── main.py           # API server with model loading
├── training/              # Training scripts
│   ├── cifar10_train.py  # Original 10-class training
│   └── cifar10_train_3class.py # 3-class variant
├── frontend/              # Web interface
│   └── index.html        # Frontend UI
├── cifar10_data/          # CIFAR-10 dataset (auto-downloaded)
├── requirements.txt       # Python dependencies
├── setup.py              # Cross-platform setup script
├── setup.sh              # Unix/Linux/macOS setup
├── setup.bat             # Windows setup
└── README.md             # This file
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

**Windows:**
```cmd
setup.bat
```

**Unix/Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**Cross-platform Python:**
```bash
python setup.py
```

### Option 2: Manual Setup

1. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   
   # Activate (Windows)
   .venv\Scripts\activate
   
   # Activate (Unix/Linux/macOS)
   source .venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Install Jupyter Kernel (for notebook support)**
   ```bash
   python -m ipykernel install --user --name=cifar10-classifier --display-name="CIFAR-10 Classifier"
   ```

4. **Start the API Server**
   ```bash
   python api/main.py
   ```

5. **Access the Application**
   - **Web Interface**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs
   - **Health Check**: http://localhost:8000/health
   - **Jupyter Lab**: `jupyter lab` (optional)

## ✅ Verification

The project includes a pre-trained model that loads automatically. You can verify everything works:

```bash
# Test model loading
python -c "
import sys; sys.path.append('api')
from main import ImageClassifier
classifier = ImageClassifier()
print('Model loaded:', classifier.get_model_info())
"

# Test API
python test_api.py
```

## 🎓 Training (Optional)

The project includes a pre-trained model, but you can retrain if needed:

### 10-Class CIFAR-10 Training
```bash
python training/cifar10_train.py --epochs 30 --batch-size 64
```

### 3-Class Training (automobile, bird, ship)
```bash
python training/cifar10_train_3class.py --epochs 30 --batch-size 64
```

## 🌐 API Usage

### Start FastAPI Server
```bash
# Make sure virtual environment is activated
python api/main.py
```

### Test Predictions

#### Using cURL
```bash
curl -X POST "http://localhost:8000/predict/image" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

#### Using Python Test Script
```bash
python test_api.py
```

#### Response Format
```json
{
  "success": true,
  "filename": "car_image.jpg",
  "predicted_class": "automobile",
  "confidence": 0.94,
  "top_predictions": [
    {"class": "automobile", "confidence": 0.94},
    {"class": "truck", "confidence": 0.04},
    {"class": "ship", "confidence": 0.02}
  ],
  "model_info": {
    "model_name": "ResNet18",
    "num_classes": 10,
    "device": "cpu",
    "trained_model_available": true
  },
  "inference_time": "0.023s"
}
```

## 🏗️ Model Architecture

### ResNet18 Transfer Learning
- **Base Model**: ResNet18 pre-trained on ImageNet
- **Modification**: Final layer adapted for CIFAR-10 classes
- **Input Processing**: Resize to 224×224 for inference
- **Optimization**: Adam optimizer with learning rate decay
- **Loss Function**: CrossEntropyLoss

### Performance Metrics
Expected accuracy on CIFAR-10 test set: **85-90%**

## 🔧 Virtual Environment Setup

### Why Virtual Environment?
- **Dependency Isolation**: Prevents conflicts with system packages
- **Reproducibility**: Ensures consistent dependency versions
- **Clean Development**: Easy to reset or share environment

### Manual Virtual Environment Commands
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Unix/Linux/macOS)  
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

## 🧪 Testing

### Web Interface Testing
1. Ensure virtual environment is activated
2. Start server: `python api/main.py`
3. Open http://localhost:8000 in your browser
4. Upload an image (JPG, PNG supported)
5. View real-time classification results

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Test with sample image
python test_api.py
```

## 🚀 Production Deployment

### Docker Deployment
```bash
# Build container
docker build -t cifar10-classifier .

# Run container
docker run -p 8000:8000 cifar10-classifier
```

### Production Features
- **Pre-trained Model**: No training required on deployment
- **GPU Acceleration**: Automatic CUDA detection
- **Model Caching**: Single model load at startup
- **Error Handling**: Robust error responses

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB for dependencies and model
- **GPU**: Optional (CUDA-compatible for faster inference)

### Python Dependencies
All dependencies are specified in `requirements.txt` with version constraints for reproducibility, including:
- **PyTorch & Torchvision**: Deep learning framework
- **FastAPI & Uvicorn**: Web framework and server
- **Jupyter & JupyterLab**: Notebook environment with isolated kernel
- **Scientific Libraries**: NumPy, Matplotlib, Scikit-learn
- **Development Tools**: Pytest, IPython kernel

## 🛠️ Development

### Project Features
- ✅ **Pre-trained model included** in repository
- ✅ **Virtual environment setup** with automated scripts
- ✅ **Jupyter integration** with isolated kernel
- ✅ **Dependency isolation** with pinned versions
- ✅ **Cross-platform support** (Windows, macOS, Linux)
- ✅ **Production-ready** FastAPI backend
- ✅ **Modern web interface** with drag-and-drop
- ✅ **Comprehensive testing** scripts included
- ✅ **Interactive notebooks** for model analysis

### Adding New Features
1. Activate virtual environment: `source .venv/bin/activate`
2. Make your changes
3. Test thoroughly: `python test_api.py`
4. Update requirements if needed: `pip freeze > requirements.txt`

## 🤝 Contributing

1. Fork the repository
2. Set up virtual environment: `./setup.sh` or `setup.bat`
3. Create a feature branch (`git checkout -b feature/improvement`)
4. Make your changes in the activated virtual environment
5. Test thoroughly
6. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**Built with PyTorch, FastAPI, and CIFAR-10 dataset | Ready to run with included pre-trained model**