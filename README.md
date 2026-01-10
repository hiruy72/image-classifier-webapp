# 🤖 CIFAR-10 Image Classification Web Application

A production-ready machine learning system that classifies images into **3 specialized categories** using a ResNet18 model trained on CIFAR-10 data.

## 🎯 Overview

This application:
- Classifies images into automobiles, birds, and ships
- Uses ResNet18 with transfer learning from ImageNet
- Serves predictions through a FastAPI backend
- Provides a modern web interface for real-time classification
- Automatically downloads and processes CIFAR-10 dataset

## 📊 Classification Classes

The model predicts one of 3 categories:

| Class | Icon | Description |
|-------|------|-------------|
| **Automobile** | 🚗 | Cars, vehicles, automobiles |
| **Bird** | 🐦 | Flying birds, avian species |
| **Ship** | 🚢 | Boats, ships, watercraft |

## 📁 Project Structure

```
cifar10-classifier/
├── cifar10_data/           # CIFAR-10 dataset (auto-downloaded)
│   └── cifar-10-batches-py/
├── models/                 # Trained models
│   ├── model.pth          # PyTorch model weights
│   ├── class_names.json   # Class mapping
│   └── training_curves.png # Training visualization
├── api/                   # FastAPI backend
│   └── main.py           # API server
├── training/              # Training scripts
│   └── cifar10_train.py  # Model training
├── frontend/              # Web interface
│   └── index.html        # Frontend UI
├── test_api.py           # API testing
├── requirements.txt      # Dependencies
├── Dockerfile           # Container setup
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
The training script automatically downloads CIFAR-10 data:
```bash
python training/cifar10_train.py
```

### 3. Start the API Server
```bash
python api/main.py
```

### 4. Access the Application
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🎓 Training Process

### Automatic CIFAR-10 Setup
1. **Data Download**: Automatically downloads CIFAR-10 dataset (170MB)
2. **Class Filtering**: Extracts only automobile, bird, and ship classes
3. **Data Preprocessing**: Normalizes and augments images
4. **Model Training**: Fine-tunes ResNet18 with transfer learning
5. **Model Export**: Saves trained model and class mappings

### Training Configuration
- **Dataset**: CIFAR-10 (filtered to 3 classes)
- **Model**: ResNet18 pre-trained on ImageNet
- **Input Size**: 32×32×3 (CIFAR-10 native resolution)
- **Batch Size**: 128
- **Learning Rate**: 0.001 with scheduler
- **Epochs**: 50 with early stopping
- **Augmentation**: Random horizontal flip, rotation, normalization

### Manual Training Options
```bash
cd training
python cifar10_train.py --epochs 50 --batch-size 128 --lr 0.001
```

## 🌐 API Usage

### Start FastAPI Server
```bash
python api/main.py
# Server runs on http://localhost:8000
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
    {"class": "ship", "confidence": 0.04},
    {"class": "bird", "confidence": 0.02}
  ],
  "model_info": {
    "model_name": "ResNet18",
    "num_classes": 3,
    "device": "cuda",
    "trained_model_available": true
  },
  "inference_time": "0.023s"
}
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/image` | POST | Upload image for classification |
| `/health` | GET | Service health status |
| `/docs` | GET | Interactive API documentation |
| `/` | GET | Web interface |

## 🏗️ Model Architecture

### ResNet18 Transfer Learning
- **Base Model**: ResNet18 pre-trained on ImageNet
- **Modification**: Final layer adapted for 3-class output
- **Input Processing**: Resize to 224×224 for inference
- **Optimization**: Adam optimizer with learning rate decay
- **Loss Function**: CrossEntropyLoss

### Performance Metrics
Expected accuracy on CIFAR-10 test set:
- **Overall Accuracy**: 85-90%
- **Automobile**: 88-92%
- **Bird**: 82-87%
- **Ship**: 85-90%

## 🧪 Testing

### Web Interface Testing
1. Open http://localhost:8000 in your browser
2. Upload an image (JPG, PNG supported)
3. View real-time classification results

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Test with sample image
python test_api.py

# Manual image test
curl -X POST "http://localhost:8000/predict/image" \
     -F "file=@sample_image.jpg"
```

## 🚀 Production Deployment

### Docker Deployment
```bash
# Build container
docker build -t cifar10-classifier .

# Run container
docker run -p 8000:8000 cifar10-classifier
```

### Performance Features
- **GPU Acceleration**: Automatic CUDA detection
- **Model Caching**: Single model load at startup
- **Fast Inference**: Optimized preprocessing pipeline
- **Error Handling**: Robust error responses

## 🔧 Configuration

### Training Parameters
Edit `training/cifar10_train.py`:
```python
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_CLASSES = 3  # automobile, bird, ship
```

### API Configuration
Edit `api/main.py`:
```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
HOST = "0.0.0.0"
PORT = 8000
```

## 📊 Dataset Information

### CIFAR-10 Subset
- **Total Images**: ~15,000 (from original 60,000)
- **Classes**: 3 (automobile, bird, ship)
- **Image Size**: 32×32 pixels
- **Format**: RGB color images
- **Split**: 80% training, 20% validation

### Class Distribution
- **Automobile**: ~5,000 images
- **Bird**: ~5,000 images  
- **Ship**: ~5,000 images

## 🛠️ Development

### Adding New Classes
To extend beyond the current 3 classes:
1. Modify `training/cifar10_train.py` to include additional CIFAR-10 classes
2. Update `NUM_CLASSES` parameter
3. Retrain the model
4. Update frontend class icons

### Model Improvements
- Try different architectures (ResNet50, EfficientNet)
- Experiment with data augmentation
- Implement ensemble methods
- Add model quantization for faster inference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**Built with PyTorch, FastAPI, and CIFAR-10 dataset**