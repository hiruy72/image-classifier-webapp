# 🤖 AI Image Classification Web Application

A production-ready end-to-end machine learning system that automatically trains and serves image classification models using the **CIFAR-10 dataset**.

## 🎯 Overview

This application:
- Uses the famous CIFAR-10 dataset with 60,000 images
- Trains a deep learning model using transfer learning
- Serves predictions through a FastAPI backend
- Provides a modern web interface for image classification

## 📊 CIFAR-10 Classes

The model classifies images into 10 categories:

| Class | Icon | Description |
|-------|------|-------------|
| **Airplane** | ✈️ | Aircraft, planes |
| **Automobile** | 🚗 | Cars, vehicles |
| **Bird** | 🐦 | Flying birds |
| **Cat** | 🐱 | Domestic cats |
| **Deer** | 🦌 | Wild deer |
| **Dog** | 🐕 | Domestic dogs |
| **Frog** | 🐸 | Amphibians |
| **Horse** | 🐴 | Horses |
| **Ship** | 🚢 | Boats, ships |
| **Truck** | 🚛 | Large vehicles |

## 📁 Project Structure

```
image-classification-app/
├── dataset/                 # Training images (auto-detected)
│   ├── animal/             # Animal images
│   ├── human/              # Human images  
│   └── house/              # House images
├── models/                 # Trained models
│   ├── model.pth          # PyTorch model
│   └── class_names.json   # Class mapping
├── api/                   # FastAPI backend
│   └── main.py           # API server
├── training/             # Training scripts
│   └── train.py         # Model training
├── frontend/            # Web interface
│   └── index.html      # Frontend UI
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Create the dataset structure and add your images:
```bash
mkdir -p dataset/{animal,human,house}
# Add your images to respective folders
```

### 3. Train the Model
```bash
python training/train.py
```

### 4. Start the API Server
```bash
python api/main.py
```

### 5. Access the Application
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📚 Dataset Instructions

### Folder Structure
Place your training images in the following structure:
```
dataset/
├── animal/
│   ├── cat1.jpg
│   ├── dog1.png
│   └── bird1.jpeg
├── human/
│   ├── person1.jpg
│   └── group1.png
└── house/
    ├── home1.jpg
    └── apartment1.png
```

### Image Requirements
- **Formats**: JPG, JPEG, PNG
- **Size**: Any size (automatically resized to 224×224)
- **Quality**: Clear, well-lit images work best
- **Quantity**: Minimum 50 images per class recommended

## 🎓 Training Steps

### Automatic Training Process
1. **Dataset Detection**: Automatically scans folders and creates class mapping
2. **Data Loading**: Loads images with augmentation (flip, rotation, normalization)
3. **Model Creation**: Uses transfer learning with ResNet18
4. **Training**: Trains with GPU if available, shows progress
5. **Validation**: Evaluates accuracy and saves best model
6. **Export**: Saves model.pth and class_names.json

### Manual Training
```bash
cd training
python train.py --epochs 20 --batch-size 32 --learning-rate 0.001
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
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

#### Response Format
```json
{
  "success": true,
  "filename": "your_image.jpg",
  "predicted_class": "animal",
  "confidence": 0.95,
  "top_predictions": [
    {"class": "animal", "confidence": 0.95},
    {"class": "human", "confidence": 0.03},
    {"class": "car", "confidence": 0.02}
  ]
}
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/image` | POST | Upload image for classification |
| `/health` | GET | Service health status |
| `/docs` | GET | Interactive API documentation |

## 🏗️ Model Architecture

### Transfer Learning Approach
- **Base Model**: ResNet18 (pre-trained on ImageNet)
- **Modification**: Replace final layer for 4-class classification
- **Optimization**: Adam optimizer with learning rate scheduling
- **Augmentation**: Random horizontal flip, rotation, normalization

### Training Configuration
- **Input Size**: 224×224×3
- **Batch Size**: 32 (adjustable)
- **Learning Rate**: 0.001 with decay
- **Epochs**: 20 (early stopping enabled)
- **Device**: Auto-detects GPU/CPU

## 🧪 Testing

### Web Interface Testing
1. Open http://localhost:8000
2. Drag and drop an image or click to browse
3. View prediction results with confidence scores

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Image prediction
curl -X POST "http://localhost:8000/predict/image" \
     -F "file=@test_image.jpg"
```

## 🚀 Production Deployment

### Docker Support
```bash
# Build image
docker build -t image-classifier .

# Run container
docker run -p 8000:8000 image-classifier
```

### Performance Optimization
- **GPU Acceleration**: Automatically uses CUDA if available
- **Model Caching**: Loads model once at startup
- **Fast Inference**: Optimized preprocessing pipeline
- **Batch Processing**: Supports multiple image uploads

## 🔧 Configuration

### Training Parameters
Edit `training/train.py` to modify:
```python
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
```

### API Configuration
Edit `api/main.py` to modify:
```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
```

## 📈 Model Performance

Expected accuracy with sufficient training data:
- **Animal Classification**: 90-95%
- **Human Detection**: 85-92%
- **Car Recognition**: 88-94%
- **House Identification**: 85-90%

## 🛠️ Development

### Adding New Classes
1. Create new folder in `dataset/`
2. Add training images
3. Retrain model: `python training/train.py`
4. Restart API server

### Custom Model Architecture
Modify `training/train.py` to use different models:
```python
# Options: resnet18, resnet50, mobilenet_v2, efficientnet_b0
model = create_model('resnet18', num_classes=4)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ using PyTorch, FastAPI, and modern web technologies**