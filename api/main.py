
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import io
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
MODEL_PATH = 'models/model.pth'
CLASS_NAMES_PATH = 'models/class_names.json'

class ImageClassifier:

    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.num_classes = 0
        self.model_name = 'resnet18'
        self.transform = None
        
        print(f"🔧 Initializing classifier on {self.device}")
        self._load_model()
        self._setup_transforms()
    
    def _load_model(self):

        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH):
                self._load_trained_model()
            else:
                self._create_mock_classifier()
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            self._create_mock_classifier()
    
    def _load_trained_model(self):
       
        print("📦 Loading trained model...")
        
        # Load class information
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_info = json.load(f)
        
        self.class_to_idx = class_info['class_to_idx']
        self.idx_to_class = class_info['idx_to_class']
        self.num_classes = class_info['num_classes']
        self.class_names = class_info['class_names']
        
        
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        self.model_name = checkpoint.get('model_name', 'resnet18')
        
      
        if self.model_name == 'resnet18':
            self.model = models.resnet18(pretrained=False)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
            
        elif self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=False)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
            
        elif self.model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=False)
            self.model.classifier[1] = nn.Linear(self.model.last_channel, self.num_classes)
        
       
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(self.device)
        
        print(f"✅ Loaded trained {self.model_name} model")
        print(f"Classes: {self.class_names}")
    
    def _create_mock_classifier(self):
       
        print("⚠️  No trained model found, using mock classifier")
        
       
        try:
            from torchvision import datasets
            temp_dataset = datasets.CIFAR10(root='cifar10_data', train=False, download=False)
            self.class_names = temp_dataset.classes
            self.class_to_idx = temp_dataset.class_to_idx
            self.idx_to_class = {str(idx): name for idx, name in enumerate(self.class_names)}
            print(f"✅ Using CIFAR-10 classes from dataset: {self.class_names}")
        except:
      
            self.class_names = [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.idx_to_class = {str(idx): name for idx, name in enumerate(self.class_names)}
            print(f"✅ Using default CIFAR-10 classes: {self.class_names}")
        
        self.num_classes = len(self.class_names)
        print(f"🎭 Mock classifier with {self.num_classes} classes")
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  
        return tensor.to(self.device)
    
    def predict(self, image: Image.Image, top_k: int = 3) -> Dict[str, Any]:
     
        try:
            if self.model is not None:
                return self._predict_with_model(image, top_k)
            else:
                return self._predict_mock(image, top_k)
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def _predict_with_model(self, image: Image.Image, top_k: int = 3) -> Dict[str, Any]:
      
        input_tensor = self._preprocess_image(image)
        
       
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
       
        top_prob, top_indices = torch.topk(probabilities, min(top_k, len(self.class_names)))
        
       
        top_predictions = []
        for i in range(len(top_indices)):
            class_idx = top_indices[i].item()
            confidence = top_prob[i].item()
            class_name = self.class_names[class_idx]
            
            top_predictions.append({
                "class": class_name,
                "confidence": round(confidence, 4)
            })
        
        return {
            "predicted_class": top_predictions[0]["class"],
            "confidence": top_predictions[0]["confidence"],
            "top_predictions": top_predictions
        }
    
    def _predict_mock(self, image: Image.Image, top_k: int = 3) -> Dict[str, Any]:
       
        import random
        import hashlib
        
        img_bytes = image.tobytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()
        random.seed(int(img_hash[:8], 16))
        
        
        predictions = []
        remaining_prob = 1.0
        
        for i in range(min(top_k, len(self.class_names))):
            if i == top_k - 1:
                confidence = remaining_prob
            else:
                confidence = random.uniform(0.1, remaining_prob * 0.7)
                remaining_prob -= confidence
            
            class_name = random.choice(self.class_names)
            predictions.append({
                "class": class_name,
                "confidence": round(confidence, 4)
            })
        
       
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "predicted_class": predictions[0]["class"],
            "confidence": predictions[0]["confidence"],
            "top_predictions": predictions
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "device": str(self.device),
            "trained_model_available": self.model is not None,
            "input_size": "224x224"
        }


app = FastAPI(
    title="AI Image Classification API",
    description="Production-ready image classification for Animals, Humans, Cars, and Houses",
    version="1.0.0"
)

classifier = ImageClassifier()


try:
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
except:
    pass  # Frontend directory might not exist yet

@app.get("/", response_class=HTMLResponse)
async def home():

    try:
        frontend_path = Path("frontend/index.html")
        if frontend_path.exists():
            with open(frontend_path, "r", encoding='utf-8') as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(content="""
            <html>
                <head><title>AI Image Classifier</title></head>
                <body>
                    <h1>🤖 AI Image Classification API</h1>
                    <p>The API is running! Visit <a href="/docs">/docs</a> for interactive documentation.</p>
                    <p>Frontend file not found at: frontend/index.html</p>
                </body>
            </html>
            """)
    except Exception as e:
        return HTMLResponse(content=f"""
        <html>
            <head><title>AI Image Classifier</title></head>
            <body>
                <h1>🤖 AI Image Classification API</h1>
                <p>The API is running! Visit <a href="/docs">/docs</a> for interactive documentation.</p>
                <p>Error loading frontend: {str(e)}</p>
            </body>
        </html>
        """)

@app.get("/health")
async def health_check():
 
    try:
        model_info = classifier.get_model_info()
        return {
            "status": "healthy",
            "service": "ai-image-classification",
            "timestamp": time.time(),
            "model": model_info
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
 
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    try:
  
        image = Image.open(io.BytesIO(contents))
        
      
        start_time = time.time()
        results = classifier.predict(image, top_k=3)
        inference_time = time.time() - start_time
        
        return {
            "success": True,
            "filename": file.filename,
            "predicted_class": results["predicted_class"],
            "confidence": results["confidence"],
            "top_predictions": results["top_predictions"],
            "inference_time": round(inference_time, 4),
            "model_info": classifier.get_model_info()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.exception_handler(413)
async def request_entity_too_large(request, exc):
    
    return {"error": "File too large", "max_size": f"{MAX_FILE_SIZE // (1024*1024)}MB"}

@app.exception_handler(500)
async def internal_server_error(request, exc):
    
    return {"error": "Internal server error", "detail": str(exc)}

def main():
   
    print("🚀 Starting AI Image Classification API")
    print("=" * 50)
    print(f"📊 Model: {classifier.model_name}")
    print(f"🏷️  Classes: {classifier.class_names}")
    print(f"🔧 Device: {classifier.device}")
    print("=" * 50)
    print("🌐 Server will be available at:")
    print("   - Web Interface: http://localhost:8000")
    print("   - API Docs: http://localhost:8000/docs")
    print("   - Health Check: http://localhost:8000/health")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
