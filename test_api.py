#!/usr/bin/env python3
"""
Test script for the AI Image Classification API
"""

import requests
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw

def create_test_images():
    """Create sample test images for each class."""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create simple test images
    classes = {
        "animal": ("🐾", "orange"),
        "human": ("👤", "lightblue"), 
        "car": ("🚗", "red"),
        "house": ("🏠", "green")
    }
    
    for class_name, (emoji, color) in classes.items():
        # Create a simple colored image with text
        img = Image.new('RGB', (224, 224), color=color)
        draw = ImageDraw.Draw(img)
        
        # Draw a simple shape
        if class_name == "animal":
            draw.ellipse([50, 50, 174, 174], fill='brown', outline='black', width=3)
        elif class_name == "human":
            draw.rectangle([80, 50, 144, 150], fill='blue', outline='black', width=3)
            draw.ellipse([90, 30, 134, 74], fill='#FFDBAC', outline='black', width=3)
        elif class_name == "car":
            draw.rectangle([40, 100, 184, 150], fill='darkred', outline='black', width=3)
            draw.ellipse([60, 140, 90, 170], fill='black')
            draw.ellipse([154, 140, 184, 170], fill='black')
        elif class_name == "house":
            draw.rectangle([60, 100, 164, 180], fill='brown', outline='black', width=3)
            draw.polygon([(60, 100), (112, 50), (164, 100)], fill='darkgreen', outline='black', width=3)
        
        img_path = test_dir / f"{class_name}_test.png"
        img.save(img_path)
        print(f"Created test image: {img_path}")

def test_health_endpoint():
    """Test the health check endpoint."""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check passed!")
            print(f"   Status: {result['status']}")
            print(f"   Model: {result['model']['model_name']}")
            print(f"   Classes: {result['model']['num_classes']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_prediction(image_path, expected_class=None):
    """Test image prediction endpoint."""
    print(f"\n🔍 Testing prediction: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            start_time = time.time()
            response = requests.post("http://localhost:8000/predict/image", files=files, timeout=30)
            end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"   Filename: {result['filename']}")
            print(f"   Predicted: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']:.4f}")
            print(f"   Inference time: {result['inference_time']}s")
            print(f"   Request time: {end_time - start_time:.2f}s")
            
            print("   Top predictions:")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"     {i}. {pred['class']}: {pred['confidence']:.4f}")
            
            if expected_class and result['predicted_class'] == expected_class:
                print(f"🎯 Correct prediction!")
            
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_invalid_requests():
    """Test invalid request handling."""
    print("\n🚫 Testing invalid requests...")
    
    # Test without file
    try:
        response = requests.post("http://localhost:8000/predict/image", timeout=10)
        if response.status_code == 422:
            print("✅ Correctly rejected request without file")
        else:
            print(f"❌ Unexpected response for no file: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing no file: {e}")
    
    # Test with invalid file type
    try:
        with open("test_file.txt", "w") as f:
            f.write("This is not an image")
        
        with open("test_file.txt", "rb") as f:
            files = {'file': ('test.txt', f, 'text/plain')}
            response = requests.post("http://localhost:8000/predict/image", files=files, timeout=10)
        
        if response.status_code == 400:
            print("✅ Correctly rejected invalid file type")
        else:
            print(f"❌ Unexpected response for invalid file: {response.status_code}")
        
        Path("test_file.txt").unlink()  # Clean up
        
    except Exception as e:
        print(f"❌ Error testing invalid file: {e}")

def main():
    """Run comprehensive API tests."""
    print("🧪 AI Image Classification API Test Suite")
    print("=" * 60)
    
    # Test health endpoint
    if not test_health_endpoint():
        print("\n❌ Server not running or unhealthy!")
        print("💡 Start the server with: python api/main.py")
        return
    
    # Create test images
    print("\n📸 Creating test images...")
    create_test_images()
    
    # Test predictions with each class
    test_cases = [
        ("test_images/animal_test.png", "animal"),
        ("test_images/human_test.png", "human"),
        ("test_images/car_test.png", "car"),
        ("test_images/house_test.png", "house")
    ]
    
    successful_tests = 0
    for image_path, expected_class in test_cases:
        if test_prediction(image_path, expected_class):
            successful_tests += 1
    
    # Test invalid requests
    test_invalid_requests()
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {successful_tests}/{len(test_cases)} predictions successful")
    
    if successful_tests == len(test_cases):
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the server logs.")
    
    print("\n🌐 Access points:")
    print("   - Web Interface: http://localhost:8000")
    print("   - API Docs: http://localhost:8000/docs")
    print("   - Health Check: http://localhost:8000/health")

if __name__ == "__main__":
    main()