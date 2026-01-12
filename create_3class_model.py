#!/usr/bin/env python3
"""
Script to create a 3-class model from the existing 10-class CIFAR-10 model.
Filters to only automobile, bird, and ship classes.
"""

import torch
import torch.nn as nn
from torchvision import models
import json
import os

def create_3class_model():
    """Create a 3-class model from the existing 10-class model."""
    
    # Define the 3 classes we want to keep
    target_classes = ['automobile', 'bird', 'ship']
    
    # Load the existing 10-class model
    if not os.path.exists('models/model.pth'):
        print("❌ No existing model found. Please train a model first.")
        return False
    
    print("📦 Loading existing 10-class model...")
    checkpoint = torch.load('models/model.pth', map_location='cpu')
    
    # Create new 3-class model architecture
    print("🔧 Creating 3-class model architecture...")
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 3)  # 3 classes instead of 10
    
    # We'll initialize with random weights since we can't directly transfer
    # the final layer weights from 10 classes to 3 classes
    print("⚠️  Note: Final layer will be randomly initialized for 3-class output")
    
    # Create new class mapping for 3 classes
    new_class_to_idx = {name: idx for idx, name in enumerate(target_classes)}
    new_idx_to_class = {str(idx): name for idx, name in enumerate(target_classes)}
    
    # Save the new 3-class model
    new_checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': new_class_to_idx,
        'num_classes': 3,
        'model_name': 'resnet18',
        'class_names': target_classes
    }
    
    torch.save(new_checkpoint, 'models/model_3class.pth')
    
    # Save new class names
    class_info = {
        'class_to_idx': new_class_to_idx,
        'idx_to_class': new_idx_to_class,
        'num_classes': 3,
        'class_names': target_classes
    }
    
    with open('models/class_names_3class.json', 'w') as f:
        json.dump(class_info, f, indent=2)
    
    print("✅ Created 3-class model:")
    print(f"   - Model: models/model_3class.pth")
    print(f"   - Classes: models/class_names_3class.json")
    print(f"   - Classes: {target_classes}")
    print("\n⚠️  Important: This model needs to be retrained for good performance!")
    print("   Run: python training/cifar10_train_3class.py")
    
    return True

if __name__ == "__main__":
    create_3class_model()