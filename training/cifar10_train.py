#!/usr/bin/env python3
"""
CIFAR-10 Image Classification Training Script
Downloads and trains on the CIFAR-10 dataset with 10 classes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import os
import json
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class CIFAR10Trainer:
    """CIFAR-10 image classification trainer."""
    
    def __init__(self, model_name='resnet18', data_dir='cifar10_data'):
        self.model_name = model_name
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.num_classes = 10
        
        # Load CIFAR-10 class names from the dataset
        self.class_names = None
        self.class_to_idx = None
        self._load_cifar10_classes()
        
        print(f"🔧 Using device: {self.device}")
        print(f"📊 CIFAR-10 dataset will be loaded automatically")
        
        # Create models directory
        Path('models').mkdir(exist_ok=True)
    
    def _load_cifar10_classes(self):
        """Load CIFAR-10 class names from the dataset."""
        # Create a temporary dataset to get class names
        temp_dataset = datasets.CIFAR10(root=self.data_dir, train=False, download=True)
        self.class_names = temp_dataset.classes
        self.class_to_idx = temp_dataset.class_to_idx
        print(f"✅ Loaded CIFAR-10 classes: {self.class_names}")
    
    def get_transforms(self, is_training=True):
        """Get data transforms for CIFAR-10."""
        if is_training:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def create_model(self):
        """Create model with transfer learning for CIFAR-10."""
        if self.model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
            
        elif self.model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
            
        elif self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.last_channel, self.num_classes)
            
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model.to(self.device)
    
    def load_cifar10_data(self, batch_size=32):
        """Load CIFAR-10 dataset."""
        print("📦 Loading CIFAR-10 dataset...")
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Get transforms
        train_transform = self.get_transforms(is_training=True)
        test_transform = self.get_transforms(is_training=False)
        
        # Load datasets
        train_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform
        )
        
        test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=test_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"📊 Training samples: {len(train_dataset)}")
        print(f"📊 Test samples: {len(test_dataset)}")
        print(f"✅ Classes: {train_dataset.classes}")
        
        return train_loader, test_loader
    
    def train(self, epochs=20, batch_size=32, learning_rate=0.001):
        """Train the model on CIFAR-10."""
        print("🚀 Starting CIFAR-10 training...")
        
        # Load data
        train_loader, test_loader = self.load_cifar10_data(batch_size)
        
        # Create model
        self.model = self.create_model()
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Training loop
        train_losses = []
        test_accuracies = []
        best_test_acc = 0.0
        
        print(f"\n🎯 Training {self.model_name} for {epochs} epochs...")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_idx, (images, labels) in enumerate(train_pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                # Update progress bar
                if batch_idx % 100 == 0:
                    train_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*correct_train/total_train:.2f}%'
                    })
            
            # Test phase
            self.model.eval()
            correct_test = 0
            total_test = 0
            
            with torch.no_grad():
                test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Test]')
                for images, labels in test_pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
                    
                    test_pbar.set_postfix({
                        'Acc': f'{100.*correct_test/total_test:.2f}%'
                    })
            
            # Calculate metrics
            epoch_loss = running_loss / len(train_loader)
            train_acc = 100. * correct_train / total_train
            test_acc = 100. * correct_test / total_test
            
            train_losses.append(epoch_loss)
            test_accuracies.append(test_acc)
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                self.save_model('models/model.pth')
                print(f"💾 New best model saved! Test accuracy: {test_acc:.2f}%")
            
            scheduler.step()
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s")
            print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
            print("-" * 60)
        
        print(f"✅ Training completed! Best test accuracy: {best_test_acc:.2f}%")
        
        # Plot training curves
        self.plot_training_curves(train_losses, test_accuracies)
        
        return True
    
    def save_model(self, save_path):
        """Save the trained model and class information."""
        # Save model checkpoint
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_to_idx': self.class_to_idx,
            'num_classes': self.num_classes,
            'model_name': self.model_name,
            'class_names': self.class_names
        }, save_path)
        
        # Save class names mapping
        class_info = {
            'class_to_idx': self.class_to_idx,
            'idx_to_class': {str(v): k for k, v in self.class_to_idx.items()},
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }
        
        with open('models/class_names.json', 'w') as f:
            json.dump(class_info, f, indent=2)
        
        print(f"💾 Model saved to {save_path}")
        print(f"💾 Class mapping saved to models/class_names.json")
    
    def plot_training_curves(self, train_losses, test_accuracies):
        """Plot training curves."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Training loss
            ax1.plot(train_losses)
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
            
            # Test accuracy
            ax2.plot(test_accuracies)
            ax2.set_title('Test Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('models/training_curves.png', dpi=150, bbox_inches='tight')
            print("📊 Training curves saved to models/training_curves.png")
            
        except Exception as e:
            print(f"⚠️  Could not save training curves: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 Classification Model')
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet50', 'mobilenet_v2'])
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    print("🤖 CIFAR-10 Image Classification Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = CIFAR10Trainer(args.model)
    
    # Start training
    success = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("🚀 Start the API server: python api/main.py")
    else:
        print("\n❌ Training failed!")

if __name__ == "__main__":
    main()