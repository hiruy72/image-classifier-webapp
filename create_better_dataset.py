#!/usr/bin/env python3
"""
Create better training images for improved car detection
"""

from PIL import Image, ImageDraw
import os
from pathlib import Path

def create_diverse_car_images():
    """Create multiple diverse car images."""
    car_dir = Path("dataset/car")
    car_dir.mkdir(parents=True, exist_ok=True)
    
    # Car variations
    car_configs = [
        {"name": "sedan", "color": "red", "body": [60, 120, 280, 160], "windows": [[80, 100, 140, 120], [200, 100, 260, 120]]},
        {"name": "suv", "color": "blue", "body": [50, 110, 290, 170], "windows": [[70, 90, 130, 110], [190, 90, 250, 110]]},
        {"name": "truck", "color": "green", "body": [40, 100, 300, 180], "windows": [[60, 80, 120, 100]]},
        {"name": "sports_car", "color": "yellow", "body": [70, 130, 270, 150], "windows": [[90, 110, 150, 130], [190, 110, 250, 130]]},
        {"name": "van", "color": "purple", "body": [45, 105, 295, 175], "windows": [[65, 85, 125, 105], [185, 85, 245, 105]]}
    ]
    
    for i, config in enumerate(car_configs):
        img = Image.new('RGB', (340, 240), color='lightgray')
        draw = ImageDraw.Draw(img)
        
        # Draw car body
        draw.rectangle(config["body"], fill=config["color"], outline='black', width=3)
        
        # Draw windows
        for window in config["windows"]:
            draw.rectangle(window, fill='lightblue', outline='black', width=2)
        
        # Draw wheels
        body = config["body"]
        wheel_y = body[3] - 10
        wheel1_x = body[0] + 20
        wheel2_x = body[2] - 40
        
        draw.ellipse([wheel1_x, wheel_y, wheel1_x + 30, wheel_y + 30], fill='black', outline='gray', width=2)
        draw.ellipse([wheel2_x, wheel_y, wheel2_x + 30, wheel_y + 30], fill='black', outline='gray', width=2)
        
        # Add headlights
        if config["name"] != "truck":
            draw.ellipse([body[2] - 15, body[1] + 10, body[2] - 5, body[1] + 20], fill='white', outline='black')
            draw.ellipse([body[2] - 15, body[3] - 20, body[2] - 5, body[3] - 10], fill='white', outline='black')
        
        # Save image
        img.save(car_dir / f"car_{config['name']}.png")
        print(f"Created car image: car_{config['name']}.png")

def create_diverse_animal_images():
    """Create multiple diverse animal images."""
    animal_dir = Path("dataset/animal")
    animal_dir.mkdir(parents=True, exist_ok=True)
    
    animal_configs = [
        {"name": "cat", "color": "orange", "body": [120, 120, 200, 180], "head": [140, 90, 180, 130]},
        {"name": "dog", "color": "brown", "body": [100, 130, 220, 170], "head": [80, 110, 120, 150]},
        {"name": "bird", "color": "yellow", "body": [140, 140, 180, 170], "head": [150, 120, 170, 140]},
        {"name": "fish", "color": "blue", "body": [120, 130, 200, 160], "head": [200, 135, 220, 155]},
        {"name": "rabbit", "color": "white", "body": [130, 140, 190, 180], "head": [145, 110, 175, 140]}
    ]
    
    for config in animal_configs:
        img = Image.new('RGB', (300, 250), color='lightgreen')
        draw = ImageDraw.Draw(img)
        
        # Draw body
        draw.ellipse(config["body"], fill=config["color"], outline='black', width=3)
        
        # Draw head
        draw.ellipse(config["head"], fill=config["color"], outline='black', width=2)
        
        # Add eyes
        head = config["head"]
        eye1_x = head[0] + 10
        eye2_x = head[2] - 20
        eye_y = head[1] + 10
        
        draw.ellipse([eye1_x, eye_y, eye1_x + 8, eye_y + 8], fill='black')
        draw.ellipse([eye2_x, eye_y, eye2_x + 8, eye_y + 8], fill='black')
        
        # Special features for different animals
        if config["name"] == "cat":
            # Cat ears
            draw.polygon([(head[0] + 5, head[1]), (head[0] + 15, head[1] - 15), (head[0] + 25, head[1])], fill=config["color"], outline='black')
            draw.polygon([(head[2] - 25, head[1]), (head[2] - 15, head[1] - 15), (head[2] - 5, head[1])], fill=config["color"], outline='black')
        elif config["name"] == "dog":
            # Dog ears
            draw.ellipse([head[0] - 5, head[1] + 5, head[0] + 15, head[1] + 25], fill=config["color"], outline='black')
            draw.ellipse([head[2] - 15, head[1] + 5, head[2] + 5, head[1] + 25], fill=config["color"], outline='black')
        elif config["name"] == "rabbit":
            # Rabbit ears
            draw.ellipse([head[0] + 10, head[1] - 20, head[0] + 20, head[1] + 5], fill=config["color"], outline='black')
            draw.ellipse([head[2] - 20, head[1] - 20, head[2] - 10, head[1] + 5], fill=config["color"], outline='black')
        
        img.save(animal_dir / f"animal_{config['name']}.png")
        print(f"Created animal image: animal_{config['name']}.png")

def create_diverse_human_images():
    """Create multiple diverse human images."""
    human_dir = Path("dataset/human")
    human_dir.mkdir(parents=True, exist_ok=True)
    
    human_configs = [
        {"name": "person1", "shirt_color": "blue", "head_pos": [140, 60, 180, 100]},
        {"name": "person2", "shirt_color": "red", "head_pos": [130, 70, 170, 110]},
        {"name": "person3", "shirt_color": "green", "head_pos": [145, 65, 185, 105]},
        {"name": "child", "shirt_color": "yellow", "head_pos": [150, 80, 180, 110]},
        {"name": "adult", "shirt_color": "purple", "head_pos": [135, 55, 175, 95]}
    ]
    
    for config in human_configs:
        img = Image.new('RGB', (320, 280), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw head
        head = config["head_pos"]
        draw.ellipse(head, fill='#FFDBAC', outline='black', width=3)
        
        # Draw body
        body_top = head[3]
        body_bottom = body_top + 80
        body_left = head[0] + 5
        body_right = head[2] - 5
        
        draw.rectangle([body_left, body_top, body_right, body_bottom], fill=config["shirt_color"], outline='black', width=2)
        
        # Draw arms
        draw.line([(body_left, body_top + 20), (body_left - 30, body_top + 50)], fill='black', width=4)
        draw.line([(body_right, body_top + 20), (body_right + 30, body_top + 50)], fill='black', width=4)
        
        # Draw legs
        draw.line([(body_left + 10, body_bottom), (body_left, body_bottom + 40)], fill='black', width=4)
        draw.line([(body_right - 10, body_bottom), (body_right, body_bottom + 40)], fill='black', width=4)
        
        # Draw eyes
        draw.ellipse([head[0] + 10, head[1] + 15, head[0] + 18, head[1] + 23], fill='black')
        draw.ellipse([head[2] - 18, head[1] + 15, head[2] - 10, head[1] + 23], fill='black')
        
        img.save(human_dir / f"human_{config['name']}.png")
        print(f"Created human image: human_{config['name']}.png")

def create_diverse_house_images():
    """Create multiple diverse house images."""
    house_dir = Path("dataset/house")
    house_dir.mkdir(parents=True, exist_ok=True)
    
    house_configs = [
        {"name": "cottage", "wall_color": "brown", "roof_color": "red"},
        {"name": "modern", "wall_color": "white", "roof_color": "gray"},
        {"name": "cabin", "wall_color": "#8B4513", "roof_color": "darkgreen"},
        {"name": "apartment", "wall_color": "lightgray", "roof_color": "blue"},
        {"name": "villa", "wall_color": "beige", "roof_color": "orange"}
    ]
    
    for config in house_configs:
        img = Image.new('RGB', (320, 280), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw house walls
        wall_coords = [80, 140, 240, 220]
        draw.rectangle(wall_coords, fill=config["wall_color"], outline='black', width=3)
        
        # Draw roof
        roof_peak_x = (wall_coords[0] + wall_coords[2]) // 2
        roof_peak_y = wall_coords[1] - 40
        draw.polygon([
            (wall_coords[0] - 10, wall_coords[1]),
            (roof_peak_x, roof_peak_y),
            (wall_coords[2] + 10, wall_coords[1])
        ], fill=config["roof_color"], outline='black', width=2)
        
        # Draw door
        door_width = 30
        door_height = 50
        door_x = roof_peak_x - door_width // 2
        door_y = wall_coords[3] - door_height
        draw.rectangle([door_x, door_y, door_x + door_width, door_y + door_height], fill='#8B4513', outline='black', width=2)
        
        # Draw windows
        window_size = 25
        window1_x = wall_coords[0] + 20
        window2_x = wall_coords[2] - 20 - window_size
        window_y = wall_coords[1] + 30
        
        draw.rectangle([window1_x, window_y, window1_x + window_size, window_y + window_size], fill='lightblue', outline='black', width=2)
        draw.rectangle([window2_x, window_y, window2_x + window_size, window_y + window_size], fill='lightblue', outline='black', width=2)
        
        # Draw window crosses
        draw.line([(window1_x + window_size//2, window_y), (window1_x + window_size//2, window_y + window_size)], fill='black', width=1)
        draw.line([(window1_x, window_y + window_size//2), (window1_x + window_size, window_y + window_size//2)], fill='black', width=1)
        draw.line([(window2_x + window_size//2, window_y), (window2_x + window_size//2, window_y + window_size)], fill='black', width=1)
        draw.line([(window2_x, window_y + window_size//2), (window2_x + window_size, window_y + window_size//2)], fill='black', width=1)
        
        img.save(house_dir / f"house_{config['name']}.png")
        print(f"Created house image: house_{config['name']}.png")

def main():
    """Create diverse training dataset."""
    print("🎨 Creating diverse training dataset...")
    print("=" * 50)
    
    create_diverse_car_images()
    print()
    create_diverse_animal_images()
    print()
    create_diverse_human_images()
    print()
    create_diverse_house_images()
    
    print("\n✅ Diverse dataset created!")
    print("📊 Dataset now contains:")
    
    for class_name in ["animal", "car", "house", "human"]:
        class_dir = Path(f"dataset/{class_name}")
        if class_dir.exists():
            image_count = len(list(class_dir.glob("*.png")))
            print(f"   {class_name}: {image_count} images")
    
    print("\n🎓 Ready to retrain with: python training/train.py --epochs 15")

if __name__ == "__main__":
    main()