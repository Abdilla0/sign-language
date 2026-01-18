"""
Convert Label Studio export to YOLO training format
Splits data into train/val/test and creates data.yaml
"""

import os
import shutil
import random
from pathlib import Path

def convert_label_studio_to_yolo(label_studio_path, output_path):
    """
    Convert Label Studio YOLO export to proper YOLO training structure
    
    Args:
        label_studio_path: Path to extracted Label Studio export
        output_path: Where to save organized dataset
    """
    print("="*60)
    print("LABEL STUDIO TO YOLO CONVERTER")
    print("="*60)
    
    # Paths
    ls_images = Path(label_studio_path) / "images"
    ls_labels = Path(label_studio_path) / "labels"
    classes_file = Path(label_studio_path) / "classes.txt"
    
    output = Path(output_path)
    
    # Check input exists
    if not ls_images.exists():
        print(f"❌ Images folder not found: {ls_images}")
        return
    
    if not ls_labels.exists():
        print(f"❌ Labels folder not found: {ls_labels}")
        return
    
    if not classes_file.exists():
        print(f"❌ classes.txt not found: {classes_file}")
        return
    
    # Read classes
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    print(f"\n📋 Found {len(classes)} classes:")
    for i, cls in enumerate(classes):
        print(f"   {i}: {cls}")
    
    # Get all image files
    image_files = list(ls_images.glob("*.jpg")) + list(ls_images.glob("*.png"))
    print(f"\n📸 Found {len(image_files)} images")
    
    # Check for corresponding labels
    valid_pairs = []
    for img_file in image_files:
        label_file = ls_labels / f"{img_file.stem}.txt"
        if label_file.exists():
            valid_pairs.append((img_file, label_file))
    
    print(f"✅ Found {len(valid_pairs)} image-label pairs")
    
    if len(valid_pairs) == 0:
        print("❌ No valid image-label pairs found!")
        return
    
    # Shuffle and split
    random.shuffle(valid_pairs)
    
    train_split = int(0.7 * len(valid_pairs))
    val_split = int(0.9 * len(valid_pairs))
    
    train_pairs = valid_pairs[:train_split]
    val_pairs = valid_pairs[train_split:val_split]
    test_pairs = valid_pairs[val_split:]
    
    print(f"\n📊 Dataset Split:")
    print(f"   Train: {len(train_pairs)} images ({len(train_pairs)/len(valid_pairs)*100:.1f}%)")
    print(f"   Val:   {len(val_pairs)} images ({len(val_pairs)/len(valid_pairs)*100:.1f}%)")
    print(f"   Test:  {len(test_pairs)} images ({len(test_pairs)/len(valid_pairs)*100:.1f}%)")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (output / split / 'images').mkdir(parents=True, exist_ok=True)
        (output / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy files
    print("\n📦 Copying files...")
    
    def copy_split(pairs, split_name):
        for img_file, label_file in pairs:
            # Copy image
            shutil.copy2(img_file, output / split_name / 'images' / img_file.name)
            # Copy label
            shutil.copy2(label_file, output / split_name / 'labels' / label_file.name)
    
    copy_split(train_pairs, 'train')
    print(f"   ✅ Train set copied")
    
    copy_split(val_pairs, 'val')
    print(f"   ✅ Validation set copied")
    
    copy_split(test_pairs, 'test')
    print(f"   ✅ Test set copied")
    
    # Create data.yaml
    data_yaml_content = f"""# ASL Sign Language Dataset
# Converted from Label Studio

path: {output.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
nc: {len(classes)}
names: {classes}
"""
    
    yaml_path = output / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"\n✅ Created data.yaml at: {yaml_path}")
    
    # Print final structure
    print("\n" + "="*60)
    print("DATASET READY!")
    print("="*60)
    print(f"Location: {output.absolute()}")
    print("\nStructure:")
    print(f"""
{output.name}/
├── train/
│   ├── images/ ({len(train_pairs)} images)
│   └── labels/ ({len(train_pairs)} labels)
├── val/
│   ├── images/ ({len(val_pairs)} images)
│   └── labels/ ({len(val_pairs)} labels)
├── test/
│   ├── images/ ({len(test_pairs)} images)
│   └── labels/ ({len(test_pairs)} labels)
└── data.yaml ✅
    """)
    
    print("\n🚀 NEXT STEP:")
    print(f"   Train your model with:")
    print(f"   python yolo_training.py")
    print(f"   (Make sure to update the data path in the script!)")
    print("="*60)


if __name__ == "__main__":
    print("\n🎯 Label Studio to YOLO Format Converter\n")
    
    # Get paths from user
    default_input = "C:/myfile/yoloproject/dataset/label_studio_export"
    default_output = "C:/myfile/yoloproject/dataset/asl_yolo_dataset"
    
    print("📁 Input: Label Studio export folder")
    print(f"   (Should contain: images/, labels/, classes.txt)")
    input_path = input(f"\nLabel Studio export path [{default_input}]: ").strip()
    if not input_path:
        input_path = default_input
    
    print("\n📁 Output: Where to save YOLO dataset")
    output_path = input(f"Output path [{default_output}]: ").strip()
    if not output_path:
        output_path = default_output
    
    print("\n🔄 Converting...")
    convert_label_studio_to_yolo(input_path, output_path)
    
    print("\n✅ Done! Press ENTER to exit...")
    input()                                                                                         