"""
YOLOv8 Training Script for ASL Sign Language Recognition
Updated for dataset location: dataset/asl_yolo_dataset
Requires: ultralytics, torch
Install: pip install ultralytics torch torchvision opencv-python
"""

from ultralytics import YOLO
import yaml
import os
import torch
from pathlib import Path

class ASLYOLOTrainer:
    def __init__(self, data_yaml_path, model_size='n'):
        """
        Initialize YOLO trainer
        
        Args:
            data_yaml_path: Path to data.yaml file
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        """
        self.data_yaml = data_yaml_path
        self.model_size = model_size
        self.model_path = f'yolov8{model_size}.pt'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🔧 Device: {self.device}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"📦 Model: YOLOv8{model_size.upper()}")
        
    def validate_dataset(self):
        """Check if dataset configuration is valid"""
        if not os.path.exists(self.data_yaml):
            print(f"❌ data.yaml not found at {self.data_yaml}")
            return False
        
        with open(self.data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        print("\n📊 Dataset Configuration:")
        print(f"  Path: {config.get('path', 'N/A')}")
        print(f"  Classes: {config.get('nc', 'N/A')}")
        print(f"  Names: {config.get('names', [])}")
        print(f"  Train: {config.get('train', 'N/A')}")
        print(f"  Val: {config.get('val', 'N/A')}")
        
        # Check if paths exist
        base_path = Path(config.get('path', ''))
        train_path = base_path / config.get('train', '')
        val_path = base_path / config.get('val', '')
        
        if train_path.exists():
            train_images = len(list(train_path.glob('*')))
            print(f"  Train images: {train_images}")
        else:
            print(f"  ⚠️  Train path not found: {train_path}")
        
        if val_path.exists():
            val_images = len(list(val_path.glob('*')))
            print(f"  Val images: {val_images}")
        else:
            print(f"  ⚠️  Val path not found: {val_path}")
        
        return True
    
    def train(self, epochs=100, img_size=640, batch_size=16, patience=50):
        """
        Train YOLO model
        
        Args:
            epochs: Number of training epochs
            img_size: Input image size
            batch_size: Batch size (reduce if GPU memory error)
            patience: Early stopping patience
        """
        if not self.validate_dataset():
            return None
        
        print(f"\n🚀 Starting Training...")
        print(f"  Epochs: {epochs}")
        print(f"  Image Size: {img_size}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Patience: {patience}")
        print(f"  Device: {self.device}")
        
        # Load pre-trained YOLO model
        print(f"\n📥 Loading {self.model_path}...")
        model = YOLO(self.model_path)
        
        # Train
        print("\n⏳ Training started (this may take a while)...")
        results = model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=patience,
            device=self.device,
            project='runs/asl_train',
            name='exp',
            exist_ok=True,
            
            # Optimization settings
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            
            # Augmentation (helps with small datasets)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            
            # Validation
            val=True,
            plots=True,
            save=True,
            save_period=10,
            
            # Performance
            verbose=True,
            workers=8,
            cache=False
        )
        
        print("\n✅ Training completed!")
        return results
    
    def evaluate(self, weights_path=None):
        """Evaluate trained model"""
        if weights_path is None:
            weights_path = 'runs/asl_train/exp/weights/best.pt'
        
        if not os.path.exists(weights_path):
            print(f"❌ Weights not found at {weights_path}")
            return None
        
        print(f"\n📊 Evaluating model: {weights_path}")
        model = YOLO(weights_path)
        
        # Validation
        results = model.val(
            data=self.data_yaml,
            imgsz=640,
            batch=16,
            device=self.device,
            plots=True
        )
        
        print("\n📈 Evaluation Results:")
        print(f"  mAP50: {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")
        print(f"  Precision: {results.box.mp:.4f}")
        print(f"  Recall: {results.box.mr:.4f}")
        
        return results
    
    def copy_best_model_to_models_folder(self):
        """Copy trained model to models folder for easy access"""
        best_weights = 'runs/asl_train/exp/weights/best.pt'
        
        if not os.path.exists(best_weights):
            print(f"⚠️  Best weights not found: {best_weights}")
            return
        
        # Create models folder if doesn't exist
        models_folder = Path('../models')
        models_folder.mkdir(parents=True, exist_ok=True)
        
        # Copy best.pt to models folder
        import shutil
        dest = models_folder / 'best.pt'
        shutil.copy2(best_weights, dest)
        
        print(f"\n✅ Model copied to: {dest.absolute()}")
        print(f"   You can now use this with sentence_builder.py!")
    
    def export_model(self, weights_path=None, format='onnx'):
        """Export model for deployment"""
        if weights_path is None:
            weights_path = 'runs/asl_train/exp/weights/best.pt'
        
        print(f"\n📤 Exporting model to {format}...")
        model = YOLO(weights_path)
        model.export(format=format)
        print("✅ Export completed!")


def find_data_yaml():
    """Search for data.yaml in common locations"""
    
    # List of possible locations
    possible_paths = [
        'dataset/asl_yolo_dataset/data.yaml',  # NEW: Your Label Studio dataset
        '../dataset/asl_yolo_dataset/data.yaml',
        'dataset/ASL-Static-Test-1/data.yaml',  # OLD: Your Roboflow dataset
        '../dataset/ASL-Static-Test-1/data.yaml',
        'dataset/merged_asl_dataset/data.yaml',  # Merged dataset
        '../dataset/merged_asl_dataset/data.yaml',
        'data.yaml',
    ]
    
    print("🔍 Searching for data.yaml...")
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found: {path}")
            return path
    
    print("❌ data.yaml not found in common locations")
    return None


def create_sample_data_yaml():
    """Create a sample data.yaml template"""
    sample_yaml = """# ASL Sign Language Dataset Configuration
# Replace paths with your actual dataset paths

path: C:/myfile/yoloproject/dataset/asl_yolo_dataset  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images      # validation images (relative to 'path')
test: test/images    # test images (optional)

# Classes
nc: 10  # number of classes
names: ['hello', 'thank_you', 'yes', 'no', 'please', 
        'sorry', 'help', 'stop', 'love', 'good']

# Dataset info
# Edit 'names' to match your actual gesture classes
"""
    
    with open('data_sample.yaml', 'w') as f:
        f.write(sample_yaml)
    
    print("📝 Sample data.yaml created: data_sample.yaml")
    print("   Edit this file with your dataset paths and classes")


if __name__ == "__main__":
    print("="*70)
    print("🤖 ASL YOLO Training System")
    print("="*70)
    
    # Try to find data.yaml automatically
    data_yaml = find_data_yaml()
    
    if not data_yaml:
        print("\n⚠️  data.yaml not found!")
        print("\n📋 Please provide the path to your data.yaml file")
        print("   Example: dataset/asl_yolo_dataset/data.yaml")
        
        custom_path = input("\nEnter path to data.yaml (or ENTER to create template): ").strip()
        
        if custom_path and os.path.exists(custom_path):
            data_yaml = custom_path
        else:
            print("\nCreating sample template...")
            create_sample_data_yaml()
            print("\n📋 STEPS TO GET STARTED:")
            print("1. Collect images using dataset_collector.py")
            print("2. Annotate with Label Studio or Roboflow")
            print("3. Export in YOLOv8 format")
            print("4. Place dataset in: dataset/asl_yolo_dataset/")
            print("5. Make sure data.yaml exists with correct paths")
            print("6. Run this script again")
            exit(0)
    
    print(f"\n✅ Using dataset: {data_yaml}")
    
    # Initialize trainer
    print("\n🔧 Select model size:")
    print("  n = nano (fastest, good for testing)")
    print("  s = small (fast, good accuracy)")
    print("  m = medium (balanced - RECOMMENDED) ⭐")
    print("  l = large (slower, higher accuracy)")
    print("  x = xlarge (slowest, best accuracy)")
    
    model_size = input("\nEnter model size (default: m): ").strip().lower() or 'm'
    
    if model_size not in ['n', 's', 'm', 'l', 'x']:
        print(f"⚠️  Invalid size '{model_size}', using 'm' (medium)")
        model_size = 'm'
    
    trainer = ASLYOLOTrainer(data_yaml, model_size=model_size)
    
    # Training configuration
    print("\n⚙️  Training Configuration:")
    print("   (Press ENTER to use defaults)")
    
    epochs_input = input("Epochs (default: 100): ").strip()
    epochs = int(epochs_input) if epochs_input else 100
    
    batch_input = input("Batch size (default: 16, reduce to 8 if memory error): ").strip()
    batch_size = int(batch_input) if batch_input else 16
    
    img_input = input("Image size (default: 640): ").strip()
    img_size = int(img_input) if img_input else 640
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Dataset: {data_yaml}")
    print(f"Model: YOLOv8{model_size.upper()}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {img_size}")
    print(f"Device: {trainer.device}")
    print("="*70)
    
    # Confirm
    confirm = input("\n⚠️  Training may take 30 min - 2 hours. Continue? (y/n): ").lower()
    if confirm != 'y':
        print("❌ Training cancelled")
        exit(0)
    
    print("\n🚀 Starting training...\n")
    
    # Start training
    results = trainer.train(
        epochs=epochs,
        img_size=img_size,
        batch_size=batch_size,
        patience=50
    )
    
    if results:
        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETE!")
        print("="*70)
        print("\n📁 Results saved in: runs/asl_train/exp/")
        print("\n📊 Check the following files:")
        print("  • weights/best.pt - Best model weights ⭐")
        print("  • weights/last.pt - Last epoch weights")
        print("  • results.png - Training metrics graph")
        print("  • confusion_matrix.png - Performance analysis")
        print("  • F1_curve.png - F1 score curve")
        print("  • PR_curve.png - Precision-Recall curve")
        
        # Copy model to models folder
        print("\n📦 Copying best model to models folder...")
        trainer.copy_best_model_to_models_folder()
        
        # Evaluate
        print("\n" + "="*70)
        evaluate_now = input("Evaluate model now? (y/n, default: y): ").lower()
        if evaluate_now != 'n':
            trainer.evaluate()
        
        # Export
        export_now = input("\nExport model to ONNX? (y/n, default: n): ").lower()
        if export_now == 'y':
            trainer.export_model()
        
        # Final instructions
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("✅ Model training complete!")
        print(f"✅ Best model saved to: models/best.pt")
        print("\n🚀 To test your model:")
        print("   python sentence_builder.py")
        print("\n📸 Remember to take screenshots for your report!")
        print("="*70)
    else:
        print("\n❌ Training failed. Check errors above.")
    
    print("\nPress ENTER to exit...")
    input()