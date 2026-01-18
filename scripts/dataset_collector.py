"""
ASL Sign Language Dataset Collection Tool
Captures images for YOLO training with automatic organization
"""

import cv2
import os
from datetime import datetime
import time

class DatasetCollector:
    def __init__(self, base_dir="asl_dataset"):
        self.base_dir = base_dir
        self.gestures = [
            "Hello", "Thank_You", "Yes", "No", "Please",
            "Sorry", "Help", "Stop", "Love", "Good"
        ]
        self.current_gesture = 0
        self.image_count = 0
        self.setup_directories()
        
    def setup_directories(self):
        """Create directory structure for dataset"""
        for gesture in self.gestures:
            path = os.path.join(self.base_dir, "raw_images", gesture)
            os.makedirs(path, exist_ok=True)
        print(f"✅ Created directories for {len(self.gestures)} gestures")
    
    def collect_images(self, images_per_gesture=120, countdown=3):
        """Capture images for each gesture"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ Cannot access webcam")
            return
        
        print("🎥 Webcam ready! Press 'n' for next gesture, 'q' to quit")
        
        for gesture in self.gestures:
            self.image_count = 0
            gesture_dir = os.path.join(self.base_dir, "raw_images", gesture)
            
            # Wait for user to get ready
            print(f"\n📸 Prepare for gesture: {gesture}")
            print(f"   Target: {images_per_gesture} images")
            
            ready = False
            while not ready:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Display instructions
                display_frame = frame.copy()
                cv2.putText(display_frame, f"GESTURE: {gesture}", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(display_frame, "Press SPACE when ready", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(display_frame, "Press 'n' to skip | 'q' to quit", 
                           (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                
                cv2.imshow('Dataset Collection', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    ready = True
                elif key == ord('n'):
                    print(f"⏭️  Skipped {gesture}")
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            
            if not ready:
                continue
            
            # Countdown
            for i in range(countdown, 0, -1):
                ret, frame = cap.read()
                if ret:
                    display_frame = frame.copy()
                    cv2.putText(display_frame, str(i), 
                               (frame.shape[1]//2 - 50, frame.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
                    cv2.imshow('Dataset Collection', display_frame)
                    cv2.waitKey(1000)
            
            # Capture images
            print(f"📹 Recording {gesture}...")
            while self.image_count < images_per_gesture:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{gesture}_{timestamp}_{self.image_count:04d}.jpg"
                filepath = os.path.join(gesture_dir, filename)
                cv2.imwrite(filepath, frame)
                
                # Display progress
                display_frame = frame.copy()
                progress = (self.image_count / images_per_gesture) * 100
                cv2.putText(display_frame, f"CAPTURING: {gesture}", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Progress: {self.image_count}/{images_per_gesture} ({progress:.1f}%)", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Progress bar
                bar_width = 600
                bar_height = 30
                bar_x, bar_y = 50, 130
                cv2.rectangle(display_frame, (bar_x, bar_y), 
                            (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
                fill_width = int(bar_width * (self.image_count / images_per_gesture))
                cv2.rectangle(display_frame, (bar_x, bar_y), 
                            (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
                
                cv2.imshow('Dataset Collection', display_frame)
                
                self.image_count += 1
                
                # Small delay for variety
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
            
            print(f"✅ Captured {self.image_count} images for {gesture}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n🎉 Dataset collection complete!")
        self.generate_summary()
    
    def generate_summary(self):
        """Generate dataset summary"""
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        total_images = 0
        for gesture in self.gestures:
            path = os.path.join(self.base_dir, "raw_images", gesture)
            count = len([f for f in os.listdir(path) if f.endswith('.jpg')])
            total_images += count
            print(f"{gesture:15s}: {count:4d} images")
        print("-"*60)
        print(f"{'TOTAL':15s}: {total_images:4d} images")
        print("="*60)
        print(f"\n📁 Dataset saved in: {self.base_dir}/raw_images/")
        print("\n📋 NEXT STEPS:")
        print("1. Upload images to Roboflow (https://roboflow.com)")
        print("2. Annotate with bounding boxes")
        print("3. Export in YOLOv8 format")
        print("4. Train the model!")

if __name__ == "__main__":
    print("🤖 ASL Dataset Collection Tool")
    print("="*60)
    
    collector = DatasetCollector()
    
    # Customize settings
    images_per_gesture = int(input("Images per gesture (default 120): ") or "120")
    countdown = int(input("Countdown seconds (default 3): ") or "3")
    
    print("\n🎬 Starting collection...")
    print("TIPS:")
    print("  • Use good lighting")
    print("  • Vary hand positions slightly")
    print("  • Keep hand in center of frame")
    print("  • Try different backgrounds")
    
    input("\nPress ENTER to start...")
    collector.collect_images(images_per_gesture, countdown)