"""
Real-Time ASL Sign Language Recognition System - OPTIMIZED VERSION
Features: Live detection, text display, speech output, maximum performance
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import pyttsx3
from datetime import datetime
import os
import threading


class ASLDetector:
    def __init__(self, model_path, conf_threshold=0.25, use_speech=True):
        """
        Initialize ASL Detector
        
        Args:
            model_path: Path to trained YOLO weights
            conf_threshold: Confidence threshold for detection
            use_speech: Enable text-to-speech
        """
        print("🚀 Initializing ASL Detector...")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        print("✅ MODEL LOADED FROM:", model_path)
        print("🧠 MODEL CLASSES:", self.model.names)
        self.conf_threshold = conf_threshold
        
        # Text-to-speech setup
        self.use_speech = use_speech
        self.speech_queue = []
        if use_speech:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
                # Start speech thread to avoid blocking
                self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
                self.speech_thread.start()
                print("🔊 Text-to-speech enabled")
            except:
                print("⚠️  Text-to-speech unavailable")
                self.use_speech = False
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_prediction = None
        self.last_prediction_time = 0
        self.prediction_cooldown = 1.5  # Reduced from 2.0 for faster response
        
        # Detection history for stability (increased for smoother predictions)
        self.detection_history = deque(maxlen=15)
        
        # Statistics
        self.total_detections = 0
        self.detection_counts = {}
        
        # Colors for display
        self.colors = self._generate_colors()
        
        # Frame skip for optimization (process every N frames)
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_count = 0
        self.last_result = None
        
        print("✅ Detector ready!")
    
    def _generate_colors(self):
        """Generate distinct colors for each class"""
        np.random.seed(42)
        class_names = self.model.names
        colors = {}
        for class_id in class_names:
            # Bright, high-contrast colors
            colors[class_id] = tuple(map(int, np.random.randint(100, 255, 3)))
        return colors
    
    def _speech_worker(self):
        """Background thread for text-to-speech to avoid blocking"""
        while True:
            if self.speech_queue:
                text = self.speech_queue.pop(0)
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except:
                    pass
            time.sleep(0.1)
    
    def _smooth_prediction(self, current_pred):
        """Smooth predictions to reduce jitter - improved algorithm"""
        if current_pred is None:
            return None
        
        self.detection_history.append(current_pred)
        
        # Need at least 7 detections for smoothing
        if len(self.detection_history) < 7:
            return current_pred
        
        # Look at recent history (last 10 frames)
        recent = list(self.detection_history)[-10:]
        pred_counts = {}
        for pred in recent:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        # Return most common prediction if it appears 5+ times (50%+)
        most_common = max(pred_counts, key=pred_counts.get)
        if pred_counts[most_common] >= 5:
            return most_common
        
        return current_pred
    
    def speak(self, text):
        """Convert text to speech (non-blocking)"""
        if self.use_speech and text and text not in self.speech_queue:
            self.speech_queue.append(text)
    
    def draw_info_panel(self, frame):
        """Draw information panel on frame - optimized"""
        h, w = frame.shape[:2]
        panel_height = 180
        
        # Semi-transparent panel (optimized alpha blend)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Title with shadow for better visibility
        shadow_color = (0, 0, 0)
        text_color = (0, 255, 255)
        cv2.putText(frame, "ASL Sign Language Recognition", 
                   (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, shadow_color, 3)
        cv2.putText(frame, "ASL Sign Language Recognition", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, text_color, 2)
        
        # FPS with color coding
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            fps_color = (0, 255, 0) if avg_fps > 20 else (0, 200, 255) if avg_fps > 15 else (0, 100, 255)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", 
                       (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # Current prediction with emphasis
        if self.last_prediction:
            cv2.putText(frame, f"Detected: {self.last_prediction.upper()}", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No detection", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        # Total detections
        cv2.putText(frame, f"Total: {self.total_detections}", 
                   (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Instructions at bottom
        cv2.putText(frame, "Q:Quit | S:Screenshot | R:Reset | T:Speech", 
                   (w - 600, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_statistics_panel(self, frame):
        """Draw statistics sidebar - optimized"""
        h, w = frame.shape[:2]
        panel_width = 280
        
        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_width, 0), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, "Statistics", 
                   (w - panel_width + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Detection counts
        y_pos = 70
        sorted_counts = sorted(self.detection_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (gesture, count) in enumerate(sorted_counts[:8]):
            percentage = (count / self.total_detections * 100) if self.total_detections > 0 else 0
            text = f"{gesture}: {count} ({percentage:.0f}%)"
            cv2.putText(frame, text, 
                       (w - panel_width + 20, y_pos + i * 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        return frame
    
    def detect_frame(self, frame):
        """Detect signs in a single frame - optimized with frame skipping"""
        self.frame_count += 1
        
        # Frame skipping for performance (process every Nth frame)
        if self.frame_count % self.frame_skip != 0 and self.last_result is not None:
            # Use cached result from last detection
            return self._draw_cached_detections(frame)
        
        # Run inference with optimized settings
        results = self.model(
            frame, 
            conf=self.conf_threshold, 
            verbose=False,
            imgsz=416,  # Optimal balance between speed and accuracy
            half=False,  # Use FP16 if GPU available (automatic)
            device='cpu'  # Change to 'cuda' or 0 if you have GPU
        )
        
        # Store result for frame skipping
        self.last_result = results
        
        current_prediction = None
        max_conf = 0
        detected_boxes = []
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                
                detected_boxes.append({
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'cls': cls,
                    'name': class_name
                })
                
                # Track highest confidence detection
                if conf > max_conf:
                    max_conf = conf
                    current_prediction = class_name
        
        # Draw all detected boxes
        for det in detected_boxes:
            x1, y1, x2, y2 = det['box']
            color = self.colors[det['cls']]
            
            # Draw bounding box with thickness based on confidence
            thickness = 3 if det['conf'] > 0.7 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with background
            label = f"{det['name']} {det['conf']:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Label background
            cv2.rectangle(frame, (x1, y1 - label_h - 8), 
                         (x1 + label_w + 6, y1), color, -1)
            # Label text
            cv2.putText(frame, label, (x1 + 3, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Smooth prediction
        smoothed_pred = self._smooth_prediction(current_prediction)
        
        # Update statistics and speak if new prediction
        current_time = time.time()
        if smoothed_pred and smoothed_pred != self.last_prediction:
            if current_time - self.last_prediction_time > self.prediction_cooldown:
                self.last_prediction = smoothed_pred
                self.last_prediction_time = current_time
                self.total_detections += 1
                self.detection_counts[smoothed_pred] = self.detection_counts.get(smoothed_pred, 0) + 1
                
                # Speak prediction (non-blocking)
                self.speak(smoothed_pred)
        
        return frame
    
    def _draw_cached_detections(self, frame):
        """Draw detections from cached results (for skipped frames)"""
        if not self.last_result:
            return frame
        
        for result in self.last_result:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                color = self.colors[cls]
                
                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_h - 8), 
                             (x1 + label_w + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self, camera_id=0, show_stats=True):
        """Run real-time detection"""
        print(f"\n🎥 Starting camera {camera_id}...")
        
        cap = cv2.VideoCapture(camera_id)
        
        # Optimized camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return
        
        print("✅ Camera ready!")
        print("\n⚡ Performance Optimizations:")
        print(f"  • Frame skip: Every {self.frame_skip} frames")
        print(f"  • Inference size: 416x416")
        print(f"  • Confidence threshold: {self.conf_threshold}")
        print("\n🎮 Controls:")
        print("  Q - Quit")
        print("  S - Screenshot")
        print("  R - Reset statistics")
        print("  T - Toggle speech")
        print("  + - Increase frame skip (faster, less smooth)")
        print("  - - Decrease frame skip (slower, more smooth)")
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Run detection
            frame = self.detect_frame(frame)
            
            # Draw UI
            frame = self.draw_info_panel(frame)
            if show_stats:
                frame = self.draw_statistics_panel(frame)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            self.fps_history.append(fps)
            
            # Display
            cv2.imshow('ASL Sign Language Recognition', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('r'):
                # Reset statistics
                self.total_detections = 0
                self.detection_counts = {}
                self.detection_history.clear()
                print("🔄 Statistics reset")
            elif key == ord('t'):
                # Toggle speech
                self.use_speech = not self.use_speech
                status = "enabled" if self.use_speech else "disabled"
                print(f"🔊 Speech {status}")
            elif key == ord('+') or key == ord('='):
                # Increase frame skip (faster)
                self.frame_skip = min(self.frame_skip + 1, 5)
                print(f"⚡ Frame skip: {self.frame_skip} (faster)")
            elif key == ord('-') or key == ord('_'):
                # Decrease frame skip (smoother)
                self.frame_skip = max(self.frame_skip - 1, 1)
                print(f"⚡ Frame skip: {self.frame_skip} (smoother)")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        self._print_session_stats()
    
    def _print_session_stats(self):
        """Print session statistics"""
        print("\n" + "="*60)
        print("SESSION STATISTICS")
        print("="*60)
        print(f"Total Detections: {self.total_detections}")
        
        if self.detection_counts:
            print("\nGesture Frequency:")
            for gesture, count in sorted(self.detection_counts.items(), 
                                        key=lambda x: x[1], reverse=True):
                percentage = (count / self.total_detections) * 100
                print(f"  {gesture:15s}: {count:4d} ({percentage:.1f}%)")
        
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            min_fps = min(self.fps_history)
            max_fps = max(self.fps_history)
            print(f"\nFPS Statistics:")
            print(f"  Average: {avg_fps:.2f}")
            print(f"  Min: {min_fps:.2f}")
            print(f"  Max: {max_fps:.2f}")
        
        print("="*60)


if __name__ == "__main__":
    print("="*70)
    print("🤖 Real-Time ASL Sign Language Recognition - OPTIMIZED")
    print("="*70)
    
    # Check for model weights
    default_weights = "models/best.pt"
    
    if not os.path.exists(default_weights):
        print(f"\n❌ Model weights not found at: {default_weights}")
        print("\n📋 Searching for model in other locations...")
        
        # Try alternative locations
        alternatives = [
            "runs/asl_train/exp/weights/best.pt",
            "models/best.pt"
        ]
        
        found = False
        for alt_path in alternatives:
            if os.path.exists(alt_path):
                default_weights = alt_path
                print(f"✅ Found model at: {alt_path}")
                found = True
                break
        
        if not found:
            weights_path = input("\nEnter path to weights file (or press ENTER to exit): ").strip()
            if not weights_path or not os.path.exists(weights_path):
                print("❌ Exiting...")
                exit(0)
            default_weights = weights_path
    else:
        print(f"✅ Found model at: {default_weights}")
    
    # Configuration
    print(f"\n📦 Loading model: {default_weights}")
    
    # Ask for settings
    print("\n⚙️  Configuration:")
    conf_input = input("Confidence threshold (0.0-1.0, default 0.25): ").strip()
    conf = float(conf_input) if conf_input else 0.25
    
    use_speech = input("Enable text-to-speech? (y/n, default y): ").lower() != 'n'
    show_stats = input("Show statistics panel? (y/n, default y): ").lower() != 'n'
    
    print("\n💡 TIP: For maximum speed, close other programs!")
    print("💡 TIP: Press + or - during detection to adjust frame skip")
    
    # Initialize detector
    detector = ASLDetector(default_weights, conf_threshold=conf, use_speech=use_speech)
    
    # Start detection
    print("\n🎬 Starting real-time detection...")
    input("Press ENTER to begin...")
    
    try:
        detector.run(camera_id=0, show_stats=show_stats)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Goodbye!")