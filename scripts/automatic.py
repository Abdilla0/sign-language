"""
Real-Time ASL Sign Language Recognition with Sentence Building
Features: Live detection, sentence construction, text-to-speech, word history
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


class ASLSentenceBuilder:
    def __init__(self, model_path, conf_threshold=0.25, use_speech=True):
        """
        Initialize ASL Sentence Builder
        
        Args:
            model_path: Path to trained YOLO weights
            conf_threshold: Confidence threshold for detection
            use_speech: Enable text-to-speech
        """
        print("🚀 Initializing ASL Sentence Builder...")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        print("✅ MODEL LOADED FROM:", model_path)
        print("🧠 MODEL CLASSES:", self.model.names)
        self.conf_threshold = conf_threshold
        
        # Sentence building
        self.sentence = []  # List of detected words
        self.sentence_display = ""  # Current sentence as string
        self.word_history = deque(maxlen=20)  # History of all detected words
        
        # Text-to-speech setup
        self.use_speech = use_speech
        self.speech_queue = []
        if use_speech:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
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
        self.prediction_cooldown = 0.8  # Short cooldown for same word repeat
        
        # Detection history for stability
        self.detection_history = deque(maxlen=15)
        
        # Allow repeating same word easily
        self.allow_duplicates = True  # Easy duplicate word addition
        
        # Statistics
        self.total_words = 0
        self.word_counts = {}
        
        # Colors for display
        self.colors = self._generate_colors()
        
        # Frame optimization
        self.frame_skip = 2
        self.frame_count = 0
        self.last_result = None
        
        print("✅ Sentence Builder ready!")
        print("📝 Words will be added to build sentences!")
    
    def _generate_colors(self):
        """Generate distinct colors for each class"""
        np.random.seed(42)
        class_names = self.model.names
        colors = {}
        for class_id in class_names:
            colors[class_id] = tuple(map(int, np.random.randint(100, 255, 3)))
        return colors
    
    def _speech_worker(self):
        """Background thread for text-to-speech"""
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
        """Smooth predictions to reduce jitter"""
        if current_pred is None:
            return None
        
        self.detection_history.append(current_pred)
        
        # Need fewer frames for faster response
        if len(self.detection_history) < 4:  # REDUCED from 7
            return current_pred
        
        # Look at recent history (last 6 frames)
        recent = list(self.detection_history)[-6:]
        pred_counts = {}
        for pred in recent:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        # Return most common prediction if it appears 3+ times (50%+)
        most_common = max(pred_counts, key=pred_counts.get)
        if pred_counts[most_common] >= 3:
            return most_common
        
        return current_pred
    
    def add_word_to_sentence(self, word):
        """Add a new word to the sentence"""
        # Add to sentence list
        self.sentence.append(word)
        
        # Add to word history
        self.word_history.append({
            'word': word,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        # Update sentence display
        self.sentence_display = " ".join(self.sentence)
        
        # Update statistics
        self.total_words += 1
        self.word_counts[word] = self.word_counts.get(word, 0) + 1
        
        print(f"📝 Added word: '{word}' | Sentence: '{self.sentence_display}'")
        
        # Speak the word
        self.speak(word)
    
    def clear_sentence(self):
        """Clear current sentence"""
        self.sentence = []
        self.sentence_display = ""
        print("🗑️  Sentence cleared")
    
    def delete_last_word(self):
        """Delete last word from sentence"""
        if self.sentence:
            removed = self.sentence.pop()
            self.sentence_display = " ".join(self.sentence)
            print(f"⬅️  Removed: '{removed}' | Sentence: '{self.sentence_display}'")
    
    def speak_sentence(self):
        """Speak the complete sentence"""
        if self.sentence_display:
            print(f"🔊 Speaking: '{self.sentence_display}'")
            self.speak(self.sentence_display)
        else:
            print("⚠️  No sentence to speak")
    
    def speak(self, text):
        """Convert text to speech (non-blocking)"""
        if self.use_speech and text and text not in self.speech_queue:
            self.speech_queue.append(text)
    
    def draw_sentence_panel(self, frame):
        """Draw sentence building panel at top"""
        h, w = frame.shape[:2]
        panel_height = 150
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "ASL Sentence Builder", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Current word being detected
        if self.last_prediction:
            cv2.putText(frame, f"Detecting: {self.last_prediction.upper()}", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Waiting for sign...", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        
        # Current sentence (large and prominent)
        sentence_text = self.sentence_display if self.sentence_display else "[Empty Sentence]"
        
        # Word wrap for long sentences
        max_chars = 50
        if len(sentence_text) > max_chars:
            # Split into multiple lines
            words = sentence_text.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + " " + word) <= max_chars:
                    current_line += (" " + word) if current_line else word
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            # Draw lines
            y_pos = 105
            for line in lines[:2]:  # Max 2 lines
                cv2.putText(frame, line, 
                           (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                y_pos += 30
        else:
            cv2.putText(frame, sentence_text, 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        
        return frame
    
    def draw_controls_panel(self, frame):
        """Draw control instructions at bottom"""
        h, w = frame.shape[:2]
        
        # Instructions
        instructions = [
            "⚡ Change signs → Instant! | 🔁 Same sign → Hold 0.8s → Repeats!",
            "BACKSPACE: Delete | C: Clear | ENTER: Speak | S: Screenshot | Q: Quit"
        ]
        
        y_start = h - 55
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, 
                       (20, y_start + i*22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def draw_word_history_panel(self, frame):
        """Draw word history sidebar"""
        h, w = frame.shape[:2]
        panel_width = 280
        
        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_width, 0), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Word History", 
                   (w - panel_width + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            fps_color = (0, 255, 0) if avg_fps > 20 else (0, 200, 255)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", 
                       (w - panel_width + 15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        
        # Total words
        cv2.putText(frame, f"Total Words: {self.total_words}", 
                   (w - panel_width + 15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Recent words
        cv2.putText(frame, "Recent:", 
                   (w - panel_width + 15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        y_pos = 150
        recent_words = list(self.word_history)[-10:]  # Last 10 words
        recent_words.reverse()  # Show newest first
        
        for i, entry in enumerate(recent_words):
            word = entry['word']
            timestamp = entry['timestamp']
            text = f"{timestamp} - {word}"
            cv2.putText(frame, text, 
                       (w - panel_width + 20, y_pos + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # Word frequency
        if self.word_counts:
            cv2.putText(frame, "Frequency:", 
                       (w - panel_width + 15, h - 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            y_pos = h - 155
            sorted_counts = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
            for i, (word, count) in enumerate(sorted_counts[:5]):
                text = f"{word}: {count}"
                cv2.putText(frame, text, 
                           (w - panel_width + 20, y_pos + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        return frame
    
    def detect_frame(self, frame):
        """Detect signs in a single frame"""
        self.frame_count += 1
        
        # Frame skipping for performance
        if self.frame_count % self.frame_skip != 0 and self.last_result is not None:
            return self._draw_cached_detections(frame)
        
        # Run inference
        results = self.model(
            frame, 
            conf=self.conf_threshold, 
            verbose=False,
            imgsz=416,
            device='cpu'  # Change to 'cuda' or 0 if GPU available
        )
        
        self.last_result = results
        
        current_prediction = None
        max_conf = 0
        detected_boxes = []
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
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
                
                if conf > max_conf:
                    max_conf = conf
                    current_prediction = class_name
        
        # Draw detected boxes
        for det in detected_boxes:
            x1, y1, x2, y2 = det['box']
            color = self.colors[det['cls']]
            thickness = 3 if det['conf'] > 0.7 else 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{det['name']} {det['conf']:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(frame, (x1, y1 - label_h - 8), 
                         (x1 + label_w + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Update current prediction with EASY DUPLICATE SUPPORT
        smoothed_pred = self._smooth_prediction(current_prediction)
        
        current_time = time.time()
        
        # Add word if it's different OR cooldown passed (allows easy duplicates!)
        if smoothed_pred:
            # NEW WORD: Add immediately
            if smoothed_pred != self.last_prediction:
                self.last_prediction = smoothed_pred
                self.last_prediction_time = current_time
                self.add_word_to_sentence(smoothed_pred)
            
            # SAME WORD: Add if cooldown passed (0.8 seconds)
            elif current_time - self.last_prediction_time > self.prediction_cooldown:
                self.last_prediction_time = current_time
                self.add_word_to_sentence(smoothed_pred)
                print(f"🔁 Repeated word: {smoothed_pred}")
        
        return frame
    
    def _draw_cached_detections(self, frame):
        """Draw detections from cached results"""
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
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_h - 8), 
                             (x1 + label_w + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self, camera_id=0, show_history=True):
        """Run real-time sentence building"""
        print(f"\n🎥 Starting camera {camera_id}...")
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return
        
        print("✅ Camera ready!")
        print("\n🎮 DYNAMIC SENTENCE BUILDING:")
        print("  ✨ Words added instantly when you change signs!")
        print("  🔁 SAME WORD? Just keep holding for 0.8 seconds - adds again!")
        print("\n  BACKSPACE   - Delete last word")
        print("  C           - Clear entire sentence")
        print("  ENTER       - Speak complete sentence")
        print("  S           - Screenshot")
        print("  Q           - Quit")
        print("\n💡 TIP: Different signs → Instant add!")
        print("💡 TIP: Same sign → Hold 0.8s → Adds duplicate!")
        print("\n🔥 Example: 'Please' (hold) → 'Please' (hold) → 'Please' = 'Please Please Please'")
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Run detection
            frame = self.detect_frame(frame)
            
            # Draw UI panels
            frame = self.draw_sentence_panel(frame)
            frame = self.draw_controls_panel(frame)
            if show_history:
                frame = self.draw_word_history_panel(frame)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            self.fps_history.append(fps)
            
            # Display
            cv2.imshow('ASL Sentence Builder', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == 8 or key == 127:  # BACKSPACE - Delete last word
                self.delete_last_word()
            elif key == ord('c') or key == ord('C'):  # C - Clear sentence
                self.clear_sentence()
            elif key == 13 or key == 10:  # ENTER - Speak sentence
                self.speak_sentence()
            elif key == ord('s'):  # S - Screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sentence_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('t'):  # T - Toggle speech
                self.use_speech = not self.use_speech
                status = "enabled" if self.use_speech else "disabled"
                print(f"🔊 Speech {status}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        self._print_session_stats()
    
    def _print_session_stats(self):
        """Print session statistics"""
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Final Sentence: {self.sentence_display if self.sentence_display else '[Empty]'}")
        print(f"Total Words Added: {self.total_words}")
        
        if self.word_counts:
            print("\nWord Frequency:")
            for word, count in sorted(self.word_counts.items(), 
                                     key=lambda x: x[1], reverse=True):
                print(f"  {word:15s}: {count:4d}")
        
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            print(f"\nAverage FPS: {avg_fps:.2f}")
        
        print("="*60)


if __name__ == "__main__":
    print("="*70)
    print("🤖 ASL Sentence Builder - Real-Time Sign Language Recognition")
    print("="*70)
    
    # Find model
    default_weights = "models/best.pt"
    
    if not os.path.exists(default_weights):
        alternatives = ["runs/asl_train/exp/weights/best.pt", "../models/best.pt"]
        for alt in alternatives:
            if os.path.exists(alt):
                default_weights = alt
                break
        else:
            weights_path = input("\nEnter path to model weights: ").strip()
            if not weights_path or not os.path.exists(weights_path):
                print("❌ Model not found. Exiting...")
                exit(0)
            default_weights = weights_path
    
    print(f"✅ Found model: {default_weights}")
    
    # Configuration
    print("\n⚙️  Configuration:")
    conf_input = input("Confidence threshold (0.0-1.0, default 0.25): ").strip()
    conf = float(conf_input) if conf_input else 0.25
    
    use_speech = input("Enable text-to-speech? (y/n, default y): ").lower() != 'n'
    show_history = input("Show word history panel? (y/n, default y): ").lower() != 'n'
    
    # Initialize
    builder = ASLSentenceBuilder(default_weights, conf_threshold=conf, use_speech=use_speech)
    
    print("\n🎬 Starting DYNAMIC sentence builder...")
    print("⚡ Words added INSTANTLY when you change signs - no delays!")
    print("🔥 Try it: Change between signs quickly and watch the sentence flow!")
    input("Press ENTER to begin...")
    
    try:
        builder.run(camera_id=0, show_history=show_history)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Goodbye!")