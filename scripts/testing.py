"""
Quick Model Test - Check if model works
"""

from ultralytics import YOLO
import cv2

print("="*60)
print("MODEL DIAGNOSTIC TEST")
print("="*60)

# Load model
print("\n1. Loading model...")
try:
    model = YOLO('models/best.pt')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Check model info
print("\n2. Model Information:")
print(f"   Classes: {model.names}")
print(f"   Number of classes: {len(model.names)}")

# Test on camera
print("\n3. Testing detection...")
print("   Opening camera...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera!")
    exit()

print("✅ Camera opened!")
print("\n4. Running detection test...")
print("   Make your gesture and look at terminal output!")
print("   Press 'q' to quit\n")

frame_count = 0
detection_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Run detection with VERY LOW threshold
    results = model(frame, conf=0.1, verbose=False)
    
    frame_count += 1
    
    # Check detections
    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            detection_count += 1
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                print(f"Frame {frame_count}: DETECTED! {class_name} (confidence: {conf:.2f})")
                
                # Draw box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('Model Test', frame)
    
    # Every 30 frames, show status
    if frame_count % 30 == 0:
        print(f"Status: {detection_count} detections in {frame_count} frames ({detection_count/frame_count*100:.1f}%)")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print(f"Total frames: {frame_count}")
print(f"Detections: {detection_count}")
print(f"Detection rate: {detection_count/frame_count*100:.1f}%")
print("="*60)

if detection_count == 0:
    print("\n⚠️  NO DETECTIONS!")
    print("\nPossible reasons:")
    print("1. Model trained on too few images (8 is not enough)")
    print("2. Current gesture doesn't match training data")
    print("3. Poor lighting conditions")
    print("4. Hand not positioned correctly")
    print("\n💡 SOLUTION: Train on all 120 images!")
else:
    print("\n✅ Model is working!")
    if detection_count/frame_count < 0.3:
        print("⚠️  But detection rate is low - consider retraining with more images")