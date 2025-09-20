import cv2
import numpy as np
from ultralytics import YOLO
import os

# Manual dataset location (update this path after unzipping the downloaded dataset)
dataset_location = r"C:\Users\91902\Documents\Arrow Dataset"  # Adjust to your actual path (use the unzipped folder name)

# Train the model using yolo12s.pt (or yolov8s.pt if YOLOv12 isn't available)
model_path = "yolov8s.pt" 

try:
    model = YOLO(model_path)
    results = model.train(data=f"{dataset_location}/data.yaml", epochs=2, imgsz=320)  # Quick test settings
except Exception as e:
    print(f"Error during training: {e}")
    exit()

# Load the trained model for inference (use dynamic path from training results)
trained_model_path = os.path.join(results.save_dir, "weights", "best.pt")
if not os.path.exists(trained_model_path):
    print(f"Trained model not found at {trained_model_path}. Ensure training completed successfully.")
    exit()
model = YOLO(trained_model_path)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture the frame")
        break

    # Shadow rejection preprocessing
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gframe, (7, 7), 0)  # Strong blur for shadow smoothing
    contrast = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhancedframe = contrast.apply(gaussian)
    thresh = cv2.adaptiveThreshold(
        enhancedframe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    # Convert thresholded image back to BGR for YOLO compatibility
    frame_preprocessed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # Run YOLO inference
    results = model(frame_preprocessed)

    direction = "NOTDEFINED"
    max_conf = 0.0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            if conf > 0.2:  # Confidence threshold
                dir_name = model.names[cls_id].upper()  # e.g., 'LEFT' or 'RIGHT' (adjusted based on your dataset classes)
                if conf > max_conf:
                    max_conf = conf
                    direction = dir_name.replace('_ARROW', '')  # Simplify to 'LEFT' or 'RIGHT'

                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{direction} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(direction)

    if max_conf > 0:
        cv2.putText(
            frame,
            f"Decision: {direction} ({max_conf:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    # Display the frame locally
    cv2.imshow('Arrow Detection with YOLO', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()