from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import yaml

# --- Video Capture ---
cap = cv2.VideoCapture("traffic.mp4")

# --- Load class names from your custom data.yaml ---
with open("auto.v1i.yolov11/data.yaml", "r") as f:
    data = yaml.safe_load(f)
classNames = data['names']  # your custom class names from training

# Adjust this list to exactly match your class names for filtering vehicles
vehicle_classes = ["car", "autorickshaw", "bike", "truck", "bus"]

# --- Load your trained YOLO model ---
model = YOLO("C:/Users/paramvir singh/runs/detect/train4/weights/best.pt")

# --- Load mask ---
mask = cv2.imread("mask.png")
if mask is None:
    raise FileNotFoundError("mask.png not found")

# --- FPS calculation variables ---
prev_frame_time = 0

# --- Desired frame size ---
desired_width = 1280
desired_height = 720

# --- Main Loop ---
while True:
    success, img = cap.read()
    if not success:
        break

    # Resize frame and mask
    img = cv2.resize(img, (desired_width, desired_height))
    mask_resized = cv2.resize(mask, (desired_width, desired_height))

    # Apply mask (comment out for debugging if needed)
    imgRegion = cv2.bitwise_and(img, mask_resized)
    # If you want to test without mask, uncomment the next line:
    # imgRegion = img

    # YOLO detection
    results = model(imgRegion, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence and class
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Debug print for detections
            print(f"Detected: {currentClass} with confidence {conf}")

            # Draw bounding box and label if confidence is high and class is vehicle
            if conf > 0.3 and currentClass in vehicle_classes:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                   scale=1, thickness=1, offset=2)
                cvzone.cornerRect(img, (x1, y1, w, h), l=7)

    # Calculate and print FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
    prev_frame_time = new_frame_time
    print(f"FPS: {fps:.2f}")

    # Show frames
    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
