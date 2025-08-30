from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# --- Video Capture ---
cap = cv2.VideoCapture("traffic.mp4")

# --- YOLO Model ---
model = YOLO("../Yolo-Weights/yolov8l.pt")

# --- Class Names ---
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

vehicle_classes = ["car", "truck", "bus", "motorbike"]

# --- Mask ---
mask = cv2.imread("mask.png")
if mask is None:
    raise FileNotFoundError("mask.png not found")

# --- FPS ---
prev_frame_time = 0

# --- Desired frame size ---
desired_width = 1280
desired_height = 720

# --- Main Loop ---
while True:
    success, img = cap.read()
    if not success:
        break

    # Resize frame
    img = cv2.resize(img, (desired_width, desired_height))

    # Resize mask
    mask_resized = cv2.resize(mask, (desired_width, desired_height))

    # Apply mask
    imgRegion = cv2.bitwise_and(img, mask_resized)

    # YOLO Detection
    results = model(imgRegion, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Draw if vehicle
            if conf > 0.6 and currentClass in vehicle_classes:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                   scale=0.7, thickness=1, offset=2)
                cvzone.cornerRect(img, (x1, y1, w, h), l=3)

    # FPS calculation
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
    prev_frame_time = new_frame_time
    print(f"FPS: {fps:.2f}")

    # Show frames
    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
