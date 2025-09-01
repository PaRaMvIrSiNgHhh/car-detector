from ultralytics import YOLO
import cv2
import cvzone
import math
import time
 
# cap = cv2.VideoCapture(0)# For Webcam
# cap = cv2.VideoCapture("cars.mp4")  

cap = cv2.VideoCapture("bmw.mp4")

model = YOLO("../Yolo-Weights/yolov8l.pt")



 
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
 
prev_frame_time = 0
new_frame_time = 0

 
# mask = cv2.imread("mask.png")

desired_width = 800
desired_height = 900

while True:
    
       
    new_frame_time = time.time()
    success, img = cap.read()
    # print(img.shape, mask.shape)

 # Resize frame
    img = cv2.resize(img, (desired_width, desired_height))

    # Resize mask
    # mask_resized = cv2.resize(mask, (desired_width, desired_height))    
    
    # imgRegion = cv2.bitwise_and(img,mask_resized)
    
    currentClass =("car" , "truck" ,"bus" ,"motorbike")
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
           
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass =classNames[cls]
            
            
            if  conf > 0.5 and currentClass:
                 cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset = 2)
                 cvzone.cornerRect(img, (x1, y1, w, h), l= 8)
 
           
 
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)
    
 
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
     # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
                                    












# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
# import time
 
# # cap = cv2.VideoCapture(0)# For Webcam
# # cap = cv2.VideoCapture("cars.mp4")  

# cap = cv2.VideoCapture("traffic.mp4")
# cap.set(3, 1280)
# cap.set(4, 1980)
# # cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video
 
 
# model = YOLO("../Yolo-Weights/yolov8n.pt")
 
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
#               ]
 
# prev_frame_time = 0
# new_frame_time = 0

 
# mask = cv2.imread("mask.png")



# while True:
    
       
#     new_frame_time = time.time()
#     success, img = cap.read()
#     print(img.shape, mask.shape)
#     desired_width = 1280
#     desired_height = 720
#  # Resize frame
#     img = cv2.resize(img, (desired_width, desired_height))

#     # Resize mask
#     mask_resized = cv2.resize(mask, (desired_width, desired_height))    
    
#     imgRegion = cv2.bitwise_and(img,mask_resized)
    
    
#     results = model(imgRegion, stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding Box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
#             w, h = x2 - x1, y2 - y1
           
#             # Confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100
#             # Class Name
#             cls = int(box.cls[0])
#             currentClass =classNames[cls]
            
            
#             if  conf > 0.6 and (currentClass == "car" or currentClass == "truck" or  currentClass == "bus" or currentClass == "motorbike") :
#                  cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1, offset = 2)
#                  cvzone.cornerRect(img, (x1, y1, w, h), l= 3)
 
           
 
#     fps = 1 / (new_frame_time - prev_frame_time)
#     prev_frame_time = new_frame_time
#     print(fps)
    
 
#     cv2.imshow("Image", img)
#     # cv2.imshow("ImageRegion", imgRegion)
#      # Exit when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
                                    