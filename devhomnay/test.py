import cv2 as cv
import numpy as np
import easyocr
import os

# 1. Configuration & Loading Models
MODELS_DIR = "models"
PROTOTXT = os.path.join(MODELS_DIR, "SSD_MobileNet.prototxt")
MODEL = os.path.join(MODELS_DIR, "SSD_MobileNet.caffemodel")

# Classes MobileNet-SSD can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Targets classes we want to box as "Cars"
TARGET_CLASSES = ["car", "bus", "motorbike", "train"]

print("[INFO] Loading SSD model...")
net = cv.dnn.readNetFromCaffe(PROTOTXT, MODEL)

print("[INFO] Initializing EasyOCR...")
reader = easyocr.Reader(["en"], gpu=True)

# 2. Reading Video
video_path = 'plate2.mp4' 
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print(f"[ERROR] Could not open video: {video_path}")
    exit()

def detect_plate(car_img):
    """Refined plate detection using traditional OpenCV methods"""
    if car_img is None or car_img.size == 0:
        return None, None
    
    h, w = car_img.shape[:2]
    # License plates are usually in the lower half of the car
    roi = car_img[int(h*0.4):h, :]
    
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    # Apply filters to highlight plate-like regions
    bfilter = cv.bilateralFilter(gray, 11, 17, 17)
    edged = cv.Canny(bfilter, 30, 200)
    
    contours, _ = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]
    
    plate_contour = None
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            x, y, w_p, h_p = cv.boundingRect(c)
            aspect_ratio = w_p / float(h_p)
            # Standard license plates have aspect ratio between 2 and 5
            if 2.0 < aspect_ratio < 5.5:
                plate_contour = approx
                break
                
    if plate_contour is not None:
        px, py, pw, ph = cv.boundingRect(plate_contour)
        # Shift py because we are in a ROI
        actual_py = py + int(h*0.4)
        return (px, actual_py, pw, ph), roi[py:py+ph, px:px+pw]
    
    return None, None

print("[INFO] Starting detection loop...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    (H, W) = frame.shape[:2]
    
    # Draw the yellow "counting/detection" line (horizontal)
    line_y = int(H * 0.6)
    cv.line(frame, (0, line_y), (W, line_y), (0, 255, 255), 3)
    
    # --- STEP 1: Detect Vehicles using SSD ---
    # Create blob from frame
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter weak detections
        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])
            
            if CLASSES[idx] in TARGET_CLASSES:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Validation of box
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(W, endX), min(H, endY)
                
                # Draw Car Box (Yellow)
                cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
                
                # --- STEP 2: Detect Plate inside Car BBox ---
                car_roi = frame[startY:endY, startX:endX]
                plate_box, plate_img = detect_plate(car_roi)
                
                if plate_box is not None:
                    px, py, pw, ph = plate_box
                    # Global coordinates for plate
                    g_px, g_py = startX + px, startY + py
                    
                    # Draw Plate Box (Green)
                    cv.rectangle(frame, (g_px, g_py), (g_px + pw, g_py + ph), (0, 255, 0), 2)
                    
                    # OCR
                    if plate_img.size > 0:
                        results = reader.readtext(plate_img, detail=0)
                        plate_text = "".join(results).replace(" ", "").upper()
                        
                        # Labels: "BSX" in red and possibly the text
                        # Draw "BSX" text above the plate box
                        cv.putText(frame, "BSX", (g_px, g_py - 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if plate_text:
                            cv.putText(frame, plate_text, (g_px, g_py - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show result
    cv.imshow('Vehicle & Plate Detection (Non-YOLO)', frame)
    
    # DEBUG: Save one frame to verify aesthetics
    if not os.path.exists("debug_frame.jpg"):
        cv.imwrite("debug_frame.jpg", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
print("[INFO] Finished.")