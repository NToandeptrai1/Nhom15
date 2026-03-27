# 1 import thư viện
import cv2
from ultralytics import YOLO

# 2 khởi tạo model và video

MODEL_PATH = "yolov8m.pt"
VIDEO_PATH = "mvideo.mp4"

# 3 CHỌN LOẠI XE 
VEHICLE_CKASSES = [2, 7]

# 4 CẤU HÌNH THAM GIA
COUNT_DICRECTION = "both"
SHOW_ID = True
SHOW_CONF = True


# 5 load model và video
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# 6 kiêm tra video có mở được không
if not cap.isOpened():
    print("Không mở được video")
    exit()  

# 7 lấy kích thước video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 8 thiết lập vùng đếm và đường đếm
line_y = int(height * 0.7)
roi_x1, roi_y1 = 100, 100
roi_x2, roi_y2 = 600, 400




# 9 Biến lưu trang thái
counted_ids = set()
total_count = 0
previous_centers = {}
frame_count = 0

# 10 vòng lặp chính
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    roi_count = 0
    max_id = -1

    results = model.track(
        frame,
        persist=True,
        classes=VEHICLE_CKASSES,
        verbose=False
    )

    # 11 vẽ đường đếm và vùng đếm
    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)


    
    # 12 vẽ bounding box và thông tin
    if results[0].boxes is not None:
        for box in results[0].boxes:
            if box.id is None:
                continue
            track_id = int(box.id[0].item())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if track_id > max_id:
                max_id = track_id
            
            
            
            # 13 lấy tọa độ trung tâm của bbox
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2:
                roi_count += 1

            
            


            

            # 15 Đổi tên class
            class_name = model.names[cls_id]
            if class_name == "car":
                show_name = "car"
            elif class_name == "truck":
                show_name = "truck"
            else:
                show_name = class_name
            
            
            
            
            # LOGIC ĐẾM XE
            if track_id not in counted_ids:
                prev_cx, prev_cy = previous_centers.get(track_id, (cx, cy))

                if COUNT_DICRECTION == "down":
                    crossed = (prev_cy < line_y <= cy)
                elif COUNT_DICRECTION == "up":
                    crossed = (prev_cy > line_y >= cy)
                else:
                    crossed = (prev_cy < line_y <= cy) or (prev_cy > line_y >= cy)
                
            
            

            # 17 tăng count
            if crossed and track_id not in counted_ids:
                total_count += 1
                counted_ids.add(track_id)
            
            
            
            

            #18 Lưu tọa độ trung tâm hiện tại
            previous_centers[track_id] = (cx, cy)
            # 19 vẽ bbox và thông tin
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lable_parts = [show_name]
            if SHOW_ID:
                lable_parts.append(f"ID:{track_id}")
            if SHOW_CONF:
                lable_parts.append(f"{conf:.2f}")
            label = " | ".join(lable_parts)
            if roi_count >10:
                cv2.putText(frame, "Traffic Jam", (600, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"COUNT: {total_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"ROI COUNT: {roi_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
           
        
        

            
            
        cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27: 
        break
cap.release()
cv2.destroyAllWindows()