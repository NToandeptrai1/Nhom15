import cv2
from ultralytics import YOLO

# --- 1. LOAD CẢ 2 MODEL ---
model_dog = YOLO("day6/dog.pt")      # Model nhận diện chó
model_chicken = YOLO("day6/chicken.pt") # Model nhận diện gà

# --- 2. MỞ VIDEO/WEBCAM ---
video_path = "day6/dogs.mp4" 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Không thể mở file video hoặc webcam!")
    exit()

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # --- 3. NHẬN DIỆN VỚI MODEL DOG ---
        # results_dog là danh sách kết quả cho model chó
        results_dog = model_dog(frame, conf=0.5, verbose=False)
        
        # Vẽ kết quả của chó lên frame gốc
        annotated_frame = results_dog[0].plot()

        # --- 4. NHẬN DIỆN VỚI MODEL CHICKEN ---
        # Chạy nhận diện gà trên frame ĐÃ CÓ kết quả của chó
        results_chicken = model_chicken(annotated_frame, conf=0.5, verbose=False)
        
        # Vẽ tiếp kết quả của gà lên frame đã có chó
        final_frame = results_chicken[0].plot()

        # --- 5. HIỂN THỊ ---
        cv2.imshow("YOLOv8 Dual Detection: Dog & Chicken", final_frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Hết video hoặc lỗi khung hình
        break

# Giải phóng bộ nhớ
cap.release()
cv2.destroyAllWindows()