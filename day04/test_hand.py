import cv2
import mediapipe as mp
import numpy as np

# ====== MediaPipe Tasks ======
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ====== Cấu hình model ======
MODEL_PATH = "day04/hand_landmarker.task"  # nhớ để file model cùng thư mục hoặc sửa path

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# ====== Tạo detector ======
detector = HandLandmarker.create_from_options(options)

# ====== Tự định nghĩa các đường nối bàn tay ======
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17)              # palm
]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không mở được camera!")
    exit()

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi camera!")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Chuyển sang MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # detect_for_video cần timestamp tăng dần (ms)
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms == 0:
        timestamp_ms = frame_idx * 33
    frame_idx += 1

    # Detect tay
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            points = []

            # Vẽ 21 keypoints
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
                cv2.circle(frame, (x, y), 5, (255, 255, 0), -1)

            # Vẽ các đường nối
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, points[start_idx], points[end_idx], (255, 0, 255), 2)

    cv2.imshow("Hand Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # bấm X vẫn thoát được
    if cv2.getWindowProperty("Hand Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()