import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Lấy đúng thư mục chứa file .py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "gesture_recognizer.task")

# STEP 1: Tạo GestureRecognizer
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# STEP 2: Mở webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không mở được camera!")
    exit()

timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Lật ảnh để nhìn như gương
    frame = cv2.flip(frame, 1)

    # BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Tạo MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Nhận diện gesture theo video mode
    result = recognizer.recognize_for_video(mp_image, timestamp)

    # Nếu có gesture
    if result.gestures and len(result.gestures) > 0 and len(result.gestures[0]) > 0:
        gesture = result.gestures[0][0].category_name
        score = result.gestures[0][0].score

        if gesture == "Thumb_Up":
            cv2.putText(
                frame,
                f"Thumbs Up 👍 [{score:.2f}]",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

        if gesture == "ILoveYou":
            cv2.putText(
                frame,
                f"I Love You 🤟 [{score:.2f}]",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

        # Nếu muốn hiện mọi gesture đang nhận được
        cv2.putText(
            frame,
            f"{gesture} [{score:.2f}]",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2
        )

    cv2.imshow("Gesture", frame)

    timestamp += 1

    # ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # bấm X vẫn thoát được
    if cv2.getWindowProperty("Gesture", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()