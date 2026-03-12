# =========================
# GESTURE RECOGNITION & CONTROL
# =========================

import time
import cv2
import mediapipe as mp
import pygame
from config import *
from utils import distance, calc_ear, play_sound, asset_path

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands


class GestureController:
    """Quản lý nhận dạng cử chỉ để điều khiển game"""
    
    def __init__(self):
        """Khởi tạo MediaPipe và Camera"""
        # Khởi tạo Face Mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Khởi tạo Hand Detection
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Mở camera
        self.camera = self._open_camera()
        
        # Biến ghi nhớ thời gian
        self.last_blink_time = 0
        self.blink_state_closed = False
        
        self.last_hand_time = 0
        self.last_head_time = 0

    def _open_camera(self):
        """Mở camera với các backend khác nhau"""
        backends = [
            cv2.CAP_DSHOW,
            cv2.CAP_MSMF,
            cv2.CAP_ANY,
        ]

        for backend in backends:
            cam = cv2.VideoCapture(0, backend)
            time.sleep(1)
            if cam.isOpened():
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cam

        raise RuntimeError("Khong mo duoc camera!")

    def _is_head_up(self, points):
        """Kiểm tra đầu hướng lên"""
        nose = points[NOSE_TIP]
        left_eye = points[LEFT_EYE_CORNER]
        right_eye = points[RIGHT_EYE_CORNER]

        eye_center_y = (left_eye[1] + right_eye[1]) / 2

        return nose[1] < eye_center_y - HEAD_UP_THRESHOLD

    def _is_index_finger_up(self, hand_landmarks):
        """Kiểm tra chỉ số lên (gesture nhảy)"""
        lm = hand_landmarks.landmark

        index_up = lm[8].y < lm[6].y
        middle_down = lm[12].y > lm[10].y
        ring_down = lm[16].y > lm[14].y
        pinky_down = lm[20].y > lm[18].y

        return index_up and middle_down and ring_down and pinky_down

    def detect_command(self):
        """
        Nhận dạng cử chỉ và trả về lệnh nhảy
        
        Return:
            1  -> phát hiện lệnh (chớp mắt / ngón tay / hướng đầu lên)
            0  -> không có lệnh
            -1 -> ESC được nhấn
        """
        ret, frame = self.camera.read()
        if not ret:
            return 0

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_result = self.face_mesh.process(rgb)
        hand_result = self.hands.process(rgb)

        command = 0
        label_text = "No face / no hand"

        # ===== FACE DETECTION =====
        if face_result.multi_face_landmarks:
            face_landmarks = face_result.multi_face_landmarks[0]
            points = []

            for lm in face_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))

            # Vẽ điểm mắt trái
            eye_points = [points[i] for i in LEFT_EYE_IDX]
            for p in eye_points:
                cv2.circle(frame, p, 2, (0, 255, 255), -1)

            # Vẽ mũi + 2 mắt ngoài để debug
            cv2.circle(frame, points[NOSE_TIP], 3, (0, 0, 255), -1)
            cv2.circle(frame, points[LEFT_EYE_CORNER], 3, (255, 255, 0), -1)
            cv2.circle(frame, points[RIGHT_EYE_CORNER], 3, (255, 255, 0), -1)

            # ----- PHÁT HIỆN ĐẦU HƯỚNG LÊN -----
            if self._is_head_up(points):
                label_text = "Head Up -> JUMP"
                if (now - self.last_head_time) > HEAD_COOLDOWN:
                    command = 1
                    self.last_head_time = now

            # ----- PHÁT HIỆN CHỚP MẮT -----
            ear = calc_ear(eye_points)

            if ear < EAR_THRESHOLD:
                self.blink_state_closed = True
                if command == 0:
                    label_text = f"Eye Closed | EAR={ear:.2f}"
            else:
                if self.blink_state_closed and (now - self.last_blink_time) > BLINK_COOLDOWN:
                    label_text = "Blink -> JUMP"
                    command = 1
                    self.last_blink_time = now
                    self.blink_state_closed = False
                else:
                    self.blink_state_closed = False
                    if command == 0:
                        label_text = f"Eye Open | EAR={ear:.2f}"

        # ===== HAND DETECTION =====
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

                if self._is_index_finger_up(hand_landmarks):
                    label_text = "Index Finger Up -> JUMP"
                    if (now - self.last_hand_time) > HAND_COOLDOWN:
                        command = 1
                        self.last_hand_time = now

        # Hiển thị thông tin trên màn hình camera
        cv2.putText(
            frame,
            label_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Blink / Index finger / Head up to jump",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            "ESC to quit",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )

        cv2.imshow("Blink Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            return -1

        return command

    def cleanup(self):
        """Giải phóng tài nguyên"""
        self.camera.release()
        self.face_mesh.close()
        self.hands.close()
        cv2.destroyAllWindows()


# Audio paths
WING_SOUND = asset_path("assets", "audio", "wing.wav")
HIT_SOUND = asset_path("assets", "audio", "hit.wav")
