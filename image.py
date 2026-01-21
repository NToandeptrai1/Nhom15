
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import math
import time

# N =1025
# N = 728
# img = np.zeros((N,N), dtype=np.uint8)
# cv.line(img, (0,0), (N-1,N-1), (255),1)
# # cv.imshow('line', img)
# if cv.waitKey(0) & 0xFF == 27:
#     cv.destroyAllWindows()



# # ====== Chessboard 8x8 ======
# cell = 100
# rows = 8
# cols = 8

# board_size = cell * rows

# chessboard = np.zeros((board_size, board_size,3), dtype=np.uint8)

# for i in range(rows):
#     for j in range(cols):
#         if (i + j) % 2 == 0:
#             chessboard[
#                 i*cell:(i+1)*cell,
#                 j*cell:(j+1)*cell
#              ] = (255, 0, 255)  
#         else:
#             chessboard[
#                 i*cell:(i+1)*cell,
#                 j*cell:(j+1)*cell
#             ] = (0, 0, 255)        
# cv.imshow("Chessboard 8x8", chessboard)
# cv.waitKey(0)
# cv.destroyAllWindows()
# chesboard = np.zeros((800, 800,3), dtype=np.uint8)
# chesboard(0:99, 0:99) = (128, 0, 128)
# chesboard(100:100, 100:199) = (128, 0, 128)
# cv.rectangle(chesboard, (100,100), (199,199), (128,0,128), -1)
# cv.imshow("Chessboard 8x8", chessboard)
# cv.waitKey(0)
# cv.destroyAllWindows()
# Kích thước ảnh
SIZE = 600
CENTER = (SIZE//2, SIZE//2)
RADIUS = 250

# Màu BGR
BLACK = (0, 0, 0)
BLUE   = (255, 0, 0)
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
WHITE  = (255, 255, 255)
YELLOW = (0, 255, 255)

# ================== HÀM VẼ KIM ==================
def draw_hand(img, angle, length, color, thickness):
    x = int(CENTER[0] + length * math.cos(angle))
    y = int(CENTER[1] + length * math.sin(angle))
    cv.line(img, CENTER, (x, y), color, thickness)

# ================== VÒNG LẶP ĐỒNG HỒ ==================
while True:
    # Nền tím
    img = np.ones((SIZE, SIZE, 3), dtype=np.uint8)
    img[:] = BLACK

    # Mặt đồng hồ
    cv.circle(img, CENTER, RADIUS, YELLOW, 3)

    # ========= VẠCH PHÚT (LEVEL 5) =========
    for i in range(60):
        angle = math.radians(i * 6 - 90)
        x1 = int(CENTER[0] + (RADIUS - 10) * math.cos(angle))
        y1 = int(CENTER[1] + (RADIUS - 10) * math.sin(angle))
        x2 = int(CENTER[0] + RADIUS * math.cos(angle))
        y2 = int(CENTER[1] + RADIUS * math.sin(angle))
        thickness = 3 if i % 5 == 0 else 1
        cv.line(img, (x1, y1), (x2, y2), WHITE, thickness)

    # ========= SỐ LA MÃ =========
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, "XII", (285, 80),  font, 1, RED,   2)
    cv.putText(img, "III", (470, 310), font, 1, BLUE,  2)
    cv.putText(img, "VI",  (295, 520), font, 1, GREEN, 2)
    cv.putText(img, "IX",  (90, 310),  font, 1, WHITE, 2)

    # ========= THỜI GIAN HIỆN TẠI =========
    t = time.localtime()
    hour = t.tm_hour % 12
    minute = t.tm_min
    second = t.tm_sec

    # ========= TÍNH GÓC (LOGIC THẬT) =========
    second_angle = math.radians(second * 6 - 90)
    minute_angle = math.radians(minute * 6 + second * 0.1 - 90)
    hour_angle   = math.radians(hour * 30 + minute * 0.5 - 90)

    # ========= VẼ KIM =========
    draw_hand(img, hour_angle,   130, BLUE,  6)   # Kim giờ
    draw_hand(img, minute_angle, 180, GREEN, 4)   # Kim phút
    draw_hand(img, second_angle, 220, RED,   2)   # Kim giây

    # Tâm đồng hồ
    cv.circle(img, CENTER, 8, WHITE, -1)

    cv.imshow("Analog Clock", img)

    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()