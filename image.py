
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
import cv2 as cv
import numpy as np
import math
import time

# =========================================================
# 1) CẤU HÌNH (CONFIG)
# =========================================================
SIZE = 600
CENTER = (SIZE // 2, SIZE // 2)
RADIUS = 250

# Màu BGR (OpenCV dùng BGR, không phải RGB)
BLACK  = (0, 0, 0)
BLUE   = (255, 0, 0)
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
WHITE  = (255, 255, 255)
YELLOW = (0, 255, 255)

FONT = cv.FONT_HERSHEY_SIMPLEX

# =========================================================
# 2) HÀM TIỆN ÍCH (HELPER FUNCTIONS)
# =========================================================
def polar_to_xy(center, radius, angle_rad):
    x = int(center[0] + radius * math.cos(angle_rad))
    y = int(center[1] + radius * math.sin(angle_rad))
    return (x, y)

def draw_hand(img, angle_rad, length, color, thickness):
    end_point = polar_to_xy(CENTER, length, angle_rad)
    cv.line(img, CENTER, end_point, color, thickness, lineType=cv.LINE_AA)

def draw_ticks(img):

    angles_deg = [-90, 0, 90, 180] 

    for deg in angles_deg:
        angle = math.radians(deg)

        p1 = polar_to_xy(CENTER, RADIUS - 5, angle)
        p2 = polar_to_xy(CENTER, RADIUS, angle)

        cv.line(img, p1, p2, WHITE, 6, lineType=cv.LINE_AA)


def draw_roman_numbers(img):

    cv.putText(img, "XII", (285,  85), FONT, 1, RED,   2, cv.LINE_AA)
    cv.putText(img, "III", (470, 315), FONT, 1, BLUE,  2, cv.LINE_AA)
    cv.putText(img, "VI",  (295, 525), FONT, 1, GREEN, 2, cv.LINE_AA)
    cv.putText(img, "IX",  ( 95, 315), FONT, 1, WHITE, 2, cv.LINE_AA)

def get_clock_angles():
    
    t = time.localtime()
    h = t.tm_hour % 12
    m = t.tm_min
    s = t.tm_sec

    second_angle = math.radians(s * 6 - 90)
    minute_angle = math.radians(m * 6 + s * 0.1 - 90)
    hour_angle   = math.radians(h * 30 + m * 0.5 - 90)

    return hour_angle, minute_angle, second_angle

# =========================================================
# 3) VÒNG LẶP VẼ ĐỒNG HỒ (MAIN LOOP)
# =========================================================
while True:
    # Tạo khung hình nền đen
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

    # Vẽ viền mặt đồng hồ
    cv.circle(img, CENTER, RADIUS, YELLOW, 3, lineType=cv.LINE_AA)

    # Vẽ vạch phút + số
    draw_ticks(img)
    draw_roman_numbers(img)

    # Tính góc kim theo thời gian hiện tại
    hour_a, minute_a, second_a = get_clock_angles()

    # Vẽ kim (độ dài + độ dày tùy ý)
    draw_hand(img, hour_a,   130, BLUE,  6)   # Kim giờ
    draw_hand(img, minute_a, 180, GREEN, 4)   # Kim phút
    draw_hand(img, second_a, 220, RED,   2)   # Kim giây

    # Vẽ chấm tâm đồng hồ
    cv.circle(img, CENTER, 8, WHITE, -1, lineType=cv.LINE_AA)

    # Hiển thị
    cv.imshow("Analog Clock", img)

    # ESC để thoát
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()








