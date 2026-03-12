import cv2 as cv
import numpy as np
import math

cap = cv.VideoCapture("chieu.mp4")

line_x = 600
DIST_THRESHOLD = 70
MAX_MISSED = 12
MERGE_DIST = 25          # gộp circles gần nhau
COOLDOWN_FRAMES = 10     # chống đếm lại do nhảy
MIN_TRACK_AGE = 3        # so frame toi thieu truoc khi dem
SIDE_HYSTERESIS = 5      # vung dem quanh line de giam nhay
FRAME_DELAY_MS = 30      # tang de chay cham hon

objs = []
next_id = 0
count = 0
frame_idx = 0

def merge_circles(circles, merge_dist=25):
    """Gộp các circle có tâm gần nhau (giảm detect trùng)."""
    kept = []
    for (x, y, r) in circles:
        found = False
        for k in kept:
            if math.hypot(k[0]-x, k[1]-y) < merge_dist:
                # giữ circle "to hơn" (hoặc bạn có thể lấy trung bình)
                if r > k[2]:
                    k[0], k[1], k[2] = x, y, r
                found = True
                break
        if not found:
            kept.append([x, y, r])
    return np.array(kept, dtype=int)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    circles = cv.HoughCircles(
        gray, cv.HOUGH_GRADIENT,
        dp=1, minDist=20,
        param1=60, param2=35,   # bạn có thể tăng param2 để giảm detect rác
        minRadius=5, maxRadius=50
    )

    cv.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0,255,0), 2)

    # reset matched per frame
    for o in objs:
        o["matched"] = False

    detections = []
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        circles = merge_circles(circles, MERGE_DIST)  # (A) gộp trùng
        detections = circles.tolist()

    # (B) match one-to-one: mỗi detection tìm obj gần nhất chưa matched
    for (x, y, r) in detections:
        cv.circle(frame, (x, y), r, (0,0,255), 2)

        best = None
        best_dist = 1e9
        for o in objs:
            if o["matched"]:
                continue
            d = math.hypot(o["x"]-x, o["y"]-y)
            if d < best_dist:
                best_dist = d
                best = o

        if best is not None and best_dist < DIST_THRESHOLD:
            best["prev_x"] = best["x"]
            best["x"], best["y"] = x, y
            best["matched"] = True
            best["last_seen"] = frame_idx
            best["age"] += 1

            prev_side = best["side"]
            if best["x"] < line_x - SIDE_HYSTERESIS:
                best["side"] = "L"
            elif best["x"] > line_x + SIDE_HYSTERESIS:
                best["side"] = "R"

            dx = best["x"] - best["prev_x"]

            # (C) đếm khi cắt line, đúng hướng, và cooldown
            crossed = (prev_side == "L" and best["side"] == "R")
            if (not best["counted"]
                and crossed
                and dx > 0
                and best["age"] >= MIN_TRACK_AGE
                and frame_idx - best["last_count_frame"] > COOLDOWN_FRAMES):
                count += 1
                best["counted"] = True
                best["last_count_frame"] = frame_idx
                print(f"COUNT={count}  id={best['id']}  x={best['x']}")

        else:
            # tạo obj mới
            objs.append({
                "id": next_id,
                "x": x, "y": y,
                "prev_x": x,
                "last_seen": frame_idx,
                "matched": True,
                "counted": False,
                "last_count_frame": -999999,
                "age": 0,
                "side": "L" if x < line_x - SIDE_HYSTERESIS else ("R" if x > line_x + SIDE_HYSTERESIS else "U")
            })
            next_id += 1

    # xoá obj mất dấu
    objs = [o for o in objs if frame_idx - o["last_seen"] <= MAX_MISSED]

    cv.putText(frame, f"Count: {count}", (20, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    cv.imshow("f", frame)
    if cv.waitKey(FRAME_DELAY_MS) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
