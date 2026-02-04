import cv2 as cv
import numpy as np


# =========================
# CONFIG
# =========================
VIDEO_PATH = "chieu.mp4"
MAX_DISTANCE = 70                 # khoảng cách match object
LOST_FRAMES_TO_REMOVE = 5         # mất dấu bao nhiêu frame thì xóa
MIN_FRAMES_ALIVE_TO_COUNT = 2     # object phải sống >= N frame mới được đếm
FALLBACK_LINE_RATIO = 0.60        # nếu không dò được vạch trắng thì dùng 60% width


# =========================
# LINE DETECTOR (WHITE LINE IN VIDEO)
# =========================
def detect_white_vertical_line_x(frame):
    """
    Dò vị trí vạch trắng DỌC có sẵn trên video.
    Trả về x (int) nếu tìm được, nếu không trả về None.
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Ngưỡng "trắng" (tùy video có thể chỉnh 170-220)
    _, mask = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)

    # Nối nét dọc cho liền hơn (vạch trắng thường cao và mảnh)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 35))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape[:2]
    best = None
    best_score = 0.0

    for cnt in contours:
        x, y, cw, ch = cv.boundingRect(cnt)

        # Điều kiện vạch dọc: cao + mảnh
        if ch > 0.60 * h and cw < 0.05 * w:
            score = ch / max(cw, 1)   # càng "cao/mảnh" càng tốt
            if score > best_score:
                best_score = score
                best = (x, y, cw, ch)

    if best is None:
        return None

    x, y, cw, ch = best
    return int(x + cw // 2)


# =========================
# CIRCLE DETECTOR
# =========================
def detect_all_circles(frame):
    """Phát hiện TẤT CẢ hình tròn (nhỏ + lớn), trả circles_data và binary_display."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (9, 9), 2)

    # Adaptive threshold để bắt circles có độ sáng khác nhau
    binary1 = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY, 11, 2
    )

    # Threshold thường
    _, binary2 = cv.threshold(blurred, 50, 255, cv.THRESH_BINARY)

    binary = cv.bitwise_or(binary1, binary2)

    # Morphological nhẹ
    kernel = np.ones((3, 3), np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=1)
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=1)

    circles_data = []

    # --- Hough: lớn/vừa
    circles_large = cv.HoughCircles(
        blurred, cv.HOUGH_GRADIENT,
        dp=1, minDist=20,
        param1=50, param2=20,
        minRadius=20, maxRadius=100
    )

    # --- Hough: nhỏ
    circles_small = cv.HoughCircles(
        blurred, cv.HOUGH_GRADIENT,
        dp=1, minDist=15,
        param1=50, param2=18,
        minRadius=8, maxRadius=25
    )

    all_hough = []
    if circles_large is not None:
        all_hough.extend(circles_large[0])
    if circles_small is not None:
        all_hough.extend(circles_small[0])

    # Remove duplicates
    def is_duplicate(new_circle, existing_circles, threshold=15):
        nx, ny, nr = new_circle
        for ex, ey, er in existing_circles:
            dist = np.sqrt((nx - ex) ** 2 + (ny - ey) ** 2)
            if dist < threshold:
                return True
        return False

    unique_circles = []
    for c in all_hough:
        if not is_duplicate(c, unique_circles):
            unique_circles.append(c)

    for cx, cy, r in unique_circles:
        circles_data.append({
            "centroid": (int(cx), int(cy)),
            "radius": int(r),
            "method": "hough"
        })

    # --- Contour fallback: bắt circle bị miss
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv.contourArea(contour)
        if 100 < area < 50000:
            perimeter = cv.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity > 0.35:
                M = cv.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                radius = int(np.sqrt(area / np.pi))

                # check trùng với hough
                dup = False
                for ex in circles_data:
                    exx, exy = ex["centroid"]
                    dist = np.sqrt((cx - exx) ** 2 + (cy - exy) ** 2)
                    if dist < 20:
                        dup = True
                        break

                if not dup:
                    circles_data.append({
                        "centroid": (cx, cy),
                        "radius": radius,
                        "method": "contour"
                    })

    binary_display = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    return circles_data, binary_display


# =========================
# TRACKER + COUNTER
# =========================
def update_tracking(tracked_objects, current_circles, next_object_id, count, line_x):
    matched_ids = set()

    current_circles = sorted(current_circles, key=lambda c: c["centroid"][0])

    for circle in current_circles:
        cx, cy = circle["centroid"]
        r = circle["radius"]

        best_match = None
        min_dist = float("inf")

        for obj_id, obj_data in list(tracked_objects.items()):
            if obj_id in matched_ids:
                continue

            old_cx, old_cy = obj_data["centroid"]

            dist_x = abs(cx - old_cx)
            dist_y = abs(cy - old_cy)
            weighted_dist = np.sqrt((dist_x * 1.2) ** 2 + dist_y ** 2)

            if weighted_dist < MAX_DISTANCE and weighted_dist < min_dist:
                min_dist = weighted_dist
                best_match = obj_id

        if best_match is not None:
            old_x = tracked_objects[best_match]["centroid"][0]

            tracked_objects[best_match]["centroid"] = (cx, cy)
            tracked_objects[best_match]["radius"] = r
            tracked_objects[best_match]["last_seen"] = 0
            tracked_objects[best_match]["frames_alive"] = tracked_objects[best_match].get("frames_alive", 0) + 1
            matched_ids.add(best_match)

            # Count crossing (chỉ đếm nếu sống đủ frame)
            if not tracked_objects[best_match]["counted"]:
                if tracked_objects[best_match]["frames_alive"] >= MIN_FRAMES_ALIVE_TO_COUNT:
                    if old_x < line_x and cx >= line_x:
                        tracked_objects[best_match]["counted"] = True
                        count += 1
                        print(f"✓ Object #{best_match} (r={r}) crossed line! Total: {count}")

        else:
            tracked_objects[next_object_id] = {
                "centroid": (cx, cy),
                "radius": r,
                "counted": False,
                "last_seen": 0,
                "frames_alive": 0
            }
            next_object_id += 1

    # aging + remove lost
    ids_to_remove = []
    for obj_id in list(tracked_objects.keys()):
        if obj_id not in matched_ids:
            tracked_objects[obj_id]["last_seen"] += 1
            if tracked_objects[obj_id]["last_seen"] > LOST_FRAMES_TO_REMOVE:
                ids_to_remove.append(obj_id)

    for obj_id in ids_to_remove:
        del tracked_objects[obj_id]

    return tracked_objects, next_object_id, count


# =========================
# MAIN
# =========================
def main():
    vid = cv.VideoCapture(VIDEO_PATH)
    if not vid.isOpened():
        print("Không mở được video:", VIDEO_PATH)
        return

    tracked_objects = {}
    next_object_id = 0
    count = 0
    frame_count = 0

    # Read first frame to detect line position
    ret, first_frame = vid.read()
    if not ret or first_frame is None:
        print("Không đọc được frame đầu.")
        vid.release()
        return

    h, w = first_frame.shape[:2]

    line_x = detect_white_vertical_line_x(first_frame)
    if line_x is None:
        line_x = int(w * FALLBACK_LINE_RATIO)
        print(f"Không dò được vạch trắng -> fallback LINE_X_POSITION = {line_x}")
    else:
        print(f"Dò được vạch trắng -> LINE_X_POSITION = {line_x}")

    # Restart video
    vid.set(cv.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        if frame is None:
            continue

        frame_count += 1

        circles_data, binary = detect_all_circles(frame)

        tracked_objects, next_object_id, count = update_tracking(
            tracked_objects, circles_data, next_object_id, count, line_x
        )

        # ====== DRAW (KHÔNG VẼ VẠCH ĐỎ) ======
        # Bạn muốn "lấy line trắng ở sau" => để nguyên video gốc, không cv.line nữa.

        # Vẽ circles + id
        for obj_id, obj_data in tracked_objects.items():
            cx, cy = obj_data["centroid"]
            r = obj_data["radius"]

            if obj_data["counted"]:
                color = (0, 255, 255)   # đã đếm
                thickness = 3
            else:
                color = (0, 255, 0)     # chưa đếm
                thickness = 2

            cv.circle(frame, (cx, cy), r, color, thickness)
            cv.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

            label = f"#{obj_id}"
            font_scale = 0.4 if r < 20 else 0.6
            cv.putText(frame, label, (cx - 15, cy - r - 8),
                       cv.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

        # Count display
        cv.rectangle(frame, (5, 5), (220, 70), (0, 0, 0), -1)
        cv.putText(frame, f"Count: {count}", (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        info_text = f"Tracking: {len(tracked_objects)} | Frame: {frame_count} | LineX: {line_x}"
        cv.putText(frame, info_text, (10, frame.shape[0] - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ====== BINARY WINDOW (cũng không vẽ vạch đỏ) ======
        for circle in circles_data:
            cx, cy = circle["centroid"]
            r = circle["radius"]
            color = (0, 255, 0) if circle["method"] == "hough" else (255, 255, 0)
            cv.circle(binary, (cx, cy), r, color, 2)
            cv.circle(binary, (cx, cy), 2, (255, 0, 0), -1)

        cv.putText(binary, f"Detected: {len(circles_data)}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv.imshow("video", frame)
        cv.imshow("Binary/Processed", binary)

        key = cv.waitKey(25)
        if key == ord("q"):
            break
        elif key == ord("r"):
            count = 0
            tracked_objects = {}
            next_object_id = 0
            print("Reset!")

    print("\n" + "=" * 50)
    print(f"TỔNG SỐ ĐÃ ĐẾM: {count}")
    print(f"TỔNG FRAMES: {frame_count}")
    print("=" * 50)

    vid.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
