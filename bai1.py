import cv2
import numpy as np
import os
from datetime import datetime


def nothing(x):
    pass


# -----------------------------
# Sketch filters (đẹp + mượt hơn)
# -----------------------------
def dodge(gray, blur_inv):
    # gray: 0..255, blur_inv: 0..255
    return cv2.divide(gray, 255 - blur_inv, scale=256)


def pencilSketch(frame):
    """Pencil mượt, nền sáng sạch, ít bệt."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cân bằng nhẹ để da/mặt không bị bệt
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    inv = 255 - gray
    blur_inv = cv2.GaussianBlur(inv, (0, 0), sigmaX=12, sigmaY=12)
    sketch = dodge(gray, blur_inv)

    # mịn nhẹ cho “giấy” sạch
    sketch = cv2.GaussianBlur(sketch, (3, 3), 0)
    return sketch


def detailedSketch(frame, low, high, k):
    """Nét chi tiết nhưng mảnh, ít răng cưa/ít gớm."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # giữ biên, giảm noise để Canny ra đẹp
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=60, sigmaSpace=60)

    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1

    blur = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(blur, low, high, L2gradient=True)

    sketch = 255 - edges

    # sạch nhiễu nhỏ (đỡ “bệt” hơn erode)
    kernel = np.ones((2, 2), np.uint8)
    sketch = cv2.morphologyEx(sketch, cv2.MORPH_OPEN, kernel, iterations=1)
    return sketch


def hatching(frame, low, high):
    """Gạch bóng nghệ hơn (đậm nhạt theo sáng)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    smooth = cv2.bilateralFilter(gray, 9, 60, 60)

    edges = cv2.Canny(smooth, low, high, L2gradient=True)
    edges = 255 - edges  # nền trắng, nét tối

    hatch = np.full_like(gray, 255)
    step = 6

    for y in range(0, gray.shape[0], step):
        for x in range(0, gray.shape[1], step):
            v = gray[y, x]
            if v < 70:
                cv2.line(hatch, (x, y), (x + step, y + step), 120, 1)
                cv2.line(hatch, (x, y + step), (x + step, y), 120, 1)
            elif v < 120:
                cv2.line(hatch, (x, y), (x + step, y + step), 170, 1)

    # trộn hatch + edges (min = kiểu “multiply”)
    result = cv2.min(edges, hatch)
    result = cv2.GaussianBlur(result, (3, 3), 0)
    return result


def combinedSketch(frame, low, high, k, method="detailed"):
    if method == "pencil":
        return pencilSketch(frame)
    elif method == "hatching":
        return hatching(frame, low, high)
    elif method == "hybrid":
        d = detailedSketch(frame, low, high, k)
        p = pencilSketch(frame)
        return cv2.min(d, p)  # trộn nét + shading nhìn thật hơn
    else:
        return detailedSketch(frame, low, high, k)


# -----------------------------
# Main
# -----------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win = "SKETCH PRO"
    cv2.namedWindow(win)

    # trackbar: set default hợp lý hơn
    cv2.createTrackbar("Low", win, 35, 255, nothing)
    cv2.createTrackbar("High", win, 120, 255, nothing)
    cv2.createTrackbar("Blur", win, 5, 21, nothing)   # k (odd)
    cv2.createTrackbar("Mode", win, 0, 3, nothing)    # 0..3

    mode_names = ["Detailed", "Pencil", "Hatching", "Hybrid"]
    method_map = {0: "detailed", 1: "pencil", 2: "hatching", 3: "hybrid"}

    print("=" * 60)
    print("SKETCH PRO - BẢN MƯỢT / ĐỠ 'GỚM'")
    print("=" * 60)
    print("Phím tắt:")
    print("  q : Thoát")
    print("  s : Lưu ảnh")
    print("  0-3 : Chuyển chế độ (Detailed/Pencil/Hatching/Hybrid)")
    print("")
    print("Trackbar gợi ý:")
    print("  Low  : 20-60")
    print("  High : 80-160")
    print("  Blur : 3-9 (odd)")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi đọc frame!")
            break

        low = cv2.getTrackbarPos("Low", win)
        high = cv2.getTrackbarPos("High", win)
        k = cv2.getTrackbarPos("Blur", win)
        mode = cv2.getTrackbarPos("Mode", win)

        # ensure odd k >= 1
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1

        sketch = combinedSketch(frame, low, high, k, method_map.get(mode, "detailed"))

        # Hiển thị chữ đẹp hơn: convert sang BGR rồi putText (màu xám đậm)
        sketch_vis = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

        mode_name = mode_names[mode]
        cv2.putText(
            sketch_vis,
            f"Mode: {mode_name}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            sketch_vis,
            f"L:{low} H:{high} B:{k}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("CAM (Goc)", frame)
        cv2.imshow(win, sketch_vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            os.makedirs("output", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"output/original_{ts}.jpg", frame)
            cv2.imwrite(f"output/sketch_{mode_name}_{ts}.png", sketch)  # lưu bản xám sạch
            print(f"✓ Đã lưu: output/sketch_{mode_name}_{ts}.png")
        elif key in [ord("0"), ord("1"), ord("2"), ord("3")]:
            mode_idx = int(chr(key))
            cv2.setTrackbarPos("Mode", win, mode_idx)
            print(f"→ Chuyển sang chế độ: {mode_names[mode_idx]}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Đã thoát chương trình!")


if __name__ == "__main__":
    main()
