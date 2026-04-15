import os
import sys
import io
import time
import glob

# Đảm bảo in được tiếng Việt trên console Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import cv2
import numpy as np

def preprocess_image(image_path, output_path="temp_preprocessed.jpg"):
    start_total = time.time()
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 1. Denoise
    start = time.time()
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    print(f"  [Log] Denoise: {time.time() - start:.4f}s")

    # 2. Upscale 2x
    start = time.time()
    img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2),
                     interpolation=cv2.INTER_CUBIC)
    print(f"  [Log] Upscale 2x: {time.time() - start:.4f}s")

    # 3. Deskew
    start = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    coords = np.column_stack(np.where(edges > 0))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 5:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    print(f"  [Log] Deskew: {time.time() - start:.4f}s")

    # 4. CLAHE
    start = time.time()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    print(f"  [Log] CLAHE: {time.time() - start:.4f}s")

    cv2.imwrite(output_path, img)
    print(f"--- Tổng thời gian tiền xử lý: {time.time() - start_total:.4f}s ---")
    return output_path



def test_ocr(image_path):
    print(f"\n--- Đang xử lý: {image_path} ---")
    start_test = time.time()

    # Tiền xử lý ảnh trước khi OCR
    preprocessed = preprocess_image(image_path)

    try:
        start_ocr = time.time()
        from vncv.ocr import extract_text
        results = extract_text(preprocessed, lang="vi")
        ocr_time = time.time() - start_ocr
    finally:
        # Xóa file tạm
        if os.path.exists(preprocessed):
            os.remove(preprocessed)

    print(f"  [Log] OCR Extraction: {ocr_time:.4f}s")
    print("Kết quả OCR:")
    for text in results:
        print(f"  - {text}")
    
    print(f"--- Hoàn tất trong: {time.time() - start_test:.4f}s ---")


if __name__ == "__main__":
    folder_path = "anh"
    extensions = ("*.jpg", "*.jpeg", "*.png")
    
    # Tìm tất cả file ảnh trong thư mục 'anh'
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    if image_files:
        print(f"Tìm thấy {len(image_files)} ảnh trong thư mục '{folder_path}'")
        for img_file in image_files:
            test_ocr(img_file)
    else:
        # Nếu không có ảnh trong thư mục 'anh', thử chạy file mặc định
        image_file = "hinh1.jpg"
        if os.path.exists(image_file):
            print(f"Không có ảnh trong '{folder_path}', chạy thử với file mặc định: {image_file}")
            test_ocr(image_file)
        else:
            print(f"⚠️ Không tìm thấy ảnh trong '{folder_path}' hoặc file: {image_file}")
