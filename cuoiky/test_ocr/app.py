import os
import sys
import cv2
import numpy as np
import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import werkzeug
# Trong file app.py
from vncv.ocr import get_recognition

# Warm-up model tiếng Việt ngay khi khởi động
print("--- Đang khởi tạo model OCR (Warm-up)... ---")
get_recognition('vi')
print("--- Model đã sẵn sàng! ---")


app = Flask(__name__)
CORS(app)

# Cấu hình thư mục upload
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    start_total = time.time()
    img = cv2.imread(image_path)
    if img is None:
        return None

    #
    start = time.time()
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    print(f"  [Log] Denoise: {time.time() - start:.4f}s")

    # 2. 
    start = time.time()
    img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2),
                     interpolation=cv2.INTER_CUBIC)
    print(f"  [Log] Upscale 2x: {time.time() - start:.4f}s")

    # 3. 
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

    # 4. 
    start = time.time()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    print(f"  [Log] CLAHE: {time.time() - start:.4f}s")

    output_path = os.path.join(UPLOAD_FOLDER, "temp_preprocessed.jpg")
    cv2.imwrite(output_path, img)
    print(f"--- Tổng thời gian tiền xử lý: {time.time() - start_total:.4f}s ---")
    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400

    filename = werkzeug.utils.secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        print(f"\n--- Đang xử lý file: {filename} ---")
        start_request = time.time()

        # Tiền xử lý
        preprocessed = preprocess_image(filepath)
        if preprocessed is None:
            return jsonify({'error': 'Lỗi khi xử lý ảnh'}), 500

        # OCR
        start_ocr = time.time()
        from vncv.ocr import extract_text
        results = extract_text(preprocessed, lang="vi")
        print(f"  [Log] OCR Extraction: {time.time() - start_ocr:.4f}s")
        
        # Tính số ký tự nhận diện được
        text_content = ''.join(results)
        char_count = len(text_content)
        print(f"  [Log] Character Count: {char_count}")
        
        # Xóa file tạm
        if os.path.exists(preprocessed):
            os.remove(preprocessed)
        if os.path.exists(filepath):
            os.remove(filepath)

        print(f"--- Hoàn tất trong: {time.time() - start_request:.4f}s ---\n")
        return jsonify({'results': results, 'char_count': char_count})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
