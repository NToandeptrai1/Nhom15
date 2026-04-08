"""
OCR Tiếng Việt – Fix cho ảnh chữ TRẮNG trên nền MÀU (slide, banner, v.v.)
Copy toàn bộ file này vào 1 cell Jupyter rồi chạy.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────────────────────────────
# BƯỚC 1: Detect tự động  →  chữ tối-trên-sáng  hay  chữ sáng-trên-tối
# ─────────────────────────────────────────────────────────────────────

def is_dark_background(gray: np.ndarray) -> bool:
    """
    Trả về True nếu nền ảnh tối (chữ sáng).
    Dùng ngưỡng Otsu: nếu pixel trung bình < 128 → nền tối.
    """
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Tỷ lệ pixel trắng sau Otsu
    white_ratio = np.sum(otsu == 255) / otsu.size
    # Nếu > 60% pixel trắng → chữ là màu tối trên nền trắng (bình thường)
    # Nếu < 40% → chữ sáng trên nền tối → cần đảo
    return white_ratio < 0.40


def best_channel(img_bgr: np.ndarray) -> np.ndarray:
    """
    Chọn channel tốt nhất cho ảnh màu.
    Với chữ trắng trên nền xanh lam: channel R hoặc grayscale thường tốt nhất.
    Thử tất cả 4 cách → chọn cái có độ tương phản cao nhất.
    """
    gray_methods = {
        'Grayscale'  : cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
        'Blue ch.'   : img_bgr[:, :, 0],
        'Green ch.'  : img_bgr[:, :, 1],
        'Red ch.'    : img_bgr[:, :, 2],
        'Max(RGB)'   : np.max(img_bgr, axis=2),       # lấy max 3 kênh
        'LAB-L'      : cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)[:, :, 0],
    }
    best_name, best_img, best_std = 'Grayscale', gray_methods['Grayscale'], 0

    for name, g in gray_methods.items():
        std = float(np.std(g))
        if std > best_std:
            best_std = std
            best_name = name
            best_img  = g

    print(f'  → Chọn kênh: [{best_name}]  (std={best_std:.1f})')
    return best_img.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
# BƯỚC 2: Tiền xử lý thông minh (auto-invert + auto-channel)
# ─────────────────────────────────────────────────────────────────────

def smart_preprocess(image_path: str,
                     upscale: bool = True,
                     show: bool = True) -> np.ndarray:
    """
    Pipeline tiền xử lý THÔNG MINH:
    - Tự chọn channel tốt nhất
    - Tự đảo ngược nếu chữ sáng-trên-tối
    - CLAHE → Sauvola (hoặc Adaptive nếu Sauvola không có)
    - Morph cleanup
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    steps = {}
    steps['Original'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Chọn channel tốt nhất ──
    print('[1] Phân tích kênh màu...')
    gray = best_channel(img)
    steps['Best Channel'] = gray

    # ── Upscale (chỉ nếu ảnh nhỏ) ──
    if upscale:
        h, w = gray.shape
        if w < 1200:
            scale = 2 if w < 800 else 1.5
            gray = cv2.resize(gray,
                              (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)
            steps[f'Upscale x{scale}'] = gray
            print(f'[2] Upscale {w}→{int(w*scale)}px')
        else:
            print(f'[2] Ảnh đủ lớn ({w}px), bỏ qua upscale')

    # ── Khử nhiễu nhẹ ──
    gray = cv2.bilateralFilter(gray, 9, 75, 75)   # giữ cạnh tốt hơn Gaussian
    steps['Bilateral Denoise'] = gray
    print('[3] Khử nhiễu (Bilateral Filter)')

    # ── CLAHE ──
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    steps['CLAHE'] = gray
    print('[4] CLAHE contrast enhancement')

    # ── Auto-invert ──
    need_invert = is_dark_background(gray)
    if need_invert:
        gray = cv2.bitwise_not(gray)
        steps['Invert (auto)'] = gray
        print('[5] ✅ Đảo ngược (chữ sáng-nền tối được phát hiện)')
    else:
        print('[5] Không cần đảo ngược (chữ tối-nền sáng)')

    # ── Binarization: Sauvola hoặc Adaptive ──
    try:
        from skimage import filters as skf
        thresh = skf.threshold_sauvola(gray, window_size=25, k=0.2)
        binary = (gray > thresh).astype(np.uint8) * 255
        steps['Sauvola Binary'] = binary
        print('[6] Binarize: Sauvola')
    except Exception:
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 8
        )
        steps['Adaptive Binary'] = binary
        print('[6] Binarize: Adaptive (skimage không có)')

    # ── Morphological cleanup ──
    # Dùng kernel nhỏ để không xóa dấu tiếng Việt
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k)
    steps['Morph Cleanup'] = binary
    print('[7] Morphological cleanup xong')

    # ── Hiển thị ──
    if show:
        n = len(steps)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
        if n == 1:
            axes = [axes]
        for ax, (title, im) in zip(axes, steps.items()):
            cmap = 'gray' if len(im.shape) == 2 else None
            ax.imshow(im, cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        plt.suptitle('Smart Preprocessing Pipeline', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return binary


# ─────────────────────────────────────────────────────────────────────
# BƯỚC 3: OCR với Tesseract tối ưu tiếng Việt
# ─────────────────────────────────────────────────────────────────────

def ocr_vietnamese(binary_img: np.ndarray,
                   lang: str = 'vie',
                   psm: int = 6) -> str:
    """
    OCR tiếng Việt với cấu hình tối ưu.
    - lang='vie'      → chỉ tiếng Việt
    - lang='vie+eng'  → Việt + Anh (cho văn bản trộn)
    - psm=6  → khối văn bản nhiều dòng (phổ biến nhất)
    - psm=11 → sparse text (văn bản rải rác trên slide)
    """
    pil_img = Image.fromarray(binary_img)
    config  = f'--oem 3 --psm {psm}'
    text    = pytesseract.image_to_string(pil_img, lang=lang, config=config)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────
# BƯỚC 4: Pipeline đầy đủ + thử nhiều PSM để chọn kết quả tốt nhất
# ─────────────────────────────────────────────────────────────────────

def run_ocr_smart(image_path: str,
                  lang: str = 'vie',
                  upscale: bool = True,
                  show: bool = True) -> str:
    """
    Pipeline đầy đủ: tiền xử lý thông minh → OCR nhiều PSM → chọn tốt nhất.

    Trả về kết quả OCR có nhiều từ nhất (heuristic đơn giản).
    """
    print(f'\n{"─"*55}')
    print(f'📷 Xử lý: {os.path.basename(image_path)}')
    print(f'{"─"*55}')

    binary = smart_preprocess(image_path, upscale=upscale, show=show)

    # Thử nhiều PSM
    results = {}
    for psm in [6, 11, 3]:
        t = ocr_vietnamese(binary, lang=lang, psm=psm)
        results[psm] = t
        word_count = len(t.split())
        print(f'  PSM {psm}: {word_count:>3} từ  | {t[:60].replace(chr(10)," ")!r}')

    # Chọn PSM cho nhiều từ nhất
    best_psm = max(results, key=lambda p: len(results[p].split()))
    best_text = results[best_psm]

    print(f'\n✅ Chọn PSM {best_psm} ({len(best_text.split())} từ)')
    print('━' * 55)
    print('KẾT QUẢ OCR CUỐI CÙNG:')
    print('━' * 55)
    print(best_text)
    print('━' * 55)

    # Visualize bounding box
    _visualize_boxes(image_path, binary, lang=lang, psm=best_psm)

    return best_text


def _visualize_boxes(image_path, binary, lang='vie', psm=6):
    """Vẽ bounding box màu sắc theo confidence."""
    pil_img = Image.fromarray(binary)
    data = pytesseract.image_to_data(
        pil_img, lang=lang,
        config=f'--oem 3 --psm {psm}',
        output_type=pytesseract.Output.DICT
    )
    orig = cv2.imread(image_path)
    vis  = orig.copy()

    # Scale factor (vì binary có thể đã upscale)
    sx = orig.shape[1] / binary.shape[1]
    sy = orig.shape[0] / binary.shape[0]

    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        conf = int(data['conf'][i])
        if word and conf > 30:
            x = int(data['left'][i]   * sx)
            y = int(data['top'][i]    * sy)
            w = int(data['width'][i]  * sx)
            h = int(data['height'][i] * sy)
            # Màu: xanh lá (conf cao) → đỏ (conf thấp)
            green = min(255, conf * 3)
            red   = max(0, 255 - conf * 3)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, green, red), 2)
            cv2.putText(vis, f'{conf}', (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, green, red), 1)

    plt.figure(figsize=(14, 7))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f'Bounding box nhận dạng  |  lang={lang}  psm={psm}\n'
              f'Màu: xanh=tin cậy cao, đỏ=thấp', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# CHẠY THỬ NGAY  ← Thay đường dẫn ảnh của bạn vào đây
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ← SỬA ĐƯỜNG DẪN NÀY THÀNH ẢNH CỦA BẠN
    YOUR_IMAGE = r'C:\thaytri\dayhomnay\hinh10.jpg'   # hoặc .png

    if not os.path.exists(YOUR_IMAGE):
        print(f'❌ Không tìm thấy ảnh: {YOUR_IMAGE}')
        print('   👉 Sửa biến YOUR_IMAGE thành đường dẫn ảnh thật của bạn')
    else:
        result = run_ocr_smart(
            image_path = YOUR_IMAGE,
            lang       = 'vie',      # 'vie' hoặc 'vie+eng'
            upscale    = True,
            show       = True,
        )
