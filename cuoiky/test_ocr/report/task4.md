# Kế hoạch chi tiết Chương 3: Thiết kế & Cài đặt hệ thống (Bản chuẩn 2.5 điểm)

Kế hoạch này tập trung vào việc bóc tách kỹ thuật của thư viện lõi `vncv` và cách nó vận hành trong hệ thống OCR. Phần giao diện Web sẽ được trình bày ở mức độ phối hợp luồng dữ liệu.

## 3.1. Thiết kế quy trình xử lý dữ liệu (Sơ đồ & Luồng)
Sử dụng Mermaid để mô tả luồng dữ liệu đi sâu vào các lớp của hệ thống:
- **Luồng tổng quát**: Người dùng -> Flask -> Tiền xử lý (OpenCV) -> Pipeline VNCV -> Kết quả.
- **Chi tiết Pipeline VNCV (ocr.py)**:
    1. **Detection**: Tìm vùng chứa văn bản (DB model).
    2. **Classification**: Kiểm tra hướng ảnh (0/180 độ).
    3. **Preprocessing Core**: Cắt ảnh vùng chữ (Crop & Warp).
    4. **Recognition**: Nhận diện ký tự Tiếng Việt (VietOCR ONNX).

## 3.2. Cài đặt môi trường phát triển
- **Hệ điều hành & Ngôn ngữ**: Windows, Python 3.10+.
- **Cấu trúc Virtual Environment (`venv_vncv`)**: Cô lập các thư viện để chạy model AI nặng.
- **Thành phần Model (Weights)**:
    - `detection.onnx`: Model phát hiện vùng chữ.
    - `classification.onnx`: Model phân loại hướng.
    - `model_encoder.onnx` & `model_decoder.onnx`: Bộ đôi model nhận diện Tiếng Việt.
    - `vocab.json`: Từ điển ký tự.

## 3.3. Trình bày và Giải thích mã nguồn cốt lõi (Trọng tâm)

### 3.3.1. Orchestration Pipeline (`vncv/ocr.py`)
- Giải thích hàm `extract_text`: Đây là "trái tim" của thư viện, điều phối 3 giai đoạn (Phát hiện -> Phân loại -> Nhận diện).
- Giải thích kỹ thuật `sort_polygon`: Sắp xếp các dòng chữ theo thứ tự từ trên xuống dưới, từ trái sang phải để văn bản đầu ra có nghĩa.
- Giải thích `crop_image` sử dụng `getPerspectiveTransform`: Chuyển đổi ảnh nghiêng thành ảnh phẳng để AI nhận diện tốt hơn.

### 3.3.2. Nhận diện Tiếng Việt (`vncv/vietocr_onnx.py`)
- Giải thích cấu trúc **Encoder-Decoder (Transformer)**:
    - **Encoder**: Dùng CNN để trích xuất đặc trưng hình ảnh.
    - **Decoder**: Sử dụng cơ chế Attention (Transformer) để dự đoán từng ký tự tiếp theo dựa trên đặc trưng hình ảnh và các ký tự đã biết.
- Giải thích cơ chế **Greedy Search**: Chọn ký tự có xác suất cao nhất tại mỗi bước thời gian.
- Cách xử lý ảnh trước khi đưa vào model: Resize ảnh về chiều cao cố định (thường là 32px) để đảm bảo tính nhất quán.

### 3.3.3. Tiền xử lý bổ trợ (`test_vncv.py` / `app.py`)
- Giải thích các bước tăng cường chất lượng ảnh trước khi đưa vào thư viện `vncv`:
    - Denoise (Khử nhiễu).
    - Upscale 2x (Tăng độ nét).
    - Deskew (Xoay thẳng tự động bằng Canny & minAreaRect).

---
## Tệp tin mục tiêu
- [NEW] [chapter3_detail.md](file:///c:/thaytri/cuoiky/test_ocr/report/chapter3_detail.md): Nội dung bản thảo Chương 3 chi tiết.

## Câu hỏi cho người dùng
- Bạn có muốn tôi trích dẫn các đoạn code cụ thể (snippet) vào báo cáo không, hay chỉ giải thích phương pháp?
- Bạn có cần tôi vẽ sơ đồ chi tiết cho phần Encoder-Decoder của Transformer không?
