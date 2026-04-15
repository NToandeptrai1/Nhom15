# Kế hoạch nâng cấp Chương 3: Thiết kế & Cài đặt (Bản siêu chi tiết)

Mục tiêu là làm rõ các khái niệm trừu tượng thành các bước logic dễ hiểu nhưng vẫn đảm bảo tính kỹ thuật cao để lấy trọn 2.5 điểm.

## 3.2.3. Chi tiết về các thành phần mô hình (Weights)
- Giải thích cơ chế **Differentiable Binarization (DB)**: Tại sao nó tốt hơn các phương pháp cũ (nó dự đoán bản đồ xác định biên chữ thay vì chỉ nhị phân hóa cứng).
- Giải thích **Classification**: Tại sao cần xoay 180 độ (đề phòng người dùng cầm ngược máy khi chụp).
- Giải thích cấu trúc file **ONNX**: Tại sao dùng định dạng này (tối ưu tính toán đồ thị, giúp chạy nhanh trên CPU mà không cần GPU đắt tiền).

## 3.3.1. Phân tích logic Điều phối (ocr.py)
- **Hàm `extract_text`**: Mô tả luồng như một dây chuyền sản xuất (Pipeline).
- **Thuật toán `sort_polygon`**: Giải thích chi tiết bước so sánh tọa độ Y. Nếu 2 vùng chữ có Delta Y < 10px thì coi như cùng dòng, sau đó mới so sánh tọa độ X. Điều này giải quyết bài toán đọc văn bản từ trái sang phải mà không bị nhảy dòng.
- **Biến đổi phối cảnh (`crop_image`)**: Giải thích về ma trận chuyển đổi 3x3 giúp duỗi phẳng các vùng chữ bị méo do góc chụp (Perspective transform).

## 3.3.2. Chi tiết kiến trúc VietOCR (vietocr_onnx.py)
- **CNN (Feature Extractor)**: Giải thích cách các bộ lọc (filters) nhận diện nét gạch, nét cong của chữ.
- **Attention Mechanism**: Giải thích trực quan cách mô hình "nhìn" vào từng đoạn của dòng chữ để đoán ký tự.
- **Giải thuật Greedy Search**: Giải thích bước chọn `argmax` (xác suất cao nhất) tại mỗi bước thời gian và cách nó ghép nối thành từ hoàn chỉnh.

## 3.3.3. Toán học trong Tiền xử lý (test_vncv.py)
- **Denoising**: Tại sao dùng Non-Local Means (vì nó bảo vệ được các đường biên sắc nét của chữ tốt hơn các bộ lọc làm mờ thông thường).
- **Deskewing**: Giải thích quy trình 3 bước: Canny (tìm cạnh) -> Coords (tập hợp điểm) -> minAreaRect (xác định hình chữ nhật bao quanh nhỏ nhất) -> Lấy góc nghiêng để xoay.
- **CLAHE**: Giải thích sự khác biệt với Equalization thông thường (xử lý theo từng khối nhỏ 8x8 để không bị cháy sáng ở các vùng có độ sáng không đều).

## Câu hỏi cho người dùng
- Bạn có muốn tôi kèm theo các công thức toán học cơ bản (như ma trận xoay, hoặc công thức xác suất) để báo cáo trông chuyên nghiệp hơn không?
- Bạn có cần tôi giải thích thêm về cách dữ liệu được truyền qua JSON giữa Flask và React/JS không?
