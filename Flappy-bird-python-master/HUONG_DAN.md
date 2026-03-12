# 📁 Cấu Trúc Code Flappy Bird - Hướng Dẫn Chi Tiết

Game Flappy Bird được điều khiển bằng cử chỉ đã được tách thành **5 file Python** dễ hiểu:

---

## 📋 Danh Sách Các File

### 1️⃣ **config.py** - Các Hằng Số & Cấu Hình
```
Chứa: Các giá trị không đổi của game
- Kích thước màn hình (400x600)
- Tốc độ game (SPEED, GRAVITY, GAME_SPEED)
- Kích thước ống, mặt đất, khoảng cách
- Cấu hình nhận dạng cử chỉ (blink, finger, head)
- Các tham số MediaPipe
```

**Lý do tách**: Khi bạn muốn thay đổi độ khó game, chỉ cần sửa file này!

---

### 2️⃣ **utils.py** - Các Hàm Tiện Ích
```
Chứa: Các hàm được dùng lặp lại nhiều lần
- asset_path() → Tạo đường dẫn file nhạc/hình ảnh
- distance() → Tính khoảng cách giữa 2 điểm
- calc_ear() → Tính EAR để phát hiện chớp mắt
- play_sound() → Phát âm thanh
- is_off_screen() → Kiểm tra sprite ra khỏi màn hình
```

**Lý do tách**: Các hàm này được dùng ở nhiều chỗ, tách ra tái sử dụng dễ hơn.

---

### 3️⃣ **game_objects.py** - Các Đối Tượng Game
```
Chứa: 3 class chính của game

📦 Bird (Chim)
   - __init__(): Tạo chim với 3 hình cánh
   - update(): Cập nhật vị trí & hình cánh
   - bump(): Tạo cử động nhảy

📦 Pipe (Ống nước)
   - __init__(): Tạo ống trên hoặc dưới ngẫu nhiên
   - update(): Di chuyển ống sang trái

📦 Ground (Mặt đất)
   - __init__(): Tạo mặt đất
   - update(): Di chuyển mặt đất sang trái
```

**Lý do tách**: Mỗi đối tượng game có logic riêng, tách ra dễ quản lý.

---

### 4️⃣ **gesture_control.py** - Nhận Dạng Cử Chỉ
```
Chứa: Class GestureController - Quản lý camera & cử chỉ

🎮 GestureController
   - __init__(): Mở camera, khởi tạo MediaPipe
   - detect_command(): Phát hiện 3 loại cử chỉ:
     | Chớp mắt (Eye Blink)
     | Ngón tay (Index Finger Up)
     | Hướng đầu lên (Head Up)
   - cleanup(): Giải phóng camera & tài nguyên
```

**3 cách nhảy**:
- 👀 **Chớp mắt**: Tính EAR (Eye Aspect Ratio) < 0.19
- 👆 **Ngón tay**: Chỉ số lên, các ngón khác xuống
- ⬆️ **Đầu lên**: Mũi cao hơn tâm mắt

**Lý do tách**: Quản lý camera đòi hỏi khá nhiều code, tách ra dễ bảo trì.

---

### 5️⃣ **flappy_main.py** - Vòng Lặp Game Chính
```
Chứa: Luồng chính của game

🎮 Các Function:
   - init_pygame(): Khởi tạo Pygame
   - load_assets(): Tải hình ảnh nền
   - begin_screen(): Màn hình chờ bắt đầu
   - main_game_loop(): Vòng lặp game chính
   - game_over_screen(): Màn hình kết thúc

📊 Quy trình:
   1. Khởi tạo Pygame & tải hình ảnh
   2. Hiển thị màn hình chờ
      → Chờ người dùng nhấn SPACE hoặc cử chỉ
   3. Chạy vòng lặp game chính:
      → Phát hiện cử chỉ
      → Cập nhật vị trí chim, ống, mặt đất
      → Kiểm tra va chạm
      → Tính điểm
      → Vẽ lên màn hình
   4. Hiển thị màn hình Game Over
   5. Cleanup và thoát
```

**Lý do tách**: File này chỉ tập trung vào logic chính, không rối loạn.

---

## 🚀 Cách Chạy

```bash
cd Flappy-bird-python-master
python flappy_main.py
```

---

## 📊 Biểu Đồ Phụ Thuộc Giữa Các File

```
flappy_main.py (MAIN - Điểm bắt đầu)
    ├── config.py (Đọc hằng số)
    ├── utils.py (Sử dụng các hàm tiện ích)
    ├── game_objects.py (Bird, Pipe, Ground)
    │   ├── config.py
    │   └── utils.py
    └── gesture_control.py (Quản lý camera & cử chỉ)
        ├── config.py
        └── utils.py
```

---

## 💡 Lợi Ích Của Cách Tách Này

✅ **Dễ hiểu**: Mỗi file có 1 trách nhiệm rõ ràng
✅ **Dễ sửa**: Muốn thay đổi độ khó? Chỉnh config.py
✅ **Dễ tái sử dụng**: Có thể dùng game_objects.py ở game khác
✅ **Dễ bảo trì**: Tìm bug nhanh hơn
✅ **Dễ mở rộng**: Thêm tính năng mới không ảnh hưởng khác

---

## 🎮 Vòng Lặp Game Chi Tiết

```
WHILE game_running:
    1. detect_command() từ camera → nhận lệnh nhảy
    2. Cập nhật vị trí:
       - bird.update() → thêm trọng lực
       - pipe_group.update() → di chuyển sang trái
       - ground_group.update() → di chuyển sang trái
    
    3. Xoá & tạo mới:
       - Nếu ống ra khỏi màn hình → xoá & tạo cái mới
       - Nếu mặt đất ra khỏi màn hình → xoá & tạo cái mới
    
    4. Tính điểm:
       - Khi ống vượt qua chim → +1 điểm
    
    5. Vẽ màn hình:
       - Vẽ background
       - Vẽ bird, pipes, ground
       - Vẽ điểm số
    
    6. Kiểm tra va chạm:
       - Chim va chạm mặt đất → Game Over
       - Chim va chạm ống → Game Over
       - Chim vượt trên cùng → Game Over
```

---

## 🔧 Muốn Thay Đổi Gì?

| Yêu cầu | Sửa File Nào |
|--------|-------------|
| Thay đổi độ khó game | config.py |
| Thay đổi cách nhảy (blink/finger/head) | gesture_control.py |
| Thêm âm thanh mới | gesture_control.py hoặc utils.py |
| Thay đổi hình ảnh chim | game_objects.py |
| Thêm tính năng game mới | flappy_main.py |

---

## 📚 Ghi Chú

- **MediaPipe**: Thư viện nhận dạng tư thế, tay, khuôn mặt
- **OpenCV (cv2)**: Xử lý video/camera
- **Pygame**: Engine để tạo game 2D
- **EAR (Eye Aspect Ratio)**: Công thức tính để phát hiện chớp mắt

Chúc bạn hiểu rõ code! 🎉
