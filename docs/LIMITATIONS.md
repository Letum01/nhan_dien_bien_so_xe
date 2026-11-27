# Giới hạn

- Demo chỉ nhận biển có score cao nhất trong ảnh; không trả mọi biển và có thể chọn nhầm.
- OpenCV nhạy với ánh sáng, góc, viền; YOLO Character có chi phí GPU và vẫn có box thiếu/thừa. OCR có nhầm ký tự gần giống. Đồng thuận không xác suất hóa độ đúng.
- Ngưỡng Plate 0,4 / Character 0,25 là cấu hình giữ nguyên; không tối ưu bằng pilot/New10.
- Validation đã dùng trong phát triển, pilot đã xuất hiện trong diagnostics; New10 không có bằng chứng nguồn ngoài ngày nhập. Không có số đo blind test, không nghiệm thu production.
- Warm backend không phải độ trễ trình duyệt hoặc FPS. CUDA, checkpoint và hai venv là điều kiện chạy; không tự fallback.
- Demo local, không có auth nhiều người dùng; giới hạn upload theo hướng dẫn. Test không bao phủ mọi GPU/ảnh độ phân giải cực lớn hoặc stress đa phiên.