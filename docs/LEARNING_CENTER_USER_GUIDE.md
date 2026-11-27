# ALPR Learning Center

1. Chạy `OPEN_ALPR_LEARNING_CENTER.bat` hoặc mở http://localhost:8513 nếu dịch vụ đã chạy. Fast V1 và Deep Analysis không đổi.
2. Chọn ảnh JPG/PNG (tối đa 20 MiB và 24 megapixel). Điền nguồn, mã nhóm và quyền sử dụng nếu có bằng chứng; ngày chụp/thu thập không bắt buộc. Không ghi tên, địa chỉ, GPS hoặc thông tin chủ xe; không gửi ảnh ngoài phạm vi được phép sử dụng.
3. Chỉ tích đồng ý nếu muốn lưu ảnh và kết quả nghiên cứu trên máy này. Tệp upload được băm; ảnh lưu dưới dạng PNG bỏ metadata nhúng. Điều này không xóa khuôn mặt hoặc thông tin hiện trong pixel. Dữ liệu nằm riêng trong `data/alpr_learning_center.sqlite3`.
4. Chọn mẫu trong hàng đợi bằng bàn phím. Xem crop/glyph và top-k CNN/SVM/RF nếu có. Nhập text bạn đọc trực tiếp từ ảnh, xác nhận rồi lưu; không cần vẽ box. Mỗi lần sửa là revision mới. Nếu ảnh mờ, bấm **Bỏ qua ảnh không rõ**.
5. `AUTO_ALIGNED` chỉ là đề xuất box/lớp, không phải glyph reviewed. Split `test` không chạy teacher và chỉ lưu text thủ công; nó chưa phải test chính thức. Split dự kiến được giữ cố định sau nhập, không tự phân chia lại.
6. Có thể tải JSON nghiên cứu (không pixel) hoặc xóa ảnh và toàn bộ revision của mẫu qua xác nhận riêng. JSON có thể chứa text biển số và thông tin nguồn; không chia sẻ nếu chưa được phép. Xóa mẫu không thu hồi JSON đã tải hoặc bản backup bên ngoài.

**Training BLOCKED:** ứng dụng chỉ lưu phản hồi, không fit hoặc cập nhật model. Thông tin nguồn do người dùng khai báo chưa đồng nghĩa đã kiểm định tính độc lập.