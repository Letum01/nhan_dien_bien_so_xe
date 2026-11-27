# Hướng dẫn trình diễn

Mở `START_ALPR.bat` từ thư mục dự án, sau đó mở http://localhost:8511. Nếu server đã chạy, dùng URL này; nếu cổng bị ứng dụng khác chiếm, giải phóng cổng trước. Nếu thiếu CUDA/checkpoint, xem thông báo BAT hoặc `reports/dual_yolo_worker.log`; không fallback CPU cho detector.

1. **Ảnh đơn:** upload JPG/JPEG/PNG, giữ YOLO Character + CNN Mix hoặc chọn cấu hình khác, bấm **Chạy nhận diện**. Box xanh là đề xuất detector. Nếu không có plate ≥0,4, ứng dụng báo không thấy biển.
2. **So sánh thuật toán:** dùng cùng crop/segmentation cho bốn backend; xem thứ tự, phiếu đồng thuận và timing trong JSON. Phiếu không phải xác suất.
3. **Hàng loạt:** tối đa 20 ảnh/lượt, mỗi ảnh tối đa 20 MiB và 24 MP. Lỗi tệp được báo riêng, ảnh sau vẫn xử lý. Tải CSV sau khi hoàn tất.
4. **Lịch sử:** bật lưu trước khi chạy nếu muốn lưu tên ảnh, chuỗi, thuật toán, thời điểm, độ trễ và trạng thái. Không lưu pixel. Có tìm kiếm, CSV và xóa có xác nhận.
5. **Đánh giá:** validation, pilot và trạng thái New10 được hiển thị riêng. Không suy diễn chỉ số pilot thành accuracy tổng quát.

Đường dẫn nhận diện không dùng ảnh lịch sử làm train. Giao diện gắn nhãn cũ và ứng dụng production vẫn độc lập.