# Hướng dẫn trình diễn

Mở `START_ALPR.bat` từ thư mục dự án, sau đó mở http://localhost:8511. Nếu server đã chạy, dùng URL này; nếu cổng bị ứng dụng khác chiếm, giải phóng cổng trước. Nếu thiếu CUDA/checkpoint, xem thông báo BAT hoặc `reports/dual_yolo_worker.log`; không fallback CPU cho detector.

1. **Ảnh đơn:** upload JPG/JPEG/PNG, giữ YOLO Character + CNN Mix hoặc chọn cấu hình khác, bấm **Chạy nhận diện**. Box xanh là đề xuất detector. Nếu không có plate ≥0,4, ứng dụng báo không thấy biển.
2. **So sánh thuật toán:** dùng cùng crop/segmentation cho CNN Mix, SVM Mix, RF Mix, Majority Mix và hai đầu SVM/RF Fusion. Xem thứ tự box, cảnh báo bất đồng và timing. Phiếu không phải xác suất; không tự sửa chuỗi.
3. **Hàng loạt:** tối đa 20 ảnh/lượt, mỗi ảnh tối đa 20 MiB và 24 MP. Lỗi tệp được báo riêng, ảnh sau vẫn xử lý. Tải CSV sau khi hoàn tất.
4. **Lịch sử:** bật lưu trước khi chạy nếu muốn lưu tên ảnh, chuỗi, thuật toán, thời điểm, độ trễ và trạng thái. Không lưu pixel. Có tìm kiếm, CSV và xóa có xác nhận.
5. **Đánh giá:** validation, pilot và trạng thái New10 được hiển thị riêng. Không suy diễn chỉ số pilot thành accuracy tổng quát.

Đường dẫn nhận diện không dùng ảnh lịch sử làm train. Giao diện gắn nhãn cũ và ứng dụng production vẫn độc lập.

**Deep Analysis:** mở `OPEN_ALPR_DEEP.bat` hoặc [8512](http://localhost:8512). Dùng để chẩn đoán glyph, box và top-k; không tự lưu pixel. **Learning Center:** mở `OPEN_ALPR_LEARNING_CENTER.bat` hoặc [8513](http://localhost:8513); đường dẫn [cohort review](http://localhost:8513/?mode=cohort) dùng các revision hiện có. Xem [hướng dẫn cohort](ALPR_COHORT_REVIEW_GUIDE.md). Nếu chuyển máy, xem [bộ artifact cần khôi phục](MODEL_REGISTRY.md) trước khi chạy.