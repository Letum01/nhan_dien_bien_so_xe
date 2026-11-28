# Demo

Khôi phục checkpoint theo MODEL_REGISTRY, chạy START_ALPR.bat, mở http://localhost:8511.

- Ảnh đơn: JPG/PNG, chọn CNN Mix/SVM Fusion/Random Forest Fusion, bấm Chạy nhận diện.
- So sánh: ba classifier dùng cùng crop/box/thứ tự; bất đồng chỉ cảnh báo, không sửa chuỗi.
- Batch: tối đa20 ảnh, mỗi ảnh20MiB/24MP; lỗi riêng từng ảnh, xuất CSV.
- History opt-in: tên file/text/timing, không ảnh; tìm kiếm/CSV/xóa xác nhận.
- Đánh giá: artifact development nếu đã khôi phục; không phải final test.

Score box chưa hiệu chuẩn OCR; một biển score cao nhất mỗi ảnh. Không phát hiện ở .40 hoặc không có glyph sẽ báo rõ. Lỗi CUDA/checkpoint xem reports/dual_yolo_worker.log, không fallback CPU. Chỉ giữ app8511, không UI nghiên cứu/review.
