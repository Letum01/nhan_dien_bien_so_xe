# Kiến trúc

`app_alpr_demo → alpr.demo_service → dual_yolo_client.Worker → dual_yolo_worker` (CUDA). YOLO Plate Round2 → padding15% → Character Round1 → order/normalize32×32 → Keras CNN Mix V1 hoặc HOG324+dense256 → SVM/RF Fusion → UI/CSV/SQLite opt-in.

Fusion nhận đúng batch pixel tại boundary CNN, không crop/threshold lần nữa. HOG/embedding dùng chung hai đầu. Không nạp SVM/RF Mix mặc định. Worker serialize request, timeout đóng tiến trình, lần sau restart.

12 module khóa protocol giữ nguyên byte; một số nhánh đối chứng cũ bên trong vẫn cần giữ. `main.py` là support training/evaluation, không còn app OpenCV. Không đổi pixel, class mapping, threshold hoặc checkpoint.

Helper tên lịch sử `train_ocr_v2`, `svm_recovery_audit`, `multiclass_run`, `run_hard13_v2` đã rút gọn, không còn runner nghiên cứu tương ứng. Recipe/validators ở gốc để giữ CLI.
