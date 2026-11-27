# Kiến trúc ALPR

```mermaid
flowchart LR
    UI[Streamlit demo 8511] --> C[Điều phối .venv]
    C -->|Request ID và PNG tạm| W[Persistent worker .venv-yolo]
    W --> P[YOLOv8n Plate trên CUDA]
    W --> G[YOLOv8n Character trên CUDA]
    W -->|Box và detector score| C
    C --> OCR[OCR Mix trên CPU]
    C -->|Nếu người dùng bật lưu| DB[(SQLite lịch sử demo)]
    OCR --> UI
```

```mermaid
flowchart LR
    I[JPG/PNG hợp lệ] --> P[Plate 0.4]
    P --> B[Chọn score cao nhất và padding 15% mỗi phía]
    B --> S[Character 0.25 hoặc OpenCV]
    S --> O[Sắp xếp glyph theo hàng]
    O --> N[Nhị phân và pad vuông 32x32]
    N --> H[HOG 324]
    N --> C[CNN Mix]
    H --> M[SVM và RF Mix]
    M --> V[Majority tùy chọn]
    C --> V
    V --> R[Text thô và kết quả]
```

Hai YOLOv8n chỉ tìm box: Plate tìm biển trên ảnh gốc, Character tìm glyph không phân lớp ký tự trên crop. OpenCV giải mã, crop, nhị phân hóa và chuẩn hóa; hai đối chứng OpenCV dùng contour, với `border_components` bỏ thành phần nối biên. YOLO Character cải thiện segmentation trên development nhưng cần GPU và có thể bỏ/thừa glyph. HOG tóm tắt hướng gradient thành 324 đặc trưng; SVM RBF tìm biên quyết định sau scaler, RF bỏ phiếu qua cây (không cần scaler), CNN học đặc trưng từ ảnh 32×32×1 và rescale nội bộ. Majority dùng ít nhất hai phiếu trong ba, không đa số trả `?`; không hiệu chỉnh xác suất.

Worker giữ hai checkpoint và warmup một lần mỗi tiến trình, đồng bộ CUDA để đo predict, tuần tự hóa IPC, có timeout và khởi động lại ở yêu cầu tiếp theo sau lỗi. Classifier cache trong Streamlit; không cache prediction. SQLite demo chỉ lưu metadata khi bật lưu, tách khỏi SQLite nhãn test và train. Không có PaddleOCR trong demo.

## Ranh giới module sau refactor

`app_alpr_demo.py` giữ widget, cache tài nguyên và điều phối upload. `alpr/demo_service.py` chọn backend, chuyển kết quả thành dòng CSV/lịch sử và lưu case nghiên cứu khi được yêu cầu. `alpr/demo_ui.py` dùng chung hiển thị ảnh BGR và vẽ thứ tự glyph cho Fast V1/Deep Analysis. `demo_alpr_core.py` giữ decode, xuất CSV và SQLite. Adapter giữ nguyên schema dictionary của kết quả inference (`plate`, `glyph_boxes`, `glyph_scores`, `text`, `timings`, `worker`); box ký tự là tọa độ tương đối theo crop, thứ tự là thứ tự trong danh sách. `DemoRow` là hợp đồng riêng cho CSV/lịch sử, không đổi schema database.

`dual_yolo_client.py` và `dual_yolo_worker.py` giữ việc vận hành tiến trình/IPC. `dual_yolo_pipeline.py` gọi crop, ordering và classifier theo hợp đồng đã khóa. Fusion giữ HOG 324 + embedding 256 theo đúng thứ tự, chia sẻ vector 580 cho hai đầu Fusion trong một lượt. Không áp scaler của SVM sang RF, không đổi framework, dtype hoặc confidence. Xem [Model Registry](MODEL_REGISTRY.md) và [Developer Guide](DEVELOPER_GUIDE.md).