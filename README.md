# ALPR — Nhận diện biển số xe Việt Nam

Đồ án Máy học: **YOLO Plate → YOLO Character → CNN / SVM Fusion / Random Forest Fusion**. CNN Mix V1 mặc định; không OCR bên ngoài.

## Windows

Python3.11, NVIDIA driver hỗ trợ CUDA12.1; RTX2060 6GB đã kiểm tra. Hai môi trường: `.venv` (Keras CPU/Streamlit/scikit-learn) và `.venv-yolo` (PyTorch CUDA/Ultralytics).

```powershell
# Trong thư mục clone
.\setup_environment.bat
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

**Clone source chưa đủ chạy demo.** Khôi phục artifact tin cậy và được phép sử dụng theo [registry](docs/MODEL_REGISTRY.md), rồi:

```powershell
.\.venv\Scripts\python.exe -c "from final_ocr_dataset import check_protocol; check_protocol()"
.\START_ALPR.bat
```

Chỉ một ứng dụng **http://localhost:8511**: ảnh đơn, batch, CNN/SVM Fusion/RF Fusion trên cùng crop và boxes, CSV, lịch sử opt-in không lưu pixel. Learning Center/Deep Analysis độc lập đã bỏ source; không xóa SQLite review.

## Kiến trúc

Ảnh → kiểm tra giới hạn → Plate(conf.40) → padding15% → Character(conf.25) → sắp dòng → uint8 32×32 → classifier → UI/CSV.

Fusion = HOG324 + CNN dense256 =580 chiều; hai đầu dùng chung extractor. Persistent CUDA worker không reload mỗi ảnh. `alpr/`, `app_alpr_demo.py`, `demo_*.py` là UI/điều phối; `dual_yolo_*` là worker/pipeline. Recipe training và validators giữ ở gốc để bảo toàn đường dẫn. Một số đối chứng bên trong module khóa byte phải giữ nhưng không nạp mặc định.

## Kết quả

| Classifier | Matched glyph | Exact | CER |
|---|---:|---:|---:|
| CNN Mix V1 |95/98|8/12|4,04%|
| SVM Fusion |95/98|8/12|4,04%|
| RF Fusion |94/98|7/12|6,06%|

Plate Round2 mAP50 97,15%; Character Round1 F1 98,49% trên crop YOLO. Khác phạm vi; **12 development đã dùng nhiều lần, không phải test độc lập hoặc accuracy90%**. Chỉ trả biển score cao nhất mỗi ảnh. Score/đồng thuận không phải xác suất OCR đúng.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv-yolo\Scripts\python.exe -m pip check
```

[Kiến trúc](docs/ARCHITECTURE.md) · [Sử dụng](docs/USER_GUIDE.md) · [Training](docs/TRAINING_AND_DATASET.md) · [Đánh giá](docs/EVALUATION_REPORT.md) · [Deep Clean](reports/ALPR_DEEP_CLEAN_REPORT.md).

YOLO: [Ultralytics](https://github.com/ultralytics/ultralytics), tuân thủ giấy phép phần mềm/dữ liệu. Không phát hành ảnh/ZIP/SQLite/checkpoint chưa đủ quyền. Source không đủ tái lập bit-exact nếu thiếu snapshot/manifest gốc.
