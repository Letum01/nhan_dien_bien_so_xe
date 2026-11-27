# ALPR — Nhận diện biển số xe Việt Nam

Đồ án máy học phát hiện biển và ký tự bằng **hai detector YOLOv8n**, sau đó nhận dạng ký tự bằng **CNN, SVM hoặc Random Forest**. Bản demo giữ nguyên checkpoint V1 đã kiểm định; không dùng dịch vụ OCR bên ngoài.

## Chạy trên Windows

Từ thư mục dự án, dùng Python 3.11, driver NVIDIA hỗ trợ CUDA 12.1 và GPU tương thích. Cấu hình đã kiểm tra: RTX 2060 6 GB, hai môi trường `.venv` (Streamlit/TensorFlow CPU/scikit-learn) và `.venv-yolo` (PyTorch CUDA/Ultralytics).

```powershell
Set-Location 'C:\Users\Admin\Documents\nhan_dien_bien_so_xe'
.\setup_environment.bat
```

**Clone mã nguồn chưa đủ để chạy demo.** Khôi phục các checkpoint và artifact được phép chia sẻ từ bản sao cục bộ đáng tin cậy theo [Model Registry](docs/MODEL_REGISTRY.md). Giữ nguyên đường dẫn và hash; không dùng pointer Git LFS thay cho trọng số. Không cần copy database cá nhân để chạy nhận diện ảnh.

| Ứng dụng | Khởi động | URL |
|---|---|---|
| Fast V1 | `START_ALPR.bat` | [8511](http://localhost:8511) |
| Deep Analysis | `OPEN_ALPR_DEEP.bat` | [8512](http://localhost:8512) |
| Learning Center | `OPEN_ALPR_LEARNING_CENTER.bat` | [8513](http://localhost:8513) |

Fast V1 mặc định Dual-YOLO + CNN Mix. Hỗ trợ ảnh đơn, batch, so sánh cả hai đầu Fusion, box biển/glyph, CSV và lịch sử opt-in. Xem [hướng dẫn sử dụng](docs/USER_GUIDE.md).

## Kiến trúc và bố cục

Ảnh hợp lệ → YOLO Plate → crop → YOLO Character → thứ tự glyph → OpenCV chuẩn hóa 32×32 → CNN hoặc HOG/SVM/RF → UI/CSV/SQLite. Fusion ghép HOG 324 chiều với embedding CNN V1 256 chiều. Worker CUDA persistent giữ hai checkpoint trong một tiến trình riêng. Xem [sơ đồ và hợp đồng runtime](docs/ARCHITECTURE.md).

- `alpr/`: adapter runtime và thành phần hiển thị dùng chung; entrypoint `app_*.py` vẫn ở gốc.
- `configs/`, `tests/`, `docs/`: cấu hình thí nghiệm, kiểm thử và tài liệu.
- Script train/evaluation/research vẫn ở đường dẫn cũ để bảo toàn CLI, IPC và hash protocol. Xem [Developer Guide](docs/DEVELOPER_GUIDE.md) để tìm đúng entrypoint.
- `data/`, `dataset/`, `datasets/`, `external_datasets/`, `experiments/`, `reports/`: dữ liệu và artifact cục bộ, không phải mã nguồn để stage hàng loạt.

## Kết quả và giới hạn

Trên **12 biển development đã dùng nhiều lần**, CNN Mix V1 và SVM Fusion đạt **8/12 exact, CER 4,04%**; RF Fusion đạt **7/12, CER 6,06%**. Plate Round 2 mAP50 **97,15%** trên validation detection; Character Round 1 F1 **98,49%** trên crop YOLO. Các phép đo có phạm vi khác nhau, không phải final test hay accuracy production.

Demo chỉ trả một biển có score detector cao nhất trên mỗi ảnh. Điểm detector và số phiếu không phải xác suất đúng. Ảnh mờ, nghiêng hoặc ký tự nhỏ có thể gây lỗi. [Báo cáo đồ án](reports/ALPR_FINAL_ACADEMIC_REPORT.md) · [Báo cáo refactor](reports/ALPR_CLEAN_CODE_REPORT.md).

## Huấn luyện và trích dẫn

Huấn luyện là quy trình riêng, **không cần để chạy demo**. Xem [dữ liệu/huấn luyện](docs/TRAINING_AND_DATASET.md) và [registry](docs/MODEL_REGISTRY.md). Không dùng dữ liệu không rõ quyền, không dùng Pilot/New10/Final Test để fit hoặc tuning. UCI Optdigits CC BY 4.0 chỉ là đối chứng viết tay: [Alpaydin & Kaynak](https://doi.org/10.24432/C50P49), không dùng để báo accuracy biển Việt Nam. YOLO sử dụng [Ultralytics](https://github.com/ultralytics/ultralytics). Nhật ký các thử nghiệm giữ tại [Development History](docs/DEVELOPMENT_HISTORY.md).