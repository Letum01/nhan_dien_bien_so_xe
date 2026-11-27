# Developer Guide — ALPR

## Bản đồ phụ thuộc vận hành

| Nhóm | Entry point và quan hệ cần giữ |
|---|---|
| Runtime | `START_ALPR.bat` → `app_alpr_demo.py` → `alpr/demo_service.py` → `dual_yolo_pipeline.py` / `demo_fusion.py` → `dual_yolo_client.py` → subprocess `dual_yolo_worker.py` trong `.venv-yolo`. |
| Hiển thị và lưu trữ | `alpr/demo_ui.py` dùng chung phần vẽ box; `demo_alpr_core.py` kiểm tra ảnh, CSV, SQLite lịch sử opt-in. Hai app Fast/Deep giữ API import cũ. |
| Deep và review | `OPEN_ALPR_DEEP.bat` → `app_alpr_deep.py` → `alpr_recovery.py`; `OPEN_ALPR_LEARNING_CENTER.bat` → `app_learning_center.py` → `learning_center.py`, dynamic `learning_center_teacher.py` và `cohort_review.py`. |
| Training | `train_plate_detector.py`, `train_yolo_character.py`, `train_real_characters.py`, `progressive_train.py`, bộ chuẩn bị dataset; chỉ gọi qua CLI và preflight riêng. |
| Evaluation | `benchmark_yolo_character.py`, `profile_dual_yolo.py`, validators và regression scripts. Lưu ý hàm crop và một số helper từ các module benchmark vẫn là dependency runtime. |
| Research / legacy | Script V2–V5, UCI, recovery, annotation cũ, `app.py`, tích hợp OCR cũ. Giữ nguyên đường dẫn để tái lập; không kết luận unused chỉ vì không có static import. |
| Generated | `.venv*`, cache, logs, output train, dataset và artifacts. Không stage; dữ liệu, checkpoint và kết quả thí nghiệm không được mặc định xóa. |

Kiểm tra import trực tiếp, dynamic import, BAT, subprocess arguments, literal path và hash protocol trước khi di chuyển module. `final_ocr_protocol.json` khóa nhiều file Python và trọng số; không đổi file hoặc cập nhật hash chỉ vì format. Không gom hai môi trường vào một venv.

## Kiểm thử trên bản sao được phép

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv-yolo\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe browser_check_alpr_demo.py
.\.venv\Scripts\python.exe browser_check_alpr_deep.py
.\.venv\Scripts\python.exe browser_check_learning_center.py
.\.venv\Scripts\python.exe browser_check_cohort_review.py
```

Hai check đầu cần cổng 8511/8512 đã chạy và ảnh development được phép. Hai check review dùng SQLite tạm. Đối chiếu replay bằng cùng ảnh, model hash, box/crop, glyph order, score và predicted text trước/sau. Không đụng test bảo vệ để kiểm thử refactor.

Mã mới dùng PEP 8 và type hints ở viền adapter. Không cài formatter mới hoặc format hàng loạt file khóa. Không dùng `git add -A` trên workspace chứa artifact; kiểm tra danh sách path staged và blob trước khi commit. `.gitignore` không bỏ theo dõi file đã tracked.