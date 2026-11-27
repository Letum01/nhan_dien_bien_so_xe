# Model Registry — cấu hình demo đã khóa

| Vai trò | Đường dẫn tương đối từ workspace | Hợp đồng |
|---|---|---|
| YOLO Plate Round 2 | `experiments/yolo_round2_splitB_run01/weights/best.pt` | CUDA; conf 0,40; chọn box score cao nhất |
| YOLO Character Round 1 | `experiments/yolo_character_round1/weights/best.pt` | CUDA; một lớp; conf 0,25 |
| CNN Mix V1 | `experiments/ocr_mix_experiment02_20261125T080339560263Z/cnn_mix/cnn_model.keras` | Keras CPU; `cnn_contract.json` cùng thư mục; 32×32×1, rescale một lần |
| SVM Mix | `experiments/ocr_mix_experiment01_20261125T072726619780Z/svm_mix/` | `svm_model.joblib` + `hog_scaler.joblib`, HOG 324 |
| RF Mix | `experiments/ocr_mix_experiment01_20261125T072726619780Z/rf_mix/rf_model.joblib` | HOG 324, không áp scaler SVM |
| SVM/RF Fusion | `experiments/hybrid_features_01/{svm_fusion,rf_fusion}.joblib` | HOG 324 rồi CNN dense 256; SVM dùng scaler đi kèm |

Class order: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`. Hash baseline và các file inference nằm trong `reports/final_ocr_protocol.json`; hash hai đầu Fusion ở `reports/adaptive_score_probe.json` (`head_hashes`). Không tải trọng số từ nguồn không kiểm chứng; Keras/joblib chỉ đọc bản tin cậy.

Khôi phục các artifact theo đúng đường dẫn từ bản sao được phép, gồm protocol, `cnn_contract.json` và các báo cáo dùng cho tab Đánh giá (`experiments/yolo_character_round1/comparison/results.json`, `reports/dual_yolo_profile.json`, tùy chọn `reports/alpr_demo_pilot.json`). Learning Center/cohort yêu cầu kho nguồn và manifest tương ứng; không phát hành các SQLite, ảnh hoặc archive riêng tư để làm bản clone tự chạy. `restore_assets.py` có chế độ dry-run, chỉ copy khi bật `--apply`, không ghi đè đích khác hash.

```powershell
.\.venv\Scripts\python.exe -c "from final_ocr_dataset import check_protocol; check_protocol()"
```

`START_ALPR.bat` kiểm tra hash và CUDA trước khi mở Fast V1. Không sửa protocol để chấp nhận checkpoint khác.