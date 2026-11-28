# Model Registry

## Runtime và research validation riêng

`python runtime_validation.py` dùng `configs/runtime-inference.json`, chỉ cần năm
checkpoint chính và CNN contract; tổng binary **23.612.029 byte**. Manifest này
được lấy từ hash release đã kiểm chứng, không tự học hash của file bất kỳ.

`final_ocr_dataset.check_protocol()` giữ nguyên hash/protocol nghiên cứu và vẫn
đòi ba file đối chứng HOG cũ. Registry `runtime-assets.json` phía dưới tiếp tục là
pack lịch sử đầy đủ để restore research, không phải minimum runtime.
Không xóa vĩnh viễn đối chứng: test đã tạm giấu rồi khôi phục hash nguyên vẹn.

Runtime: YOLO Plate Round2, Character Round1, CNN Mix V1, SVM Fusion, RF Fusion. Hash/bytes trong configs/runtime-assets.json.

SVM Mix/RF Mix/scaler cũ không nạp runtime mặc định nhưng vẫn cần phục hồi để gate final_ocr_protocol xác minh baseline khóa. Không sửa protocol. Checkpoint chỉ khôi phục từ nguồn tin cậy có quyền sử dụng/chia sẻ; không tự download.

```powershell
# Trong project, thay <asset-pack> bằng bản sao được phép
.\.venv\Scripts\python.exe restore_assets.py configs/runtime-assets.json '<asset-pack>' '.'
# Sau dry-run thành công
.\.venv\Scripts\python.exe restore_assets.py configs/runtime-assets.json '<asset-pack>' '.' --apply
.\.venv\Scripts\python.exe -c "from final_ocr_dataset import check_protocol; check_protocol()"
```

Không overwrite đích khác hash. CNN cần cnn_contract.json; tab Đánh giá tùy chọn cần experiments/yolo_character_round1/comparison/results.json. Không cần SQLite review để inference.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `experiments/yolo_round2_splitB_run01/weights/best.pt` | 6230819 | `b5c7261ec9db97878b276b7c02e853736b209110f9f59a068e84fcadd6d98cdd` |
| `experiments/yolo_character_round1/weights/best.pt` | 6243619 | `aff3e82866b3b0993fa6ea91d9dc3e420c165ccc4dc15be88e15d47bad4bdf2a` |
| `experiments/ocr_mix_experiment01_20261125T072726619780Z/svm_mix/svm_model.joblib` | 7283039 | `cdd6a43584486a3b4547f25119f4a27bf3f5f99acf110d685f3518dcd966afc0` |
| `experiments/ocr_mix_experiment01_20261125T072726619780Z/svm_mix/hog_scaler.joblib` | 8359 | `e27e2e6f6436521b7d7b4b0d2fc7d0c72c86cde648a9bfa304367cdb3378a940` |
| `experiments/ocr_mix_experiment01_20261125T072726619780Z/rf_mix/rf_model.joblib` | 28147609 | `c849de4f88a76159784685e5fe7664d63c7a2c2baaf9ed6e632d2024175c54fd` |
| `experiments/ocr_mix_experiment02_20261125T080339560263Z/cnn_mix/cnn_model.keras` | 8204189 | `34e88fdfad393aa2bd0da1812e0cc96740a0aee1c61381064af3163a21072d42` |
| `experiments/hybrid_features_01/svm_fusion.joblib` | 1785992 | `3dd58e0bc674f201d8887f70d51808cce86ce234a8dda40f9a6bd2ba9aae2824` |
| `experiments/hybrid_features_01/rf_fusion.joblib` | 1147410 | `13b818f045d9b230d018065901d9dca1df778fe9b33eca2eafddabb87c4ff7cd` |
| `experiments/ocr_mix_experiment02_20261125T080339560263Z/cnn_mix/cnn_contract.json` | 852 | `8fcd8de6adef80e5fee2eca582bccf438d1f6ca8db45c58261e982fba1b4b771` |
| `reports/final_ocr_protocol.json` | 3151 | `865ca7d977b0615da890303628d4c23cbad6c48063d829538a7f892a3e22d59a` |
| `reports/adaptive_score_probe.json` | 6105 | `428ab7d21593ca0be02a409244d6112e84356b0546d69f7790bbd7e1f9e41186` |
