# Recipe checkpoint chính

## Sau remediation — nguồn artifact bên ngoài

Đặt `ALPR_TRAINING_ROOT` tới artifact pack tin cậy có cấu trúc tương đối gốc.
Resolver chỉ rebasing các scope đã biết; hash trong receipt/protocol không thay đổi.
Không sửa backup hoặc copy dataset vào runtime project.

```powershell
$env:ALPR_TRAINING_ROOT = '<artifact-pack>'
python training_preflight.py plate
python training_preflight.py character
python training_preflight.py cnn
python training_preflight.py svm
python training_preflight.py rf
```

Các lệnh trên KHÔNG fit. PASS chỉ xác nhận input/validation đã được chạy, không
chứng minh execution hoặc tái lập trọng số. YOLO dataset YAML lịch sử mang root
cũ được xác minh bằng resolver; trước fitting phải export dataset/YAML vào output
mới hợp lệ, không ghi đè snapshot backup. Không chạy fitting trực tiếp trên backup.

Helper mới: `fusion_dataset.py`, `image_regions.py`, `text_metrics.py`,
`annotation_gate.py`. Các tên V2/multiclass/recovery cũ trong phần lịch sử bên dưới
chỉ mô tả nguồn gốc, không còn entrypoint. `segmentation_v2.py` được phục hồi nguyên
byte vì hash gate cần, không được đưa vào lựa chọn runtime.

`configs/alpr_progressive_protocol_v1.json`, `external_ocr_benchmark_protocol_v1.json`,
`external_task_quality_v1.json`, `ocr_v2_preregistered.json` là metadata protocol
lịch sử, không phải config inference và không còn runner progressive tương ứng.
Giữ chúng để bảo toàn protocol; không quảng bá là training feature hiện hành.

Không fit trong cleanup. Output mới, không overwrite model. Source không kèm ảnh/snapshot/manifest nên không đảm bảo tái lập bit-exact.

## YOLO

Plate Round2: label_detection.py CLI export annotation đã duyệt; prepare_round2.py/apply_approved_split_b.py/audit_round2_training.py/validate_yolo_dataset.py/train_plate_detector.py giữ validation và recipe. Split46/12/3 ảnh, imgsz640,batch4,57epoch(best37),seed42; config yolo_round2.json. Cần approval receipt và pretrained local.

Character Round1: glyph_review.py/label_characters.py export có validation, không UI. yolo_character_dataset.py kiểm tra boxes/revision/group/leakage. train_yolo_character.py giữ gate receipt gốc; annotation_gate.py chỉ còn membership helper, không benchmark runner cũ. 46crop/370box train,12crop/99box validation,imgsz640,batch4,53epoch(best33). Không nới gate.

## CNN Mix V1

ocr_mix_experiment.py, run_ocr_mix_experiment01/02.py, models.py, dataset.py giữ recipe/augmentation/preprocessing. Synthetic V1 thực sự dùng generate_synthetic_character_dataset trong dataset.py; cần glyph real và manifest riêng. Keras32×32,uint8,class order36,Rescaling/255 duy nhất. Không convert checkpoint.

## Fusion

hybrid_experiment.py giữ HOG+V1 dense, hybrid_features.py chỉ trích V1. fusion_dataset.py nay chỉ helper snapshot/domain weights(mean1), không trainer V2. ocr_v2_synthetic.py vẫn cần vì archive synthetic thực sự dùng cho Fusion bắt nguồn ở recipe này.

Cần pixels.npy/samples.json/features/hash/manifest gốc experiments/hybrid_features_01, đưa vào output bản sao mới. GroupKFold3 chọn Linear/RBF SVM và RF, scaler train-only. Frozen CNN có thể từng tiếp xúc holdout: chỉ head CV, không độc lập toàn hệ thống.

Giữ SHA/pHash/source-group/parent/revision validation. Không dùng dữ liệu chưa đủ quyền hoặc Pilot/New10/Final Test để fit/tuning. Không review lại mọi glyph, không dự đoán làm nhãn thật. CLI --help cho tham số từng recipe. Không giữ UI review hoặc benchmark nghiên cứu thất bại.
