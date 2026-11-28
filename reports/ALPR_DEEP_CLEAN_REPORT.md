# ALPR Deep Clean — bàn giao

## Phạm vi và số lượng

Branch `release/alpr-source-only`, không tạo branch, không commit/push. Không fit model, không đổi checkpoint/threshold/preprocessing. Backup người dùng giữ nguyên. Kiểm chứng ở staging cùng máy `C:\Users\Admin\AppData\Local\Temp\alpr-deep-clean-20261127`, không phải máy mới.

- Source tracked tồn tại:292 →99; xóa thực sự193file (không archive).
- Python:249 →73file;20.171 →6.437dòng (giảm khoảng68%).
- Test modules:63 →11; bỏ52module thuộc chức năng nghiên cứu/UI/legacy đã loại. Suite cuối45PASS,6warning sklearn fixture.
- Dung lượng99source tracked còn lại369.256byte trước bổ sung registry/báo cáo; không tính dataset/venv.
- Thêm configs/runtime-assets.json chứa tên/bytes/SHA, không chứa model.
- Danh sách file xóa và thống kê: reports/deep_clean_deleted.json, deep_clean_stats.json. Ba file xóa bổ sung: label_crops.py,database.py,detection_backends.py.

## Xóa và tinh giản

Đã bỏ Learning Center/cohort/glyph/box review UI, Deep Analysis độc lập và startup phụ; Recovery/Boost/Oracle/adaptive selector/hard-negative, progressive/external audit và runner nghiên cứu không cần. Paddle client/worker/requirements bị xóa. Không tạo ứng dụng/framework/archive mới.

Helper tên lịch sử được rút gọn: train_ocr_v2 chỉ snapshot/weights/HOG; svm_recovery_audit chỉ checked image/crop; multiclass_run chỉ edit_distance; run_hard13_v2 chỉ annotation gate. label_detection/label_characters chỉ export, không UI. hybrid_features chỉ V1; hybrid_experiment chỉ HOG+V1 dense. Giữ synthetic generator thật sự dùng cho V1/Fusion.

12module khóa protocol giữ nguyên byte. Một số nhánh OpenCV/Mix/Majority bên trong buộc giữ để không phá protocol nhưng không xuất hiện trên UI hoặc nạp mặc định. Các config/protocol khóa và artifact nghiên cứu cục bộ không bị xóa, không đưa cả reports lên Git. main.py là support training/evaluation, không còn app chính. Không tuyên bố đã xóa mọi dòng legacy bên trong file khóa.

## Runtime cuối

Một app8511: ảnh đơn,batch,CNN Mix mặc định/SVM Fusion/RF Fusion,compare cùng crop/box/thứ tự,cảnh báo bất đồng,CSV,history opt-in không ảnh. Fusion nhận đúng batch normalized tại CNN boundary, tái sử dụng HOG/embedding cho hai đầu. Persistent CUDA worker giữ nguyên. Requirements bỏ drawable-canvas, chuyển pytest sang dev, không uninstall/nâng cấp môi trường.

## Kiểm chứng đã thực hiện

- Full retained pytest:45PASS,6warning,không skip ở workspace cuối.
- Staging trước điều chỉnh unit test:44PASS/1SKIP vì thiếu characters.csv. Unit test registry nay mock gate model để kiểm tra fixture độc lập; gate runtime thật vẫn kiểm tra trong CUDA replay, không hạ chốt.
- AST compile source còn lại và import UI/training:PASS.
- CUDA replay12/12: plate bbox/crop coordinates,glyph boxes/order,scores,chuỗi ba classifier khớp baseline.
- CNN95/98 matched,8/12 exact,CER4/99=4,04%; SVM Fusion95/98,8/12,4/99; RF Fusion94/98,7/12,6/99=6,06%. Geometry/prediction giống hệt nên matched counts bảo toàn. Không nâng accuracy.
- Hash12source khóa và checkpoint protocol:PASS. Không normalize CRLF các file khóa.
- Chromium:ảnh thật/CNN/hai Fusion/compare trio/batch/CSV/corrupt/no-plate PASS. Không kiểm tra UI đã xóa.
- pip check cả .venv và .venv-yolo:không broken requirements.
-9SQLite trong backup chỉ đọc,integrity_check=ok; workspace source-only không có data gốc. Không ghi SQLite review hoặc tạo quyết định.
- Warm3classifier,cùng worker:11lượt,median205,89ms,p95261,21ms. Không so với124msCNN đơn. Chưa có benchmark xen kẽ before/after cùng3backend nên không tuyên bố tăng tốc. Memory/worker metadata trong deep_clean_replay.json.
- git diff --check dùng core.whitespace=cr-at-eol vì module khóa CRLF. Các file sửa được normalize LF, không sửa file khóa.

## Triển khai và hạn chế

Không xóa dataset/ZIP/checkpoint/scaler/SQLite/annotation/protocol/backup hoặc file ngoài workspace. Registry runtime-assets và MODEL_REGISTRY hướng dẫn restore_assets (dry-run,không overwrite khác hash). Clone không tự có model. Demo đang kiểm tra tại staging có artifact hợp lệ; workspace chính source-only cần restore trước START_ALPR.bat.

Source/CLI chủ yếu giữ ở root để không phá dependency. Recipe cần dữ liệu/snapshot/approval/receipt gốc, không tuyên bố tái lập bit-exact chỉ từ clone. Không fit pthang23/VNLP,không review thêm.12development đã dùng nhiều lần,không độc lập,không tuyên bố90–95%.

Không xóa đệ quy artifact/cache trong venv. Cache không thuộc bản source phát hành. Một số support dùng tên cũ để tránh đổi đường dẫn.

## Rollback và Git

Chưa stage/commit/push. Source gốc có trong HEAD và backup người dùng. Có thể phục hồi từng file sau sao lưu diff; không reset-hard/clean. Xem git diff và danh sách xóa trước commit. Không đổi lịch sử/remote branch.
