# Đánh giá thực nghiệm

Nguồn số liệu: `experiments/yolo_character_round1/comparison/results.json`, `experiments/yolo_character_round1/validation_ap/ap.json`, `reports/alpr_demo_pilot.json`, `reports/dual_yolo_profile.json` và `reports/alpr_demo_new10_audit.json`.

## Validation: 12 biển, 99 ký tự

Các cấu hình dưới đây cùng crop YOLO Plate từ ảnh gốc. Mỗi ô là **exact trên 12 biển / CER**.

| Segmentation | SVM Mix | RF Mix | CNN Mix | Majority |
|---|---:|---:|---:|---:|
| OpenCV baseline | 4 / 35,35% | 3 / 38,38% | 3 / 36,36% | 4 / 35,35% |
| border_components | 5 / 19,19% | 3 / 21,21% | 4 / 20,20% | 5 / 19,19% |
| YOLO Character | 5 / 8,08% | 3 / 13,13% | 8 / 4,04% | 6 / 7,07% |

Character mAP50/95 trên crop GT: mAP50 99,50%, mAP50–95 72,98%. Trên crop YOLO: 98 TP, 2 FP, 1 FN, F1 98,49%; CNN đúng 95/98 match. CNN Mix là lựa chọn demo đã xác định trước vì kết quả development, không chọn lại bằng pilot. Profiling trước đây đo warm CNN backend median 124 ms trên RTX 2060, không gồm render trình duyệt.

## Pilot riêng: 3 biển, 24 ký tự, không blind

Cả ba ảnh đã xuất hiện trong diagnostics; Plate phát hiện đủ 3/3 ở ngưỡng 0,4 trong lượt này. Mỗi ô là **exact trên 3 biển / CER**.

| Segmentation | SVM Mix | RF Mix | CNN Mix | Majority |
|---|---:|---:|---:|---:|
| OpenCV baseline | 0 / 41,67% | 0 / 41,67% | 1 / 37,50% | 0 / 41,67% |
| border_components | 0 / 20,83% | 0 / 20,83% | 1 / 16,67% | 0 / 20,83% |
| YOLO Character | 1 / 8,33% | 1 / 8,33% | 1 / 8,33% | 1 / 8,33% |

Dual-YOLO + CNN Mix: 2 edits gồm 1 thay thế, 0 thiếu, 1 thừa. Toàn bộ breakdown và kết quả từng ảnh của 12 cấu hình nằm trong `reports/alpr_demo_pilot.json`. Cold n=1: 6,287 s; warm n=8 gộp ba segmentation: mean 135,7 ms, median 131,1 ms, p95 160,1 ms. OCR load riêng khoảng 0,902 s. CUDA detector, CPU OCR. Không gộp các tập hoặc suy ra accuracy tổng quát.

## New10 — BLOCKED

10 ảnh được duyệt, 85 ký tự, 8 ID nhóm khai báo, trong đó 1 glyph-reviewed và 9 text-reviewed. Kỹ thuật fingerprint và chốt leakage theo chỉ mục vượt kiểm tra, nhưng cả 10 bằng chứng chỉ ghi ngày nhập; số nhóm độc lập chưa xác minh. Không freeze, không inference, không tính accuracy/F1/IoU New10. Corpus mở rộng giữ trạng thái chưa hoàn tất.