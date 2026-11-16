# Nhận diện biển số xe Việt Nam (ALPR)

## Cập nhật detector và Paddle persistent

Paddle nay giữ một worker CPU cho mỗi interpreter/config trong tiến trình app,
load model một lần, khóa tuần tự yêu cầu và cache 32 kết quả theo bytes crop +
script + config. Worker timeout/exit được dừng/reap; lỗi hiện tại được báo rõ,
yêu cầu tiếp theo khởi tạo lại. Không âm thầm retry kết quả lỗi. Khởi động lại
app sẽ mất worker/cache. `OCRResult.timings` tách cold start, inference, decode,
encode, request và cache hit. Raw text không bị thay đổi.

```powershell
.\.venv\Scripts\python.exe compare_ocr.py
.\.venv\Scripts\python.exe benchmark_paddle.py
```

Đo trên máy này: cold start ~7.31 giây, warm inference ~0.43–0.48 giây/crop
biển đọc được; cache hit được đo riêng. Xem `persistent_benchmark.json` trong
`debug_outputs/ocr_comparison/`. Đây không phải FPS video hoặc tuyên bố realtime.
Chi phí detector chưa nằm trong benchmark crop OCR này.

OpenCV được giữ lại; bổ sung proposal từ khung tứ giác kín giúp 0405 cắt đủ
biển và Paddle đọc `65-H6 / 9154`. 0407 vẫn crop nhầm đèn; hiện Paddle trả rỗng.
Không suy ra detection accuracy từ sáu ảnh phát triển này.

### Detector nhiều biển và YOLO tùy chọn

`detection_backends.detect_plates` trả nhiều crop/xywh/score; `recognize_multiple`
trả kết quả OCR từng proposal, giữ nhãn `unverified_proposal` cho OpenCV (không
coi mọi proposal là biển đúng). UI ảnh hiện tại vẫn dùng một kết quả tốt nhất.
`end_to_end_pipeline(..., detector_type="yolo", detector_weights="path.pt")`
dùng adapter plate-only. Không có trọng số phù hợp trong workspace: chưa bật
YOLO mặc định và chưa cài ultralytics. Adapter không tải model generic; yêu cầu
file checkpoint local, class plate-only. Tên lớp không tự chứng minh chất lượng
checkpoint: người cung cấp phải xác minh nguồn và đánh giá trên test độc lập.

```powershell
.\.venv\Scripts\python.exe -m streamlit run label_detection.py
# Gắn TẤT CẢ biển trong ảnh bằng [x,y,w,h], kiểm tra overlay, xác nhận thủ công.
.\.venv\Scripts\python.exe label_detection.py --export experiments/plate_yolo_dataset
# Chỉ khi đủ nhãn train/val/test độc lập theo xe/video và cài ultralytics riêng:
python train_plate_detector.py experiments/plate_yolo_dataset/data.yaml local_architecture.yaml experiments/plate_detector_run --epochs 50
.\.venv\Scripts\python.exe evaluate_detection.py dataset/reviewed_detection.json --output opencv_detection.json
```

Nhập [] chỉ cho ảnh thực sự không có biển; không lấy prediction làm box thật.
Exporter kiểm tra source_group/split, không ghi đè output. Trainer bắt đầu từ
architecture YAML local (`pretrained=False`), không download checkpoint generic,
và đánh giá checkpoint best trên split test. Chưa chạy huấn luyện vì thiếu nhãn.
Evaluator tính precision/recall tại IoU=.5 cho proposal (kể cả false positives),
mean matched IoU, không giả lập mAP. OCR vẫn dùng label_crops.py và
evaluate_labeled.py --crop-mode; chưa có nhãn được người dùng duyệt để tính accuracy.

## PaddleOCR tùy chọn — đã kiểm tra với API 3.3.2

Giữ nguyên venv chính và detector OpenCV. Paddle dùng môi trường riêng vì
PaddleX yêu cầu OpenCV-contrib khác phiên bản với ứng dụng. Không cài
`requirements-paddle.txt` vào venv TensorFlow đang dùng:

```powershell
py -3.11 -m venv .venv-paddle
.\.venv-paddle\Scripts\python.exe -m pip install -r requirements-paddle.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Chọn **PaddleOCR (chuỗi trực tiếp)** ở thanh bên. Backend tải lười các model
chính thức `PP-OCRv5_mobile_det` (phát hiện chữ trong crop, không phải detector
biển số) và `en_PP-OCRv4_mobile_rec`, lưu cache mặc định của PaddleX.
Lần đầu cần Internet. Có thể chỉ định Python worker bằng `ALPR_PADDLE_PYTHON`.
Lỗi dependency/model/timeout được ghi rõ; CNN/SVM/RF không phụ thuộc Paddle.

`ocr_backends.recognize_crop` trả cùng `OCRResult` cho bốn backend: raw text,
text chuẩn hóa, trạng thái, score nếu có, thời gian, tokens và thông tin lỗi.
Raw Paddle giữ nguyên dấu phân cách/xuống dòng; text chỉ bỏ separator, không
thay B/8, O/0. Score Paddle là min điểm các text token, **chưa hiệu chuẩn**.
Worker CPU khởi tạo lại model cho mỗi crop mới (~8–12 giây trên máy kiểm thử),
có cache theo nội dung cho crop lặp lại; không dành cho realtime.

```powershell
.\.venv\Scripts\python.exe compare_ocr.py
.\.venv\Scripts\python.exe verify_paddle.py
.\.venv\Scripts\python.exe -m streamlit run label_crops.py
```

Bảng so sánh, histogram lớp dự đoán, hash crop và thời gian nằm ở
`debug_outputs/ocr_comparison/results.json` và `results.csv`. Crop 0405 thiếu
biển và 0407 là đèn xe: loại chúng khỏi đánh giá chất lượng OCR, không dùng
kết quả trống để kết luận Paddle kém. Những ảnh còn lại cũng phải được duyệt
nhãn trước khi báo cáo accuracy. AppTest upload là kiểm thử phía server với
payload UploadedFile, không phải kiểm thử browser file picker.

### Duyệt nhãn và huấn luyện dữ liệu thật

`label_crops.py` hiển thị crop không kèm dự đoán gợi ý, yêu cầu nhập text,
source_group và split; lưu `dataset/reviewed_crops.csv` sau xác nhận bằng người.
Ảnh cùng xe/video phải cùng group/split. Chỉ lưu crop có đầy đủ biển số.

```powershell
.\.venv\Scripts\python.exe evaluate_labeled.py dataset/reviewed_crops.csv --crop-mode --model paddle --output paddle_evaluation.json
.\.venv\Scripts\python.exe collect_real_characters.py dataset/reviewed_crops.csv experiments/real_glyphs
```

Exporter chỉ sinh **ứng viên ký tự chưa nhãn** và `characters_to_review.csv`.
Không gán nhãn theo vị trí chuỗi biển vì segmentation có thể dư/thiếu. Xem
từng glyph, nhập `class_label` và đánh dấu `human_reviewed`; loại mảnh viền.
Sau khi đủ cả 36 lớp trong train và có validation/test độc lập theo nguồn:

```powershell
.\.venv\Scripts\python.exe train_real_characters.py experiments/real_glyphs/characters_to_review.csv experiments/real_run_01 --epochs 20
```

Trainer từ chối thư mục đã tồn tại/thư mục trong saved_models; kiểm tra group
leakage và lớp, dùng augmentation chỉ ở train, checkpoint/early stopping theo
validation. Chưa chạy huấn luyện dữ liệu thật vì chưa đủ nhãn. Cần so sánh model
mới với model gốc trên cùng tập reviewed test trước khi đưa vào sử dụng.
> **Automatic License Plate Recognition System**  
> Dự án ứng dụng **Thị giác Máy tính (Computer Vision)** và **Học máy / Học sâu (Machine Learning & Deep Learning)** để nhận diện biển số xe tự động theo quy trình đường ống khép kín (End-to-End Pipeline).

---

> **Lưu ý về kết quả:** các chỉ số 100%/96.94% nêu trong tài liệu phiên bản cũ không có artifact/báo cáo test độc lập đi kèm và chưa được xác minh; không xem đó là độ chính xác thực tế. Hãy xóa các con số đó khỏi báo cáo bảo vệ cho đến khi đo lại.

## Trạng thái và phạm vi đã xác minh

Pipeline hiện tại xử lý **một ảnh và một vùng biển số ứng viên** bằng OpenCV contour, sau đó phân đoạn contour và phân loại ký tự bằng CNN/SVM/Random Forest. Chưa có triển khai video, webcam, YOLO, OCR Paddle/EasyOCR hay đánh giá whole-plate có ground truth. Workspace có các file model, nhưng khả năng tải/suy luận cần xác minh trong môi trường Python đã cài dependencies.

##  1. TỔNG QUAN HỆ THỐNG

Hệ thống cho phép người dùng tải lên ảnh phương tiện giao thông (ô tô, xe máy) và tự động trích xuất chuỗi ký tự biển số xe với độ chính xác cao. 

###  Tính năng nổi bật
- **Định vị biển số tự động (Plate Localization)**: Ứng dụng các kỹ thuật xử lý ảnh nâng cao (Black-hat, Sobel-X, Otsu, Canny, phân tích Contour).
- **Phân tách ký tự thông minh (Character Segmentation)**: Tự động tách từng ký tự, lọc nhiễu ốc vít/viền, hỗ trợ cả **biển số 1 dòng và biển số 2 dòng**, chuẩn hóa ký tự về kích thước chuẩn **32x32 pixels**.
- **Đa dạng mô hình nhận diện (OCR Classification)**:
  - **SVM (Support Vector Machine)**: RBF Kernel kết hợp đặc trưng HOG.
  - **Random Forest**: 100 cây quyết định kết hợp đặc trưng HOG.
  - **CNN (Convolutional Neural Network)**: Mạng học sâu đa tầng Conv2D + BatchNorm + Dropout.
- **Hậu xử lý cú pháp biển số Việt Nam (`Domain Heuristics`)**: Tự động sửa các lỗi nhầm lẫn quang học kinh điển (`I` ↔ `1`, `O` ↔ `0`, `S` ↔ `5`, `B` ↔ `8`) dựa trên quy tắc biển số xe Việt Nam.
- **Giao diện Web trực quan (Streamlit GUI)**: Kéo thả ảnh dễ dàng, hiển thị minh bạch các bước xử lý trung gian.

---

##  2. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
may_hoc/
│
├── localization.py        # Module 1: Tiền xử lý & định vị khung biển số xe
├── segmentation.py        # Module 2: Nhị phân hóa Otsu & phân tách từng ký tự (32x32)
├── dataset.py             # Module 3: Trích xuất HOG, Data Augmentation & Cú pháp biển số VN
├── models.py              # Module 4: Xây dựng kiến trúc & huấn luyện SVM, RF, CNN
├── main.py                # Module 5: Đường ống End-to-End & đánh giá định lượng Train/Test
├── app.py                 # Module 6: Giao diện web ứng dụng Streamlit GUI
│
├── saved_models/          # Thư mục lưu trữ các mô hình đã huấn luyện
│   ├── svm_model.joblib   # Trọng số mô hình SVM
│   ├── rf_model.joblib    # Trọng số mô hình Random Forest
│   ├── cnn_model.keras    # Trọng số mô hình CNN
│   └── hog_scaler.joblib  # Bộ chuẩn hóa StandardScaler fit trên Train set
│
├── dataset/               # Dữ liệu hình ảnh biển số xe
└── README.md              # Tài liệu hướng dẫn sử dụng dự án
```

---

## 🔄 3. QUY TRÌNH HOẠT ĐỘNG (END-TO-END WORKFLOW)

```text
[ Ảnh xe gốc ]
      │
      ▼
┌──────────────────────────────┐
│       localization.py        │  ==> Cắt khung biển số xe (Plate ROI)
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│       segmentation.py        │  ==> Tách các ký tự chuẩn 32x32
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│          dataset.py          │  ==> Trích xuất vector đặc trưng HOG (9 bins, 8x8 cell)
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│   models.py (RF / SVM / CNN) │  ==> Dự đoán nhãn từng ký tự (0-9, A-Z)
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Hậu xử lý cú pháp VN (Rule) │  ==> Sửa nhầm lẫn quang học theo vị trí chuẩn
└──────────┬───────────────────┘
           │
           ▼
    KẾT QUẢ: "51A00530" (Hiển thị lên Streamlit Web)
```

---

## 📊 4. ĐÁNH GIÁ

Chưa có kết quả được xác minh trên dữ liệu biển số thực có nhãn độc lập. Dữ liệu ký tự tổng hợp có thể dùng để kiểm tra pipeline huấn luyện, nhưng không đại diện độ chính xác trên ảnh đường phố. Tập đánh giá đúng cần ground truth box và text, rồi đo precision/recall/IoU, character accuracy/CER, exact plate match và latency.

---

## 🚀 5. HƯỚNG DẪN CÀI ĐẶT & SỬ DỤNG

### **Yêu cầu môi trường**
- Python 3.10 hoặc 3.11
- Hệ điều hành: Windows, macOS, hoặc Linux

### **Bước 1: Cài đặt các thư viện cần thiết**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### **Bước 2: Huấn luyện mô hình ký tự tổng hợp (tùy chọn)**
Chạy lệnh sau để tự động sinh dữ liệu, huấn luyện 3 mô hình, lưu checkpoint vào thư mục `saved_models/` và in báo cáo kết quả chi tiết:
```powershell
python main.py
```

### **Bước 3: Khởi chạy Giao diện Web (Streamlit GUI)**
Khởi động ứng dụng web trực quan:
```powershell
python -m streamlit run app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.

---

## 🛠️ 6. CÔNG NGHỆ SỬ DỤNG
- **Computer Vision**: OpenCV (`cv2`), Scikit-Image (`skimage`).
- **Machine Learning**: Scikit-Learn (SVM, RandomForestClassifier, StandardScaler, Metrics).
- **Deep Learning**: TensorFlow / Keras (Sequential, Conv2D, BatchNormalization, Dropout).
- **Web App GUI**: Streamlit.
- **Model Checkpoint**: Joblib, HDF5/Keras.

### Giới hạn và phát triển tiếp

- Dùng TensorFlow CPU 2.16.2/Keras 3.15 cho tương thích với artifact CNN `.keras`; SVM/RF lưu từ scikit-learn 1.9.0. Không nên nâng riêng các gói này khi đang dùng artifact cũ.
- Không có trọng số detector YOLO đã xác minh. Chỉ tích hợp mô hình được huấn luyện trên bounding box biển số có nhãn, đánh giá trên cùng tập test với OpenCV; không dùng YOLO generic.
- Để tạo tập đánh giá, gắn nhãn box + text bằng người gán nhãn, kiểm tra chéo, chia tách theo nguồn/xe; tuyệt đối không lấy dự đoán mô hình làm nhãn thật.
- Video, webcam trình duyệt, multi-plate end-to-end, OCR hiện đại và UI chuyên biệt cho so sánh chưa được hoàn thành. Localization có proposal nhiều vùng nhưng pipeline vẫn chỉ phân loại vùng tốt nhất.
- Smoke test trên một số ảnh không nhãn cho thấy dự đoán ký tự còn sai rõ rệt; detector contour và classifier huấn luyện ký tự tổng hợp chưa đạt độ tin cậy thực tế. Không trình bày kết quả các ảnh đó như đánh giá accuracy.
- Lịch sử SQLite được lưu tại `data/recognitions.sqlite3`; ứng dụng chỉ ghi khi bấm lưu và không giữ ảnh tải lên.

### Chẩn đoán pipeline theo từng ảnh

```powershell
python debug_pipeline.py --images dataset/license_plate_dataset/0/images/0402.jpg dataset/license_plate_dataset/0/images/0403.jpg --output debug_outputs --model cnn
```

Mỗi ảnh sinh ảnh gốc, edge map, proposal box, crop từng proposal, crop baseline, crop hiệu chỉnh phối cảnh (nếu tìm được quadrilateral), binary, ký tự, overlay box và JSON/CSV prediction. Trạng thái `ocr_prediction_only` nghĩa là mô hình đưa ra chuỗi, không khẳng định chuỗi chính xác. Ảnh hiện có trong workspace **không có ground truth**, nên kết quả debug hiện tại không phải accuracy report. `dataset/evaluation_ground_truth_template.csv` cần người xác minh; hai ví dụ từ ảnh người dùng được đánh dấu `manual_estimate_from_user_image`, không được evaluator tính cho đến khi đổi thành `human_reviewed` sau khi xác minh.

Sau khi gắn nhãn bằng người:

```powershell
python evaluate_labeled.py dataset/evaluation_ground_truth_template.csv --model cnn --output evaluation_results.json
```

Evaluator bỏ qua các hàng chưa đánh dấu `human_reviewed`, ghi exact match/CER/thời gian cho hàng được duyệt. Đánh giá detection IoU/precision/recall cần box ground truth được xác minh riêng.

Evaluator ghi `null` khi chưa có mẫu được duyệt (không coi đó là accuracy/CER bằng 0).
CER toàn tập được tính bằng tổng edit distance chia tổng số ký tự nhãn; UNKNOWN
được tính như dự đoán rỗng và ảnh không đọc được vẫn là mẫu thất bại. Không tự
loại mẫu lỗi để làm đẹp chỉ số. Danh sách kết quả từng ảnh nằm trong `results`.

Debug JSON hiện chứa `production_pipeline`: box, chuỗi raw/post, confidence và
vùng ký tự thực sự dùng bởi `end_to_end_pipeline`. Ảnh `08_production_*` và thư
mục `production_characters` ghi đúng nhánh này. Các candidate rectification và
kết quả cấp trên được đánh dấu `experimental_geometry_not_production`; không
coi đó là kết quả của ứng dụng khi hai nhánh chọn crop khác nhau.

Chạy thử nghiệm hình học đầu vào mà **không thay trọng số/production preprocessing**:

```powershell
python diagnose_recognition.py
```

Kết quả `debug_outputs/geometry_diagnostic.json` chỉ gồm hai ví dụ phát triển.
Thử kéo giãn foreground thay vì square padding chưa nhận đúng hai biển nên
không được bật trong pipeline. Với ảnh 0403, 9 thành phần phân đoạn không có
nghĩa là segmentation đúng: chuỗi nhìn thấy `83FD9972` chỉ có 8 ký tự.

