#  HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE TỰ ĐỘNG (ALPR PIPELINE)
> **Automatic License Plate Recognition System**  
> Dự án ứng dụng **Thị giác Máy tính (Computer Vision)** và **Học máy / Học sâu (Machine Learning & Deep Learning)** để nhận diện biển số xe tự động theo quy trình đường ống khép kín (End-to-End Pipeline).

---

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

## 📊 4. KẾT QUẢ ĐÁNH GIÁ ĐỊNH LƯỢNG TRÊN TẬP TEST (720 MẪU)

Toàn bộ quy trình đánh giá tuân thủ nghiêm ngặt nguyên tắc **chống rò rỉ dữ liệu (No Data Leakage)**: `StandardScaler` chỉ được `fit` trên tập Train set, tập Test set hoàn toàn độc lập.

| Mô hình | Đầu vào đặc trưng | Accuracy | Macro Precision | Macro Recall | Macro F1-score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (100 Trees)** | **HOG Features** | **100.0%** | **100.0%** | **100.0%** | **1.00** |
| **SVM (RBF Kernel)** | **HOG Features** | **100.0%** | **100.0%** | **100.0%** | **1.00** |
| **CNN (Deep Learning)** | **Ma trận Pixel $(32, 32, 1)$** | **96.94%** | **97.39%** | **96.94%** | **0.97** |

---

## 🚀 5. HƯỚNG DẪN CÀI ĐẶT & SỬ DỤNG

### **Yêu cầu môi trường**
- Python 3.10 hoặc 3.11
- Hệ điều hành: Windows, macOS, hoặc Linux

### **Bước 1: Cài đặt các thư viện cần thiết**
```powershell
pip install opencv-python numpy scikit-learn scikit-image tensorflow streamlit pillow joblib matplotlib
```

### **Bước 2: Huấn luyện mô hình và đánh giá hệ thống**
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

