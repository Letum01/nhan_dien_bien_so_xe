"""
Module 6: Streamlit GUI for ALPR
Author: ALPR Team
"""

import os
from typing import Optional, Dict, Any
import streamlit as st
import cv2
import numpy as np
from PIL import Image

from main import end_to_end_pipeline, load_trained_models
from localization import locate_license_plate
from segmentation import segment_characters

st.set_page_config(
    page_title="Hệ Thống Nhận Diện Biển Số Xe (ALPR)",
    page_icon="",
    layout="wide"
)

st.title(" Hệ Thống Nhận Diện Biển Số Xe Tự Động (ALPR Pipeline)")
st.markdown(
    """
    Preprocessing & Localization -> Character Segmentation -> Classification (CNN / SVM / RF)**.
    """
)


@st.cache_resource
def get_models() -> Optional[Dict[str, Any]]:
    """Tải và lưu vào bộ nhớ cache các mô hình."""
    return load_trained_models("saved_models")


models_dict = get_models()

st.sidebar.header(" Cấu hình Nhận diện")
model_choice = st.sidebar.selectbox(
    "Chọn mô hình phân loại ký tự:",
    options=["Random Forest (HOG + Ensembles)", "SVM (HOG + Support Vector)", "CNN (Deep Learning)"],
    index=0
)

model_type_map = {
    "CNN (Deep Learning)": "cnn",
    "SVM (HOG + Support Vector)": "svm",
    "Random Forest (HOG + Ensembles)": "rf"
}
selected_model_type = model_type_map[model_choice]

if models_dict is None:
    st.sidebar.error(" Chưa tìm thấy mô hình trong thư mục `saved_models`!")
    st.sidebar.info("Vui lòng chạy `python main.py` trong terminal để huấn luyện và lưu mô hình.")
else:
    st.sidebar.success(" Đã nạp thành công 3 mô hình (CNN, SVM, Random Forest).")

uploaded_file = st.sidebar.file_uploader(
    "Chọn ảnh xe hoặc biển số cần nhận diện:",
    type=["jpg", "jpeg", "png"]
)

tab1, tab2 = st.tabs([" Nhận diện Trực quan", " Chi tiết Kỹ thuật"])

with tab1:
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Ảnh xe gốc tải lên")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        with st.spinner("Đang xử lý toàn bộ luồng ALPR..."):
            plate_text, annotated, plate_crop, char_images = end_to_end_pipeline(
                image_bgr,
                model_type=selected_model_type,
                models_dict=models_dict
            )

        with col2:
            st.subheader("2. Kết quả nhận diện trên xe")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.success(f"###  Biển số nhận diện được: `{plate_text}`")

        st.markdown("---")
        st.subheader("3. Các bước xử lý trung gian (Intermediate Steps)")
        step_col1, step_col2 = st.columns([1, 2])

        with step_col1:
            st.markdown("**Vùng ảnh cắt biển số (Plate Localization):**")
            if plate_crop is not None:
                st.image(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB), width=280)
            else:
                st.warning("Không tách được khung biển số.")

        with step_col2:
            st.markdown(f"**Các ký tự phân tách được ({len(char_images)} ký tự chuẩn 32x32):**")
            if char_images:
                char_cols = st.columns(min(len(char_images), 10))
                for i, char_img in enumerate(char_images):
                    col_idx = i % 10
                    with char_cols[col_idx]:
                        st.image(char_img, caption=f"#{i+1}", width=48)
            else:
                st.info("Chưa có ký tự nào được phân tách.")

    else:
        st.info(" Hãy tải lên một ảnh phương tiện giao thông từ thanh bên trái để bắt đầu nhận diện.")

with tab2:
    st.markdown(
        """
        ### Kiến trúc Hệ thống ALPR:
        - **Module 1: Preprocessing & Plate Localization**:
          - `cv2.cvtColor`, `cv2.GaussianBlur`, toán tử hình thái học `cv2.morphologyEx` (Black-hat, Sobel, Close).
          - Lọc các ứng viên bằng `cv2.findContours` dựa trên aspect ratio và area ratio.
        - **Module 2: Character Segmentation**:
          - Phân ngưỡng Otsu nghịch đảo `cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV`.
          - Lọc contour ký tự theo kích thước, sắp xếp đọc từ trái sang phải / trên xuống dưới, padding và resize chuẩn **32x32**.
        - **Module 3: Feature Extraction & Data Augmentation**:
          - Trích xuất HOG: `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`.
          - Data Augmentation cho CNN: Xoay, dịch chuyển, zoom, thay đổi tương phản.
        - **Module 4: Models Architecture**:
          - **SVM**: Kernel RBF với HOG Features.
          - **Random Forest**: 100 Trees với HOG Features.
          - **CNN**: 3 Khối Conv2D + BatchNorm + MaxPool + Dropout + Dense (36 classes).
        - **Module 5: Strict Evaluation**:
          - Đánh giá trên Test set: Accuracy, Precision, Recall, Macro F1-score, Confusion Matrix.
        """
    )
