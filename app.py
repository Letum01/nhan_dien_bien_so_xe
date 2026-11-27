"""
Module 6: Streamlit GUI for ALPR
Author: ALPR Team
"""

import logging
import os
from pathlib import Path
import time
import pandas as pd
from typing import Optional, Dict, Any
import streamlit as st
import cv2
import numpy as np
from PIL import Image

from main import end_to_end_pipeline, load_trained_models
from localization import locate_license_plate
from segmentation import segment_characters
from database import RecognitionDatabase
from image_io import decode_uploaded_image
from ocr_backends import recognize_crop

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Hệ Thống Nhận Diện Biển Số Xe (ALPR)",
    page_icon="",
    layout="wide"
)

@st.cache_resource
def get_database() -> RecognitionDatabase:
    return RecognitionDatabase()

database = get_database()


@st.cache_data(show_spinner=False)
def decode_image_for_display(data: bytes) -> Optional[np.ndarray]:
    """Decode an uploaded image once per content for this Streamlit session."""
    return decode_uploaded_image(data)

st.title("Hệ thống nhận diện biển số xe Việt Nam")
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
model_type_map = {
    "CNN (Deep Learning)": "cnn",
    "SVM (HOG + Support Vector)": "svm",
    "Random Forest (HOG + Ensembles)": "rf",
    "Ensemble (Majority — thử nghiệm)": "ensemble",
}
available_choices = [label for label, key in model_type_map.items() if models_dict and models_dict.get(key) is not None]
if models_dict and all(models_dict.get(key) is not None for key in ("svm", "rf", "cnn", "scaler")):
    available_choices.append("Ensemble (Majority — thử nghiệm)")
if not available_choices:
    st.sidebar.warning("Không có mô hình nhận dạng khả dụng; thêm trọng số đã huấn luyện vào saved_models.")
    available_choices = list(model_type_map)
model_choice = st.sidebar.selectbox("Chọn mô hình phân loại ký tự:", options=available_choices)
selected_model_type = model_type_map[model_choice]
detector_options = ["OpenCV"]
from yolo_ui_bridge import available as round2_available, CHECKPOINT as ROUND2_CHECKPOINT
if round2_available():
    detector_options.append("YOLO vòng 2 — thử nghiệm")
yolo_weights = os.environ.get("ALPR_YOLO_WEIGHTS", "")
if yolo_weights and Path(yolo_weights).is_file():
    from importlib.util import find_spec
    if find_spec("ultralytics") is not None:
        detector_options.append("YOLO")
detector_choice = st.sidebar.selectbox("Detector", detector_options)
detector_backend=detector_choice.lower()
if detector_choice=="YOLO vòng 2 — thử nghiệm":
    detector_backend="yolo_isolated"
    yolo_weights=str(ROUND2_CHECKPOINT)
    st.sidebar.caption("Checkpoint split B; confidence cố định 0.4. Chạy venv YOLO riêng; chưa là detector mặc định.")
if len(detector_options) == 1:
    st.sidebar.caption("YOLO chưa khả dụng: cần ultralytics và ALPR_YOLO_WEIGHTS trỏ tới checkpoint biển số đã xác minh.")
if selected_model_type == "ensemble":
    st.sidebar.warning("Majority là baseline thử nghiệm, chưa chứng minh tốt hơn model đơn. 3 phiếu khác nhau → UNKNOWN. Đồng thuận không đồng nghĩa chính xác.")

if models_dict is None:
    st.sidebar.error(" Chưa tìm thấy mô hình trong thư mục `saved_models`!")
    st.sidebar.info("Cung cấp model hợp lệ hoặc dùng train_real_characters.py với dữ liệu thật đã duyệt; không ghi đè model gốc.")
else:
    st.sidebar.success("Đã nạp: " + ", ".join(key.upper() for key in ("cnn", "svm", "rf") if models_dict.get(key) is not None))

uploaded_file = st.sidebar.file_uploader(
    "Chọn ảnh xe hoặc biển số cần nhận diện:",
    type=["jpg", "jpeg", "png"]
)

tab1, tab2, tab3 = st.tabs(["Nhận diện ảnh", "Lịch sử", "Thông tin mô hình"])

with tab1:
    if uploaded_file is not None:
        image_bgr = decode_image_for_display(uploaded_file.getvalue())

        if image_bgr is None:
            st.error("Không đọc được ảnh. Hãy tải lên tệp JPG hoặc PNG hợp lệ.")
            st.stop()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Ảnh xe gốc tải lên")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

        started = time.perf_counter()
        with st.spinner("Đang xử lý ảnh..."):
            try:
                recognition_trace = {}
                plate_text, annotated, plate_crop, char_images = end_to_end_pipeline(
                    image_bgr, model_type=selected_model_type, models_dict=models_dict, trace=recognition_trace,
                    detector_type=detector_backend, detector_weights=yolo_weights or None,
                )
            except (ValueError, KeyError, cv2.error) as exc:
                logger.exception("ALPR pipeline rejected uploaded image")
                st.error(f"Không thể nhận diện với cấu hình hiện tại: {exc}")
                st.stop()
            except Exception as exc:
                logger.exception("Unexpected ALPR inference failure")
                st.error(f"Lỗi không mong đợi khi nhận diện ({type(exc).__name__}). Xem log ứng dụng để biết chi tiết.")
                st.stop()
        elapsed = time.perf_counter() - started

        with col2:
            st.subheader("2. Kết quả nhận diện trên xe")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.success(f"Biển số: `{plate_text}`")
            st.caption(f"OCR gốc: {recognition_trace.get('raw_text', '')}")
            st.caption(f"Thời gian xử lý: {elapsed:.3f} giây")
            if recognition_trace.get('detections'):
                st.write('YOLO detections (xywh / model confidence, không phải OCR confidence):',recognition_trace['detections'])
            if st.button("Lưu kết quả vào lịch sử"):
                status = "recognized" if not plate_text.startswith("UNKNOWN") else "not_found"
                database.add(recognition_trace.get("raw_text", ""), plate_text, selected_model_type,
                             uploaded_file.name, status=status,
                             confidence=recognition_trace.get("ocr_result", {}).get("confidence"))
                st.success("Đã lưu lịch sử (không lưu ảnh tải lên).")

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

        if plate_crop is not None and st.button("So sánh SVM / RF / CNN / Ensemble trên cùng crop"):
            rows = []
            segments = recognition_trace.get("segments", [])
            for backend in ("svm", "rf", "cnn", "ensemble"):
                comparison = recognize_crop(plate_crop, backend, models_dict, segments)
                rows.append({"model": backend, "raw": comparison.raw_text, "text": comparison.text,
                             "status": comparison.status, "seconds": comparison.seconds,
                             "error": comparison.error})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if st.button('OCR Diagnostics — YOLO vòng 2 (lưu artifact)'):
            from ocr_diagnostics import render
            with st.spinner('Đang tạo diagnostics baseline và biến thể; không sửa model...'):
                try:render(image_bgr,models_dict)
                except Exception as exc:
                    logger.exception('OCR Diagnostics failed')
                    st.error(f'Diagnostics lỗi: {type(exc).__name__}: {exc}')

    else:
        st.info(" Hãy tải lên một ảnh phương tiện giao thông từ thanh bên trái để bắt đầu nhận diện.")

with tab3:
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
    st.caption("Sản phẩm: OpenCV/YOLO → segmentation → SVM/RF/CNN/Ensemble. Không sử dụng OCR bên ngoài. YOLO cần checkpoint biển số đã xác minh. Weighted/soft voting chưa bật khi chưa có validation/calibration.")

with tab2:
    query = st.text_input("Tìm theo biển số")
    records = database.search(query)
    if records:
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
        st.download_button("Xuất CSV", pd.DataFrame(records).to_csv(index=False).encode("utf-8-sig"), "alpr_history.csv", "text/csv")
    else:
        st.info("Chưa có lịch sử nhận diện.")
    if st.button("Xóa toàn bộ lịch sử", type="secondary"):
        database.clear()
        st.rerun()
