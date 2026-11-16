"""
Module 3: Feature Extraction & Data Augmentation
Author: ALPR Team
"""

import os
import string
from typing import Tuple, List, Optional
import cv2
import numpy as np
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras import layers, models

# Bộ ký tự chuẩn (36 lớp: 10 chữ số + 26 chữ cái viết hoa)
CHARACTER_CLASSES: List[str] = list(string.digits) + list(string.ascii_uppercase)
CLASS_TO_IDX = {ch: idx for idx, ch in enumerate(CHARACTER_CLASSES)}
IDX_TO_CLASS = {idx: ch for idx, ch in enumerate(CHARACTER_CLASSES)}


def extract_hog_features(
    image_32x32: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2)
) -> np.ndarray:
    """
    Trích xuất đặc trưng HOG (Histogram of Oriented Gradients) từ ảnh ký tự 32x32.

    Args:
        image_32x32: Ảnh ma trận xám hoặc nhị phân kích thước (32, 32).
        orientations: Số lượng hướng gradient bin (mặc định 9).
        pixels_per_cell: Kích thước mỗi cell (mặc định 8x8).
        cells_per_block: Số cell trong mỗi block chuẩn hóa (mặc định 2x2).

    Returns:
        Vector đặc trưng 1 chiều dạng float32.
    """
    if image_32x32.shape != (32, 32):
        image_32x32 = cv2.resize(image_32x32, (32, 32), interpolation=cv2.INTER_AREA)

    # Đảm bảo kiểu float hoặc uint8 hợp lệ cho skimage
    if image_32x32.dtype != np.uint8 and image_32x32.dtype != np.float32:
        image_32x32 = image_32x32.astype(np.uint8)

    features = hog(
        image_32x32,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        transform_sqrt=True,
        visualize=False
    )
    return features.astype(np.float32)

def post_process_plate_text(raw_text: str) -> str:
    """
    Áp dụng quy tắc cú pháp biển số xe Việt Nam (Domain Heuristics):
    - 2 ký tự đầu là MÃ TỈNH (bắt buộc là SỐ: ví dụ 51, 29, 30).
    - Ký tự thứ 3 là SÊ-RI (bắt buộc là CHỮ: ví dụ A, B, C, F, G...).
    - Các ký tự còn lại là SỐ THỨ TỰ (bắt buộc là SỐ: ví dụ 00530).
    Tự động hiệu chỉnh các cặp ký tự dễ nhầm lẫn quang học (I <-> 1, O <-> 0, B <-> 8, S <-> 5).
    """
    if not raw_text or len(raw_text) < 6:
        return raw_text

    char_to_digit = {
        'I': '1', 'L': '1', 'J': '1', 'T': '1',
        'O': '0', 'D': '0', 'Q': '0', 'U': '0',
        'B': '8', 'S': '5', 'Z': '2', 'G': '6', 'A': '4'
    }
    digit_to_char = {
        '1': 'I', '0': 'O', '8': 'B', '5': 'S', '2': 'Z', '6': 'G', '4': 'A'
    }

    chars = list(raw_text)
    n = len(chars)

    # Nếu biển số có 7 đến 9 ký tự (dạng chuẩn của xe ô tô và mô tô Việt Nam)
    if 7 <= n <= 9:
        # Vị trí 0 và 1: Bắt buộc là số
        for idx in [0, 1]:
            if chars[idx] in char_to_digit:
                chars[idx] = char_to_digit[chars[idx]]

        # Vị trí 2: Bắt buộc là chữ cái
        if chars[2] in digit_to_char:
            chars[2] = digit_to_char[chars[2]]

        # Các vị trí từ 3 trở đi (dãy số cuối): Bắt buộc là số
        for idx in range(3, n):
            if chars[idx] in char_to_digit:
                chars[idx] = char_to_digit[chars[idx]]

    return "".join(chars)



def get_data_augmentation_pipeline() -> tf.keras.Sequential:
    """
    Khởi tạo luồng tăng cường dữ liệu (Data Augmentation) cho tập huấn luyện CNN
    sử dụng Keras Preprocessing Layers: xoay nhẹ, dịch chuyển, thay đổi độ tương phản.

    Returns:
        Keras Sequential pipeline để áp dụng tăng cường dữ liệu.
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.08, fill_mode='constant', fill_value=0.0),
        layers.RandomTranslation(height_factor=0.08, width_factor=0.08, fill_mode='constant', fill_value=0.0),
        layers.RandomZoom(height_factor=(-0.08, 0.08), fill_mode='constant', fill_value=0.0),
        layers.RandomContrast(factor=0.15),
    ], name="character_data_augmentation")

    return data_augmentation


def generate_synthetic_character_dataset(
    samples_per_class: int = 120
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo bộ dữ liệu ký tự tổng hợp (Synthetic Dataset) chuẩn xác với nhiều kiểu font,
    độ nghiêng, độ mờ để đảm bảo mô hình có thể huấn luyện và đánh giá độc lập
    ngay cả khi chưa gắn nhãn tay từng ký tự.

    Args:
        samples_per_class: Số lượng mẫu trên mỗi ký tự (0-9, A-Z).

    Returns:
        X: Mảng numpy shape (N, 32, 32, 1) giá trị [0, 255] uint8.
        y: Mảng nhãn nhị phân / số nguyên (N,) dạng int32.
    """
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
    ]

    images: List[np.ndarray] = []
    labels: List[int] = []

    np.random.seed(42)

    for class_idx, char in enumerate(CHARACTER_CLASSES):
        for _ in range(samples_per_class):
            canvas = np.zeros((64, 64), dtype=np.uint8)
            font = fonts[np.random.randint(0, len(fonts))]
            scale = np.random.uniform(1.2, 1.7)
            thickness = np.random.randint(2, 4)

            # Tính kích thước chữ để căn giữa
            (text_w, text_h), baseline = cv2.getTextSize(char, font, scale, thickness)
            org_x = max(0, (64 - text_w) // 2 + np.random.randint(-3, 4))
            org_y = max(text_h, (64 + text_h) // 2 + np.random.randint(-3, 4))

            cv2.putText(canvas, char, (org_x, org_y), font, scale, 255, thickness, cv2.LINE_AA)

            # Phép biến đổi Affine nhẹ (xoay, kéo nghiêng)
            angle = np.random.uniform(-10.0, 10.0)
            M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
            transformed = cv2.warpAffine(canvas, M, (64, 64), flags=cv2.INTER_LINEAR, borderValue=0)

            # Cắt bounding box và đưa về 32x32
            pts = cv2.findNonZero(transformed)
            if pts is not None:
                bx, by, bw, bh = cv2.boundingRect(pts)
                roi = transformed[by:by+bh, bx:bx+bw]
                dim = max(bw, bh)
                sq = np.zeros((dim, dim), dtype=np.uint8)
                sq[(dim - bh) // 2:(dim - bh) // 2 + bh, (dim - bw) // 2:(dim - bw) // 2 + bw] = roi
                char_img = cv2.resize(sq, (32, 32), interpolation=cv2.INTER_AREA)
            else:
                char_img = cv2.resize(transformed, (32, 32), interpolation=cv2.INTER_AREA)

            # Thêm nhiễu Gaussian nhẹ ngẫu nhiên
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 15, (32, 32)).astype(np.int16)
                noisy = np.clip(char_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                char_img = noisy

            _, char_img = cv2.threshold(char_img, 60, 255, cv2.THRESH_BINARY)
            images.append(char_img)
            labels.append(class_idx)

    X = np.array(images, dtype=np.uint8)[..., np.newaxis]
    y = np.array(labels, dtype=np.int32)
    return X, y
