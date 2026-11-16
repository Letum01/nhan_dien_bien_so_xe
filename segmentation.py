"""
Module 2: Character Segmentation
Author: ALPR Team
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np


def preprocess_plate(plate_img: np.ndarray) -> np.ndarray:
    """
    Tiền xử lý vùng ảnh biển số: chuyển xám, cân bằng sáng và nhị phân hóa Otsu nghịch đảo.

    Args:
        plate_img: Ảnh BGR của biển số.

    Returns:
        Ảnh nhị phân (chữ/số là màu trắng 255 trên nền đen 0).
    """
    if plate_img is None or plate_img.size == 0:
        raise ValueError("Ảnh biển số không hợp lệ (None).")

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Nâng cao tương phản cục bộ bằng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Làm mịn nhẹ để giảm nhiễu hạt
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Phân ngưỡng Otsu nghịch đảo: Ký tự thường có màu sẫm (đen) trên nền biển số sáng (trắng/vàng)
    # Sau THRESH_BINARY_INV, chữ số sẽ là điểm ảnh trắng (255) trên nền đen (0)
    _, thresh_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Kiểm tra tỷ lệ pixel trắng: Nếu > 55% thì có thể biển số có chữ trắng trên nền tối -> đảo lại
    white_ratio = np.sum(thresh_inv == 255) / float(thresh_inv.size)
    if white_ratio > 0.60:
        thresh_inv = cv2.bitwise_not(thresh_inv)

    return thresh_inv


def segment_characters(
    plate_img: np.ndarray,
    target_size: Tuple[int, int] = (32, 32)
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Phân tách từng ký tự (chữ và số) trong ảnh biển số xe.
    Lọc nhiễu theo kích thước và tỷ lệ aspect ratio,
    sau đó sắp xếp theo đúng thứ tự đọc (hỗ trợ cả biển 1 dòng và 2 dòng),
    cuối cùng đưa về kích thước chuẩn (32x32).

    Args:
        plate_img: Ảnh BGR của biển số.
        target_size: Kích thước đầu ra chuẩn hóa (width, height), mặc định (32, 32).

    Returns:
        Danh sách các tuple: (char_img_32x32, (x, y, w, h))
        với char_img_32x32 là ảnh nhị phân uint8 kích thước 32x32.
    """
    if plate_img is None or plate_img.size == 0:
        return []

    try:
        binary_plate = preprocess_plate(plate_img)
    except Exception:
        return []

    plate_h, plate_w = binary_plate.shape[:2]
    contours, _ = cv2.findContours(binary_plate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_chars: List[Tuple[int, int, int, int, np.ndarray]] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / float(h)
        h_ratio = float(h) / float(plate_h)
        w_ratio = float(w) / float(plate_w)

        # Bộ lọc đặc trưng hình thái ký tự chuẩn:
        # - Ký tự không quá nhỏ và không chiếm toàn bộ biển số
        # - Chiều cao thường từ 25% đến 95% chiều cao biển (hoặc dòng)
        # - Tỷ lệ w/h của ký tự nằm trong khoảng 0.15 đến 1.1 (trừ chữ 1 rất mỏng hoặc W hơi rộng)
        if not (0.20 <= h_ratio <= 0.95):
            continue
        if not (0.02 <= w_ratio <= 0.60):
            continue
        if not (0.12 <= aspect_ratio <= 1.25):
            continue

        # Trích xuất ROI của ký tự
        char_roi = binary_plate[y:y+h, x:x+w]

        # Padding vuông để giữ nguyên tỷ lệ ký tự trước khi resize về 32x32
        max_dim = max(w, h)
        padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
        offset_x = (max_dim - w) // 2
        offset_y = (max_dim - h) // 2
        padded[offset_y:offset_y+h, offset_x:offset_x+w] = char_roi

        resized_char = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
        # Nhị phân hóa lại sau khi resize
        _, resized_char = cv2.threshold(resized_char, 127, 255, cv2.THRESH_BINARY)

        valid_chars.append((x, y, w, h, resized_char))

    if not valid_chars:
        return []

    # Sắp xếp ký tự theo đúng chiều đọc (trên xuống dưới, trái sang phải nếu biển 2 dòng)
    # Xác định xem biển số có phải là 2 dòng hay không:
    # Nếu sự phân tách trục Y lớn hơn một ngưỡng (ví dụ > 35% chiều cao biển)
    y_coords = [item[1] for item in valid_chars]
    min_y, max_y = min(y_coords), max(y_coords)
    is_two_lines = (max_y - min_y) > (plate_h * 0.25)

    if is_two_lines:
        # Phân chia 2 dòng dựa trên ngưỡng y trung bình
        median_y = np.median(y_coords)
        line1 = [item for item in valid_chars if item[1] < median_y]
        line2 = [item for item in valid_chars if item[1] >= median_y]

        line1.sort(key=lambda item: item[0])  # Sắp xếp từ trái sang phải
        line2.sort(key=lambda item: item[0])
        sorted_chars = line1 + line2
    else:
        # Biển 1 dòng: Chỉ sắp xếp từ trái qua phải theo x
        sorted_chars = sorted(valid_chars, key=lambda item: item[0])

    # Trả về kết quả: (char_img, (x, y, w, h))
    return [(char[4], (char[0], char[1], char[2], char[3])) for char in sorted_chars]
