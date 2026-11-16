"""
Module 1: Preprocessing & Plate Localization
Author: ALPR Team
"""

from typing import Optional, Tuple, List
import cv2
import numpy as np


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Tiền xử lý ảnh xe: Chuyển sang ảnh xám, làm mờ Gaussian để lọc nhiễu,
    áp dụng phép toán hình thái học (Black-hat) và phát hiện biên Canny.

    Args:
        image: Ảnh BGR đầu vào dạng numpy array.

    Returns:
        Ảnh nhị phân biên (edge map) dạng uint8.
    """
    if image is None:
        raise ValueError("Ảnh đầu vào không hợp lệ (None).")

    # 1. Chuyển ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Lọc nhiễu Gaussian
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Tăng cường độ tương phản cục bộ bằng Black-hat hoặc Sobel
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, rect_kernel)

    # Gradient theo trục X để làm nổi bật các vạch biên dọc của biển số
    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)
    min_val, max_val = np.min(grad_x), np.max(grad_x)
    if max_val > min_val:
        grad_x = (255 * ((grad_x - min_val) / (max_val - min_val))).astype(np.uint8)
    else:
        grad_x = np.zeros(blackhat.shape, dtype=np.uint8)

    # 4. Làm mịn và đóng các khoảng trống biên
    grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 7))
    morph = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, close_kernel)

    # 5. Phân ngưỡng Otsu và toán tử hình thái học để tạo vùng liên thông
    _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # 6. Phát hiện biên Canny kết hợp
    canny = cv2.Canny(blurred, 50, 200)
    edges = cv2.bitwise_or(thresh, canny)

    return edges


def locate_license_plate(
    image: np.ndarray,
    min_aspect_ratio: float = 1.2,
    max_aspect_ratio: float = 6.5,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.35
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Định vị và trích xuất vùng chứa biển số xe dựa trên phân tích hình thái contours.

    Args:
        image: Ảnh BGR đầu vào.
        min_aspect_ratio: Tỷ lệ w/h tối thiểu của biển số (chuẩn 1 dòng hoặc 2 dòng).
        max_aspect_ratio: Tỷ lệ w/h tối đa của biển số dài.
        min_area_ratio: Tỷ lệ diện tích tối thiểu so với toàn bộ ảnh xe.
        max_area_ratio: Tỷ lệ diện tích tối đa so với toàn bộ ảnh xe.

    Returns:
        (plate_crop, bbox): 
            plate_crop: Vùng ảnh cắt của biển số (hoặc None nếu không tìm thấy).
            bbox: Tuple (x, y, w, h) tương ứng với vị trí biển số trên ảnh gốc.
    """
    if image is None or image.size == 0:
        return None, None

    img_h, img_w = image.shape[:2]
    img_area = float(img_h * img_w)

    try:
        edges = preprocess_image(image)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except Exception:
        return None, None

    # Lọc và chấm điểm các contour tiềm năng
    candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = float(w * h)
        area_ratio = area / img_area
        aspect_ratio = float(w) / float(h)

        if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
            continue
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue

        # Điểm ưu tiên: Biển số thường ở nửa dưới hoặc trung tâm xe, tỷ lệ gần 1.3 - 4.5
        center_y = y + h / 2.0
        y_score = center_y / img_h  # Nằm dưới có điểm cao hơn
        rect_coverage = cv2.contourArea(cnt) / (area + 1e-5)
        
        # Biển số chuẩn có hình chữ nhật rõ ràng
        score = rect_coverage * 0.4 + y_score * 0.4 + (area_ratio * 10) * 0.2
        candidates.append((score, (x, y, w, h)))

    if not candidates:
        # Phương án dự phòng: Nếu không tìm thấy contour chuẩn, thử lấy vùng nghi ngờ trung tâm
        return None, None

    # Lấy ứng viên có điểm số cao nhất
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_x, best_y, best_w, best_h = candidates[0][1]

    # Thêm viền đệm nhỏ (padding) an toàn
    pad_x = int(best_w * 0.03)
    pad_y = int(best_h * 0.05)
    x1 = max(0, best_x - pad_x)
    y1 = max(0, best_y - pad_y)
    x2 = min(img_w, best_x + best_w + pad_x)
    y2 = min(img_h, best_y + best_h + pad_y)

    plate_crop = image[y1:y2, x1:x2].copy()
    return plate_crop, (x1, y1, x2 - x1, y2 - y1)
