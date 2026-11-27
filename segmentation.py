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

    if plate_img.ndim == 2:
        gray = plate_img
    elif plate_img.ndim == 3 and plate_img.shape[2] == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("plate_img must be grayscale or BGR")

    # Nâng cao tương phản cục bộ bằng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Làm mịn nhẹ để giảm nhiễu hạt
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Phân ngưỡng Otsu nghịch đảo: Ký tự thường có màu sẫm (đen) trên nền biển số sáng (trắng/vàng)
    # Sau THRESH_BINARY_INV, chữ số sẽ là điểm ảnh trắng (255) trên nền đen (0)
    _, otsu_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    adaptive_inv = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )

    # Select the threshold with more plausible foreground coverage; avoid a
    # fixed Otsu choice on uneven illumination. Both candidates stay local.
    def plausible(binary: np.ndarray) -> bool:
        ratio = float(np.mean(binary > 0))
        return 0.04 <= ratio <= 0.55

    thresh_inv = otsu_inv if plausible(otsu_inv) else adaptive_inv
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh_inv = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, kernel)

    # Kiểm tra tỷ lệ pixel trắng: Nếu > 55% thì có thể biển số có chữ trắng trên nền tối -> đảo lại
    white_ratio = np.sum(thresh_inv == 255) / float(thresh_inv.size)
    if white_ratio > 0.60:
        thresh_inv = cv2.bitwise_not(thresh_inv)

    return thresh_inv


def segment_characters(
    plate_img: np.ndarray,
    target_size: Tuple[int, int] = (32, 32),
    trace: Optional[dict] = None,
    binary_override: Optional[np.ndarray] = None,
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

    # Invalid preprocessing must be visible to caller/UI rather than silently
    # reported as an empty successful segmentation.
    binary_plate = preprocess_plate(plate_img) if binary_override is None else binary_override.copy()
    if binary_plate.ndim != 2 or binary_plate.dtype != np.uint8 or binary_plate.shape != plate_img.shape[:2]:
        raise ValueError('Expected uint8 binary image matching the crop dimensions')
    if trace is not None:
        trace.clear()
        trace.update({'contours': [], 'binary': binary_plate, 'final_boxes': []})

    plate_h, plate_w = binary_plate.shape[:2]
    contours, _ = cv2.findContours(binary_plate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_chars: List[Tuple[int, int, int, int, np.ndarray]] = []

    for contour_index, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / float(h)
        h_ratio = float(h) / float(plate_h)
        w_ratio = float(w) / float(plate_w)

        # Bộ lọc đặc trưng hình thái ký tự chuẩn:
        # - Ký tự không quá nhỏ và không chiếm toàn bộ biển số
        # - Chiều cao thường từ 25% đến 95% chiều cao biển (hoặc dòng)
        # - Tỷ lệ w/h của ký tự nằm trong khoảng 0.15 đến 1.1 (trừ chữ 1 rất mỏng hoặc W hơi rộng)
        reasons=[]
        if not (0.12 <= h_ratio <= 0.95):reasons.append('height_ratio_outside_0.12_0.95')
        if not (0.02 <= w_ratio <= 0.60):reasons.append('width_ratio_outside_0.02_0.60')
        if not (0.12 <= aspect_ratio <= 1.25):reasons.append('aspect_outside_0.12_1.25')
        edge_fragment=((x <= max(2, plate_w*.01) or x+w >= plate_w*.99)
                       and (h >= plate_h*.45 or w < h*.24))
        if edge_fragment:reasons.append('edge_fragment_filter')
        if trace is not None:
            trace['contours'].append({'index':contour_index,'bbox':[x,y,w,h],
                                      'area':float(cv2.contourArea(cnt)), 'aspect_ratio':aspect_ratio,
                                      'height_ratio':h_ratio,'width_ratio':w_ratio,
                                      'status':'rejected' if reasons else 'candidate_before_merge',
                                      'reasons':reasons})
        if not (0.12 <= h_ratio <= 0.95):
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

    # Plate frames and screws frequently touch the outer 2% of a crop. Reject
    # only tiny edge fragments (not full-height characters), then merge
    # vertically overlapping components that belong to one split glyph.
    valid_chars = [item for item in valid_chars
                   if not ((item[0] <= max(2, plate_w * 0.01) or item[0] + item[2] >= plate_w * 0.99)
                           and (item[3] >= plate_h * 0.45 or item[2] < item[3] * 0.24))]
    if not valid_chars:
        return []

    # Merge touching/overlapping sub-contours such as the split strokes of D,
    # F or 8. Do not merge adjacent normal glyphs: require strong x overlap.
    merged_chars: List[Tuple[int, int, int, int, np.ndarray]] = []
    for char in sorted(valid_chars, key=lambda item: item[0]):
        x, y, w, h, _ = char
        match = None
        for i, other in enumerate(merged_chars):
            ox, oy, ow, oh, _ = other
            x_intersection = max(0, min(x + w, ox + ow) - max(x, ox))
            y_intersection = max(0, min(y + h, oy + oh) - max(y, oy))
            x_overlap = x_intersection / max(1, min(w, ow))
            y_overlap = y_intersection / max(1, min(h, oh))
            if x_overlap >= 0.35 and y_overlap >= 0.35:
                match = i
                break
        if match is None:
            merged_chars.append(char)
        else:
            ox, oy, ow, oh, _ = merged_chars[match]
            nx, ny = min(x, ox), min(y, oy)
            nr, nb = max(x + w, ox + ow), max(y + h, oy + oh)
            roi = binary_plate[ny:nb, nx:nr]
            side = max(nr - nx, nb - ny)
            padded = np.zeros((side, side), dtype=np.uint8)
            padded[(side - (nb - ny)) // 2:(side - (nb - ny)) // 2 + nb - ny,
                   (side - (nr - nx)) // 2:(side - (nr - nx)) // 2 + nr - nx] = roi
            merged_chars[match] = (nx, ny, nr - nx, nb - ny,
                                   cv2.threshold(cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA),
                                                 127, 255, cv2.THRESH_BINARY)[1])
    valid_chars = merged_chars
    if not valid_chars:
        return []

    # Sắp xếp ký tự theo đúng chiều đọc (trên xuống dưới, trái sang phải nếu biển 2 dòng)
    # Xác định xem biển số có phải là 2 dòng hay không:
    # Nếu sự phân tách trục Y lớn hơn một ngưỡng (ví dụ > 35% chiều cao biển)
    centers = np.array([item[1] + item[3] / 2.0 for item in valid_chars])
    is_two_lines = float(np.percentile(centers, 90) - np.percentile(centers, 10)) > plate_h * 0.27
    if is_two_lines:
        # 1-D two-cluster split using the largest vertical gap between centers.
        ordered = sorted(valid_chars, key=lambda item: item[1] + item[3] / 2.0)
        gaps = [((b[1] + b[3] / 2) - (a[1] + a[3] / 2), i)
                for i, (a, b) in enumerate(zip(ordered, ordered[1:]))]
        largest_gap, split = max(gaps, default=(0.0, 0))
        if largest_gap > plate_h * 0.12:
            line1, line2 = ordered[:split + 1], ordered[split + 1:]
            sorted_chars = sorted(line1, key=lambda item: item[0]) + sorted(line2, key=lambda item: item[0])
        else:
            sorted_chars = sorted(valid_chars, key=lambda item: item[0])
    else:
        sorted_chars = sorted(valid_chars, key=lambda item: item[0])

    # Trả về kết quả: (char_img, (x, y, w, h))
    output = [(char[4], (char[0], char[1], char[2], char[3])) for char in sorted_chars]
    if trace is not None:
        trace['final_boxes']=[list(box) for _,box in output]
        trace['is_two_lines']=bool(is_two_lines)
        for entry in trace['contours']:
            if not entry['reasons']:
                matched=[i for i,(_,box) in enumerate(output)
                         if box[0]<=entry['bbox'][0] and box[1]<=entry['bbox'][1]
                         and box[0]+box[2]>=entry['bbox'][0]+entry['bbox'][2]
                         and box[1]+box[3]>=entry['bbox'][1]+entry['bbox'][3]]
                entry['final_indices']=matched
                entry['status']='kept_or_merged' if matched else 'filtered_after_candidate_stage'
    return output
