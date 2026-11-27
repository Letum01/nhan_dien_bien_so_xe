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

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Ảnh phải là ảnh grayscale hoặc BGR 3 kênh")

    # 1. Chuyển ảnh xám
    # 2. Lọc nhiễu và cân bằng tương phản để giảm ảnh thiếu sáng
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
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
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    except cv2.error:
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

        # Penalize degenerate contours; do not rank solely by lower image position
        # or area, both of which favor headlights/body panels in crowded scenes.
        center_y = y + h / 2.0
        rect_coverage = cv2.contourArea(cnt) / (area + 1e-5)
        aspect_score = float(np.exp(-abs(np.log(max(aspect_ratio, 1e-5) / 3.0))))
        size_score = min(area_ratio / 0.025, 1.0)
        edge_density = float(np.mean(edges[y:y + h, x:x + w] > 0))
        score = (0.40 * aspect_score + 0.25 * min(rect_coverage, 1.0)
                 + 0.20 * min(edge_density * 2.0, 1.0) + 0.15 * size_score)
        candidates.append((score, (x, y, w, h)))

    if not candidates:
        # Phương án dự phòng: Nếu không tìm thấy contour chuẩn, thử lấy vùng nghi ngờ trung tâm
        return None, None

    # Contours are often nested; merge highly-overlapping boxes, retaining the
    # union instead of allowing a tiny inner contour to win by its shape.
    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
    merged: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for score, box in candidates:
        x, y, w, h = box
        duplicate_index = None
        for i, (_, other) in enumerate(merged):
            ox, oy, ow, oh = other
            ix1, iy1 = max(x, ox), max(y, oy)
            ix2, iy2 = min(x + w, ox + ow), min(y + h, oy + oh)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = w * h + ow * oh - intersection
            if union > 0 and intersection / union >= 0.45:
                duplicate_index = i
                break
        if duplicate_index is None:
            merged.append((score, box))
        else:
            old_score, old = merged[duplicate_index]
            ax, ay, aw, ah = old
            nx1, ny1 = min(x, ax), min(y, ay)
            nx2, ny2 = max(x + w, ax + aw), max(y + h, ay + ah)
            merged[duplicate_index] = (max(score, old_score), (nx1, ny1, nx2 - nx1, ny2 - ny1))
    candidates = sorted(merged, key=lambda item: item[0], reverse=True)
    best_score, (best_x, best_y, best_w, best_h) = candidates[0]
    if best_score < 0.30:
        return None, None

    # Thêm viền đệm nhỏ (padding) an toàn
    pad_x = int(best_w * 0.03)
    pad_y = int(best_h * 0.05)
    x1 = max(0, best_x - pad_x)
    y1 = max(0, best_y - pad_y)
    x2 = min(img_w, best_x + best_w + pad_x)
    y2 = min(img_h, best_y + best_h + pad_y)

    plate_crop = image[y1:y2, x1:x2].copy()
    return plate_crop, (x1, y1, x2 - x1, y2 - y1)


def locate_license_plates(image: np.ndarray, max_plates: int = 10) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
    """Return distinct plate proposals as (crop, xywh, heuristic score).

    This is an OpenCV proposal API; scores are not calibrated probabilities.
    The legacy single-result API remains unchanged.
    """
    if image is None or image.size == 0 or max_plates <= 0:
        return []
    height, width = image.shape[:2]
    area = float(height * width)
    try:
        edge_map = preprocess_image(image)
        contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    except cv2.error:
        return []
    proposals: List[Tuple[np.ndarray, Tuple[int, int, int, int], float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        ratio, area_ratio = w / max(float(h), 1.0), (w * h) / area
        # Reject small vehicle details and edge fragments before segmentation.
        # At these image dimensions, plate text is not recoverable from tiny
        # 10-40px-wide contours (the failure mode seen in 0402/0407 debug crops).
        if not (1.25 <= ratio <= 6.5 and 0.003 <= area_ratio <= 0.35 and w >= 55 and h >= 18):
            continue
        coverage = cv2.contourArea(contour) / max(float(w * h), 1.0)
        score = 0.45 * np.exp(-abs(np.log(max(ratio, 1e-5) / 3.0))) + 0.30 * min(coverage, 1) + 0.25 * min(area_ratio / 0.025, 1)
        padding_x, padding_y = int(w * 0.03), int(h * 0.05)
        x1, y1 = max(0, x - padding_x), max(0, y - padding_y)
        x2, y2 = min(width, x + w + padding_x), min(height, y + h + padding_y)
        box = (x1, y1, x2 - x1, y2 - y1)
        if any(_box_iou(box, existing[1]) > 0.45 for existing in proposals):
            continue
        proposals.append((image[y1:y2, x1:x2].copy(), box, float(score)))

    # Add closed plate-frame proposals independently of text-gradient contours.
    # Evidence: 0405 has a complete frame but the gradient branch joins the
    # taillight and only the TOP half of the plate. Preserve existing branch.
    proposals.extend(plate_border_proposals(image))

    # For two-row motorcycle plates, contour closure often yields one contour
    # per row. Merge vertically adjacent, horizontally aligned row proposals.
    row_proposals = list(proposals)
    for first_idx, first in enumerate(row_proposals):
        _, (x1, y1, w1, h1), score1 = first
        for second in row_proposals[first_idx + 1:]:
            _, (x2, y2, w2, h2), score2 = second
            if abs(y1 - y2) < min(h1, h2) * 0.65:
                continue
            upper_y, lower_y = (y1, y2) if y1 < y2 else (y2, y1)
            upper_h = h1 if y1 < y2 else h2
            lower_h = h2 if y1 < y2 else h1
            horizontal_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_ratio = horizontal_overlap / max(1, min(w1, w2))
            gap = lower_y - (upper_y + upper_h)
            if overlap_ratio < 0.55 or gap < -min(h1, h2) * 0.45 or gap > max(h1, h2) * 1.2:
                continue
            left, top = min(x1, x2), min(y1, y2)
            right, bottom = max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2)
            box = (left, top, right - left, bottom - top)
            bw, bh = box[2:]
            if bw / max(bh, 1) < 1.05:
                continue
            crop = image[top:bottom, left:right].copy()
            score = float(min(1.0, (score1 + score2) / 2 + 0.12 * overlap_ratio))
            if not any(_box_iou(box, existing[1]) > 0.85 for existing in proposals):
                proposals.append((crop, box, score))
    proposals.sort(key=lambda item: item[2], reverse=True)
    distinct: List[Tuple[np.ndarray, Tuple[int, int, int, int], float]] = []
    for proposal in proposals:
        if any(_box_iou(proposal[1], existing[1]) > 0.55 for existing in distinct):
            continue
        distinct.append(proposal)
        if len(distinct) >= max_plates:
            break
    return distinct


def plate_border_proposals(image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
    """Closed convex plate-sized frames, validated by interior character layout.

    Scores are heuristic, not probabilities. This conservative supplementary
    branch may miss broken frames; it does not fabricate a plate on 0407.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    found = []
    from segmentation import segment_characters
    for contour in contours:
        polygon = cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        x, y, w, h = cv2.boundingRect(polygon)
        area = cv2.contourArea(polygon)
        if not (.01 <= area / (height * width) <= .2 and 1.15 <= w / h <= 6.5):
            continue
        # Small outward padding includes the physical frame without cutting ink.
        pad = max(2, int(min(w, h) * .03))
        left, top = max(0, x-pad), max(0, y-pad)
        right, bottom = min(width, x+w+pad), min(height, y+h+pad)
        crop = image[top:bottom, left:right].copy()
        segments = segment_characters(crop)
        if not 6 <= len(segments) <= 11:
            continue
        centers = np.array([box[1] + box[3]/2 for _, box in segments])
        if w/h < 2.3 and np.ptp(centers) < .25*h:
            continue
        box = (left, top, right-left, bottom-top)
        if any(_box_iou(box, previous[1]) > .8 for previous in found):
            continue
        found.append((crop, box, .9))
    return found


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_w, inter_h = max(0, min(ax + aw, bx + bw) - max(ax, bx)), max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0
