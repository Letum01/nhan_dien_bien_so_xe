"""Optional license-plate-only YOLO adapter. Never downloads generic weights."""
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import numpy as np


@lru_cache(maxsize=2)
def load_plate_yolo(weights: str):
    path = Path(weights).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Provide verified local license-plate weights: {path}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Optional ultralytics dependency is not installed") from exc
    model = YOLO(str(path))
    names = set(str(name).lower().replace("-", "_").replace(" ", "_") for name in model.names.values())
    if not names or not names <= {"license_plate", "licence_plate", "plate"}:
        raise ValueError(f"Checkpoint classes are not plate-only: {model.names}")
    return model


def detect_yolo(image: np.ndarray, weights: str, confidence: float = .4):
    """Return multiple crops, xywh boxes and YOLO model scores."""
    model = load_plate_yolo(weights)
    result = model.predict(source=image, conf=confidence, verbose=False)[0]
    detections = []
    height, width = image.shape[:2]
    for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1:
            detections.append((image[y1:y2, x1:x2].copy(), (x1, y1, x2-x1, y2-y1), float(score)))
    return detections