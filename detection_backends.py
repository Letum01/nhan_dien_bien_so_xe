"""Multiple-plate detector interface; no OCR or pseudo ground-truth involved."""
from localization import locate_license_plates


def detect_plates(image, backend="opencv", weights=None, max_plates=30):
    if backend == "opencv":
        return locate_license_plates(image, max_plates=max_plates)
    if backend == "yolo":
        if not weights:
            raise ValueError("YOLO requires verified local license-plate weights")
        from yolo_detector import detect_yolo
        return detect_yolo(image, weights)[:max_plates]
    raise ValueError(f"Unknown detector backend: {backend}")


def recognize_multiple(image, ocr_backend="paddle", models=None, detector="opencv", weights=None):
    """Return all proposals and OCR results; OpenCV proposals remain unverified.

    No claim that every candidate is a plate. Caller must preserve proposal
    status/score kind and reject unreadable/error results as appropriate.
    """
    from ocr_backends import recognize_crop
    results = []
    for crop, box, score in detect_plates(image, detector, weights):
        result = recognize_crop(crop, ocr_backend, models)
        results.append({"bbox": box, "detector": detector, "detection_score": score,
                        "score_kind": "heuristic" if detector == "opencv" else "model_score_uncalibrated",
                        "candidate_status": "unverified_proposal", "ocr": result.to_dict()})
    return results