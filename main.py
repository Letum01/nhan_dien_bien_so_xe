"""
Module 5: Evaluation & Main End-to-End Pipeline
Author: ALPR Team
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import joblib
import cv2
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
try:
    import keras as keras_api
except ImportError:
    keras_api = None

from localization import locate_license_plate, locate_license_plates
from segmentation import segment_characters
from dataset import (
    CHARACTER_CLASSES,
    IDX_TO_CLASS,
    extract_hog_features,
    generate_synthetic_character_dataset,
    post_process_plate_text
)
from models import train_and_save_all_models


def evaluate_models(
    models_dict: Dict[str, Any],
    X_test_img: np.ndarray,
    X_test_hog: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Đánh giá định lượng trên Test set: In đầy đủ Accuracy, Precision, Recall,
    Macro F1-score và ma trận nhầm lẫn (Confusion Matrix).
    """
    scaler = models_dict.get("scaler")
    X_test_hog_scaled = scaler.transform(X_test_hog) if scaler is not None else X_test_hog
    results = {}

    svm_preds = models_dict["svm"].predict(X_test_hog_scaled)
    results["SVM"] = _compute_metrics("Support Vector Machine (SVM)", y_test, svm_preds)

    rf_preds = models_dict["rf"].predict(X_test_hog)
    results["RandomForest"] = _compute_metrics("Random Forest (RF)", y_test, rf_preds)

    cnn_probs = models_dict["cnn"].predict(X_test_img, verbose=0)
    cnn_preds = np.argmax(cnn_probs, axis=1)
    results["CNN"] = _compute_metrics("Convolutional Neural Network (CNN)", y_test, cnn_preds)

    return results


def _compute_metrics(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Hàm phụ trợ tính toán và hiển thị các chỉ số đánh giá."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 65)
    print(f"               KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH: {model_name}")
    print("=" * 65)
    print(f" - Accuracy (Độ chính xác):          {acc * 100:.2f}%")
    print(f" - Macro Precision (Độ chuẩn xác):   {prec * 100:.2f}%")
    print(f" - Macro Recall (Độ thu hồi):        {rec * 100:.2f}%")
    print(f" - Macro F1-score:                   {f1 * 100:.2f}%")
    print(f" - Confusion Matrix Shape:           {cm.shape}")
    print("-" * 65)
    print(classification_report(y_true, y_pred, labels=list(range(len(CHARACTER_CLASSES))),
                                target_names=CHARACTER_CLASSES, zero_division=0))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}


def load_trained_models(models_dir: str = "saved_models") -> Optional[Dict[str, Any]]:
    """Tải độc lập từng mô hình có sẵn; lỗi một mô hình không chặn mô hình khác."""
    svm_path = os.path.join(models_dir, "svm_model.joblib")
    rf_path = os.path.join(models_dir, "rf_model.joblib")
    cnn_path = os.path.join(models_dir, "cnn_model.keras")
    scaler_path = os.path.join(models_dir, "hog_scaler.joblib")

    loaded: Dict[str, Any] = {"svm": None, "rf": None, "cnn": None, "scaler": None}
    cnn_loader = keras_api.models.load_model if keras_api is not None else tf.keras.models.load_model
    for key, path, loader in (
        ("svm", svm_path, joblib.load),
        ("rf", rf_path, joblib.load),
        ("cnn", cnn_path, cnn_loader),
        ("scaler", scaler_path, joblib.load),
    ):
        if os.path.isfile(path):
            try:
                loaded[key] = loader(path, compile=False) if key == "cnn" else loader(path)
            except Exception as exc:
                print(f"[CẢNH BÁO] Không thể tải {key}: {exc}")
    return loaded if any(loaded[key] is not None for key in ("svm", "rf", "cnn")) else None


def recognize_characters(
    char_images: List[np.ndarray],
    model_type: str = "cnn",
    models_dict: Optional[Dict[str, Any]] = None,
    return_confidences: bool = False,
) -> Any:
    """Predict characters; optionally return uncalibrated model scores."""
    if not char_images:
        return ("", None) if return_confidences else ""
    if models_dict is None:
        raise ValueError("Không có mô hình nhận dạng")

    recognized_chars: List[str] = []

    if model_type.lower() == "cnn":
        if models_dict.get("cnn") is None:
            raise ValueError("CNN model is not available")
        batch_chars = np.stack(char_images, axis=0)[..., np.newaxis].astype(np.float32)
        probs = models_dict["cnn"].predict(batch_chars, verbose=0)
        pred_indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1).astype(float).tolist()
        recognized_chars = [IDX_TO_CLASS[idx] for idx in pred_indices]

    elif model_type.lower() == "svm":
        if models_dict.get("svm") is None:
            raise ValueError("SVM model is not available")
        hog_feats = np.array([extract_hog_features(img) for img in char_images], dtype=np.float32)
        if models_dict.get("scaler") is None:
            raise ValueError("SVM requires its training HOG scaler")
        hog_feats = models_dict["scaler"].transform(hog_feats)
        pred_indices = models_dict["svm"].predict(hog_feats)
        recognized_chars = [IDX_TO_CLASS[int(idx)] for idx in pred_indices]
        confidences = None
        if hasattr(models_dict["svm"], "predict_proba"):
            probabilities = models_dict["svm"].predict_proba(hog_feats)
            # SVC predict() can disagree with argmax(predict_proba()). Report
            # the score for the actual selected class, not another class.
            class_columns = {int(label): index for index, label in enumerate(models_dict["svm"].classes_)}
            confidences = [float(probabilities[i, class_columns[int(label)]])
                           for i, label in enumerate(pred_indices)]

    elif model_type.lower() in ("rf", "random_forest"):
        if models_dict.get("rf") is None:
            raise ValueError("Random Forest model is not available")
        hog_feats = np.array([extract_hog_features(img) for img in char_images], dtype=np.float32)
        pred_indices = models_dict["rf"].predict(hog_feats)
        recognized_chars = [IDX_TO_CLASS[int(idx)] for idx in pred_indices]
        confidences = None

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    result = "".join(recognized_chars)
    return (result, confidences) if return_confidences else result


def end_to_end_pipeline(
    image_path_or_bgr: Any,
    model_type: str = "cnn",
    models_dict: Optional[Dict[str, Any]] = None,
    trace: Optional[Dict[str, Any]] = None,
    detector_type: str = "opencv",
    detector_weights: Optional[str] = None,
) -> Tuple[str, np.ndarray, Optional[np.ndarray], List[np.ndarray]]:
    """
    Quy trình khép kín (End-to-End ALPR Pipeline):
    1. Đọc ảnh xe đầu vào.
    2. Định vị biển số xe (Localization).
    3. Phân tách từng ký tự (Segmentation).
    4. Nhận diện ký tự bằng mô hình (OCR Classification).
    5. Vẽ hộp và ghi kết quả chuỗi ký tự nhận diện lên ảnh.
    """
    if model_type not in ("svm", "rf", "cnn", "ensemble"):
        raise ValueError("Product recognition supports only SVM, RF, CNN and Ensemble")
    if isinstance(image_path_or_bgr, (str, os.PathLike)):
        image = cv2.imread(image_path_or_bgr)
        if image is None:
            raise ValueError(f"Không thể mở ảnh từ đường dẫn: {image_path_or_bgr}")
    else:
        if not isinstance(image_path_or_bgr, np.ndarray) or image_path_or_bgr.size == 0:
            raise ValueError("Ảnh đầu vào phải là mảng ảnh BGR hợp lệ.")
        image = image_path_or_bgr.copy()
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError("Ảnh phải có 2 chiều grayscale hoặc 3/4 kênh màu")
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    annotated = image.copy()
    if trace is not None:
        trace.clear()
        trace.update({"selected_box": None, "raw_text": "", "processed_text": "",
                      "segments": [], "confidences": None, "status": "not_found"})

    detection_started = time.perf_counter()
    if detector_type == "opencv":
        proposals = locate_license_plates(image, max_plates=30)
    elif detector_type == "yolo":
        if not detector_weights:
            raise ValueError("YOLO requires verified license-plate weights")
        from yolo_detector import detect_yolo
        proposals = detect_yolo(image, detector_weights)
    elif detector_type == 'yolo_isolated':
        from yolo_ui_bridge import detect
        proposals=detect(image,detector_weights)
    else:
        raise ValueError(f"Unsupported detector: {detector_type}")
    if detector_type in ('yolo','yolo_isolated'):
        for _,(dx,dy,dw,dh),score in proposals:
            cv2.rectangle(annotated,(dx,dy),(dx+dw,dy+dh),(0,180,255),2)
            cv2.putText(annotated,f'YOLO {score:.3f}',(dx,max(15,dy-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,180,255),1)
        if trace is not None:
            trace['detections']=[{'bbox':box,'confidence':float(score)} for _,box,score in proposals]
    viable = []
    for crop, box, score in proposals:
        h_candidate, w_candidate = crop.shape[:2]
        ratio = w_candidate / max(h_candidate, 1)
        if h_candidate < 24 or w_candidate < 60 or not 1.25 <= ratio <= 4.5:
            continue
        chars_candidate = segment_characters(crop)
        if 6 <= len(chars_candidate) <= 11:
            geometry_score = float(np.exp(-abs(np.log(ratio / 1.8))))
            candidate_score = 0.65 * geometry_score + 0.35 * min(score, 1.0)
            viable.append((candidate_score, crop, box, chars_candidate))

    if viable:
        viable.sort(key=lambda item: item[0], reverse=True)
        _, plate_crop, bbox, segmented = viable[0]
    elif detector_type in ('yolo','yolo_isolated') and proposals:
        plate_crop,bbox,_=max(proposals,key=lambda p:p[2])
        segmented=segment_characters(plate_crop)
    else:
        plate_crop, bbox = locate_license_plate(image) if detector_type == "opencv" else (None, None)
        segmented = segment_characters(plate_crop) if plate_crop is not None else []
        if plate_crop is not None and (plate_crop.shape[1] < 60 or plate_crop.shape[0] < 24
                                       or not 1.25 <= plate_crop.shape[1] / max(plate_crop.shape[0], 1) <= 4.5
                                       or not 6 <= len(segmented) <= 11):
            plate_crop, bbox, segmented = None, None, []

    if plate_crop is None or bbox is None:
        h, w = image.shape[:2]
        if detector_type == "opencv" and 1.2 <= (w / h) <= 6.5 and h < 300 and 6 <= len(segment_characters(image)) <= 11:
            plate_crop = image.copy()
            bbox = (0, 0, w, h)
            segmented = segment_characters(plate_crop)
        else:
            return "UNKNOWN (Plate Not Found)", annotated, None, []

    x, y, w, h = bbox
    if trace is not None:
        trace.update({"selected_box": bbox, "segments": segmented, "status": "segmented",
                      "detection_selection_seconds": time.perf_counter() - detection_started})
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)

    required_models = ("svm", "rf", "cnn", "scaler") if model_type == "ensemble" else (model_type,)
    if models_dict is None or any(models_dict.get(key) is None for key in required_models):
        raise ValueError(f"Mô hình '{model_type}' không khả dụng")

    if not segmented or (detector_type in ('yolo','yolo_isolated') and not 6<=len(segmented)<=11):
        cv2.putText(annotated, "Plate Detected - No Chars", (x, max(30, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if trace is not None:trace['status']='segmentation_uncertain'
        return "UNKNOWN (Segmentation Uncertain)", annotated, plate_crop, [s[0] for s in segmented]

    char_images = [item[0] for item in segmented]
    from ocr_backends import recognize_crop
    ocr = recognize_crop(plate_crop, model_type, models_dict, segmented)
    if ocr.status == "error":
        raise RuntimeError(ocr.error)
    raw_text = ocr.raw_text
    confidences = [token.get("score") for token in ocr.tokens] if model_type != "ensemble" else None
    plate_text = ocr.text or ("UNKNOWN (Ensemble Disagreement)" if ocr.status == "uncertain" else "UNKNOWN (OCR Unreadable)")
    if trace is not None:
        trace.update({"raw_text": raw_text, "processed_text": plate_text,
                      "confidences": confidences, "status": "ocr_prediction_only" if ocr.text else "ocr_unreadable",
                      "ocr_result": ocr.to_dict()})

    cv2.putText(annotated, f"Plate: {plate_text}", (x, max(35, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    # Preserve the established four-value API; callers needing raw OCR may
    # use recognize_characters(..., return_confidences=True) directly.
    return plate_text, annotated, plate_crop, char_images
