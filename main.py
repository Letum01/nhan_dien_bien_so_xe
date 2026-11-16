"""
Module 5: Evaluation & Main End-to-End Pipeline
Author: ALPR Team
"""

import os
from typing import Dict, Any, Optional, Tuple, List
import joblib
import cv2
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf

from localization import locate_license_plate
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
    print(classification_report(y_true, y_pred, target_names=CHARACTER_CLASSES, zero_division=0))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}


def load_trained_models(models_dir: str = "saved_models") -> Optional[Dict[str, Any]]:
    """Tải các mô hình và scaler đã lưu."""
    svm_path = os.path.join(models_dir, "svm_model.joblib")
    rf_path = os.path.join(models_dir, "rf_model.joblib")
    cnn_path = os.path.join(models_dir, "cnn_model.keras")
    scaler_path = os.path.join(models_dir, "hog_scaler.joblib")

    if not (os.path.exists(svm_path) and os.path.exists(rf_path) and os.path.exists(cnn_path)):
        return None

    try:
        svm_model = joblib.load(svm_path)
        rf_model = joblib.load(rf_path)
        cnn_model = tf.keras.models.load_model(cnn_path)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        return {"svm": svm_model, "rf": rf_model, "cnn": cnn_model, "scaler": scaler}
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể tải mô hình: {e}")
        return None


def recognize_characters(
    char_images: List[np.ndarray],
    model_type: str = "cnn",
    models_dict: Optional[Dict[str, Any]] = None
) -> str:
    """Nhận diện danh sách các ảnh ký tự cắt bằng mô hình đã chọn."""
    if not char_images or models_dict is None:
        return ""

    recognized_chars: List[str] = []

    if model_type.lower() == "cnn":
        batch_chars = np.stack(char_images, axis=0)[..., np.newaxis].astype(np.float32)
        probs = models_dict["cnn"].predict(batch_chars, verbose=0)
        pred_indices = np.argmax(probs, axis=1)
        recognized_chars = [IDX_TO_CLASS[idx] for idx in pred_indices]

    elif model_type.lower() == "svm":
        hog_feats = np.array([extract_hog_features(img) for img in char_images], dtype=np.float32)
        if models_dict.get("scaler") is not None:
            hog_feats = models_dict["scaler"].transform(hog_feats)
        pred_indices = models_dict["svm"].predict(hog_feats)
        recognized_chars = [IDX_TO_CLASS[idx] for idx in pred_indices]

    elif model_type.lower() in ("rf", "random_forest"):
        hog_feats = np.array([extract_hog_features(img) for img in char_images], dtype=np.float32)
        pred_indices = models_dict["rf"].predict(hog_feats)
        recognized_chars = [IDX_TO_CLASS[idx] for idx in pred_indices]

    return "".join(recognized_chars)


def end_to_end_pipeline(
    image_path_or_bgr: Any,
    model_type: str = "cnn",
    models_dict: Optional[Dict[str, Any]] = None
) -> Tuple[str, np.ndarray, Optional[np.ndarray], List[np.ndarray]]:
    """
    Quy trình khép kín (End-to-End ALPR Pipeline):
    1. Đọc ảnh xe đầu vào.
    2. Định vị biển số xe (Localization).
    3. Phân tách từng ký tự (Segmentation).
    4. Nhận diện ký tự bằng mô hình (OCR Classification).
    5. Vẽ hộp và ghi kết quả chuỗi ký tự nhận diện lên ảnh.
    """
    if isinstance(image_path_or_bgr, str):
        image = cv2.imread(image_path_or_bgr)
        if image is None:
            raise ValueError(f"Không thể mở ảnh từ đường dẫn: {image_path_or_bgr}")
    else:
        image = image_path_or_bgr.copy()

    annotated = image.copy()

    plate_crop, bbox = locate_license_plate(image)
    if plate_crop is None or bbox is None:
        h, w = image.shape[:2]
        if 1.2 <= (w / h) <= 6.5 and h < 300:
            plate_crop = image.copy()
            bbox = (0, 0, w, h)
        else:
            return "UNKNOWN (Plate Not Found)", annotated, None, []

    x, y, w, h = bbox
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)

    segmented = segment_characters(plate_crop)
    if not segmented:
        cv2.putText(annotated, "Plate Detected - No Chars", (x, max(30, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return "UNKNOWN (No Characters)", annotated, plate_crop, []

    char_images = [item[0] for item in segmented]
    raw_text = recognize_characters(char_images, model_type=model_type, models_dict=models_dict)
    plate_text = post_process_plate_text(raw_text)

    cv2.putText(annotated, f"Plate: {plate_text}", (x, max(35, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    return plate_text, annotated, plate_crop, char_images


if __name__ == "__main__":
    print("[1] Đang sinh tập dữ liệu huấn luyện ký tự...")
    X_all_img, y_all = generate_synthetic_character_dataset(samples_per_class=100)

    print(f"[2] Kích thước dữ liệu: {X_all_img.shape}, Số lớp: {len(np.unique(y_all))}")
    print("[3] Chia tập Train/Test theo tỷ lệ 80/20 (Stratified)...")
    X_train_img, X_test_img, y_train, y_test = train_test_split(
        X_all_img, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    print("[4] Trích xuất đặc trưng HOG cho Train và Test set...")
    X_train_hog = np.array([extract_hog_features(img.squeeze()) for img in X_train_img], dtype=np.float32)
    X_test_hog = np.array([extract_hog_features(img.squeeze()) for img in X_test_img], dtype=np.float32)

    models_dict = train_and_save_all_models(
        X_train_img, y_train, X_train_hog,
        save_dir="saved_models",
        epochs=12,
        batch_size=64
    )

    evaluate_models(models_dict, X_test_img, X_test_hog, y_test)

    sample_img_path = os.path.join("dataset", "license_plate_dataset", "0", "images", "0402.jpg")
    if os.path.exists(sample_img_path):
        print(f"\n[5] Thử nghiệm End-to-End Pipeline trên ảnh: {sample_img_path}")
        plate_text, _, _, chars = end_to_end_pipeline(sample_img_path, model_type="cnn", models_dict=models_dict)
        print(f" -> Kết quả nhận diện biển số xe: {plate_text} (Tách được {len(chars)} ký tự)")

