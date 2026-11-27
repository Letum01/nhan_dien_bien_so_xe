"""
Module 4: Models Architecture & Training
Author: ALPR Team
"""

from typing import Tuple, Dict, Any, Optional
import os
import json
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_svm_model(kernel: str = 'rbf', C: float = 10.0, gamma: str = 'scale') -> SVC:
    """
    Khởi tạo mô hình Support Vector Machine (SVM) để phân loại ký tự trên đặc trưng HOG.

    Args:
        kernel: Loại kernel ('rbf' hoặc 'linear').
        C: Hệ số phạt điều hòa (penalty parameter).
        gamma: Tham số hệ số hạt nhân kernel.

    Returns:
        Đối tượng SVC đã cấu hình.
    """
    return SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)


def build_rf_model(n_estimators: int = 100, max_depth: Optional[int] = 20) -> RandomForestClassifier:
    """
    Khởi tạo mô hình Random Forest Classifier để phân loại ký tự trên đặc trưng HOG.

    Args:
        n_estimators: Số lượng cây trong rừng (mặc định 100).
        max_depth: Độ sâu tối đa của mỗi cây.

    Returns:
        Đối tượng RandomForestClassifier.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=42
    )


def build_cnn_model(
    input_shape: Tuple[int, int, int] = (32, 32, 1),
    num_classes: int = 36
) -> tf.keras.Model:
    """
    Xây dựng kiến trúc mạng nơ-ron tích chập (CNN) chuẩn cho bài toán nhận diện ký tự:
    Conv2D -> BatchNormalization -> MaxPooling2D -> Dropout -> Dense.

    Args:
        input_shape: Kích thước ma trận ảnh đầu vào, mặc định (32, 32, 1).
        num_classes: Số lượng lớp phân loại đầu ra (mặc định 36).

    Returns:
        Mô hình tf.keras.Model đã compile.
    """
    model = models.Sequential([
        # Khối Conv Block 1
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255.0),  # Chuẩn hóa ma trận pixel về khoảng [0, 1]
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Khối Conv Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.30),

        # Khối Conv Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.35),

        # Khối Fully-Connected (Dense)
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.50),
        layers.Dense(num_classes, activation='softmax')
    ], name="ALPR_Character_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


def train_and_save_all_models(
    X_train_img: np.ndarray,
    y_train: np.ndarray,
    X_train_hog: np.ndarray,
    save_dir: str = "saved_models",
    epochs: int = 15,
    batch_size: int = 64,
    X_val_img: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Huấn luyện cả 3 mô hình (SVM, Random Forest, CNN) CHỈ TRÊN TẬP TRAIN SET
    và lưu lại vào thư mục chỉ định, đảm bảo không rò rỉ dữ liệu (Data Leakage).

    Args:
        X_train_img: Ảnh ký tự tập train (N, 32, 32, 1).
        y_train: Nhãn lớp dạng số (N,).
        X_train_hog: Vector đặc trưng HOG tập train (N, D).
        save_dir: Thư mục lưu checkpoint mô hình.
        epochs: Số epoch huấn luyện CNN.
        batch_size: Kích thước batch huấn luyện CNN.

    Returns:
        Dictionary chứa các mô hình đã huấn luyện và scaler.
    """
    os.makedirs(save_dir, exist_ok=True)
    if (X_val_img is None) != (y_val is None):
        raise ValueError("X_val_img and y_val must either both be supplied or both be omitted")

    # 1. Chuẩn hóa đặc trưng HOG chỉ fit trên Train set
    scaler = StandardScaler()
    X_train_hog_scaled = scaler.fit_transform(X_train_hog)
    joblib.dump(scaler, os.path.join(save_dir, "hog_scaler.joblib"))

    # 2. Huấn luyện SVM
    print("[INFO] Đang huấn luyện SVM (Kernel RBF)...")
    svm_model = build_svm_model()
    svm_model.fit(X_train_hog_scaled, y_train)
    joblib.dump(svm_model, os.path.join(save_dir, "svm_model.joblib"))
    print("[INFO] Đã lưu SVM model thành công.")

    # 3. Huấn luyện Random Forest
    print("[INFO] Đang huấn luyện Random Forest (n_estimators=100)...")
    rf_model = build_rf_model(n_estimators=100)
    rf_model.fit(X_train_hog, y_train)  # Tree-based không bắt buộc scale
    joblib.dump(rf_model, os.path.join(save_dir, "rf_model.joblib"))
    print("[INFO] Đã lưu Random Forest model thành công.")

    # 4. Huấn luyện CNN
    print("[INFO] Đang huấn luyện CNN với Data Augmentation...")
    cnn_model = build_cnn_model(input_shape=(32, 32, 1), num_classes=36)
    # Mild augmentation in TRAIN mode only; original checkpoints are unaffected.
    from dataset import get_data_augmentation_pipeline
    cnn_model = models.Sequential([layers.Input(shape=(32, 32, 1)),
                                   get_data_augmentation_pipeline(), cnn_model],
                                  name="ALPR_Augmented_Character_CNN")
    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    validation_data = (X_val_img, y_val) if X_val_img is not None else None
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss' if validation_data else 'loss', patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss' if validation_data else 'loss', factor=0.5, patience=2, min_lr=1e-6),
    ]
    if validation_data is not None:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            os.path.join(save_dir, "cnn_best.keras"), monitor="val_loss", save_best_only=True
        ))

    cnn_model.fit(
        X_train_img,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )
    cnn_model.save(os.path.join(save_dir, "cnn_model.keras"))
    with open(os.path.join(save_dir, "character_model_config.json"), "w", encoding="utf-8") as config_file:
        json.dump({"classes": list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "image_size": [32, 32], "input_range": [0, 255]}, config_file, indent=2)
    print("[INFO] Đã lưu CNN model thành công.")

    return {
        "svm": svm_model,
        "rf": rf_model,
        "cnn": cnn_model,
        "scaler": scaler
    }
