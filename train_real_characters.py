"""Train NEW artifacts only from manually reviewed character labels.

Requires characters_to_review.csv with explicit class, split and source_group.
The original saved_models directory is never a valid output destination.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import CLASS_TO_IDX, extract_hog_features
from main import evaluate_models
from models import train_and_save_all_models

ROOT = Path(__file__).resolve().parent


def load_reviewed_characters(manifest: Path):
    data = {split: [] for split in ("train", "validation", "test")}
    groups, paths = {}, set()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("annotation_status") != "human_reviewed":
                continue
            label, split, group = row.get("class_label"), row.get("split"), row.get("source_group")
            if label not in CLASS_TO_IDX or split not in data or not group:
                raise ValueError("Reviewed rows need valid class_label, split, source_group")
            if group in groups and groups[group] != split:
                raise ValueError(f"Group leakage: {group}")
            groups[group] = split
            path = (manifest.parent / row["image_path"]).resolve()
            if path in paths:
                raise ValueError(f"Duplicate character image: {path}")
            paths.add(path)
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (32, 32):
                raise ValueError(f"Expected reviewed 32x32 glyph: {path}")
            data[split].append((image, CLASS_TO_IDX[label]))
    if any(len(rows) < 2 for rows in data.values()):
        raise ValueError("At least two reviewed glyphs per train/validation/test split required")
    arrays = {split: (np.stack([x for x, _ in rows])[..., None], np.array([y for _, y in rows]))
              for split, rows in data.items()}
    train_classes = set(arrays["train"][1].tolist())
    if train_classes != set(CLASS_TO_IDX.values()):
        raise ValueError("Training requires all 36 classes; do not silently change class mapping")
    return arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path, help="New experiment directory OUTSIDE saved_models")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    destination = args.output.resolve()
    original = ROOT / "saved_models"
    if destination == original or original in destination.parents or destination.exists():
        raise ValueError("Output must be a NEW directory outside saved_models")
    arrays = load_reviewed_characters(args.manifest)
    x, y = arrays["train"]
    vx, vy = arrays["validation"]
    tx, ty = arrays["test"]
    features = np.array([extract_hog_features(image.squeeze()) for image in x])
    trained = train_and_save_all_models(x, y, features, save_dir=str(destination),
                                        X_val_img=vx, y_val=vy, epochs=args.epochs)
    test_features = np.array([extract_hog_features(image.squeeze()) for image in tx])
    report = evaluate_models(trained, tx, test_features, ty)
    (destination / "reviewed_test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    from main import recognize_characters
    from dataset import CHARACTER_CLASSES, IDX_TO_CLASS
    from sklearn.metrics import confusion_matrix, classification_report
    expected = [IDX_TO_CLASS[int(label)] for label in ty]
    details = {}
    for backend in ("svm", "rf", "cnn"):
        predicted = list(recognize_characters([image.squeeze() for image in tx], backend, trained))
        details[backend] = {
            "confusion_matrix": confusion_matrix(expected, predicted, labels=CHARACTER_CLASSES).tolist(),
            "class_order": CHARACTER_CLASSES,
            "classification_report": classification_report(expected, predicted, labels=CHARACTER_CLASSES,
                                                             output_dict=True, zero_division=0),
        }
    (destination / "reviewed_test_per_class.json").write_text(json.dumps(details, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()