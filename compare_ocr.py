"""Freeze production crops, compare four OCR backends; never generate labels."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import cv2

from main import end_to_end_pipeline, load_trained_models
from ocr_backends import recognize_crop

ROOT = Path(__file__).resolve().parent


def main():
    models = load_trained_models(str(ROOT / "saved_models"))
    output = ROOT / "debug_outputs" / "ocr_comparison"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    histograms = {backend: Counter() for backend in ("cnn", "svm", "rf", "paddle")}
    audit = {"cnn_input_shape": str(models["cnn"].input_shape),
             "cnn_output_shape": str(models["cnn"].output_shape),
             "cnn_rescaling": [layer.get_config() for layer in models["cnn"].layers if layer.__class__.__name__ == "Rescaling"],
             "svm_classes": models["svm"].classes_.tolist(), "rf_classes": models["rf"].classes_.tolist(),
             "scaler_features": int(models["scaler"].n_features_in_),
             "class_mapping": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", "ground_truth_used": False}
    for name in ("0402", "0403", "0404", "0405", "0406", "0407"):
        image_path = ROOT / f"dataset/license_plate_dataset/0/images/{name}.jpg"
        image = cv2.imread(str(image_path))
        trace = {}
        _, _, crop, _ = end_to_end_pipeline(image, "cnn", models, trace)
        if crop is None:
            for backend in histograms:
                results.append({"image": name, "backend": backend, "status": "no_crop", "raw_text": "",
                                "text": "", "seconds": None, "error": "Detector did not provide a crop"})
            continue
        crop_path = output / f"{name}_crop.png"
        cv2.imwrite(str(crop_path), crop)
        digest = hashlib.sha256(crop_path.read_bytes()).hexdigest()
        for backend in histograms:
            result = recognize_crop(crop, backend, models, trace["segments"])
            row = {"image": name, **result.to_dict(), "crop_path": str(crop_path.relative_to(ROOT)),
                   "crop_sha256": digest, "bbox": list(trace["selected_box"]),
                   "crop_review_status": "unreviewed_candidate_not_accuracy_data"}
            # Visual audit of saved crop images, not inferred from predictions.
            if name == "0407":
                row["crop_review_status"] = "visually_not_a_plate_exclude_from_ocr_accuracy"
            results.append(row)
            histograms[backend].update(ch for ch in result.raw_text if ch.isalnum())
            print(name, backend, result.status, repr(result.raw_text), round(result.seconds, 3), flush=True)
    audit["predicted_class_counts"] = {backend: dict(counts) for backend, counts in histograms.items()}
    (output / "results.json").write_text(json.dumps({"audit": audit, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["image", "backend", "raw_text", "text", "status", "confidence", "seconds", "error", "crop_path", "crop_sha256", "crop_review_status"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()