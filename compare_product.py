"""Product-only development predictions on fixed shared crops. No accuracy."""
import csv
import json
from pathlib import Path

import cv2

from main import end_to_end_pipeline, load_trained_models
from ocr_backends import recognize_crop


def main():
    root = Path(__file__).resolve().parent
    output = root / "debug_outputs" / "product_ensemble"
    output.mkdir(parents=True, exist_ok=True)
    models = load_trained_models(str(root / "saved_models"))
    rows = []
    for name in ("0402", "0403", "0404", "0405", "0406", "0407"):
        image = cv2.imread(str(root / f"dataset/license_plate_dataset/0/images/{name}.jpg"))
        trace = {}
        _, _, crop, _ = end_to_end_pipeline(image, "cnn", models, trace)
        if crop is not None:
            cv2.imwrite(str(output / f"{name}_crop.png"), crop)
        for backend in ("svm", "rf", "cnn", "ensemble"):
            if crop is None:
                row = {"image": name, "backend": backend, "status": "not_found", "raw_text": "", "text": ""}
            else:
                result = recognize_crop(crop, backend, models, trace["segments"])
                row = {"image": name, **result.to_dict()}
            rows.append(row)
            print(name, backend, row["status"], row["raw_text"], flush=True)
    (output / "predictions.json").write_text(json.dumps({"ground_truth_used": False, "results": rows}, indent=2), encoding="utf-8")
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "backend", "raw_text", "text", "status", "seconds", "error"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()