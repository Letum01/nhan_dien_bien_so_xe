"""Evaluate saved ALPR predictions against a human-reviewed CSV manifest."""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import cv2

from main import end_to_end_pipeline, load_trained_models


def normalize_plate(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def summarize_results(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate edit counts, not average CERs of unequal-length plates.

    Missing predictions are deletions. Image errors remain in the report and
    are counted as failures; zero eligible labels means unavailable metrics.
    """
    count = len(rows)
    characters = sum(len(str(row["expected"])) for row in rows)
    errors = sum(int(row["character_errors"]) for row in rows)
    return {
        "samples": count,
        "plate_exact_match": sum(bool(row["exact_match"]) for row in rows) / count if count else None,
        "cer": errors / characters if characters else None,
        "mean_cer": sum(float(row["cer"]) for row in rows) / count if count else None,
        "mean_seconds": sum(float(row["seconds"]) for row in rows) / count if count else None,
        "total_reference_characters": characters,
        "total_character_errors": errors,
        "failed_samples": sum(row["status"] != "prediction" for row in rows),
    }


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def evaluate_manifest(manifest: Path, model_name: str = "cnn", crop_mode: bool = False) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    models = None
    models_loaded = model_name == "paddle"
    skipped = 0
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        for sample in csv.DictReader(stream):
            if not sample.get("image_path") or not sample.get("plate_text"):
                continue
            if sample.get("annotation_status", "").lower() != "human_reviewed":
                skipped += 1
                continue
            if not models_loaded:
                models = load_trained_models(str(Path(__file__).resolve().parent / "saved_models"))
                if models is None or models.get(model_name) is None:
                    raise RuntimeError(f"Model unavailable: {model_name}")
                models_loaded = True
            path = Path(sample["image_path"])
            if not path.is_absolute():
                path = manifest.parent / path
            image = cv2.imread(str(path))
            expected = normalize_plate(sample["plate_text"])
            if not expected:
                raise ValueError(f"Reviewed annotation has no alphanumeric text: {path}")
            started = time.perf_counter()
            if image is None:
                prediction, actual, status = "", "", "image_error"
            else:
                if crop_mode:
                    from ocr_backends import recognize_crop
                    ocr = recognize_crop(image, model_name, models)
                    prediction, actual, status = ocr.raw_text, ocr.text, ocr.status
                else:
                    prediction, _, _, _ = end_to_end_pipeline(image, model_name, models)
                    status = "not_found" if prediction.startswith("UNKNOWN") else "prediction"
                    actual = "" if status == "not_found" else normalize_plate(prediction)
            elapsed = time.perf_counter() - started
            distance = edit_distance(expected, actual)
            rows.append({"image": str(path), "expected": expected, "prediction": actual,
                         "pipeline_output": prediction, "status": status,
                         "exact_match": expected == actual, "character_errors": distance,
                         "cer": distance / max(1, len(expected)), "seconds": elapsed,
                         "annotation_status": sample.get("annotation_status", "")})
    return {"model": model_name, **summarize_results(rows), "ground_truth_source": str(manifest),
            "scope": "ocr_on_reviewed_crops" if crop_mode else "end_to_end",
            "unreadable_rate": sum(row["status"] in ("unreadable", "not_found") for row in rows) / len(rows) if rows else None,
            "results": rows, "skipped_unreviewed": skipped,
            "warning": "Only rows marked human_reviewed are included. Detector metrics require verified box annotations and are not inferred from this OCR-only report."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", default="dataset/evaluation_ground_truth_template.csv")
    parser.add_argument("--model", choices=["cnn", "svm", "rf", "paddle"], default="cnn")
    parser.add_argument("--crop-mode", action="store_true", help="Evaluate OCR directly on reviewed plate crops, bypass detection")
    parser.add_argument("--output", default="evaluation_results.json")
    args = parser.parse_args()
    report = evaluate_manifest(Path(args.manifest), args.model, args.crop_mode)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model", "samples", "plate_exact_match", "mean_cer", "mean_seconds", "warning")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()