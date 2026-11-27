"""Reproducible ALPR stage artifacts for diagnosing unlabeled images.

Usage: python debug_pipeline.py --images paths... --output debug_outputs
The report contains predictions only; it never creates ground-truth labels.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from dataset import post_process_plate_text
from localization import locate_license_plate, locate_license_plates, preprocess_image
from main import end_to_end_pipeline, load_trained_models, recognize_characters
from segmentation import preprocess_plate, segment_characters

LOGGER = logging.getLogger("alpr.debug")


def rectify_plate(crop: np.ndarray) -> np.ndarray:
    """Perspective rectify a dominant quadrilateral when one is measurable."""
    if crop is None or crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    for contour in candidates:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or cv2.contourArea(approx) < crop.shape[0] * crop.shape[1] * 0.12:
            continue
        points = approx.reshape(4, 2).astype(np.float32)
        sums, diffs = points.sum(axis=1), np.diff(points, axis=1).ravel()
        ordered = np.array([points[np.argmin(sums)], points[np.argmin(diffs)],
                            points[np.argmax(sums)], points[np.argmax(diffs)]], dtype=np.float32)
        tl, tr, br, bl = ordered
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        if width < 20 or height < 10 or width / height < 1.1:
            continue
        target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        return cv2.warpPerspective(crop, cv2.getPerspectiveTransform(ordered, target), (width, height))
    return crop.copy()


def _save_image(path: Path, image: Optional[np.ndarray]) -> None:
    if image is not None and image.size:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write debug image: {path}")


def inspect_image(image_path: Path, output_root: Path, models: Dict[str, Any], model_name: str = "cnn") -> Dict[str, Any]:
    """Run all inspectable stages and save pixel-level evidence for one image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image decode failed: {image_path}")
    folder = output_root / image_path.stem
    folder.mkdir(parents=True, exist_ok=True)
    _save_image(folder / "00_input.jpg", image)
    preprocessed = preprocess_image(image)
    _save_image(folder / "01_localization_edges.png", preprocessed)

    proposals = locate_license_plates(image, max_plates=30)
    result: Dict[str, Any] = {
        "image": str(image_path), "proposal_count": len(proposals), "selected_box": None,
        "character_count": 0, "raw_text": "", "processed_text": "", "status": "not_found",
        "model": model_name, "characters": [], "ground_truth_used": False,
    }
    production_trace: Dict[str, Any] = {}
    production_text, production_overlay, production_crop, _ = end_to_end_pipeline(
        image, model_name, models, trace=production_trace
    )
    _save_image(folder / "08_production_overlay.jpg", production_overlay)
    _save_image(folder / "08_production_crop.png", production_crop)
    if production_crop is not None:
        _save_image(folder / "08_production_binary.png", preprocess_plate(production_crop))
    production_segments = production_trace.pop("segments")
    production_trace["character_boxes"] = [list(box) for _, box in production_segments]
    production_trace["character_count"] = len(production_segments)
    production_trace["display_text"] = production_text
    result["production_pipeline"] = production_trace
    for i, (char, _) in enumerate(production_segments):
        _save_image(folder / "production_characters" / f"{i:02d}.png", char)
    # Top-level report below describes the experimental rectification/ranking
    # branch. Never confuse it with the exact production trace saved above.
    result["selection_branch"] = "experimental_geometry_not_production"
    annotated = image.copy()
    for idx, (_, box, score) in enumerate(proposals):
        x, y, w, h = box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 180, 255), 2)
        cv2.putText(annotated, f"{idx}: {score:.2f}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 180, 255), 1)
    # Always save the single best baseline ROI on top of proposal overview.
    baseline_crop, baseline_box = locate_license_plate(image)
    if baseline_box is not None:
        bx, by, bw, bh = baseline_box
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
        cv2.putText(annotated, "baseline", (bx, max(12, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 0, 0), 1)
    _save_image(folder / "02_proposals.jpg", annotated)
    _save_image(folder / "02b_baseline_crop.jpg", baseline_crop)

    if not proposals:
        result["annotated"] = str(folder / "02_proposals.jpg")
        _write_json(folder / "report.json", result)
        return result

    # Evaluate every proposal through segmentation and prefer character evidence,
    # but never turn the heuristic into a calibrated detection confidence.
    ranked = []
    for idx, (crop, box, score) in enumerate(proposals):
        rectified = rectify_plate(crop)
        binary = preprocess_plate(rectified)
        segments = segment_characters(rectified)
        char_count = len(segments)
        ratio = crop.shape[1] / max(crop.shape[0], 1)
        aspect_fit = float(np.exp(-abs(np.log(max(ratio, 1e-6) / 1.7)))) if 1.0 <= ratio <= 3.2 else 0.0
        count_fit = max(0.0, 1.0 - abs(char_count - 8.0) / 8.0)
        if char_count < 5 or char_count > 12:
            count_fit = 0.0
        # Geometry/text-layout evidence only; it is intentionally not reported
        # as a probability or model confidence.
        selection_score = 0.50 * aspect_fit + 0.30 * count_fit + 0.20 * score
        ranked.append((selection_score, char_count, score, idx, crop, rectified, binary, segments, box))
        candidate_visual = crop.copy()
        for char_idx, (_, (cx, cy, cw, ch)) in enumerate(segments):
            cv2.rectangle(candidate_visual, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 1)
            cv2.putText(candidate_visual, str(char_idx), (cx, max(9, cy - 2)), cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 0, 0), 1)
        _save_image(folder / f"03_candidate_{idx}_crop.jpg", candidate_visual)
        _save_image(folder / f"04_candidate_{idx}_rectified.jpg", rectified)
        _save_image(folder / f"05_candidate_{idx}_binary.png", binary)
    ranked.sort(key=lambda item: item[0], reverse=True)
    selection_score, count, score, idx, crop, rectified, binary, segments, box = ranked[0]
    # The old code always selected something, even when all candidates were a
    # tiny tail-light fragment. Now return UNKNOWN unless there is plausible
    # aspect and character-count evidence.
    plausible = count in range(6, 11) and 1.25 <= crop.shape[1] / max(crop.shape[0], 1) <= 3.2
    result.update({"selected_box": list(box) if plausible else None, "selected_candidate": idx if plausible else None,
                   "candidate_score_heuristic": float(selection_score), "character_count": len(segments),
                   "status": "segmented" if plausible else "low_quality_candidates_rejected"})
    if not plausible:
        _write_json(folder / "report.json", result)
        return result

    char_images = [item[0] for item in segments]
    raw_text, confidences = recognize_characters(char_images, model_name, models, return_confidences=True)
    processed = post_process_plate_text(raw_text, confidences)
    result.update({"status": "ocr_prediction_only", "raw_text": raw_text, "processed_text": processed,
                   "confidence": confidences, "characters": []})
    char_grid = np.full((40, max(1, 40 * len(segments)), 3), 255, dtype=np.uint8)
    for char_idx, (char, bbox) in enumerate(segments):
        _save_image(folder / f"char_{char_idx:02d}_{raw_text[char_idx]}.png", char)
        char_grid[4:36, char_idx * 40 + 4:char_idx * 40 + 36] = cv2.cvtColor(char, cv2.COLOR_GRAY2BGR)
        result["characters"].append({"index": char_idx, "label": raw_text[char_idx], "bbox_in_crop": list(bbox),
                                     "confidence": confidences[char_idx] if confidences else None})
    _save_image(folder / "06_character_strip.png", char_grid)
    overlay = image.copy()
    x, y, w, h = box
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(overlay, f"raw={raw_text} post={processed}", (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 2)
    _save_image(folder / "07_selected_box_ocr.jpg", overlay)
    result["artifacts"] = {
        "input": str(folder / "00_input.jpg"), "preprocess": str(folder / "01_localization_edges.png"),
        "proposals": str(folder / "02_proposals.jpg"), "baseline_crop": str(folder / "02b_baseline_crop.jpg"),
        "selected_crop": str(folder / f"03_candidate_{idx}_crop.jpg"),
        "rectified": str(folder / f"04_candidate_{idx}_rectified.jpg"),
        "binary": str(folder / f"05_candidate_{idx}_binary.png"),
        "character_strip": str(folder / "06_character_strip.png"),
        "ocr_overlay": str(folder / "07_selected_box_ocr.jpg"),
    }
    _write_json(folder / "report.json", result)
    return result


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", default="debug_outputs")
    parser.add_argument("--model", choices=["cnn", "svm", "rf"], default="cnn")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    models = load_trained_models("saved_models")
    if not models or models.get(args.model) is None:
        raise RuntimeError(f"Required model unavailable: {args.model}")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for path in map(Path, args.images):
        try:
            results.append(inspect_image(path, root, models, args.model))
        except Exception:
            LOGGER.exception("Debug inspection failed for %s", path)
            results.append({"image": str(path), "status": "error"})
    with (root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["image", "model", "proposal_count", "selected_box", "character_count", "raw_text", "processed_text", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in results)
    _write_json(root / "summary.json", {"results": results, "ground_truth_used": False})


if __name__ == "__main__":
    main()