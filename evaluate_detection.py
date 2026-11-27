"""Detection precision/recall at IoU=.5 using explicit human-reviewed boxes."""
import argparse
import json
from pathlib import Path
import time

import cv2

from detection_backends import detect_plates
from label_detection import ROOT, validate_annotation
from localization import _box_iou


def match_boxes(predicted, expected, threshold=.5):
    """Greedy one-to-one matching by IoU, independent of predicted text."""
    pairs = sorted([(_box_iou(p, g), i, j) for i,p in enumerate(predicted) for j,g in enumerate(expected)], reverse=True)
    used_p, used_g, matched = set(), set(), []
    for iou, i, j in pairs:
        if iou >= threshold and i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            matched.append(iou)
    return len(matched), len(predicted)-len(matched), len(expected)-len(matched), matched


def evaluate(manifest, backend="opencv", weights=None):
    rows = json.loads(Path(manifest).read_text(encoding="utf-8"))
    results, all_ious = [], []
    tp = fp = fn = 0
    for row in rows:
        if not row.get("human_reviewed") or row.get("split") != "test":
            continue
        validate_annotation(row)
        image = cv2.imread(str(ROOT / row["image_path"]))
        start = time.perf_counter()
        proposals = detect_plates(image, backend, weights)
        elapsed = time.perf_counter()-start
        counts = match_boxes([p[1] for p in proposals], row["boxes"])
        tp += counts[0]; fp += counts[1]; fn += counts[2]; all_ious.extend(counts[3])
        results.append({"image":row["image_path"], "tp":counts[0], "fp":counts[1], "fn":counts[2],
                        "matched_ious":counts[3], "seconds":elapsed})
    return {"backend":backend, "samples":len(results), "iou_threshold":.5,
            "precision":tp/(tp+fp) if tp+fp else None, "recall":tp/(tp+fn) if tp+fn else None,
            "mean_matched_iou":sum(all_ious)/len(all_ious) if all_ious else None,
            "tp":tp,"fp":fp,"fn":fn,"results":results,
            "scope":"proposal detection; OpenCV false proposals included, no mAP claimed"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("manifest",type=Path)
    parser.add_argument("--backend",choices=["opencv","yolo"],default="opencv")
    parser.add_argument("--weights")
    parser.add_argument("--output",type=Path,default=Path("detection_metrics.json"))
    args=parser.parse_args()
    args.output.write_text(json.dumps(evaluate(args.manifest,args.backend,args.weights),indent=2),encoding="utf-8")