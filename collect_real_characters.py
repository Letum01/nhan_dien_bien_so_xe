"""Export unlabeled character candidates ONLY from reviewed training crops.

This never assigns a plate string position to a contour (counts can be wrong).
Review each exported glyph and enter its true class in the generated CSV.
"""
import argparse
import csv
import hashlib
from pathlib import Path

import cv2

from segmentation import segment_characters


def export_characters(manifest: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    candidates = []
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        if row.get("annotation_status") != "human_reviewed":
            continue
        group = row.get("source_group", "")
        if not group or row.get("split") not in ("train", "validation", "test"):
            raise ValueError("Reviewed rows require source_group and train/validation/test split")
        if group in groups and groups[group] != row["split"]:
            raise ValueError("Source leakage: same group occurs across splits")
        groups[group] = row["split"]
        path = manifest.parent / row["image_path"]
        crop = cv2.imread(str(path))
        if crop is None:
            raise ValueError(f"Could not decode {path}")
        key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
        for index, (char, box) in enumerate(segment_characters(crop)):
            filename = f"{key}_{index:02d}.png"
            if not cv2.imwrite(str(output / filename), char):
                raise OSError(filename)
            candidates.append(dict(image_path=filename, class_label="", annotation_status="unlabeled",
                                   source_group=group, split=row["split"], source_crop=str(path), bbox=str(box)))
    with (output / "characters_to_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "class_label", "annotation_status", "source_group", "split", "source_crop", "bbox"])
        writer.writeheader()
        writer.writerows(candidates)
    return len(candidates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path, help="Must be a NEW directory; never overwrites data")
    args = parser.parse_args()
    print("Unlabeled candidates exported:", export_characters(args.manifest, args.output))