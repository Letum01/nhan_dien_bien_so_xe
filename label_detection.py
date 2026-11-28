"""Grouped YOLO export from existing reviewed annotations (no review UI)."""
import argparse
import json
from pathlib import Path
import shutil
import cv2
ROOT = Path(__file__).resolve().parent


def validate_annotation(row):
    if not row.get("human_reviewed") or not row.get("source_group") or row.get("split") not in ("train", "val", "test"):
        raise ValueError("Human review, source_group and train/val/test split required")
    image = cv2.imread(str(ROOT / row["image_path"]))
    if image is None:
        raise ValueError("Unreadable source image")
    height, width = image.shape[:2]
    from dataset_validation import validate_boxes
    validate_boxes(row["boxes"], width, height)
    for box in row["boxes"]:
        if len(box) != 4:
            raise ValueError("Boxes must be [x,y,width,height]")
        x, y, w, h = box
        if min(x, y) < 0 or min(w, h) <= 0 or x+w > width or y+h > height:
            raise ValueError(f"Box outside image: {box}")
    return width, height



def export_yolo(manifest: Path, destination: Path):
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    groups, images = {}, set()
    # Validate ALL annotations before creating output.
    for row in rows:
        validate_annotation(row)
        group, split = row["source_group"], row["split"]
        if group in groups and groups[group] != split:
            raise ValueError("Source group leakage across splits")
        groups[group] = split
        if row["image_path"] in images:
            raise ValueError("Duplicate source image")
        images.add(row["image_path"])
    if not rows or set(row["split"] for row in rows) != {"train", "val", "test"}:
        raise ValueError("Need reviewed train, val and test annotations")
    from dataset_validation import check_split_leakage
    check_split_leakage(rows, ROOT)
    destination.mkdir(parents=True, exist_ok=False)
    for index, row in enumerate(rows):
        source = ROOT / row["image_path"]
        width, height = validate_annotation(row)
        stem = f"{index:06d}"
        image_dir = destination / "images" / row["split"]
        label_dir = destination / "labels" / row["split"]
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_dir / (stem + source.suffix))
        lines = [f"0 {(x+w/2)/width:.8f} {(y+h/2)/height:.8f} {w/width:.8f} {h/height:.8f}"
                 for x,y,w,h in row["boxes"]]
        (label_dir / (stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    # JSON is valid YAML and avoids a mandatory YAML dependency.
    config = {"path": str(destination.resolve()), "train": "images/train", "val": "images/val",
              "test": "images/test", "names": ["license_plate"]}
    (destination / "data.yaml").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps({"rows":rows,"split_policy":"human_assigned_group_validated","seed":42},indent=2),encoding="utf-8")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT/'dataset/reviewed_detection.json')
    args = parser.parse_args()
    export_yolo(args.manifest, args.export)
