"""Human box labeling + grouped YOLO export, no model-suggested ground truth."""
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


def ui():
    import streamlit as st
    st.title("Gắn nhãn bounding box biển số bằng người")
    st.info("Nhập từng box [x,y,width,height]. Gắn TẤT CẢ biển trong ảnh. [] chỉ dùng cho ảnh thực sự không có biển. Không lưu nhãn từ dự đoán.")
    sources = sorted((ROOT / "dataset" / "license_plate_dataset").glob("*/images/*.jpg"))
    selected = Path(st.selectbox("Ảnh", [str(p) for p in sources]))
    image = cv2.imread(str(selected))
    st.caption(f"Kích thước: {image.shape[1]} x {image.shape[0]}; tọa độ tính trên ảnh gốc")
    boxes_text = st.text_area("Danh sách box JSON", value="[]")
    group = st.text_input("source_group (cùng xe/video = cùng nhóm)")
    split = st.selectbox("Split", ["train", "val", "test"])
    try:
        boxes = json.loads(boxes_text)
        row = {"image_path": str(selected.relative_to(ROOT)), "boxes": boxes,
               "source_group": group, "split": split, "human_reviewed": True}
        if group:
            validate_annotation(row)
        preview = image.copy()
        for x,y,w,h in boxes:
            cv2.rectangle(preview, (int(x),int(y)), (int(x+w),int(y+h)), (0,255,0), 2)
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_column_width=True)
    except (ValueError, TypeError, KeyError) as exc:
        st.error(str(exc))
        return
    confirmed = st.checkbox("Tôi đã kiểm tra thủ công tất cả box và nguồn/split")
    if st.button("Lưu annotation", disabled=not confirmed):
        try:
            validate_annotation(row)
            manifest = ROOT / "dataset" / "reviewed_detection.json"
            rows = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
            if any(old["source_group"] == group and old["split"] != split for old in rows):
                raise ValueError("Group already exists in another split")
            rows = [old for old in rows if old["image_path"] != row["image_path"]] + [row]
            temp = manifest.with_suffix(".tmp")
            temp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(manifest)
            st.success("Đã lưu reviewed_detection.json")
        except ValueError as exc:
            st.error(str(exc))


if __name__ == "__main__":
    import sys
    if "--export" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--export", type=Path, required=True)
        parser.add_argument("--manifest", type=Path, default=ROOT / "dataset" / "reviewed_detection.json")
        args = parser.parse_args()
        export_yolo(args.manifest, args.export)
    else:
        ui()