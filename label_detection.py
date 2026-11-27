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


def ui():
    from annotation_ui import run
    run()


def annotation_ui():
    import streamlit as st
    from PIL import Image
    from streamlit_drawable_canvas import st_canvas
    from annotation_store import AnnotationStore, image_id, canvas_boxes
    st.title("Gắn nhãn bounding box biển số bằng người")
    store = AnnotationStore(ROOT / "data" / "annotations.sqlite3")
    sources = sorted((ROOT / "dataset" / "license_plate_dataset").glob("*/images/*.jpg"))
    if not sources:
        st.warning("Không có ảnh nguồn")
        return
    selected = Path(st.selectbox("Ảnh nguồn", [str(p) for p in sources], key="source_image"))
    identity = image_id(selected)
    previous = store.latest(identity) or {}
    image = Image.open(selected).convert("RGB")
    scale = st.slider("Tỷ lệ hiển thị", .25, 2.0, 1.0, .25)
    mode = st.radio("Thao tác", ["rect", "transform"], horizontal=True)
    st.caption("rect: vẽ box; transform: di chuyển/resize; dùng công cụ canvas để xóa/undo. Chỉ box axis-aligned. Save draft trước khi chuyển ảnh.")
    show_proposals = st.checkbox("Xem gợi ý OpenCV (không tự duyệt)")
    if show_proposals:
        import numpy as np
        from localization import locate_license_plates
        preview = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for _, (x,y,w,h), _ in locate_license_plates(preview):
            cv2.rectangle(preview,(x,y),(x+w,y+h),(0,180,255),2)
        st.image(cv2.cvtColor(preview,cv2.COLOR_BGR2RGB),caption="Proposal màu vàng, không phải ground truth")
    objects = [{"type":"rect","left":x*scale,"top":y*scale,"width":w*scale,"height":h*scale,
                "fill":"rgba(0,180,0,0.12)","stroke":"green","strokeWidth":2}
               for x,y,w,h in previous.get("boxes", [])]
    canvas = st_canvas(background_image=image, width=int(image.width*scale),height=int(image.height*scale),
                       drawing_mode=mode, stroke_width=2, stroke_color="green",
                       initial_drawing={"objects":objects}, update_streamlit=True,
                       key=f"boxes_{identity}_{scale}")
    group = st.text_input("source_group", value=previous.get("source_group", ""), key=f"group_{identity}")
    split = st.selectbox("Split", ["train","val","test"], index=["train","val","test"].index(previous.get("split","train")),key=f"split_{identity}")
    status = st.selectbox("Review status", ["unlabeled","proposed","needs_review","rejected","reviewed"],key=f"status_{identity}")
    confirmed = st.checkbox("Tôi đã xem tất cả box; [] xác nhận không có biển",key=f"confirm_{identity}")
    if canvas.json_data is not None:
        try:
            boxes = canvas_boxes(canvas.json_data.get("objects",[]),scale,image.width,image.height)
        except ValueError as exc:
            st.error(str(exc)); return
        row = {"image_id":identity,"image_path":str(selected.relative_to(ROOT)),"boxes":boxes,
               "source_group":group,"split":split,"review_status":status,"confirmed_by_user":confirmed}
        if st.button("Lưu phiên bản / draft"):
            try:
                if status=="reviewed":
                    if not group or not confirmed:
                        raise ValueError("Reviewed requires group and explicit human confirmation")
                    # Maintain existing exporter-compatible manifest; immutable history stays in SQLite.
                    manifest=ROOT/"dataset"/"reviewed_detection.json"
                    rows=json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
                    if any(old["source_group"]==group and old["split"]!=split for old in rows):
                        raise ValueError("Group exists in another split; reviewed export not changed")
                    validate_annotation({**row,"human_reviewed":True})
                saved=store.save(row,expected_version=previous.get("annotation_version",0))
                if status!="reviewed":
                    # Revoking review must also revoke eligibility in the legacy
                    # exporter manifest; SQLite retains the old reviewed revision.
                    manifest=ROOT/"dataset"/"reviewed_detection.json"
                    if manifest.exists():
                        rows=json.loads(manifest.read_text(encoding="utf-8"))
                        rows=[old for old in rows if old["image_path"]!=row["image_path"]]
                        temporary=manifest.with_suffix(".tmp")
                        temporary.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
                        temporary.replace(manifest)
                if status=="reviewed":
                    reviewed_row={**saved,"human_reviewed":True}
                    rows=[old for old in rows if old["image_path"]!=row["image_path"]]+[reviewed_row]
                    temporary=manifest.with_suffix(".tmp")
                    temporary.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
                    temporary.replace(manifest)
                st.success("Đã lưu phiên bản; có thể tải lại để tiếp tục")
            except ValueError as exc:
                st.error(str(exc))
    with st.expander("Lịch sử annotation"):
        st.json(store.history(identity))
    return


def legacy_json_ui():
    """Retained numeric-box editor for environments without canvas."""
    import streamlit as st
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