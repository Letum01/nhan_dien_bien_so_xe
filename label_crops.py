"""Local annotation UI: streamlit run label_crops.py (no suggested OCR labels)."""
import csv
import os
from pathlib import Path
import tempfile

import streamlit as st

ROOT = Path(__file__).resolve().parent
FIELDS = ["image_path", "plate_text", "annotation_status", "source_group", "split",
          "line_layout", "has_unreadable_characters", "reviewed_at", "annotation_version", "image_id"]


def save_label(destination: Path, row: dict) -> None:
    """Upsert one human label with atomic replacement, preserving other rows."""
    if not row["plate_text"].strip() or not row["source_group"].strip():
        raise ValueError("Text and source_group are required")
    if row.get("split") not in ("train","validation","test"):
        raise ValueError("Invalid split")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if destination.exists():
        with destination.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        # Preserve an immutable snapshot before replacing previously reviewed CSV.
        from datetime import datetime, timezone
        history=destination.parent/"annotation_history"
        history.mkdir(exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        (history/f"{destination.stem}_{stamp}.csv").write_bytes(destination.read_bytes())
    for old in rows:
        if old["source_group"] == row["source_group"] and old["split"] != row["split"]:
            raise ValueError("Same source_group must remain in the same split")
    from datetime import datetime, timezone
    version=max([int(old.get("annotation_version") or 0) for old in rows if old["image_path"]==row["image_path"]] or [0])+1
    row={**row,"reviewed_at":datetime.now(timezone.utc).isoformat(),"annotation_version":version}
    rows = [old for old in rows if old["image_path"] != row["image_path"]]
    rows.append(row)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".csv")
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    st.title("Gắn nhãn crop biển số bằng người")
    st.info("Chỉ lưu nếu crop chứa đầy đủ biển số. Không lấy dự đoán OCR làm nhãn. Ảnh cùng xe/video phải cùng source_group và split.")
    crops = sorted((ROOT / "debug_outputs" / "ocr_comparison").glob("*_crop.png"))
    if not crops:
        st.warning("Chạy compare_ocr.py trước để tạo crop.")
        return
    crop = Path(st.selectbox("Crop cần duyệt", [str(p) for p in crops]))
    st.image(str(crop), width=400)
    with st.form(f"human_label_{crop.name}", clear_on_submit=False):
        text = st.text_input("Biển số đúng (nhập bằng tay)")
        group = st.text_input("Nguồn/xe/video (source_group)")
        split = st.selectbox("Tập dữ liệu", ["test", "validation", "train"])
        layout=st.selectbox("Số dòng",["UNKNOWN","one_line","two_lines"])
        unreadable=st.checkbox("Có ký tự không đọc được (không dùng để train)")
        confirmed = st.checkbox("Đã xem và xác nhận crop chứa đầy đủ biển số, text đúng")
        if st.form_submit_button("Lưu nhãn"):
            if not confirmed:
                st.error("Cần xác nhận nhãn bằng người")
            else:
                destination = ROOT / "dataset" / "reviewed_crops.csv"
                try:
                    save_label(destination, {"image_path": os.path.relpath(crop, destination.parent),
                                             "plate_text": text, "annotation_status": "needs_review" if unreadable else "human_reviewed",
                                             "source_group": group, "split": split,"line_layout":layout,
                                             "has_unreadable_characters":str(unreadable).lower(),
                                             "image_id":__import__('hashlib').sha256(crop.read_bytes()).hexdigest()})
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success("Đã lưu reviewed_crops.csv")


if __name__ == "__main__":
    main()