"""Human glyph-region editor: manual boxes/labels/rows, no positional pseudo-labels."""
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from annotation_store import AnnotationStore, image_id
from dataset_validation import validate_boxes

ROOT=Path(__file__).resolve().parent


def export_reviewed_glyphs(crop_path, annotations, destination, group, split):
    """Export originals + normalized white-on-black glyphs to NEW directory."""
    from dataset import CHARACTER_CLASSES
    image=cv2.imread(str(crop_path))
    if image is None:
        raise ValueError("Cannot decode crop")
    validate_boxes([item["bbox"] for item in annotations],image.shape[1],image.shape[0])
    if not annotations or not group or split not in ("train","validation","test"):
        raise ValueError("Reviewed glyphs, group and split required")
    if any(item.get("label") not in CHARACTER_CLASSES or not item.get("confirmed") for item in annotations):
        raise ValueError("Each glyph needs exact single-character class and human confirmation")
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    originals=destination/"originals";originals.mkdir()
    rows=[]
    for index,item in enumerate(sorted(annotations,key=lambda item:(int(item["line"]),int(item["order"])))):
        x,y,w,h=map(int,item["bbox"])
        original=image[y:y+h,x:x+w]
        cv2.imwrite(str(originals/f"{index:03d}.png"),original)
        gray=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
        _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        # Polarity explicitly selected by human; do not guess based on OCR.
        if item.get("invert",False):
            binary=cv2.bitwise_not(binary)
        side=max(w,h);canvas=np.zeros((side,side),np.uint8)
        canvas[(side-h)//2:(side-h)//2+h,(side-w)//2:(side-w)//2+w]=binary
        normalized=cv2.resize(canvas,(32,32),interpolation=cv2.INTER_AREA)
        _,normalized=cv2.threshold(normalized,127,255,cv2.THRESH_BINARY)
        cv2.imwrite(str(destination/f"{index:03d}.png"),normalized)
        rows.append({"image_path":f"{index:03d}.png","class_label":item["label"],
                     "annotation_status":"human_reviewed","source_group":group,"split":split,
                     "source_crop":str(Path(crop_path).resolve()),"bbox":json.dumps(item["bbox"]),
                     "line":item["line"],"order":item["order"]})
    with (destination/"characters_to_review.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    from glyph_review_ui import run
    run()


def legacy_json_editor():
    st.title("Duyệt và sửa vùng ký tự thật")
    crops=sorted((ROOT/"debug_outputs").glob("ocr_comparison/*_crop.png"))
    if not crops:
        st.warning("Cần crop đã được duyệt trước");return
    crop=Path(st.selectbox("Crop",[str(p) for p in crops]))
    st.image(str(crop),width=500)
    st.warning("Chỉ duyệt ký tự từ crop đã kiểm tra đủ biển. Mỗi box một nhãn bằng người; không chia chuỗi plate tự động.")
    store=AnnotationStore(ROOT/"data"/"characters.sqlite3")
    identity=image_id(crop);prior=store.latest(identity) or {}
    if st.button("Tạo proposal segmentation (chưa duyệt)"):
        from segmentation import segment_characters
        image=cv2.imread(str(crop))
        candidates=[{"bbox":list(box),"label":"","line":0,"order":i,"confirmed":False,"invert":False}
                    for i,(_,box) in enumerate(segment_characters(image))]
        st.session_state[f"glyph_json_{identity}"]=json.dumps(candidates,indent=2)
    st.caption("Sửa bbox bằng số pixel gốc; thêm/xóa phần tử để tách/ghép vùng, đặt line/order. Chỉ confirmed=true sau khi xem glyph, không dựa vào model.")
    text=st.text_area("Glyph regions JSON",value=json.dumps(prior.get("characters",[]),indent=2),
                      key=f"glyph_json_{identity}",height=300)
    group=st.text_input("source_group",value=prior.get("source_group",""),key=f"char_group_{identity}")
    split=st.selectbox("split",["train","validation","test"])
    confirmed=st.checkbox("Đã kiểm tra từng box, ký tự, thứ tự dòng và nét không bị cắt")
    try:
        annotations=json.loads(text)
        image=cv2.imread(str(crop))
        validate_boxes([item["bbox"] for item in annotations],image.shape[1],image.shape[0])
        for item in annotations:
            x,y,w,h=map(int,item["bbox"])
            st.image(cv2.cvtColor(image[y:y+h,x:x+w],cv2.COLOR_BGR2RGB),caption=f"{item.get('label','')} line={item.get('line')} order={item.get('order')}",width=90)
        if st.button("Lưu draft ký tự"):
            store.save({"image_id":identity,"image_path":str(crop),"characters":annotations,
                        "source_group":group,"split":split,"review_status":"needs_review"})
            st.success("Draft đã lưu; các phiên bản trước được giữ")
        destination=st.text_input("Thư mục output MỚI",value=f"experiments/glyphs_{crop.stem}")
        if st.button("Duyệt và xuất glyph",disabled=not confirmed):
            export_reviewed_glyphs(crop,annotations,ROOT/destination,group,split)
            store.save({"image_id":identity,"image_path":str(crop),"characters":annotations,
                        "source_group":group,"split":split,"review_status":"reviewed","confirmed_by_user":True})
            st.success("Đã xuất ảnh gốc + normalized + CSV; không sửa model")
    except (ValueError,KeyError,TypeError,FileExistsError) as exc:
        st.error(str(exc))


if __name__=="__main__":
    main()