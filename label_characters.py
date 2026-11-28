"""Export validated glyph crops without a review application."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from dataset_validation import validate_boxes

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
