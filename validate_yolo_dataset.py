"""Validate exported image-label pairs BEFORE any detector training."""
import argparse
import json
import math
from pathlib import Path

import cv2

from dataset_validation import check_split_leakage


def validate_dataset(root: Path):
    root=Path(root).resolve()
    manifest=root/"manifest.json"
    if not manifest.is_file():
        raise ValueError("Reviewed export manifest required; raw labels alone do not prove review")
    data=json.loads(manifest.read_text(encoding="utf-8"))
    rows=data["rows"]
    from label_detection import ROOT, validate_annotation
    for row in rows:
        validate_annotation(row)
    check_split_leakage(rows,ROOT)
    counts={}
    for split in ("train","val","test"):
        files=sorted((root/"images"/split).glob("*"))
        labels=sorted((root/"labels"/split).glob("*.txt"))
        if not files or {p.stem for p in files}!={p.stem for p in labels}:
            raise ValueError(f"Missing/orphan image-label pairs: {split}")
        for path in files:
            image=cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Corrupt exported image: {path}")
            for line in (root/"labels"/split/(path.stem+".txt")).read_text().splitlines():
                values=line.split()
                if len(values)!=5 or values[0]!="0":
                    raise ValueError(f"Invalid YOLO label: {path}")
                x,y,w,h=map(float,values[1:])
                if w*image.shape[1]<2-1e-5 or h*image.shape[0]<2-1e-5:
                    raise ValueError(f"Degenerate plate box below 2 pixels: {path}")
                if not all(math.isfinite(v) for v in (x,y,w,h)) or not (0<w<=1 and 0<h<=1 and w/2-1e-7<=x<=1-w/2+1e-7 and h/2-1e-7<=y<=1-h/2+1e-7):
                    raise ValueError(f"Out-of-range YOLO box: {path}")
        counts[split]=len(files)
    if sum(counts.values())!=len(rows):
        raise ValueError("Export/manifest sample count mismatch")
    return {"status":"PASS","images_by_split":counts,"manifest":str(manifest)}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("root",type=Path);args=parser.parse_args()
    print(json.dumps(validate_dataset(args.root),indent=2))