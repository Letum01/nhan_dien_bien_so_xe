"""Character-only inference in .venv-yolo; outputs original crop coordinates."""
import json
import math
from pathlib import Path
import sys

import cv2


def load_model(weights):
    from ultralytics import YOLO
    model=YOLO(str(weights))
    if model.task!='detect' or model.names!={0:'character'}:
        raise ValueError('Expected one-class character detector, not plate/COCO weights')
    return model


def predict(image,model):
    result = model.predict(image,imgsz=640,conf=.25,iou=.7,max_det=40,device=0,verbose=False)[0]
    h,w=image.shape[:2]
    rows=[]
    for (a,b,c,d),score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
        x,y=max(0,math.floor(a)),max(0,math.floor(b))
        r,s=min(w,math.ceil(c)),min(h,math.ceil(d))
        if r>x and s>y:
            rows.append({'bbox':[x,y,r-x,s-y],'score':score})
    return rows


def main():
    source, target, weights = map(Path, sys.argv[1:4])
    image = cv2.imread(str(source))
    if image is None or not weights.is_file():
        raise ValueError('Readable crop and trusted local checkpoint required')
    rows=predict(image,load_model(weights))
    target.write_text(json.dumps(rows),encoding='utf-8')


if __name__=='__main__':
    main()