"""Synthetic composites exercise runtime only; NOT real multi-plate accuracy."""
import json
from pathlib import Path
import cv2
import numpy as np
from yolo_ui_bridge import detect,CHECKPOINT

ROOT=Path(__file__).resolve().parent


def main():
    folder=ROOT/'reports/round2_streamlit'
    source=cv2.imread(str(ROOT/'dataset/license_plate_dataset/0/images/14.jpg'))
    cases={'blank_synthetic_negative':np.zeros((480,640,3),np.uint8),
           'two_vehicle_composite_NOT_real_multiplate':np.concatenate([source,source],axis=1),
           'downscaled_small_plate_development':cv2.resize(source,(150,120))}
    rows=[]
    for name,image in cases.items():
        detections=detect(image,str(CHECKPOINT));overlay=image.copy()
        for _,(x,y,w,h),score in detections:
            cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.imwrite(str(folder/(name+'.jpg')),overlay)
        rows.append({'case':name,'detection_count':len(detections),
                     'detections':[{'bbox':box,'confidence':score} for _,box,score in detections],
                     'scope':'runtime fixture only, no real-data accuracy claim'})
    (folder/'edgecases.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(rows)


if __name__=='__main__':main()