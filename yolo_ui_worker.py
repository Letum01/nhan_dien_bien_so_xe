"""Isolated YOLO inference for TensorFlow Streamlit environment."""
import json
from pathlib import Path
import sys

import cv2
from yolo_detector import detect_yolo

if __name__=='__main__':
    image=cv2.imread(sys.argv[1])
    if image is None:raise ValueError('Invalid worker image')
    detections=detect_yolo(image,sys.argv[3],confidence=.4)
    Path(sys.argv[2]).write_text(json.dumps([{'bbox':box,'confidence':score} for _,box,score in detections]),encoding='utf-8')