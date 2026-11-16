"""Inspect Otsu bright connected regions; no OCR-derived labels."""
from pathlib import Path
import cv2

output = Path("debug_outputs/bright_regions")
output.mkdir(parents=True, exist_ok=True)
for name in ("0402", "0403", "0404", "0405", "0406", "0407"):
    image = cv2.imread(f"dataset/license_plate_dataset/0/images/{name}.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if not (.01 <= w*h/(image.shape[0]*image.shape[1]) <= .2 and 1.15 <= w/h <= 6.5):
            continue
        boxes.append((x,y,w,h,round(cv2.contourArea(contour)/(w*h),3)))
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imwrite(str(output/f"{name}.jpg"),image)
    print(name, threshold, boxes)