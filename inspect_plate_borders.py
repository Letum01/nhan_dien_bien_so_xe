"""Development evidence: export convex plate-sized border quadrilaterals."""
from pathlib import Path
import cv2


def main():
    output = Path(__file__).resolve().parent / "debug_outputs" / "border_audit"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("0402", "0403", "0404", "0405", "0406", "0407"):
        image = cv2.imread(f"dataset/license_plate_dataset/0/images/{name}.jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            polygon = cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            x, y, w, h = cv2.boundingRect(polygon)
            area = cv2.contourArea(polygon)
            if area < .01 * image.shape[0] * image.shape[1] or not 1.15 < w / h < 6.5:
                continue
            boxes.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite(str(output / f"{name}.jpg"), image)
        print(name, boxes)


if __name__ == "__main__":
    main()