"""Compare input geometry variants without changing weights or production OCR."""
import json
from pathlib import Path

import cv2
import numpy as np

from main import load_trained_models, recognize_characters
from segmentation import segment_characters


def main() -> None:
    models = load_trained_models()
    if models is None:
        raise RuntimeError("Models unavailable")
    report = []
    root = Path(__file__).resolve().parent
    for name, box, visible_text in [
        ("0402", (208, 192, 145, 101), "65L100780"),
        ("0403", (209, 207, 177, 122), "83FD9972"),
    ]:
        image = cv2.imread(str(root / f"dataset/license_plate_dataset/0/images/{name}.jpg"))
        if image is None:
            raise ValueError(f"Cannot decode {name}")
        x, y, w, h = box
        segments = segment_characters(image[y:y+h, x:x+w])
        variants = {"square_padding_baseline": [ch for ch, _ in segments], "stretch_foreground": []}
        for ch, _ in segments:
            points = cv2.findNonZero(ch)
            if points is None:
                variants["stretch_foreground"].append(ch)
            else:
                cx, cy, cw, chh = cv2.boundingRect(points)
                variants["stretch_foreground"].append(cv2.resize(ch[cy:cy+chh, cx:cx+cw], (32, 32)))
        for variant, chars in variants.items():
            for model in ("cnn", "svm", "rf"):
                text = recognize_characters(chars, model, models)
                row = {"image": name, "variant": variant, "model": model, "prediction": text,
                       "visible_manual_transcription": visible_text, "segments": len(chars),
                       "note": "Two diagnostic development examples, not an independent benchmark"}
                print(row)
                report.append(row)
    output = root / "debug_outputs" / "geometry_diagnostic.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()