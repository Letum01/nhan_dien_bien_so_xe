"""Six fixed images through real detection + persistent Paddle, no labels."""
import json
from pathlib import Path
import time

import cv2

from main import end_to_end_pipeline
from ocr_backends import paddle_client, paddle_python


def main():
    root = Path(__file__).resolve().parent
    client = paddle_client(str(paddle_python()))
    client.close()
    client.clear_cache()
    results = []
    for name in ("0402", "0403", "0404", "0405", "0406", "0407"):
        image = cv2.imread(str(root / f"dataset/license_plate_dataset/0/images/{name}.jpg"))
        trace = {}
        start = time.perf_counter()
        text, overlay, crop, _ = end_to_end_pipeline(image, "paddle", trace=trace)
        total = time.perf_counter() - start
        report = {"image": name, "text": text, "box": trace.get("selected_box"),
                  "detection_selection_seconds": trace.get("detection_selection_seconds"),
                  "ocr": trace.get("ocr_result"), "total_pipeline_seconds": total}
        results.append(report)
        print(name, text, total, flush=True)
        output = root / "debug_outputs" / "persistent_pipeline"
        output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output / f"{name}_overlay.jpg"), overlay)
        if crop is not None:
            cv2.imwrite(str(output / f"{name}_crop.png"), crop)
    (output / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    client.close()


if __name__ == "__main__":
    main()