"""Measure cold/warm/cache separately, without changing OCR configuration."""
import json
from pathlib import Path
import time

import cv2

from ocr_backends import paddle_client, paddle_python


def main():
    root = Path(__file__).resolve().parent
    folder = root / "debug_outputs" / "ocr_comparison"
    client = paddle_client(str(paddle_python()))
    client.close()
    client.clear_cache()
    results = []
    for repeat in range(2):
        for name in ("0402", "0403", "0404", "0405", "0406", "0407"):
            start = time.perf_counter()
            image = cv2.imread(str(folder / f"{name}_crop.png"))
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise ValueError(name)
            preprocessing = time.perf_counter() - start
            result = client.request(encoded.tobytes(), use_cache=False)
            row = {"image": name, "repeat": repeat, "raw_text": result["raw_text"],
                   "preprocessing_seconds": preprocessing, **result["timings"],
                   "total_seconds": time.perf_counter() - start}
            results.append(row)
            print(row, flush=True)
    encoded = (folder / "0402_crop.png").read_bytes()
    client.request(encoded)
    cached = client.request(encoded)
    report = {"measurements": results, "cache_probe": cached["timings"],
              "note": "No accuracy claims; invalid crops 0407 are not successful OCR speed samples"}
    (folder / "persistent_benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    client.close()


if __name__ == "__main__":
    main()