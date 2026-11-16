"""PaddleOCR 3.x CPU worker. No TensorFlow imports or old .ocr API."""
import json
from pathlib import Path
import sys
import os
import time
import traceback
from contextlib import redirect_stdout

import numpy as np


def ordered_tokens(texts, scores, polygons):
    """Cluster by vertical centers relative to text height, then read left-right."""
    tokens = []
    for text, score, polygon in zip(texts, scores, polygons):
        points = np.asarray(polygon, dtype=float)
        tokens.append({"text": str(text), "score": float(score), "polygon": points.tolist(),
                       "x": float(points[:, 0].min()), "y": float(points[:, 1].mean()),
                       "height": max(1.0, float(np.ptp(points[:, 1])))})
    lines = []
    for token in sorted(tokens, key=lambda t: t["y"]):
        for line in lines:
            if abs(token["y"] - np.mean([t["y"] for t in line])) <= 0.5 * max(token["height"], np.median([t["height"] for t in line])):
                line.append(token)
                break
        else:
            lines.append([token])
    ordered = [sorted(line, key=lambda t: t["x"]) for line in lines]
    flat = [token for line in ordered for token in line]
    return "\n".join(" ".join(t["text"] for t in line) for line in ordered), flat


def create_engine():
    from paddleocr import PaddleOCR
    return PaddleOCR(
        device="cpu", use_doc_orientation_classify=False,
        use_doc_unwarping=False, use_textline_orientation=False,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="en_PP-OCRv4_mobile_rec",
        enable_mkldnn=False,
    )


def predict(engine, image):
    start = time.perf_counter()
    tokens = []
    raw = []
    for page in engine.predict(input=image):
        text, entries = ordered_tokens(page["rec_texts"], page["rec_scores"], page["rec_polys"])
        raw.append(text)
        tokens.extend(entries)
    result = {"raw_text": "\n".join(raw), "tokens": tokens,
              "confidence": min(t["score"] for t in tokens) if tokens else None,
               "confidence_kind": "minimum text-token model score, uncalibrated",
               "timings": {"inference_seconds": time.perf_counter() - start,
                           "worker_pid": os.getpid()}}
    return result


def serve():
    import base64
    import cv2
    protocol = sys.stdout
    # Paddle/PaddleX may print progress to stdout; reserve stdout for JSON.
    with redirect_stdout(sys.stderr):
        engine = create_engine()
    print(json.dumps({"event": "ready", "pid": os.getpid()}), file=protocol, flush=True)
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            start = time.perf_counter()
            encoded = np.frombuffer(base64.b64decode(request["png_base64"], validate=True), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Invalid PNG payload")
            decode_seconds = time.perf_counter() - start
            with redirect_stdout(sys.stderr):
                result = predict(engine, image)
            result["timings"]["decode_seconds"] = decode_seconds
            response = {"id": request["id"], "result": result}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {"id": request.get("id"), "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=True), file=protocol, flush=True)


def main():
    if sys.argv[1:] == ["--serve"]:
        serve()
    else:
        result = predict(create_engine(), sys.argv[1])
        Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()