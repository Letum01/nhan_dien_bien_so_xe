"""Shared OCR results; Paddle is isolated from the TensorFlow environment."""
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


@dataclass
class OCRResult:
    backend: str
    raw_text: str = ""
    text: str = ""
    status: str = "unreadable"
    confidence: Optional[float] = None
    seconds: float = 0.0
    error: Optional[str] = None
    tokens: List[Dict[str, Any]] = field(default_factory=list)
    timings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def paddle_python() -> Path:
    return Path(os.environ.get("ALPR_PADDLE_PYTHON", str(ROOT / ".venv-paddle" / "Scripts" / "python.exe")))


def paddle_available() -> bool:
    """Environment existence only; weights/runtime are validated on first use."""
    return paddle_python().is_file()


@lru_cache(maxsize=4)
def paddle_client(python: str):
    from paddle_client import PaddleClient
    return PaddleClient(Path(python), ROOT / "paddle_worker.py")


def _paddle_cached(encoded: bytes) -> Dict[str, Any]:
    return paddle_client(str(paddle_python())).request(encoded)


def recognize_crop(crop: np.ndarray, backend: str, models=None, segments=None) -> OCRResult:
    """Common crop interface; errors are explicit, never synthetic predictions.

    Confidence is a model score, not a calibrated plate-correctness probability.
    Paddle raw text keeps punctuation/newlines; text only removes separators.
    """
    start = time.perf_counter()
    result = OCRResult(backend=backend)
    try:
        if crop is None or crop.size == 0:
            raise ValueError("Empty plate crop")
        if backend == "ensemble":
            from segmentation import segment_characters
            from ensemble_recognition import recognize_ensemble
            segments = segment_characters(crop) if segments is None else segments
            votes = recognize_ensemble([char for char, _ in segments], models)
            result.raw_text = votes.raw_text
            result.text = votes.raw_text if votes.status == "prediction" else ""
            result.status = votes.status
            result.tokens = [{"index": i, "text": ch, "uncertain": i in votes.uncertain_positions}
                             for i, ch in enumerate(votes.raw_text)]
            result.timings["method"] = votes.method
            result.timings["members"] = votes.members
        elif backend == "paddle":
            if not paddle_available():
                raise ImportError("Paddle environment missing; install requirements-paddle.txt in .venv-paddle")
            ok, buffer = cv2.imencode(".png", crop)
            if not ok:
                raise ValueError("Could not encode crop")
            data = _paddle_cached(buffer.tobytes())
            result.timings = dict(data.get("timings", {}))
            result.timings["encode_seconds"] = time.perf_counter() - start - result.timings.get("request_seconds", 0.0)
            result.raw_text = data["raw_text"]
            result.tokens = data["tokens"]
            result.confidence = data.get("confidence")
            result.text = "".join(ch for ch in result.raw_text.upper() if ch.isascii() and ch.isalnum())
        else:
            from main import recognize_characters
            from segmentation import segment_characters
            from dataset import post_process_plate_text
            segments = segment_characters(crop) if segments is None else segments
            raw, scores = recognize_characters([char for char, _ in segments], backend, models, True)
            result.raw_text = raw
            result.text = post_process_plate_text(raw, scores)
            result.tokens = [{"text": ch, "score": scores[i] if scores else None} for i, ch in enumerate(raw)]
        if backend != "ensemble":
            result.status = "prediction" if result.text else "unreadable"
    except Exception as exc:
        logging.getLogger(__name__).exception("OCR backend %s failed", backend)
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
    result.seconds = time.perf_counter() - start
    return result