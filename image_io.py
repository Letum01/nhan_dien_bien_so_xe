"""Validated image decoding shared by the Streamlit UI and tests."""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def decode_uploaded_image(data: bytes) -> Optional[np.ndarray]:
    """Decode supported image bytes to BGR, returning None for invalid files."""
    if not data:
        return None
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        logger.warning("Uploaded file could not be decoded as an image")
        return None
    return image