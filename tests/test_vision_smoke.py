import cv2
import numpy as np
import pytest

from localization import locate_license_plate, locate_license_plates, preprocess_image
from segmentation import preprocess_plate, segment_characters
from image_io import decode_uploaded_image


@pytest.mark.parametrize("shape", [(120, 300), (120, 300, 3)])
def test_segmentation_handles_single_and_two_line_layouts(shape):
    plate = np.full(shape, 235, dtype=np.uint8)
    if len(shape) == 2:
        cv2.putText(plate, "51A", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 10, 2)
        cv2.putText(plate, "12345", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 10, 2)
    else:
        cv2.putText(plate, "51A", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
        cv2.putText(plate, "12345", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    result = segment_characters(plate)
    assert len(result) >= 6


def test_invalid_and_blank_images_fail_safely():
    assert locate_license_plate(None) == (None, None)
    assert locate_license_plates(np.zeros((0, 0, 3), dtype=np.uint8)) == []
    assert segment_characters(np.zeros((0, 0, 3), dtype=np.uint8)) == []
    assert segment_characters(np.zeros((120, 300, 3), dtype=np.uint8)) == []


def test_preprocess_accepts_gray_and_bgr():
    gray = np.zeros((120, 300), dtype=np.uint8)
    assert preprocess_image(gray).shape == gray.shape
    assert preprocess_plate(gray).shape == gray.shape
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert preprocess_image(bgr).shape == gray.shape


def test_segment_synthetic_one_line_characters():
    plate = np.full((90, 320, 3), 235, dtype=np.uint8)
    cv2.putText(plate, "51A12345", (8, 67), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                (15, 15, 15), 3, cv2.LINE_AA)
    chars = segment_characters(plate)
    assert 6 <= len(chars) <= 10
    assert all(char.shape == (32, 32) for char, _ in chars)


def test_multi_proposal_contract():
    image = np.zeros((240, 640, 3), dtype=np.uint8)
    assert locate_license_plates(image) == []


def test_uploaded_jpeg_png_and_corrupt_bytes():
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    for extension in (".jpg", ".png"):
        ok, encoded = cv2.imencode(extension, image)
        assert ok
        decoded = decode_uploaded_image(encoded.tobytes())
        assert decoded is not None and decoded.shape == image.shape
        assert decoded[20, 20, 2] > decoded[20, 20, 0]
    assert decode_uploaded_image(b"not an image") is None
    assert decode_uploaded_image(b"") is None