import cv2
import numpy as np
import pytest

from main import end_to_end_pipeline, load_trained_models
from localization import locate_license_plates
from segmentation import segment_characters


@pytest.fixture(scope="module")
def loaded_models():
    models = load_trained_models("saved_models")
    assert models is not None
    return models


def make_plate(two_lines=False, dark=False, tilt=0):
    plate = np.full((150 if two_lines else 90, 360, 3), 225, dtype=np.uint8)
    if two_lines:
        cv2.putText(plate, "51A", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (10, 10, 10), 2)
        cv2.putText(plate, "12345", (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (10, 10, 10), 2)
    else:
        cv2.putText(plate, "51A12345", (12, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (10, 10, 10), 3)
    if dark:
        plate = (plate.astype(np.float32) * 0.25).astype(np.uint8)
    if tilt:
        matrix = cv2.getRotationMatrix2D((plate.shape[1] / 2, plate.shape[0] / 2), tilt, 1)
        plate = cv2.warpAffine(plate, matrix, (plate.shape[1], plate.shape[0]), borderValue=(40, 40, 40))
    return plate


@pytest.mark.parametrize("model_name", ["cnn", "svm", "rf"])
def test_pipeline_runs_each_model(model_name, loaded_models):
    result, annotated, crop, chars = end_to_end_pipeline(make_plate(), model_name, loaded_models)
    assert isinstance(result, str) and result
    assert annotated.shape == make_plate().shape
    assert crop is None or crop.size > 0
    assert isinstance(chars, list)


@pytest.mark.parametrize("image", [
    make_plate(), make_plate(two_lines=True), make_plate(dark=True), make_plate(tilt=8),
    np.zeros((360, 640, 3), dtype=np.uint8),
])
def test_pipeline_failure_modes_return_structured_results(image, loaded_models):
    result, annotated, crop, chars = end_to_end_pipeline(image, "svm", loaded_models)
    assert isinstance(result, str) and result
    assert annotated.shape == image.shape
    assert isinstance(chars, list)


def test_missing_models_fails_explicitly_on_plate_roi():
    with pytest.raises(ValueError, match="Mô hình"):
        end_to_end_pipeline(make_plate(), "svm", None)


def test_bad_path_fails_with_clear_error():
    with pytest.raises(ValueError, match="mở ảnh"):
        end_to_end_pipeline("missing-image-that-does-not-exist.jpg", "svm", None)


@pytest.mark.parametrize("name,expected_box,observed_components", [
    ("0402", (203, 185, 157, 115), 9),
    ("0403", (204, 200, 181, 131), 9),
])
def test_user_supplied_two_row_plate_regression(name, expected_box, observed_components):
    """Check box coverage and current component count, NOT OCR correctness.

    0403 has eight visible characters but nine components; that is a known
    segmentation defect, not evidence of successful character extraction.
    """
    image = cv2.imread(f"dataset/license_plate_dataset/0/images/{name}.jpg")
    assert image is not None
    proposals = locate_license_plates(image, max_plates=30)
    assert proposals
    crop, box, _ = max(proposals, key=lambda item: _iou(item[1], expected_box))
    assert _iou(box, expected_box) >= 0.65
    assert len(segment_characters(crop)) == observed_components


def test_production_trace_matches_returned_characters(loaded_models):
    trace = {"stale": True}
    image = cv2.imread("dataset/license_plate_dataset/0/images/0402.jpg")
    text, _, crop, chars = end_to_end_pipeline(image, "cnn", loaded_models, trace=trace)
    assert "stale" not in trace
    assert trace["processed_text"] == text
    assert len(trace["raw_text"]) == len(chars) == len(trace["segments"])
    assert trace["selected_box"] is not None
    assert crop is not None
    for actual, (recorded, _) in zip(chars, trace["segments"]):
        np.testing.assert_array_equal(actual, recorded)


def test_trace_is_reset_for_not_found_image(loaded_models):
    trace = {"raw_text": "old result"}
    text, _, crop, chars = end_to_end_pipeline(
        np.zeros((360, 640, 3), np.uint8), "cnn", loaded_models, trace=trace
    )
    assert text.startswith("UNKNOWN")
    assert trace["status"] == "not_found"
    assert trace["raw_text"] == ""
    assert trace["selected_box"] is None
    assert crop is None and not chars


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter) if inter else 0.0