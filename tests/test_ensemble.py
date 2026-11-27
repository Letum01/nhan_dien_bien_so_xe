import numpy as np
import pytest

from ensemble_recognition import CLASSES, vote, soft_vote


def test_majority_and_abstention():
    result = vote({"svm": "ABC", "rf": "ABD", "cnn": "AXE"})
    assert result.raw_text == "AB?"
    assert result.uncertain_positions == [2]
    assert result.status == "uncertain"


def test_equal_wrong_predictions_are_not_accuracy_proof():
    result = vote({name: "IIII" for name in ("svm", "rf", "cnn")})
    assert result.raw_text == "IIII"
    assert result.status == "prediction"  # agreement only, not correctness


def test_weighted_requires_validation():
    with pytest.raises(ValueError, match="validation"):
        vote({"svm": "A", "rf": "B", "cnn": "C"}, "weighted", {"svm": 1, "rf": 1, "cnn": 2})
    result = vote({"svm": "A", "rf": "B", "cnn": "C"}, "weighted",
                  {"svm": 2, "rf": 1, "cnn": 1}, "validation_manifest_hash")
    assert result.raw_text == "A"


def test_mismatched_segments_fail():
    with pytest.raises(ValueError, match="same segmented"):
        vote({"svm": "A", "rf": "AB", "cnn": "A"})


def test_soft_rejects_uncalibrated_and_aligns_labels():
    names = ("svm", "rf", "cnn")
    probabilities = {name: np.eye(36)[[10]] for name in names}
    orders = {name: CLASSES for name in names}
    weights = {name: 1 for name in names}
    with pytest.raises(ValueError, match="Calibration"):
        soft_vote(probabilities, orders, weights, None, "val")
    orders["cnn"] = tuple(reversed(CLASSES))
    probabilities["cnn"] = np.eye(36)[[25]]
    assert soft_vote(probabilities, orders, weights, "calibration", "val").raw_text == "A"


def test_product_pipeline_rejects_external_ocr_by_default():
    from main import end_to_end_pipeline
    with pytest.raises(ValueError, match="Product recognition"):
        end_to_end_pipeline(np.zeros((100, 200, 3), np.uint8), "paddle")


def test_product_ui_has_no_external_ocr():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
    assert not app.exception
    options = " ".join(app.selectbox[0].options).lower()
    assert not any(external in options for external in ("paddle", "easyocr", "tesseract"))
    assert "ensemble" in options


def test_syntax_is_not_used_to_rewrite_ensemble():
    from unittest.mock import patch
    from ocr_backends import recognize_crop
    predictions = {"svm": "B0", "rf": "B0", "cnn": "80"}
    with patch("main.recognize_characters", side_effect=lambda chars, name, models: predictions[name]):
        result = recognize_crop(np.zeros((20, 40, 3), np.uint8), "ensemble", {},
                                [(np.zeros((32, 32), np.uint8), (0,0,5,10))] * 2)
    assert result.raw_text == "B0" and result.text == "B0"
    assert result.confidence is None