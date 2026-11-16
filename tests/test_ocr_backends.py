import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from label_crops import save_label
from ocr_backends import recognize_crop
from paddle_worker import ordered_tokens


def test_paddle_missing_dependency_is_explicit():
    with patch("ocr_backends.paddle_available", return_value=False):
        result = recognize_crop(np.zeros((40, 80, 3), np.uint8), "paddle")
    assert result.status == "error"
    assert "environment missing" in result.error
    assert not result.text


def test_paddle_keeps_raw_punctuation_and_no_glyph_substitution():
    with patch("ocr_backends.paddle_available", return_value=True), patch(
        "ocr_backends._paddle_cached", return_value={"raw_text": "83-FD\n99B2", "tokens": [], "confidence": .8}
    ):
        result = recognize_crop(np.zeros((40, 80, 3), np.uint8), "paddle")
    assert result.raw_text == "83-FD\n99B2"
    assert result.text == "83FD99B2"
    assert result.confidence == .8


def test_paddle_no_text_is_unreadable():
    with patch("ocr_backends.paddle_available", return_value=True), patch(
        "ocr_backends._paddle_cached", return_value={"raw_text": "", "tokens": [], "confidence": None}
    ):
        assert recognize_crop(np.zeros((40, 80, 3), np.uint8), "paddle").status == "unreadable"


def test_token_order_two_rows_and_same_row():
    polygons = [[[20, 50], [60, 50], [60, 70], [20, 70]],
                [[30, 5], [60, 5], [60, 25], [30, 25]],
                [[0, 5], [25, 5], [25, 25], [0, 25]]]
    text, tokens = ordered_tokens(["9972", "FD", "83"], [.9] * 3, polygons)
    assert text == "83 FD\n9972"
    assert len(tokens) == 3


def test_label_upsert_and_group_isolation(tmp_path):
    path = tmp_path / "labels.csv"
    row = dict(image_path="a.png", plate_text="83FD9972", annotation_status="human_reviewed", source_group="vehicle1", split="test")
    save_label(path, row)
    save_label(path, row)
    assert len(list(csv.DictReader(path.open(encoding="utf-8-sig")))) == 1
    with pytest.raises(ValueError, match="same split"):
        save_label(path, {**row, "image_path": "b.png", "split": "train"})


def test_old_model_missing_is_error_not_fake_prediction():
    result = recognize_crop(np.zeros((40, 80, 3), np.uint8), "cnn", None,
                            [(np.zeros((32, 32), np.uint8), (0, 0, 20, 20))])
    assert result.status == "error" and not result.raw_text