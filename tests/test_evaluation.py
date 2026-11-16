from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from evaluate_labeled import edit_distance, evaluate_manifest, summarize_results


def test_empty_evaluation_is_unavailable_not_zero():
    report = summarize_results([])
    assert report["samples"] == 0
    assert report["plate_exact_match"] is None
    assert report["cer"] is None


def test_corpus_cer_is_length_weighted_and_can_exceed_one():
    rows = [
        dict(expected="12", character_errors=3, cer=1.5, seconds=1, exact_match=False, status="prediction"),
        dict(expected="12345678", character_errors=1, cer=0.125, seconds=1, exact_match=False, status="prediction"),
    ]
    assert summarize_results(rows)["cer"] == pytest.approx(0.4)
    assert edit_distance("ABC", "") == 3
    assert edit_distance("ABC", "AXBC") == 1


def test_unknown_counts_as_missing_plate_not_error_message(tmp_path):
    image = tmp_path / "image.png"
    cv2.imwrite(str(image), np.zeros((100, 200, 3), dtype=np.uint8))
    manifest = tmp_path / "labels.csv"
    manifest.write_text("image_path,plate_text,annotation_status\nimage.png,51A12345,human_reviewed\n", encoding="utf-8")
    with patch("evaluate_labeled.load_trained_models", return_value={"cnn": object()}), patch(
        "evaluate_labeled.end_to_end_pipeline", return_value=("UNKNOWN (Plate Not Found)", None, None, [])
    ):
        report = evaluate_manifest(manifest)
    assert report["cer"] == 1.0
    assert report["results"][0]["prediction"] == ""
    assert report["failed_samples"] == 1


def test_missing_images_remain_in_denominator(tmp_path):
    manifest = tmp_path / "labels.csv"
    manifest.write_text("image_path,plate_text,annotation_status\nabsent.png,51A12345,human_reviewed\n", encoding="utf-8")
    with patch("evaluate_labeled.load_trained_models", return_value={"cnn": object()}):
        report = evaluate_manifest(manifest)
    assert report["samples"] == 1
    assert report["cer"] == 1
    assert report["results"][0]["status"] == "image_error"


def test_unreviewed_annotations_do_not_load_models(tmp_path):
    manifest = tmp_path / "labels.csv"
    manifest.write_text("image_path,plate_text,annotation_status\nabsent.png,51A12345,manual_estimate\n", encoding="utf-8")
    with patch("evaluate_labeled.load_trained_models") as loader:
        report = evaluate_manifest(manifest)
    loader.assert_not_called()
    assert report["samples"] == 0
    assert report["skipped_unreviewed"] == 1
    assert report["results"] == []