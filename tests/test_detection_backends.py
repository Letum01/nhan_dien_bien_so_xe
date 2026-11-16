import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from detection_backends import detect_plates, recognize_multiple
from evaluate_detection import match_boxes
from label_detection import export_yolo
from localization import plate_border_proposals, _box_iou
from yolo_detector import load_plate_yolo

ROOT = Path(__file__).resolve().parents[1]


def test_0405_border_contains_full_plate():
    image = cv2.imread(str(ROOT / "dataset/license_plate_dataset/0/images/0405.jpg"))
    proposals = plate_border_proposals(image)
    # Development-only visual box; not a human reviewed benchmark annotation.
    expected = (193, 165, 162, 114)
    assert any(_box_iou(box, expected) > .75 for _, box, _ in proposals)


def test_yolo_missing_weights_never_downloads_generic_model():
    with pytest.raises(FileNotFoundError):
        load_plate_yolo(str(ROOT / "saved_models/nonexistent-plate.pt"))
    with pytest.raises(ValueError, match="weights"):
        detect_plates(np.zeros((100,200,3),np.uint8), "yolo")


def test_multi_plate_contract_preserves_two_results():
    image=np.zeros((100,200,3),np.uint8)
    proposals=[(image[:40,:80],(0,0,80,40),.7),(image[50:90,100:180],(100,50,80,40),.8)]
    from ocr_backends import OCRResult
    with patch("detection_backends.detect_plates", return_value=proposals), patch(
        "ocr_backends.recognize_crop", return_value=OCRResult("paddle",raw_text="test",text="test",status="prediction")
    ):
        results=recognize_multiple(image)
    assert len(results)==2
    assert results[0]["candidate_status"] == "unverified_proposal"


def test_detection_matching_no_double_credit():
    tp,fp,fn,ious=match_boxes([(0,0,20,20),(0,0,20,20)],[(0,0,20,20)])
    assert (tp,fp,fn)==(1,1,0)
    assert ious==[1.0]


def test_yolo_export_refuses_unreviewed_or_missing_splits(tmp_path):
    manifest=tmp_path/"annotations.json"
    manifest.write_text(json.dumps([]),encoding="utf-8")
    with pytest.raises(ValueError,match="train, val and test"):
        export_yolo(manifest,tmp_path/"export")
    assert not (tmp_path/"export").exists()