import pytest

from dataset_validation import validate_boxes


def test_reviewed_one_pixel_plate_must_be_corrected_by_human():
    boxes=[[157,155,247,201],[586,42,1,1]]
    with pytest.raises(ValueError,match="human correction"):
        validate_boxes(boxes,600,480)
    assert boxes[1]==[586,42,1,1]