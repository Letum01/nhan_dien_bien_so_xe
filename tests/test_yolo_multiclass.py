import pytest

from multiclass_dataset import labels
from multiclass_run import class_geometry, edit_distance
from segmentation_benchmark import match_boxes


def test_multiclass_label_mapping_and_strict_count():
    box = (.5, .5, .25, .25)
    assert [row.split()[0] for row in labels('09AZ', [box]*4).splitlines()] == ['0', '9', '10', '35']
    with pytest.raises(ValueError):
        labels('09', [box])


def test_wrong_class_is_localization_tp_but_class_aware_fp_and_fn():
    box = [0, 0, 10, 10]
    assert match_boxes([box], [box])['tp'] == 1
    total, per_class = class_geometry([{'bbox': box, 'label': '0'}], [box], 'D')
    assert total == {'tp': 0, 'fp': 1, 'fn': 1}
    assert per_class['0'] == [0, 1, 0]
    assert per_class['D'] == [0, 0, 1]
    assert edit_distance('D', '') == 1