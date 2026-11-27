import pytest
from run_ocr_mix_experiment01 import metrics


def test_metrics_accept_character_string_and_count_unknown_as_wrong():
    rows=[{'plate_id':'p','group':'g'}]*3
    result=metrics('A1B','A?B',rows)
    assert result['correct']==2 and result['samples']==3
    assert result['accuracy']==pytest.approx(2/3)


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError,match='counts'):metrics('AB','A',[{'plate_id':'p','group':'g'}]*2)