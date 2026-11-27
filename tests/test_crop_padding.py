import pytest
from benchmark_crop_padding import padded_box,summarize


def test_padding_is_per_side_based_on_original_box_no_mutation():
    box=[20,30,100,40]
    assert padded_box(box,.1,(200,300,3))==(10,26,120,48)
    assert box==[20,30,100,40]
    assert padded_box(box,0,(200,300,3))==tuple(box)


def test_padding_clips_and_rounds_outwards():
    assert padded_box([0,0,10,10],.15,(11,12,3))==(0,0,12,11)
    with pytest.raises(ValueError):padded_box([1,2,0,4],.1,(20,20,3))
    with pytest.raises(ValueError):padded_box([1,2,3,4],-.1,(20,20,3))