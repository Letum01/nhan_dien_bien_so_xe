import cv2
import numpy as np
import pytest

from segmentation import preprocess_plate,segment_characters


def test_trace_does_not_change_production_segmentation():
    image=np.full((100,360,3),240,np.uint8)
    cv2.putText(image,'51A12345',(8,70),cv2.FONT_HERSHEY_SIMPLEX,1.6,(10,10,10),3)
    baseline=segment_characters(image)
    trace={}
    traced=segment_characters(image,trace=trace)
    assert [b for _,b in baseline]==[b for _,b in traced]
    for (a,_),(b,_) in zip(baseline,traced):np.testing.assert_array_equal(a,b)
    assert len(trace['contours'])>=len(trace['final_boxes'])
    assert all('reasons' in c and 'status' in c for c in trace['contours'])


def test_binary_override_is_explicit_experiment_and_validated():
    image=np.zeros((60,120,3),np.uint8)
    trace={}
    assert segment_characters(image,trace=trace,binary_override=np.zeros((60,120),np.uint8))==[]
    with pytest.raises(ValueError,match='matching'):
        segment_characters(image,binary_override=np.zeros((30,30),np.uint8))


def test_invalid_channel_shape_does_not_silently_look_like_no_glyphs():
    with pytest.raises(ValueError,match='grayscale or BGR'):
        segment_characters(np.zeros((60,120,2),np.uint8))