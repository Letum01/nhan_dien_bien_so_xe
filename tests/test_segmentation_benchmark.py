import cv2
import numpy as np

from segmentation_benchmark import match_boxes,experimental_segment,aggregate
from segmentation import segment_characters


def test_one_to_one_does_not_pair_by_index_or_double_count():
    truth=[[0,0,10,20],[30,0,10,20]]
    score=match_boxes([[30,0,10,20],[0,0,10,20],[0,0,10,20]],truth)
    assert (score['tp'],score['fp'],score['fn'])==(2,1,0)
    assert not score['complete_order_correct']
    assert len({m['gt'] for m in score['matches']})==2


def test_missing_predictions_are_fn_not_classifier_success():
    result=match_boxes([],[[0,0,10,20]])
    assert result['fn']==1 and not result['full_match']
    assert not result['complete_order_correct']


def test_experimental_baseline_matches_production():
    crop=np.full((100,320,3),235,np.uint8)
    cv2.putText(crop,'51A12345',(5,70),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,0),2)
    direct=segment_characters(crop);baseline,trace=experimental_segment(crop,'baseline')
    assert [b for _,b in direct]==[b for _,b in baseline]
    for (a,_),(b,_) in zip(direct,baseline):np.testing.assert_array_equal(a,b)


def test_border_removal_can_remove_edge_glyph_so_not_automatically_safe():
    crop=np.full((100,200,3),240,np.uint8)
    cv2.putText(crop,'A',(0,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),3)
    _,trace=experimental_segment(crop,'border_components')
    assert trace['binary'].shape==crop.shape[:2]