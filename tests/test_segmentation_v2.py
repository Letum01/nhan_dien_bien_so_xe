import cv2
import numpy as np

from segmentation_v2 import segment_v2,VARIANTS


def plate(two_rows=False):
    image=np.full((150,320,3),235,np.uint8)
    cv2.rectangle(image,(2,2),(317,147),(0,0,0),2)
    cv2.putText(image,'51A' if two_rows else '51A12345',(8,58 if two_rows else 95),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,0),2)
    if two_rows:cv2.putText(image,'12345',(8,125),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,0),2)
    return image


def test_frame_mask_does_not_erase_entire_border_component():
    image=plate();original=image.copy()
    _,trace=segment_v2(image,'v2_a')
    assert np.count_nonzero(trace['frame_mask'])>0
    assert np.count_nonzero(trace['binary'])>100
    np.testing.assert_array_equal(image,original)


def test_thin_glyph_and_row_contracts():
    for two in (False,True):
        image=plate(two)
        for variant in VARIANTS:
            seg,trace=segment_v2(image,variant)
            assert trace['frame_mask'].shape==image.shape[:2]
            assert all(char.shape==(32,32) for char,_ in seg)
    image=np.full((90,180,3),235,np.uint8)
    cv2.putText(image,'11',(0,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),2)
    _,trace=segment_v2(image,'v2_b')
    assert np.count_nonzero(trace['binary'])>0