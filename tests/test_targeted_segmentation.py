import cv2
import numpy as np
from segmentation_train_ablation import variant_segments


def test_targeted_variants_never_read_ground_truth_and_preserve_source():
    image=np.full((100,300,3),235,np.uint8)
    cv2.putText(image,'51A12345',(5,70),cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,0,0),2)
    original=image.copy()
    for variant in ('frozen_border','protected_border','geometry','projection_split','row_sort'):
        segments,trace=variant_segments(image,variant)
        assert trace['binary'].shape==image.shape[:2]
        assert all(c.shape==(32,32) and c.dtype==np.uint8 for c,_ in segments)
    np.testing.assert_array_equal(image,original)