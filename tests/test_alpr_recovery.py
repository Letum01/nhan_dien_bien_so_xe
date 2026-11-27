import cv2
import numpy as np

from alpr_recovery import VARIANTS, pixels, stable_candidate
from yolo_character_pipeline import normalize_glyph


def test_pixel_contract_and_baseline_equivalence():
    roi=np.full((35,19,3),240,np.uint8)
    cv2.line(roi,(9,4),(9,30),(10,10,10),2)
    assert np.array_equal(pixels(roi,'raw'),normalize_glyph(roi))
    for name in VARIANTS:
        result=pixels(roi,name)
        assert result.shape==(32,32) and result.dtype==np.uint8
        assert set(np.unique(result))<={0,255}


def test_stability_does_not_force_disagreeing_models():
    baseline={'svm':'D','rf':'D','cnn':'D'}
    ambiguous={'svm':'D','rf':'D','cnn':'0'}
    assert stable_candidate('D',[baseline,ambiguous,ambiguous,ambiguous])==('D',[1])
    supported={**ambiguous,'rf':'0'}
    assert stable_candidate('D',[baseline,supported,supported,supported])==('0',[1])
    assert stable_candidate('',[])==('',[])