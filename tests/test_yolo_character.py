import cv2
import numpy as np
import pytest

from benchmark_yolo_character import score, summarize
from yolo_character_dataset import integer_boxes, normalized
from yolo_character_pipeline import normalize_glyph, order_boxes, segment


def test_xywh_conversion_and_bounds():
    assert normalized([0,5,10,10],20,20)==(.25,.5,.5,.5)
    with pytest.raises(ValueError): integer_boxes([[15,0,6,6]],20,20)
    with pytest.raises(ValueError): integer_boxes([[1.5,0,6,6]],20,20)


def test_row_order_and_polarity():
    a,b,c,d=[{'bbox':box} for box in ([13,4,4,12],[4,5,4,12],[4,27,4,12],[13,27,4,12])]
    assert order_boxes([a,c,d,b])==[b,a,c,d]
    assert order_boxes([a,b])==[b,a]
    roi=np.full((64,12,3),255,np.uint8); cv2.rectangle(roi,(4,8),(7,55),(0,0,0),-1)
    glyph=normalize_glyph(roi)
    assert glyph.shape==(32,32) and 0<np.count_nonzero(glyph)<200
    np.testing.assert_array_equal(glyph,normalize_glyph(255-roi))


def test_match_uses_geometry_and_missing_detection_counts_as_error():
    row={'plate_text':'12','characters':[{'label':'1'},{'label':'2'}]}
    gt=[[1,1,5,10],[8,1,5,10]]
    p={'glyph_boxes':list(reversed(gt)), 'text':dict.fromkeys(('svm','rf','cnn','ensemble'),'21')}
    result=score(p,[0,0,20,20],gt,row)
    assert result['geometry']['tp']==2 and not result['geometry']['complete_order_correct']
    assert result['ocr']['cnn']['matched_correct']==2 and not result['ocr']['cnn']['exact']
    empty=score({'glyph_boxes':[],'text':dict.fromkeys(p['text'],'')},None,gt,row)
    assert summarize([empty])['ocr']['cnn']['cer']==1 and empty['geometry']['fn']==2


def test_character_path_with_injected_detections(monkeypatch):
    monkeypatch.setattr('yolo_character_pipeline.character_boxes',lambda crop, weights:[{'bbox':[2,3,10,12]}])
    glyphs=segment(np.full((20,30,3),200,np.uint8),'character_yolo','fixture')
    assert glyphs[0][0].shape==(32,32) and glyphs[0][1]==[2,3,10,12]