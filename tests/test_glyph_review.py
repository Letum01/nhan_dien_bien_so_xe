import copy

import pytest

from glyph_review import validate_glyph_plate,plates_from_detection,export_glyph_dataset
from annotation_store import AnnotationStore


def sample():
    return dict(plate_text='A-1',text_confirmed=True,unreadable=False,layout='one_line',source_group='g',split='train',
                characters=[dict(bbox=[5,5,15,30],label='A',confirmed=True,line=0,order=0),
                            dict(bbox=[30,5,10,30],label='1',confirmed=True,line=0,order=1)])


def test_glyph_text_order_and_class_validation():
    row=sample();assert validate_glyph_plate(row,100,50)=='A1'
    row['characters'][0]['label']='B'
    with pytest.raises(ValueError,match='mismatch'):validate_glyph_plate(row,100,50)
    assert row['plate_text']=='A-1'


def test_overlap_rejected_and_proposal_not_reviewed():
    row=sample();row['characters'][1]['bbox']=[6,5,15,30]
    with pytest.raises(ValueError,match='overlapping'):validate_glyph_plate(row,100,50)
    row=sample();row['characters'][0]['confirmed']=False
    with pytest.raises(ValueError,match='confirmation'):validate_glyph_plate(row,100,50)


def test_ocr_work_never_updates_detection_store(tmp_path):
    detection=AnnotationStore(tmp_path/'d.db');ocr=AnnotationStore(tmp_path/'o.db')
    original=dict(image_id='source',image_path='source.jpg',boxes=[[0,0,100,50]],source_group='g',split='test',
                  review_status='reviewed',confirmed_by_user=True,plates=[dict(plate_text='A1',text_confirmed=True)])
    detection.save(original,0);before=detection.all_latest()
    entries=plates_from_detection(detection)
    ocr.save_draft(entries[0],0)
    assert detection.all_latest()==before
    assert entries[0]['split']=='test' and not entries[0]['characters']
    with pytest.raises(ValueError,match='No Glyph reviewed'):export_glyph_dataset(ocr,detection,tmp_path/'export')
    assert not (tmp_path/'export').exists()


def test_reviewed_glyph_export_and_validator_fixture_only(tmp_path):
    import cv2
    import numpy as np
    import hashlib
    from validate_glyph_dataset import validate
    path=tmp_path/'fixture.png';cv2.imwrite(str(path),np.full((50,100,3),240,np.uint8))
    detection=AnnotationStore(tmp_path/'d.db');ocr=AnnotationStore(tmp_path/'o.db')
    identity=hashlib.sha256(path.read_bytes()).hexdigest()
    detection.save(dict(image_id=identity,image_path=str(path),boxes=[[0,0,100,50]],source_group='fixture',split='train',
                        review_status='reviewed',confirmed_by_user=True,plates=[]),0)
    parent=plates_from_detection(detection)[0]
    row={**parent,**sample(),'review_status':'reviewed','confirmed_by_user':True}
    row['source_group']=parent['source_group']
    ocr.save(row,0)
    assert export_glyph_dataset(ocr,detection,tmp_path/'out')==2
    result=validate(tmp_path/'out')
    assert result['glyphs']==2 and result['status']=='PASS'