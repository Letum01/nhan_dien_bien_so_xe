import json

import pytest
import cv2
import numpy as np

from final_ocr_dataset import conflicts, validate_manual
from yolo_character_dataset import validate


def test_replacement_preserves_original_and_rejects_duplicate(tmp_path, monkeypatch):
    from annotation_store import AnnotationStore
    from final_ocr_intake import upload, registered
    import final_ocr_dataset as final

    db=AnnotationStore(tmp_path/'test.sqlite3').path
    old=tmp_path/'old.png'
    assert cv2.imwrite(str(old),np.zeros((32,32,3),np.uint8))
    before=old.read_bytes()
    old_key=final.relative_image(old)
    original={'image':str(old),'sha256':final.sha(old),'phash':'0'*16,'status':'UNKNOWN'}
    monkeypatch.setattr(final,'audit_candidates',lambda path,write: {'rows':[original,*registered(path)],'known':[]})
    pixels=cv2.resize(np.random.default_rng(19).integers(0,256,(16,16,3),dtype=np.uint8),(160,160))
    data=cv2.imencode('.png',pixels)[1].tobytes()
    result=upload(db,'new.png',data,{'source_group':'','capture':'fixture'},old_key)
    assert result['replaces']==old_key and result['status']=='UNKNOWN'
    assert old.read_bytes()==before and not AnnotationStore(db).all_latest()
    with pytest.raises(ValueError,match='trùng SHA-256'):upload(db,'renamed.png',data,{})
    assert registered(db)==[result]


def test_group_hash_and_perceptual_conflicts():
    known=[{'image':'source.jpg','sha256':'original','phash':'0000000000000000','source_group':'Car-1'}]
    assert conflicts('original',2**64-1,'another',known)==['source.jpg']
    assert conflicts('new',63,'another',known)==['source.jpg']
    assert conflicts('new',127,'car-1',known)==['source.jpg']
    assert not conflicts('new',127,'another',known)


def test_explicit_source_evidence_and_manual_box_required():
    row=dict(split='test',cohort='final',plate_bbox=[0,0,50,30],plate_box_confirmed=True,
             characters=[dict(bbox=[5,5,10,20],label='A',confirmed=True,line=0)],plate_text='A',
             text_confirmed=True,layout='one_line',source_group='UNKNOWN',source_evidence='')
    with pytest.raises(ValueError,match='evidence'):validate_manual(row,(50,100))
    row.update(source_group='real-capture-1',source_evidence='Fixture capture and history checked',provenance_confirmed=True,
               confirmed_group='real-capture-1',confirmed_evidence='Fixture capture and history checked')
    validate_manual(row,(50,100))
    with pytest.raises(ValueError,match='evidence'):validate_manual({**row,'source_group':'edited'},(50,100))
    row['characters'][0]['bbox']=[45,5,10,20]
    with pytest.raises(ValueError,match='outside'):validate_manual(row,(50,100))


def test_training_snapshot_rejects_test_even_offline(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'schema':'reviewed-character-yolo-v1',
         'rows':[{'review':{'image_id':'test','split':'test'}}]}),encoding='utf-8')
    with pytest.raises(ValueError,match='exclude test'):validate(tmp_path,verify_live=False)