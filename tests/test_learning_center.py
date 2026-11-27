import hashlib
import sqlite3

import cv2
import numpy as np
import pytest

from learning_center import Feedback, alignment, dashboard, decode_upload


def fixture(seed=42):
    im=np.random.default_rng(seed).integers(0,256,(70,150,3),np.uint8)
    return cv2.imencode('.png',im)[1].tobytes()


def metadata(split='train'):
    return dict(source='synthetic test fixture',group='fixture-group',rights='research_only',split=split)


def test_opt_in_and_duplicate_transaction(tmp_path):
    db=Feedback(tmp_path/'test.sqlite3'); im=fixture()
    for data,consent in [(im,False),(b'broken',True)]:
        with pytest.raises((ValueError,OSError)):db.add(data,metadata=metadata(),consent=consent)
    assert not db.samples()
    db.add(im,metadata=metadata(),consent=True)
    with pytest.raises(ValueError,match='SHA-256'):db.add(im,metadata=metadata(),consent=True)
    with pytest.raises(ValueError,match='Source group'):db.add(fixture(24),metadata=metadata('test'),consent=True)
    assert len(db.samples())==1


def test_protected_reference_and_near_duplicate(tmp_path):
    im=fixture(); _,digest,ph=decode_upload(im)
    for reff,match in [({'sha256':digest},'SHA-256'),({'sha256':'other','phash':ph},'pHash'),
                        ({'sha256':'other','group':'fixture-group','protected':True},'Source group')]:
        db=Feedback(tmp_path/'test.sqlite3',[reff])
        with pytest.raises(ValueError,match=match):db.add(im,metadata=metadata(),consent=True)
        assert not db.samples()


def test_alignment_revision_export_delete_and_gate(tmp_path):
    db=Feedback(tmp_path/'test.sqlite3'); key=db.add(fixture(),metadata=metadata(),consent=True)
    raw=dict(plate_count=1,plate_score=.8,glyphs=[dict(score=.8,foreground=.2,bbox=[4,6,9,15],top_k={k:['A','B','C'] for k in ('cnn','svm','rf')})])
    db.set_prediction(key,raw)
    assert alignment('B',raw)[0]=='ALIGNMENT_REJECTED' # count and top-3 alone are insufficient
    assert db.confirm(key,'A',confirmed=True)=='AUTO_ALIGNED'
    assert db.confirm(key,'B',confirmed=True)=='ALIGNMENT_REJECTED'
    db=Feedback(db.path); assert dashboard(db.samples())['confirmed_text']==1
    with pytest.raises(ValueError): db.export(key)
    data=db.export(key,allowed=True); assert data['sample']['raw']==raw
    assert [r['text'] for r in data['sample']['revisions']]==[None,'A','B'] and not data['training_allowed']
    with pytest.raises(ValueError,match='BLOCKED'): db.training_snapshot()
    with pytest.raises(ValueError):db.delete(key)
    db.delete(key,confirmed=True)
    with sqlite3.connect(db.path) as con:
        assert con.execute('SELECT count(*) FROM revisions').fetchone()[0]==0
        assert con.execute('PRAGMA integrity_check').fetchone()[0]=='ok'


def test_planned_test_is_blind(tmp_path):
    db=Feedback(tmp_path/'test.sqlite3'); data=fixture(); key=db.add(data,metadata=metadata('test'),consent=True)
    assert key==hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError):db.set_prediction(key,{'text':'A'})
    assert db.confirm(key,'65-H1.12345',confirmed=True)=='TEXT_CONFIRMED'
    assert db.confirm(key,'',skip=True)=='AMBIGUOUS'
    assert db.samples()[0]['raw']=={}