import json
from pathlib import Path

from annotation_store import AnnotationStore
from ocr_train_queue import load_queue
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def test_frozen_train_queue_matches_parent_and_prioritizes_confirmed():
    document,entries=load_queue(AnnotationStore(ROOT/'data/annotations.sqlite3'))
    assert len(entries)==15
    assert all(r['split']=='train' for r in entries)
    assert all(r['text_confirmed'] for r in entries[:7])
    assert all(not r['text_confirmed'] for r in entries[7:])
    assert not document['reviewed_by_agent']


def test_train_ui_queue_and_draft_reload(tmp_path,monkeypatch):
    database=tmp_path/'ocr.db'
    monkeypatch.setenv('ALPR_GLYPH_DB',str(database))
    monkeypatch.setenv('ALPR_OCR_QUEUE','train')
    app=AppTest.from_file(str(ROOT/'label_characters.py'),default_timeout=60).run()
    assert not app.exception
    chooser=next(s for s in app.selectbox if s.label=='Biển cần gắn glyph')
    assert len(chooser.options)==15
    assert all(' / train / ' in option for option in chooser.options)
    store=AnnotationStore(database)
    assert store.all_latest()=={}
    app.run()
    assert not app.exception


def test_reviewed_payload_survives_close_reopen(tmp_path):
    path=tmp_path/'ocr.db';store=AnnotationStore(path)
    payload={'image_id':'fixture','image_path':'fixture.png','review_status':'reviewed',
             'confirmed_by_user':True,'source_group':'fixture-group','split':'train',
             'characters':[{'bbox':[1,2,10,20],'label':'A','confirmed':True,'line':0,'order':0}]}
    saved=store.save(payload,0)
    reopened=AnnotationStore(path)
    assert reopened.latest('fixture')==saved
    assert reopened.latest('fixture')['characters']==payload['characters']