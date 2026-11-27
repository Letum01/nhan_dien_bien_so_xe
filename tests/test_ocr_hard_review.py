import json
from pathlib import Path

from annotation_store import AnnotationStore
from ocr_hard_review import load_hard_queue
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def test_hard_queue_excludes_completed_without_deleting(tmp_path):
    detection=AnnotationStore(ROOT/'data/annotations.sqlite3');ocr=AnnotationStore(tmp_path/'ocr.db')
    queue=load_hard_queue(detection,ocr)
    assert queue['total']==28 and not queue['errors']
    row=queue['pending'][0]
    ocr.save({**row,'review_status':'reviewed','confirmed_by_user':True},0) # isolated fixture only
    result=load_hard_queue(detection,ocr)
    assert len(result['completed'])==1 and len(result['pending'])==27
    assert ocr.latest(row['image_id'])['review_status']=='reviewed'


def test_focused_hard_queue_ui_has_only_train(tmp_path,monkeypatch):
    monkeypatch.setenv('ALPR_GLYPH_DB',str(tmp_path/'test.db'))
    monkeypatch.setenv('ALPR_OCR_QUEUE','hard')
    app=AppTest.from_file(str(ROOT/'label_characters.py'),default_timeout=60).run()
    assert not app.exception
    choices=next(s for s in app.selectbox if s.label=='Biển cần gắn glyph')
    assert len(choices.options)==28
    assert all(' / train / ' in item for item in choices.options)
    assert not AnnotationStore(tmp_path/'test.db').all_latest()