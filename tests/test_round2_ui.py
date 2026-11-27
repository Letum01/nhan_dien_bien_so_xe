from pathlib import Path

from streamlit.testing.v1 import AppTest
from annotation_store import AnnotationStore
from round2_review import load_candidates,progress

ROOT=Path(__file__).resolve().parents[1]


def test_round2_exact_queue_and_no_automatic_assignment(tmp_path,monkeypatch):
    monkeypatch.setenv('ALPR_ANNOTATION_DB',str(tmp_path/'labels.db'))
    monkeypatch.setenv('ALPR_ANNOTATION_ROUND','2')
    monkeypatch.setenv('ALPR_ROUND_ONLY','0')
    app=AppTest.from_file(str(ROOT/'label_detection.py'),default_timeout=60).run()
    assert not app.exception
    candidates,errors=load_candidates()
    assert not errors and len(candidates)==120
    sources=next(w for w in app.selectbox if w.label=='Ảnh nguồn')
    assert set(sources.options)=={r['absolute_path'] for r in candidates}
    split=next(w for w in app.selectbox if w.label=='Split')
    assert split.value=='UNASSIGNED' and 'test' not in split.options
    assert len(app.metric)==5
    next(b for b in app.button if b.label=='Ảnh tiếp theo →').click().run()
    assert not app.exception
    store=AnnotationStore(tmp_path/'labels.db')
    assert store.all_latest()=={}


def test_round2_progress_draft_is_not_reviewed(tmp_path):
    candidates,_=load_candidates()
    store=AnnotationStore(tmp_path/'labels.db')
    row=candidates[0]
    store.save_draft(dict(image_id=row['sha256'],image_path=row['image_path'],review_status='proposed'),0)
    stats=progress(store,candidates)
    assert sum(r['status']=='draft' for r in stats)==1
    assert not any(r['status']=='reviewed' for r in stats)