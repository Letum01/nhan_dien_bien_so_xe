from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_unified_navigation_filter_and_dashboard(tmp_path, monkeypatch):
    monkeypatch.setenv('ALPR_ANNOTATION_ROUND','1')
    monkeypatch.setenv('ALPR_ROUND_ONLY','0')
    monkeypatch.setenv('ALPR_ANNOTATION_DB',str(tmp_path/'labels.db'))
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'label_detection.py'),default_timeout=60).run()
    assert not app.exception
    assert len(app.tabs)==4
    assert len(app.metric)==4
    next(b for b in app.button if b.label=='Ảnh tiếp theo →').click().run()
    assert not app.exception
    next(b for b in app.button if b.label=='← Ảnh trước').click().run()
    assert not app.exception
    next(t for t in app.text_input if t.label=='Tìm tên ảnh').set_value('0402').run()
    assert not app.exception
    assert all('0402' in value for box in app.selectbox if box.label=='Ảnh nguồn' for value in box.options)