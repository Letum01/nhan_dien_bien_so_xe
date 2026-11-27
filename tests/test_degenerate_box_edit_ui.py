import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from annotation_store import AnnotationStore,image_id


def test_user_can_remove_invalid_box_without_changing_reviewed_revision(tmp_path,monkeypatch):
    monkeypatch.setenv('ALPR_ANNOTATION_ROUND','1')
    root=Path(__file__).resolve().parents[1]
    path=tmp_path/'fixture.jpg'
    cv2.imwrite(str(path),np.full((100,200,3),200,np.uint8))
    database=tmp_path/'test.db'
    store=AnnotationStore(database)
    row=dict(image_id=image_id(path),image_path=os.path.relpath(path,root),
             boxes=[[20,20,100,50],[180,10,1,1]],plates=[],source_group='fixture',
             split='test',review_status='reviewed',confirmed_by_user=True)
    store.save(row,0)
    monkeypatch.setenv('ALPR_ANNOTATION_DB',str(database))
    monkeypatch.setenv('ALPR_ANNOTATION_SOURCE',str(tmp_path))
    monkeypatch.setenv('ALPR_ROUND_ONLY','0')
    app=AppTest.from_file(str(root/'label_detection.py'),default_timeout=60).run()
    assert not app.exception
    widget=next(w for w in app.multiselect if w.label.startswith('Xóa box theo index'))
    widget.set_value([1]).run()
    next(b for b in app.button if b.label=='Xóa chính xác các box đã chọn').click().run()
    assert not app.exception
    assert store.latest(row['image_id'])['boxes']==row['boxes']
    assert store.draft(row['image_id'])[0]['boxes']==[[20,20,100,50]]
    assert store.draft(row['image_id'])[0]['review_status']=='needs_review'