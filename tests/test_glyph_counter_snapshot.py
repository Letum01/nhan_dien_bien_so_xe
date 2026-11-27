from unittest.mock import patch
from annotation_store import AnnotationStore
from glyph_progress import summarize


def test_progress_uses_supplied_snapshot_not_second_read(tmp_path):
    detection=AnnotationStore(tmp_path/'d.db');ocr=AnnotationStore(tmp_path/'o.db')
    row=dict(image_id='fixture',image_path='fixture.jpg',review_status='reviewed',split='test',plate_bbox=[0,0,50,50],
             source_group='g',characters=[],confirmed_by_user=True)
    with patch.object(ocr,'all_latest',side_effect=AssertionError('Unexpected second snapshot')):
        result=summarize(detection,ocr,reviewed_snapshot=[row])
    assert result['reviewed_plates']==1