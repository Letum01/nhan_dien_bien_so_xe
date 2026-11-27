from unittest.mock import patch
from annotation_store import AnnotationStore
from glyph_progress import summarize


def test_validation_priority_never_changes_split_or_labels(tmp_path):
    detection=AnnotationStore(tmp_path/'d.db');ocr=AnnotationStore(tmp_path/'o.db')
    for name,split,group in [('a','train','train_group'),('b','val','new_group'),('c','test','test_group')]:
        detection.save(dict(image_id=name,image_path=name+'.jpg',boxes=[[0,0,100,50]],plates=[],
                             source_group=group,split=split,review_status='reviewed',confirmed_by_user=True),0)
    before=detection.all_latest()
    with patch('glyph_progress.fingerprint',return_value=('abc',1,(50,100,3))):
        report=summarize(detection,ocr)
    assert report['queue'][0]['split']=='val'
    assert report['queue'][0]['independent_of_reviewed_ocr_train'] is True
    assert detection.all_latest()==before
    assert ocr.all_latest()=={}
    assert report['reviewed_glyphs']==0
    assert len(report['missing_classes'])==36