import hashlib
import json

import cv2
import numpy as np

from annotation_store import AnnotationStore
from prepare_round_proposals import prepare


def test_proposal_generation_never_approves_or_assigns_source(tmp_path):
    (tmp_path/'reports').mkdir()
    (tmp_path/'dataset').mkdir()
    image=tmp_path/'dataset/test.jpg'
    cv2.imwrite(str(image),np.zeros((200,400,3),np.uint8))
    queue={'images':[{'image_path':'dataset/test.jpg','rank':1}]}
    (tmp_path/'reports/annotation_round_01.json').write_text(json.dumps(queue),encoding='utf-8')
    report=prepare(tmp_path)
    entry=report['results'][0]
    assert entry['human_reviewed'] is False
    assert entry['source_group'] is None and entry['split'] is None
    assert AnnotationStore(tmp_path/'data/annotations.sqlite3').all_latest()=={}
    assert (tmp_path/entry['preview']).exists()