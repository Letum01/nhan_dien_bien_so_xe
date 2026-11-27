import csv
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from annotation_store import AnnotationStore, image_id
from annotation_ui import export_snapshot
from validate_yolo_dataset import validate_dataset


def test_synthetic_only_review_export_validate_roundtrip(tmp_path):
    """Fixture labels are never copied into production data; no accuracy claim."""
    store=AnnotationStore(tmp_path/'test.db')
    rng=np.random.default_rng(42)
    for i,split in enumerate(('train','val','test')):
        image=rng.integers(0,256,(120,240,3),dtype=np.uint8)
        path=tmp_path/f'fixture{i}.jpg';cv2.imwrite(str(path),image)
        row=dict(image_id=image_id(path),image_path=str(path),source_group=f'fixture_{i}',split=split,
                 boxes=[[20,20,100,70]],review_status='reviewed',confirmed_by_user=True,human_reviewed=True,
                 plates=[dict(plate_text='A',text_confirmed=True,unreadable=False,layout='one_line',
                              characters=[dict(bbox=[10,10,20,40],label='A',confirmed=True,line=0,order=0)])])
        store.save_draft({**row,'review_status':'needs_review'},0)
        store.save(row,0)
    destination=tmp_path/'fixture_export'
    export_snapshot(store,destination)
    result=validate_dataset(destination/'license_plates')
    assert result['status']=='PASS'
    assert result['images_by_split']=={'train':1,'val':1,'test':1}
    with (destination/'reviewed_crops.csv').open(encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==3 and all(r['plate_text']=='A' for r in rows)
    glyphs=list((destination/'glyphs').glob('*/characters_to_review.csv'))
    assert len(glyphs)==3
    for manifest in glyphs:
        with manifest.open(encoding='utf-8-sig') as stream:
            row=next(csv.DictReader(stream))
        assert row['class_label']=='A' and row['line']=='0'
        assert cv2.imread(str(manifest.parent/row['image_path']),0).shape==(32,32)