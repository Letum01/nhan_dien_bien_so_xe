from unittest.mock import patch
import hashlib

import cv2
import numpy as np

from audit_recent_annotations import inspect_payload


def test_draft_never_becomes_export_eligible(tmp_path):
    path=tmp_path/'source.jpg';cv2.imwrite(str(path),np.zeros((60,100,3),np.uint8))
    payload={'image_path':str(path),'image_id':hashlib.sha256(path.read_bytes()).hexdigest(),
             'boxes':[[5,5,40,20]],'source_group':'g','split':'train','confirmed_by_user':True,
             'review_status':'reviewed','plates':[]}
    result=inspect_payload(payload,'draft')
    assert result['status']=='NEEDS_REVIEW'
    assert not result['detection_export_eligible']


def test_invalid_box_is_fail_not_auto_corrected(tmp_path):
    path=tmp_path/'source.jpg';cv2.imwrite(str(path),np.zeros((60,100,3),np.uint8))
    payload={'image_path':str(path),'image_id':hashlib.sha256(path.read_bytes()).hexdigest(),
             'boxes':[[90,5,40,20]],'source_group':'','split':'UNASSIGNED'}
    result=inspect_payload(payload,'draft')
    assert result['status']=='FAIL'
    assert result['boxes']==payload['boxes']