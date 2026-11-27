import csv
from io import StringIO

import cv2
import numpy as np
import pytest

from acceptance_pilot import edit_counts
from demo_alpr_core import History, csv_bytes, decode


def test_input_bounds_and_history(tmp_path):
    image=np.zeros((1300,2400,3),np.uint8)
    assert decode(cv2.imencode('.jpg',image)[1].tobytes()).shape==image.shape
    with pytest.raises(OSError):decode(b'bad PNG')
    h=History(tmp_path/'demo.sqlite3')
    h.save([dict(filename='=1+1.jpg',prediction='29A12345',algorithm='CNN',seconds=.2,status='OK')])
    rows=h.search('29a');assert len(rows)==1
    assert next(csv.DictReader(StringIO(csv_bytes(rows).decode('utf-8-sig'))))['filename']=="'=1+1.jpg"
    h.clear();assert not h.search()


def test_edit_breakdown_and_missing_plate():
    assert edit_counts('ABC','ADC')=={'substitutions':1,'deletions':0,'insertions':0}
    assert edit_counts('ABC','')=={'substitutions':0,'deletions':3,'insertions':0}
    assert edit_counts('ABC','AXBCC')['insertions']==2


def test_intake_date_does_not_pass_provenance_gate():
    from quick_ocr_test import require_evaluation_evidence
    with pytest.raises(ValueError,match='intake date'):
        require_evaluation_evidence([{'image':'a.jpg','evidence':'\nNgày nhập\t26/11/2026 '}])