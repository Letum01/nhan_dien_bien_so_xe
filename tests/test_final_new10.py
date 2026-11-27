import pytest

from app_ocr_test_dataset import apply_plate_box, confirmed_source


def test_box_edit_resets_only_working_annotation():
    row=dict(plate_bbox=[1,2,10,12],characters=[{'label':'A'}],plate_text='A',
             text_confirmed=True,plate_box_confirmed=True)
    apply_plate_box(row,[1,2,10,12],30,30)
    assert row['characters']
    with pytest.raises(ValueError):apply_plate_box(row,[20,20,20,20],30,30)
    assert row['characters']
    apply_plate_box(row,[2,4,10,12],30,30)
    assert not row['characters'] and not row['plate_box_confirmed'] and not row['text_confirmed']


def test_source_count_requires_fresh_evidence_and_no_known_group():
    evidence='Fixture with a separately documented capture history'
    row=dict(source_group='new',source_evidence=evidence,confirmed_group='new',
             confirmed_evidence=evidence,provenance_confirmed=True)
    assert confirmed_source(row,[])
    assert not confirmed_source(row,['new'])
    assert not confirmed_source({**row,'source_evidence':evidence+' changed'},[])