import pytest
from quick_ocr_test import verify_text, freeze
from final_ocr_dataset import check_final_conflicts


def test_final_conflicts_distinguishes_self_peers_and_exposure():
    item=dict(image='a.jpg',sha256='a',phash='0000000000000000',source_group='may')
    self_ref={**item,'split':'test','final_peer':True}
    peer={**self_ref,'image':'b.jpg','sha256':'b','phash':'ffffffffffffffff'}
    check_final_conflicts(item,'may',[self_ref,peer])
    with pytest.raises(ValueError,match=r'source_group.*b.jpg.*train.*may'):
        check_final_conflicts(item,'may',[{**peer,'split':'train'}])
    with pytest.raises(ValueError,match=r'SHA-256.*b.jpg'):
        check_final_conflicts(item,'may',[{**peer,'sha256':'a'}])
    with pytest.raises(ValueError,match=r'pHash gần trùng \(2 bit'):
        check_final_conflicts(item,'may',[{**peer,'phash':'3'}])
    with pytest.raises(ValueError,match=r'SHA-256.*a.jpg.*val'):
        check_final_conflicts(item,'may',[{**item,'split':'val'}])


def test_text_gate_rejects_unconfirmed_or_duplicate():
    e=dict(image='sample.jpg',sha256='digest',phash='ffffffffffffffff',status='UNKNOWN')
    evidence='Confirmed original capture and provenance record'
    row=dict(image_id='text:digest',image_path='sample.jpg',source_sha256='digest',source_phash=e['phash'],
             label_scope='TEXT_REVIEWED',review_status='reviewed',confirmed_by_user=True,split='test',plate_text='30A12345',
             text_confirmed=True,source_group='fresh',confirmed_group='fresh',source_evidence=evidence,
             confirmed_evidence=evidence,provenance_confirmed=True)
    verify_text(row,e,[])
    with pytest.raises(ValueError):verify_text({**row,'text_confirmed':False},e,[])
    with pytest.raises(ValueError):verify_text(row,e,[{**e,'source_group':'old'}])


def test_pending_freeze_does_not_write_file(tmp_path,monkeypatch):
    monkeypatch.setattr('quick_ocr_test.resolved',lambda: ([],[],['pending']))
    output=tmp_path/'freeze.json'
    with pytest.raises(ValueError,match='BLOCKED'):freeze(output)
    assert not output.exists()


def test_propose_rejects_non_development_before_worker(monkeypatch):
    from development_autolabel import propose
    monkeypatch.setattr('development_autolabel.pool',lambda: ([],[]))
    monkeypatch.setattr('development_autolabel.check_protocol',lambda:None)
    with pytest.raises(ValueError,match='allowlist'):list(propose([{'image':'test.jpg'}],None,None))