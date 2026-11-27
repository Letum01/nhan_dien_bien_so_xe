import copy
import sqlite3

import pytest
from annotation_store import AnnotationStore


def fixture(tmp_path):
    store=AnnotationStore(tmp_path/'a.db')
    for i,split in enumerate(('val','val','test')):
        store.save(dict(image_id=str(i),image_path=f'{i}.jpg',boxes=[[1,2,30,40]],
                        source_group=f'g{i}',split=split,review_status='reviewed',confirmed_by_user=True,
                        plates=[{'plate_text':'ABC'}]),0)
    rows=store.all_latest()
    changes=[dict(image_id=str(i),source_group=f'g{i}',base_revision=1,**{'from':'val','proposed_to':'train'}) for i in (0,1)]
    return store,rows,changes


def test_atomic_split_revisions_preserve_payload_and_pilot(tmp_path):
    store,rows,changes=fixture(tmp_path)
    result=store.migrate_splits(changes,rows,'approved')
    assert len(result['changes'])==2
    after=store.all_latest()
    assert after['2']==rows['2']
    for i in ('0','1'):
        assert after[i]['boxes']==rows[i]['boxes']
        assert after[i]['source_group']==rows[i]['source_group']
        assert after[i]['plates']==rows[i]['plates']
        assert after[i]['split']=='train' and after[i]['annotation_version']==2
        assert store.history(i)[0]==rows[i]
    assert store.migrate_splits(changes,rows,'approved')['already_applied']
    assert len(store.history('0'))==2


def test_failure_rolls_back_all_rows(tmp_path):
    store,rows,changes=fixture(tmp_path)
    changes[1]['base_revision']=99
    with pytest.raises(ValueError):store.migrate_splits(changes,rows,'bad')
    assert store.all_latest()==rows
    assert len(store.history('0'))==1


def test_stale_proposal_changes_nothing(tmp_path):
    store,rows,changes=fixture(tmp_path)
    stale=copy.deepcopy(rows);stale['0']['source_group']='different'
    with pytest.raises(ValueError,match='preflight'):store.migrate_splits(changes,stale,'bad')
    assert store.all_latest()==rows