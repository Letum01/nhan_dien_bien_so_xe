"""Apply user-approved immutable proposal after exact snapshot/integrity checks."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from annotation_store import AnnotationStore
from dataset_validation import check_split_leakage
from round2_review import check_frozen_pilot
from label_detection import validate_annotation
from audit_round2_training import quality_flags
import cv2

ROOT=Path(__file__).resolve().parent


def main():
    proposal=ROOT/'reports/split_options_20261124T152525043845Z'
    options=json.loads((proposal/'options.json').read_text(encoding='utf-8'))
    planned=json.loads((proposal/'option_B_proposal_rows_NOT_APPLIED.json').read_text(encoding='utf-8'))
    changes=options['option_B']['changes']
    operation=hashlib.sha256((proposal/'options.json').read_bytes()).hexdigest()
    store=AnnotationStore(ROOT/'data/annotations.sqlite3')
    with sqlite3.connect(store.path) as connection:
        exists=connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='split_migrations'").fetchone()
        applied=connection.execute('SELECT receipt FROM split_migrations WHERE operation_id=?',(operation,)).fetchone() if exists else None
    if applied:
        receipt=json.loads(applied[0])
        current=store.all_latest()
        if any(current.get(c['image_id'])!=c['after'] for c in receipt['changes']):
            raise ValueError('BLOCKED: applied migration has later changes; no automatic retry or restore')
        print('ALREADY_APPLIED',operation,'no revisions created')
        return
    latest={k:v for k,v in store.all_latest().items() if v.get('review_status')=='reviewed'}
    expected={row['image_id']:dict(row) for row in planned}
    for change in changes:expected[change['image_id']]['split']=change['from']
    if latest!=expected:raise ValueError('BLOCKED: approved proposal no longer exactly matches SQLite')
    plan=json.loads((ROOT/'reports/round2_plan/plan.json').read_text(encoding='utf-8'))
    pilot={r['image_id'] for r in plan['frozen_pilot']};candidates={r['sha256'] for r in plan['candidates']}
    if len(changes)!=33 or any(c['image_id'] in pilot or c['image_id'] not in candidates for c in changes):
        raise ValueError('BLOCKED: approved identity list mismatch')
    if Counter((c['from'],c['proposed_to']) for c in changes)!={('val','train'):31,('train','val'):2}:
        raise ValueError('BLOCKED: direction list differs from explicitly approved 31+2 clarification')
    for collection in (list(latest.values()),planned):
        check_frozen_pilot(collection);check_split_leakage(collection,ROOT)
        for row in collection:
            validate_annotation(row)
            image=cv2.imread(str(ROOT/row['image_path']))
            if hashlib.sha256((ROOT/row['image_path']).read_bytes()).hexdigest()!=row['image_id']:
                raise ValueError('Source hash mismatch')
            if quality_flags(row,image.shape[1],image.shape[0]):raise ValueError('Quality flags remain')
    folder=ROOT/'reports'/('splitB_applied_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));folder.mkdir()
    backup=folder/'annotations_before.sqlite3'
    with sqlite3.connect(store.path) as source,sqlite3.connect(backup) as target:source.backup(target)
    with sqlite3.connect(backup) as connection:
        assert connection.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        before=list(connection.execute('SELECT * FROM revisions ORDER BY id'))
    (folder/'before.json').write_text(json.dumps(list(latest.values()),ensure_ascii=False,indent=2),encoding='utf-8')
    receipt=store.migrate_splits(changes,expected,operation)
    after=store.all_latest();reviewed={k:v for k,v in after.items() if v.get('review_status')=='reviewed'}
    assert Counter(r['split'] for r in reviewed.values())=={'train':46,'val':12,'test':3}
    changed={c['image_id'] for c in changes}
    for identity,old in latest.items():
        new=reviewed[identity]
        if identity not in changed:assert old==new
        else:
            assert new['annotation_version']==old['annotation_version']+1
            assert {k:v for k,v in new.items() if k not in ('split','annotation_version')}=={k:v for k,v in old.items() if k not in ('split','annotation_version')}
    check_frozen_pilot(list(reviewed.values()));check_split_leakage(list(reviewed.values()),ROOT)
    with sqlite3.connect(store.path) as connection:
        assert list(connection.execute('SELECT * FROM revisions WHERE id<=? ORDER BY id',(before[-1][0],)))==before
    receipt['backup']=str(backup);receipt['after_counts']=dict(Counter(r['split'] for r in reviewed.values()))
    (folder/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'after.json').write_text(json.dumps(list(reviewed.values()),ensure_ascii=False,indent=2),encoding='utf-8')
    print('APPLIED',folder,receipt['after_counts'])


if __name__=='__main__':main()