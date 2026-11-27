"""Read-only audit of the 2,932-image pool and bounded 100-image teacher pilot."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2

from dataset_audit import fingerprint
from development_autolabel import flags
from dual_yolo_client import Worker, CHECKPOINT, CHARACTER
from dual_yolo_pipeline import MixClassifiers, run
from final_ocr_dataset import ROOT, STORE, audit_candidates, check_protocol, conflicts, read, relative_image
from yolo_character_dataset import ReadOnlyStore, sha


def audit_pool():
    check_protocol()
    audit=audit_candidates(write=False)
    detection=ReadOnlyStore(ROOT/'data/annotations.sqlite3').all_latest()
    by_path={relative_image(r['image_path']):r for r in detection.values()}
    forbidden=[{**r,'source_group':r.get('metadata',{}).get('source_group','')} for r in audit['rows']]
    forbidden.extend(r for r in audit['known'] if r['split'] in ('val','test'))
    forbidden.extend({'image':r['image_path'],'sha256':r['source_sha256'],'phash':r['source_phash'],'source_group':r['source_group']}
                     for r in ReadOnlyStore(STORE).all_latest().values() if r.get('source_sha256') and r.get('source_phash'))
    known_paths={r['image'] for r in audit['known']}
    rows=list(csv.DictReader((ROOT/'reports/dataset_audit.csv').open(encoding='utf-8-sig',newline='')))
    eligible=[]; excluded=Counter(); hashes=Counter(r['sha256'] for r in rows)
    for r in rows:
        key=relative_image(r['image_path']); det=by_path.get(key,{})
        if r['readable']!='True' or not r['phash']:excluded['unreadable']+=1;continue
        if det.get('split') in ('val','test'):excluded['val_or_test_split']+=1;continue
        if conflicts(r['sha256'],int(r['phash'],16),det.get('source_group',''),forbidden):excluded['leakage_or_test_conflict']+=1;continue
        if det.get('review_status')=='reviewed' and det.get('split')=='train':excluded['already_reviewed_train']+=1;continue
        if key in known_paths:excluded['prior_experiment_or_queue']+=1;continue
        eligible.append({'image':r['image_path'],'sha256':r['sha256'],'phash':r['phash'],
                         'source_group':det.get('source_group') or 'UNKNOWN','provenance':'UNKNOWN'})
    # Deduplicate within the unlabeled pool by exact hash and pHash <=6.
    unique=[]; seen=[]
    for r in sorted(eligible,key=lambda x:x['image']):
        if conflicts(r['sha256'],int(r['phash'],16),None,seen):excluded['pool_duplicate_or_near']+=1;continue
        unique.append(r); seen.append(r)
    return {'total_inventory':len(rows),'readable':sum(r['readable']=='True' for r in rows),
            'exact_duplicate_files':sum(v-1 for v in hashes.values() if v>1),'reviewed_detection':sum(r['review_status']=='reviewed' for r in detection.values()),
            'excluded':dict(excluded),'eligible_unique':len(unique),'eligible':unique}


def pilot(out, limit=100):
    summary=audit_pool(); out.mkdir(exist_ok=False)
    (out/'pool_audit.json').write_text(json.dumps({k:v for k,v in summary.items() if k!='eligible'},indent=2),encoding='utf-8')
    # Evaluate the filter on the 46 reviewed train plates first (never validation/test).
    from train_ocr_v2 import prepare
    from glyph_review import normalized_text
    _,reviewed=prepare(); train_rows=[r for r in reviewed.values() if r['split']=='train']
    models=MixClassifiers(); hashes=read(ROOT/'reports/final_ocr_protocol.json')['hashes']
    with Worker() as worker:
        filter_eval=[]
        for r in train_rows:
            res=run(cv2.imread(str(ROOT/r['image_path'])),worker,models,'all','character_yolo')
            w=flags(res,cv2.imread(str(ROOT/r['image_path'])).shape)
            filter_eval.append({'image':r['image_path'],'accepted':not w,'flags':w,'exact':res['text']['cnn']==normalized_text(r['plate_text'])})
        accepted=[r for r in filter_eval if r['accepted']]
        (out/'filter_on_reviewed_train.json').write_text(json.dumps({'plates':len(filter_eval),'accepted':len(accepted),
            'accepted_exact':sum(r['exact'] for r in accepted),'rows':filter_eval},indent=2),encoding='utf-8')
        pseudo=[]
        for r in summary['eligible'][:limit]:
            im=cv2.imread(str(ROOT/r['image'])); res=run(im,worker,models,'all','character_yolo')
            w=flags(res,im.shape)
            pseudo.append({**r,'status':'AUTO_PSEUDO_LABEL' if not w else 'NEEDS_REVIEW','flags':w,
                           'teacher_hashes':{'plate':sha(CHECKPOINT),'character':sha(CHARACTER),'cnn':hashes['experiments\\ocr_mix_experiment02_20261125T080339560263Z\\cnn_mix\\cnn_model.keras']},
                           'plate':res['plate'],'glyph_boxes':res['glyph_boxes'],'glyph_scores':res['glyph_scores'],'text':res['text']})
    (out/'pseudo_pilot_100.json').write_text(json.dumps({'scope':'unverified UNKNOWN-group pseudo-labels; NOT training ground truth',
        'count':len(pseudo),'accepted_by_filter':sum(r['status']=='AUTO_PSEUDO_LABEL' for r in pseudo),'rows':pseudo},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Pool',summary['eligible_unique'],'filter_train',len(accepted),'/',len(filter_eval),'pseudo_accepted',sum(r['status']=='AUTO_PSEUDO_LABEL' for r in pseudo))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--limit',type=int,default=100)
    a=p.parse_args();pilot(a.output,a.limit)