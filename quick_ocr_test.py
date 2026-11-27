"""Blind text-only review. No model import or prediction on the labeling path."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path

from annotation_store import AnnotationStore
from annotation_workflow import validate_plate_text
from audit_final_new10 import QUEUE
from final_ocr_dataset import ROOT, STORE, PROTOCOL_HASH, audit_candidates, check_protocol, check_final_conflicts, read, relative_image
from glyph_review import normalized_text, validate_glyph_plate
from yolo_character_dataset import ReadOnlyStore, sha

TEXT_STORE=ROOT/'data/final_ocr_text.sqlite3'
POLICY={'version':1,'base_protocol_sha256':PROTOCOL_HASH,'scope':'text-only; no character F1 or plate IoU',
        'methods':['baseline','border_components','character_yolo'],'backends':['svm','rf','cnn','ensemble'],
        'primary':['character_yolo','cnn'],'normalization':'uppercase ASCII alphanumeric, strip separators',
        'missing_plate':'empty prediction, counted in all denominators','cohort':'fixed New10; small development-independent pilot if provenance passes'}


def policy_hash():
    return hashlib.sha256(json.dumps(POLICY,sort_keys=True).encode()).hexdigest()


def entries(audit):
    ids=read(QUEUE)
    if len(ids)!=10 or len(set(ids))!=10:raise ValueError('Invalid pinned cohort')
    by_hash={r['sha256']:r for r in audit['rows'] if r.get('imported_at')}
    if not set(ids)<=by_hash.keys():raise ValueError('Pinned source missing')
    return [by_hash[k] for k in ids]


def key(entry):return 'text:'+entry['sha256']


def verify_text(row, entry, known):
    if (row.get('label_scope')!='TEXT_REVIEWED' or row.get('review_status')!='reviewed' or
        not row.get('confirmed_by_user') or row.get('split')!='test' or row.get('image_id')!=key(entry) or
        row.get('image_path')!=entry['image'] or row.get('source_sha256')!=entry['sha256'] or
        row.get('source_phash')!=entry['phash'] or entry['status']=='REPLACE' or entry.get('superseded_by')):
        raise ValueError('Invalid text-only record or source')
    validate_plate_text(row)
    if not normalized_text(row['plate_text']):raise ValueError('Text must contain letters or digits')
    group=row.get('source_group','').strip()
    if (group.casefold() in ('','unknown','unverified','needs_human_confirmation') or not row.get('provenance_confirmed') or
        row.get('confirmed_group')!=group or row.get('confirmed_evidence')!=row.get('source_evidence') or
        len(row.get('source_evidence','').strip())<20):raise ValueError('Missing explicit provenance confirmation')
    check_final_conflicts(entry,group,known)


def resolved(store_path=STORE,text_path=TEXT_STORE):
    check_protocol()
    if read(ROOT/'reports/final_ocr_text_protocol_v1.json')!={'policy':POLICY,'policy_sha256':policy_hash()}:
        raise ValueError('Text-only protocol differs from the declared policy')
    audit=audit_candidates(store_path,write=False)
    full={relative_image(r['image_path']):r for r in ReadOnlyStore(store_path).all_latest().values() if r.get('cohort')=='final'}
    texts=ReadOnlyStore(text_path).all_latest() if Path(text_path).is_file() else {}
    accepted=[]; excluded=[]; pending=[]
    for e in entries(audit):
        prior=full.get(relative_image(e['image']),{})
        if e['status']=='REPLACE' or e.get('superseded_by') or prior.get('review_status')=='rejected':
            excluded.append({'image':e['image'],'sha256':e['sha256'],'reason':prior.get('exclusion_reason') or e['reasons'] or e['superseded_by']});continue
        row=texts.get(key(e))
        if prior.get('review_status')=='reviewed':
            validate_glyph_plate(prior,*prior['plate_bbox'][2:]); row={**prior,'label_scope':'TEXT_REVIEWED','image_id':key(e),
                'source_revision_scope':'GLYPH_REVIEWED'}
        if not row:pending.append(e['filename']);continue
        verify_text(row,e,audit['known']+[{**r,'split':'test','final_peer':True} for r in accepted])
        accepted.append({'image':e['image'],'sha256':e['sha256'],'phash':e['phash'],'source_group':row['source_group'],
                         'text':row['plate_text'],'evidence':row['source_evidence'],'revision':row['annotation_version'],
                         'source_revision_scope':row.get('source_revision_scope','TEXT_REVIEWED')})
    return accepted,excluded,pending


def require_evaluation_evidence(rows):
    """Intake dates alone do not document origin or development exposure."""
    lacking=[Path(r['image']).name for r in rows if re.fullmatch(
        r'ngày\s+nhập\s+\d{1,2}/\d{1,2}/\d{4}',r['evidence'].strip(),re.I)]
    if lacking:
        raise ValueError(f'BLOCKED: {len(lacking)} records document only an intake date, not independent provenance')


def freeze(output):
    accepted,excluded,pending=resolved()
    if pending or not accepted:raise ValueError(f'BLOCKED: {len(pending)} pending; {len(accepted)} eligible')
    require_evaluation_evidence(accepted)
    manifest={'policy':POLICY,'policy_sha256':policy_hash(),'rows':accepted,'excluded':excluded,
              'independent_group_count':len({r['source_group'].strip().casefold() for r in accepted}),
              'created_at':datetime.now(timezone.utc).isoformat()}
    with output.open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)


def evaluate(manifest_path, expected_hash, output):
    if sha(manifest_path)!=expected_hash:raise ValueError('Frozen text manifest hash mismatch')
    manifest=read(manifest_path); accepted,excluded,pending=resolved()
    if (pending or not accepted or manifest['policy']!=POLICY or manifest['policy_sha256']!=policy_hash() or
        manifest['rows']!=accepted or manifest['excluded']!=excluded):raise ValueError('Text freeze does not match audited revisions')
    require_evaluation_evidence(accepted)
    # No model is imported or invoked above the gate.
    import cv2
    from statistics import median
    from time import perf_counter
    from dual_yolo_client import Worker
    from dual_yolo_pipeline import run, MixClassifiers
    from evaluate_labeled import edit_distance
    results=[]
    # Reserve the output exclusively before inference to avoid accidental reruns.
    with output.open('x',encoding='utf-8') as f:
        start=perf_counter(); models=MixClassifiers(); load=perf_counter()-start
        with Worker() as worker:
            for row in accepted:
                image=cv2.imread(str(ROOT/row['image'])); ref=normalized_text(row['text'])
                for method in POLICY['methods']:
                    result=run(image,worker,models,'all',method)
                    results.append({'image':row['image'],'reference':ref,'method':method,'cold':result['cold'],
                                    'seconds':result['timings']['total'],'plate_missing':not result['plate'],
                                    'ocr':{k:{'prediction':p,'edits':edit_distance(ref,p),'exact':bool(result['plate'] and p==ref)} for k,p in result['text'].items()}})
        summary={}
        for method in POLICY['methods']:
            rows=[r for r in results if r['method']==method]
            summary[method]={k:{'correct':sum(r['ocr'][k]['exact'] for r in rows),'total':len(rows),
                                'edits':sum(r['ocr'][k]['edits'] for r in rows),
                                'cer':sum(r['ocr'][k]['edits'] for r in rows)/sum(len(r['reference']) for r in rows),
                                'plate_missing':sum(r['plate_missing'] for r in rows)} for k in POLICY['backends']}
        json.dump({'scope':'small fixed text-only cohort','manifest_sha256':expected_hash,'classifiers_load_seconds':load,
                   'latency':{label:{'n':len(samples),'median_seconds':median(samples) if samples else None}
                              for label,cold in [('cold',True),('warm',False)] for samples in [[r['seconds'] for r in results if r['cold']==cold]]},
                   'independent_group_count':len({r['source_group'].strip().casefold() for r in accepted}),
                   'summary':summary,'rows':results},f,ensure_ascii=False,indent=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='action',required=True)
    fr=sub.add_parser('freeze');fr.add_argument('output',type=Path)
    ev=sub.add_parser('evaluate');ev.add_argument('manifest',type=Path);ev.add_argument('sha256');ev.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.action=='freeze':freeze(args.output)
    else:evaluate(args.manifest,args.sha256,args.output)