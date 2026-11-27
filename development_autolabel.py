"""Explicit train-only proposals. Nothing here writes to reviewed annotation stores."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import re

import cv2

from final_ocr_dataset import ROOT,STORE,audit_candidates, conflicts,check_protocol,relative_image,read
from dataset_audit import fingerprint
from yolo_character_dataset import ReadOnlyStore,sha


def pool(deduplicate=False):
    audit=audit_candidates(write=False)
    source=ReadOnlyStore(ROOT/'data/annotations.sqlite3').all_latest().values()
    blocked=[{**r,'source_group':r.get('metadata',{}).get('source_group','')} for r in audit['rows']]
    blocked += [{'image':r['image_path'],'sha256':r['source_sha256'],'phash':r['source_phash'],'source_group':r['source_group']}
                for r in ReadOnlyStore(STORE).all_latest().values() if r.get('source_sha256') and r.get('source_phash')]
    candidates=[]; skipped=[]
    for r in sorted(source,key=lambda r:r['image_path']):
        digest,p,shape=fingerprint(ROOT/r['image_path'])
        row={'image':r['image_path'],'sha256':digest,'phash':f'{p:016x}' if p is not None else None,'source_group':r.get('source_group','')}
        if r['split']!='train':
            if shape:blocked.append(row)
        elif r['review_status']=='reviewed' and shape and row['source_group']:candidates.append(row)
    kept=[]
    for r in candidates:
        match=conflicts(r['sha256'],int(r['phash'],16),r['source_group'],blocked+(kept if deduplicate else []))
        if match:skipped.append({**r,'matches':match})
        else:kept.append(r)
    return kept,skipped


def flags(result,shape):
    plate=result['plate']; boxes=result['glyph_boxes']; labels=result['text']; warnings=[]
    if not plate:return ['plate_missing']
    if plate['score']<.60 or any(s<.50 for s in result['glyph_scores']):warnings.append('low_detector_score')
    if not 7<=len(boxes)<=10:warnings.append('glyph_count_outside_heuristic')
    if len({labels.get(k) for k in ('svm','rf','cnn')})>1:warnings.append('classifier_disagreement')
    _,_,cw,ch=plate['crop_box']; x,y,w,h=plate['bbox']
    if min(x,y,shape[1]-x-w,shape[0]-y-h)<=1 or any(min(a,b,cw-a-c,ch-b-d)<=1 for a,b,c,d in boxes):warnings.append('possible_clipping')
    if any(max(0,min(a+c,x+w)-max(a,x))*max(0,min(b+d,y+h)-max(b,y))>.25*min(c*d,w*h)
           for i,(a,b,c,d) in enumerate(boxes) for x,y,w,h in boxes[:i]):warnings.append('overlapping_glyph_boxes')
    if len(boxes)>1:
        import numpy as np
        gaps=np.diff(sorted(b+d/2 for _,b,_,d in boxes)); ratio=max(gaps)/np.median([d for *_,d in boxes])
        if .40<=ratio<=.70:warnings.append('ambiguous_reading_order')
    if not re.fullmatch(r'[0-9]{2}[A-Z][A-Z0-9]{0,2}[0-9]{4,6}',labels.get('ensemble','')):warnings.append('format_heuristic')
    return warnings


def propose(rows,worker,models):
    from dual_yolo_pipeline import run
    allowed={relative_image(r['image']):r for r in pool()[0]}
    check_protocol()
    for row in rows:
        if allowed.get(relative_image(row['image']))!=row:raise ValueError('Not in current train-only development allowlist')
    for row in rows:
        image=cv2.imread(str(ROOT/row['image']))
        if image is None or sha(ROOT/row['image'])!=row['sha256']:raise ValueError('Source changed')
        result=run(image,worker,models,'all','character_yolo')
        warnings=flags(result,image.shape)
        yield {**row, 'image_hw':list(image.shape[:2]),'result':result,'flags':warnings,
               'status':'NEEDS_REVIEW' if warnings else 'AUTO_PSEUDO_LABEL','label_origin':'AUTO_PSEUDO_LABEL',
               'created_at':datetime.now(timezone.utc).isoformat(),'plate_conf':.4,'character_conf':.25,
               'model_and_code_hashes':read(ROOT/'reports/final_ocr_protocol.json')['hashes']}


def export(rows,path):
    from glyph_review import CLASSES
    samples=[]; rejected=[]
    for r in rows:
        if r['label_origin']!='AUTO_PSEUDO_LABEL' or r['status'] not in ('AUTO_PSEUDO_LABEL','NEEDS_REVIEW','REJECTED'):
            raise ValueError('Invalid pseudo-label status')
        if r['status']=='REJECTED':rejected.append(r);continue
        text=r.get('edited_text',r['result']['text']['ensemble']).strip().upper()
        boxes=r['result']['glyph_boxes']
        mapped=list(zip(boxes,text)) if boxes and len(boxes)==len(text) and all(c in CLASSES for c in text) else None
        samples.append({**r,'mapped_glyphs_crop_xywh':mapped})
    labels=Counter(c for r in samples for _,c in r['mapped_glyphs_crop_xywh'] or [])
    scores=[x for r in samples for x in r['result']['glyph_scores']]
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump({'scope':'development pseudo-labels only; not a training snapshot','samples':samples,'rejected':rejected,
                   'counts':dict(Counter(r['status'] for r in rows)),'class_counts':dict(labels),
                   'missing_classes':[c for c in CLASSES if not labels[c]],
                   'confidence':{'n':len(scores),'min':min(scores,default=None),'max':max(scores,default=None),
                                 'mean':sum(scores)/len(scores) if scores else None},
                   'training_plan':{'status':'BLOCKED until human QA and explicit experiment approval','detector':'character',
                                    'config_reference':'configs/yolo_character_pilot.json','split':'train-only proposals; retain reviewed validation'}},f,ensure_ascii=False,indent=2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--limit',type=int,default=10)
    a=p.parse_args(); rows,skipped=pool(deduplicate=True)
    if a.output.exists():p.error('Choose a new output path')
    from dual_yolo_client import Worker
    from dual_yolo_pipeline import MixClassifiers
    with Worker() as worker:export(list(propose(rows[:max(0,a.limit)],worker,MixClassifiers())),a.output)
    print('Eligible:',len(rows),'excluded:',len(skipped))