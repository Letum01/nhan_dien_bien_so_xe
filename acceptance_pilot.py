"""Non-blind, separate pilot evaluation and read-only New10 evidence audit."""
import argparse
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from quick_ocr_test import ROOT, resolved, POLICY
from final_ocr_dataset import check_protocol, pilot_rows, read
from glyph_review import normalized_text, validate_glyph_plate
from yolo_character_dataset import sha


def new10_audit():
    accepted,excluded,pending=resolved()
    intake_only=[Path(r['image']).name for r in accepted if re.fullmatch(r'ngày\s+nhập\s+\d{1,2}/\d{1,2}/\d{4}',r['evidence'].strip(),re.I)]
    return {'reviewed':len(accepted),'glyph_reviewed':sum(r['source_revision_scope']=='GLYPH_REVIEWED' for r in accepted),
            'text_reviewed':sum(r['source_revision_scope']=='TEXT_REVIEWED' for r in accepted),
            'characters':sum(len(normalized_text(r['text'])) for r in accepted),'groups_declared':len({r['source_group'].strip().casefold() for r in accepted}),
            'pending':pending,'excluded':excluded,'intake_date_only':intake_only,'independence_verified':False,
            'status':'BLOCKED','reason':'No independently verified capture/source evidence; a recorded intake date does not establish provenance.'}


def edit_counts(ref,pred):
    """Minimum edit alignment; deterministic tie order substitution/deletion/insertion."""
    dp=[[(0,0,j) for j in range(len(pred)+1)]]
    for i,a in enumerate(ref,1):
        current=[(0,i,0)]
        for j,b in enumerate(pred,1):
            s,d,n=dp[-1][j-1]; x,y,z=dp[-1][j];u,v,w=current[-1]
            current.append(min([(s+int(a!=b),d,n),(x,y+1,z),(u,v,w+1)],key=sum))
        dp.append(current)
    return dict(zip(('substitutions','deletions','insertions'),dp[-1][-1]))


def pilot(out):
    import cv2
    from dual_yolo_client import Worker
    from dual_yolo_pipeline import MixClassifiers,run
    check_protocol()
    manifest=ROOT/'reports/pilot_ocr_labels_v1.json'; payload=read(manifest)
    rows=pilot_rows(); saved={r['image_id']:r for r in payload['rows']}
    if len(rows)!=3 or sum(len(r['characters']) for r in rows)!=24:raise ValueError('Pilot not fully reviewed')
    inputs=[]
    for r in rows:
        path=ROOT/r['image_path']; e=saved[r['image_id']]
        if any(e.get(k)!=v for k,v in r.items()) or sha(path)!=e['source_sha256']:raise ValueError('Pilot revision or file changed')
        validate_glyph_plate(r,*r['plate_bbox'][2:]); image=cv2.imread(str(path))
        if image is None:raise ValueError(f'Unreadable pilot image: {path}')
        inputs.append((path,image,normalized_text(r['plate_text'])))
    with out.open('x',encoding='utf-8') as f:
        start=perf_counter();models=MixClassifiers();model_seconds=perf_counter()-start;records=[]
        with Worker() as worker:
            for method in POLICY['methods']:
                for path,image,ref in inputs:
                    r=run(image,worker,models,'all',method)
                    records.append({'image':path.name,'reference':ref,'method':method,'plate_missing':r['plate'] is None,
                                    'cold':r['cold'],'timings':r['timings'],'gpu':r['worker']['gpu'],
                                    'predictions':{k:{'text':p,'exact':bool(r['plate'] and p==ref),**edit_counts(ref,p)} for k,p in r['text'].items()}})
        def latency(values):return {'n':len(values),'mean':float(np.mean(values)),'median':float(np.median(values)),'p95':float(np.percentile(values,95))} if values else {'n':0}
        summary=[]
        for method in POLICY['methods']:
            cases=[r for r in records if r['method']==method]; n=sum(len(r['reference']) for r in cases)
            for backend in POLICY['backends']:
                preds=[r['predictions'][backend] for r in cases]
                counts={k:sum(p[k] for p in preds) for k in ('substitutions','deletions','insertions')}
                summary.append({'method':method,'backend':backend,'images':len(cases),'characters':n,
                                'plate_missing':sum(r['plate_missing'] for r in cases),'exact':sum(p['exact'] for p in preds),
                                'levenshtein':sum(counts.values()),'cer':sum(counts.values())/n,**counts})
        json.dump({'scope':'pilot diagnostics-exposed; not blind; end-to-end text metrics only',
                   'label_manifest_sha256':sha(manifest),'gpu_mode':'CUDA YOLO / CPU OCR', 'ocr_load_seconds':model_seconds,
                   'latency':{name:latency([r['timings']['total'] for r in records if r['cold']==cold]) for name,cold in [('cold',True),('warm',False)]},
                   'summary':summary,'records':records},f,ensure_ascii=False,indent=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['audit-new10','pilot']);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.mode=='pilot':pilot(args.output)
    else:
        with args.output.open('x',encoding='utf-8') as f:json.dump(new10_audit(),f,indent=2)