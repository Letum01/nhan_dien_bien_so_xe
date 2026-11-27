"""Compare rejection arms on the exact same saved development boxes."""
from collections import Counter
import json
import subprocess
from pathlib import Path
from time import perf_counter

import cv2
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from acceptance_pilot import edit_counts
from dual_yolo_client import PYTHON
from final_ocr_dataset import check_protocol,read
from glyph_review import CLASSES, ROOT
from hard_negative_data import OUT,diagnostics,geometry_keep,intersection
from recovery_features import features
from segmentation_benchmark import match_boxes
from yolo_character_dataset import validate
from yolo_character_pipeline import normalize_glyph


def main():
    check_protocol(); validate(ROOT/'datasets/yolo_character_round1')
    dest=OUT/'evaluation';dest.mkdir(exist_ok=False)
    manifest=read(OUT/'manifest.json'); bank=np.load(OUT/'data.npz',allow_pickle=False)
    X,Y,G=[bank[k] for k in ('x','y','groups')]
    if len(X)!=len(manifest['rows']) or any(int(y)!=r['class'] or g!=r['group'] for y,g,r in zip(Y,G,manifest['rows'])):
        raise ValueError('Manifest mismatch')
    old=read(ROOT/'reports/accuracy_recovery/benchmark.json')
    rows=[r for r in old['records'] if r['arm']=='baseline']; assert len(rows)==12
    glyphs=[]
    for row in rows:
        image=cv2.imread(str(ROOT/row['image'])); row['start']=len(glyphs)
        glyphs.extend(normalize_glyph(image[y:y+h,x:x+w]) for x,y,w,h in row['pred_boxes'])
        row['end']=len(glyphs)
    dev_count=len(glyphs)
    case=old['replays'][0];case_dir=ROOT/'reports/accuracy_recovery'/case['case']
    case_images=[cv2.imread(str(p),0) for p in sorted(case_dir.glob('*_normalized.png'))]; glyphs.extend(case_images)
    np.save(dest/'glyphs.npy',np.stack(glyphs)); predictions={}; training={}; times={}
    targets=(Y==36).astype(int)
    folds=list(GroupKFold(3).split(X,Y,G))
    for name,arms in [('svm',('hog','zoning')),('rf',('hog','lbp','zoning','geometry'))]:
        for arm in arms:
            F=np.stack([features(g,arm) for g in X]); Q=np.stack([features(g,arm) for g in glyphs])
            def fit(labels,ids):
                m=(make_pipeline(StandardScaler(),SVC(C=10,gamma='scale',class_weight='balanced')) if name=='svm' else
                   RandomForestClassifier(n_estimators=100,max_depth=20,class_weight='balanced_subsample',random_state=42,n_jobs=4))
                m.fit(F[ids],labels[ids]);return m
            for n,labels in [(2,targets),(37,Y)]:
                k=f'{name}_{arm}_{n}'; oof=np.empty(len(Y),int)
                for tr,va in folds:oof[va]=fit(labels,tr).predict(F[va])
                m=fit(labels,np.arange(len(Y)));joblib.dump(m,dest/(k+'.joblib'))
                t=perf_counter();predictions[k]=m.predict(Q);times[k]=perf_counter()-t
                rejects=oof==(1 if n==2 else 36)
                training[k]={'oof_confusion':confusion_matrix(labels,oof,labels=list(range(n))).tolist(),
                             'binary_confusion':confusion_matrix(targets,rejects,labels=[0,1]).tolist()}
                if name=='rf':
                    importance=m.feature_importances_;training[k]['importance']={'hog_sum':float(importance[:324].sum()),'extra_sum':float(importance[324:].sum()),
                        'top':[[int(i),float(importance[i])] for i in np.argsort(-importance)[:8]]}
            print('finished',name,arm,flush=True)
    subprocess.run([str(PYTHON),str(ROOT/'hard_negative_probe.py'),str(dest/'glyphs.npy'),str(dest/'cuda.json')],cwd=ROOT,check=True)
    cuda=read(dest/'cuda.json')
    for n,arm in cuda.items():predictions['cnn_'+n]=np.array(arm['predictions']);times['cnn_'+n]=arm['batch_median_s']

    def arm_eval(name,reject=None):
        out=[]
        for row in rows:
            start,end=row['start'],row['end']; boxes=row['pred_boxes']; raw=row['ocr']['cnn']['prediction']; gt=row['gt_boxes']
            labels=predictions[name][start:end].tolist() if name in predictions else None
            if name=='geometry':keep=[geometry_keep(d) for d in diagnostics(boxes,glyphs[start:end])]
            elif labels is None:keep=[True]*len(boxes)
            else:keep=[c!=reject for c in labels]
            kept=[i for i,k in enumerate(keep) if k]; dropped=[i for i,k in enumerate(keep) if not k]
            pred=''.join(CLASSES[labels[i]] if reject==36 else raw[i] for i in kept)
            match=match_boxes([boxes[i] for i in kept],gt); orig={m['pred'] for m in row['geometry']['matches']}
            unmatched=set(range(len(boxes)))-orig
            # Unmatched is not necessarily noise: only zero-overlap FP is treated
            # as a geometric proxy for a negative on this benchmark.
            clear=[i for i in unmatched if not any(intersection(boxes[i],g) for g in gt)]
            out.append({'image':row['image'],'reference':row['reference'],'text':pred,'raw':raw,'removed':dropped,'geometry':match,
                        'exact':pred==row['reference'],'edits':sum(edit_counts(row['reference'],pred).values()),
                        'true_removed':len(orig.intersection(dropped)),'unmatched_removed':len(unmatched.intersection(dropped)),
                        'clear_negative_total':len(clear),'clear_negative_removed':len(set(clear).intersection(dropped))})
        tp,fp,fn=[sum(r['geometry'][k] for r in out) for k in ('tp','fp','fn')]
        return {'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn),'exact':sum(r['exact'] for r in out),
                'cer':sum(r['edits'] for r in out)/sum(len(r['reference']) for r in out),
                'regressions':sum(r['raw']==r['reference'] and not r['exact'] for r in out),
                'true_removed':sum(r['true_removed'] for r in out),'unmatched_removed':sum(r['unmatched_removed'] for r in out),
                'clear_negative_total':sum(r['clear_negative_total'] for r in out),'clear_negative_removed':sum(r['clear_negative_removed'] for r in out),
                'inference_batch_seconds':times.get(name),'rows':out}
    results={k:arm_eval(k) for k in ('baseline','geometry')}
    results.update({k:arm_eval(k,1 if k.endswith('_2') else 36) for k in predictions})
    raw=case['v1']['text']['cnn']; detail=diagnostics(case['v1']['glyph_boxes'],case_images)
    case_rows={}
    for name,p in predictions.items():
        labels=p[dev_count:].tolist(); reject=1 if name.endswith('_2') else 36
        case_rows[name]={'labels':labels,'candidate':''.join(CLASSES[c] if reject==36 else raw[i] for i,c in enumerate(labels) if c!=reject),
                         'removed':[i+1 for i,c in enumerate(labels) if c==reject]}
    show=cv2.imread(str(case_dir/'glyph_order.png')); tiles=[]
    for i,(g,d) in enumerate(zip(case_images,detail),1):
        tile=cv2.cvtColor(cv2.resize(g,(64,64),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
        cv2.putText(tile,str(i),(2,14),0,.45,(40,200,40),1);tiles.append(tile)
    cv2.imwrite(str(dest/'case_glyph_sheet.png'),np.hstack(tiles))
    data={'dataset_counts':dict(Counter(map(int,Y))),'origins':dict(Counter(r['kind'] for r in manifest['rows'])),
          'ambiguous':len(manifest['ambiguous_excluded']),'class_order':list(CLASSES)+['NON_CHARACTER'],
          'training':training,'cuda':{k:{a:b for a,b in v.items() if a!='predictions'} for k,v in cuda.items()},
          'results':results,'case':{'raw':raw,'geometry':detail,'models':case_rows,'ground_truth':None}}
    (dest/'report.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print({k:{a:r[a] for a in ('exact','cer','tp','fp','fn','true_removed','unmatched_removed')} for k,r in results.items()},flush=True)


if __name__=='__main__':main()