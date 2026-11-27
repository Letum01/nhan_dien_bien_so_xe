"""Reproduce the historical SVC failure; no checkpoint is written."""
import argparse
import hashlib
from collections import Counter
from time import perf_counter

import cv2
import joblib
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from alpr_feature_correction import generate_targeted_synthetic, get_hog_features
from dual_yolo_client import Worker
from segmentation_benchmark import match_boxes
from train_ocr_v2 import CHAR, REAL, ROOT, prepare, put, read, edit_counts, CLASSES
from yolo_character_pipeline import WEIGHTS, normalize_glyph, order_boxes


def weight_cases(target, n_real):
    n = len(target)
    if not 0 < n_real < n:
        raise ValueError('Both training domains must be present')
    domain = np.r_[np.full(n_real, .5/n_real), np.full(n-n_real, .5/(n-n_real))]
    return {'A_unweighted': (None, np.ones(n)), 'B_class': ('balanced', np.ones(n)),
            'C_domain': (None, domain*n), 'D_joint_historical': ('balanced', domain),
            'D_joint_mean_one_domain': ('balanced', domain*n),
            'scale_only_control': (None, np.full(n, 1/n))}


def image(path, flags=cv2.IMREAD_COLOR):
    pixels = cv2.imread(str(path), flags)
    if pixels is None:
        raise ValueError(f'Unreadable {path}')
    return pixels


def crop(im, box):
    x, y, w, h = box
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > im.shape[1] or y+h > im.shape[0]:
        raise ValueError(f'Out-of-bounds ROI {box}')
    return im[y:y+h, x:x+w]


def summarize_hog(domains, out):
    def stats(f):
        return {'samples': len(f), 'all_zero': int((~f.any(axis=1)).sum()),
                'norm_q05_50_95': np.percentile(np.linalg.norm(f, axis=1), [5,50,95]).tolist(),
                'nonzero_q05_50_95': np.percentile(np.count_nonzero(f, axis=1), [5,50,95]).tolist(),
                'feature_mean': f.mean(axis=0).tolist(), 'feature_std': f.std(axis=0).tolist()}

    train = np.concatenate([domains[k][0] for k in ('real_train','targeted_synthetic','detector_train')])
    pca = PCA(2, random_state=42, svd_solver='randomized').fit(train)
    canvas=np.full((515,1100,3),255,np.uint8)
    cv2.putText(canvas,'PCA fit on train only / Nonzero HOG bins', (30,25),0,.7,(0,0,0),1)
    cv2.rectangle(canvas,(30,45),(535,419),(210,210,210),1)
    cv2.rectangle(canvas,(565,45),(1070,419),(210,210,210),1)
    bounds=np.maximum(np.max(np.abs(pca.transform(train)),axis=0)*1.1,.01)
    result = {}
    for index,(name, (f, y)) in enumerate(domains.items()):
        color=((180,55,0),(0,130,0),(135,135,135),(0,80,205),(175,0,180))[index]
        proj = pca.transform(f)
        xy=np.clip(proj/bounds,-1,1)*[250,-185]+[282,232]
        for xx,yy in xy.astype(int):cv2.circle(canvas,(xx,yy),1,color,-1)
        bins=np.histogram(np.count_nonzero(f,axis=1),bins=np.arange(0,331,10))[0]/len(f)
        points=np.column_stack((np.linspace(573,1063,len(bins)),418-370*bins)).astype(np.int32)
        cv2.polylines(canvas,[points],False,color,2)
        cv2.putText(canvas,name,(30+index*210,460),0,.48,color,1)
        result[name] = {**stats(f), 'classes': {c:stats(f[y==i]) for i,c in enumerate(CLASSES) if np.any(y==i)}}
    if not cv2.imwrite(str(out/'hog_diagnostics.png'),canvas):raise OSError('Cannot write HOG diagnostics')
    real, ry = domains['real_train']; synth, sy = domains['targeted_synthetic']
    rc = {i:real[ry==i].mean(axis=0) for i in np.unique(ry)}
    result['paired_class_domains'] = {CLASSES[i]:{
        'mean_bin_wasserstein': float(np.mean([wasserstein_distance(a,b) for a,b in zip(real[ry==i].T,synth[sy==i].T)])),
        'centroid_gap_l2': float(np.linalg.norm(rc[i]-synth[sy==i].mean(axis=0))),
        'nearest_other_real_centroid_l2': min(float(np.linalg.norm(rc[i]-v)) for j,v in rc.items() if i!=j)
    } for i in rc}
    return result


def main(out):
    buckets, reviews = prepare()
    out.mkdir(parents=True, exist_ok=False)
    manifest = read(CHAR/'manifest.json')['rows']
    real = {k:(get_hog_features([image(REAL/r['image_path'],0) for r in rows]),
               np.array([CLASSES.index(r['class_label']) for r in rows])) for k,rows in buckets.items()}
    detector, labels, timings = [], [], []
    with Worker() as worker:
        # Historical run generated Character predictions on reviewed plate crops (NOT Dual-YOLO).
        for r in manifest:
            row = r['review']
            if row['split']!='train':
                continue
            im = image(CHAR/r['image']); response = worker.request('character', im)
            boxes = [d['bbox'] for d in order_boxes(response['rows'])]
            gt = row['characters']; match = match_boxes(boxes,[g['bbox'] for g in gt],.5)
            for m in match['matches']:
                detector.append(normalize_glyph(crop(im,boxes[m['pred']]))); labels.append(CLASSES.index(gt[m['gt']]['label']))
            timings.append({k:response[k] for k in ('worker_seconds','peak_reserved')})
        gpu = worker.info
    sx, sy = generate_targeted_synthetic(200,400,42)
    dx, dy = get_hog_features(detector), np.asarray(labels)
    rx, ry = real['train']; vx, vy = real['validation']; sh = get_hog_features(sx)
    x = np.concatenate([rx,dx,sh]); y = np.r_[ry,dy,sy]
    if (len(rx),len(dx),len(sx),len(vx))!=(370,370,12400,99) or x.shape[1]!=324 or not np.isfinite(x).all():
        raise ValueError('Historical corpus or HOG contract did not reproduce')
    plates=[]; dev_features=[]; dev_labels=[]
    for rec in read(ROOT/'experiments/yolo_character_round1/comparison/results.json')['records']:
        if (rec['scope'],rec['method'])!=('yolo','character_yolo'):
            continue
        row=reviews[rec['id']]
        if row['split']!='val':
            raise ValueError('Non-development row in replay')
        im=image(ROOT/row['image_path']); boxes=rec['pred_boxes']; px,py,_,_=row['plate_bbox']
        gt=row['characters']; gt_boxes=[[px+a,py+b,w,h] for a,b,w,h in (g['bbox'] for g in gt)]
        matches=match_boxes(boxes,gt_boxes,.5)['matches']
        feats=get_hog_features([normalize_glyph(crop(im,b)) for b in boxes]) if boxes else np.empty((0,324),np.float32)
        idx=[m['pred'] for m in matches]; target=[CLASSES.index(gt[m['gt']]['label']) for m in matches]
        dev_features.extend(feats[idx]); dev_labels.extend(target)
        plates.append((row, feats, idx, target))
    if len(plates)!=12:
        raise ValueError('Expected 12 development plates')
    dev=np.asarray(dev_features); devy=np.asarray(dev_labels)
    domains={'real_train':(rx,ry),'detector_train':(dx,dy),'targeted_synthetic':(sh,sy),
             'reviewed_dev':(vx,vy),'matched_detector_dev':(dev,devy)}
    put(out/'hog.json',summarize_hog(domains,out))
    scaler=StandardScaler().fit(x); scaled=scaler.transform(x)

    def evaluate(model,s):
        if list(model.classes_)!=list(range(36)) or model.n_features_in_!=324:
            raise ValueError('Incompatible HOG/class mapping')
        def predict(f):
            return model.predict(s.transform(f)) if len(f) else np.empty(0,dtype=int)
        def metrics(t,p):
            cm=confusion_matrix(t,p,labels=range(36)); support=cm.sum(axis=1)
            return {'correct':int(np.trace(cm)),'total':len(t),'confusion':cm.tolist(),
                    'histogram':dict(Counter(CLASSES[i] for i in p)),
                    'recall':[float(cm[i,i]/n) if n else None for i,n in enumerate(support)]}
        records=[]
        for row,f,_,_ in plates:
            ref=''.join(g['label'] for g in row['characters']); pred=''.join(CLASSES[i] for i in predict(f))
            records.append({'plate_id':row['image_id'],'group':row['source_group'],'exact':pred==ref,'prediction':pred,'reference':ref,**edit_counts(ref,pred)})
        edits=sum(r[k] for r in records for k in ('substitutions','deletions','insertions'))
        return {'exact':sum(r['exact'] for r in records),'edits':edits,'cer':edits/len(vy),'plates':records,
                'gt_glyph':metrics(vy,predict(vx)),'matched_detector_glyph':metrics(devy,predict(dev))}

    prior=ROOT/'experiments/alpr_feature_correction'; v2=ROOT/'experiments/ocr_v2_200'
    result={'classes':list(CLASSES),'real_source_groups':len({r['source_group'] for r in buckets['train']}),
            'scope':'Historical same-cohort diagnosis, not independent testing, no winner selection',
            'x_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'gpu':gpu,'cuda_requests':timings,'cases':{}}
    for name,m,s in [('V1',WEIGHTS['svm'],WEIGHTS['scaler']),('V2',v2/'svm.joblib',v2/'scaler.joblib'),
                     ('saved_corrected',prior/'svm_balanced.joblib',prior/'scaler.joblib')]:
        saved=joblib.load(s); result['cases'][name]=evaluate(joblib.load(m),saved)
        if name=='saved_corrected':
            result['historical_scaler_equal']=bool(np.array_equal(saved.mean_,scaler.mean_) and np.array_equal(saved.var_,scaler.var_))
    for name,(cw,sw) in weight_cases(y,len(rx)+len(dx)).items():
        model=SVC(C=10,gamma='scale',class_weight=cw,random_state=42)
        t=perf_counter(); model.fit(scaled,y,sample_weight=sw); seconds=perf_counter()-t
        effective=10*model.class_weight_[y]*sw
        result['cases'][name]={**evaluate(model,scaler),'seconds_fit':seconds,'gamma':model._gamma,
            'class_weight':model.class_weight_.tolist(),'support_vectors':int(model.n_support_.sum()),
            'sample_weight_mean':float(sw.mean()),'effective_C_by_class':{c:{
                'count':int(len(e)), 'min':float(e.min()),'max':float(e.max()),'mean':float(e.mean())}
                for i,c in enumerate(CLASSES) for e in [effective[y==i]]}}
        put(out/'svm_ablation.json',result)
        print(name,result['cases'][name]['exact'],result['cases'][name]['edits'],flush=True)
    result['historical_predictions_equal']=result['cases']['D_joint_historical']['plates']==result['cases']['saved_corrected']['plates']
    put(out/'svm_ablation.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    from pathlib import Path
    parser.add_argument('output',type=Path)
    main(parser.parse_args().output)