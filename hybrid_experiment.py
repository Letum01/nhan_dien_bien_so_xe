"""Controlled frozen-feature experiment on train groups and development only."""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dual_yolo_client import Worker
from hybrid_features import MODELS, V2, V3
from multiclass_run import edit_distance
from segmentation_benchmark import match_boxes, overlap_fraction
from svm_recovery_audit import crop, image
from train_ocr_v2 import CHAR, REAL, ROOT, CLASSES, hog, prepare, put, read, sha, weights
from yolo_character_pipeline import PYTHON, WEIGHTS, normalize_glyph, order_boxes

SETS = {'hog':('hog',), 'v1_pool':('v1_pool',), 'v1_dense':('v1_embedding',),
        'fusion':('hog','v1_embedding'), 'v2_dense':('v2_embedding',),
        'v3_pool':('v3_pool',), 'v3_flat':('v3_embedding',)}


def collect(out):
    buckets,reviews=prepare()
    out.mkdir(exist_ok=False)
    ref=ROOT/'experiments/yolo_character_round1/comparison/results.json'
    syn=V2/'synthetic.npz'
    if read(V2/'synthetic_manifest.json')['arrays_sha256']!=sha(syn):raise ValueError('Synthetic snapshot changed')
    for file,digest in read(V2/'checkpoint_hashes.json').items():
        if sha(V2/file)!=digest:raise ValueError('V2 checkpoint changed')
    if sha(MODELS['v3'])!=read(V3/'training.json')['robust']['sha256']:raise ValueError('V3 checkpoint changed')
    put(out/'protocol.json',{'features':SETS,'seed':42,'synthetic':'First 50 per class from fixed V2 archive',
        'fit':'3 GroupKFold, synthetic fixed, domain 50/50 mean weight 1; detector variants grouped with originals',
        'selection':'For each feature set and model choose hyperparameters by mean fold macro F1 on GT holdout. No dev selection.',
        'svm':[('linear',1),('rbf',10)],'rf':[(12,2),(22,1)],
        'limitations':'Head-only CV. Frozen CNN, detector and synthetic archive may have prior exposure to held-out folds.',
        'sources':{str(p):sha(p) for p in (*MODELS.values(),ref,syn,CHAR/'manifest.json',Path(__file__))}})
    xs=[]; rows=[]; plates=[]
    def add(x, domain, label, **metadata):
        i=len(xs);xs.append(x);rows.append(dict(domain=domain,label=label,**metadata));return i
    for split,rs in buckets.items():
        for r in rs:
            add(image(REAL/r['image_path'],0),split,r['class_label'],group=r['source_group'],
                parent_sha256=r['source_image_id'],plate_id=r['plate_id'],revision=r['ocr_revision'],order=r['order'])
    with np.load(syn,allow_pickle=False) as d:
        for k,c in enumerate(CLASSES):
            ids=np.flatnonzero(d['labels']==k)[:50]
            if len(ids)!=50:raise ValueError('Insufficient synthetic support')
            for i in ids:add(d['pixels'][i],'synthetic',c,group=f'synthetic_render_{i}',parent_sha256=None,archive_sha256=sha_cached,render_index=int(i))
    rejects=[]; peaks=[]
    with Worker() as worker:
        for row in reviews.values():
            if row['split']!='train':continue
            im=image(ROOT/row['image_path']); p=worker.request('plate',im)
            if not p['rows']:rejects.append({'image':row['image_id'],'reason':'plate miss'});continue
            from benchmark_crop_padding import padded_box
            b=padded_box(max(p['rows'],key=lambda r:r['score'])['bbox'],.15,im.shape); x,y=b[:2]
            q=worker.request('character',crop(im,b));peaks.append(q['peak_reserved']); px,py=row['plate_bbox'][:2]
            gs=row['characters']; truth=[[px+a,py+d,w,h] for a,d,w,h in (g['bbox'] for g in gs)]
            boxes=[[x+a,y+d,w,h] for a,d,w,h in (r['bbox'] for r in order_boxes(q['rows']))]
            match=match_boxes(boxes,truth,.5)
            for m in match['matches']:
                j=m['gt']; box=boxes[m['pred']]; coverage=overlap_fraction(truth[j],box)
                neighbor=max((overlap_fraction(g,box) for k,g in enumerate(truth) if k!=j),default=0.)
                if coverage<.85 or neighbor>.10:
                    rejects.append({'image':row['image_id'],'reason':'target clipping or adjacent glyph','match':m});continue
                add(normalize_glyph(crop(im,box)),'detector_train',gs[j]['label'],group=row['source_group'],
                    parent_sha256=row['source_image_id'],plate_id=row['image_id'],gt_box=truth[j],det_box=box,iou=m['iou'],coverage=coverage,other_glyph_coverage=neighbor)
            if match['fp'] or match['fn']:rejects.append({'image':row['image_id'],'unmatched':{k:match[k] for k in ('fp','fn')}})
    for r in read(ref)['records']:
        if (r['scope'],r['method'])!=('yolo','character_yolo'):continue
        row=reviews[r['id']]
        if row['split']!='val':raise ValueError('Unexpected benchmark split')
        im=image(ROOT/row['image_path']); matches={m['pred']:m['gt'] for m in r['geometry']['matches']}
        ids=[add(normalize_glyph(crop(im,b)),'detector_dev',r['reference'][matches[j]] if j in matches else None,
                 group=row['source_group'],parent_sha256=row['source_image_id'],plate_id=r['id'],det_box=b)
             for j,b in enumerate(r['pred_boxes'])]
        plates.append(dict(ids=ids,id=r['id'],reference=r['reference'],baseline=r['ocr']['cnn']['prediction']))
    np.save(out/'pixels.npy',np.stack(xs),allow_pickle=False)
    put(out/'samples.json',dict(samples=rows,plates=plates,rejects=rejects,detector_peak_reserved=max(peaks,default=0),pixels_sha256=sha(out/'pixels.npy')))


def run(out):
    meta=read(out/'samples.json'); rows=meta['samples']; plates=meta['plates']
    pixels=np.load(out/'pixels.npy',allow_pickle=False)
    if sha(out/'pixels.npy')!=meta['pixels_sha256']:raise ValueError('Pixels changed')
    domains=np.array([r['domain'] for r in rows]); groups=np.array([r['group'] for r in rows]); y=np.array([CLASSES.index(r['label']) if r['label'] else -1 for r in rows])
    ix=lambda d:np.flatnonzero(domains==d)
    gt,det,syn,vg,vd=map(ix,('train','detector_train','synthetic','validation','detector_dev'))
    if (len(gt),len(vg),len(plates),sum(y[vd]>=0))!=(370,99,12,98):raise ValueError('Development/train inventory mismatch')
    all_fit=np.r_[syn,gt,det]; eval_ids=np.r_[vg,vd]; supported=np.unique(y[vg]); report={}; features={}; desc={}
    t=perf_counter(); features['hog']=hog(pixels); desc['hog_glyphs_per_second']=len(pixels)/(perf_counter()-t)
    for name in MODELS:
        file=out/f'{name}_features.npz'
        if not file.exists():
            subprocess.run([str(PYTHON) if name=='v3' else sys.executable,'-u',str(ROOT/'hybrid_features.py'),name,str(out)],check=True,cwd=ROOT)
        extraction=read(out/f'{name}_extract.json')
        if extraction['sha256']!=sha(MODELS[name]):raise ValueError('Stale feature extraction')
        with np.load(file,allow_pickle=False) as data:
            for k in ('pool','embedding'):features[f'{name}_{k}']=data[k]
            report[f'cnn_{name}']=score(data['prediction'][eval_ids],eval_ids,y,vg,vd,plates)
            render_maps(pixels,data['maps'],out/f'{name}_maps.png')
    np.save(out/'hog.npy',features['hog'],allow_pickle=False)
    for name,f in features.items():
        if f.dtype!=np.float32 or len(f)!=len(y) or not np.isfinite(f).all():raise ValueError(f'Bad features {name}')
        desc[name]={'dimension':f.shape[1],'dtype':str(f.dtype),'domains':{d:{'n':len(ix(d)),
            'norm_q':np.percentile(np.linalg.norm(f[ix(d)],axis=1),[5,50,95]).tolist(),
            'constant':int((f[ix(d)].std(0)<1e-8).sum())} for d in np.unique(domains)},
            **visualize(f,name,gt,syn,vg,y,out)}
    put(out/'distributions.json',desc)
    for version in ('v1','v2'):
        s=joblib.load(WEIGHTS['scaler'] if version=='v1' else V2/'scaler.joblib')
        h=features['hog'][eval_ids]
        for family in ('svm','rf'):
            model_path=WEIGHTS[family] if version=='v1' else V2/f'{family}.joblib'
            model=joblib.load(model_path); t=perf_counter();p=model.predict(s.transform(h) if family=='svm' else h)
            report[f'{family}_{version}']={**score(p,eval_ids,y,vg,vd,plates),'classifier_eval_ms':1000*(perf_counter()-t),'model_bytes':model_path.stat().st_size}
    expected=(8,4)
    if (report['cnn_v1']['exact'],report['cnn_v1']['edits'])!=expected:raise ValueError('V1 replay mismatch')
    folds=list(GroupKFold(3).split(gt,y[gt],groups[gt]))
    put(out/'folds.json',[{'fit_groups':sorted(set(groups[gt[t]])), 'hold_groups':sorted(set(groups[gt[v]]))} for t,v in folds])
    for key,columns in SETS.items():
        x=np.concatenate([features[c] for c in columns],axis=1)
        for family in ('svm','rf'):
            options=[('linear',1),('rbf',10)] if family=='svm' else [(12,2),(22,1)]
            def fit(ids,p):
                w=weights(len(syn),len(ids)-len(syn)); assert np.isclose(w.mean(),1)
                scaler=StandardScaler().fit(x[ids]) if family=='svm' else None
                model=(SVC(kernel=p[0],C=p[1],gamma='scale',random_state=42) if scaler else
                       RandomForestClassifier(n_estimators=64,max_depth=p[0],min_samples_leaf=p[1],max_features='sqrt',random_state=42,n_jobs=4))
                model.fit(scaler.transform(x[ids]) if scaler else x[ids],y[ids],sample_weight=w)
                return model,scaler
            def predict(model,s,xx):return model.predict(s.transform(xx) if s else xx)
            cv=[]
            for params in options:
                scores=[]
                for train,val in folds:
                    hold=set(groups[gt[val]]); safe=det[~np.isin(groups[det],list(hold))]
                    ids=np.r_[syn,gt[train],safe]; assert not hold&set(groups[ids]); model,s=fit(ids,params)
                    scores.append(float(f1_score(y[gt[val]],predict(model,s,x[gt[val]]),labels=np.unique(y[gt[val]]),average='macro',zero_division=0)))
                cv.append(scores)
            chosen=int(np.argmax(np.mean(cv,axis=1))); t=perf_counter(); model,s=fit(all_fit,options[chosen]); elapsed=perf_counter()-t
            path=out/f'{family}_{key}.joblib';joblib.dump({'classifier':model,'scaler':s,'features':columns,'classes':list(CLASSES)},path,compress=3)
            measures=[]
            for _ in range(10):
                t=perf_counter();predict(model,s,x[eval_ids[:8]]);measures.append(perf_counter()-t)
            entry={**score(predict(model,s,x[eval_ids]),eval_ids,y,vg,vd,plates),'fold_macro_f1':cv,
                   'options':options,'chosen':options[chosen],'dimensions':x.shape[1],'fit_seconds':elapsed,
                   'model_bytes':path.stat().st_size,'head_batch8_median_ms':1000*float(np.median(measures))}
            if family=='rf':
                rng=np.random.default_rng(42); offset=0; imp={}
                for col in columns:
                    size=features[col].shape[1]; base=x[vg].copy(); baseline=(model.predict(base)==y[vg]).mean()
                    drops=[]
                    for _ in range(5):
                        perm=base.copy();perm[:,offset:offset+size]=base[rng.permutation(len(vg)),offset:offset+size]
                        drops.append(float(baseline-(model.predict(perm)==y[vg]).mean()))
                    imp[col]={'impurity_sum':float(model.feature_importances_[offset:offset+size].sum()),'dev_permutation_accuracy_drop':drops};offset+=size
                entry['group_importance_descriptive']=imp
            report[f'{family}_{key}']=entry;put(out/'results.json',report)
            print(family,key,entry['exact'],entry['edits'],flush=True)


def score(p,ids,y,vg,vd,plates):
    pred=dict(zip(ids,map(int,p),strict=True)); answer={}
    for name,indices in [('gt',vg),('detector',vd[y[vd]>=0])]:
        t=y[indices]; z=[pred[i] for i in indices]; cm=confusion_matrix(t,z,labels=range(36)); support=cm.sum(1)
        answer[name]={'correct':int(cm.trace()),'total':len(t),'macro_f1':float(f1_score(t,z,labels=np.unique(t),average='macro',zero_division=0)),
                      'recall':{c:float(cm[i,i]/support[i]) if support[i] else None for i,c in enumerate(CLASSES)},'confusion':cm.tolist()}
    result=[]
    for r in plates:
        text=''.join(CLASSES[pred[i]] for i in r['ids']); ref=r['reference']
        result.append({**r,'prediction':text,'exact':text==ref,'edits':edit_distance(ref,text)})
    edits=sum(r['edits'] for r in result)
    return {**answer,'exact':sum(r['exact'] for r in result),'edits':edits,'cer':edits/sum(len(r['reference']) for r in result),
            'fixed_vs_cnn_v1':sum(r['exact'] and r['baseline']!=r['reference'] for r in result),
            'regressed_vs_cnn_v1':sum(not r['exact'] and r['baseline']==r['reference'] for r in result),'plates':result}


def render_maps(pixels,maps,path):
    tiles=[]
    for im,f in zip(pixels,maps):
        imgs=[im]+[cv2.normalize(f[...,i],None,0,255,cv2.NORM_MINMAX).astype(np.uint8) for i in range(8)]
        tiles.append(np.hstack([cv2.resize(z,(64,64),interpolation=cv2.INTER_NEAREST) for z in imgs]))
    if not cv2.imwrite(str(path),np.vstack(tiles)):raise OSError(path)


def visualize(f,name,gt,syn,vg,y,out):
    pca=PCA(2,random_state=42,svd_solver='randomized').fit(f[np.r_[gt,syn]])
    proj=pca.transform(f); scale=np.maximum(np.percentile(np.abs(proj[gt]),98,axis=0),.01)
    canvas=np.full((560,780,3),245,np.uint8); focus='D0TIMKNA'
    colors=[(2,90,180),(50,150,0),(180,20,20),(95,0,140),(50,100,120),(110,0,0),(120,90,0),(0,0,160)]
    for c,color in zip(focus,colors):
        for i in np.r_[gt,vg]:
            if y[i]!=CLASSES.index(c):continue
            x,z=(np.clip(proj[i]/scale,-1,1)*[320,-217]+[390,265]).astype(int)
            cv2.circle(canvas,(x,z),4,color,-1 if i in gt else 1)
        cv2.putText(canvas,c,(35+focus.index(c)*55,525),0,.65,color,2)
    cv2.putText(canvas,f'{name}: train PCA; filled=train, ring=development; axes clipped', (20,29),0,.55,(0,0,0),1)
    if not cv2.imwrite(str(out/f'pca_{name}.png'),canvas):raise OSError(name)
    normal=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-8)
    def center(ids,c):return normal[ids[y[ids]==CLASSES.index(c)]].mean(0)
    real=[c for c in focus if np.any(y[gt]==CLASSES.index(c))]
    return {'pca_variance':pca.explained_variance_ratio_.tolist(),'focus_counts':{c:int(sum(y[gt]==CLASSES.index(c))) for c in focus},
            'real_centroid_l2_unit_vectors':{a+b:float(np.linalg.norm(center(gt,a)-center(gt,b))) for a in real for b in real if a<b},
            'real_synthetic_centroid_l2':{a:float(np.linalg.norm(center(gt,a)-center(syn,a))) for a in real}}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);args=ap.parse_args();out=args.output.resolve()
    sha_cached=sha(V2/'synthetic.npz')
    collect(out);run(out)