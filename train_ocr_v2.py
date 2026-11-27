"""Isolated train-only classifier experiment; no inference on test images."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from acceptance_pilot import edit_counts
from dataset import extract_hog_features
from final_ocr_dataset import ROOT, check_protocol, read, conflicts, audit_candidates
from dataset_audit import fingerprint
from glyph_review import CLASSES, normalized_text
from ocr_mix_experiment import contact
from ocr_v2_synthetic import generate
from validate_glyph_dataset import validate
from yolo_character_dataset import validate as validate_character, sha
from yolo_character_pipeline import MixClassifiers, normalize_glyph, WEIGHTS

REAL=ROOT/'reports/glyph_audit_20261125T145221295978Z/dataset'
CHAR=ROOT/'datasets/yolo_character_round1'
CFG=ROOT/'configs/ocr_v2_preregistered.json'


def put(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def prepare():
    check_protocol(); validate(REAL); validate_character(CHAR)
    exported={r['image_id']:r for r in read(REAL/'ocr_revisions.json') if r['split'] in ('train','val')}
    expected={r['review']['image_id']:r['review'] for r in read(CHAR/'manifest.json')['rows']}
    if exported!=expected:raise ValueError('Glyph snapshot and validated Character revisions differ')
    a=audit_candidates(write=False)
    forbidden=[{**r,'source_group':r.get('metadata',{}).get('source_group','')} for r in a['rows']]
    forbidden.extend(r for r in a['known'] if r['split']=='test')
    for row in exported.values():
        if row['split']=='train':
            d,p,shape=fingerprint(ROOT/row['image_path'])
            if not shape or conflicts(d,p,row['source_group'],forbidden):raise ValueError('Test exposure conflict: '+row['image_path'])
    with (REAL/'characters.csv').open(encoding='utf-8-sig',newline='') as f:
        buckets={s:[] for s in ('train','validation')}
        for row in csv.DictReader(f):
            if row['split'] not in buckets:continue
            rev=exported[row['plate_id']]
            if (row['source_image_id'],row['source_group'],row['split'])!=(rev['source_image_id'],rev['source_group'],'train' if rev['split']=='train' else 'validation'):
                raise ValueError('Glyph manifest provenance mismatch')
            buckets[row['split']].append({**row,'sha256':sha(REAL/row['image_path'])})
    if tuple(map(len,buckets.values()))!=(370,99):raise ValueError('Unexpected glyph inventory')
    return buckets,exported


def weights(s,r):
    total=(s+r)/2
    return np.r_[np.full(s,total/s),np.full(r,total/r)]


def hog(images):return np.stack([extract_hog_features(x) for x in images])


def classify(model,x):
    h=hog(x); start=perf_counter()
    p={'svm':model.svm.predict(model.scaler.transform(h)), 'rf':model.rf.predict(h),
       'cnn':np.asarray(model.cnn(x[...,None],training=False)).argmax(1)}
    from ensemble_recognition import vote
    texts={k:''.join(CLASSES[i] for i in v) for k,v in p.items()}
    return {**texts,'ensemble':vote(texts).raw_text},perf_counter()-start


def char_metrics(model,x,y):
    text,seconds=classify(model,x); ref=''.join(CLASSES[i] for i in y);result={}
    for name,p in text.items():
        supported=sorted(set(ref)); per=classification_report(list(ref),list(p),labels=supported,output_dict=True,zero_division=0)
        result[name]={'correct':sum(a==b for a,b in zip(ref,p)),'total':len(y),'prediction':p,'reference':ref,
                      'macro_supported':per['macro avg'],'per_class':per,'unsupported':sorted(set(CLASSES)-set(ref)),
                      'confusion_order':list(CLASSES)+'?'.split(),'confusion':confusion_matrix(list(ref),list(p),labels=list(CLASSES)+['?']).tolist()}
    return {'models':result,'predict_all_seconds_excludes_hog':seconds}


def plate_metrics(models,rows):
    # Use precisely the original evaluation boxes, not GT crops or retuned detectors.
    artifact=read(ROOT/'experiments/yolo_character_round1/comparison/results.json')
    records=[]
    for r in artifact['records']:
        if r['scope']!='yolo' or r['method']!='character_yolo':continue
        row=rows[r['id']]
        image=cv2.imread(str(ROOT/row['image_path'])); ref=normalized_text(row['plate_text'])
        boxes=r['pred_boxes']; glyphs=np.stack([normalize_glyph(image[y:y+h,x:x+w]) for x,y,w,h in boxes]) if boxes else None
        one={'image':row['image_path'],'reference':ref,'geometry':r['geometry'],'outputs':{}}
        for name,model in models.items():
            start=perf_counter()
            strings,_=classify(model,glyphs) if glyphs is not None else (dict.fromkeys(('svm','rf','cnn','ensemble'),''),0)
            one['outputs'][name]={'ocr_seconds':perf_counter()-start,'backends':{k:{'text':v,'exact':v==ref,**edit_counts(ref,v)} for k,v in strings.items()}}
        records.append(one)
    if len(records)!=12:raise ValueError('Expected 12 development plates')
    summary={}
    for name in models:
        summary[name]={}
        for backend in ('svm','rf','cnn','ensemble'):
            scores=[r['outputs'][name]['backends'][backend] for r in records]
            edits=sum(sum(s[k] for k in ('substitutions','deletions','insertions')) for s in scores)
            summary[name][backend]={'exact':sum(s['exact'] for s in scores),'plates':len(scores),'edits':edits,
                                    'cer':edits/sum(len(r['reference']) for r in records)}
        summary[name]['ocr_all_median_seconds']=float(np.median([r['outputs'][name]['ocr_seconds'] for r in records]))
    return {'summary':summary,'records':records,'timing_scope':'warm cached-box OCR incl HOG, not end-to-end latency'}


def main(out):
    buckets,rows=prepare(); config=read(CFG); out.mkdir(parents=True,exist_ok=False)
    put(out/'real_manifest.json',{'glyph_snapshot_sha256':sha(REAL/'ocr_revisions.json'),'buckets':buckets})
    put(out/'config.json',config)
    def batch(s):return np.stack([cv2.imread(str(REAL/r['image_path']),0) for r in buckets[s]]),np.array([CLASSES.index(r['class_label']) for r in buckets[s]])
    x,y=batch('train'); vx,vy=batch('validation')
    groups=np.array([r['source_group'] for r in buckets['train']]); folds=list(GroupKFold(3).split(x,y,groups))
    legacy=read(ROOT/'experiments/ocr_mix_bundle_20261125/manifest.json')
    v1=MixClassifiers()
    baseline=plate_metrics({'v1':v1},rows)
    put(out/'baseline_audit_before_fit.json',{'legacy_real_train':sum(r['origin']=='reviewed_real' and r['split']=='train' for r in legacy['manifest']),
        'model_hashes':{k:sha(v) for k,v in WEIGHTS.items()},'real_class_counts':dict(Counter(CLASSES[i] for i in y)),
        'baseline':baseline})
    print('Baseline audit saved before fit',flush=True)
    sx,sy=generate(config['synthetic_per_class'],config['seed']); np.savez_compressed(out/'synthetic.npz',pixels=sx,labels=sy)
    put(out/'synthetic_manifest.json',{'seed':42,'class_order':list(CLASSES),'font':'OpenCV built-in Hershey strokes (OpenCV distribution license; not official VN fonts)',
        'generator_sha256':sha(ROOT/'ocr_v2_synthetic.py'),'arrays_sha256':sha(out/'synthetic.npz'),'counts':dict(Counter(CLASSES[i] for i in sy)),
        'blank':int(sum(not im.any() for im in sx))})
    if any(not im.any() for im in sx):raise ValueError('Blank synthetic glyph')
    contact(sx,sy,out/'synthetic_contact.png'); h=hog(x); sh=hog(sx); fitted={}; cv_report={}
    for name,cls in [('svm',SVC),('rf',RandomForestClassifier)]:
        def fit(indices,params):
            features=np.concatenate([sh,h[indices]]); target=np.r_[sy,y[indices]]; w=weights(len(sy),len(indices))
            scaler=StandardScaler().fit(features) if name=='svm' else None
            model=cls(random_state=42,**({'n_jobs':4} if name=='rf' else {}),**params)
            model.fit(scaler.transform(features) if scaler else features,target,sample_weight=w);return model,scaler
        scores=[]
        for params in config[name]:
            fold=[]
            for train,val in folds:
                m,s=fit(train,params);fold.append(float(np.mean(m.predict(s.transform(h[val]) if s else h[val])==y[val])))
            scores.append({'params':params,'fold_accuracy':fold,'mean':float(np.mean(fold))})
        best=max(scores,key=lambda r:r['mean']); m,s=fit(np.arange(len(y)),best['params']);fitted[name]=m
        joblib.dump(m,out/f'{name}.joblib')
        if s is not None:joblib.dump(s,out/'scaler.joblib');fitted['scaler']=s
        cv_report[name]={'search':scores,'selected':best};print('Completed',name,flush=True)
    put(out/'group_cv.json',cv_report)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    conf=config['cnn']; train,val=folds[0]
    def cnn():
        model=tf.keras.models.load_model(WEIGHTS['cnn'],compile=False)
        model.compile(optimizer=tf.keras.optimizers.Adam(conf['learning_rate']),loss='sparse_categorical_crossentropy');return model
    start=perf_counter(); model=cnn()
    def fit_cnn(model,indices,epochs,**kwargs):
        return model.fit(np.concatenate([sx,x[indices]])[...,None],np.r_[sy,y[indices]],sample_weight=weights(len(sy),len(indices)),
                         batch_size=conf['batch'],epochs=epochs,verbose=2,**kwargs)
    history=fit_cnn(model,train,conf['epochs'],validation_data=(x[val,...,None],y[val]),callbacks=[tf.keras.callbacks.EarlyStopping(patience=conf['patience'])])
    epoch=int(np.argmin(history.history['val_loss']))+1
    model=cnn(); final=fit_cnn(model,np.arange(len(y)),epoch)
    model.save(out/'cnn.keras');fitted['cnn']=model
    put(out/'cnn_training.json',{'tensorflow':tf.__version__,'gpus':[str(d) for d in tf.config.list_physical_devices('GPU')],
        'seconds':perf_counter()-start,'selected_epochs':epoch,'group_holdout_history':history.history,'full_train_history':final.history})
    v2=SimpleNamespace(svm=joblib.load(out/'svm.joblib'),rf=joblib.load(out/'rf.joblib'),scaler=joblib.load(out/'scaler.joblib'),cnn=tf.keras.models.load_model(out/'cnn.keras',compile=False))
    glyph={k:char_metrics(v,vx,vy) for k,v in {'v1':v1,'v2':v2}.items()};put(out/'glyph_validation.json',glyph)
    result=plate_metrics({'v1':v1,'v2':v2},rows)
    a,b=[result['summary'][k]['cnn'] for k in ('v1','v2')]
    result['experimental_ui_gate']=b['exact']>a['exact'] and b['cer']<=a['cer']
    put(out/'plate_development.json',result)
    put(out/'checkpoint_hashes.json',{n:sha(out/n) for n in ('svm.joblib','rf.joblib','scaler.joblib','cnn.keras')})
    print('RESULT',result['summary'],result['experimental_ui_gate'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);main(parser.parse_args().output)