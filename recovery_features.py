"""Controlled HOG feature ablations; the deployed model contracts stay unchanged."""
import json
from collections import Counter

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.feature import local_binary_pattern
from skimage.morphology import skeletonize

from acceptance_pilot import edit_counts
from dataset import extract_hog_features, CHARACTER_CLASSES
from train_ocr_v2 import REAL, prepare, weights
from final_ocr_dataset import ROOT, read
from alpr_recovery import pixels

ARMS=('hog','lbp','zoning','geometry')


def features(im, arm):
    h=extract_hog_features(im); ink=im>0
    if arm=='hog':return h
    if arm=='lbp':
        code=local_binary_pattern(im,8,1,'uniform').astype(int)
        extra=np.concatenate([np.bincount(code[y:y+16,x:x+16].ravel(),minlength=10)/256 for y in (0,16) for x in (0,16)])
    elif arm=='zoning':
        extra=np.r_[ink.reshape(4,8,4,8).mean((1,3)).ravel(),ink.mean(0),ink.mean(1)]
    elif arm=='geometry':
        mask=ink.astype(np.uint8); m=cv2.moments(mask); hu=cv2.HuMoments(m).ravel()
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        width=cv2.distanceTransform(mask,cv2.DIST_L2,3)[skeletonize(ink)]
        extra=np.r_[np.sign(hu)*np.log1p(abs(hu)*1e6),cv2.connectedComponents(mask)[0]-1,len(contours),
                    ink.mean(),width.mean() if len(width) else 0,width.std() if len(width) else 0,
                    cv2.Sobel(im,cv2.CV_32F,1,0).__abs__().mean()/255,
                    cv2.Sobel(im,cv2.CV_32F,0,1).__abs__().mean()/255]
    else:raise ValueError(arm)
    return np.r_[h,extra].astype(np.float32)


def main():
    buckets,_=prepare(); out=ROOT/'reports/accuracy_recovery/feature_study.json'
    if out.exists():raise FileExistsError(out)
    def load(split):
        rows=buckets[split];return np.stack([cv2.imread(str(REAL/r['image_path']),0) for r in rows]),np.array([CHARACTER_CLASSES.index(r['class_label']) for r in rows])
    tr,y=load('train'); va,vy=load('validation');groups=[r['source_group'] for r in buckets['train']]
    source=np.load(ROOT/'experiments/ocr_v2_200/synthetic.npz',allow_pickle=False)
    ids=np.concatenate([np.flatnonzero(source['labels']==i)[:100] for i in range(36)])
    sx,sy=source['pixels'][ids],source['labels'][ids]
    records=[r for r in read(ROOT/'experiments/yolo_character_round1/comparison/results.json')['records'] if r['method']=='character_yolo' and r['scope']=='yolo']
    inputs=[]
    for r in records:
        im=cv2.imread(str(ROOT/r['image']));inputs.append([pixels(im[y:y+h,x:x+w],'raw') for x,y,w,h in r['pred_boxes']])
    result={}
    for arm in ARMS:
        def matrix(xs):return np.stack([features(x,arm) for x in xs])
        A,B,S=map(matrix,(tr,va,sx));G=[matrix(x) if x else None for x in inputs]
        result[arm]={}
        for name in ('svm','rf'):
            def fit(idx):
                T=np.r_[S,A[idx]]; scaler=StandardScaler().fit(T) if name=='svm' else None
                clf=SVC(C=10,gamma='scale') if scaler is not None else RandomForestClassifier(n_estimators=80,max_depth=20,n_jobs=4,random_state=42)
                clf.fit(scaler.transform(T) if scaler is not None else T,np.r_[sy,y[idx]],sample_weight=weights(len(sy),len(idx)))
                return lambda x:clf.predict(scaler.transform(x) if scaler is not None else x)
            folds=[float(np.mean(fit(a)(A[b])==y[b])) for a,b in GroupKFold(3).split(A,y,groups)]
            predict=fit(np.arange(len(y))); yp=predict(B)
            texts=[''.join(CHARACTER_CLASSES[i] for i in predict(g)) if g is not None else '' for g in G]
            refs=[r['reference'] for r in records]; edits=sum(sum(edit_counts(a,b).values()) for a,b in zip(refs,texts))
            result[arm][name]={'fold_accuracy':folds,'glyph_correct':int((yp==vy).sum()),'exact':sum(a==b for a,b in zip(refs,texts)),
                'cer':edits/sum(map(len,refs)),'texts':texts,'glyph_confusions':dict(Counter(f'{CHARACTER_CLASSES[a]}>{CHARACTER_CLASSES[b]}' for a,b in zip(vy,yp) if a!=b))}
        print(arm,result[arm],flush=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()