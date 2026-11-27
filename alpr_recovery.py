"""Bounded research inference and pixel traces, separate from frozen V1."""
from collections import Counter
from time import perf_counter

import cv2
import numpy as np

from benchmark_crop_padding import padded_box
from dataset import extract_hog_features
from dual_yolo_client import Worker,PYTHON,ROOT,CHECKPOINT,CHARACTER
from ensemble_recognition import vote
from glyph_review import CLASSES
from yolo_character_pipeline import normalize_glyph,order_boxes

VARIANTS=('raw','clahe','gamma','sharpen','adaptive')


def worker():
    return Worker(timeout=20,command=[str(PYTHON),'-u',str(ROOT/'recovery_worker.py'),str(CHECKPOINT),str(CHARACTER)])


def pixels(roi,mode):
    if mode=='raw':return normalize_glyph(roi)
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    if mode=='clahe':gray=cv2.createCLAHE(2,(4,4)).apply(gray)
    elif mode=='gamma':gray=np.rint((gray/255)**.8*255).astype(np.uint8)
    elif mode=='sharpen':gray=cv2.addWeighted(gray,1.5,cv2.GaussianBlur(gray,(3,3),.7),-.5,0)
    elif mode=='adaptive':
        gray=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,9,3)
    else:raise ValueError(mode)
    return normalize_glyph(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR))


def classify(models,x,details=False):
    if not len(x):return dict.fromkeys(('svm','rf','cnn','ensemble'),''),[]
    X=np.asarray(x); H=np.stack([extract_hog_features(g) for g in X]); scaled=models.scaler.transform(H)
    c=np.asarray(models.cnn(X[...,None],training=False)); rf=models.rf.predict_proba(H)
    text={k:''.join(CLASSES[int(i)] for i in ids) for k,ids in {'svm':models.svm.predict(scaled),'rf':models.rf.predict(H),'cnn':c.argmax(1)}.items()}
    top=[]
    if details:
        svm=models.svm.predict_proba(scaled) if getattr(models.svm,'probability',False) else None
        for i in range(len(X)):
            top.append({k:[[CLASSES[int(j)],float(s[i,j])] for j in np.argsort(-s[i])[:3]]
                        for k,s in [('cnn',c),('rf',rf),('svm_probability',svm)] if s is not None})
    return {**text,'ensemble':vote(text).raw_text},top


def geometry(image,client,ps=640,cs=640,padding=.15):
    p=client.request(f'plate:{ps}',image);r={'plate_rows':p['rows'],'peak_reserved':p['peak_reserved'],'glyphs':[],'plate':None,
                                          'gpu_seconds':p['worker_seconds'],'cold':p['cold']}
    if not p['rows']:return r,[]
    best=max(p['rows'],key=lambda q:q['score']); x,y,w,h=padded_box(best['bbox'],padding,image.shape)
    crop=image[y:y+h,x:x+w]; c=client.request(f'character:{cs}',crop)
    r.update(plate={**best,'crop_box':[x,y,w,h]},glyphs=order_boxes(c['rows']),peak_reserved=c['peak_reserved'],
             gpu_seconds=r['gpu_seconds']+c['worker_seconds'])
    return r,[crop[b:b+d,a:a+e] for a,b,e,d in [q['bbox'] for q in r['glyphs']]]


def stable_candidate(raw,texts):
    """At least three agreeing pixel variants and two distinct model families at that position."""
    result=[];uncertain=[]
    for i,c in enumerate(raw):
        counts=Counter(t['cnn'][i] for t in texts); best,n=counts.most_common(1)[0]
        support={k for t in texts for k in ('svm','rf','cnn') if t[k][i]==best}
        if n>=3 and len(support)>=2:result.append(best)
        else:result.append(c)
        if len(counts)>1 or len({t[k][i] for t in texts for k in ('svm','rf','cnn')})>1:uncertain.append(i+1)
    return ''.join(result),uncertain


def analyze(image,client,models,ps=640,cs=640,padding=.15,variants=VARIANTS,limit_seconds=12):
    start=perf_counter(); data,rois=geometry(image,client,ps,cs,padding); runs={}; top=[]
    for mode in variants:
        if runs and perf_counter()-start>=limit_seconds:break  # soft deadline; IPC timeout is separate
        text,detail=classify(models,[pixels(im,mode) for im in rois],details=mode=='raw')
        runs[mode]=text
        if mode=='raw':top=detail
    raw=runs['raw']['cnn']; candidate,uncertain=stable_candidate(raw,list(runs.values()))
    return {**data,'runs':runs,'top_k':top,'raw':raw,'candidate':candidate,'uncertain_positions':uncertain,
            'seconds':perf_counter()-start,'budget_exhausted':len(runs)<len(variants)},rois