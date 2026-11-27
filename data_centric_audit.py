"""Data-centric audit: phân phối nguồn, lớp, crop policy, thuộc tính pixel giữa real/synthetic/detector."""
from __future__ import annotations
import csv, json
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np

from glyph_review import CLASSES, ROOT
from dataset import extract_hog_features, CHARACTER_CLASSES
from final_ocr_dataset import read

REAL=ROOT/'reports/glyph_audit_20261125T145221295978Z/dataset/characters.csv'
REAL_DIR=ROOT/'reports/glyph_audit_20261125T145221295978Z/dataset'
CHAR=ROOT/'datasets/yolo_character_round1/manifest.json'
SYNTH=ROOT/'experiments/ocr_v2_200/synthetic.npz'
OUT=ROOT/'reports/data_centric_audit'

def load_real():
    rows=[]
    with REAL.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def glyph_attrs(img_gray):
    if img_gray.ndim==3:
        img_gray=cv2.cvtColor(img_gray,cv2.COLOR_BGR2GRAY)
    _,bin=cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    edge=np.concatenate((bin[0],bin[-1],bin[1:-1,0],bin[1:-1,-1]))
    if np.mean(edge>0)>.5:
        bin=cv2.bitwise_not(bin)
    h,w=bin.shape
    ys,xs=np.where(bin>0)
    if len(xs)==0:
        return {'fg':0,'aspect':0,'fill':0,'width':0,'height':0,'center_x':0,'center_y':0,'intensity_mean':float(img_gray.mean()),'intensity_std':float(img_gray.std())}
    cx=float(xs.mean())/w; cy=float(ys.mean())/h
    bw=float(xs.max()-xs.min()+1)/w; bh=float(ys.max()-ys.min()+1)/h
    return {'fg':float((bin>0).mean()),'aspect':bw/max(bh,1e-6),'fill':float((bin>0).sum()/(bin.size)),'width':bw,'height':bh,'center_x':cx,'center_y':cy,'intensity_mean':float(img_gray.mean()),'intensity_std':float(img_gray.std())}

def hog_stats(images):
    feats_list=[]
    for i in images:
        if i.ndim==3:
            i=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
        feats_list.append(extract_hog_features(i))
    if not feats_list:
        return {'mean':0,'std':0,'min':0,'max':0,'q25':0,'q75':0}
    feats=np.stack(feats_list)
    return {'mean':float(feats.mean()),'std':float(feats.std()),'min':float(feats.min()),'max':float(feats.max()),'q25':float(np.percentile(feats,25)),'q75':float(np.percentile(feats,75))}

def pct(vals):
    vals=[v for v in vals if v is not None]
    if not vals:
        return [0,0,0]
    return [round(float(v),4) for v in np.percentile(vals,[5,50,95])]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    real=load_real()
    by_group=Counter(); by_class=Counter(); by_split=Counter(); attrs=defaultdict(list); hog_real=[]
    for r in real:
        by_group[r['source_group']]+=1; by_class[r['class_label']]+=1; by_split[r['split']]+=1
        img=cv2.imread(str(REAL_DIR/r['image_path']),0)
        if img is None:
            continue
        attrs[r['split']].append(glyph_attrs(img))
        hog_real.append(extract_hog_features(img))
    synth=np.load(SYNTH); sx=synth['pixels']; sy=synth['labels']
    attrs['synthetic']=[glyph_attrs(cv2.cvtColor(i,cv2.COLOR_GRAY2BGR)) for i in sx]
    hog_synth=[extract_hog_features(i) for i in sx]
    report={'real_count':len(real),'synthetic_count':int(len(sx)),'by_group':dict(by_group),'by_class':dict(by_class),'by_split':dict(by_split),
            'class_order':list(CLASSES),'missing_classes':sorted(set(CLASSES)-set(by_class)),
            'sparse_classes':sorted(c for c,n in by_class.items() if n<3),
            'attr':{k:{field:pct([d[field] for d in vs]) for field in ('fg','aspect','fill','width','height','center_x','center_y','intensity_mean','intensity_std')} for k,vs in attrs.items()},
            'hog_real':hog_stats(hog_real),
            'hog_synth':hog_stats(hog_synth)}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'class_by_split.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(['class','train','validation','synthetic','total'])
        for c in CLASSES:
            tr=sum(1 for r in real if r['class_label']==c and r['split']=='train')
            va=sum(1 for r in real if r['class_label']==c and r['split']=='validation')
            sy_n=int(np.sum(sy==CLASSES.index(c)))
            w.writerow([c,tr,va,sy_n,tr+va+sy_n])
    print(json.dumps({'groups':len(by_group),'classes':len(by_class),'missing':report['missing_classes'],'sparse':report['sparse_classes']},indent=2))

if __name__=='__main__':main()