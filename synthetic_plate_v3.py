"""Procedural train-only plates, with pixel masks and a shared homography."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from glyph_review import CLASSES
from yolo_character_pipeline import normalize_glyph


def render(rng, two, forced):
    w,h=(224,112) if two else (290,64)
    plate=np.full((h,w,3),int(rng.integers(176,244)),np.uint8)
    cv2.rectangle(plate,(2,2),(w-3,h-3),(50,50,50),2)
    for x in (9,w-10):cv2.circle(plate,(9 if x==9 else x,8),3,(90,90,90),-1)
    # Research layouts only; no claim of regulatory coverage.
    prefix=f'{rng.integers(10,100)}{CLASSES[forced]}'
    suffix=str(rng.integers(10000,100000))
    lines=[prefix,suffix] if two else [prefix+suffix]
    masks=[]; chars=[]
    font=int(rng.choice([cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_DUPLEX])); thick=int(rng.integers(2,4))
    for line,text in enumerate(lines):
        step=min(34,(w-24)//len(text)); baseline=38+line*53
        for i,c in enumerate(text):
            mask=np.zeros((h,w),np.uint8);scale=float(rng.uniform(.78,1.00));tw=cv2.getTextSize(c,font,scale,thick)[0][0]
            cv2.putText(mask,c,((w-step*len(text))//2+i*step+(step-tw)//2,baseline),font,scale,255,thick,cv2.LINE_8)
            plate[mask>0]=int(rng.integers(8,45)); masks.append(mask);chars.append((c,line))
    src=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    dst=src*np.float32(rng.uniform(.60,.96))+[36,34]+rng.uniform(-6,6,(4,2))
    transform=cv2.getPerspectiveTransform(src,dst.astype(np.float32)); size=(360,180)
    bg=np.full((180,360,3),int(rng.integers(45,140)),np.uint8)
    warped=cv2.warpPerspective(plate,transform,size)
    foreground=cv2.warpPerspective(np.full((h,w),255,np.uint8),transform,size,flags=cv2.INTER_NEAREST)>0
    bg[foreground]=warped[foreground]
    glyphs=[]
    for mask,(c,line) in zip(masks,chars):
        warped=cv2.warpPerspective(mask,transform,size,flags=cv2.INTER_NEAREST)
        x,y,bw,bh=cv2.boundingRect(warped)
        if min(bw,bh)<2:
            return render(rng,two,forced)
        glyphs.append({'class':c,'row':line,'order':len(glyphs),'xywh':[x,y,bw,bh]})
    yy,xx=np.indices(bg.shape[:2]); illumination=1+rng.uniform(-.25,.25)*xx/360-.16*(yy>rng.integers(30,150))
    bg=np.clip(bg*illumination[...,None]+rng.normal(0,rng.uniform(0,4),bg.shape),0,255).astype(np.uint8)
    if rng.random()<.45:bg=cv2.GaussianBlur(bg,(3,3),.65)
    elif rng.random()<.35:bg=cv2.filter2D(bg,-1,np.ones((1,3),np.float32)/3)
    encoded=cv2.imencode('.jpg',bg,[cv2.IMWRITE_JPEG_QUALITY,int(rng.integers(55,96))])[1]
    box=list(cv2.boundingRect(foreground.astype(np.uint8)))
    return encoded,{'text':prefix+suffix,'layout':'two' if two else 'one','glyphs':glyphs,'plate_xywh':box,
                    'quad':dst.tolist(),'homography':transform.tolist()}


def generate(root,count=720,seed=42):
    root.mkdir(parents=True,exist_ok=False); (root/'images').mkdir(); rng=np.random.default_rng(seed)
    rows=[]; crops=[]; labels=[]
    for i in range(count):
        encoded,row=render(rng,i%2==0,i%36); path=f'images/{i:05}.jpg'
        (root/path).write_bytes(encoded.tobytes()); im=cv2.imdecode(encoded,1)
        row.update(image=path,sha256=hashlib.sha256(encoded).hexdigest(),split='train',source_group=f'synthetic-{seed}-{i}')
        rows.append(row)
        for g in row['glyphs']:
            x,y,w,h=g['xywh']; p=int(rng.integers(0,3))
            crops.append(normalize_glyph(im[max(0,y-p):y+h+p,max(0,x-p):x+w+p]));labels.append(CLASSES.index(g['class']))
    (root/'manifest.json').write_text(json.dumps({'seed':seed,'font':'OpenCV built-in Hershey Simplex/Duplex; procedural backgrounds; no external font/photo',
              'class_order':list(CLASSES),'glyph_arrays_order':'manifest row then glyph order','rows':rows},indent=2),encoding='utf-8')
    np.savez_compressed(root/'glyphs.npz',x=np.stack(crops),y=np.array(labels))
    sheet=[]
    for row in rows[:16]:
        im=cv2.imread(str(root/row['image']))
        for box in [row['plate_xywh']]+[g['xywh'] for g in row['glyphs']]:
            x,y,w,h=box;cv2.rectangle(im,(x,y),(x+w-1,y+h-1),(0,255,0),1)
        sheet.append(im)
    if sheet:cv2.imwrite(str(root/'contact.png'),np.vstack(sheet))
    return validate(root)


def validate(root):
    meta=json.loads((root/'manifest.json').read_text()); rows=meta['rows']; expected=[]
    if meta['class_order']!=list(CLASSES) or not rows:raise ValueError('Invalid manifest')
    seen=set()
    for r in rows:
        p=root/r['image']; im=cv2.imread(str(p))
        if im is None or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256'] or r['split']!='train' or r['source_group'] in seen:
            raise ValueError('Image/hash/split/group mismatch')
        seen.add(r['source_group']); h,w=im.shape[:2]
        px,py,pw,ph=r['plate_xywh']
        if not (0<=px<px+pw<=w and 0<=py<py+ph<=h):raise ValueError('Invalid plate box')
        if ''.join(g['class'] for g in r['glyphs'])!=r['text']:raise ValueError('Text/order mismatch')
        if [g['order'] for g in r['glyphs']]!=list(range(len(r['glyphs']))):raise ValueError('Invalid order indices')
        for g in r['glyphs']:
            x,y,bw,bh=g['xywh']
            if not (px<=x<x+bw<=px+pw and py<=y<y+bh<=py+ph and min(bw,bh)>=2):raise ValueError('Invalid glyph box')
            expected.append(CLASSES.index(g['class']))
    with np.load(root/'glyphs.npz',allow_pickle=False) as data:
        if data['x'].shape!=(len(expected),32,32) or data['x'].dtype!=np.uint8 or data['y'].tolist()!=expected or not np.all(data['x'].any(axis=(1,2))):
            raise ValueError('Glyph crop/label contract mismatch')
    return {'plates':len(rows),'glyphs':len(expected),'classes':len(set(expected))}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--validate',action='store_true');p.add_argument('--count',type=int,default=720)
    a=p.parse_args();print(validate(a.root) if a.validate else generate(a.root,a.count))