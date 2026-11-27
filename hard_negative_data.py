"""Train-only mining and reusable relative box diagnostics."""
import json
from pathlib import Path

import cv2
import numpy as np

from glyph_review import ROOT, CLASSES
from yolo_character_pipeline import normalize_glyph

OUT=ROOT/'experiments/hard_negative_v1'


def intersection(a,b,margin=0):
    x,y,w,h=a;u,v,s,t=b
    return max(0,min(x+w,u+s+margin)-max(x,u-margin))*max(0,min(y+h,v+t+margin)-max(y,v-margin))


def negative(box,truth,plate):
    x,y,w,h=box; a,b,c,d=plate
    return a<=x and b<=y and x+w<=a+c and y+h<=b+d and not any(intersection(box,g,2) for g in truth)


def diagnostics(boxes, images):
    items=[]
    for i,(x,y,w,h) in enumerate(boxes):
        row=[b for b in boxes if abs(b[1]+b[3]/2-y-h/2)<.45*max(h,b[3])]
        mh=float(np.median([b[3] for b in row])); mw=float(np.median([b[2] for b in row]))
        ink=(images[i]>0).astype(np.uint8); dist=cv2.distanceTransform(ink,cv2.DIST_L2,3)
        ios=[]
        for j,b in enumerate(boxes):
            if j!=i:
                a=intersection(boxes[i],b);ios.append(a/(w*h+b[2]*b[3]-a))
        near=[abs(x+w/2-b[0]-b[2]/2)/mh for b in row if b is not boxes[i]]
        items.append({'relative_height':h/mh,'relative_width':w/mw,'aspect':w/h,'density':float(ink.mean()),
                      'stroke_proxy':float(dist[ink>0].mean()/32) if ink.any() else 0.,
                      'components':cv2.connectedComponents(ink)[0]-1,'iou':max(ios,default=0.),
                      'center_distance':min(near,default=None),'row_offset':float(abs(y+h/2-np.median([b[1]+b[3]/2 for b in row]))/mh)})
    return items


def geometry_keep(d):
    # No width-based rejection: a slender vertical stroke is a valid glyph.
    return not (d['density']==0 or (d['relative_height']<.48 and d['aspect']>1.35))


def main():
    from train_ocr_v2 import prepare,REAL
    from alpr_recovery import worker,geometry
    from final_ocr_dataset import read
    from synthetic_plate_v3 import validate
    buckets,reviewed=prepare(); syn=ROOT/'experiments/synthetic_plate_v3_validated'; validate(syn)
    OUT.mkdir(exist_ok=False); images=[];labels=[];groups=[]; provenance=[];ambiguous=[]
    def add(im,label,group,kind,source,box=None):
        images.append(im);labels.append(label);groups.append(group)
        provenance.append({'class':label,'group':group,'kind':kind,'source':source,'box':box})
    for r in buckets['train']:
        add(cv2.imread(str(REAL/r['image_path']),0),CLASSES.index(r['class_label']),r['source_group'],'reviewed',r['image_path'])
    with np.load(syn/'glyphs.npz',allow_pickle=False) as a:
        sx,sy=a['x'],a['y']; pos=0
        # Keep a bounded subset of positive glyphs and retain their synthetic source IDs.
        counts=np.zeros(36,int)
        for row in read(syn/'manifest.json')['rows']:
            for j,g in enumerate(row['glyphs']):
                k=int(sy[pos+j]); assert CLASSES[k]==g['class']
                if counts[k]<20:add(sx[pos+j],k,row['source_group'],'synthetic_glyph',row['image'],g['xywh']);counts[k]+=1
            pos+=len(row['glyphs'])
    rng=np.random.default_rng(42)
    # Render isolated non-glyph motifs, guaranteed to contain no alphanumeric labels.
    for i in range(384):
        bg=np.full((int(rng.integers(14,40)),int(rng.integers(7,33)),3),int(rng.integers(150,250)),np.uint8)
        h,w=bg.shape[:2]; mid=(w//2,h//2); color=(45,45,45); kind=('dash','dot','screw','rim','shadow','background')[i%6]
        if kind=='dash':cv2.line(bg,(w//5,mid[1]),(w-w//5-1,mid[1]),color,int(rng.integers(1,3)))
        if kind in ('dot','screw'):cv2.circle(bg,mid,int(rng.integers(1,max(2,min(h,w)//3))),color,-1 if kind=='dot' else 1)
        if kind=='rim':cv2.line(bg,(0,0),(w-1,0),color,2)
        if kind=='shadow':bg[:h//3]//=2
        bg=np.clip(bg.astype(float)+rng.normal(0,5,bg.shape[:2])[...,None],0,255).astype(np.uint8)
        add(normalize_glyph(bg),36,f'noise-{i}',kind,'procedural')
    with worker() as client:
        for r in reviewed.values():
            if r['split']!='train':continue
            im=cv2.imread(str(ROOT/r['image_path']));x,y,w,h=r['plate_bbox']
            truth=[[x+a,y+b,c,d] for a,b,c,d in [g['bbox'] for g in r['characters']]]
            data,rois=geometry(im,client)
            if not data['plate']:continue
            px,py,_,_=data['plate']['crop_box']
            for g,roi in zip(data['glyphs'],rois):
                a,b,c,d=g['bbox']; box=[px+a,py+b,c,d]
                if negative(box,truth,r['plate_bbox']):
                    add(normalize_glyph(roi),36,r['source_group'],'real_detector_no_glyph_overlap',r['image_path'],box)
                elif not any(intersection(box,k)/(c*d+k[2]*k[3]-intersection(box,k))>=.5 for k in truth):
                    ambiguous.append({'source':r['image_path'],'box':box,'status':'AMBIGUOUS'})
    np.savez_compressed(OUT/'data.npz',x=np.stack(images),y=np.array(labels),groups=np.array(groups))
    (OUT/'manifest.json').write_text(json.dumps({'scope':'train only; synthetic negatives and reviewed-box-excluded detector samples',
        'rows':provenance,'ambiguous_excluded':ambiguous,'mapping':list(CLASSES)+['NON_CHARACTER']},indent=2),encoding='utf-8')
    print('saved',len(labels),'negatives',labels.count(36),'ambiguous',len(ambiguous),flush=True)


if __name__=='__main__':main()