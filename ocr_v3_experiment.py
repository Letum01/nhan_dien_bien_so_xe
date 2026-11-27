"""Fixed development experiment; run with .venv-yolo (CUDA required)."""
import json
import hashlib
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from torch import nn

from glyph_review import ROOT, CLASSES
from final_ocr_dataset import check_protocol, read
from synthetic_plate_v3 import validate
from yolo_character_pipeline import normalize_glyph

OUT=ROOT/'experiments/ocr_v3_cuda_retry'
SYN=ROOT/'experiments/synthetic_plate_v3_validated'
REAL=ROOT/'reports/glyph_audit_20261125T145221295978Z/dataset'


def network():
    layers=[]; prev=1
    for width in (24,48,96):
        layers += [nn.Conv2d(prev,width,3,padding=1),nn.BatchNorm2d(width),nn.ReLU(),nn.MaxPool2d(2)]
        prev=width
    return nn.Sequential(*layers,nn.Flatten(),nn.Dropout(.2),nn.Linear(96*16,36))


def jitter(im,rng):
    # Scale down first so translations cannot trim essential strokes.
    k=float(rng.uniform(.80,.95)); x,y=rng.uniform(-.7,.7,2)
    return cv2.warpAffine(im,np.float32([[k,0,16*(1-k)+x],[0,k,16*(1-k)+y]]),(32,32))


def vote(strings,weights):
    if len({len(x) for x in strings})!=1:raise ValueError('Different glyph ordering/length')
    answer=[]
    for letters in zip(*strings):
        totals={c:sum(w for a,w in zip(letters,weights) if a==c) for c in set(letters)}
        best=[c for c,v in totals.items() if v==max(totals.values())]
        answer.append(best[0] if len(best)==1 else '?')
    return ''.join(answer)


def edits(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]


def write(name,data): (OUT/name).write_text(json.dumps(data,indent=2),encoding='utf-8')


def main():
    check_protocol(); audit=validate(SYN)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required; no CPU fallback')
    OUT.mkdir(exist_ok=False)
    write('preregistered.json',{'seed':42,'dataset':audit,'arms':['standard','robust'],'epochs_max':18,'batch':128,
        'split':'sorted train source groups every fifth group held out; select epoch by held-out real loss then refit all train',
        'criterion':'strict exact improvement and CER no higher than V1 CNN; no integration from repeated development alone',
        'fusion':['majority V1/V2','hybrid SVM/RF V2 + CNN V1','weighted hard votes 1/1/2','V2 SVM/RF + CNN V3 each arm']})
    revfile=REAL/'ocr_revisions.json'; manifest=read(ROOT/'experiments/ocr_v2_200/real_manifest.json')
    if manifest['glyph_snapshot_sha256']!=hashlib.sha256(revfile.read_bytes()).hexdigest():raise ValueError('Glyph snapshot changed')
    def load(split):
        rs=manifest['buckets'][split]
        imgs=[]
        for r in rs:
            path=(REAL/r['image_path']).resolve()
            if not path.is_relative_to(REAL) or r['split']!=split or r['annotation_status']!='human_reviewed' or hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:
                raise ValueError('Untrusted glyph')
            img=cv2.imread(str(path),0)
            if img is None or img.shape!=(32,32):raise ValueError('Invalid glyph')
            imgs.append(img)
        return np.stack(imgs),np.array([CLASSES.index(r['class_label']) for r in rs]),np.array([r['source_group'] for r in rs])
    x,y,g=load('train'); vx,vy,_=load('validation')
    assert len(x)==370 and len(vx)==99
    synthetic=np.load(SYN/'glyphs.npz',allow_pickle=False); sx,sy=synthetic['x'],synthetic['y']
    hold=np.isin(g,np.unique(g)[::5]); assert not set(g[hold])&set(g[~hold])
    old=read(ROOT/'experiments/ocr_v2_200/plate_development.json')['records']
    geom={r['image']:r['pred_boxes'] for r in read(ROOT/'experiments/yolo_character_round1/comparison/results.json')['records'] if r['scope']=='yolo' and r['method']=='character_yolo'}
    assert len(old)==12 and {r['image'] for r in old}==set(geom)
    metrics={}; preds={}
    def tensor(z):return torch.as_tensor(z,device='cuda')[:,None].float()/255
    def fit(indices,robust,epochs,monitor):
        torch.manual_seed(42);rng=np.random.default_rng(42); model=network().cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.001); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,18)
        scaler=torch.amp.GradScaler('cuda'); loss=nn.CrossEntropyLoss(reduction='none'); history=[]
        ax=np.concatenate([sx,x[indices]]);ay=np.r_[sy,y[indices]]
        # Balance domains and temper dominant classes within each domain.
        ws=[]
        for yy in (sy,y[indices]):
            w=1/np.sqrt(np.bincount(yy,minlength=36)[yy]);ws.extend(w/w.sum()*len(ay)/2)
        ws=np.asarray(ws,dtype=np.float32)
        best=float('inf'); stale=0
        for ep in range(epochs):
            model.train(); losses=[]
            for ids in np.array_split(rng.permutation(len(ay)),int(np.ceil(len(ay)/128))):
                images=ax[ids]; images=np.stack([jitter(im,rng) for im in images]) if robust else images
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    z=loss(model(tensor(images)),torch.as_tensor(ay[ids],device='cuda',dtype=torch.long))
                    value=(z*torch.as_tensor(ws[ids],device='cuda')).mean()
                scaler.scale(value).backward();scaler.step(opt);scaler.update();losses.append(value.detach())
            sch.step(); model.eval()
            with torch.inference_mode():v=float(loss(model(tensor(x[hold])),torch.as_tensor(y[hold],device='cuda',dtype=torch.long)).mean()) if monitor else None
            history.append({'epoch':ep+1,'train':float(torch.stack(losses).mean()),'heldout':v})
            if monitor:
                if v<best:best=v;stale=0
                else:stale+=1
                if stale>=4:break
        return model,history
    for name in ('standard','robust'):
        print('Training',name,flush=True);t=perf_counter();torch.cuda.reset_peak_memory_stats()
        _,hist=fit(np.flatnonzero(~hold),name=='robust',18,True); epoch=min(hist,key=lambda r:r['heldout'])['epoch']
        model,full=fit(np.arange(len(y)),name=='robust',epoch,False)
        path=OUT/f'{name}.pt'; torch.save(model.state_dict(),path)
        model.load_state_dict(torch.load(path,weights_only=True));model.eval()
        with torch.inference_mode(): yp=model(tensor(vx)).argmax(1).cpu().numpy()
        confusion=np.zeros((36,36),int); np.add.at(confusion,(vy,yp),1)
        per={}; times=[]
        for im,boxes in geom.items():
            image=cv2.imread(str(ROOT/im)); assert image is not None
            z=np.stack([normalize_glyph(image[y:y+h,x:x+w]) for x,y,w,h in boxes])
            torch.cuda.synchronize();start=perf_counter()
            with torch.inference_mode(): pred=model(tensor(z)).argmax(1).cpu().tolist()
            torch.cuda.synchronize(); times.append(perf_counter()-start);per[im]=''.join(CLASSES[i] for i in pred)
        preds[name]=per
        metrics[name]={'glyph_correct':int((yp==vy).sum()),'glyph_total':len(vy),'confusion':confusion.tolist(),
                       'class_order':CLASSES,'heldout_history':hist,'refit_history':full,'selected_epochs':epoch,
                       'seconds':perf_counter()-t,'peak_allocated':torch.cuda.max_memory_allocated(),
                       'peak_reserved':torch.cuda.max_memory_reserved(),'cuda_glyph_batch_median_s':float(np.median(times)),
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        write('training.json',metrics);print(name,epoch,metrics[name]['glyph_correct'],flush=True)
    records=[]
    for row in old:
        a,b=[{k:r['text'] for k,r in row['outputs'][v]['backends'].items()} for v in ('v1','v2')]
        r={'image':row['image'],'reference':row['reference'],'cnn_v1':a['cnn'],'cnn_v2':b['cnn'],
           'svm_v1':a['svm'],'rf_v1':a['rf'],'svm_v2':b['svm'],'rf_v2':b['rf'],
           'majority_v1':a['ensemble'],'majority_v2':b['ensemble'],
           'hybrid':vote([b['svm'],b['rf'],a['cnn']],[1,1,1]),'weighted_hard':vote([b['svm'],b['rf'],a['cnn']],[1,1,2])}
        for n,outputs in preds.items():
            r[n]=outputs[row['image']]; r[n+'_hybrid']=vote([b['svm'],b['rf'],r[n]],[1,1,1])
        records.append(r)
    summary={k:{'exact':sum(r[k]==r['reference'] for r in records),'cer':sum(edits(r['reference'],r[k]) for r in records)/sum(len(r['reference']) for r in records)} for k in records[0] if k not in ('image','reference')}
    write('development.json',{'summary':summary,'records':records,'geometry':'cached Round 1 YOLO boxes and glyph ordering','test_run':False});print(summary,flush=True)


if __name__=='__main__':main()