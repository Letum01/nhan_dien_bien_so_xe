"""CUDA-only pilot; retain all frozen ALPR checkpoints and use development only."""
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from final_ocr_dataset import check_protocol
from glyph_review import CLASSES, ROOT
from localization import _box_iou
from multiclass_dataset import OUT, validate, write
from segmentation_benchmark import match_boxes
from yolo_character_dataset import sha
from yolo_character_pipeline import order_boxes

REFERENCE = ROOT/'experiments/yolo_character_round1/comparison/results.json'
PRE = ROOT/'pretrained/yolov8n.pt'
SETTINGS = dict(imgsz=640, batch=4, workers=0, device=0, amp=True, seed=42,
                deterministic=True, mosaic=0., mixup=0., fliplr=0., flipud=0.,
                degrees=3., translate=.05, scale=.15, perspective=.0002, patience=8)
RULE = ('Rank best.pt and last.pt by exact descending then CER ascending; '
        'retain no candidate unless exact >8/12 and CER <=4/99. '
        'Open B only if A reaches >=6 exact and <=8 edits; open C only if B improves '
        'exact without worse CER or improves CER without worse exact. D requires final gate and evidence of capacity limitation.')


def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):
            new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]


def class_geometry(rows, truth, ref):
    same=Counter(tp=0,fp=0,fn=0)
    per={}
    for c in CLASSES:
        m=match_boxes([r['bbox'] for r in rows if r['label']==c],
                      [b for b,g in zip(truth,ref,strict=True) if g==c],.5)
        tp,fp,fn=(m[k] for k in ('tp','fp','fn'))
        same.update(tp=tp,fp=fp,fn=fn); per[c]=[tp,fp,fn]
    return dict(same),per


def gpu_sample():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu,memory.used',
                                  '--format=csv,noheader,nounits','-i','0'],text=True,timeout=5)
    return dict(zip(('temp_c','util_pct','system_used_mib'),map(int,text.strip().split(',')),strict=True))


def train(name, data, init, epochs, batch=4):
    import torch
    from ultralytics import YOLO
    validate(data)
    settings={**SETTINGS,'batch':batch,'epochs':epochs}
    if name=='C_pretrain':settings['patience']=0
    sample=gpu_sample()
    if sample['temp_c']>=85 or torch.cuda.mem_get_info(0)[0]<1.5*1024**3:
        raise RuntimeError(f'GPU preflight thermal/free-VRAM gate: {sample}')
    path=OUT/name
    if path.exists():raise FileExistsError(path)
    samples=[sample]; batch_times=[]
    torch.cuda.reset_peak_memory_stats(0)
    model=YOLO(str(init))
    def monitor(trainer):
        point=gpu_sample(); samples.append(point)
        if point['temp_c']>=87:
            raise RuntimeError(f'GPU thermal stop during {name}: {point}')
    def start(trainer):
        batch_times.append(perf_counter())
    model.add_callback('on_train_epoch_end',monitor)
    if name=='smoke':model.add_callback('on_train_batch_end',start)
    t=perf_counter()
    model.train(data=str(data/'data.yaml'),project=str(OUT),name=name,exist_ok=False,**settings)
    torch.cuda.synchronize(0)
    elapsed=perf_counter()-t
    with (path/'results.csv').open() as stream: actual=len(list(csv.DictReader(stream)))
    best,last=(path/'weights'/f'{k}.pt' for k in ('best','last'))
    write(path/'receipt.json',{'seconds':elapsed,'epochs':actual,'settings':settings,'init_sha256':sha(init),
          'dataset_sha256':sha(data/'manifest.json'),'best_sha256':sha(best),'last_sha256':sha(last),
          'peak_allocated':torch.cuda.max_memory_allocated(0),'peak_reserved':torch.cuda.max_memory_reserved(0),
          'gpu_samples':samples,'smoke_batches':len(batch_times),
          'nominal_train_images_per_wall_second':actual*validate(data)['images']['train']/elapsed})
    del model
    torch.cuda.empty_cache()
    return best,last


def benchmark(checkpoint, data, output):
    import torch
    from ultralytics import YOLO
    model=YOLO(str(checkpoint))
    if list(model.names.values())!=list(CLASSES):raise ValueError('Unexpected checkpoint mapping')
    model.predict(np.zeros((96,192,3),np.uint8),imgsz=640,device=0,verbose=False)
    dev={e['key']:e['review'] for e in json.loads((data/'manifest.json').read_text(encoding='utf-8'))['rows'] if e['split']=='val'}
    base=[r for r in json.loads(REFERENCE.read_text(encoding='utf-8'))['records'] if r['scope']=='yolo' and r['method']=='character_yolo']
    if len(dev)!=12 or {r['id'] for r in base}!=set(dev):raise ValueError('Development cohort mismatch')
    records=[]; conf=np.zeros((36,36),int); local=Counter(); per=Counter(); timings=[]
    for r in base:
        im=cv2.imread(str(ROOT/dev[r['id']]['image_path'])); box=r['crop_box']; rows=[]
        if box:
            x,y,w,h=box; region=im[y:y+h,x:x+w]
            torch.cuda.synchronize(0); start=perf_counter()
            p=model.predict(region,imgsz=640,conf=.25,iou=.7,max_det=40,device=0,verbose=False)[0]
            torch.cuda.synchronize(0); timings.append(perf_counter()-start)
            for (a,b,c,d),idx,score in zip(p.boxes.xyxy.cpu().tolist(),p.boxes.cls.cpu().tolist(),p.boxes.conf.cpu().tolist(),strict=True):
                a,b=max(0,math.floor(a)),max(0,math.floor(b)); c,d=min(w,math.ceil(c)),min(h,math.ceil(d))
                if c>a and d>b:rows.append({'bbox':[x+a,y+b,c-a,d-b],'label':CLASSES[int(idx)],'score':score})
        rows=order_boxes(rows); ref=r['reference']; truth=r['gt_boxes']
        pred=''.join(z['label'] for z in rows)
        geom=match_boxes([z['bbox'] for z in rows],truth,.5)
        for m in geom['matches']:conf[CLASSES.index(ref[m['gt']]),CLASSES.index(pred[m['pred']])]+=1
        loc,cls=class_geometry(rows,truth,ref); local.update({k:geom[k] for k in ('tp','fp','fn')})
        for c,values in cls.items():
            for k,v in zip(('tp','fp','fn'),values):per[c,k]+=v
        records.append({'id':r['id'],'source_group':dev[r['id']]['source_group'],'reference':ref,'prediction':pred,
                        'edits':edit_distance(ref,pred),'exact':ref==pred,'class_geometry':loc,'geometry':geom,
                        'boxes':rows,'baseline':r['ocr']})
    counts={}
    for c in CLASSES:
        tp,fp,fn=[per[c,k] for k in ('tp','fp','fn')]
        counts[c]={'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,
                   'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if tp+fp+fn else None}
    result={'checkpoint_sha256':sha(checkpoint),'exact':sum(r['exact'] for r in records),
            'edits':sum(r['edits'] for r in records),'glyphs':sum(len(r['reference']) for r in records),
            'fixed':sum(r['exact'] and not r['baseline']['cnn']['exact'] for r in records),
            'regressed':sum(not r['exact'] and r['baseline']['cnn']['exact'] for r in records),
            'localization':dict(local),'class_aware':{k:sum(v[k] for v in counts.values()) for k in ('tp','fp','fn')},
            'matched_correct':int(conf.trace()),'per_class':counts,'classes':list(CLASSES),'confusion':conf.tolist(),
            'character_latency_ms':{'n':len(timings),'median':1000*statistics.median(timings),
                'mean':1000*statistics.mean(timings),'p95':1000*float(np.percentile(timings,95))},'records':records}
    write(output,result)
    return result


def evaluate(name, checkpoints, real):
    results=[]; seen=set()
    for p in checkpoints:
        if sha(p) in seen:continue
        seen.add(sha(p)); results.append((p,benchmark(p,real,OUT/name/f'{p.stem}_ocr.json')))
    checkpoint,r=max(results,key=lambda a:(a[1]['exact'],-a[1]['edits']))
    from ultralytics import YOLO
    metrics=YOLO(str(checkpoint)).val(data=str(real/'data.yaml'),split='val',device=0,imgsz=640,batch=4,
                conf=.001,iou=.7,max_det=40,project=str(OUT/name),name='ap',exist_ok=False,workers=0)
    write(OUT/name/'evaluation.json',{'checkpoint':str(checkpoint),'map50':float(metrics.box.map50),
                  'map50_95':float(metrics.box.map),'ap_scope':'reviewed GT crops',**r})
    print(name,'exact',r['exact'],'edits',r['edits'],flush=True)
    return r


def main():
    import torch
    import ultralytics
    check_protocol()
    assert torch.cuda.is_available() and '2060' in torch.cuda.get_device_name(0), 'RTX 2060 CUDA required'
    a,b,c=(OUT/f'data_{s}' for s in ('A_real','B_mix','C_pretrain'))
    protocol={'gate':RULE,'settings':SETTINGS,'budgets':{'smoke':1,'A':20,'B':20,'C_pretrain':10,'C_finetune':20},
              'references':{str(p):sha(p) for p in (REFERENCE,PRE,Path(__file__),*(p/'manifest.json' for p in (a,b,c)))},
              'scope':'Frozen cached plate crops. Character .25; class-aware NMS .7; geometric matching IoU .5.',
              'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'ultralytics':ultralytics.__version__}
    path=OUT/'protocol.json'
    with path.open('x',encoding='utf-8') as f:json.dump(protocol,f,indent=2)
    train('smoke',a,PRE,1)
    ra=evaluate('A_real',train('A_real',a,PRE,20),a)
    decisions={'B':'STOP: A below continuation gate','C':'NOT RUN','D':'NOT RUN','hybrid':'NOT RUN'}
    write(OUT/'decisions.json',decisions)
    if ra['exact']>=6 and ra['edits']<=8:
        rb=evaluate('B_mix',train('B_mix',b,PRE,20),a); decisions['B']='COMPLETED'
        if (rb['exact']>=ra['exact'] and rb['edits']<=ra['edits'] and
                (rb['exact']>ra['exact'] or rb['edits']<ra['edits'])):
            _,last=train('C_pretrain',c,PRE,10)
            evaluate('C_finetune',train('C_finetune',a,last,20),a); decisions['C']='COMPLETED'
        else:decisions['C']='STOP: no paired improvement from B'
    write(OUT/'decisions.json',decisions)


if __name__=='__main__':main()