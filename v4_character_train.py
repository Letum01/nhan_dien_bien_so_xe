"""Fixed-budget CUDA ablation, independent of the frozen product modules."""
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import cv2
import numpy as np

from final_ocr_dataset import ROOT, check_protocol, read
from synthetic_plate_v3 import validate as validate_synthetic
from yolo_character_dataset import sha, validate

OUT=ROOT/'experiments/alpr_v4'
REAL=ROOT/'datasets/yolo_character_round1'
SYN=ROOT/'experiments/synthetic_plate_v3_validated'
PRE=ROOT/'pretrained/yolov8n.pt'
PROTOCOL={'seed':42,'arms':{'mixed':{'real':46,'synthetic':46,'epochs':35},'pretrain':{'real':0,'synthetic':360,'epochs':12},
           'finetune':{'real':46,'synthetic':0,'epochs':35}},'fixed':{'imgsz':640,'batch':4,'device':0,'amp':True,'workers':0,'patience':12,
           'fliplr':0.,'flipud':0.,'mosaic':0.,'mixup':0.,'seed':42},
           'selection':'Report both arms separately; experimental candidate must exceed V1 CNN exact and not increase CER on YOLO crops. Never use test.',
           'pretrain_selection':'last checkpoint after fixed budget; no real development validation in pretraining'}


def write(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def synthetic_crop(im,row):
    x,y,w,h=row['plate_xywh']; left=max(0,x-w//12); top=max(0,y-h//12)
    right=min(im.shape[1],x+w+w//12); bottom=min(im.shape[0],y+h+h//12)
    result=im[top:bottom,left:right]; H,W=result.shape[:2]
    labels=[]
    for glyph in row['glyphs']:
        gx,gy,gw,gh=glyph['xywh']
        if not (left<=gx and top<=gy and gx+gw<=right and gy+gh<=bottom):raise ValueError('Glyph clipped')
        labels.append(((gx-left+gw/2)/W,(gy-top+gh/2)/H,gw/W,gh/H))
    return result,labels


def stage(name, real_count, synth_train, synth_val=()):
    dst=OUT/('data_'+name);dst.mkdir(); rows=[]
    for split in ('train','val'):
        for f in ('images','labels'):(dst/f/split).mkdir(parents=True)
        if (split=='train' and real_count) or (split=='val' and not synth_val):
            for source in sorted((REAL/'images'/split).glob('*.png')):
                label=REAL/'labels'/split/(source.stem+'.txt')
                for f,p in [('images',source),('labels',label)]:shutil.copy2(p,dst/f/split/p.name)
                rows.append({'split':split,'origin':'reviewed_real','id':source.stem,'source_sha256':sha(source),'label_sha256':sha(label)})
        for r in (synth_train if split=='train' else synth_val):
            src=SYN/r['image']; crop,labels=synthetic_crop(cv2.imread(str(src)),r); id='syn_'+src.stem
            img=dst/'images'/split/(id+'.png'); lab=dst/'labels'/split/(id+'.txt')
            if not cv2.imwrite(str(img),crop):raise OSError(img)
            lab.write_text(''.join('0 '+' '.join(f'{v:.10f}' for v in label)+'\n' for label in labels),encoding='utf-8')
            rows.append({'split':split,'origin':'procedural_synthetic','id':id,'group':r['source_group'],'source_sha256':r['sha256'],
                         'image_sha256':sha(img),'label_sha256':sha(lab)})
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Cross-split image ID conflict')
    write(dst/'manifest.json',rows)
    (dst/'data.yaml').write_text(json.dumps({'path':dst.as_posix(),'train':'images/train','val':'images/val','names':['character']}),encoding='utf-8')
    return dst


def main():
    import torch
    from ultralytics import YOLO
    check_protocol();validate(REAL);validate_synthetic(SYN)
    # Reuse the OCR preflight to check reviewed train against test source/hash/pHash.
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-c','from train_ocr_v2 import prepare; prepare()'],cwd=ROOT,check=True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no fallback')
    OUT.mkdir(exist_ok=False);write(OUT/'protocol.json',PROTOCOL)
    synth=read(SYN/'manifest.json')['rows']; rng=np.random.default_rng(42)
    # Stratify both arms by layout, then reserve disjoint synthetic validation.
    bins=[[r for r in synth if r['layout']==layout] for layout in ('one','two')]
    for g in bins:rng.shuffle(g)
    data={'mixed':stage('mixed',46,[r for g in bins for r in g[:23]]),
          'pretrain':stage('pretrain',0,[r for g in bins for r in g[:180]],[r for g in bins for r in g[180:204]]),
          'finetune':stage('finetune',46,[])}
    torch.cuda.init(); initial=PRE
    for name in ('mixed','pretrain','finetune'):
        settings={**PROTOCOL['fixed'],'epochs':PROTOCOL['arms'][name]['epochs']}
        if name=='pretrain':settings['patience']=0
        model=YOLO(str(initial)); torch.cuda.reset_peak_memory_stats(); tick=perf_counter()
        print('START',name,flush=True)
        model.train(data=str(data[name]/'data.yaml'),project=str(OUT),name=name,exist_ok=False,**settings)
        path=OUT/name/'weights'; last=path/'last.pt'; best=path/'best.pt'
        write(OUT/name/'receipt.json',{'initial_sha256':sha(initial),'data_manifest_sha256':sha(data[name]/'manifest.json'),
            'seconds':perf_counter()-tick,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
            'best_sha256':sha(best),'last_sha256':sha(last),'device':torch.cuda.get_device_name(0)})
        if name=='pretrain':initial=last
        del model;torch.cuda.empty_cache();print('DONE',name,flush=True)


if __name__=='__main__':main()