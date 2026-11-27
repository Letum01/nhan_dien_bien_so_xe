"""Read-only research audit of local ALPR archives; never runs archive code or trains."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import sys
import zipfile

import cv2
import numpy as np

from dataset_audit import EXTENSIONS, ROOT
from final_ocr_dataset import audit_candidates, read, check_protocol

CLASSES='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# Author's training/character.yaml and training/license.yaml on main, checked 2026-11-27.
OCR_NAMES='123456789ABCDEFGHKLMNPSTUVXYZ0'
SOURCES={'OCR_Data':('pthang23',r'OCR_Data(?:-[\w-]+)?(?:\.zip)?'),
         'License_Data':('pthang23',r'License_Data(?:-[\w-]+)?(?:\.zip)?'),
         'VNLP_detection':('VNLP',r'VNLP_detection(?:\.zip)?')}


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def safe_members(z):
    seen={}; nodes={}
    for m in z.infolist():
        s=m.filename.rstrip('/'); parts=s.split('/')
        mode=stat.S_IFMT(m.external_attr >> 16) if m.create_system==3 else 0
        if any(not p or p in ('.','..') or p.endswith(('.', ' ')) or re.search(r'[\\:< >"|?*\x00-\x1f]'.replace(' ',''),p)
               or re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?',p,re.I) for p in parts):
            raise ValueError(f'Unsafe ZIP path: {m.filename!r}')
        if mode not in (0,stat.S_IFREG,stat.S_IFDIR) or m.flag_bits & 1:
            raise ValueError(f'Unsupported ZIP member: {m.filename}')
        key=s.casefold()
        if key in seen:raise ValueError(f'Duplicate ZIP path: {m.filename}')
        seen[key]=m
        for i in range(1,len(parts)+1):
            sub='/'.join(parts[:i]); directory=i<len(parts) or m.is_dir()
            if nodes.setdefault(sub.casefold(),(sub,directory))!=(sub,directory):
                raise ValueError(f'ZIP path collision: {m.filename}')
    return list(seen.values())


def extract(archive,dest):
    before=digest(archive)
    with zipfile.ZipFile(archive) as z:
        members=safe_members(z); total=sum(m.file_size for m in members)
        if total+512*2**20>shutil.disk_usage(dest.parent).free:raise ValueError('Insufficient disk space')
        dest.mkdir(); skipped=Counter(); extracted=0
        for m in members:
            p=dest.joinpath(*PurePosixPath(m.filename).parts)
            if m.is_dir():continue
            suffix=p.suffix.lower()
            # Only data/docs are written; nevertheless stream every member to validate its CRC.
            allowed=suffix in EXTENSIONS|{'.txt','.yaml','.yml','.md','.csv','.json','.xml'} or p.name.lower() in ('vnlp_readme','license','readme')
            try:
                with z.open(m) as src:
                    if allowed:
                        p.parent.mkdir(parents=True,exist_ok=True)
                        with p.open('xb') as f:shutil.copyfileobj(src,f,1024*1024)
                        extracted+=m.file_size
                    else:
                        while src.read(1024*1024):pass
                        skipped[suffix]+=1
            except (OSError,zipfile.BadZipFile) as e:raise ValueError(f'{archive.name}: {m.filename}: {e}') from e
    if digest(archive)!=before:raise ValueError(f'ZIP changed: {archive}')
    return dict(file=archive.name,zip_bytes=archive.stat().st_size,sha256=before,uncompressed_bytes=total,
                extracted_bytes=extracted,entries=len(members),crc='PASS',not_extracted=dict(skipped))


def phash(gray):
    d=cv2.dct(cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32))[:8,:8].ravel()
    return f'{sum(int(v)<<i for i,v in enumerate(d>np.median(d[1:]))):016x}'


def parse_yolo(text,n,w,h):
    seen=set(); boxes=[]; errors=Counter()
    for line in text.splitlines():
        if not line.strip():continue
        try:
            c,*v=line.split(); c=int(c); x,y,bw,bh=map(float,v)
            if not 0<=c<n:errors['class_out_of_range']+=1;continue
            if not all(math.isfinite(k) for k in (x,y,bw,bh)) or min(bw,bh)<=0 or min(x-bw/2,y-bh/2)<-1e-6 or max(x+bw/2,y+bh/2)>1+1e-6:
                errors['invalid_box']+=1;continue
            key=(c,x,y,bw,bh)
            if key in seen:errors['duplicate_box']+=1;continue
            seen.add(key);boxes.append([c,(x-bw/2)*w,(y-bh/2)*h,bw*w,bh*h])
        except ValueError:errors['invalid_line']+=1
    return boxes,errors


def vn_box(coords,w,h,xyxy):
    x,y,a,b=coords; bw,bh=(a-x,b-y) if xyxy else (a,b)
    return [0,x,y,bw,bh] if 0<=x<x+bw<=w and 0<=y<y+bh<=h else None


def audit(root,tag,out):
    ocr=tag=='OCR_Data'; vn=tag=='VNLP_detection'; names=OCR_NAMES if ocr else ['license']
    rows=[]; errs=Counter(); dist=Counter(); splits=defaultdict(Counter); used=set(); sizes=[]; examples=[]
    docs=[p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and re.search(r'readme|license\.|terms|\.ya?ml$',p.name,re.I)]
    lists={p.name:[s.strip().lstrip('/') for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()] for p in root.rglob('list_*.txt')} if vn else {}
    images=sorted(p for p in root.rglob('*') if p.suffix.lower() in EXTENSIONS)
    paths={p.relative_to(root).as_posix() for p in images}
    spec=Counter(); textids=Counter(); layout=Counter(); identity=defaultdict(set)
    for p in images:
        rel=p.relative_to(root).as_posix(); parts=rel.split('/'); split='unassigned' if vn else parts[2]; s=splits[split]; s['images']+=1
        r={'path':rel,'split':split,'dataset':tag,'sha256':digest(p),'group':None}; rows.append(r)
        im=cv2.imread(str(p),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if im is None:errs['unreadable_image']+=1;continue
        h,w=im.shape[:2]; gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); s['readable']+=1
        r.update(width=w,height=h,phash=phash(gray)); boxes=[]; status='ok'; diagbox=False
        if vn:
            layout[parts[1]]+=1
            m=re.fullmatch(r'[^_]+_[^_]+_[^_]+_([A-Za-z0-9]+)_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)',p.stem)
            if m:
                t=m[1].upper();r['plate_text_candidate']=t; textids[t]+=1
                coords=list(map(int,m.groups()[1:])); a,b=vn_box(coords,w,h,False),vn_box(coords,w,h,True)
                spec['documented_xywh_in_bounds']+=bool(a);spec['alternative_xyxy_in_bounds']+=bool(b);spec['only_xyxy']+=bool(b and not a);spec['only_xywh']+=bool(a and not b)
                r['filename_coords']=coords
                if a: boxes=[a]
                elif b:diagbox=True; boxes=[b];status='documented_box_invalid'
                else:status='both_coordinate_interpretations_invalid'
            else:status='filename_schema_mismatch'
        else:
            lab=root.joinpath(parts[0],'labels',*parts[2:]).with_suffix('.txt');used.add(lab)
            if not lab.is_file():status='missing_label'
            else:
                try:boxes,e=parse_yolo(lab.read_text(encoding='utf-8-sig'),len(names),w,h)
                except UnicodeError: boxes=[];e=Counter(invalid_encoding=1)
                errs.update(e)
                r['label_sha256']=digest(lab)
                if e:status='invalid_annotation'
                elif not boxes:status='empty_annotation'
            if status=='ok' and ocr and len(boxes)>1:
                cy=sorted(b[2]+b[4]/2 for b in boxes); ratio=max(np.diff(cy))/np.median([b[4] for b in boxes]);layout['two_proxy' if ratio>.55 else 'one_proxy']+=1
        r.update(annotation=status,boxes=[] if diagbox else boxes)
        s[status]+=1
        if status!='ok':errs[status]+=1
        if status=='ok':
            s['boxes']+=len(boxes);dist.update(str(names[b[0]]) if not vn else 'plate' for b in boxes)
            sizes.extend((b[3],b[4],b[3]*b[4]/(w*h)) for b in boxes)
            if not ocr:
                r['crop_hashes']=[phash(gray[max(0,int(y)):min(h,math.ceil(y+bh)),max(0,int(x)):min(w,math.ceil(x+bw))]) for _,x,y,bw,bh in boxes]
                s['multi_box_images']+=len(boxes)>1
                s['small_box_images']+=any(b[3]*b[4]/(w*h)<.01 for b in boxes)
        if len(examples)<20 and boxes and (not vn or (diagbox and sum(k[1] for k in examples)<10) or (not diagbox and sum(not k[1] for k in examples)<10)):
            for c,x,y,bw,bh in boxes:
                cv2.rectangle(im,(round(x),round(y)),(round(x+bw),round(y+bh)),(0,220,0),2)
                if ocr:cv2.putText(im,names[c],(round(x),max(10,round(y)-2)),0,.55,(0,0,255),1)
            pad=np.zeros((180,320,3),np.uint8);scale=min(320/w,155/h); sm=cv2.resize(im,(max(1,int(w*scale)),max(1,int(h*scale))));pad[:sm.shape[0],:sm.shape[1]]=sm
            cv2.putText(pad,f'{split} / {status}'+(' xyxy diagnostic' if diagbox else ''),(3,174),0,.41,(240,240,240),1);examples.append((pad,diagbox))
    if examples:
        sheet=[a for a,_ in examples];sheet += [np.zeros_like(sheet[0])]*(-len(sheet)%4)
        if not cv2.imwrite(str(out/f'{tag}_overlay.jpg'),np.vstack([np.hstack(sheet[i:i+4]) for i in range(0,len(sheet),4)])):raise OSError('Overlay write failed')
    extra=sum(1 for p in root.rglob('*.txt') if 'labels' in p.relative_to(root).parts and p not in used)
    return rows,dict(splits={k:dict(v) for k,v in splits.items()},classes=dict(dist),errors=dict(errs),orphan_labels=extra,layout=dict(layout),docs=docs,
                     resolution=Counter(f"{r['width']}x{r['height']}" for r in rows if 'width' in r).most_common(10),
                     box_wh_area_q05_50_95=np.percentile(sizes,[5,50,95],axis=0).tolist() if sizes else [],
                     vnlp_coordinate_evidence=dict(spec),list_counts={k:len(v) for k,v in lists.items()},
                     list_missing=sum(p not in paths for v in lists.values() for p in v),
                     list_unlisted=len(paths-{p for v in lists.values() for p in v}) if vn else None,
                     text_candidates=sum(textids.values()),repeated_text_proxy_groups=sum(v>1 for v in textids.values()),verified_source_groups=0)


def references():
    import csv
    with (ROOT/'reports/dataset_audit.csv').open(encoding='utf-8-sig') as f:
        old=[{'dataset':'legacy_2932','path':r['image_path'],'sha256':r['sha256'],'phash':r['phash']} for r in csv.DictReader(f) if 'license_plate_dataset' in r['image_path']]
    protected=audit_candidates(write=False)
    old += [{'dataset':f"protected_{r.get('split','final_candidate')}",'path':r['image'],'sha256':r['sha256'],'phash':r['phash']} for r in
            protected['rows']+[r for r in protected['known'] if r['split'] in ('train','val','test')]]
    path=ROOT/'data/alpr_learning_center.sqlite3'
    if path.exists():
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
            old.extend(dict(dataset='learning_center',path=f'local:{i}',sha256=i,phash=p) for i,p in db.execute('SELECT id,phash FROM samples'))
    return old


def bands(h):
    # Seven disjoint bands: any hashes at distance <=6 share at least one band.
    return tuple((i,(h>>shift)&((1<<width)-1)) for i,(shift,width) in enumerate(zip((0,10,19,28,37,46,55),(10,9,9,9,9,9,9))))


def overlaps(rows,refs,crop=False):
    index=defaultdict(set); hash_ids=defaultdict(list); buckets=defaultdict(list); counts=Counter(); examples=defaultdict(list)
    def add(r):
        if r.get('sha256'):hash_ids[r['sha256']].append(r)
        if r.get('phash'):
            h=int(r['phash'],16)
            if h not in buckets:
                for key in bands(h):index[key].add(h)
            buckets[h].append(r)
    for r in refs:add(r)
    def record(a,b,kind,d=None):
        sa=f"{a['dataset']}:{a.get('split','')}";sb=f"{b['dataset']}:{b.get('split','')}"; key=' | '.join((*sorted((sa,sb)),kind)); counts[key]+=1
        if len(examples[key])<8:examples[key].append([a['path'],b['path'],d])
    for r in rows:
        if not crop:
            for b in hash_ids.get(r['sha256'],[]):record(r,b,'exact')
        for value in r.get('crop_hashes',[]) if crop else [r.get('phash')]:
            if not value:continue
            h=int(value,16)
            for c in set().union(*(index[k] for k in bands(h))):
                distance=(h^c).bit_count()
                if distance<=6:
                    for b in buckets[c]:
                        if crop or r['sha256']!=b['sha256']:record(r,b,'crop_pHash_candidate' if crop else 'near_nonexact',distance)
        if not crop:add(r)
    return {'counts':dict(counts),'examples':dict(examples),'note':'Pair counts, not unique images or verified same vehicle.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path);args=ap.parse_args()
    base=ROOT/'external_datasets'; tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=args.output or ROOT/'reports'/f'external_dataset_audit_{tag}';out.mkdir(parents=True,exist_ok=False)
    ext=base/f'extracted_{tag}';ext.mkdir();check_protocol(); result={}; allrows=[]
    for name,(folder,pattern) in SOURCES.items():
        candidates=[p for p in (base/folder).iterdir() if p.is_file() and re.fullmatch(pattern,p.name,re.I)]
        if len(candidates)!=1:raise ValueError(f'Expected one {name} archive; found {len(candidates)}')
        dest=ext/name; print('Extract',name,flush=True); info=extract(candidates[0],dest);write(out/f'{name}_archive.json',info)
        print('Audit',name,flush=True);rows,summary=audit(dest,name,out); allrows.extend(rows);write(out/f'{name}_manifest.json',rows)
        result[name]={**info,**summary};write(out/'summary.json',result)
    mapping={i:{'source_name':c,'target_id':CLASSES.index(c),'valid_boxes':result['OCR_Data']['classes'].get(c,0)} for i,c in enumerate(OCR_NAMES)}
    write(out/'class_mapping.json',{'source':'https://raw.githubusercontent.com/pthang23/License_Plate_Recognition/main/training/character.yaml',
          'target_order':CLASSES,'mapping':mapping,'missing_source_classes':sorted(set(CLASSES)-set(OCR_NAMES)),
          'target_counts':{c:result['OCR_Data']['classes'].get(c,0) for c in CLASSES},'scope':'Mapping audit; no training export'})
    print('Overlap index',flush=True);ref=references();write(out/'duplicates.json',overlaps(allrows,ref))
    write(out/'crop_duplicates.json',overlaps([r for r in allrows if r['dataset']!='OCR_Data'],[r for r in allrows if r['dataset']=='OCR_Data'],crop=True))
    print(out,ext,flush=True)


if __name__=='__main__':main()