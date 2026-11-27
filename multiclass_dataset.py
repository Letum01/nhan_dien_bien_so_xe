"""Isolated 36-class snapshot; only reviewed real and procedural labels."""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from glyph_review import CLASSES, ROOT, normalized_text
from synthetic_plate_v3 import validate as validate_synthetic
from v4_character_train import REAL, SYN, synthetic_crop
from yolo_character_dataset import integer_boxes, normalized, sha, validate as validate_real

OUT = ROOT/'experiments/yolo_multiclass_01'


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def labels(glyphs, coords):
    return ''.join(f"{CLASSES.index(g)} " + ' '.join(f'{v:.10f}' for v in box) + '\n'
                   for g, box in zip(glyphs, coords, strict=True))


def build(root=OUT):
    from train_ocr_v2 import prepare
    prepare()
    if validate_synthetic(SYN)['glyphs'] != 5760:
        raise ValueError('Unexpected synthetic snapshot')
    root.mkdir(exist_ok=False)
    specs = json.loads((REAL/'manifest.json').read_text(encoding='utf-8'))['rows']
    synth = json.loads((SYN/'manifest.json').read_text(encoding='utf-8'))['rows']
    rng = np.random.default_rng(42)
    mixed, pre, synval = [], [], []
    for layout in ('one','two'):
        rows = [r for r in synth if r['layout']==layout]
        rng.shuffle(rows)
        mixed += rows[:23]; pre += rows[:336]; synval += rows[336:]
    for tag in ('A_real','B_mix','C_pretrain'):
        dest = root/('data_'+tag); entries = []
        for split in ('train','val'):
            for folder in ('images','labels'):
                (dest/folder/split).mkdir(parents=True)

        def add(split, key, pixels, text, coords, meta, source=None):
            img = dest/'images'/split/(key+'.png')
            lab = dest/'labels'/split/(key+'.txt')
            if source:
                shutil.copyfile(source,img)
            elif not cv2.imwrite(str(img),pixels):
                raise OSError(img)
            lab.write_text(labels(text,coords),encoding='ascii')
            entries.append({**meta,'split':split,'key':key,'text':text,'coords':coords,
                            'image_sha256':sha(img),'label_sha256':sha(lab)})

        if tag!='C_pretrain':
            for e in specs:
                r=e['review']; src=REAL/e['image']; im=cv2.imread(str(src)); h,w=im.shape[:2]
                gs=r['characters']; boxes=[g['bbox'] for g in gs]; integer_boxes(boxes,w,h)
                text=''.join(g['label'] for g in gs)
                if text!=normalized_text(r['plate_text']) or [g['order'] for g in gs]!=list(range(len(gs))):
                    raise ValueError('Reviewed text/order mismatch')
                add(r['split'],r['image_id'],im,text,[normalized(b,w,h) for b in boxes],
                    {'origin':'human_reviewed','source_group':r['source_group'],'review':r,
                     'source_sha256':e['source_sha256'],'source_phash':e['source_phash']},src)
        for split,rows in ([('train',mixed)] if tag=='B_mix' else [('train',pre),('val',synval)] if tag=='C_pretrain' else []):
            for r in rows:
                path=SYN/r['image']; im,coords=synthetic_crop(cv2.imread(str(path)),r)
                add(split,'syn_'+path.stem,im,r['text'],coords,{'origin':'procedural_synthetic','source_group':r['source_group'],
                    'source_sha256':r['sha256'],'synthetic_annotation':r})
        write(dest/'manifest.json',{'classes':list(CLASSES),'rows':entries})
        write(dest/'data.yaml',{'path':dest.resolve().as_posix(),'train':'images/train','val':'images/val','names':list(CLASSES)})
        write(dest/'validation.json',validate(dest))


def validate(dest):
    meta=json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
    yaml=json.loads((dest/'data.yaml').read_text(encoding='utf-8'))
    if meta['classes']!=list(CLASSES) or yaml!={'path':dest.resolve().as_posix(),'train':'images/train','val':'images/val','names':list(CLASSES)}:
        raise ValueError('Class order or data YAML differs')
    groups={}; keys=set(); hashes={}; counts={}; expected=set()
    for r in meta['rows']:
        split=r['split']; key=r['key']
        if split not in ('train','val') or key in keys or not key.replace('_','').isalnum():
            raise ValueError('Unexpected split or ID')
        keys.add(key)
        for group,index in ((r['source_group'],groups),(r['source_sha256'],hashes),(r['image_sha256'],hashes)):
            if not group or index.setdefault(group,split)!=split:
                raise ValueError('Split overlap')
        img=dest/'images'/split/(key+'.png'); lab=dest/'labels'/split/(key+'.txt'); expected.update((img,lab))
        if sha(img)!=r['image_sha256'] or sha(lab)!=r['label_sha256'] or cv2.imread(str(img)) is None:
            raise ValueError('Snapshot bytes changed')
        text=r['text']; coords=np.asarray(r['coords']); lines=lab.read_text(encoding='ascii').splitlines()
        if not text or coords.shape!=(len(text),4) or (coords[:,2:]<=0).any() or not np.isfinite(coords).all() or (coords[:,:2]-coords[:,2:]/2 < -1e-9).any() or (coords[:,:2]+coords[:,2:]/2 > 1+1e-9).any():
            raise ValueError('Invalid YOLO box')
        if lines!=labels(text,coords).splitlines():
            raise ValueError('Label mapping/coordinates differ')
        counts.setdefault(split,Counter()).update(text)
    if set(counts)!= {'train','val'} or {p for folder in ('images','labels') for p in (dest/folder).rglob('*') if p.suffix in ('.png','.txt')}!=expected:
        raise ValueError('Missing split or unmanifested file')
    return {'images':dict(Counter(r['split'] for r in meta['rows'])),
            'support':{s:{c:counts[s][c] for c in CLASSES} for s in counts}}


if __name__=='__main__':
    if len(sys.argv)==1:
        build()
    else:
        print(validate(Path(sys.argv[1])))