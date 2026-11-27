"""Read-only SQLite audit and reviewed character-box export; no training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from dataset_audit import fingerprint
from dataset_validation import check_split_leakage, validate_boxes
from glyph_review import ROOT, plates_from_detection, validate_glyph_plate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReadOnlyStore:
    def __init__(self, path):
        self.path = Path(path).resolve()

    def all_latest(self):
        with sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True) as db:
            rows = db.execute('''SELECT r.image_id,r.payload FROM revisions r JOIN
                (SELECT image_id,MAX(version) AS version FROM revisions GROUP BY image_id) m
                ON r.image_id=m.image_id AND r.version=m.version''').fetchall()
        return {key: json.loads(payload) for key, payload in rows}


def live_rows(root=ROOT):
    parents = {r['image_id']: r for r in plates_from_detection(ReadOnlyStore(root/'data/annotations.sqlite3'))}
    reviews = ReadOnlyStore(root/'data/ocr_glyphs.sqlite3').all_latest()
    rows = sorted((r for r in reviews.values() if r['review_status']=='reviewed'), key=lambda r:r['image_id'])
    for row in rows:
        parent = parents.get(row['image_id'])
        if not parent or any(row.get(k)!=parent[k] for k in ('source_image_id','image_path','plate_index','plate_bbox','detection_revision','source_group','split')):
            raise ValueError(f"Stale detection parent: {row['image_path']}")
    return rows


def normalized(box, width, height):
    x,y,w,h = box
    return ((x+w/2)/width,(y+h/2)/height,w/width,h/height)


def integer_boxes(boxes, width, height):
    validate_boxes(boxes,width,height)
    if any(type(v) is not int for box in boxes for v in box):
        raise ValueError('Crop/label coordinates must be integer pixel xywh')


def source_crop(row, source_root):
    path = source_root/row['image_path']
    digest, phash, shape = fingerprint(path)
    if shape is None:
        raise ValueError(f'Unreadable source: {path}')
    x,y,w,h = row['plate_bbox']
    integer_boxes([row['plate_bbox']],shape[1],shape[0])
    if not row.get('confirmed_by_user') or row.get('review_status')!='reviewed':
        raise ValueError('Explicit Glyph reviewed required')
    integer_boxes([c['bbox'] for c in row['characters']],w,h)
    validate_glyph_plate(row,w,h)
    return cv2.imread(str(path))[y:y+h,x:x+w], digest, f'{phash:016x}'


def validate(root, source_root=ROOT, *, verify_live=True):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    rows = manifest['rows']
    by_id = {r['review']['image_id']: r['review'] for r in rows}
    if not rows or len(by_id)!=len(rows) or manifest['schema']!='reviewed-character-yolo-v1':
        raise ValueError('Empty, duplicate or incompatible manifest')
    if any(r['split'] not in ('train','val') for r in by_id.values()):
        raise ValueError('Character training snapshots must exclude test records')
    if verify_live and by_id!={r['image_id']:r for r in live_rows(source_root) if r['split'] in ('train','val')}:
        raise ValueError('Manifest differs from latest reviewed SQLite rows')
    check_split_leakage(by_id.values(),source_root)
    counts = Counter(r['split'] for r in by_id.values())
    if not counts['train'] or not counts['val']:
        raise ValueError('Train and validation required')
    expected_yaml = {'path':str(root),'names':['character'],'nc':1,
                     **{s:f'images/{s}' for s in ('train','val','test') if counts[s]}}
    # JSON is a YAML-compatible subset, avoiding PyYAML in the OCR venv.
    if json.loads((root/'data.yaml').read_text(encoding='utf-8'))!=expected_yaml:
        raise ValueError('Incorrect data.yaml paths or class mapping')
    images, labels, crop_rows, boxes = set(), set(), [], Counter()
    for entry in rows:
        row = entry['review']; split = row['split']; key = row['image_id']
        if not key or not key.isascii() or not key.isalnum():
            raise ValueError('Unsafe plate ID')
        img = root/'images'/split/(key+'.png'); label = root/'labels'/split/(key+'.txt')
        images.add(img); labels.add(label)
        crop, digest, phash = source_crop(row,source_root)
        if (digest,phash,sha(img),sha(label))!=tuple(entry[k] for k in ('source_sha256','source_phash','crop_sha256','label_sha256')):
            raise ValueError(f'Artifact/source hash mismatch: {key}')
        if not np.array_equal(cv2.imread(str(img)),crop) or list(crop.shape[:2])!=entry['crop_hw']:
            raise ValueError(f'Crop differs from reviewed parent: {key}')
        actual = [line.split() for line in label.read_text(encoding='utf-8').splitlines()]
        if len(actual)!=len(row['characters']):
            raise ValueError(f'Box count mismatch: {key}')
        for fields, glyph in zip(actual,row['characters']):
            if len(fields)!=5 or fields[0]!='0' or not np.allclose(list(map(float,fields[1:])),normalized(glyph['bbox'],crop.shape[1],crop.shape[0]),atol=1e-9,rtol=0):
                raise ValueError(f'Box/class differs from reviewed xywh: {key}')
        boxes[split] += len(actual)
        crop_rows.append({'image_path':str(img),'source_group':row['source_group'],'split':split})
    for kind,expected in (('images',images),('labels',labels)):
        if {p for p in (root/kind).rglob('*') if p.is_file() and p.suffix!='.cache'}!=expected:
            raise ValueError(f'Missing/orphan files: {kind}')
    check_split_leakage(crop_rows,root)
    return {'status':'PASS','images_by_split':dict(counts),'boxes_by_split':dict(boxes),'boxes':sum(boxes.values()),
            'groups_by_split':{s:len({r['source_group'] for r in by_id.values() if r['split']==s}) for s in counts},
            'crop_edge_glyphs':sum(len(r['edge_glyph_indices']) for r in rows),
            'min_glyph_wh':[min(c['bbox'][i] for r in by_id.values() for c in r['characters']) for i in (2,3)]}


def export(destination, source_root=ROOT):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    rows = [r for r in live_rows(source_root) if r['split'] in ('train','val')]
    check_split_leakage(rows,source_root)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        stage=Path(tmp)/'export'; stage.mkdir()
        entries=[]
        for row in rows:
            crop,digest,phash=source_crop(row,source_root); h,w=crop.shape[:2]
            key=row['image_id']; split=row['split']
            img=stage/'images'/split/(key+'.png'); label=stage/'labels'/split/(key+'.txt')
            overlay=stage/'overlays'/split/(key+'.png')
            for path in (img,label,overlay): path.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(img),crop): raise OSError(img)
            label.write_text(''.join('0 '+' '.join(f'{v:.12f}' for v in normalized(c['bbox'],w,h))+'\n' for c in row['characters']),encoding='utf-8')
            vis=crop.copy(); edges=[]
            for i,c in enumerate(row['characters']):
                x,y,cw,ch=c['bbox']; edge=min(x,y,w-x-cw,h-y-ch)<=1
                if edge: edges.append(i)
                cv2.rectangle(vis,(x,y),(x+cw-1,y+ch-1),(0,180,255) if edge else (0,255,0),1)
                cv2.putText(vis,str(i),(x,max(8,y)),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,0,0),1)
            if not cv2.imwrite(str(overlay),vis): raise OSError(overlay)
            entries.append({'review':row,'source_sha256':digest,'source_phash':phash,'crop_hw':[h,w],
                            'crop_sha256':sha(img),'label_sha256':sha(label),'edge_glyph_indices':edges,
                            'image':img.relative_to(stage).as_posix(),'label':label.relative_to(stage).as_posix()})
        spec={'names':['character'],'nc':1,**{s:f'images/{s}' for s in sorted({r['split'] for r in rows})}}
        (stage/'manifest.json').write_text(json.dumps({'schema':'reviewed-character-yolo-v1','rows':entries},ensure_ascii=False,indent=2),encoding='utf-8')
        (stage/'data.yaml').write_text(json.dumps({'path':str(stage),**spec},indent=2),encoding='utf-8')
        result=validate(stage,source_root)
        (stage/'data.yaml').write_text(json.dumps({'path':str(destination),**spec},indent=2),encoding='utf-8')
        (stage/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        stage.rename(destination)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('export','validate')); parser.add_argument('path',type=Path)
    args=parser.parse_args()
    print(json.dumps((export if args.action=='export' else validate)(args.path),indent=2))