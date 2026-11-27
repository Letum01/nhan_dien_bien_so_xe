"""Prepare local review artifacts from the existing external archive audit; never fit a model."""
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
import math
from pathlib import Path

import cv2
import numpy as np

from external_dataset_audit import CLASSES, bands, digest, parse_yolo, phash, vn_box, write
from final_ocr_dataset import ROOT, STORE, ReadOnlyStore, check_protocol, read, relative_image
from final_ocr_intake import registered
from yolo_character_pipeline import normalize_glyph, order_boxes

AUDIT = ROOT / 'reports/external_dataset_audit_local01'
YAML_SHA = '718e101cf1f9ae07ca4d7467abe185dcb5ddf8a5413a1186c1c2238e1cac7d12'
TAGS = ('OCR_Data', 'License_Data', 'VNLP_detection')


def mapping(config):
    names = config['names']
    if not isinstance(names, list) or len(names) != 30 or config['nc'] != len(names):
        raise ValueError('Unexpected source schema')
    if any(not isinstance(n, str) or len(n) != 1 or n not in CLASSES for n in names) or len(set(names)) != len(names):
        raise ValueError('Duplicate or invalid source class')
    forward = {i: CLASSES.index(n) for i, n in enumerate(names)}
    return forward, {v: k for k, v in forward.items()}


def source_config(text):
    """Read the two scalar/inline-list fields of this pinned source config."""
    values = defaultdict(list)
    for line in text.splitlines():
        key, _, value = line.partition(':')
        if key in ('nc', 'names'):
            values[key].append(ast.literal_eval(value.strip()))
    if any(len(values[key]) != 1 for key in ('nc', 'names')):
        raise ValueError('Missing or repeated source config fields')
    return {key: values[key][0] for key in ('nc','names')}


def safe_path(base, name):
    result = (base / name).resolve()
    if not result.is_relative_to(base.resolve()) or result == base.resolve():
        raise ValueError(f'Path escapes source: {name}')
    return result


def load_image(path):
    im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if im is None:
        raise ValueError(f'Cannot decode {path}')
    return im


def crop(im, xywh):
    x, y, w, h = xywh
    height, width = im.shape[:2]
    a, b = max(0, math.floor(x + 1e-8)), max(0, math.floor(y + 1e-8))
    c, d = min(width, math.ceil(x+w-1e-8)), min(height, math.ceil(y+h-1e-8))
    if c <= a or d <= b:
        raise ValueError('Empty pixel crop')
    return im[b:d, a:c], [a, b, c-a, d-b]


def iou(a, b):
    x,y,w,h = a
    u,v,s,t = b
    area = max(0,min(x+w,u+s)-max(x,u))*max(0,min(y+h,v+t)-max(y,v))
    return area/(w*h+s*t-area)


class HashIndex:
    def __init__(self):
        self.exact = defaultdict(set)
        self.values = defaultdict(set)
        self.band = defaultdict(set)

    def add(self, key, sha, h):
        if sha:
            self.exact[sha].add(key)
        if h is not None:
            h = int(h, 16)
            if h not in self.values:
                for b in bands(h):
                    self.band[b].add(h)
            self.values[h].add(key)

    def hits(self, sha, h):
        exact = self.exact.get(sha, ()) if sha else ()
        for key in sorted(exact):
            yield key, 'sha256', None
        if h is not None:
            h = int(h, 16)
            for v in sorted(set().union(*(self.band[b] for b in bands(h)))):
                distance = (v ^ h).bit_count()
                if distance <= 6:
                    for key in sorted(self.values[v]):
                        if key not in exact:
                            yield key, 'phash_candidate', distance


def components(nodes, edges):
    parent = {n: n for n in nodes}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    return {n: root(n) for n in nodes}


def protected():
    provenance = read(ROOT/'reports/ocr_test_provenance.json')
    rows = {}
    groups = set()
    registry = ReadOnlyStore(ROOT/'data/annotations.sqlite3').all_latest().values()
    for r in registry:
        if r['split'] in ('val','test') and r.get('human_reviewed'):
            groups.add(r['source_group'])
    for r in registry:
        if r['split'] in ('val','test') or r['source_group'] in groups:
            rows[relative_image(r['image_path'])] = [r['split'], r['source_group'], r['boxes'], None]
    for r in provenance['candidates'] + provenance['pilot_test'] + registered(STORE):
        key = relative_image(r['image'])
        if key not in rows:
            rows[key] = ['final_candidate' if r not in provenance['pilot_test'] else 'pilot', None, [], r.get('sha256')]
    for r in ReadOnlyStore(STORE).all_latest().values():
        key = relative_image(r['image_path'])
        if r.get('human_reviewed') and key in rows and r.get('plate_bbox'):
            rows[key][2] = [r['plate_bbox']]
    index = HashIndex(); meta = {}
    for key, (split, group, boxes, expected) in sorted(rows.items()):
        path = safe_path(ROOT, key)
        im = load_image(path); sha = digest(path)
        if expected and expected != sha:
            raise ValueError(f'Protected image changed: {key}')
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        index.add(key, sha, phash(gray))
        oriented = cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
        index.add(key, None, phash(cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY)))
        for box in boxes:
            region, _ = crop(gray, box)
            index.add(key, None, phash(region))
        meta[key] = dict(split=split, source_group=group, sha256=sha, crop_count=len(boxes))
    return index, meta


def draw(im, boxes, color):
    for label,x,y,w,h in boxes:
        cv2.rectangle(im,(round(x),round(y)),(round(x+w),round(y+h)),color,2)
        cv2.putText(im,str(label),(round(x),max(12,round(y)-2)),0,.6,color,2)
    return im


def sheet(out, tiles):
    panels = []
    for title, im in tiles:
        tile = np.zeros((172,320,3),np.uint8)
        scale = min(320/im.shape[1],148/im.shape[0])
        small = cv2.resize(im,(max(1,round(im.shape[1]*scale)),max(1,round(im.shape[0]*scale))))
        tile[:small.shape[0],:small.shape[1]] = small
        cv2.putText(tile,title[:51],(3,165),0,.46,(255,255,255),1)
        panels.append(tile)
    if panels:
        panels += [np.zeros_like(panels[0])]*(-len(panels)%4)
        if not cv2.imwrite(str(out),np.vstack([np.hstack(panels[i:i+4]) for i in range(0,len(panels),4)])):
            raise OSError(f'Cannot write {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source_yaml',type=Path)
    ap.add_argument('--extracted',type=Path,required=True)
    args = ap.parse_args()
    ext = args.extracted.resolve()
    check_protocol()
    if digest(args.source_yaml) != YAML_SHA:
        raise ValueError('Source YAML differs from verified GitHub blob')
    config = source_config(args.source_yaml.read_text(encoding='utf-8-sig'))
    forward, reverse = mapping(config)
    old_map = read(AUDIT/'class_mapping.json')
    if any(old_map['mapping'][str(i)]['target_id'] != v or old_map['mapping'][str(i)]['source_name'] != CLASSES[v] for i,v in forward.items()):
        raise ValueError('Prior manifest class mapping mismatch')
    archive_hashes = {}
    for tag in TAGS:
        prior = read(AUDIT/f'{tag}_archive.json')
        archive = ROOT/'external_datasets'/('VNLP' if tag.startswith('VNLP') else 'pthang23')/prior['file']
        if prior['crc'] != 'PASS' or archive.stat().st_size != prior['zip_bytes'] or digest(archive) != prior['sha256']:
            raise ValueError(f'Archive identity changed: {tag}')
        archive_hashes[tag] = prior['sha256']
    print('Archive identity PASS',flush=True)
    manifests = {tag: AUDIT/f'{tag}_manifest.json' for tag in TAGS}
    ocr,plates,vnlp = (read(manifests[t]) for t in TAGS)
    out = ROOT/'reports'/('external_preparation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    ref, refmeta = protected(); index = HashIndex(); edges = []; direct = []; internal = []
    rows = {r['path']: r for r in (*ocr,*plates)}
    for tag, collection in [('OCR_Data',ocr),('License_Data',plates)]:
        for r in collection:
            key = r['path']
            for view, sha, h in [('full',r['sha256'],r.get('phash'))]+[('plate_crop',None,x) for x in r.get('crop_hashes',[])]:
                direct.extend([key,k,view,kind,d] for k,kind,d in ref.hits(sha,h))
                if view=='full' or tag=='License_Data':
                    internal.extend([key,k,view,kind,d] for k,kind,d in index.hits(sha,h) if key!=k)
            index.add(key,r['sha256'],r.get('phash'))
    edges.extend((a,b) for a,b,*_ in internal)
    group = components(rows,edges)
    protected_groups = {group[a] for a,*_ in direct}
    splits = defaultdict(set)
    for a,r in rows.items():
        splits[group[a]].add((r['dataset'],r['split']))
    mixed = {g for g,s in splits.items() if len({split for _,split in s})>1}
    leakage = {'protected_references':refmeta,'direct':direct,'internal':internal,
               'scope':'OCR_Data and License_Data full-image; License_Data plate crops against OCR full images and available protected plate crops. pHash <=6 is a conservative candidate, not proof.',
               'quarantined':{a: [reason for yes,reason in [(g in protected_groups,'protected_component'),(g in mixed,'cross_split_component')] if yes]
                              for a,g in group.items() if g in protected_groups|mixed}}
    write(out/'leakage.json',leakage)
    print('Leakage candidate index complete',flush=True)
    write(out/'class_mapping.json',{'source_yaml_sha256':YAML_SHA,'source_order':config['names'],'target_order':CLASSES,
                                   'source_to_target':forward,'target_to_source':reverse,'missing':[c for c in CLASSES if c not in config['names']]})
    glyphs=[]; samples=defaultdict(list); counts=defaultdict(Counter); issues=Counter(); before=defaultdict(Counter)
    for r in ocr:
        if r['annotation']!='ok':
            continue
        p=safe_path(ext/'OCR_Data',r['path']); parts=Path(r['path']).parts
        lab=(ext/'OCR_Data').joinpath(parts[0],'labels',*parts[2:]).with_suffix('.txt')
        if digest(p)!=r['sha256'] or digest(lab)!=r['label_sha256']:
            raise ValueError(f'Extracted OCR input differs from audit: {r["path"]}')
        w,h=r['width'],r['height']; boxes, errors = parse_yolo(lab.read_text(encoding='utf-8-sig'),len(forward),w,h)
        if errors or boxes != r['boxes']:
            raise ValueError(f'Annotation replay differs: {r["path"]}')
        im=load_image(p)
        if im.shape[:2]!=(h,w):raise ValueError(f'Dimension mismatch: {p}')
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); blur=float(cv2.Laplacian(cv2.resize(gray,(256,128)),cv2.CV_64F).var())
        overlap = max((iou(a[1:],b[1:]) for a,b in combinations(boxes,2)),default=0)
        flags=[]
        if not 6<=len(boxes)<=10:flags.append('unusual_glyph_count')
        if overlap>=.2:flags.append('overlap_iou_ge_point2')
        if any(min(x,y,w-x-bw,h-y-bh)<=2 for _,x,y,bw,bh in boxes):flags.append('edge_le_2px')
        issues.update(flags)
        for c in sorted({CLASSES[forward[b[0]]] for b in boxes}):samples['class_'+c].append((blur,r))
        _, pixelboxes = zip(*(crop(im,b[1:]) for b in boxes))
        ordered=order_boxes([{'bbox': b} for b in pixelboxes])
        gap=max((abs(a['bbox'][1]-b['bbox'][1]) for a,b in zip(ordered,ordered[1:])),default=0)
        layout='two_row_proxy' if gap>.55*np.median([b[3] for b in pixelboxes]) else 'one_row_proxy'
        for label,val in [('min_size',min(min(x[3:]) for x in boxes)),('blur',blur),('aspect',w/h),(layout,blur)]+[(f,blur) for f in flags]:samples[label].append((val,r))
        reasons = leakage['quarantined'].get(r['path'], [])
        for tid,x,y,bw,bh in boxes:
            c=CLASSES[forward[tid]]; before[r['split']][c]+=1
            if not reasons:counts[r['split']][c]+=1
            _,pixelbox=crop(im,(x,y,bw,bh))
            glyphs.append(dict(source='pthang23 OCR_Data',parent=r['path'],sha256=r['sha256'],label_sha256=r['label_sha256'],
                               source_group=None,possible_duplicate_component=group[r['path']],original_split=r['split'],split=None,
                               provenance='source_annotation_unverified_semantically',character=c,target_id=forward[tid],box_xywh=[x,y,bw,bh],
                               pixel_xywh=pixelbox,crop_policy='raw source pixels; floor start, ceil end, clipped; V1 normalize_glyph 32x32 preview only',
                               quarantine=reasons,quality_flags=flags,training_eligible=False))
    write(out/'glyph_review_manifest.json',glyphs)
    def ocr_tile(title,r):
        return title,draw(load_image(ext/'OCR_Data'/r['path']),[[CLASSES[forward[b[0]]],*b[1:]] for b in r['boxes']],(70,255,100))
    sheet(out/'ocr_by_class.jpg',[ocr_tile(k,sorted(v,key=lambda t:(t[0],t[1]['path']))[len(v)//2][1]) for k,v in sorted(samples.items()) if k.startswith('class_')])
    tiles=[]
    for k in ('min_size','blur','aspect','one_row_proxy','two_row_proxy','edge_le_2px','overlap_iou_ge_point2','unusual_glyph_count'):
        a=sorted(samples[k],key=lambda t:(t[0],t[1]['path']))
        for v,r in a[:4]+(a[-4:] if k in ('aspect','min_size','blur') else []):tiles.append(ocr_tile(f'{k} {v:.1f}',r))
    sheet(out/'ocr_quality.jpg',tiles)
    confusable=[]
    for pattern in ('D0','T1','MN','KA','B8','S5'):
        for c in pattern:
            for _,r in sorted(samples['class_'+c],key=lambda t:t[0])[::max(1,len(samples['class_'+c])//3)][:3]:
                im=load_image(ext/'OCR_Data'/r['path']); b=next(b for b in r['boxes'] if CLASSES[forward[b[0]]]==c)
                roi,_=crop(im,b[1:]); norm=cv2.cvtColor(normalize_glyph(roi),cv2.COLOR_GRAY2BGR)
                raw=cv2.resize(roi,(64,64),interpolation=cv2.INTER_NEAREST)
                confusable.append((f'{pattern} label {c} raw | 32x32',np.hstack([raw,cv2.resize(norm,(64,64),interpolation=cv2.INTER_NEAREST)])))
    sheet(out/'ocr_confusables.jpg',confusable)
    vs=defaultdict(list)
    for r in vnlp:
        w,h=r['width'],r['height']; q=r['filename_coords']; a,b=vn_box(q,w,h,False),vn_box(q,w,h,True)
        if b:
            _,x,y,bw,bh=b; score=bw*bh/(w*h)
            vs['both' if a else 'only_xyxy'].append((score,r));vs['size'].append((score,r));vs['edge'].append((min(x,y,w-x-bw,h-y-bh),r))
    tiles=[]; vn_samples=[]
    for mode, seq in [('only_xyxy',vs['only_xyxy']),('both',vs['both']),('small',vs['size']),('large',vs['size']),('edge',vs['edge'])]:
        seq=sorted(seq,key=lambda t:(t[0],t[1]['path']))
        selection=seq[-4:] if mode=='large' else seq[:4] if mode in ('small','edge') else [seq[i*len(seq)//4] for i in range(4)]
        for _,r in selection:
            p=safe_path(ext/'VNLP_detection',r['path']); im=load_image(p)
            if digest(p)!=r['sha256']:raise ValueError(f'VNLP overlay image changed: {p}')
            w,h=r['width'],r['height']; a,b=(vn_box(r['filename_coords'],w,h,xyxy) for xyxy in (False,True))
            if a:draw(im,[['xywh',*a[1:]]],(0,180,255))
            if b:draw(im,[['xyxy',*b[1:]]],(0,255,0))
            vn_samples.append(dict(stratum=mode,path=r['path'],raw=r['filename_coords'],xywh=a,xyxy=b))
            tiles.append((f'{mode} / orange xywh, green xyxy',im))
    sheet(out/'vnlp_interpretations.jpg',tiles);write(out/'vnlp_samples.json',vn_samples)
    write(out/'summary.json',dict(archive_sha256=archive_hashes,audit_manifest_sha256={k:digest(p) for k,p in manifests.items()},
         available_glyphs=len(glyphs),ocr_images=sum(r['annotation']=='ok' for r in ocr),classes_before=before,classes_after_candidate_quarantine=counts,
         quality_flags=issues,direct_unique=dict(Counter(rows[k]['dataset'] for k in {h[0] for h in direct})),
         quarantined=dict(Counter(rows[k]['dataset'] for k in leakage['quarantined'])),groups_verified=0,
         training_eligible_glyphs=0,ready=False,reason='Missing documented training permissions, source-group provenance, locked independent benchmark and semantic review. VNLP coordinate format unresolved.'))
    print(out,flush=True)


if __name__=='__main__':
    main()