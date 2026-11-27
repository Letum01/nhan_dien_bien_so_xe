"""Local audit only: report candidates, never create training labels."""
import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from dataset_audit import EXTENSIONS
from final_ocr_dataset import ROOT, STORE, audit_candidates, check_protocol, read, relative_image
from train_ocr_v2 import put
from yolo_character_dataset import ReadOnlyStore, live_rows, sha


def quantiles(values):
    return np.percentile(values, [5, 25, 50, 75, 95]).round(3).tolist() if len(values) else None


def family(name):
    # Filename taxonomy only, not capture/vehicle identity or a label.
    if re.search(r'checkin|checkout', name, re.I):return 'checkin_checkout'
    if re.fullmatch(r'\d+_\d+_[a-z]+', name, re.I):return 'number_number_suffix'
    if name.isdigit():return 'numeric'
    return 'other'


def deduplicate(items):
    chosen=[]
    for r in items:
        if any(r['sha256']==s['sha256'] or (int(r['phash'],16)^int(s['phash'],16)).bit_count()<=6 or
               (r['group']!='UNKNOWN' and r['group']==s['group']) for s in chosen):
            continue
        chosen.append(r)
    return chosen


def inventory():
    original=ROOT/'dataset/license_plate_dataset'; former=ROOT/'reports/dataset_audit.csv'
    with former.open(encoding='utf-8-sig',newline='') as stream:
        previous={relative_image(r['image_path']):r['sha256'] for r in csv.DictReader(stream) if 'license_plate_dataset' in r['image_path']}
    detection=ReadOnlyStore(ROOT/'data/annotations.sqlite3').all_latest()
    det={relative_image(r['image_path']):r for r in detection.values()}
    protected=audit_candidates(write=False)
    refs=[r for r in protected['known'] if r['split'] in ('val','test')]+protected['rows']
    refs += [{'image':r['image_path'],'sha256':r['source_sha256'],'phash':r['source_phash'],'source_group':r['source_group']}
             for r in ReadOnlyStore(STORE).all_latest().values() if r.get('source_sha256') and r.get('source_phash')]
    phrefs=[int(r['phash'],16) for r in refs if r.get('phash')]
    sha_refs={r['sha256'] for r in refs}; paths={relative_image(r['image']) for r in refs}
    group_refs={r.get('source_group','').strip().casefold() for r in refs}|{(r.get('metadata',{}).get('source_group') or '').strip().casefold() for r in refs}
    rows=[]; exif=Counter(); hashes=Counter(); distances=Counter(); changes=[]
    for p in sorted(original.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in EXTENSIONS:continue
        rel=relative_image(p); digest=sha(p); row=det.get(rel,{})
        if previous.get(rel)!=digest:changes.append(rel)
        im=cv2.imread(str(p)); hashes[digest]+=1
        entry=dict(path=rel,sha256=digest,readable=im is not None,family=family(p.stem),folder=p.relative_to(original).parts[0],
                   group=row.get('source_group') or 'UNKNOWN',split=row.get('split'),license_status='UNKNOWN')
        try:
            with Image.open(p) as pil:
                tags=pil.getexif()
                for key,tag in [('any',None),('camera_make',271),('camera_model',272),('capture_datetime',36867),('orientation',274),('gps_tag',34853),('copyright',33432),('artist',315)]:
                    if (bool(tags) if tag is None else tag in tags):exif[key]+=1
        except OSError:exif['read_error']+=1
        if im is not None:
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            small=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
            a=cv2.dct(small)[:8,:8].flatten(); ph=sum(int(b)<<i for i,b in enumerate(a>np.median(a[1:])))
            entry.update(width=gray.shape[1],height=gray.shape[0],laplacian=float(cv2.Laplacian(gray,cv2.CV_64F).var()),
                         contrast=float(np.std(gray)),phash=f'{ph:016x}',
                         protected_conflict=rel in paths or digest in sha_refs or any((ph^x).bit_count()<=6 for x in phrefs)
                         or (entry['group']!='UNKNOWN' and entry['group'].strip().casefold() in group_refs))
            for other in rows:
                if 'phash' in other:
                    d=(ph^int(other['phash'],16)).bit_count()
                    if d<=6:distances[d]+=1
        rows.append(entry)
    valid=[r for r in rows if r['readable']]
    return rows,dict(total=len(rows),readable=len(valid),exact_duplicate_groups=sum(n>1 for n in hashes.values()),
        exact_extra=sum(n-1 for n in hashes.values()),near_pairs_le6=sum(distances.values()),phash_distance=dict(distances),
        prior_sha_changes=len(changes),prior_missing=len(set(previous)-{r['path'] for r in rows}),exif=dict(exif),
        protected_conflicts=sum(r.get('protected_conflict',False) for r in rows),
        groups_known=len({r['group'] for r in rows if r['group']!='UNKNOWN'}),unknown_group=sum(r['group']=='UNKNOWN' for r in rows),
        families=dict(Counter(r['family'] for r in rows)),folders=dict(Counter(r['folder'] for r in rows)),
        resolutions=Counter(f"{r['width']}x{r['height']}" for r in valid).most_common(12),
        full_frame_quality={k:quantiles([r[k] for r in valid]) for k in ('width','height','laplacian','contrast')})


def exposure(train):
    plate=read(ROOT/'reports/round2_gate_20261124T154153249700Z/dataset/manifest.json')['rows']
    char=read(ROOT/'datasets/yolo_character_round1/manifest.json')['rows']
    cnn=read(ROOT/'experiments/ocr_mix_bundle_20261125/manifest.json')['manifest']
    fusion=read(ROOT/'experiments/hybrid_features_01/samples.json')['samples']
    sets={'plate_train':{r['source_group'] for r in plate if r['split']=='train'},
          'character_train':{r['review']['source_group'] for r in char if r['review']['split']=='train'},
          'cnn_train':{r['group'] for r in cnn if r['origin']=='reviewed_real' and r['split']=='train'},
          'fusion_train':{r['group'] for r in fusion if r['domain'] in ('train','detector_train')}}
    universe={r['source_group'] for r in train}
    return {k:{'groups_seen':len(v&universe),'plates_seen':sum(r['source_group'] in v for r in train)} for k,v in sets.items()}, len(universe-set.union(*sets.values()))


def main(out):
    check_protocol(); rows,summary=inventory()
    reviews=live_rows(); train=[r for r in reviews if r['split']=='train']; train_groups={r['source_group'] for r in train}
    dependencies=[ROOT/'experiments/alpr_v4/pool_pilot'/f for f in ('pseudo_pilot_100.json','filter_on_reviewed_train.json')]
    old,controls=map(read,dependencies)
    counts,unseen=exposure(train)
    bypath={r['path']:r for r in rows}; diagnostics=[]; queue=[]
    for p in old['rows']:
        r=bypath.get(relative_image(p['image']))
        if not r or not r.get('readable') or r.get('protected_conflict') or r.get('split') in ('val','test') or r['group'] in train_groups:
            continue
        if (p['sha256'],p['phash'])!=(r['sha256'],r['phash']):continue
        plate=p['plate']; boxes=p['glyph_boxes']; item=dict(path=r['path'],sha256=r['sha256'],phash=r['phash'],group=r['group'],
            family=r['family'],flags=p['flags'],status='AMBIGUOUS',provenance='UNKNOWN',label_export=False,
            glyph_count=len(boxes),plate_fraction=None,crop_laplacian=None,crop_contrast=None,layout_proxy=None,angle_proxy=None)
        if plate:
            im=cv2.imread(str(ROOT/p['image']));x,y,w,h=plate['crop_box'];gray=cv2.cvtColor(im[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
            item.update(plate_fraction=plate['bbox'][2]*plate['bbox'][3]/(r['width']*r['height']),crop_laplacian=float(cv2.Laplacian(gray,cv2.CV_64F).var()),crop_contrast=float(gray.std()))
            if len(boxes)>=2:
                xy=np.array([[x+w/2,y+h/2] for x,y,w,h in boxes]); gaps=np.diff(np.sort(xy[:,1])); two=max(gaps)>.55*np.median([b[3] for b in boxes]); item['layout_proxy']='two' if two else 'one'
                if not two and np.ptp(xy[:,0])>0:item['angle_proxy']=float(np.degrees(np.arctan(np.polyfit(xy[:,0],xy[:,1],1)[0])))
        diagnostics.append(item)
    priority={'classifier_disagreement','low_detector_score','possible_clipping','ambiguous_reading_order','overlapping_glyph_boxes','plate_missing'}
    pool=deduplicate(sorted((r for r in diagnostics if priority.intersection(r['flags'])),key=lambda r:(-len(priority.intersection(r['flags'])),r['path'])))
    # Round robin across filename families, not claimed independent source groups.
    families=defaultdict(list)
    for r in pool:families[r['family']].append(r)
    while any(families.values()) and len(queue)<20:
        for rs in families.values():
            if rs and len(queue)<20:queue.append(rs.pop(0))
    summary.update(snapshot_hashes={str(p.relative_to(ROOT)):sha(p) for p in dependencies},reviewed_train_plates=len(train),
        reviewed_train_glyphs=sum(len(r['characters']) for r in train),reviewed_train_groups=len(train_groups),teacher_exposure=counts,
        unexposed_reviewed_train_groups=unseen,legacy_diagnostic={k:controls[k] for k in ('plates','accepted','accepted_exact')},
        prior_pool_count=len(old['rows']),prior_pool_filter_count=old['accepted_by_filter'],cached_audited=len(diagnostics),
        cached_flags=dict(Counter(f for r in diagnostics for f in r['flags'])),cached_layout=dict(Counter(r['layout_proxy'] or 'unknown' for r in diagnostics)),
        cached_glyph_counts=dict(Counter(r['glyph_count'] for r in diagnostics)),cached_quality={k:quantiles([r[k] for r in diagnostics if r[k] is not None]) for k in ('plate_fraction','crop_laplacian','crop_contrast','angle_proxy')},
        queue_count=len(queue),queue_groups_known=len({r['group'] for r in queue if r['group']!='UNKNOWN'}),
        gate={'rights_and_provenance':'UNVERIFIED','teacher_holdout':'BLOCKED' if not unseen else 'NOT_VALIDATED',
              'new_benchmark':'BLOCKED','student_training':'BLOCKED','approved_training_labels':0})
    out.mkdir(parents=True,exist_ok=False)
    for name,data in [('inventory',rows),('summary',summary),('cached_quality',diagnostics),('active_learning_candidates',queue)]:put(out/f'{name}.json',data)
    print(out,summary)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    main(ap.parse_args().output)