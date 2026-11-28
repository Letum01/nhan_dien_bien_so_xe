"""Validated glyph snapshot helpers used by the retained Fusion recipe; no V2 trainer."""
import csv
import json
import numpy as np
from dataset import extract_hog_features
from final_ocr_dataset import ROOT, check_protocol, read, conflicts, audit_candidates
from dataset_audit import fingerprint
from validate_glyph_dataset import validate
from yolo_character_dataset import validate as validate_character
from yolo_character_dataset import sha
from glyph_review import CLASSES
REAL = ROOT/'reports/glyph_audit_20261125T145221295978Z/dataset'
CHAR = ROOT/'datasets/yolo_character_round1'


def put(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')



def prepare():
    check_protocol(); validate(REAL); validate_character(CHAR)
    exported={r['image_id']:r for r in read(REAL/'ocr_revisions.json') if r['split'] in ('train','val')}
    expected={r['review']['image_id']:r['review'] for r in read(CHAR/'manifest.json')['rows']}
    if exported!=expected:raise ValueError('Glyph snapshot and validated Character revisions differ')
    a=audit_candidates(write=False)
    forbidden=[{**r,'source_group':r.get('metadata',{}).get('source_group','')} for r in a['rows']]
    forbidden.extend(r for r in a['known'] if r['split']=='test')
    for row in exported.values():
        if row['split']=='train':
            d,p,shape=fingerprint(ROOT/row['image_path'])
            if not shape or conflicts(d,p,row['source_group'],forbidden):raise ValueError('Test exposure conflict: '+row['image_path'])
    with (REAL/'characters.csv').open(encoding='utf-8-sig',newline='') as f:
        buckets={s:[] for s in ('train','validation')}
        for row in csv.DictReader(f):
            if row['split'] not in buckets:continue
            rev=exported[row['plate_id']]
            if (row['source_image_id'],row['source_group'],row['split'])!=(rev['source_image_id'],rev['source_group'],'train' if rev['split']=='train' else 'validation'):
                raise ValueError('Glyph manifest provenance mismatch')
            buckets[row['split']].append({**row,'sha256':sha(REAL/row['image_path'])})
    if tuple(map(len,buckets.values()))!=(370,99):raise ValueError('Unexpected glyph inventory')
    return buckets,exported



def weights(s,r):
    total=(s+r)/2
    return np.r_[np.full(s,total/s),np.full(r,total/r)]



def hog(images):return np.stack([extract_hog_features(x) for x in images])
