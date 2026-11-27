"""Read-only provenance screening. Candidates are not approved independent sources."""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from dataset_audit import fingerprint
from dual_yolo_client import CHARACTER, CHECKPOINT
from glyph_review import ROOT
from yolo_character_dataset import ReadOnlyStore, sha, validate
from yolo_character_pipeline import WEIGHTS


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))


def main():
    validate(ROOT/'datasets/yolo_character_round1')
    detection=list(ReadOnlyStore(ROOT/'data/annotations.sqlite3').all_latest().values())
    reviewed=ReadOnlyStore(ROOT/'data/ocr_glyphs.sqlite3').all_latest()
    plate=read('reports/round2_gate_20261124T154153249700Z/dataset/manifest.json')['rows']
    pilot=read('reports/annotation_review_20261117T152948733793Z/yolo_trial/manifest.json')['rows']
    char=[e['review'] for e in read('datasets/yolo_character_round1/manifest.json')['rows']]
    mix=read('experiments/ocr_mix_bundle_20261125/manifest.json')['manifest']
    mix_train={r['source_image_id'] for r in mix if r.get('source_image_id') and r['split']=='train'}
    train_ids={
        'plate':{r['image_id'] for r in plate if r['split']=='train'},
        'pilot_plate':{r['image_id'] for r in pilot if r['split']=='train'},
        'character':{r['source_image_id'] for r in char if r['split']=='train'},
        'mix_real':mix_train,
    }
    exposed=[r for r in detection if r['split'] in ('train','val')]
    fingerprints=[(r,*fingerprint(ROOT/r['image_path'])[:2]) for r in exposed]
    history=read('experiments/pilot21_20261117_run01/inference_audit/predictions.json')['results']
    tests=[]
    for r in detection:
        if r['review_status']!='reviewed' or r['split']!='test': continue
        digest,p,_=fingerprint(ROOT/r['image_path'])
        tests.append({'image':r['image_path'],'group':r['source_group'],
                      'training_overlap':{k:r['image_id'] in ids for k,ids in train_ids.items()},
                      'group_or_hash_overlap':[o['image_path'] for o,d,h in fingerprints if r['source_group']==o['source_group'] or d==digest or (h^p).bit_count()<=6],
                      'prior_pilot_diagnostic':any(Path(a['image']).resolve()==(ROOT/r['image_path']).resolve() for a in history),
                      'glyph_reviewed':any(a.get('source_image_id')==r['image_id'] and a['review_status']=='reviewed' for a in reviewed.values())})
    seen={str(Path(r['image_path'])) for r in detection}
    for path in ('reports/round2_plan/candidates.csv',):
        with (ROOT/path).open(encoding='utf-8-sig',newline='') as stream:
            seen.update(str(Path(r['image_path'])) for r in csv.DictReader(stream))
    # Include all indexed debug JSON references where available; unknown exposure still blocks approval.
    for folder in ('reports','debug_outputs','experiments'):
        for path in (ROOT/folder).rglob('*.json'):
            # Inventory records show existence, not model or development exposure.
            if path.name in ('dataset_audit.json','ocr_test_provenance.json','final_ocr_protocol.json'): continue
            try: data=json.loads(path.read_text(encoding='utf-8-sig'))
            except (ValueError,OSError): continue
            stack=[data]
            while stack:
                item=stack.pop()
                if isinstance(item,dict): stack.extend(item.values())
                elif isinstance(item,list): stack.extend(item)
                elif isinstance(item,str) and 'license_plate_dataset' in item and item.lower().endswith(('.jpg','.png','.jpeg')):
                    p=Path(item)
                    if p.is_absolute():
                        try:p=p.relative_to(ROOT)
                        except ValueError:continue
                    if p.parts and p.parts[0]=='license_plate_dataset': p=Path('dataset')/p
                    seen.add(str(p))
    strata=defaultdict(list); n=0; seen_hash=[]
    with (ROOT/'reports/dataset_audit.csv').open(encoding='utf-8-sig',newline='') as stream:
        for r in csv.DictReader(stream):
            name=str(Path(r['image_path'])); path=ROOT/name
            if not path.is_file() or not name.startswith(str(Path('dataset/license_plate_dataset'))):continue
            if name in seen and r['phash']:
                seen_hash.append((r['sha256'],int(r['phash'],16)))
            if name in seen or r['readable']!='True' or not r['phash']:continue
            w,h=int(r['width']),int(r['height']); n+=1
            strata[(Path(name).parts[2],w<h,max(w,h)<800)].append(r)
    rng=random.Random(42)
    for rows in strata.values():rng.shuffle(rows)
    selected=[]; hashes=[]
    while len(selected)<40 and any(strata.values()):
        for key in sorted(strata):
            if not strata[key] or len(selected)>=40:continue
            r=strata[key].pop(); d,p,shape=fingerprint(ROOT/r['image_path'])
            if not shape or d!=r['sha256'] or any(d==other or (p^h).bit_count()<=6 for other,h in seen_hash+hashes):continue
            selected.append({'image':r['image_path'],'sha256':d,'phash':f'{p:016x}','stratum':key,'source_group':'UNVERIFIED'})
            hashes.append((d,p))
    out=ROOT/'reports'
    (out/'ocr_test_provenance.json').write_text(json.dumps({'pilot_test':tests,'candidate_pool':n,'candidates':selected,
        'limitations':'Seed42 round-robin by folder, orientation, dimensions. No detector/OCR scores used. File/group exposure outside known artifacts (including external pretraining) not proven; human metadata review required. Pilot tests previously diagnosed at .4/.001, not blind. No change to SQLite.'},indent=2),encoding='utf-8')
    files=['dual_yolo_worker.py','dual_yolo_pipeline.py','dual_yolo_client.py','yolo_detector.py','yolo_character_worker.py','yolo_character_pipeline.py','segmentation.py','segmentation_benchmark.py','benchmark_crop_padding.py','benchmark_yolo_character.py','dataset.py','ensemble_recognition.py']
    protocol={'status':'BLOCKED — no approved independent corpus','primary':'CNN Mix','secondary':['SVM Mix','RF Mix','Majority Mix'],
              'controls':['OpenCV baseline','border_components'],'plate_conf':.4,'character_conf':.25,
              'nms':{'iou':.7,'plate_max_det':300,'character_max_det':40},'precision':'FP32, single image',
              'crop':'Top score plate, 15% per side clipped; separate GT control unpadded',
              'ordering':'Existing .55 median height gap, then left-to-right',
              'normalization':'Otsu polarity from ROI edge, square pad, resize 32x32, threshold 127, existing Mix contract',
              'metrics':'IoU .5 one-to-one TP/FP/FN, matched OCR accuracy, CER and exact over all plates; missing crop is a failure',
              'gates':['human-reviewed independent source provenance','reviewed labels and validator PASS','group/sha256/pHash leakage PASS','freeze corpus IDs/hash before inference','no post-test threshold or model selection'],
              'hashes':{str(p.relative_to(ROOT)):sha(p) for p in (CHECKPOINT,CHARACTER,*WEIGHTS.values(),*(ROOT/f for f in files))}}
    protocol_path=out/'final_ocr_protocol.json'
    if protocol_path.exists():
        if read('reports/final_ocr_protocol.json')!=protocol:raise ValueError('Frozen final OCR protocol differs; refusing to overwrite')
    else:
        with protocol_path.open('x',encoding='utf-8') as stream:json.dump(protocol,stream,indent=2)
    print('Pilot',len(tests),'screened candidates',len(selected))


if __name__=='__main__':main()