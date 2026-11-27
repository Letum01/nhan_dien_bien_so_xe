"""Build hard13 contact sheet and error/class tables without editing labels."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from glyph_review import CLASSES
from yolo_character_dataset import validate, sha


def main(audit, dataset):
    validation=validate(dataset)
    result=json.loads((audit/'hard13_v2_results.json').read_text(encoding='utf-8'))
    ids={r['image_id'] for r in result['protocol']['plates']}
    with (audit/'dataset/characters.csv').open(encoding='utf-8-sig',newline='') as stream:
        glyphs=[r for r in csv.DictReader(stream) if r['plate_id'] in ids]
    if len(glyphs)!=100: raise ValueError('Expected 100 audited hard13 glyphs')
    sheet=np.full((10*86,10*78,3),245,np.uint8)
    for i,r in enumerate(glyphs):
        raw=audit/'dataset'/r['image_path']; raw=raw.parent/'originals'/raw.name
        image=cv2.imread(str(raw))
        if image is None:
            image=cv2.imread(str(audit/'dataset'/r['image_path']))
        h,w=image.shape[:2]; scale=min(68/w,59/h)
        image=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
        y,x=(i//10)*86,(i%10)*78; h,w=image.shape[:2]
        sheet[y:y+h,x:x+w]=image
        cv2.putText(sheet,f'{i}: {r["class_label"]}',(x,y+77),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
    if not cv2.imwrite(str(audit/'hard13_contact.jpg'),sheet): raise OSError('Contact sheet write failed')
    (audit/'hard13_contact_index.json').write_text(json.dumps(glyphs,indent=2),encoding='utf-8')
    manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    counts={s:Counter(c['label'] for e in manifest['rows'] if e['review']['split']==s for c in e['review']['characters']) for s in ('train','val')}
    with (audit/'character_classes_by_split.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['class','train','val']);writer.writerows((c,counts['train'][c],counts['val'][c]) for c in CLASSES)
    base={(r['plate_id'],r['crop']):r for r in result['records'] if r['variant']=='baseline'}
    errors=[]
    for r in result['records']:
        m=r['matching']; b=base[r['plate_id'],r['crop']]
        errors.append({'image':r['image'],'group':r['group'],'crop':r['crop'],'variant':r['variant'],
                       **{k:m[k] for k in ('tp','fp','fn','unmatched_gt','unmatched_pred','complete_order_correct')},
                       'covered':r['covered'],'matched_correct':r['matched_correct'],'prediction':r['prediction'],'truth':r['truth'],
                       'fn_delta_vs_baseline':m['fn']-b['matching']['fn'],
                       'lost_exact_vs_baseline':b['prediction']==b['truth'] and r['prediction']!=r['truth']})
    with (audit/'hard13_errors.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(errors[0]));writer.writeheader();writer.writerows(errors)
    out={'dataset':validation,'manifest_sha256':sha(dataset/'manifest.json'),
         'coverage':{s:{'glyphs':sum(r['covered'] for r in base.values() if r['crop']==s),
                        'plates':sum(r['covered']==len(r['gt_boxes']) for r in base.values() if r['crop']==s)} for s in ('gt','yolo')},
         'classification':{s:{v:sum(r['matched_correct'] for r in result['records'] if r['crop']==s and r['variant']==v)
                              for v in result['protocol']['variants']} for s in ('gt','yolo')},
         'regressions':[r for r in errors if r['fn_delta_vs_baseline']>0 or r['lost_exact_vs_baseline']],
         'missing':{s:[c for c in CLASSES if not n[c]] for s,n in counts.items()},
         'few':{s:[c for c in CLASSES if 0<n[c]<=2] for s,n in counts.items()}}
    (audit/'character_final_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit',type=Path);parser.add_argument('dataset',type=Path)
    args=parser.parse_args();main(args.audit,args.dataset)