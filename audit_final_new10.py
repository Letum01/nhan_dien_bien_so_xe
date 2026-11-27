"""Read-only audit of the ten registered imports; no model inference."""
import argparse
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from final_ocr_dataset import ROOT, STORE, audit_candidates, relative_image
from final_ocr_intake import registered
from yolo_character_dataset import ReadOnlyStore

QUEUE=ROOT/'reports/final_test_new10_queue.json'


def audit_new10(store_path=STORE):
    imports=registered(store_path)
    keys=json.loads(QUEUE.read_text(encoding='utf-8')) if QUEUE.is_file() else [r['sha256'] for r in imports]
    if len(keys)!=10 or len(set(keys))!=10 or not set(keys)<={r['sha256'] for r in imports}:
        raise ValueError('Expected exactly ten identifiable imported images; no automatic reselection')
    audit=audit_candidates(store_path,write=False)
    recent={r['sha256']:r for r in audit['rows'] if r.get('imported_at')}
    with sqlite3.connect(Path(store_path).resolve().as_uri()+'?mode=ro',uri=True) as db:
        integrity=db.execute('PRAGMA integrity_check').fetchone()[0]
    if integrity!='ok':raise ValueError('Final Test SQLite integrity check failed')
    reviews={relative_image(r['image_path']):r for r in ReadOnlyStore(store_path).all_latest().values()}
    rows=[]
    for key in keys:
        row=recent[key]; image=cv2.imread(str(ROOT/row['image']))
        warnings=[]; width=height=None; laplacian=None; luminance=None
        if image is None: warnings.append('Không đọc được ảnh')
        else:
            height,width=image.shape[:2]
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            luminance=round(float(np.mean(gray)),1)
            # Shrink only; the score is a whole-image indicator, not a glyph diagnosis.
            if max(width,height)>640:
                gray=cv2.resize(gray,None,fx=640/max(width,height),fy=640/max(width,height),interpolation=cv2.INTER_AREA)
            laplacian=round(float(cv2.Laplacian(gray,cv2.CV_64F).var()),1)
            if min(width,height)<256:warnings.append('Cạnh ngắn <256 px; phóng to kiểm tra nét biển')
            if laplacian<80:warnings.append('Độ nét toàn ảnh thấp theo Laplacian (<80)')
            if luminance<55 or luminance>205:warnings.append('Độ sáng trung bình thấp/cao; kiểm tra vùng biển')
        row={**row,'readable':image is not None,'width':width,'height':height,'quality_warnings':warnings,
             'laplacian':laplacian,'mean_gray':luminance,'review_status':reviews.get(relative_image(row['image']),{}).get('review_status','unlabeled')}
        rows.append(row)
    return {'integrity':integrity,'known_exposure_images':audit['known_exposure_images'],'rows':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pin',action='store_true');args=parser.parse_args()
    result=audit_new10()
    if args.pin and not QUEUE.is_file():
        with QUEUE.open('x',encoding='utf-8') as file:json.dump([r['sha256'] for r in result['rows']],file,indent=2)
    (ROOT/'reports/final_test_new10_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print([(r['filename'],r['width'],r['height'],r['status'],r['quality_warnings']) for r in result['rows']])
    print('indexed:',result['known_exposure_images'])