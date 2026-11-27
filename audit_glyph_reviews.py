"""Read-only snapshot audit/export of actual glyph reviews, no training."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np

from annotation_store import AnnotationStore
from glyph_review import plates_from_detection,validate_glyph_plate,export_glyph_dataset
from dataset_validation import check_split_leakage
from validate_glyph_dataset import validate

ROOT=Path(__file__).resolve().parent


def main(ocr_path=None,detection_path=None):
    folder=ROOT/'reports'/('glyph_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    folder.mkdir(parents=True,exist_ok=False)
    # Backup each database consistently; review compares immutable parent versions.
    for filename in ('ocr_glyphs.sqlite3','annotations.sqlite3'):
        source=Path(ocr_path if filename=='ocr_glyphs.sqlite3' else detection_path) if (ocr_path if filename=='ocr_glyphs.sqlite3' else detection_path) else ROOT/'data'/filename
        with sqlite3.connect(f'file:{source.as_posix()}?mode=ro',uri=True) as c,sqlite3.connect(folder/filename) as target:
            c.backup(target)
    ocr=AnnotationStore(folder/'ocr_glyphs.sqlite3');detection=AnnotationStore(folder/'annotations.sqlite3')
    rows=[r for r in ocr.all_latest().values() if r.get('review_status')=='reviewed']
    parents={r['image_id']:r for r in plates_from_detection(detection)}
    details=[];counts=Counter();failures=[]
    for row in rows:
        issues=[];parent=parents.get(row['image_id'])
        for field in ('source_image_id','plate_bbox','source_group','split','detection_revision'):
            if not parent or parent.get(field)!=row.get(field):issues.append('Parent mismatch: '+field)
        path=ROOT/row['image_path'];image=cv2.imread(str(path))
        if image is None:issues.append('Source unreadable')
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=row['source_image_id']:issues.append('Source hash mismatch')
        try:sequence=validate_glyph_plate(row,row['plate_bbox'][2],row['plate_bbox'][3])
        except ValueError as exc:issues.append(str(exc));sequence=None
        orders=[c.get('order') for c in row['characters']]
        if orders!=list(range(len(orders))):issues.append('Stored order indices not contiguous')
        if row['layout']=='two_lines' and {c['line'] for c in row['characters']}!={0,1}:issues.append('Two-line annotation missing a row')
        if image is not None:
            x,y,w,h=row['plate_bbox'];crop=image[y:y+h,x:x+w]
            sheet=np.full((190,len(row['characters'])*112,3),245,np.uint8)
            for i,char in enumerate(row['characters']):
                cx,cy,cw,ch=char['bbox'];glyph=crop[cy:cy+ch,cx:cx+cw]
                factor=min(96/max(cw,1),125/max(ch,1));gw,gh=max(1,int(cw*factor)),max(1,int(ch*factor))
                tile=cv2.resize(glyph,(gw,gh))
                sheet[8:8+gh,i*112+8:i*112+8+gw]=tile
                cv2.putText(sheet,f"#{i} {char['label']}",(i*112+5,155),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1)
                cv2.putText(sheet,f"row {char['line']}",(i*112+5,180),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
                counts[char['label']]+=1
            name=Path(row['image_path']).stem
            cv2.imwrite(str(folder/f'{name}_contact_raw.jpg'),sheet)
            overlay=crop.copy()
            for i,char in enumerate(row['characters']):
                cx,cy,cw,ch=char['bbox'];cv2.rectangle(overlay,(cx,cy),(cx+cw,cy+ch),(0,255,0),1)
                cv2.putText(overlay,f"{i}:{char['label']}",(cx,max(10,cy-2)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,255),1)
            cv2.imwrite(str(folder/f'{name}_glyph_overlay.jpg'),overlay)
        detail={'image':row['image_path'],'ocr_revision':row['annotation_version'],'parent_revision':row['detection_revision'],
                'text':row['plate_text'],'sequence':sequence,'layout':row['layout'],'source_group':row['source_group'],
                'split':row['split'],'glyphs':len(row['characters']),'status':'FAIL' if issues else 'PASS','issues':issues}
        details.append(detail)
        if issues:failures.append(detail)
    leakage={'status':'PASS','note':'Only available source splits checked; no val/test samples invented.'}
    try:check_split_leakage(rows,ROOT)
    except ValueError as exc:leakage={'status':'FAIL','reason':str(exc)}
    exported={'status':'BLOCKED'}
    if not failures and leakage['status']=='PASS' and rows:
        count=export_glyph_dataset(ocr,detection,folder/'dataset')
        exported=validate(folder/'dataset');exported['exported_count']=count
        # Binary-normalized contact sheets also reveal erased glyph strokes.
        import csv
        manifest=list(csv.DictReader((folder/'dataset/characters.csv').open(encoding='utf-8-sig')))
        for plate in rows:
            entries=[r for r in manifest if r['plate_id']==plate['image_id']]
            sheet=np.full((100,80*len(entries),3),240,np.uint8)
            for i,entry in enumerate(entries):
                glyph=cv2.imread(str(folder/'dataset'/entry['image_path']))
                sheet[2:66,i*80+8:i*80+72]=cv2.resize(glyph,(64,64),interpolation=cv2.INTER_NEAREST)
                cv2.putText(sheet,entry['class_label'],(i*80+25,90),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),1)
            cv2.imwrite(str(folder/(Path(plate['image_path']).stem+'_contact_normalized.png')),sheet)
    report={'actual_glyph_reviewed_plates':len(rows),'confirmed_glyphs':sum(c.get('confirmed') is True for r in rows for c in r['characters']),
            'counts_by_class':dict(sorted(counts.items())),'splits':dict(Counter(r['split'] for r in rows)),
            'details':details,'leakage':leakage,'export_validator':exported,
            'training':'NOT_RUN_USER_REQUEST','ground_truth_modified':False,
            'visual_status':'Contact sheets created for human inspection; not automatically reapproved'}
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Audit glyph reviewed',f"\nPlates: {len(rows)}; confirmed glyphs: {report['confirmed_glyphs']}; splits: {report['splits']}.",
           f"\nClass counts: {report['counts_by_class']}",f"\nLeakage: {leakage}; export/validator: {exported}",
           '\n|Image|Text/sequence|Layout|Group|Split|Glyphs|Status/issues|','|---|---|---|---|---|---:|---|']
    for r in details:lines.append(f"|{r['image']}|{r['sequence']}|{r['layout']}|{r['source_group']}|{r['split']}|{r['glyphs']}|{r['status']} {r['issues']}|")
    lines.append('\nNo training or label changes. Validation PASS does not prove visual glyph correctness or sufficient independent training data. Inspect raw and normalized contact sheets.')
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2));print('REPORT',folder)
    return folder,report


if __name__=='__main__':main()