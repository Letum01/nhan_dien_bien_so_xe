"""OCR-only annotation contract; detection revisions remain read-only."""
import copy
import hashlib
import json
from pathlib import Path

import cv2

from annotation_workflow import validate_plate_text
from dataset_validation import validate_boxes,check_split_leakage

ROOT=Path(__file__).resolve().parent
CLASSES='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def normalized_text(text):
    """Remove separators only; never substitute ambiguous letters/digits."""
    return ''.join(ch for ch in text.upper() if ch.isascii() and ch.isalnum())


def plates_from_detection(store):
    entries=[]
    for row in store.all_latest().values():
        if row.get('review_status')!='reviewed':continue
        for index,box in enumerate(row.get('boxes',[])):
            key=hashlib.sha256(json.dumps([row['image_id'],index,box],sort_keys=True).encode()).hexdigest()
            plate=row.get('plates',[])[index] if index<len(row.get('plates',[])) else {}
            confirmed=bool(plate.get('text_confirmed') and plate.get('plate_text','').strip() and not plate.get('unreadable'))
            entries.append({'image_id':key,'source_image_id':row['image_id'],'image_path':row['image_path'],
                'detection_revision':row['annotation_version'],'plate_index':index,'plate_bbox':box,
                'source_group':row['source_group'],'split':row['split'],
                'plate_text':plate.get('plate_text',''),'text_confirmed':confirmed,
                'unreadable':plate.get('unreadable',False),'layout':plate.get('layout','UNKNOWN'),
                'characters':[],'review_status':'unlabeled','confirmed_by_user':False})
    return sorted(entries,key=lambda row:(not row['text_confirmed'],row['image_path'],row['plate_index']))


def validate_glyph_plate(row,width,height):
    validate_plate_text(row)
    if row.get('layout') not in ('one_line','two_lines'):
        raise ValueError('Confirm one_line/two_lines before glyph review')
    chars=row.get('characters',[])
    if not chars:raise ValueError('No glyphs; OpenCV proposals are not reviewed labels')
    validate_boxes([c['bbox'] for c in chars],width,height)
    for index,c in enumerate(chars):
        if c.get('label') not in CLASSES or len(c.get('label',''))!=1 or not c.get('confirmed'):
            raise ValueError(f'Glyph {index}: exact single class + explicit confirmation required')
        if c.get('line') not in (0,1) or (row['layout']=='one_line' and c['line']!=0):
            raise ValueError(f'Glyph {index}: invalid row')
        x,y,w,h=c['bbox']
        for other in chars[:index]:
            ox,oy,ow,oh=other['bbox']
            intersection=max(0,min(x+w,ox+ow)-max(x,ox))*max(0,min(y+h,oy+oh)-max(y,oy))
            if intersection/min(w*h,ow*oh)>.35:
                raise ValueError('Abnormally overlapping glyphs; inspect split/merge manually')
    ordered=sorted(chars,key=lambda c:(c['line'],c['bbox'][0]))
    if chars!=ordered:raise ValueError('Sort reading order by row and left-to-right before review')
    observed=''.join(c['label'] for c in ordered)
    expected=normalized_text(row['plate_text'])
    if observed!=expected:
        raise ValueError(f'Glyph sequence/count mismatch: {observed!r} ({len(chars)}) != human text {expected!r} ({len(expected)}). Text NOT changed.')
    if not row.get('source_group') or row.get('split') not in ('train','val','test'):
        raise ValueError('Parent source group/split required')
    return observed


def export_glyph_dataset(ocr_store,detection_store,destination,root=ROOT):
    """Validate every reviewed row before creating a new export directory."""
    root=Path(root);destination=Path(destination)
    parents={r['image_id']:r for r in plates_from_detection(detection_store)}
    rows=[r for r in ocr_store.all_latest().values() if r.get('review_status')=='reviewed']
    if not rows:raise ValueError('No Glyph reviewed rows')
    for row in rows:
        parent=parents.get(row['image_id'])
        if not parent or any(row.get(k)!=parent.get(k) for k in ('source_image_id','plate_bbox','source_group','split','detection_revision')):
            raise ValueError('Parent detection changed; recheck glyph review before export')
        image=cv2.imread(str(root/row['image_path']))
        if image is None:raise ValueError('Parent image unreadable')
        validate_glyph_plate(row,row['plate_bbox'][2],row['plate_bbox'][3])
    check_split_leakage(rows,root)
    destination.mkdir(parents=True,exist_ok=False)
    from label_characters import export_reviewed_glyphs
    import csv
    manifest=[]
    for row in rows:
        image=cv2.imread(str(root/row['image_path']));x,y,w,h=row['plate_bbox']
        crop_folder=destination/'plate_crops';crop_folder.mkdir(exist_ok=True)
        crop_path=crop_folder/(row['image_id']+'.png')
        if not cv2.imwrite(str(crop_path),image[y:y+h,x:x+w]):raise OSError(crop_path)
        split='validation' if row['split']=='val' else row['split']
        target=destination/split/row['image_id']
        export_reviewed_glyphs(crop_path,row['characters'],target,row['source_group'],split)
        with (target/'characters_to_review.csv').open(encoding='utf-8-sig',newline='') as stream:
            for glyph in csv.DictReader(stream):
                glyph['image_path']=str((target/glyph['image_path']).relative_to(destination))
                glyph.update(source_image_id=row['source_image_id'],plate_id=row['image_id'],
                             ocr_revision=row['annotation_version'],detection_revision=row['detection_revision'])
                manifest.append(glyph)
    with (destination/'characters.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    (destination/'ocr_revisions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    return len(manifest)