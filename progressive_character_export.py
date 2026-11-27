"""Export one-class YOLO training data only from independently approved complete images."""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil

from external_task_curation import digest, load_image, read, safe_path, write
from external_task_train import evidence
from final_ocr_dataset import check_protocol
from progressive_ocr import approved_rows
from yolo_character_dataset import integer_boxes, normalized


def plan(curation, approval, extracted):
    rows,frozen,_=approved_rows(curation,approval)
    auth=read(approval); base=approval.parent
    receipt=auth.get('character_detector_approval')
    if not evidence(base,receipt): raise ValueError('BLOCKED: dedicated detector permission and completeness receipt required')
    scope=read(safe_path(base,receipt['path']))
    if scope.get('purpose')!='academic_character_detection_training' or scope.get('frozen_sha256')!=frozen or not evidence(base,scope.get('rights_evidence')):
        raise ValueError('Invalid detector scope or missing rights evidence')
    by_parent=defaultdict(list)
    for r in rows: by_parent[r['parent']].append(r)
    all_labels=defaultdict(list)
    for r in read(curation/'glyphs.json'): all_labels[r['parent']].append(r)
    result=[]; seen=set()
    for entry in scope['images']:
        p=entry['parent']; records=by_parent.get(p,[])
        if (entry.get('all_glyphs_verified') is not True or entry.get('is_actual_plate') is not True
                or not str(entry.get('reviewer') or '').strip() or not evidence(base,entry.get('completeness_evidence'))
                or p in seen or not records or records[0]['split']=='test'
                or len(records)!=len(all_labels[p]) or any(r['sha256']!=entry['sha256'] for r in records)):
            raise ValueError(f'Unverified, incomplete, duplicate, or held-out image: {p}')
        seen.add(p); image=safe_path(extracted/'OCR_Data',p)
        if digest(image)!=entry['sha256']: raise ValueError(f'Image changed: {p}')
        h,w=load_image(image).shape[:2]; boxes=[r['pixel_xywh'] for r in records]
        integer_boxes(boxes,w,h)
        if len(set(map(tuple,boxes)))!=len(boxes): raise ValueError(f'Duplicate box: {p}')
        result.append((image,'train' if records[0]['split']=='train' else 'val',p,records[0],
                       [normalized(b,w,h) for b in boxes]))
    if {s for _,s,*_ in result}!={'train','val'}: raise ValueError('Both complete train and development images required')
    return result, frozen


def export(items, frozen, output):
    output.mkdir(parents=True,exist_ok=False)
    for s in ('train','val'):
        for d in ('images','labels'): (output/d/s).mkdir(parents=True)
    manifest=[]
    for i,(image,s,p,row,boxes) in enumerate(items):
        dest=output/'images'/s/f'{i:06d}{image.suffix.lower()}'
        shutil.copyfile(image,dest)
        if digest(dest)!=row['sha256']: raise ValueError(f'Copy hash mismatch: {p}')
        label=output/'labels'/s/f'{i:06d}.txt'
        label.write_text(''.join('0 '+' '.join(format(v,'.12g') for v in b)+'\n' for b in boxes),encoding='ascii')
        manifest.append(dict(parent=p,split=s,sha256=row['sha256'],source_group=row['source_group'],
                             image=dest.relative_to(output).as_posix(),label=label.relative_to(output).as_posix(),
                             label_sha256=digest(label),box_count=len(boxes)))
    write(output/'data.yaml',dict(path=str(output.resolve()),train='images/train',val='images/val',nc=1,names=['character']))
    write(output/'manifest.json',dict(frozen_sha256=frozen,test_exported=False,images=manifest))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('curation',type=Path); ap.add_argument('--approval',type=Path)
    ap.add_argument('--extracted',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    try:
        check_protocol(); items,frozen=plan(a.curation,a.approval,a.extracted); export(items,frozen,a.output)
    except (OSError,ValueError,KeyError) as e: ap.exit(2,f'{e}\n')