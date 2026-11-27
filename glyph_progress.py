"""Read-only OCR coverage and validation review priorities."""
from collections import Counter
import json
from pathlib import Path
import cv2

from glyph_review import CLASSES,ROOT,plates_from_detection
from dataset_audit import fingerprint


def summarize(detections,glyphs,pilot_ids=(),reviewed_snapshot=None):
    entries=plates_from_detection(detections)
    reviewed=reviewed_snapshot if reviewed_snapshot is not None else [r for r in glyphs.all_latest().values() if r.get('review_status')=='reviewed']
    ids={r['image_id'] for r in reviewed}
    train=[r for r in reviewed if r['split']=='train']
    train_groups={r['source_group'] for r in train}
    train_hashes=[fingerprint(ROOT/r['image_path'])[:2] for r in train]
    queue=[]
    for entry in entries:
        if entry['image_id'] in ids:continue
        independent=None
        if entry['split']=='val':
            digest,phash,_=fingerprint(ROOT/entry['image_path'])
            independent=entry['source_group'] not in train_groups and not any(
                digest==other or (phash is not None and p is not None and (phash^p).bit_count()<=6)
                for other,p in train_hashes)
        queue.append({**entry,'in_original_pilot':entry['image_id'] in pilot_ids,
                      'independent_of_reviewed_ocr_train':independent})
    queue.sort(key=lambda r:(not(r['split']=='val' and r['independent_of_reviewed_ocr_train']),
                             r['split']=='test',not r['text_confirmed'],r['image_path']))
    counts=Counter(c['label'] for r in reviewed for c in r['characters'] if c.get('confirmed'))
    edge=[]
    for r in reviewed:
        w,h=r['plate_bbox'][2:]
        image=cv2.imread(str(ROOT/r['image_path']))
        px,py,_,_=r['plate_bbox']
        for i,c in enumerate(r['characters']):
            x,y,cw,ch=c['bbox']
            if min(x,y,w-x-cw,h-y-ch)<=2:
                edge.append({'image':r['image_path'],'glyph_index':i,'label':c['label'],'reason':'within_2px_plate_crop_edge_NOT_proven_cutoff'})
            if image is not None:
                roi=image[py+y:py+y+ch,px+x:px+x+cw]
                if roi.size:
                    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                    _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
                    if c.get('invert'):binary=cv2.bitwise_not(binary)
                    touching=any((side>0).mean()>.15 for side in (binary[0],binary[-1],binary[:,0],binary[:,-1]))
                    if touching:edge.append({'image':r['image_path'],'glyph_index':i,'label':c['label'],
                                              'reason':'foreground_touches_glyph_boundary_heuristic_NOT_proven_cutoff'})
    split_rows=[]
    for split in ('train','val','test'):
        subset=[r for r in reviewed if r['split']==split]
        split_rows.append({'split':split,'available_plates':sum(r['split']==split for r in entries),
                           'reviewed_plates':len(subset),'reviewed_glyphs':sum(len(r['characters']) for r in subset),
                           'remaining':sum(r['split']==split for r in queue)})
    return {'reviewed_plates':len(reviewed),'reviewed_glyphs':sum(counts.values()),'split_progress':split_rows,
            'class_counts':{c:counts[c] for c in CLASSES},'missing_classes':[c for c in CLASSES if not counts[c]],
            'low_sample_classes':[c for c in CLASSES if 0<counts[c]<5],
            'low_sample_rule':'<5 is review prioritization only, not statistical sufficiency',
            'edge_warnings':edge,'remaining_pilot':sum(r['in_original_pilot'] for r in queue),
            'remaining_all':len(queue),'queue':queue,
            'independence_scope':'relative to currently reviewed OCR train only; revalidate on export, not model/test independence claim'}


def main():
    from annotation_store import AnnotationStore
    pilot=json.loads((ROOT/'reports/glyph_pilot.json').read_text(encoding='utf-8'))
    report=summarize(AnnotationStore(ROOT/'data/annotations.sqlite3'),AnnotationStore(ROOT/'data/ocr_glyphs.sqlite3'),set(pilot['plate_ids']))
    from datetime import datetime,timezone
    folder=ROOT/'reports'/('glyph_progress_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));folder.mkdir()
    (folder/'progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(folder)
    print({k:report[k] for k in ('reviewed_plates','reviewed_glyphs','split_progress','remaining_pilot','missing_classes','low_sample_classes')})


if __name__=='__main__':main()