"""Validate frozen hard-case queue against current parents, never label data."""
import hashlib
import json
from pathlib import Path

import cv2

from glyph_review import ROOT,plates_from_detection

QUEUE=ROOT/'reports/ocr_hard_queue_20261125T091023673153Z/queue.json'
FIELDS=('source_image_id','image_path','plate_bbox','detection_revision','source_group','split','plate_index')


def load_hard_queue(detections,glyphs,queue_path=QUEUE,reviewed_snapshot=None):
    document=json.loads(Path(queue_path).read_text(encoding='utf-8'))
    parents={r['image_id']:r for r in plates_from_detection(detections)}
    reviewed=({r['image_id']:r for r in reviewed_snapshot} if reviewed_snapshot is not None else
              {k:v for k,v in glyphs.all_latest().items() if v.get('review_status')=='reviewed'})
    pending=[];completed=[];errors=[];seen=set()
    for saved in sorted(document['candidates'],key=lambda r:r['rank']):
        identity=saved['image_id'];parent=parents.get(identity)
        if identity in seen:errors.append({'image':saved['image_path'],'reason':'Duplicate queue identity'});continue
        seen.add(identity)
        if not parent or any(parent.get(k)!=saved.get(k) for k in FIELDS) or saved['split']!='train':
            errors.append({'image':saved['image_path'],'reason':'Detection parent/group/split/revision mismatch'});continue
        path=ROOT/parent['image_path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=parent['source_image_id'] or cv2.imread(str(path)) is None:
            errors.append({'image':saved['image_path'],'reason':'Source image missing, changed or unreadable'});continue
        if identity in reviewed:
            if any(reviewed[identity].get(k)!=parent.get(k) for k in FIELDS):
                errors.append({'image':saved['image_path'],'reason':'Reviewed OCR references changed parent'});continue
            completed.append(reviewed[identity]);continue
        # Current parent text flags may be newer; proposal metrics never become labels.
        entry={**parent,'hard_priority':{k:saved.get(k) for k in ('rank','pixel_area','laplacian_variance',
              'edge_rectangle_angle_proxy','border_foreground_fraction','priority_score')}}
        pending.append(entry)
    return {'total':len(document['candidates']),'pending':pending,'completed':completed,'errors':errors,
            'queue_path':str(queue_path),'metadata_kind':'unverified priority proxies, NOT ground truth'}