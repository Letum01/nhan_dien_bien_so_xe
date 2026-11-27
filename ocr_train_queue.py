"""Frozen train-only human-review queue from the latest milestone suggestions."""
import hashlib
import json
from pathlib import Path

from glyph_review import ROOT,plates_from_detection,CLASSES

QUEUE=ROOT/'reports/ocr_train_priority.json'


def load_queue(detections):
    document=json.loads(QUEUE.read_text(encoding='utf-8'))
    parents={r['image_id']:r for r in plates_from_detection(detections)}
    entries=[]
    for saved in document['entries']:
        current=parents.get(saved['plate_id'])
        if not current or any(current.get(k)!=saved.get(k) for k in ('source_group','split','detection_revision','plate_bbox')):
            raise ValueError('Train queue parent changed: '+saved['image_path']+'; audit before continuing')
        if current['split']!='train':raise ValueError('Queue contains non-train plate')
        entries.append(current)
    return document,entries


def prepare():
    from annotation_store import AnnotationStore
    source=sorted((ROOT/'reports').glob('glyph_audit_*/milestone_summary.json'))[-1]
    suggestions=json.loads(source.read_text(encoding='utf-8'))['next_train_review']
    detections=AnnotationStore(ROOT/'data/annotations.sqlite3')
    parents={r['image_id']:r for r in plates_from_detection(detections)}
    entries=[]
    for suggestion in suggestions:
        row=parents[suggestion['plate_id']]
        if row['split']!='train' or row['source_group']!=suggestion['source_group']:raise ValueError('Suggestion no longer matches parent')
        entries.append({'plate_id':row['image_id'],'image_path':row['image_path'],
                        'source_group':row['source_group'],'split':row['split'],
                        'detection_revision':row['detection_revision'],'plate_bbox':row['plate_bbox'],
                        'text_confirmed_at_preparation':row['text_confirmed']})
    entries.sort(key=lambda r:not r['text_confirmed_at_preparation'])  # stable, retain suggested order
    if len(entries)!=15:raise ValueError('Expected exactly 15 suggestions')
    document={'name':'OCR Train — ưu tiên','source_milestone':str(source.relative_to(ROOT)),
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'entries':entries,'reviewed_by_agent':False,'training':'NOT_REQUESTED'}
    if QUEUE.exists():raise FileExistsError('Queue already exists; never silently replace human work plan')
    QUEUE.write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Queue15; existing confirmed texts',sum(r['text_confirmed_at_preparation'] for r in entries))


if __name__=='__main__':prepare()