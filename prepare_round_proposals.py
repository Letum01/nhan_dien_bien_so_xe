"""Precompute OpenCV review aids for the EXISTING queue, never approve labels."""
import hashlib
import json
from pathlib import Path
import time

import cv2

from annotation_store import AnnotationStore
from localization import locate_license_plates
from prepare_annotation_round import readiness

ROOT = Path(__file__).resolve().parent


def prepare(root=ROOT):
    root = Path(root)
    queue = json.loads((root/'reports/annotation_round_01.json').read_text(encoding='utf-8'))['images']
    store = AnnotationStore(root/'data/annotations.sqlite3')
    latest = store.all_latest()
    folder = root/'reports/round_01_proposals'
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for item in queue:
        path = root/item['image_path']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if latest.get(digest, {}).get('review_status') == 'reviewed':
            results.append({'image_path': item['image_path'], 'image_id': digest, 'status': 'skipped_reviewed'})
            continue
        image = cv2.imread(str(path))
        if image is None:
            results.append({'image_path': item['image_path'], 'image_id': digest, 'status': 'image_error'})
            continue
        start = time.perf_counter()
        proposals = locate_license_plates(image, max_plates=10)
        preview = image.copy()
        entries = []
        for index, (_, (x,y,w,h), score) in enumerate(proposals):
            entries.append({'bbox': [x,y,w,h], 'score': float(score), 'score_kind': 'heuristic_not_probability'})
            cv2.rectangle(preview, (x,y), (x+w,y+h), (0,180,255), 2)
            cv2.putText(preview, str(index), (x,max(15,y-3)), cv2.FONT_HERSHEY_SIMPLEX,.5,(0,180,255),1)
        filename = f'{digest}.jpg'
        if not cv2.imwrite(str(folder/filename), preview):
            raise OSError(filename)
        results.append({'image_path': item['image_path'], 'image_id': digest,
                        'status': 'proposed' if entries else 'no_proposals', 'proposals': entries,
                        'preview': str((folder/filename).relative_to(root)),
                        'seconds': time.perf_counter()-start,
                        'source_group': None, 'split': None, 'human_reviewed': False})
    report = {'purpose': 'review_aids_only', 'queue_size': len(queue), 'results': results}
    temporary = folder/'index.tmp'
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(folder/'index.json')
    status = readiness(store, queue)
    (root/'reports/training_readiness.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps({'images':len(results), 'proposed':sum(r['status']=='proposed' for r in results),
                      'no_proposals':sum(r['status']=='no_proposals' for r in results),
                      'reviewed':status['reviewed_images'], 'training':status['status']},indent=2))
    return report


if __name__ == '__main__':
    prepare()