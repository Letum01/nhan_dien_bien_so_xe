"""Read-only round-two training gate, separate quality flags from YOLO syntax.

Never correct/drop annotations or change split membership to make validation pass.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import cv2

from audit_recent_annotations import inspect_payload
from dataset_validation import check_split_leakage
from label_detection import export_yolo
from round2_review import check_frozen_pilot
from validate_yolo_dataset import validate_dataset

ROOT = Path(__file__).resolve().parent


def quality_flags(row, image_width, image_height):
    """Human-review flags, not rules to automatically rewrite ground truth."""
    flags = []
    boxes = row.get('boxes', [])
    for index, (x, y, width, height) in enumerate(boxes):
        if min(width, height) < 8:
            flags.append({'box_index': index, 'reason': 'tiny_plate_candidate_requires_human_inspection', 'box': boxes[index]})
        if not .8 <= width / max(height, 1e-9) <= 7:
            flags.append({'box_index': index, 'reason': 'unusual_aspect_ratio', 'box': boxes[index]})
        if not .001 <= width * height / (image_width * image_height) <= .4:
            flags.append({'box_index': index, 'reason': 'unusual_area_fraction', 'box': boxes[index]})
        if min(x, y, image_width-x-width, image_height-y-height) < .01 * min(image_width, image_height):
            flags.append({'box_index': index, 'reason': 'near_image_border', 'box': boxes[index]})
    return flags


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = ROOT / 'reports' / f'round2_gate_{stamp}'
    folder.mkdir(parents=True, exist_ok=False)
    database = ROOT / 'data/annotations.sqlite3'
    source = sqlite3.connect(f'file:{database.as_posix()}?mode=ro', uri=True)
    snapshot = sqlite3.connect(':memory:')
    source.backup(snapshot)
    source.close()
    latest = [json.loads(p[0]) for p in snapshot.execute(
        'SELECT r.payload FROM revisions r JOIN (SELECT image_id,MAX(version) v FROM revisions GROUP BY image_id)t '
        'ON r.image_id=t.image_id AND r.version=t.v')]
    reviewed = [r for r in latest if r.get('review_status') == 'reviewed']
    plan = json.loads((ROOT / 'reports/round2_plan/plan.json').read_text(encoding='utf-8'))
    candidate_ids = {r['sha256'] for r in plan['candidates']}
    pilot_ids = {r['image_id'] for r in plan['frozen_pilot']}
    failures, checks, flagged = [], [], []
    for row in reviewed:
        check = inspect_payload(row, 'reviewed_revision')
        if check.get('resolution'):
            width, height = check['resolution']
            check['quality_flags'] = quality_flags(row, width, height)
            if check['quality_flags']:
                flagged.append({'image_path': row['image_path'], 'split': row['split'],
                                'source_group': row['source_group'], 'revision': row['annotation_version'],
                                'flags': check['quality_flags']})
        if not check.get('detection_export_eligible'):
            failures.append({'image': row['image_path'], 'issues': check['issues']})
        image = cv2.imread(str(ROOT / row['image_path']))
        if image is not None:
            for i, box in enumerate(row['boxes']):
                x,y,w,h = map(int,box)
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,180,255),2)
                cv2.putText(image,f'{i}: {box}',(max(0,x-10),max(15,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
            overlay = folder / f"{row['image_id'][:12]}_{Path(row['image_path']).stem}.jpg"
            cv2.imwrite(str(overlay), image)
            check['overlay'] = overlay.name
        checks.append(check)
    integrity = {}
    for name, function in [('frozen_pilot',lambda:check_frozen_pilot(reviewed)),
                           ('split_leakage',lambda:check_split_leakage(reviewed,ROOT))]:
        try:
            function(); integrity[name] = {'status':'PASS'}
        except ValueError as exc:
            integrity[name] = {'status':'FAIL','reason':str(exc)}
    # Export only if BOTH numeric validation AND human-review quality gate pass.
    clean = not failures and not flagged and all(c['status']=='PASS' for c in integrity.values())
    export = {'status':'BLOCKED','reason':'Quality flags or integrity errors require human resolution. No samples removed.'}
    if clean:
        manifest = folder/'reviewed_input.json'
        manifest.write_text(json.dumps(reviewed,ensure_ascii=False,indent=2),encoding='utf-8')
        try:
            export_yolo(manifest,folder/'dataset')
            export=validate_dataset(folder/'dataset')
        except ValueError as exc:
            export={'status':'FAIL','reason':str(exc)}
    pilot_train_groups={r['source_group'] for r in plan['frozen_pilot'] if r['split']=='train'}
    current_train_groups={r['source_group'] for r in reviewed if r['split']=='train'}
    common_val=[r for r in reviewed if r['split']=='val' and r['source_group'] not in pilot_train_groups|current_train_groups]
    history_counts=dict(snapshot.execute('SELECT image_id,COUNT(*) FROM revisions GROUP BY image_id'))
    report={'reviewed':len(reviewed),'by_split':dict(Counter(r['split'] for r in reviewed)),
            'pilot_reviewed':sum(r['image_id'] in pilot_ids for r in reviewed),
            'round2_reviewed':sum(r['image_id'] in candidate_ids for r in reviewed),
            'round2_by_split':dict(Counter(r['split'] for r in reviewed if r['image_id'] in candidate_ids)),
            'independent_source_groups_by_split':{s:len({r['source_group'] for r in reviewed if r['split']==s}) for s in ('train','val','test')},
            'multi_revision_images':[{ 'image':r['image_path'],'revision_count':history_counts.get(r['image_id'],0),
                                      'latest_version':r['annotation_version']} for r in reviewed if history_counts.get(r['image_id'],0)>1],
            'integrity':integrity,'numeric_failures':failures,'quality_needs_review':flagged,
            'audit_quality_status':'PASS' if clean else 'BLOCKED','export':export,
            'training':'NOT_STARTED','common_validation_candidates':common_val,
            'common_validation_note':'No shared train groups; leakage check applies, but only 3 old val images and all remaining val used for checkpoint selection. Not independent final test.',
            'planned_threshold_analysis':[.01,.05,.10,.20,.40],
            'validation_warning':f"Current training images={sum(r['split']=='train' for r in reviewed)}; validation={sum(r['split']=='val' for r in reviewed)}. Split proportions are reported, not automatically changed.",
            'checks':checks,'annotations_modified':False,'test_or_threshold_modified':False}
    (folder/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    text=(f"# Round 2 training gate\n\nReviewed: {report['reviewed']}; splits {report['by_split']}.\n\n"
          f"Pilot {report['pilot_reviewed']}, round2 {report['round2_reviewed']}; round2 splits {report['round2_by_split']}.\n\n"
          f"Source groups {report['independent_source_groups_by_split']}.\n\nIntegrity: {integrity}.\n\n"
          f"Quality: {report['audit_quality_status']}; export: {export}.\n\n"
          f"{report['validation_warning']}\n\n## NEEDS_REVIEW\n"+
          '\n'.join(f"- {r['image_path']} ({r['split']}, revision {r['revision']}): {r['flags']}" for r in flagged)+
          '\n\nTraining not run, no round2 best.pt, no new loss/PR/mAP or checkpoint comparison. '
          'Frozen pilot/test and all labels untouched. Do not train from syntax-only PASS snapshot while quality flags remain.\n')
    (folder/'REPORT.md').write_text(text,encoding='utf-8')
    print(text); print('REPORT',folder)


if __name__=='__main__':main()