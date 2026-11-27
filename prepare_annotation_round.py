"""Deterministic unlabeled priority queue and conservative training readiness."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from annotation_store import AnnotationStore

ROOT=Path(__file__).resolve().parent


def build_queue(audit, count=150):
    images=[r for r in audit['images'] if r['readable']]
    parents={r['image_path']:r['image_path'] for r in images}
    def find(x):
        while parents[x]!=x:
            parents[x]=parents[parents[x]];x=parents[x]
        return x
    def union(a,b):
        if a in parents and b in parents:parents[find(b)]=find(a)
    for group in audit['exact_duplicate_groups']:
        for other in group[1:]:union(group[0],other)
    for pair in audit['near_duplicate_candidates']:union(pair['a'],pair['b'])
    groups=defaultdict(list)
    for row in images:groups[find(row['image_path'])].append(row)
    failures={'0405.jpg','0407.jpg'}
    candidates=[]
    for members in groups.values():
        # Prefer known detector failures, then deterministic source representative.
        row=min(members,key=lambda r:(Path(r['image_path']).name not in failures,r['image_path']))
        candidates.append({**row,'duplicate_cluster':min(r['image_path'] for r in members),
                           'cluster_size':len(members),'priority_reason':'known_detector_failure' if Path(row['image_path']).name in failures else 'source_resolution_diversity'})
    buckets=defaultdict(list)
    for row in sorted(candidates,key=lambda r:(r['priority_reason']!='known_detector_failure',r['image_path'])):
        buckets[(str(Path(row['image_path']).parent),row['width'],row['height'])].append(row)
    chosen=[r for r in candidates if r['priority_reason']=='known_detector_failure']
    used={r['image_path'] for r in chosen}
    while len(chosen)<count and any(buckets.values()):
        for bucket in sorted(buckets):
            if buckets[bucket]:
                row=buckets[bucket].pop(0)
                if row['image_path'] not in used:
                    chosen.append(row);used.add(row['image_path'])
                if len(chosen)>=count:break
    return [{'rank':i+1,**{key:row[key] for key in ('image_path','sha256','width','height','duplicate_cluster','cluster_size','priority_reason')},
             'review_status':'unlabeled','split':'UNASSIGNED','source_group':'NEEDS_HUMAN_CONFIRMATION'} for i,row in enumerate(chosen)]


def readiness(store, queue):
    latest=store.all_latest()
    reviewed=[r for r in latest.values() if r.get('review_status')=='reviewed']
    splits=Counter(r.get('split') for r in reviewed)
    errors=[];valid=0
    from label_detection import validate_annotation
    for row in reviewed:
        try:validate_annotation({**row,'human_reviewed':True});valid+=1
        except ValueError as exc:errors.append({'image':row['image_path'],'error':str(exc)})
    required={split:max(0,1-splits[split]) for split in ('train','val','test')}
    leakage_error=None
    if reviewed and not errors:
        from dataset_validation import check_split_leakage
        try:check_split_leakage(reviewed,ROOT)
        except ValueError as exc:leakage_error=str(exc)
    export_ready=bool(reviewed) and not errors and not leakage_error and not any(required.values())
    return {'status':'EXPORT_READY' if export_ready else 'BLOCKED',
            'training_status':'NOT_AUTHORIZED_requires_export_validator_and_training_data_quality_review',
            'queue_images':len(queue),'reviewed_images':len(reviewed),'individually_valid_box_images':valid,
            'needs_review':sum(r.get('review_status')=='needs_review' for r in latest.values()),
            'reviewed_by_split':dict(splits),'minimum_missing_images_by_split':required,
            'errors':errors,'leakage_error':leakage_error,
            'warning':'One sample/split is structural minimum ONLY, not sufficient statistical training data. Group/hash export validator and GPU checks required. No accuracy claimed.'}


def main():
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    audit=json.loads((reports/'dataset_audit.json').read_text(encoding='utf-8'))
    rows=build_queue(audit)
    (reports/'annotation_round_01.json').write_text(json.dumps({'selection_is_ground_truth':False,'images':rows},indent=2),encoding='utf-8')
    with (reports/'annotation_round_01.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=readiness(AnnotationStore(ROOT/'data/annotations.sqlite3'),rows)
    (reports/'training_readiness.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()