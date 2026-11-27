"""Append-only annotation plan; freeze pilot split membership and test identities."""
import csv
import hashlib
import json
from pathlib import Path

from prepare_annotation_round import build_queue

ROOT=Path(__file__).resolve().parent
PILOT=ROOT/'reports/annotation_review_20261117T152948733793Z/yolo_trial/manifest.json'


def prepare_plan(audit, frozen_rows, candidates=120):
    frozen_hashes={r['image_id'] for r in frozen_rows}
    frozen_paths={str(Path(r['image_path'])) for r in frozen_rows}
    forbidden=set(frozen_paths)
    # Conservative transitive duplicate exclusion against ALL pilot images,
    # not just test, to avoid treating another frame as a new independent source.
    links=[(p['a'],p['b']) for p in audit['near_duplicate_candidates']]
    for group in audit['exact_duplicate_groups']:
        links.extend((group[0],p) for p in group[1:])
    changed=True
    while changed:
        changed=False
        for a,b in links:
            a,b=str(Path(a)),str(Path(b))
            if a in forbidden or b in forbidden:
                old=len(forbidden);forbidden.update((a,b));changed |= len(forbidden)>old
    remaining=[r for r in audit['images'] if str(Path(r['image_path'])) not in forbidden and r['sha256'] not in frozen_hashes]
    copy={**audit,'images':remaining}
    queue=build_queue(copy,count=candidates)
    for row in queue:
        row['split']='UNASSIGNED'
        row['split_policy']='human_confirm_source_then_train_or_val_only; never move frozen test'
        row['priority_reason']='source_resolution_diversity_proxy_not_verified_vehicle_lighting_label'
    return {'status':'BLOCKED_pending_human_review','frozen_pilot':frozen_rows,
            'frozen_test_images':[r['image_path'] for r in frozen_rows if r['split']=='test'],
            'frozen_source_group_splits':{r['source_group']:r['split'] for r in frozen_rows},
            'candidates':queue,'excluded_pilot_related_paths':sorted(forbidden),
            'target_total_if_all_reviewed':len(frozen_rows)+len(queue),
            'policy':{'seed':42,'new_independent_groups':'Suggest ~85% train /15% val only AFTER human source verification. No automatic assignment.',
                      'existing_test':'Immutable pilot test. Do not tune thresholds/config against it.',
                      'metadata':'Vehicle, lighting, skew, blur UNKNOWN unless confirmed by human.',
                      'training_gate':'Need expanded reviewed labels, resolved label quality flags, source/hash leakage check and fresh exported validator PASS.'}}


def main():
    audit=json.loads((ROOT/'reports/dataset_audit.json').read_text(encoding='utf-8'))
    frozen=json.loads(PILOT.read_text(encoding='utf-8'))['rows']
    plan=prepare_plan(audit,frozen)
    destination=ROOT/'reports/round2_plan';destination.mkdir(exist_ok=False)
    plan['pilot_manifest_sha256']=hashlib.sha256(PILOT.read_bytes()).hexdigest()
    (destination/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    with (destination/'candidates.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(plan['candidates'][0]));writer.writeheader();writer.writerows(plan['candidates'])
    print('Candidates',len(plan['candidates']),'target',plan['target_total_if_all_reviewed'],'frozen test',plan['frozen_test_images'])


if __name__=='__main__':main()