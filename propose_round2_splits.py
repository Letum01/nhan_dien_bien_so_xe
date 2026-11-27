"""Read-only A/B allocation proposals; never writes annotation revisions."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sqlite3

from dataset_validation import check_split_leakage
from round2_review import check_frozen_pilot

ROOT=Path(__file__).resolve().parent


def propose(rows, frozen, seed=42):
    """Freeze all pilot labels; move round2 source groups only in a copied plan.

    Merge source groups linked by exact/pHash<=6 so proposal cannot silently
    separate near-duplicate frames. Group identifiers are NOT renamed.
    """
    from dataset_audit import fingerprint
    pilot_ids={r['image_id'] for r in frozen}
    frozen_groups={r['source_group']:r['split'] for r in frozen}
    groups=defaultdict(list)
    for row in rows:groups[row['source_group']].append(row)
    parent={g:g for g in groups}
    def find(g):
        while parent[g]!=g:
            parent[g]=parent[parent[g]];g=parent[g]
        return g
    def union(a,b):parent[find(b)]=find(a)
    hashes=[]
    for row in rows:
        digest,phash,_=fingerprint(ROOT/row['image_path'])
        for other,other_hash,other_phash in hashes:
            if digest==other_hash or (phash is not None and other_phash is not None and (phash^other_phash).bit_count()<=6):
                union(row['source_group'],other['source_group'])
        hashes.append((row,digest,phash))
    clusters=defaultdict(list)
    for group in groups:clusters[find(group)].append(group)
    fixed={};free=[]
    for key,members in clusters.items():
        frozen_splits={frozen_groups[g] for g in members if g in frozen_groups}
        if len(frozen_splits)>1:raise ValueError('Duplicate cluster crosses frozen pilot splits')
        if frozen_splits:fixed[key]=next(iter(frozen_splits))
        else:free.append(key)
    random.Random(seed).shuffle(free)
    # Target ~80/20 among train+val images; pilot test is never used for choice.
    total=sum(r['split']!='test' for r in rows)
    val_target=max(sum(r['split']=='val' for r in frozen),round(total*.2))
    val_count=sum(len(groups[g]) for key,members in clusters.items() if fixed.get(key)=='val' for g in members)
    assignments=dict(fixed)
    # Predeclared seeded order, no model score/image test performance used.
    for key in free:
        size=sum(len(groups[g]) for g in clusters[key])
        choose_val=abs(val_count+size-val_target)<abs(val_count-val_target)
        assignments[key]='val' if choose_val else 'train'
        if choose_val:val_count+=size
    proposed=[]
    for row in rows:
        split=row['split'] if row['image_id'] in pilot_ids else assignments[find(row['source_group'])]
        proposed.append({**row,'split':split})
    return proposed


def main():
    uri=f'file:{(ROOT/"data/annotations.sqlite3").as_posix()}?mode=ro'
    with sqlite3.connect(uri,uri=True) as connection:
        all_rows=[json.loads(r[0]) for r in connection.execute('SELECT r.payload FROM revisions r JOIN (SELECT image_id,MAX(version) v FROM revisions GROUP BY image_id)t ON r.image_id=t.image_id AND r.version=t.v')]
        history=[json.loads(r[0]) for r in connection.execute('SELECT payload FROM revisions ORDER BY id')]
    rows=[r for r in all_rows if r.get('review_status')=='reviewed']
    plan=json.loads((ROOT/'reports/round2_plan/plan.json').read_text(encoding='utf-8'))
    frozen=plan['frozen_pilot'];pilot_ids={r['image_id'] for r in frozen}
    check_frozen_pilot(rows);check_split_leakage(rows,ROOT)
    b=propose(rows,frozen);check_split_leakage(b,ROOT);check_frozen_pilot(b)
    changes=[{'image_path':old['image_path'],'image_id':old['image_id'],'source_group':old['source_group'],
              'from':old['split'],'proposed_to':new['split'],'base_revision':old['annotation_version']}
             for old,new in zip(rows,b) if old['split']!=new['split']]
    split_changes=[];last={}
    for r in history:
        old=last.get(r['image_id'])
        if old and (old['split']!=r['split'] or old['source_group']!=r['source_group']):
            split_changes.append({'image':r['image_path'],'version':r['annotation_version'],
                                  'from_group_split':[old['source_group'],old['split']],
                                  'to_group_split':[r['source_group'],r['split']]})
        last[r['image_id']]=r
    train=sum(r['split']=='train' for r in rows);val=sum(r['split']=='val' for r in rows)
    # A target80 train minimum; retain all current val and test.
    report={'status':'PROPOSAL_ONLY_REQUIRES_USER_CONFIRMATION','reviewed':len(rows),
            'current':dict(Counter(r['split'] for r in rows)),
            'group_counts':{s:len({r['source_group'] for r in rows if r['split']==s}) for s in ('train','val','test')},
            'observed_revision_group_split_changes':split_changes,
            'cause_evidence':'No automatic allocation in UI. Per-image split keys, UNASSIGNED default. Revisions record values, not user click intent; cannot prove deliberate vs mistaken selection.',
            'option_A':{'policy':'No existing split changes; add verified new independent training groups.',
                        'additional_train_for_minimum80':max(0,80-train),'then_counts':{'train':max(80,train),'val':val,'test':sum(r['split']=='test' for r in rows)},
                        'recommended_new_train':80,'recommended_counts':{'train':train+80,'val':val,'test':sum(r['split']=='test' for r in rows)}},
            'option_B':{'policy':'Seed42 group+duplicate clusters, ~80/20 train/val; ALL21 pilot frozen. No test metrics used.',
                        'counts':dict(Counter(r['split'] for r in b)),
                        'group_counts':{s:len({r['source_group'] for r in b if r['split']==s}) for s in ('train','val','test')},
                        'changes':changes,'leakage_check':'PASS source/exact/pHash','frozen_pilot':'PASS',
                        'approval_required':True,'applied':False,
                        'risk':'Moved former validation examples become training; cannot claim independence versus earlier validation analysis. Freeze a new validation plan before model selection; keep original test untouched. New revisions must preserve old history.'},
            'training_readiness':'BLOCKED_by_small_train17_and_pending_user_split_decision_not_label_quality',
            'annotations_modified':False}
    folder=ROOT/'reports'/('split_options_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));folder.mkdir()
    (folder/'options.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'option_B_proposal_rows_NOT_APPLIED.json').write_text(json.dumps(b,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'REPORT.md').write_text(f"# Split options — NOT APPLIED\n\nCurrent {report['current']}, groups {report['group_counts']}.\n\n0408 reviewed v3 corrected; audit and validator PASS.\n\nA: retain all splits, add {max(0,80-train)} train for80/{val}/3; recommended +80 train gives{train+80}/{val}/3.\n\nB: {report['option_B']['counts']}, groups {report['option_B']['group_counts']}; {len(changes)} proposed moves. Group/hash/pHash and frozen pilot PASS. User approval required; SQLite not modified.\n\n{report['cause_evidence']}\n\nRisk: {report['option_B']['risk']}\n\nNo100epoch run: train17 still insufficient expansion.\n",encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('current','group_counts','option_A')},indent=2))
    print('B',report['option_B']['counts'],'changes',len(changes));print('REPORT',folder)


if __name__=='__main__':main()