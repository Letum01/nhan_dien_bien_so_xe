"""Build blind review queues or approved, nested train-only sampling manifests."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from external_task_curation import read, digest, write
from external_dataset_prepare import protected
from external_task_train import preflight, evidence
from final_ocr_dataset import ROOT, check_protocol

LEVELS=(1000,5000,10000,15000)
LEAKAGE=ROOT/'reports/external_preparation_20261127T080626995900Z/leakage.json'


def review(curation, output):
    glyphs=read(curation/'glyphs.json')
    parents={r['parent']:r for r in read(curation/'parents.json')}
    visual=read(curation/'visual_review.json')
    index=read(curation/'task_review_index.json')
    strata=defaultdict(list); scenes=defaultdict(list)
    for g in glyphs:
        if g['tier']=='D': continue
        strata['class:'+g['character']].append(g)
        for flag in ('small_pixel_roi','parent_edge_risk','box_overlap','blur_proxy'):
            if flag in g['glyph_flags']: strata[flag].append(g)
    for p,r in parents.items():
        if r['tier']=='D': continue
        scenes['no_parent_flag' if r['detector_review_candidate'] else 'risk_pool'].append(p)
        for flag in r['quality_flags']:
            scenes[flag].append(p)
    for observation in visual['observations']:
        label=observation.get('domain_candidate') or observation['issue_candidate']
        paths={index[i-1] for i in observation['tiles']}
        scenes[label].extend(sorted(p for p in paths if parents[p]['tier']!='D'))
        for g in glyphs:
            if g['tier']!='D' and g['parent'] in paths: strata[label].append(g)
    rng=random.Random(42); glyph_queue={}; scene_queue={}; sizes={}
    for label,rows in sorted(strata.items()):
        sizes[label]=len(rows)
        for g in rng.sample(rows,min(10 if label.startswith('class:') else 16,len(rows))):
            key=(g['parent'],g['glyph_index'])
            row=glyph_queue.setdefault(key,{k:g[k] for k in ('parent','glyph_index','character','pixel_xywh','sha256')})
            row.setdefault('strata',[]).append(label)
    for label,rows in sorted(scenes.items()):
        rows=sorted(set(rows))
        for p in rng.sample(rows,min(24,len(rows))):
            scene_queue.setdefault(p,[]).append(label)
    output.mkdir(parents=True,exist_ok=False)
    for name,rows in [('glyph_review',list(glyph_queue.values())),('detector_review',[
        dict(parent=p,sha256=parents[p]['sha256'],strata=s) for p,s in scene_queue.items()])]:
        write(output/f'{name}_inputs.json',rows)
        with (output/f'{name}_responses.csv').open('x',encoding='utf-8',newline='') as stream:
            fields=['review_id','decision','reason','reviewer']
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
            for i in range(len(rows)): writer.writerow({'review_id':i+1})
    write(output/'receipt.json',dict(seed=42,glyphs=len(glyph_queue),detector_images=len(scene_queue),
         glyph_pool_sizes=sizes,detector_pool_sizes={k:len(set(v)) for k,v in scenes.items()},
         glyph_selected_by_stratum=dict(Counter(s for r in glyph_queue.values() for s in r['strata'])),
         detector_selected_by_stratum=dict(Counter(s for labels in scene_queue.values() for s in labels)),
         glyph_manifest_sha256=digest(curation/'glyphs.json'),parent_manifest_sha256=digest(curation/'parents.json'),
         inference_used=False,review_results=0,error_rate=None))


def sampling_units(rows):
    """Keep linked verified groups, parents and duplicate components intact."""
    train=[r for r in rows if r['split']=='train']
    members=defaultdict(list); links={}; parent={}
    def root(g):
        while parent[g]!=g:
            g=parent[g]
        return g
    for r in train:
        g=r['source_group']; parent.setdefault(g,g)
        for kind in ('parent','sha256','possible_duplicate_component'):
            key=(kind,r[kind]); a,b=root(g),root(links.setdefault(key,g))
            if a!=b: parent[max(a,b)]=min(a,b)
    for r in train: members[root(r['source_group'])].append(r)
    return members


def nested(rows, levels=LEVELS):
    """Order intact groups with a fixed tempered class-distribution objective."""
    members=sampling_units(rows)
    train=[r for unit in members.values() for r in unit]
    profiles={k:Counter(r['character'] for r in a) for k,a in members.items()}
    totals=Counter(r['character'] for r in train)
    denom=sum(v**.5 for v in totals.values()); target={c:v**.5/denom for c,v in totals.items()}
    counts=Counter(); selected=[]; remaining=set(members); result={}
    ties={k:hashlib.sha256(f'42:{k}'.encode()).hexdigest() for k in members}
    for level in sorted(set(levels)):
        if len(train)<level: continue
        while len(selected)<level:
            def cost(k):
                n=len(members[k]); total=len(selected)+n
                # Squared distance to a tempered distribution (rare classes retain representation).
                error=sum(((counts[c]+profiles[k][c])/total-w)**2 for c,w in target.items())
                return error,ties[k]
            key=min(remaining,key=cost); remaining.remove(key)
            selected.extend(sorted(members[key],key=lambda r:(r['parent'],r['glyph_index'])))
            counts.update(profiles[key])
        result[level]=list(selected)
    return result


def approved_rows(curation, approval):
    check_protocol()
    rows, frozen=preflight(curation,approval)
    record=read(approval)
    if not evidence(approval.parent,record.get('progressive_protocol_approval')):
        raise ValueError('BLOCKED: supplemental benchmark approval is missing or changed')
    supplemental=read(approval.parent/record['progressive_protocol_approval']['path'])
    protocol=ROOT/'configs/alpr_progressive_protocol_v1.json'
    if supplemental.get('protocol_sha256')!=digest(protocol) or supplemental.get('frozen_manifest_sha256')!=frozen or supplemental.get('decision')!='approved':
        raise ValueError('BLOCKED: approval does not cover this protocol and cohort')
    expected=read(curation/'summary.json')['source_artifact_sha256']['leakage.json']
    if digest(LEAKAGE)!=expected or protected()[1]!=read(LEAKAGE)['protected_references']:
        raise ValueError('BLOCKED: protected references changed; refresh the leakage evidence and approval')
    return rows,frozen,protocol


def approved(curation, approval, output):
    rows,frozen,protocol=approved_rows(curation,approval)
    snapshots=nested(rows)
    if not snapshots: raise ValueError('BLOCKED: fewer than 1,000 approved external train glyphs')
    output.mkdir(parents=True,exist_ok=False)
    development=[r for r in rows if r['split']=='development']
    info={}
    for level,samples in snapshots.items():
        path=output/f'train_{level}.json'; write(path,samples)
        info[level]=dict(actual=len(samples),classes=dict(Counter(r['character'] for r in samples)),
                         groups=len({r['source_group'] for r in samples}),sha256=digest(path))
    write(output/'development.json',development)
    write(output/'receipt.json',dict(levels=info,frozen_sha256=frozen,development_sha256=digest(output/'development.json'),
                                    protocol_sha256=digest(protocol),seed=42,test_exported=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('review','snapshots')); ap.add_argument('curation',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--approval',type=Path); args=ap.parse_args()
    try:
        if args.mode=='review': review(args.curation,args.output)
        else: approved(args.curation,args.approval,args.output)
    except (ValueError,KeyError,OSError) as e:
        ap.exit(2,f'{e}\n')


if __name__=='__main__': main()