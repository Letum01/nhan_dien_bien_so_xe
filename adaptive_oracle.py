"""Upper bounds, not deployable predictions. No model fitting or test access."""
import argparse
import itertools
import json
from pathlib import Path

MODELS=('cnn_v1','cnn_v2','cnn_v3','svm_v2','rf_v2','svm_fusion','rf_fusion')
CORE=('cnn_v1','svm_fusion','rf_fusion')
ROOT=Path(__file__).resolve().parent


def oracle_edits(reference, candidates):
    row=list(range(len(reference)+1))
    for i, options in enumerate(candidates,1):
        new=[i]
        for j, target in enumerate(reference,1):
            new.append(min(row[j]+1,new[-1]+1,row[j-1]+(target not in options)))
        row=new
    return row[-1]


def audit():
    folder=ROOT/'experiments/hybrid_features_01'
    def read(p):return json.loads(p.read_text(encoding='utf-8'))
    source=read(folder/'samples.json'); results=read(folder/'results.json')
    reference=read(ROOT/'experiments/yolo_character_round1/comparison/results.json')
    geometry={r['id']:r for r in reference['records'] if r['scope']=='yolo' and r['method']=='character_yolo'}
    by_model={k:{r['id']:r for r in results[k]['plates']} for k in MODELS}
    plates=[]; aligned=[]
    for p in source['plates']:
        pid=p['id']; ids=p['ids']; ref=p['reference']; geo=geometry[pid]
        preds={k:by_model[k][pid]['prediction'] for k in MODELS}
        if any(by_model[k][pid]['ids']!=ids or by_model[k][pid]['reference']!=ref or len(preds[k])!=len(ids) for k in MODELS):
            raise ValueError('Prediction alignment mismatch')
        pairs=geo['geometry']['matches']; order=[(m['pred'],m['gt']) for m in pairs]
        for j,k in order:
            if source['samples'][ids[j]]['label']!=ref[k]:raise ValueError('GT alignment mismatch')
            aligned.append((ref[k],{n:v[j] for n,v in preds.items()}))
        plates.append(dict(id=pid,reference=ref,predictions=preds,matches=order,geometry={k:geo['geometry'][k] for k in ('tp','fp','fn','complete_order_correct')}))
    if len(plates)!=12 or len(aligned)!=98:raise ValueError('Unexpected development cohort')
    matrix={}
    for a,b in itertools.combinations(MODELS,2):
        def overlaps(rows):
            n=len(rows); both=sum(p[a]==t and p[b]==t for t,p in rows); neither=sum(p[a]!=t and p[b]!=t for t,p in rows)
            wa=sum(p[a]!=t for t,p in rows);wb=sum(p[b]!=t for t,p in rows)
            return dict(n=n,disagree=sum(p[a]!=p[b] for _,p in rows),both_correct=both,both_wrong=neither,double_fault_rate=neither/n,
                        only_a_correct=wb-neither,only_b_correct=wa-neither,error_jaccard=neither/(wa+wb-neither) if wa+wb else None)
        matrix[f'{a}/{b}']=dict(glyph=overlaps(aligned),plate=overlaps([(p['reference'],p['predictions']) for p in plates]))
    scopes={}
    for name, models in [('cnn_svm',CORE[:2]),('core',CORE),('all',MODELS)]:
        entries=[]
        for p in plates:
            ref=p['reference']; preds=p['predictions']; opts=[set(v) for v in zip(*(preds[k] for k in models),strict=True)]
            # Upper bound allows picking any available character per existing box.
            # A second bound respects fixed IoU-based assignments.
            edits=oracle_edits(ref,opts)
            strict=p['geometry']['complete_order_correct'] and all(ref[k] in opts[j] for j,k in p['matches'])
            entries.append(dict(id=p['id'],whole_edits=min(oracle_edits(ref,[{c} for c in preds[k]]) for k in models),
                                slot_edits=edits,strict_exact=strict,cnn_was_exact=preds['cnn_v1']==ref))
        scopes[name]=dict(models=models,matched_correct=sum(any(p[k]==t for k in models) for t,p in aligned),
                          whole_exact=sum(e['whole_edits']==0 for e in entries),whole_edits=sum(e['whole_edits'] for e in entries),
                          slot_exact=sum(e['slot_edits']==0 for e in entries),slot_edits=sum(e['slot_edits'] for e in entries),
                          strict_exact=sum(e['strict_exact'] for e in entries),fixed=sum(e['slot_edits']==0 and not e['cnn_was_exact'] for e in entries),plates=entries)
    return dict(models=MODELS,plates=plates,pairwise=matrix,oracles=scopes,reference_glyphs=sum(len(p['reference']) for p in plates),
                gate={'minimum_core_fixed':2,'pass':scopes['core']['fixed']>=2})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);args=ap.parse_args()
    data=audit()
    with args.output.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2)
    print({k:{a:b for a,b in v.items() if a!='plates'} for k,v in data['oracles'].items()},data['gate'])