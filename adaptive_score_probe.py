"""Read-only top-k and end-to-end timing diagnostics, not a selector."""
from pathlib import Path
from time import perf_counter
import argparse
import json

import joblib
import numpy as np
from scipy.special import xlogy
import tensorflow as tf

from adaptive_oracle import CORE, ROOT, oracle_edits
from benchmark_crop_padding import padded_box
from dual_yolo_client import Worker
from ensemble_recognition import vote
from final_ocr_dataset import check_protocol, read, sha
from glyph_review import CLASSES
from multiclass_run import edit_distance
from svm_recovery_audit import crop, image
from train_ocr_v2 import hog
from yolo_character_pipeline import WEIGHTS, normalize_glyph, order_boxes


def rank(scores, predicted):
    """Keep estimator predict() first, including in a score tie."""
    order=np.argsort(-scores,axis=1,kind='stable')
    return np.stack([np.r_[p,row[row!=p]] for row,p in zip(order,predicted,strict=True)])


def main(output):
    check_protocol()
    folder=ROOT/'experiments/hybrid_features_01'; oracle=read(ROOT/'reports/adaptive_oracle.json')
    if oracle['gate']['pass']:raise ValueError('This probe is for a stopped experiment')
    protocol=read(folder/'protocol.json')
    for p,h in protocol['sources'].items():
        if sha(Path(p))!=h:raise ValueError(f'Hybrid input changed: {p}')
    meta=read(folder/'samples.json'); path=folder/'pixels.npy'
    if sha(path)!=meta['pixels_sha256']:raise ValueError('Cached pixels changed')
    pixels=np.load(path,allow_pickle=False); ids=[i for p in oracle['plates'] for i in meta['plates'][[r['id'] for r in meta['plates']].index(p['id'])]['ids']]
    net=tf.keras.models.load_model(WEIGHTS['cnn'],compile=False); inner=net.layers[-1]; final=inner.layers[-1]
    probe=tf.keras.Model(inner.inputs,[inner.get_layer('dense').output,final.input,inner.outputs[0]])
    heads={n:joblib.load(folder/f'{n}.joblib') for n in CORE[1:]}
    for h in heads.values():
        if h['classes']!=list(CLASSES) or list(h['classifier'].classes_)!=list(range(36)) or h['classifier'].n_features_in_!=580:
            raise ValueError('Fusion class mapping mismatch')

    def scores(x):
        embedding,presoftmax,probs=(np.asarray(z) for z in probe([x[...,None]],training=False))
        kernel,bias=final.get_weights(); logits=presoftmax@kernel+bias
        np.testing.assert_allclose(probs,tf.nn.softmax(logits).numpy(),atol=1e-5)
        f=np.concatenate([hog(x),embedding],axis=1)
        svm,rf=(heads[k]['classifier'] for k in CORE[1:]); scaled=heads['svm_fusion']['scaler'].transform(f)
        return {'cnn_v1':(probs,probs.argmax(1)), 'svm_fusion':(svm.decision_function(scaled),svm.predict(scaled)),
                'rf_fusion':(rf.predict_proba(f),rf.predict(f))}

    raw=scores(pixels[ids]); ranks={k:rank(s,p) for k,(s,p) in raw.items()}; truth=np.array([CLASSES.index(meta['samples'][i]['label']) if meta['samples'][i]['label'] else -1 for i in ids]); keep=truth>=0
    summary={}; top={}; voting=[]; start=0
    for k,(s,p) in raw.items():
        ordered=np.sort(s,axis=1); gap=ordered[:,-1]-ordered[:,-2]; summary[k]={'median_margin_raw':float(np.median(gap)),
            'median_entropy_raw':float(np.median(-xlogy(s,s).sum(1))) if k!='svm_fusion' else None}
        top[k]={str(n):int(np.any(ranks[k][keep,:n]==truth[keep,None],axis=1).sum()) for n in (1,2,3,5)}
    bounds={str(n):{'exact':0,'edits':0} for n in (1,2,3,5)}
    for p in oracle['plates']:
        size=len(p['predictions']['cnn_v1']); span=slice(start,start+size); ref=p['reference']; baseline=p['predictions']['cnn_v1']
        for k,(_,pred) in raw.items():
            if ''.join(CLASSES[i] for i in pred[span])!=p['predictions'][k]:raise ValueError(f'Cached prediction replay mismatch: {k}')
        result=vote(dict(zip(('cnn','svm','rf'),(p['predictions'][n] for n in CORE))))
        text=result.raw_text
        voting.append({'id':p['id'],'prediction':text,'exact':text==ref,'edits':edit_distance(ref,text),'changes':sum(a!=b for a,b in zip(text,baseline)),
                       'fixed':sum(text[j]==ref[i] and baseline[j]!=ref[i] for j,i in p['matches']),
                       'regressed':sum(text[j]!=ref[i] and baseline[j]==ref[i] for j,i in p['matches']),
                       'correct':sum(text[j]==ref[i] for j,i in p['matches']),'baseline_exact':baseline==ref,'uncertain':result.uncertain_positions})
        for n in (1,2,3,5):
            choices=[{CLASSES[i] for k in CORE for i in ranks[k][j,:n]} for j in range(start,start+size)]
            edits=oracle_edits(ref,choices); bounds[str(n)]['edits']+=edits; bounds[str(n)]['exact']+=int(edits==0)
        start+=size
    top['union']={str(n):int(np.logical_or.reduce([np.any(r[keep,:n]==truth[keep,None],axis=1) for r in ranks.values()]).sum()) for n in (1,2,3,5)}

    # Alternate order to reduce first/second-run bias. Includes workers/IPC,
    # crop, sort, normalization, OCR and voting; excludes disk decode/UI.
    cached=read(ROOT/'experiments/yolo_character_round1/comparison/results.json')
    dev={r['id']:r for r in cached['records'] if (r['scope'],r['method'])==('yolo','character_yolo')}
    frames=[image(ROOT/dev[p['id']]['image']) for p in oracle['plates']]; times={'cnn_only':[],'fusion_vote':[]}
    def replay(worker, im, mode):
        begin=perf_counter(); plate=worker.request('plate',im); data=plate['rows']; boxes=[]; text=''
        if data:
            box=padded_box(max(data,key=lambda r:r['score'])['bbox'],.15,im.shape); region=crop(im,box)
            boxes=[r['bbox'] for r in order_boxes(worker.request('character',region)['rows'])]
            if boxes:
                x=np.stack([normalize_glyph(crop(region,b)) for b in boxes]);
                if mode=='cnn_only':text=''.join(CLASSES[i] for i in np.asarray(net(x[...,None],training=False)).argmax(1))
                else:
                    z=scores(x); text=vote(dict(zip(('cnn','svm','rf'),(''.join(CLASSES[i] for i in z[k][1]) for k in CORE)))).raw_text
        return 1000*(perf_counter()-begin),text,plate['worker_info']['gpu']
    with Worker() as worker:
        for mode in times:replay(worker,frames[0],mode)
        for i,(p,im) in enumerate(zip(oracle['plates'],frames,strict=True)):
            for mode in (list(times) if i%2==0 else list(reversed(times))):
                ms,text,gpu=replay(worker,im,mode)
                if text!=(p['predictions']['cnn_v1'] if mode=='cnn_only' else voting[i]['prediction']):raise ValueError(f'Live replay mismatch at {p["id"]}')
                times[mode].append(ms)
    with output.open('x',encoding='utf-8') as f:
        json.dump({'top_k_coverage':top,'top_k_sequence_oracle':bounds,'raw_score_diagnostics':summary,'voting':voting,
                   'timing':{k:{'samples_ms':v,'mean_ms':float(np.mean(v)),'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95))} for k,v in times.items()},
                   'gpu':gpu,'calibration_fit':False,'head_hashes':{k:sha(folder/f'{k}.joblib') for k in CORE[1:]}},f,indent=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);main(parser.parse_args().output)