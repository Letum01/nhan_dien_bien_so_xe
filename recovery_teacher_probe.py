"""Train-only in-sample diagnostic; emits aggregate metrics, never labels."""
import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from adaptive_score_probe import rank
from benchmark_crop_padding import padded_box
from dual_yolo_client import Worker, CHECKPOINT, CHARACTER
from final_ocr_dataset import ROOT, check_protocol, read, sha
from glyph_review import CLASSES, normalized_text
from segmentation_benchmark import match_boxes, overlap_fraction
from svm_recovery_audit import image, crop
from train_ocr_v2 import hog, put
from yolo_character_dataset import live_rows
from yolo_character_pipeline import WEIGHTS, normalize_glyph, order_boxes


def main(out):
    check_protocol()
    if out.exists():raise FileExistsError(out)
    folder=ROOT/'experiments/hybrid_features_01'
    for path,digest in read(folder/'protocol.json')['sources'].items():
        if sha(Path(path))!=digest:raise ValueError('Changed hybrid input')
    expected=read(ROOT/'reports/adaptive_score_probe.json')['head_hashes']
    paths={k:folder/f'{k}.joblib' for k in ('svm_fusion','rf_fusion')}
    if any(sha(p)!=expected[k] for k,p in paths.items()):raise ValueError('Changed Fusion head')
    svm,rf=(joblib.load(p) for p in paths.values())
    for obj in (svm,rf):
        if obj['classes']!=list(CLASSES) or list(obj['classifier'].classes_)!=list(range(36)) or obj['classifier'].n_features_in_!=580:
            raise ValueError('Feature/class contract changed')
    cnn=tf.keras.models.load_model(WEIGHTS['cnn'],compile=False).layers[-1]
    probe=tf.keras.Model(cnn.inputs,[cnn.get_layer('dense').output,cnn.outputs[0]])
    rows=[r for r in live_rows() if r['split']=='train']
    n=Counter(); errors=Counter(); categories=Counter(); accepted_groups=set(); align_groups=set()
    with Worker() as worker:
        for r in rows:
            n['plates']+=1; n['ground_truth_glyphs']+=len(r['characters']); ref=normalized_text(r['plate_text'])
            im=image(ROOT/r['image_path']); p=worker.request('plate',im)
            if not p['rows']:categories['plate_missing']+=1;continue
            best=max(p['rows'],key=lambda v:v['score']); box=padded_box(best['bbox'],.15,im.shape);x,y=box[:2];roi=crop(im,box)
            d=order_boxes(worker.request('character',roi)['rows']); absboxes=[[x+a,y+b,w,h] for a,b,w,h in (v['bbox'] for v in d)]
            if not d:categories['no_glyph']+=1;continue
            xg,yg=r['plate_bbox'][:2]; truth=[[xg+a,yg+b,w,h] for a,b,w,h in (g['bbox'] for g in r['characters'])]
            match=match_boxes(absboxes,truth,.5); paired={m['pred']:m['gt'] for m in match['matches']}
            pix=np.stack([normalize_glyph(crop(roi,v['bbox'])) for v in d]); emb,ps=(np.asarray(a) for a in probe([pix[...,None]],training=False))
            f=np.concatenate([hog(pix),emb],axis=1);z=svm['scaler'].transform(f);sc=svm['classifier'];rc=rf['classifier']
            pred=[ps.argmax(1),sc.predict(z),rc.predict(f)]; ranks=[rank(s,t) for s,t in zip((ps,sc.decision_function(z),rc.predict_proba(f)),pred,strict=True)]
            overlaps=any(max(overlap_fraction(a,b),overlap_fraction(b,a))>.25 for i,a in enumerate(absboxes) for b in absboxes[:i]);
            clipped=any(min(a,b,roi.shape[1]-a-w,roi.shape[0]-b-h)<=1 for a,b,w,h in (v['bbox'] for v in d))
            geom=not overlaps and not clipped and best['score']>=.60 and min(v['score'] for v in d)>=.50 and np.all(pix.any(axis=(1,2)))
            text=''.join(CLASSES[i] for i in pred[0]); agree=all(np.array_equal(pred[0],v) for v in pred[1:]);
            if geom and 7<=len(d)<=10 and agree:
                n['accepted_plates']+=1; accepted_groups.add(r['source_group']); n['accepted_exact']+=int(text==ref)
                n['accepted_false_plate']+=int(text!=ref or not match['complete_order_correct'])
                for i,c in enumerate(text):
                    n['accepted_boxes']+=1; j=paired.get(i); good=j is not None and c==ref[j];n['accepted_correct']+=int(good)
                    if not good:errors[f'{ref[j] if j is not None else "UNMATCHED"}->{c}']+=1
            # Confirmed text enters this diagnostic only; GT boxes are not a gate input.
            if geom and len(d)==len(ref) and all(sum(CLASSES.index(c) in rr[i,:3] for rr in ranks)>=2 and
                   any(CLASSES.index(c)==pp[i] for pp in pred) for i,c in enumerate(ref)):
                n['alignment_plates']+=1;align_groups.add(r['source_group']);n['alignment_exact_geometry_order']+=int(match['complete_order_correct'])
                n['alignment_slots']+=len(d);n['alignment_correct_identity']+=sum(ref[i]==ref[j] for i,j in paired.items())
                n['alignment_correct_position']+=sum(i==j for i,j in paired.items())
        gpu=worker.info.get('gpu')
    put(out,{'scope':'IN_SAMPLE_ONLY; geometry overlap is a proxy; no stability/image perturbation measured',
             'counts':dict(n),'pseudo_errors_by_class':dict(errors),'categories':dict(categories),'accepted_group_count':len(accepted_groups),
             'alignment_group_count':len(align_groups),'gpu':gpu,'checkpoint_hashes':{str(p.relative_to(ROOT)):sha(p) for p in (CHECKPOINT,CHARACTER,WEIGHTS['cnn'],*paths.values())},
             'exported_labels':0,'student_fit':False})
    print(dict(n),dict(errors),len(accepted_groups),len(align_groups),gpu)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);main(ap.parse_args().output)