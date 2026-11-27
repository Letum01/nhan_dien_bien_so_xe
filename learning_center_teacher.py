"""Frozen local Dual-YOLO and Fusion teacher. No fit or label writes."""
from collections import Counter

import cv2
import joblib
import numpy as np

from benchmark_crop_padding import padded_box
from dataset import extract_hog_features
from dual_yolo_client import Worker, CHECKPOINT, CHARACTER
from final_ocr_dataset import ROOT, audit_candidates, check_protocol, read, sha, STORE
from learning_center import CLASSES, MODELS
from segmentation_benchmark import overlap_fraction
from yolo_character_dataset import ReadOnlyStore
from yolo_character_pipeline import WEIGHTS, normalize_glyph, order_boxes


def references():
    check_protocol()
    old=read(ROOT/'reports/dataset_recovery_feasibility01/inventory.json')
    if len(old)!=2932: raise ValueError('Inventory 2.932 ảnh không đầy đủ.')
    audit=audit_candidates(write=False)
    final=ReadOnlyStore(STORE).all_latest().values()
    def entry(r):
        return {'sha256':r['sha256'],'phash':r.get('phash'),
                'group':(r.get('metadata',{}).get('source_group') or r.get('source_group') or r.get('group') or ''), 'protected':True}
    return [entry(r) for r in (*old,*audit['known'],*audit['rows'])]+[
        entry(dict(sha256=r['source_sha256'],phash=r['source_phash'],source_group=r['source_group']))
        for r in final if r.get('source_sha256') and r.get('source_phash')]


class Teacher:
    def __init__(self):
        import tensorflow as tf
        check_protocol()
        folder=ROOT/'experiments/hybrid_features_01'
        hashes=read(ROOT/'reports/adaptive_score_probe.json')['head_hashes']
        self.heads={}; self.hashes={k:sha(p) for k,p in [('plate',CHECKPOINT),('character',CHARACTER),('cnn',WEIGHTS['cnn'])]}
        for name in ('svm','rf'):
            p=folder/f'{name}_fusion.joblib'; digest=sha(p)
            if digest!=hashes[f'{name}_fusion']:raise ValueError('Checkpoint Fusion đã đổi.')
            h=joblib.load(p)
            if h['classes']!=list(CLASSES) or list(h['classifier'].classes_)!=list(range(36)) or h['classifier'].n_features_in_!=580:
                raise ValueError('Mapping hoặc feature Fusion không hợp lệ.')
            self.heads[name]=h; self.hashes[name]=digest
        cnn=tf.keras.models.load_model(WEIGHTS['cnn'],compile=False).layers[-1]
        self.probe=tf.keras.Model(cnn.inputs,[cnn.get_layer('dense').output,cnn.outputs[0]])
        self.worker=Worker(); self.worker_info={}

    def run(self, image):
        p=self.worker.request('plate',image); detections=p['rows']; self.worker_info=p['worker_info']
        result={'plate_count':len(detections),'model_hashes':self.hashes,'policy':'lc-alignment-v1',
                'plate_threshold':.4,'character_threshold':.25,'glyphs':[], 'text':{k:'' for k in MODELS}}
        if not detections:return result
        plate=max(detections,key=lambda r:r['score']); x,y,w,h=padded_box(plate['bbox'],.15,image.shape)
        crop=image[y:y+h,x:x+w]; gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        rows=order_boxes(self.worker.request('character',crop)['rows']); boxes=[r['bbox'] for r in rows]
        flags=[]
        if any(min(a,b,w-a-c,h-b-d)<=1 for a,b,c,d in boxes):flags.append('edge')
        if any(max(overlap_fraction(a,b),overlap_fraction(b,a))>.25 for i,a in enumerate(boxes) for b in boxes[:i]):flags.append('overlap')
        ratio=0.
        if len(boxes)>1:
            ratio=float(max(np.diff(sorted(b+d/2 for a,b,c,d in boxes)))/np.median([b[3] for b in boxes]))
            if .40<=ratio<=.70:flags.append('ambiguous_rows')
        blur=float(cv2.Laplacian(gray,cv2.CV_64F).var()); contrast=float(gray.std())
        result.update(crop_box=[x,y,w,h],plate_score=plate['score'],geometry_flags=flags,layout=('two' if ratio>.55 else 'one') if len(boxes)>1 else 'unknown',
                      low_detector=plate['score']<.60 or any(r['score']<.50 for r in rows),poor_crop=blur<100 or contrast<20,blur=blur,contrast=contrast)
        if not rows:return result
        pixels=np.stack([normalize_glyph(crop[b:b+d,a:a+c]) for a,b,c,d in boxes]); emb,prob=map(np.asarray,self.probe([pixels[...,None]],training=False))
        f=np.concatenate([np.stack([extract_hog_features(i) for i in pixels]),emb],axis=1)
        s,r=(self.heads[k]['classifier'] for k in ('svm','rf')); z=self.heads['svm']['scaler'].transform(f)
        scores={'cnn':prob,'svm':s.decision_function(z),'rf':r.predict_proba(f)}
        preds={'cnn':prob.argmax(1),'svm':s.predict(z),'rf':r.predict(f)}
        support=Counter(g['label'] for row in read(ROOT/'datasets/yolo_character_round1/manifest.json')['rows'] if row['review']['split']=='train' for g in row['review']['characters'])
        for i, row in enumerate(rows):
            ranks={}
            for k in MODELS:
                winner=int(preds[k][i]); order=np.argsort(-scores[k][i],kind='stable'); ranks[k]=[CLASSES[j] for j in [winner,*[j for j in order if j!=winner][:2]]]
            result['glyphs'].append({'bbox':row['bbox'],'score':row['score'],'foreground':float((pixels[i]>0).mean()),'top_k':ranks})
        result.update(text={k:''.join(CLASSES[i] for i in v) for k,v in preds.items()},
                      low_cnn_margin=bool(np.any(np.diff(np.sort(prob,axis=1)[:,-2:],axis=1)<.2)),
                      rare_candidates=any(support[g['top_k'][k][0]]<=2 for g in result['glyphs'] for k in MODELS))
        result['disagreement']=len(set(result['text'].values()))>1
        return result