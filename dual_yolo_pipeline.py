"""Experimental optimized path, with the existing geometry/pixel contracts."""
from time import perf_counter

from benchmark_crop_padding import padded_box
from dataset import extract_hog_features
from ensemble_recognition import vote
from glyph_review import CLASSES
from yolo_character_pipeline import MixClassifiers, normalize_glyph, order_boxes, segment
import numpy as np


def run(image, worker, models, backend='cnn', method='character_yolo'):
    start=perf_counter()
    p=worker.request('plate',image)
    timings={'plate_wall':p['wall_seconds'],'plate_predict':p['worker_seconds']}
    keys=('svm','rf','cnn','ensemble') if backend=='all' else (backend,)
    result={'plate':None,'text':dict.fromkeys(keys,''),'glyph_boxes':[], 'glyph_scores':[],
            'worker':p['worker_info'],'cold':p['cold'],'timings':timings,'memory':memory(p)}
    if not p['rows']:
        timings['total']=perf_counter()-start
        return result
    t=perf_counter(); best=max(p['rows'],key=lambda r:r['score'])
    x,y,w,h=padded_box(best['bbox'],.15,image.shape); crop=image[y:y+h,x:x+w]
    result['plate']={**best,'crop_box':[x,y,w,h]}; timings['crop']=perf_counter()-t
    if method=='character_yolo':
        c=worker.request('character',crop); result['memory']=memory(c); timings.update(character_wall=c['wall_seconds'],character_predict=c['worker_seconds'])
        t=perf_counter(); rows=order_boxes(c['rows']); timings['ordering']=perf_counter()-t
        t=perf_counter(); glyphs=[(normalize_glyph(crop[b:b+d,a:a+e]),r['bbox']) for r in rows for a,b,e,d in [r['bbox']]]
        timings['normalization']=perf_counter()-t; result['glyph_scores']=[r['score'] for r in rows]
    else:
        t=perf_counter(); glyphs=segment(crop,method); timings['segmentation']=perf_counter()-t
    result['glyph_boxes']=[list(b) for _,b in glyphs]
    if glyphs:
        needed=('svm','rf','cnn') if backend in ('all','ensemble') else (backend,)
        x=np.asarray([g for g,_ in glyphs]); texts={}
        if 'svm' in needed or 'rf' in needed:
            t=perf_counter(); hog=np.stack([extract_hog_features(g) for g in x]); timings['hog']=perf_counter()-t
        for name in needed:
            t=perf_counter()
            if name=='svm': ids=models.svm.predict(models.scaler.transform(hog))
            elif name=='rf': ids=models.rf.predict(hog)
            else: ids=np.asarray(models.cnn(x[...,None],training=False)).argmax(axis=1)
            texts[name]=''.join(CLASSES[int(i)] for i in ids); timings[name]=perf_counter()-t
        if backend in ('all','ensemble'):
            t=perf_counter(); texts['ensemble']=vote(texts).raw_text; timings['ensemble']=perf_counter()-t
        result['text']={k:texts[k] for k in keys}
    timings['total']=perf_counter()-start
    return result


def memory(response):
    return {k:response[k] for k in ('allocated','reserved','peak_allocated','peak_reserved')}