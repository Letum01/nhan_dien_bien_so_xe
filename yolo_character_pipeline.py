"""Opt-in ALPR experiment; no annotations consumed by inference."""
import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np

from glyph_review import CLASSES, ROOT
from yolo_ui_bridge import CHECKPOINT, PYTHON, detect

MIX1=ROOT/'experiments/ocr_mix_experiment01_20261125T072726619780Z'
MIX2=ROOT/'experiments/ocr_mix_experiment02_20261125T080339560263Z'
WEIGHTS={'svm':MIX1/'svm_mix/svm_model.joblib','scaler':MIX1/'svm_mix/hog_scaler.joblib',
         'rf':MIX1/'rf_mix/rf_model.joblib','cnn':MIX2/'cnn_mix/cnn_model.keras'}


def order_boxes(rows):
    """Use only geometry, splitting on a strong gap between box centers."""
    center=lambda row: row['bbox'][1]+row['bbox'][3]/2
    left=lambda row: (row['bbox'][0],row['bbox'][1])
    vertical=sorted(rows,key=lambda r:(center(r),left(r)))
    if len(vertical)>1:
        gaps=np.diff([center(r) for r in vertical]); split=int(gaps.argmax())+1
        if gaps.max()>.55*np.median([r['bbox'][3] for r in vertical]):
            return sorted(vertical[:split],key=left)+sorted(vertical[split:],key=left)
    return sorted(vertical,key=left)


def normalize_glyph(bgr):
    """Match reviewed-export geometry; infer polarity only from ROI border."""
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    edge=np.concatenate((binary[0],binary[-1],binary[1:-1,0],binary[1:-1,-1]))
    if np.mean(edge>0)>.5:
        binary=cv2.bitwise_not(binary)
    h,w=binary.shape; side=max(h,w); padded=np.zeros((side,side),np.uint8)
    y,x=(side-h)//2,(side-w)//2
    padded[y:y+h,x:x+w]=binary
    return cv2.threshold(cv2.resize(padded,(32,32),interpolation=cv2.INTER_AREA),127,255,cv2.THRESH_BINARY)[1]


class MixClassifiers:
    def __init__(self):
        import joblib
        import tensorflow as tf
        from dataset import CHARACTER_CLASSES
        if tuple(CHARACTER_CLASSES)!=tuple(CLASSES):
            raise ValueError('HOG and OCR class order differ')
        self.svm,self.scaler,self.rf=(joblib.load(WEIGHTS[k]) for k in ('svm','scaler','rf'))
        if any(m.n_features_in_!=324 for m in (self.svm,self.scaler,self.rf)) or any(list(m.classes_)!=list(range(36)) for m in (self.svm,self.rf)):
            raise ValueError('HOG/scaler/class mapping mismatch')
        contract=json.loads((WEIGHTS['cnn'].parent/'cnn_contract.json').read_text(encoding='utf-8'))
        self.cnn=tf.keras.models.load_model(WEIGHTS['cnn'],compile=False)
        def rescalers(model):
            return [layer for child in model.layers for layer in
                    ([child] if isinstance(child,tf.keras.layers.Rescaling) else rescalers(child) if hasattr(child,'layers') else [])]
        scales=rescalers(self.cnn)
        if (self.cnn.input_shape!=(None,32,32,1) or self.cnn.output_shape!=(None,36) or
            contract['class_order']!=list(CLASSES) or len(scales)!=1 or not np.isclose(scales[0].scale,1/255) or scales[0].offset!=0):
            raise ValueError('CNN input/output/rescaling mismatch')

    def predict(self, glyphs):
        from dataset import extract_hog_features
        from ensemble_recognition import vote
        if not glyphs:
            return dict.fromkeys(('svm','rf','cnn','ensemble'),'')
        x=np.asarray(glyphs)
        if x.shape[1:]!=(32,32) or x.dtype!=np.uint8:
            raise ValueError('Expected 32x32 uint8 glyphs')
        hog=np.stack([extract_hog_features(g) for g in x])
        ids={'svm':self.svm.predict(self.scaler.transform(hog)), 'rf':self.rf.predict(hog),
             'cnn':np.asarray(self.cnn(x[...,None],training=False)).argmax(axis=1)}
        text={k:''.join(CLASSES[int(i)] for i in indices) for k,indices in ids.items()}
        return {**text,'ensemble':vote(text).raw_text}


def character_boxes(crop, weights):
    weights=Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError('Trained character checkpoint required')
    with TemporaryDirectory(prefix='alpr_character_') as folder:
        src=Path(folder)/'crop.png'; dst=Path(folder)/'boxes.json'
        if not cv2.imwrite(str(src),crop):
            raise OSError('Cannot write character worker input')
        subprocess.run([str(PYTHON),str(ROOT/'yolo_character_worker.py'),str(src),str(dst),str(weights)],
                       check=True,timeout=90,cwd=ROOT)
        return order_boxes(json.loads(dst.read_text(encoding='utf-8')))


def segment(crop, method, weights=None):
    if method=='character_yolo':
        rows=character_boxes(crop,weights)
        return [(normalize_glyph(crop[y:y+h,x:x+w]),row['bbox']) for row in rows for x,y,w,h in [row['bbox']]]
    from segmentation_benchmark import experimental_segment
    if method not in ('baseline','border_components'):
        raise ValueError('Unknown segmentation method')
    return experimental_segment(crop,method)[0]


def plate_crop(image):
    from benchmark_crop_padding import padded_box
    proposals=detect(image,str(CHECKPOINT))  # plate threshold remains .4
    if not proposals:
        return None
    _,bbox,score=max(proposals,key=lambda p:p[2])
    return {'bbox':bbox,'score':score,'crop_box':padded_box(bbox,.15,image.shape)}


def recognize_crop(crop, method, classifiers, weights=None):
    start=perf_counter()
    scores=[]
    if method=='character_yolo':
        rows=character_boxes(crop,weights)
        scores=[r['score'] for r in rows]
        glyphs=[(normalize_glyph(crop[y:y+h,x:x+w]),r['bbox']) for r in rows for x,y,w,h in [r['bbox']]]
    else:
        glyphs=segment(crop,method,weights)
    segmented=perf_counter()
    text=classifiers.predict([g for g,_ in glyphs])
    end=perf_counter()
    return {'glyph_boxes':[list(b) for _,b in glyphs],'glyph_scores':scores,'text':text,'seconds':end-start,
            'segmentation_seconds':segmented-start,'classification_seconds':end-segmented}


def infer(image, weights, classifiers):
    start=perf_counter(); plate=plate_crop(image)
    if plate is None:
        return {'plate':None,'glyph_boxes':[],'text':{},'total_seconds':perf_counter()-start}
    x,y,w,h=plate['crop_box']
    result=recognize_crop(image[y:y+h,x:x+w],'character_yolo',classifiers,weights)
    return {'plate':plate,**result,'total_seconds':perf_counter()-start}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('image',type=Path);parser.add_argument('weights',type=Path)
    args=parser.parse_args();image=cv2.imread(str(args.image))
    if image is None or not args.weights.is_file():
        parser.error('Readable image and trusted trained character checkpoint required')
    print(json.dumps(infer(image,args.weights,MixClassifiers()),indent=2))