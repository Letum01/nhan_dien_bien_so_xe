"""Exact production crop diagnostics plus isolated preprocessing experiments."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import csv
from pathlib import Path
import sqlite3

import cv2
import numpy as np

from dataset import CHARACTER_CLASSES,extract_hog_features
from ensemble_recognition import vote
from main import end_to_end_pipeline,load_trained_models
from segmentation import segment_characters,preprocess_plate
from yolo_ui_bridge import CHECKPOINT

ROOT=Path(__file__).resolve().parent


def score_glyphs(chars,models):
    if not chars:return {}
    features=np.array([extract_hog_features(c) for c in chars],np.float32)
    scaled=models['scaler'].transform(features)
    batch=np.stack(chars)[...,None].astype(np.float32)
    outputs={
        'svm':{'probabilities':models['svm'].predict_proba(scaled),'labels':models['svm'].classes_,
               'predicted':models['svm'].predict(scaled),'decision_scores':models['svm'].decision_function(scaled)},
        'rf':{'probabilities':models['rf'].predict_proba(features),'labels':models['rf'].classes_,
              'predicted':models['rf'].predict(features)},
        'cnn':{'probabilities':models['cnn'].predict(batch,verbose=0),'labels':np.arange(36)},
    }
    result={}
    for name,data in outputs.items():
        probs=data['probabilities'];labels=data['labels'];predicted=data.get('predicted',labels[probs.argmax(axis=1)])
        entries=[]
        for i,row in enumerate(probs):
            order=np.argsort(row)[::-1]
            entries.append({'index':i,'prediction':CHARACTER_CLASSES[int(predicted[i])],
                'top3':[{'class':CHARACTER_CLASSES[int(labels[j])],'score':float(row[j])} for j in order[:3]],
                'top1_top2_margin':float(row[order[0]]-row[order[1]]),
                'scores':row.tolist(),'decision_scores':data['decision_scores'][i].tolist() if 'decision_scores' in data else None})
        result[name]={'raw_text':''.join(e['prediction'] for e in entries),'glyphs':entries,
                      'class_order':[CHARACTER_CLASSES[int(x)] for x in labels],
                      'score_kind':'classifier_probability_output_NOT_real_data_calibrated',
                      'prediction_histogram':dict(Counter(e['prediction'] for e in entries))}
    return result


def reviewed_inventory():
    path=ROOT/'data/annotations.sqlite3'
    with sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True) as connection:
        rows=[json.loads(r[0]) for r in connection.execute('SELECT r.payload FROM revisions r JOIN (SELECT image_id,MAX(version) v FROM revisions GROUP BY image_id)t ON r.image_id=t.image_id AND r.version=t.v')]
    plates=[p for r in rows if r.get('review_status')=='reviewed' for p in r.get('plates',[])]
    glyphs=[c for p in plates for c in p.get('characters',[]) if c.get('confirmed') and c.get('label') in CHARACTER_CLASSES]
    counts=Counter(c['label'] for c in glyphs)
    return {'detection_reviewed':sum(r.get('review_status')=='reviewed' for r in rows),
            'confirmed_readable_plate_texts':sum(bool(p.get('text_confirmed')) and bool(p.get('plate_text','').strip()) and not p.get('unreadable') for p in plates),
            'confirmed_glyphs':len(glyphs),'class_counts':dict(counts),'missing_classes':[c for c in CHARACTER_CLASSES if not counts[c]],
            'benchmark_status':'DATASET_INSUFFICIENT' if not glyphs else 'NEEDS_VALIDATION_AND_SPLIT_PROVENANCE',
            'accuracy':None,'warning':'Confirmed flags must be validated, not assumed benchmark quality; detection review is not OCR review.'}


def diagnose(image,models=None):
    if image is None or image.size==0:raise ValueError('Empty diagnostics image')
    models=models or load_trained_models(str(ROOT/'saved_models'))
    identity=hashlib.sha256(image.tobytes()+str(image.shape).encode()).hexdigest()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=ROOT/'reports/ocr_diagnostics'/f'{stamp}_{identity[:12]}'
    folder.mkdir(parents=True,exist_ok=False)
    def save(name,pixels):
        if not cv2.imwrite(str(folder/name),pixels):raise OSError(name)
    save('00_input.png',image)
    trace={}
    text,overlay,crop,_=end_to_end_pipeline(image,'cnn',models,trace,
                         detector_type='yolo_isolated',detector_weights=str(CHECKPOINT))
    save('01_detection.png',overlay)
    report={'image_id':identity,'timestamp':stamp,'detector_checkpoint':str(CHECKPOINT),
            'detector_sha256':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
            'ocr_weights_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'saved_models').glob('*') if p.is_file() and p.suffix in ('.keras','.joblib')},
            'product_text':text,'detections':trace.get('detections',[]),'selected_box':trace.get('selected_box'),
            'config':{'baseline':'preprocess_plate exact production','target_size':[32,32],
                      'experimental_only':['raw_otsu','adaptive','no_open','pad4','quadrilateral_rectification']},
            'dataset':reviewed_inventory(),'status':trace['status'],'variants':{}}
    if crop is not None:
        save('02_crop.png',crop)
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);save('03_gray.png',gray)
        enhanced=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(gray);save('04_clahe.png',enhanced)
        _,raw_otsu=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        _,no_open=cv2.threshold(cv2.GaussianBlur(enhanced,(3,3),0),0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        adaptive=cv2.adaptiveThreshold(enhanced,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,7)
        variants={'baseline':(crop,None),'raw_otsu':(crop,raw_otsu),'no_open':(crop,no_open),'adaptive':(crop,adaptive)}
        x,y,w,h=trace['selected_box'];padded=image[max(0,y-4):min(image.shape[0],y+h+4),max(0,x-4):min(image.shape[1],x+w+4)]
        variants['pad4']=(padded,None)
        from debug_pipeline import rectify_plate
        variants['quadrilateral_rectification']=(rectify_plate(crop),None)
        for name,(variant,binary) in variants.items():
            evidence={};segments=segment_characters(variant,trace=evidence,binary_override=binary)
            save(f'{name}_crop.png',variant);save(f'{name}_binary.png',evidence['binary'])
            normalized=[char for char,_ in segments]
            scores=score_glyphs(normalized,models)
            predictions={k:v['raw_text'] for k,v in scores.items()}
            ensemble=vote(predictions).__dict__ if scores else None
            vis=variant.copy()
            for entry in evidence['contours']:
                cx,cy,cw,ch=entry['bbox'];color=(0,0,255) if entry['reasons'] else (0,255,0)
                cv2.rectangle(vis,(cx,cy),(cx+cw,cy+ch),color,1)
                cv2.putText(vis,str(entry['index']),(cx,max(10,cy-1)),cv2.FONT_HERSHEY_SIMPLEX,.3,color,1)
            save(f'{name}_contours.png',vis)
            ordered=variant.copy()
            for i,(char,(cx,cy,cw,ch)) in enumerate(segments):
                save(f'{name}_glyph_{i:02d}_raw.png',variant[cy:cy+ch,cx:cx+cw])
                save(f'{name}_glyph_{i:02d}_normalized.png',char)
                cv2.rectangle(ordered,(cx,cy),(cx+cw,cy+ch),(0,255,0),1)
                cv2.putText(ordered,str(i),(cx,max(10,cy-2)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
            save(f'{name}_reading_order.png',ordered)
            evidence.pop('binary')
            for entry in evidence['contours']:
                ids=entry.get('final_indices',[])
                entry['predictions']={k:[v['glyphs'][i]['prediction'] for i in ids] for k,v in scores.items()}
            with (folder/f'{name}_contours.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                fields=['index','bbox','area','aspect_ratio','height_ratio','width_ratio','status','reasons','final_indices','predictions']
                writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(evidence['contours'])
            report['variants'][name]={'segmentation':evidence,'glyph_count':len(segments),'classifiers':scores,
                                      'ensemble':ensemble,'accuracy':None,'status':'experiment_not_promoted' if name!='baseline' else 'production_baseline'}
        report['model_contract']={'cnn_input':str(models['cnn'].input_shape),'cnn_output':str(models['cnn'].output_shape),
           'cnn_rescaling':[l.get_config() for l in models['cnn'].layers if l.__class__.__name__=='Rescaling'],
           'svm_classes':models['svm'].classes_.tolist(),'rf_classes':models['rf'].classes_.tolist(),
           'hog_dimensions':int(models['scaler'].n_features_in_),'class_mapping':CHARACTER_CLASSES,
           'input_range':'uint8 0..255 binary white glyph black background; CNN internally /255',
           'calibration':'No real-data calibration artifact. All scores uncalibrated; no calibrated confidence claimed.'}
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return folder,report


def render(image,models):
    import streamlit as st
    folder,report=diagnose(image,models)
    st.success(f'OCR Diagnostics lưu tại {folder}')
    st.caption('Không tính accuracy khi chưa có glyph ground truth. Biến thể chỉ thử nghiệm, không đổi production.')
    for name,entry in report['variants'].items():
        with st.expander(f"{name}: {entry['glyph_count']} glyph",expanded=name=='baseline'):
            st.image(str(folder/f'{name}_binary.png'),caption='Binary')
            st.image(str(folder/f'{name}_contours.png'),caption='Green candidate / red rejected (see reasons)')
            st.image(str(folder/f'{name}_reading_order.png'),caption='Reading order')
            import pandas as pd
            st.dataframe(pd.DataFrame(entry['segmentation']['contours']),use_container_width=True)
            st.json({k:{'raw_text':v['raw_text'],'prediction_histogram':v['prediction_histogram']} for k,v in entry['classifiers'].items()})
            for index in range(entry['glyph_count']):
                st.image(str(folder/f'{name}_glyph_{index:02d}_normalized.png'),width=64,caption=f'Glyph {index}')
                st.json({model:data['glyphs'][index] for model,data in entry['classifiers'].items()})
            st.json(entry['ensemble'])
    st.json(report['dataset'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('image');args=parser.parse_args()
    folder,report=diagnose(cv2.imread(args.image))
    print(folder)
    for name,r in report['variants'].items():print(name,r['glyph_count'],{k:v['raw_text'] for k,v in r['classifiers'].items()})