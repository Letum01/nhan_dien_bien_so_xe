"""Evaluate frozen legacy models on reviewed normalized validation glyphs only.

Metrics describe this development set. Historical artifact training provenance
is not recoverable from weights alone: never assert independent final accuracy.
No segmentation, training, calibration or model promotion here.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from dataset import CHARACTER_CLASSES,extract_hog_features
from ensemble_recognition import vote
from main import load_trained_models,recognize_characters
from validate_glyph_dataset import validate

ROOT=Path(__file__).resolve().parent


def draw_sheet(rows,dataset,output,title,predictions=None):
    sheet=np.full((((len(rows)+8)//9)*108,9*105,3),240,np.uint8)
    for i,row in enumerate(rows):
        glyph=cv2.imread(str(dataset/row['image_path']))
        x=(i%9)*105;y=(i//9)*108
        sheet[y:y+64,x:x+64]=cv2.resize(glyph,(64,64),interpolation=cv2.INTER_NEAREST)
        caption=f"{i}:GT {row['class_label']}"+(f" / {predictions[i]}" if predictions else '')
        cv2.putText(sheet,caption,(x,y+86),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),1)
    cv2.imwrite(str(output/title),sheet)


def visual_checks(rows,dataset,output):
    warnings=[]
    for split in ('train','validation'):
        subset=[r for r in rows if r['split']==split]
        if not subset:
            continue
        sheet=np.full((((len(subset)+8)//9)*122,9*108,3),240,np.uint8)
        for index,row in enumerate(subset):
            normalized=dataset/row['image_path'];original=normalized.parent/'originals'/normalized.name
            image=cv2.imread(str(original));gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            height,width=gray.shape;ratio=min(100/width,85/height)
            tile=cv2.resize(image,(max(1,int(width*ratio)),max(1,int(height*ratio))))
            x=index%9*108;y=index//9*122;sheet[y:y+tile.shape[0],x:x+tile.shape[1]]=tile
            cv2.putText(sheet,f"{index}:{row['class_label']}",(x,y+106),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
            binary=cv2.imread(str(normalized),0)
            coverage=float(np.mean(binary>0))
            reasons=[]
            if coverage<.03 or coverage>.75:reasons.append('extreme_normalized_foreground_coverage')
            _,raw_binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
            touches=[name for name,edge in [('top',raw_binary[0]),('bottom',raw_binary[-1]),('left',raw_binary[:,0]),('right',raw_binary[:,-1])] if np.mean(edge>0)>.15]
            if touches:reasons.append('tight_glyph_foreground_at_edges_NOT_proven_cutoff')
            components=cv2.connectedComponentsWithStats(binary,8)[2][1:]
            large=[int(c[cv2.CC_STAT_AREA]) for c in components if c[cv2.CC_STAT_AREA]>=8]
            if len(large)>1:reasons.append('multiple_ink_components_can_be_noise_or_valid_disconnected_strokes')
            if reasons:warnings.append({'split':split,'sheet_index':index,'label':row['class_label'],
                'plate_id':row['plate_id'],'source_crop':row['source_crop'],'bbox':row['bbox'],
                'glyph_file':row['image_path'],'coverage':coverage,'touch_edges':touches,
                'ink_component_areas':large,'reasons':reasons})
        cv2.imwrite(str(output/f'{split}_raw_contact.jpg'),sheet)
    return warnings


def main():
    parser=argparse.ArgumentParser();parser.add_argument('audit',type=Path);args=parser.parse_args()
    folder=args.audit;dataset=folder/'dataset';validated=validate(dataset)
    from datetime import datetime,timezone
    output=folder/('baseline_evaluation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));output.mkdir(exist_ok=False)
    rows=list(csv.DictReader((dataset/'characters.csv').open(encoding='utf-8-sig')))
    train=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    if not validation:raise ValueError('No reviewed validation glyphs')
    assert not {r['source_group'] for r in train}&{r['source_group'] for r in validation}
    assert not {r['source_image_id'] for r in train}&{r['source_image_id'] for r in validation}
    models=load_trained_models(str(ROOT/'saved_models'))
    chars=[cv2.imread(str(dataset/r['image_path']),0) for r in validation]
    expected=[r['class_label'] for r in validation]
    predictions={name:recognize_characters(chars,name,models) for name in ('svm','rf','cnn')}
    ensemble=vote(predictions);predictions['ensemble']=ensemble.raw_text
    counts={split:Counter(r['class_label'] for r in rows if r['split']==split) for split in ('train','validation','test')}
    class_counts=[{'class':c,**{s:counts[s][c] for s in counts}} for c in CHARACTER_CLASSES]
    with (folder/'class_counts_by_split.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(class_counts[0]));writer.writeheader();writer.writerows(class_counts)
    draw_sheet(train,dataset,output,'train_normalized.jpg')
    draw_sheet(validation,dataset,output,'validation_normalized.jpg')
    visual_warnings=visual_checks(rows,dataset,output)
    reports={}
    for name,text in predictions.items():
        labels=CHARACTER_CLASSES+(['?'] if name=='ensemble' else [])
        prediction=list(text)
        reports[name]={'correct':sum(a==b for a,b in zip(expected,prediction)),
                       'accuracy_on_reviewed_development_glyphs':float(accuracy_score(expected,prediction)),
                       'prediction_histogram':dict(Counter(prediction)),
                       'unknown_count':prediction.count('?'),
                       'per_class':classification_report(expected,prediction,labels=labels,output_dict=True,zero_division=0),
                       'confusion_matrix':confusion_matrix(expected,prediction,labels=labels).tolist(),'matrix_class_order':labels,
                       'predictions':prediction}
        draw_sheet(validation,dataset,output,f'validation_{name}.jpg',prediction)
    raw_feats=np.array([extract_hog_features(c) for c in chars],np.float32)
    scaled=models['scaler'].transform(raw_feats)
    test_train_matches={}
    if hasattr(models['svm'],'support_vectors_'):
        distances=[float(np.min(np.linalg.norm(models['svm'].support_vectors_-f,axis=1))) for f in scaled]
        test_train_matches={'min_distance_to_any_svm_support_vector':min(distances),'near_exact_match_count_lt_1e6':sum(d<1e-6 for d in distances),
                            'note':'No exact SV match does NOT prove disjoint training; SVM stores support vectors only.'}
    artifact={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'saved_models').iterdir() if p.suffix in ('.keras','.joblib')}
    record={'validator':validated,'validation_glyphs':len(validation),
            'validation_plates':len({r['plate_id'] for r in validation}),'validation_source_groups':len({r['source_group'] for r in validation}),
            'train_glyphs':len(train),'train_classes':sum(counts['train'][c]>0 for c in CHARACTER_CLASSES),
            'missing_train':[c for c in CHARACTER_CLASSES if not counts['train'][c]],
            'train_1_or_2':[c for c in CHARACTER_CLASSES if 1<=counts['train'][c]<=2],
            'all_class_counts':class_counts,'models':reports,'artifact_sha256':artifact,
            'provenance_status':'UNVERIFIED_HISTORICAL_TRAINING_MEMBERSHIP',
            'provenance_evidence':'Repository generator is synthetic; artifact metadata has classes/shape/range but no training manifest/hash/source IDs. No real-validation fitting performed in this run.',
            'interpretation':'Descriptive reviewed development-set metrics only, not proven independent final-test accuracy. 57 glyphs from 7 plates are clustered, not 57 independent sources.',
            'svm_support_vector_probe':test_train_matches,
            'segmentation_scope':'BYPASSED: human-reviewed glyph crops. Compare full-image segmentation separately.',
            'prediction_rows':[dict(**r,**{k:predictions[k][i] for k in predictions}) for i,r in enumerate(validation)],
            'training':'NOT_RUN','model_changed':False}
    record['visual_screening_warnings']=visual_warnings
    record['visual_warning_note']='Screening only; tight crops naturally touch ink boundaries. No automatic correction/rejection.'
    (output/'results.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    print({name:{'correct':r['correct'],'n':len(validation),'accuracy':r['accuracy_on_reviewed_development_glyphs'],'histogram':r['prediction_histogram']} for name,r in reports.items()})
    print('train classes',record['train_classes'],'missing',record['missing_train'],'1-2',record['train_1_or_2'])
    print('OUTPUT',output)


if __name__=='__main__':main()