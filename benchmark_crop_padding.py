"""Experimental YOLO crop padding grid: train-only F1 selection, frozen VAL.

Padding fraction is added PER SIDE based on original detection w/h. Integer
floor/ceil and image clipping. Detection xywh and score never modified. GT is
used for scoring only, not crop selection, padding or output correction.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from segmentation_benchmark import experimental_segment, match_boxes
from localization import _box_iou

ROOT = Path(__file__).resolve().parent
PADDINGS = (0, .05, .10, .15)
METHODS = ('baseline', 'border_components')


def padded_box(box, fraction, image_shape):
    if fraction < 0 or not math.isfinite(fraction):
        raise ValueError('Padding must be finite and nonnegative')
    x,y,w,h=box
    if w<=0 or h<=0:raise ValueError('Invalid detection dimensions')
    height,width=image_shape[:2]
    left=max(0,math.floor(x-w*fraction));top=max(0,math.floor(y-h*fraction))
    right=min(width,math.ceil(x+w+w*fraction));bottom=min(height,math.ceil(y+h+h*fraction))
    if right<=left or bottom<=top:raise ValueError('Detection outside image')
    return left,top,right-left,bottom-top


def summarize(records):
    tp=sum(r['segmentation']['tp'] for r in records)
    fp=sum(r['segmentation']['fp'] for r in records)
    fn=sum(r['segmentation']['fn'] for r in records)
    matches=[m for r in records for m in r['segmentation']['matches']]
    classifier_n=sum(r['matched_classifier_samples'] for r in records)
    return {'plates':len(records),'tp':tp,'fp':fp,'fn':fn,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
            'matched_mean_iou':float(np.mean([m['iou'] for m in matches])) if matches else None,
            'crop_full_coverage_plates':sum(r['fully_covered_glyphs']==r['gt_glyph_count'] for r in records),
            'fully_covered_glyphs':sum(r['fully_covered_glyphs'] for r in records),
            'same_glyph_count_plates':sum(r['segmentation']['same_count'] for r in records),
            'full_match_plates':sum(r['segmentation']['full_match'] for r in records),
            'full_order_plates':sum(r['segmentation']['complete_order_correct'] for r in records),
            'matched_classifier_correct':sum(r['matched_classifier_correct'] for r in records),
            'matched_classifier_samples':classifier_n,
            'matched_classifier_accuracy':sum(r['matched_classifier_correct'] for r in records)/classifier_n if classifier_n else None,
            'exact_plate_count':sum(r['exact_text'] for r in records)}


def main(audit):
    from validate_glyph_dataset import validate
    from dataset_validation import check_split_leakage
    from annotation_store import AnnotationStore
    from glyph_review import plates_from_detection, normalized_text
    from yolo_ui_bridge import detect, CHECKPOINT
    from main import recognize_characters
    import tensorflow as tf
    audit=Path(audit).resolve()
    validated=validate(audit/'dataset')
    rows=json.loads((audit/'dataset/ocr_revisions.json').read_text(encoding='utf-8'))
    live=AnnotationStore(ROOT/'data/ocr_glyphs.sqlite3').all_latest()
    parents={r['image_id']:r for r in plates_from_detection(AnnotationStore(ROOT/'data/annotations.sqlite3'))}
    if len(rows)!=25 or sum(len(r['characters']) for r in rows)!=204:raise ValueError('Snapshot not approved25/204')
    for row in rows:
        if live.get(row['image_id'])!=row:raise ValueError('OCR reviewed changed from snapshot')
        if any(row[k]!=parents[row['image_id']][k] for k in ('plate_bbox','source_group','split','detection_revision')):
            raise ValueError('Parent changed')
        if hashlib.sha256((ROOT/row['image_path']).read_bytes()).hexdigest()!=row['source_image_id']:raise ValueError('Source bytes changed')
    check_split_leakage(rows,ROOT)
    output=ROOT/'reports'/('crop_padding_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir()
    cnn_path=ROOT/'experiments/ocr_mix_experiment02_20261125T080339560263Z/cnn_mix/cnn_model.keras'
    protocol={'paddings_per_side':PADDINGS,'segmentation_methods':METHODS,'detection_confidence':.4,
              'selection':'highest pooled segmentation F1 over18 TRAIN plates; ties retain earlier configuration (lower padding, baseline first)',
              'matching':'maximum-cardinality one-to-one IoU>=.5, then sumIoU',
              'detection_selection':'highest detector score, no GT used','gt_coordinates':'crop relative glyph xywh translated to original image',
              'snapshot':str(audit),'cnn_checkpoint':str(cnn_path),'detector_checkpoint':str(CHECKPOINT),
              'checkpoint_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (cnn_path,CHECKPOINT)},
              'production_changed':False,'note':'development only; YOLO/CNN trained on train sources and val used previously for checkpoint selection'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    cnn=tf.keras.models.load_model(cnn_path,compile=False)
    if cnn.input_shape!=(None,32,32,1) or cnn.output_shape!=(None,36):raise ValueError('CNN contract mismatch')
    records=[];selection=None;summary={}
    for split in ('train','val'):
        for row in rows:
            if row['split']!=split:continue
            image=cv2.imread(str(ROOT/row['image_path']))
            start=time.perf_counter();proposals=detect(image,str(CHECKPOINT));detection_seconds=time.perf_counter()-start
            selected=max(proposals,key=lambda p:p[2]) if proposals else None
            px,py,_,_=row['plate_bbox']
            truth=[[px+c['bbox'][0],py+c['bbox'][1],c['bbox'][2],c['bbox'][3]] for c in row['characters']]
            folder=output/split/row['image_id'][:12];folder.mkdir(parents=True)
            (folder/'detections.json').write_text(json.dumps([{'original_bbox':b,'score':s} for _,b,s in proposals],indent=2),encoding='utf-8')
            for padding in PADDINGS:
                for method in METHODS:
                    label=f'p{int(padding*100):02d}_{method}'
                    start=time.perf_counter();segments=[];text='';global_boxes=[];covered=0;partial=[]
                    cropbox=None;trace={};box=None;score=None
                    if selected:
                        _,box,score=selected
                        cropbox=padded_box(box,padding,image.shape)
                        x,y,w,h=cropbox;crop=image[y:y+h,x:x+w]
                        segments,trace=experimental_segment(crop,method)
                        global_boxes=[[x+b[0],y+b[1],b[2],b[3]] for _,b in segments]
                        text=recognize_characters([char for char,_ in segments],'cnn',{'cnn':cnn})
                        for gx,gy,gw,gh in truth:
                            full=x<=gx and y<=gy and x+w>=gx+gw and y+h>=gy+gh
                            covered+=bool(full)
                            partial.append(max(0,min(x+w,gx+gw)-max(x,gx))*max(0,min(y+h,gy+gh)-max(y,gy))/(gw*gh))
                        cv2.imwrite(str(folder/(label+'_crop.png')),crop)
                        cv2.imwrite(str(folder/(label+'_binary.png')),trace['binary'])
                        (folder/(label+'_contours.json')).write_text(json.dumps({k:v for k,v in trace.items() if k!='binary'},indent=2),encoding='utf-8')
                    matched=match_boxes(global_boxes,truth)
                    item={'image':row['image_path'],'plate_id':row['image_id'],'split':split,'source_group':row['source_group'],
                          'padding':padding,'method':method,'config':label,'original_detection_bbox':box,'original_score':score,
                          'expanded_crop_bbox':cropbox,'detection_iou_unchanged':_box_iou(box,row['plate_bbox']) if box else None,
                          'gt_glyph_count':len(truth),'fully_covered_glyphs':covered,'glyph_coverage_fractions':partial,
                          'segmentation':matched,'prediction_boxes_global':global_boxes,'prediction':text,
                          'reviewed_text':normalized_text(row['plate_text']),'exact_text':text==normalized_text(row['plate_text']),
                          'matched_classifier_samples':len(matched['matches']),
                          'matched_classifier_correct':sum(text[m['pred']]==row['characters'][m['gt']]['label'] for m in matched['matches']),
                          'matched_classification':[{'pred':m['pred'],'gt':m['gt'],'prediction':text[m['pred']],
                                                    'expected':row['characters'][m['gt']]['label'],'iou':m['iou']} for m in matched['matches']],
                          'crop_segment_classify_seconds':time.perf_counter()-start,'detector_seconds':detection_seconds,
                          'timing_note':'isolated detector cold startup included; repeated model.predict overhead; not a throughput benchmark'}
                    records.append(item)
                    overlay=image.copy()
                    if box:
                        x,y,w,h=box;cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,200,255),2)
                    if cropbox:
                        x,y,w,h=cropbox;cv2.rectangle(overlay,(x,y),(x+w,y+h),(255,180,0),1)
                    for gx,gy,gw,gh in truth:cv2.rectangle(overlay,(gx,gy),(gx+gw,gy+gh),(0,255,0),1)
                    for i,(x,y,w,h) in enumerate(global_boxes):
                        cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,0,255),1)
                        cv2.putText(overlay,str(i),(x,max(10,y-2)),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,255),1)
                    cv2.putText(overlay,text,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
                    cv2.imwrite(str(folder/(label+'_overlay.jpg')),overlay)
            print(split,Path(row['image_path']).name,flush=True)
        metrics={f'p{int(p*100):02d}_{m}':summarize([r for r in records if r['split']==split and r['padding']==p and r['method']==m]) for p in PADDINGS for m in METHODS}
        summary[split]=metrics
        if split=='train':
            selection=max(metrics,key=lambda k:metrics[k]['f1'])
            (output/'selection_FROZEN_BEFORE_VALIDATION.json').write_text(json.dumps({'selected':selection,'metrics':metrics},indent=2),encoding='utf-8')
    result={'protocol':protocol,'selected_train_only':selection,'summary':summary,'records':records,'source_validator':validated,
            'test_used':False,'gt_used_to_repair_output':False,'production_changed':False}
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    table=[{'split':s,'config':k,**v} for s,metrics in summary.items() for k,v in metrics.items()]
    with (output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print('OUTPUT',output,'SELECTED',selection)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('audit',type=Path);args=parser.parse_args();main(args.audit)