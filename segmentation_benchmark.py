"""Frozen IoU>=.5 segmentation protocol. Select on TRAIN, report VAL separately.

Experimental algorithms only; production segmentation/weights remain unchanged.
GT matches score output after inference; never supply GT to candidate inference.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from segmentation import segment_characters,preprocess_plate
from localization import _box_iou

ROOT=Path(__file__).resolve().parent
VARIANTS=('baseline','raw_otsu','adaptive','clahe_otsu','open3','close2','border_components','height_consistency','row_cluster')


def match_boxes(predicted,truth,threshold=.5):
    """Max-cardinality valid one-to-one matches, then maximize total IoU.

    A 1000 bonus dominates sum IoU for these small plate glyph sets. Invalid
    assignment pairs discarded. No index pairing of unmatched glyphs.
    """
    matrix=np.array([[_box_iou(p,g) for g in truth] for p in predicted],dtype=float).reshape(len(predicted),len(truth))
    matches=[]
    if matrix.size:
        a,b=linear_sum_assignment(-((matrix>=threshold)*1000+matrix))
        matches=[{'pred':int(i),'gt':int(j),'iou':float(matrix[i,j])} for i,j in zip(a,b) if matrix[i,j]>=threshold]
    matched_pred={m['pred'] for m in matches};matched_gt={m['gt'] for m in matches}
    # overlap-based diagnostics only, not confirmed causal split/merge labels.
    split_candidates=[];merge_candidates=[]
    for j,g in enumerate(truth):
        overlapping=[i for i,p in enumerate(predicted) if overlap_fraction(p,g)>.2]
        if len(overlapping)>1:split_candidates.append({'gt':j,'predictions':overlapping})
    for i,p in enumerate(predicted):
        overlapping=[j for j,g in enumerate(truth) if overlap_fraction(g,p)>.2]
        if len(overlapping)>1:merge_candidates.append({'pred':i,'ground_truth':overlapping})
    order=[m['gt'] for m in sorted(matches,key=lambda m:m['pred'])]
    complete=len(matches)==len(truth)==len(predicted)
    return {'tp':len(matches),'fp':len(predicted)-len(matches),'fn':len(truth)-len(matches),
            'matches':matches,'unmatched_pred':[i for i in range(len(predicted)) if i not in matched_pred],
            'unmatched_gt':[j for j in range(len(truth)) if j not in matched_gt],
            'same_count':len(predicted)==len(truth),'full_match':complete,
            'relative_order_correct_on_matches':len(order)>0 and order==sorted(order),
            'complete_order_correct':complete and order==list(range(len(truth))),
            'split_overlap_candidates':split_candidates,'merge_overlap_candidates':merge_candidates}


def overlap_fraction(a,b):
    x,y,w,h=a;ox,oy,ow,oh=b
    return max(0,min(x+w,ox+ow)-max(x,ox))*max(0,min(y+h,oy+oh)-max(y,oy))/max(w*h,1)


def experimental_segment(crop,variant):
    trace={}
    binary=None
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    enhanced=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(gray)
    if variant=='raw_otsu':binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    if variant=='adaptive':binary=cv2.adaptiveThreshold(enhanced,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,7)
    if variant=='clahe_otsu':binary=cv2.threshold(cv2.GaussianBlur(enhanced,(3,3),0),0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    if variant=='open3':binary=cv2.morphologyEx(preprocess_plate(crop),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    if variant=='close2':binary=cv2.morphologyEx(preprocess_plate(crop),cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))
    if variant=='border_components':
        binary=preprocess_plate(crop)
        # Remove components connected to image border; may erase edge glyphs.
        count,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
        for i in range(1,count):
            x,y,w,h,area=stats[i]
            if x==0 or y==0 or x+w==binary.shape[1] or y+h==binary.shape[0]:binary[labels==i]=0
    segments=segment_characters(crop,trace=trace,binary_override=binary)
    trace['experimental_rejections']=[]
    if variant=='height_consistency' and segments:
        median=float(np.median([b[3] for _,b in segments]))
        retained=[]
        for char,box in segments:
            x,y,w,h=box
            reason=None
            if not .55*median<=h<=1.6*median:reason='height_vs_plate_median'
            if (x<=2 or x+w>=crop.shape[1]-2) and (w/h<.22 or h>1.35*median):reason='narrow_edge_or_tall_frame'
            if reason:trace['experimental_rejections'].append({'bbox':box,'reason':reason})
            else:retained.append((char,box))
        segments=retained
    if variant=='row_cluster' and len(segments)>1:
        # Bottom/upper centers from largest center gap, normalized by median glyph height.
        ordered=sorted(segments,key=lambda p:p[1][1]+p[1][3]/2)
        centers=[b[1]+b[3]/2 for _,b in ordered]
        gap=np.diff(centers);split=int(np.argmax(gap))+1
        median=np.median([b[3] for _,b in ordered])
        if gap.max()>.6*median:
            segments=sorted(ordered[:split],key=lambda p:p[1][0])+sorted(ordered[split:],key=lambda p:p[1][0])
        else:segments=sorted(ordered,key=lambda p:p[1][0])
    trace['output_boxes']=[list(b) for _,b in segments]
    return segments,trace


def aggregate(rows):
    tp=sum(r['segmentation']['tp'] for r in rows);fp=sum(r['segmentation']['fp'] for r in rows);fn=sum(r['segmentation']['fn'] for r in rows)
    ious=[m['iou'] for r in rows for m in r['segmentation']['matches']]
    return {'plates':len(rows),'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'mean_matched_iou':float(np.mean(ious)) if ious else None,
            'same_count_plates':sum(r['segmentation']['same_count'] for r in rows),
            'full_match_plates':sum(r['segmentation']['full_match'] for r in rows),
            'complete_order_correct_plates':sum(r['segmentation']['complete_order_correct'] for r in rows),
            'glyph_match_classifier_correct':sum(r.get('matched_classifier_correct',0) for r in rows),
            'glyph_match_classifier_samples':sum(r.get('matched_classifier_samples',0) for r in rows),
            'exact_text_on_gt_crop':sum(r.get('exact_text',False) for r in rows)}


def run(audit):
    from validate_glyph_dataset import validate
    from dataset_validation import check_split_leakage
    audit=Path(audit)
    validate(audit/'dataset')
    rows=json.loads((audit/'dataset/ocr_revisions.json').read_text(encoding='utf-8'))
    if len(rows)!=25 or sum(len(r['characters']) for r in rows)!=204:raise ValueError('Expected approved25/204 snapshot')
    check_split_leakage(rows,ROOT)
    output=ROOT/'reports'/('segmentation_benchmark_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir()
    protocol={'matching':'maximum cardinality valid pairs then IoU','iou':.5,'variants':VARIANTS,
              'selection':'TRAIN segmentation F1, ties baseline/earlier predefined variant, no OCR score selection',
              'validation':'fixed7plates57glyphs, development not final independent test',
              'production_changed':False,'snapshot':str(audit.resolve()),
              'snapshot_sha256':hashlib.sha256((audit/'dataset/ocr_revisions.json').read_bytes()).hexdigest()}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    records=[];stored={}
    # First compute all TRAIN records, freeze chosen variant before running VAL.
    selected=None
    for split in ('train','val'):
        for row in rows:
            if row['split']!=split:continue
            image=cv2.imread(str(ROOT/row['image_path']));x,y,w,h=row['plate_bbox'];crop=image[y:y+h,x:x+w]
            truth=[c['bbox'] for c in row['characters']]
            folder=output/split/row['image_id'][:12];folder.mkdir(parents=True)
            cv2.imwrite(str(folder/'crop.png'),crop)
            for variant in VARIANTS:
                start=time.perf_counter();segments,trace=experimental_segment(crop,variant)
                result=match_boxes([b for _,b in segments],truth)
                record={'image':row['image_path'],'plate_id':row['image_id'],'source_group':row['source_group'],
                        'split':split,'variant':variant,'seconds':time.perf_counter()-start,'segmentation':result,
                        'ground_truth_boxes':truth,'prediction_boxes':[list(b) for _,b in segments],
                        'contour_count':len(trace['contours']),'trace':{k:v for k,v in trace.items() if k!='binary'}}
                records.append(record);stored[(row['image_id'],variant)]=segments
                cv2.imwrite(str(folder/(variant+'_binary.png')),trace['binary'])
                overlay=crop.copy();contours=crop.copy()
                for c in trace['contours']:
                    cx,cy,cw,ch=c['bbox'];cv2.rectangle(contours,(cx,cy),(cx+cw,cy+ch),(0,0,255) if c['reasons'] else (0,255,0),1)
                cv2.imwrite(str(folder/(variant+'_all_contours.png')),contours)
                for c in row['characters']:
                    cx,cy,cw,ch=c['bbox'];cv2.rectangle(overlay,(cx,cy),(cx+cw,cy+ch),(0,255,0),1)
                for i,(_,box) in enumerate(segments):
                    cx,cy,cw,ch=box;cv2.rectangle(overlay,(cx,cy),(cx+cw,cy+ch),(0,0,255),1)
                    cv2.putText(overlay,str(i),(cx,max(10,cy-2)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,255),1)
                cv2.imwrite(str(folder/(variant+'_overlay.png')),overlay)
        if split=='train':
            train_metrics={v:aggregate([r for r in records if r['variant']==v]) for v in VARIANTS}
            selected=max(VARIANTS,key=lambda v:train_metrics[v]['f1'] or 0)
            (output/'selection_FROZEN_BEFORE_VALIDATION.json').write_text(json.dumps({'selected':selected,'train_metrics':train_metrics},indent=2),encoding='utf-8')
    # Frozen classifiers used only AFTER segmentation policy selection.
    import tensorflow as tf
    checkpoint=ROOT/'experiments/ocr_mix_experiment02_20261125T080339560263Z/cnn_mix/cnn_model.keras'
    cnn=tf.keras.models.load_model(checkpoint,compile=False)
    if cnn.input_shape!=(None,32,32,1) or cnn.output_shape!=(None,36):raise ValueError('CNN input/output mismatch')
    from main import recognize_characters
    from glyph_review import normalized_text
    metadata={r['image_id']:r for r in rows}
    for record in records:
        if record['variant'] not in ('baseline',selected):continue
        row=metadata[record['plate_id']];segments=stored[(record['plate_id'],record['variant'])]
        predicted=recognize_characters([c for c,_ in segments],'cnn',{'cnn':cnn})
        record.update(raw_text=predicted,reviewed_text=normalized_text(row['plate_text']),
                      exact_text=predicted==normalized_text(row['plate_text']),
                      matched_classifier_samples=len(record['segmentation']['matches']),
                      matched_classifier_correct=sum(predicted[m['pred']]==row['characters'][m['gt']]['label'] for m in record['segmentation']['matches']))
        record['matched_classification']=[{'pred_index':m['pred'],'gt_index':m['gt'],'iou':m['iou'],
                                           'prediction':predicted[m['pred']],'expected':row['characters'][m['gt']]['label']} for m in record['segmentation']['matches']]
    summary={split:{v:aggregate([r for r in records if r['split']==split and r['variant']==v]) for v in VARIANTS} for split in ('train','val')}
    # Independent pipeline output: select top detector score, never GT IoU to choose.
    from yolo_ui_bridge import detect,CHECKPOINT
    end_to_end=[]
    for row in rows:
        if row['split']!='val':continue
        image=cv2.imread(str(ROOT/row['image_path']));start=time.perf_counter()
        proposals=detect(image,str(CHECKPOINT))
        item={'image':row['image_path'],'scope':'7development validation sources only, not final test',
              'detector_boxes':[{'bbox':b,'score':s} for _,b,s in proposals], 'status':'not_found','prediction':'','exact_text':False}
        if proposals:
            crop,box,score=max(proposals,key=lambda p:p[2])
            segments,trace=experimental_segment(crop,selected)
            predicted=recognize_characters([c for c,_ in segments],'cnn',{'cnn':cnn})
            dx,dy,_,_=box
            # GT only for scoring after output is finalized.
            global_pred=[[dx+b[0],dy+b[1],b[2],b[3]] for _,b in segments]
            px,py,_,_=row['plate_bbox'];global_gt=[[px+c['bbox'][0],py+c['bbox'][1],c['bbox'][2],c['bbox'][3]] for c in row['characters']]
            score_matches=match_boxes(global_pred,global_gt)
            covered=sum(dx<=gx and dy<=gy and dx+box[2]>=gx+gw and dy+box[3]>=gy+gh for gx,gy,gw,gh in global_gt)
            item.update(status='prediction' if predicted else 'no_glyphs',prediction=predicted,
                        exact_text=predicted==normalized_text(row['plate_text']),reviewed_text=normalized_text(row['plate_text']),
                        selected_box=box,detection_iou=_box_iou(box,row['plate_bbox']),
                        gt_glyphs_fully_inside_crop=covered,gt_glyph_count=len(global_gt),
                        segmentation=score_matches,matched_classifier_correct=sum(predicted[m['pred']]==row['characters'][m['gt']]['label'] for m in score_matches['matches']))
            cv2.imwrite(str(output/(row['image_id'][:12]+'_yolo_crop.png')),crop)
        item['total_seconds']=time.perf_counter()-start;end_to_end.append(item)
    result={'protocol':protocol,'selected_variant_train_only':selected,'summary':summary,'per_plate':records,'yolo_end_to_end':end_to_end,
            'cnn_checkpoint':str(checkpoint),'cnn_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'test_used':False,'production_changed':False,'note':'matched classifier accuracy excludes FP/FN and must be reported together with segmentation recall. Count alone not completeness.'}
    (output/'benchmark.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    table=[]
    for split,variants in summary.items():
        for variant,m in variants.items():table.append({'split':split,'variant':variant,**m})
    with (output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print('OUTPUT',output,'SELECTED',selected)
    print(json.dumps(summary,indent=2))
    print('YOLO E2E',sum(r['exact_text'] for r in end_to_end),'/',len(end_to_end))
    return output,result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('audit',type=Path);args=parser.parse_args();run(args.audit)