"""Targeted TRAIN-only experiments on frozen YOLO padded crops.

No validation inference/selection; recorded validation errors are described
from previous frozen results only. GT used solely for post-inference scoring.
"""
from collections import Counter
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import hashlib

import cv2
import numpy as np

from benchmark_crop_padding import summarize
from segmentation_benchmark import experimental_segment,match_boxes

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT/'reports/crop_padding_20261125T084749402271Z'
VARIANTS=('frozen_border','protected_border','geometry','projection_split','row_sort')


def normalized(binary,box):
    x,y,w,h=box;roi=binary[y:y+h,x:x+w];side=max(w,h)
    canvas=np.zeros((side,side),np.uint8)
    canvas[(side-h)//2:(side-h)//2+h,(side-w)//2:(side-w)//2+w]=roi
    return cv2.threshold(cv2.resize(canvas,(32,32),interpolation=cv2.INTER_AREA),127,255,cv2.THRESH_BINARY)[1]


def variant_segments(crop,variant):
    from segmentation import preprocess_plate,segment_characters
    if variant=='protected_border':
        binary=preprocess_plate(crop)
        count,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
        removed=[]
        for i in range(1,count):
            x,y,w,h,area=stats[i];touch=x==0 or y==0 or x+w==crop.shape[1] or y+h==crop.shape[0]
            # Keep ordinary edge-touching glyphs; remove only thin or spanning
            # frame components. This may retain noise: experiment, not promotion.
            if touch and (w>crop.shape[1]*.8 or h>crop.shape[0]*.9 or w/max(h,1)<.10):
                binary[labels==i]=0;removed.append([int(x),int(y),int(w),int(h)])
        trace={};segments=segment_characters(crop,trace=trace,binary_override=binary)
        trace['removed_frame_components']=removed
        return segments,trace
    segments,trace=experimental_segment(crop,'border_components')
    if variant=='geometry' and segments:
        median=float(np.median([b[3] for _,b in segments]))
        segments=[(char,b) for char,b in segments if .60*median<=b[3]<=1.55*median and b[2]/b[3]>=.16]
    if variant=='projection_split' and segments:
        # Only wide components; valley split selected from binary projection,
        # never expected text length or GT positions.
        new=[];median=np.median([b[2] for _,b in segments]);binary=trace['binary']
        for char,b in segments:
            x,y,w,h=b
            if w>1.7*median and w/h>.8:
                projection=np.sum(binary[y:y+h,x:x+w]>0,axis=0)
                left,right=int(w*.3),int(w*.7)
                at=left+int(np.argmin(projection[left:right]))
                if projection[at]<.25*h and min(at,w-at)>=3:
                    for box in [(x,y,at,h),(x+at,y,w-at,h)]:new.append((normalized(binary,box),box))
                    continue
            new.append((char,b))
        segments=new
    if variant=='row_sort' and len(segments)>1:
        ordered=sorted(segments,key=lambda p:p[1][1]+p[1][3]/2)
        centers=np.array([b[1]+b[3]/2 for _,b in ordered]);gaps=np.diff(centers)
        split=int(gaps.argmax())+1;median=np.median([b[3] for _,b in segments])
        if gaps.max()>.5*median:
            segments=sorted(ordered[:split],key=lambda p:p[1][0])+sorted(ordered[split:],key=lambda p:p[1][0])
        else:segments=sorted(ordered,key=lambda p:p[1][0])
    return segments,trace


def main():
    previous=json.loads((PREVIOUS/'results.json').read_text(encoding='utf-8'))
    protocol=previous['protocol']
    for path,digest in protocol['checkpoint_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Checkpoint changed')
    output=ROOT/'reports'/('segmentation_train_ablation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));output.mkdir()
    from annotation_store import AnnotationStore
    from dataset_validation import check_split_leakage
    snapshot=json.loads((Path(protocol['snapshot'])/'dataset/ocr_revisions.json').read_text(encoding='utf-8'))
    live=AnnotationStore(ROOT/'data/ocr_glyphs.sqlite3').all_latest()
    if any(live.get(r['image_id'])!=r for r in snapshot):raise ValueError('OCR snapshot changed')
    check_split_leakage(snapshot,ROOT)
    meta={r['image_id']:r for r in snapshot}
    import tensorflow as tf
    from main import recognize_characters
    cnn=tf.keras.models.load_model(protocol['cnn_checkpoint'],compile=False)
    records=[]
    # Only train crops are opened in ablation loop. Previous VAL not rerun.
    for old in previous['records']:
        if old['split']!='train' or old['config']!='p15_border_components':continue
        row=meta[old['plate_id']];box=old['expanded_crop_bbox']
        image=cv2.imread(str(ROOT/row['image_path']));x,y,w,h=box;crop=image[y:y+h,x:x+w]
        px,py,_,_=row['plate_bbox']
        truth=[[px+c['bbox'][0],py+c['bbox'][1],c['bbox'][2],c['bbox'][3]] for c in row['characters']]
        folder=output/row['image_id'][:12];folder.mkdir()
        for variant in VARIANTS:
            seg,trace=variant_segments(crop,variant)
            pred=recognize_characters([c for c,_ in seg],'cnn',{'cnn':cnn})
            boxes=[[x+b[0],y+b[1],b[2],b[3]] for _,b in seg]
            matched=match_boxes(boxes,truth)
            result={**old,'config':variant,'method':variant,'prediction':pred,'prediction_boxes_global':boxes,
                    'segmentation':matched,'exact_text':pred==old['reviewed_text'],
                    'matched_classifier_samples':len(matched['matches']),
                    'matched_classifier_correct':sum(pred[m['pred']]==row['characters'][m['gt']]['label'] for m in matched['matches'])}
            records.append(result)
            overlay=crop.copy()
            for c in row['characters']:
                cx,cy,cw,ch=c['bbox'];cv2.rectangle(overlay,(px+cx-x,py+cy-y),(px+cx+cw-x,py+cy+ch-y),(0,255,0),1)
            for i,(_,b) in enumerate(seg):
                cx,cy,cw,ch=b;cv2.rectangle(overlay,(cx,cy),(cx+cw,cy+ch),(0,0,255),1)
                cv2.putText(overlay,str(i),(cx,max(10,cy-2)),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,255),1)
            cv2.imwrite(str(folder/(variant+'.png')),overlay)
            cv2.imwrite(str(folder/(variant+'_binary.png')),trace['binary'])
    metrics={v:summarize([r for r in records if r['config']==v]) for v in VARIANTS}
    selected=max(VARIANTS,key=lambda v:metrics[v]['f1'])
    # Error table uses existing frozen records, not new validation tuning.
    errors=[]
    for old in previous['records']:
        if old['config']!=previous['selected_train_only']:continue
        row=meta[old['plate_id']];matches=old['segmentation']['matches'];text=old['prediction']
        errors.append({'image':old['image'],'split':old['split'],'crop_full':old['fully_covered_glyphs']==old['gt_glyph_count'],
                       'coverage':f"{old['fully_covered_glyphs']}/{old['gt_glyph_count']}",
                       'unmatched_noise_boxes':[old['prediction_boxes_global'][i] for i in old['segmentation']['unmatched_pred']],
                       'missing_gt_indices':old['segmentation']['unmatched_gt'],
                       'split_overlap_candidates':old['segmentation']['split_overlap_candidates'],
                       'merge_overlap_candidates':old['segmentation']['merge_overlap_candidates'],
                       'complete_order_correct':old['segmentation']['complete_order_correct'],
                       'classification_errors_on_matches':[{'pred_index':m['pred'],'gt_index':m['gt'],'prediction':text[m['pred']],
                                                            'expected':row['characters'][m['gt']]['label']} for m in matches if text[m['pred']]!=row['characters'][m['gt']]['label']],
                       'prediction':text,'expected':old['reviewed_text'],
                       'evidence_folder':str(PREVIOUS/old['split']/old['plate_id'][:12]),
                       'note':'FP/FN matched geometry; overlap split/merge flags are hypotheses, not confirmed causal labels'})
    report={'selection':'TRAIN_F1_only','train_plates':18,'validation_new_inference':False,'variants':metrics,
            'selected':selected,'records':records,'previous_frozen_error_analysis':errors,
            'production_changed':False,'training':False}
    (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,items in [('train_comparison.csv',[{'variant':v,**m} for v,m in metrics.items()]),('frozen_per_plate_errors.csv',errors)]:
        with (output/name).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    print('OUTPUT',output,'selected',selected);print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()