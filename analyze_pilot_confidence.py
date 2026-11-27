"""Frozen-pilot label audit + validation-only raw/NMS confidence diagnostics.

No training, threshold selection, SQLite writes or label modification.
Run in .venv-yolo; output must be a new directory.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / 'experiments/pilot21_20261117_run01'
DATA = ROOT / 'reports/annotation_review_20261117T152948733793Z/yolo_trial'


def xywh_to_xyxy(box):
    x,y,w,h = box
    return [x,y,x+w,y+h]


def audit_labels(rows, output):
    """Flags are screening rules, NOT decisions to change human labels."""
    records=[]
    for i,row in enumerate(rows):
        image=cv2.imread(str(ROOT/row['image_path']))
        height,width=image.shape[:2]
        entries=[]
        for j,box in enumerate(row['boxes']):
            x,y,w,h=box
            flags=[]
            area=w*h/(width*height);aspect=w/h
            if area < .001 or area > .40:flags.append('unusual_area_fraction')
            if not .8 <= aspect <= 7:flags.append('unusual_aspect_ratio')
            if min(x,y,width-x-w,height-y-h) < .01*min(width,height):flags.append('near_image_border')
            if min(w,h)<8:flags.append('tiny_dimension')
            entries.append({'xywh':box,'xyxy':xywh_to_xyxy(box),'area_fraction':area,'aspect_ratio':aspect,'flags':flags})
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(image,f'GT{j} xywh={box}',(max(0,x-10),max(16,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
        path=output/'ground_truth'/f'{i:02d}_{Path(row["image_path"]).stem}.jpg'
        path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(path),image)
        # Validate exact export conversion against frozen source xywh.
        exported=DATA/'labels'/row['split']/f'{i:06d}.txt'
        labels=[list(map(float,line.split())) for line in exported.read_text().splitlines()]
        expected=[[0,(x+w/2)/width,(y+h/2)/height,w/width,h/height] for x,y,w,h in row['boxes']]
        conversion_ok=np.allclose(labels,expected,atol=1e-7)
        records.append({'image':row['image_path'],'split':row['split'],'source_group':row['source_group'],
                        'image_size':[width,height],'boxes':entries,'export_matches_xywh':bool(conversion_ok),
                        'screening_status':'NEEDS_REVIEW' if any(b['flags'] for b in entries) or not conversion_ok else 'NUMERIC_SCREEN_PASS_VISUAL_REVIEW_REQUIRED',
                        'overlay':str(path.relative_to(output))})
    return records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'reports/pilot_confidence_analysis')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.ops import non_max_suppression, scale_boxes
    from ultralytics.utils.metrics import box_iou
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))['rows']
    labels=audit_labels(rows,out)
    model=YOLO(str(PILOT/'weights/best.pt'))
    net=model.model.to('cuda').eval().float()
    head=net.model[-1]
    pretrained=YOLO(str(ROOT/'pretrained/yolov8n.pt'))
    transferred=sum(k in net.state_dict() and net.state_dict()[k].shape==v.shape for k,v in pretrained.model.state_dict().items())
    validation=[];scores_all=[];scatter=[];metric_tp=[];metric_scores=[]
    for row in rows:
        if row['split']!='val':continue  # TEST NEVER USED FOR SCORE/THRESHOLD ANALYSIS
        image=cv2.imread(str(ROOT/row['image_path']))
        letterbox=LetterBox(new_shape=(640,640),auto=False,stride=32)(image=image)
        batch=torch.from_numpy(np.ascontiguousarray(letterbox[:,:,::-1].transpose(2,0,1))).unsqueeze(0).to('cuda').float()/255
        with torch.no_grad():raw=net(batch)
        decoded=raw[0] if isinstance(raw,tuple) else raw
        assert decoded.shape[1]==5, decoded.shape  # xywh + ONE class probability
        raw_scores=decoded[0,4,:].detach().cpu().numpy()
        scores_all.extend(raw_scores.tolist())
        record={'image':row['image_path'],'raw_decoded_shape':list(decoded.shape),
                'pre_nms_total_locations':int(len(raw_scores)),
                'class_score_quantiles':dict(zip(['min','p50','p90','p99','max'],np.quantile(raw_scores,[0,.5,.9,.99,1]).tolist())),
                'thresholds':{}}
        gt=torch.tensor([xywh_to_xyxy(b) for b in row['boxes']],device='cuda',dtype=torch.float32)
        for threshold in (.001,.4):
            predictions=non_max_suppression(decoded.clone(),conf_thres=threshold,iou_thres=.7,max_det=300,nc=1)[0]
            if len(predictions):scale_boxes(batch.shape[2:],predictions[:,:4],image.shape)
            ious=box_iou(predictions[:,:4],gt) if len(predictions) else torch.zeros((0,len(gt)),device='cuda')
            # Greedy 1:1 match fixed IoU .5, not a threshold optimization.
            pairs=sorted([(float(ious[i,j]),i,j) for i in range(len(predictions)) for j in range(len(gt))],reverse=True)
            used_p=set();used_g=set()
            for overlap,i,j in pairs:
                if overlap>=.5 and i not in used_p and j not in used_g:used_p.add(i);used_g.add(j)
            entries=[]
            for i,prediction in enumerate(predictions):
                overlap=float(ious[i].max()) if len(gt) else 0
                score=float(prediction[4]);entries.append({'xyxy':prediction[:4].tolist(),'confidence':score,'best_gt_iou':overlap})
                if threshold==.001:scatter.append((score,overlap))
            record['thresholds'][str(threshold)]={'before_nms_above_threshold':int((raw_scores>threshold).sum()),
                'after_nms_capped_300':len(predictions),'tp_iou05':len(used_p),'fp_iou05':len(predictions)-len(used_p),
                'fn_iou05':len(gt)-len(used_g),'best_iou_per_gt':ious.max(dim=0).values.tolist() if len(predictions) else [0.0]*len(gt),
                'predictions':entries}
            if threshold==.001:
                for index,pred in enumerate(entries):
                    metric_scores.append(pred['confidence'])
                    metric_tp.append([index in used_p]*10)
            overlay=image.copy()
            for box in row['boxes']:
                x,y,x2,y2=map(int,xywh_to_xyxy(box));cv2.rectangle(overlay,(x,y),(x2,y2),(0,255,0),2)
            for pred in sorted(entries,key=lambda p:p['confidence'],reverse=True)[:5]:
                x,y,x2,y2=map(int,pred['xyxy']);cv2.rectangle(overlay,(x,y),(x2,y2),(0,180,255),2)
                cv2.putText(overlay,f"{pred['confidence']:.4f} IoU {pred['best_gt_iou']:.2f}",(x,max(12,y-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
            cv2.imwrite(str(out/f"val_{Path(row['image_path']).stem}_conf{threshold}.jpg"),overlay)
        validation.append(record)
    with (PILOT/'results.csv').open() as stream:history=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    plt.figure();plt.hist(scores_all,bins=60);plt.yscale('log');plt.xlabel('Decoded class score (validation)');plt.ylabel('Location count (log)');plt.savefig(out/'confidence_histogram.png');plt.close()
    plt.figure();plt.scatter([s for s,i in scatter],[i for s,i in scatter],s=6);plt.xlabel('Confidence');plt.ylabel('Best GT IoU');plt.savefig(out/'confidence_iou_validation.png');plt.close()
    plt.figure()
    for field in ('train/cls_loss','val/cls_loss'):plt.plot([h['epoch'] for h in history],[h[field] for h in history],label=field)
    plt.legend();plt.xlabel('Epoch');plt.savefig(out/'classification_loss.png');plt.close()
    from ultralytics.utils.metrics import ap_per_class, smooth
    metric_arrays=ap_per_class(np.asarray(metric_tp,dtype=bool),np.asarray(metric_scores),
                               np.zeros(len(metric_scores)),np.zeros(sum(len(r['boxes']) for r in rows if r['split']=='val')))
    grid=metric_arrays[10];f1_curve=metric_arrays[9].mean(axis=0)
    selected_index=int(smooth(f1_curve,.1).argmax())
    metric_diagnostic={'scope':'validation only, IoU05 repeated across columns for precision-curve diagnostic; NOT mAP50-95',
                       'ultralytics_smoothed_f1_grid_index':selected_index,
                       'corresponding_grid_confidence':float(grid[selected_index]),
                       'reported_precision':float(metric_arrays[2][0]),'reported_recall':float(metric_arrays[3][0]),
                       'unsmoothed_f1_max':float(f1_curve.max()),
                       'unsmoothed_f1_max_grid_confidence':float(grid[int(f1_curve.argmax())]),
                       'note':'No threshold applied to product. Smoothing over ~0.1 confidence range can flatten a signal restricted below .03.'}
    report={'coordinate_contract':'source boxes are xywh pixels; export labels normalized cxcywh; predictions xyxy',
            'label_audit':labels,'checkpoint_names':model.names,'head_nc':head.nc,'head_no':head.no,
            'reg_max':head.reg_max,'decoded_channels':5,'separate_objectness':False,
            'pretrained_names_count':len(pretrained.names),'same_shape_parameter_entries':transferred,
            'training_log_evidence':['nc=80 overridden to nc=1','Transferred 319/355 items','AMP checks passed'],
            'best_checkpoint_epoch_zero_based':model.ckpt.get('epoch'),
            'classification_loss_first_last':{k:[history[0][k],history[-1][k]] for k in ('train/cls_loss','val/cls_loss')},
            'validation_only':validation,'test_threshold_tuning':False,'product_threshold_changed':False,
            'nms_note':'Counts after NMS include confidence filtering and max_det=300 cap; not pure suppression count',
            'precision_reporting_diagnostic':metric_diagnostic,
            'hypotheses_not_proven':['15 training images / 40 optimizer batches insufficient for calibrated class scores',
              'train-domain imbalance and lack of reviewed negative images may hurt specificity',
              'more epochs may help but requires new validation, not test tuning']}
    (out/'analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'head_nc':head.nc,'labels':len(labels),'flagged':[r['image'] for r in labels if r['screening_status']=='NEEDS_REVIEW'],
                      'val':[{k:v for k,v in r.items() if k!='thresholds'} for r in validation]},indent=2))


if __name__=='__main__':main()