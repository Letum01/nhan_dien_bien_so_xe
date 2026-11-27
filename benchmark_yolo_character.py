"""Fixed development validation benchmark; labels enter only scoring and GT crop control."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from evaluate_labeled import edit_distance
from glyph_review import ROOT, normalized_text
from segmentation_benchmark import match_boxes
from yolo_character_dataset import sha, validate
from yolo_character_pipeline import CHECKPOINT, WEIGHTS, MixClassifiers, plate_crop, recognize_crop

METHODS=('baseline','border_components','character_yolo')


def score(prediction, crop_box, truth, row):
    x,y,w,h=crop_box or (0,0,0,0)
    boxes=[[x+bx,y+by,bw,bh] for bx,by,bw,bh in prediction['glyph_boxes']]
    geometry=match_boxes(boxes,truth,.5)
    coverage=sum(x<=gx and y<=gy and gx+gw<=x+w and gy+gh<=y+h for gx,gy,gw,gh in truth)
    ref=normalized_text(row['plate_text'])
    return {'geometry':geometry,'coverage':coverage,'gt_boxes':truth,'pred_boxes':boxes,'reference':ref,
            'ocr':{k:{'prediction':p,'exact':p==ref,'edits':edit_distance(ref,p),
                      'matched_correct':sum(p[m['pred']]==row['characters'][m['gt']]['label'] for m in geometry['matches'])}
                   for k,p in prediction['text'].items()}}


def summarize(rows):
    tp,fp,fn=(sum(r['geometry'][k] for r in rows) for k in ('tp','fp','fn'))
    n=sum(len(r['reference']) for r in rows)
    scores=[s for r in rows for s in r.get('glyph_scores',[])]
    ious=[m['iou'] for r in rows for m in r['geometry']['matches']]
    return {'plates':len(rows),'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,
            'mean_matched_iou':float(np.mean(ious)) if ious else None,
            'mean_seconds':float(np.mean([r.get('seconds',0) for r in rows])),
            'confidence_quantiles':dict(zip(('min','p25','median','p75','max'),np.quantile(scores,[0,.25,.5,.75,1]).tolist())) if scores else None,
            'recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn),'same_count':sum(r['geometry']['same_count'] for r in rows),
            'full_order':sum(r['geometry']['complete_order_correct'] for r in rows),
            'crop_complete':sum(r['coverage']==len(r['gt_boxes']) for r in rows),
            'ocr':{k:{'cer':sum(r['ocr'][k]['edits'] for r in rows)/n,'exact':sum(r['ocr'][k]['exact'] for r in rows),
                      'matched_correct':sum(r['ocr'][k]['matched_correct'] for r in rows),'matched_total':tp}
                   for k in ('svm','rf','cnn','ensemble')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','weights','output'): parser.add_argument(name,type=Path)
    args=parser.parse_args()
    if not args.weights.is_file(): parser.error('Trained character checkpoint required; no substitute model')
    validate(args.dataset)
    manifest=args.dataset/'manifest.json'
    rows=[e['review'] for e in json.loads(manifest.read_text(encoding='utf-8'))['rows'] if e['review']['split']=='val']
    args.output.mkdir(parents=True,exist_ok=False)
    protocol={'scope':'development validation only','manifest_sha256':sha(manifest),'box_iou':.5,
              'plate_conf':.4,'plate_pad_each_side':.15,'character_conf':.25,'character_nms_iou':.7,'character_imgsz':640,
              'threshold_selection':'Fixed .25 from readiness protocol, no validation sweep or retuning; AP measured separately',
              'hashes':{str(p):sha(p) for p in (args.weights,CHECKPOINT,*WEIGHTS.values())}}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    models=MixClassifiers(); records=[]
    for row in rows:
        image=cv2.imread(str(ROOT/row['image_path'])); px,py,_,_=row['plate_bbox']
        gt=[[px+x,py+y,w,h] for c in row['characters'] for x,y,w,h in [c['bbox']]]
        start=perf_counter(); plate=plate_crop(image); plate_seconds=perf_counter()-start
        for scope,box in (('gt',row['plate_bbox']),('yolo',plate['crop_box'] if plate else None)):
            for method in METHODS:
                if box:
                    x,y,w,h=box
                    pred=recognize_crop(image[y:y+h,x:x+w],method,models,args.weights)
                else:
                    pred={'glyph_boxes':[],'text':dict.fromkeys(('svm','rf','cnn','ensemble'),''),'seconds':0}
                records.append({'id':row['image_id'],'image':row['image_path'],'source_group':row['source_group'],'scope':scope,'method':method,
                                'glyph_scores':pred.get('glyph_scores',[]),
                                'crop_box':box,'plate_detection':plate if scope=='yolo' else None,
                                'seconds':pred['seconds']+(plate_seconds if scope=='yolo' else 0),**score(pred,box,gt,row)})
    summary={s:{m:summarize([r for r in records if r['scope']==s and r['method']==m]) for m in METHODS} for s in ('gt','yolo')}
    (args.output/'results.json').write_text(json.dumps({'summary':summary,'records':records},indent=2),encoding='utf-8')


if __name__=='__main__':
    main()