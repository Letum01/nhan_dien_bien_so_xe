"""Same validation split, fixed thresholds, no test evaluation/model promotion."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from ultralytics.utils.metrics import box_iou

ROOT=Path(__file__).resolve().parent
EXPERIMENT=ROOT/'experiments/yolo_round2_splitB_run01'
DATA=ROOT/'reports/round2_gate_20261124T154153249700Z/dataset'
PILOT=ROOT/'experiments/pilot21_20261117_run01'
THRESHOLDS=(.01,.05,.10,.20,.40)


def match(predictions,gt):
    ious=box_iou(predictions[:,:4],gt) if len(predictions) else torch.zeros((0,len(gt)),device=gt.device)
    used_p=set();used_g=set();matches=[]
    for overlap,i,j in sorted([(float(ious[i,j]),i,j) for i in range(len(predictions)) for j in range(len(gt))],reverse=True):
        if overlap>=.5 and i not in used_p and j not in used_g:
            used_p.add(i);used_g.add(j);matches.append({'confidence':float(predictions[i,4]),'iou':overlap,'prediction_index':i,'gt_index':j})
    return {'tp':len(matches),'fp':len(predictions)-len(matches),'fn':len(gt)-len(matches),'matches':matches,
            'predictions':[{'xyxy':p[:4].tolist(),'confidence':float(p[4])} for p in predictions]}


def main():
    output=EXPERIMENT/'validation_comparison';output.mkdir(exist_ok=False)
    rows=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))['rows']
    pilot_rows=json.loads((ROOT/'reports/round2_plan/plan.json').read_text(encoding='utf-8'))['frozen_pilot']
    training=[r for r in rows+pilot_rows if r['split']=='train']
    train_ids={r['image_id'] for r in training};train_groups={r['source_group'] for r in training}
    validation=[r for r in rows if r['split']=='val']
    assert len(validation)==12
    assert all(r['image_id'] not in train_ids and r['source_group'] not in train_groups for r in validation)
    from dataset_validation import check_split_leakage
    check_split_leakage(rows,ROOT)
    nets={name:YOLO(str(path)) for name,path in {'pilot':PILOT/'weights/best.pt','round2':EXPERIMENT/'weights/best.pt'}.items()}
    predictions=[];metrics={}
    for name,model in nets.items():
        assert model.names=={0:'license_plate'}
        metric=model.val(data=str(DATA/'data.yaml'),split='val',device=0,imgsz=640,batch=4,workers=2,
                         conf=.001,iou=.7,rect=False,project=str(output),name=name+'_metrics',plots=True,verbose=False)
        metrics[name]=metric.results_dict
        network=model.model.to('cuda').eval().float()
        for row in validation:
            image=cv2.imread(str(ROOT/row['image_path']))
            letterbox=LetterBox(new_shape=(640,640),auto=False,stride=32)(image=image)
            tensor=torch.from_numpy(np.ascontiguousarray(letterbox[:,:,::-1].transpose(2,0,1))).unsqueeze(0).to('cuda').float()/255
            with torch.no_grad():raw=network(tensor)
            decoded=raw[0] if isinstance(raw,tuple) else raw
            scores=decoded[0,4,:].detach().cpu().numpy()
            gt=torch.tensor([[x,y,x+w,y+h] for x,y,w,h in row['boxes']],device='cuda',dtype=torch.float32)
            record={'model':name,'image_id':row['image_id'],'image':row['image_path'],
                    'raw_locations':int(scores.size),'raw_score_max':float(scores.max()),
                    'raw_score_quantiles':np.quantile(scores,[.5,.9,.99,1]).tolist(),'thresholds':{}}
            for threshold in THRESHOLDS:
                boxes=non_max_suppression(decoded.clone(),conf_thres=threshold,iou_thres=.7,max_det=300,nc=1)[0]
                if len(boxes):scale_boxes(tensor.shape[2:],boxes[:,:4],image.shape)
                record['thresholds'][str(threshold)]={**match(boxes,gt),'before_nms_above_threshold':int((scores>threshold).sum())}
            predictions.append(record)
    summary=[]
    for name in nets:
        for threshold in THRESHOLDS:
            entries=[r['thresholds'][str(threshold)] for r in predictions if r['model']==name]
            tp=sum(r['tp'] for r in entries);fp=sum(r['fp'] for r in entries);fn=sum(r['fn'] for r in entries)
            matches=[m for r in entries for m in r['matches']]
            summary.append({'model':name,'threshold':threshold,'tp':tp,'fp':fp,'fn':fn,
                            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
                            'images_with_correct_detection':sum(r['tp']>0 for r in entries),
                            'mean_tp_confidence':float(np.mean([m['confidence'] for m in matches])) if matches else None,
                            'mean_tp_iou':float(np.mean([m['iou'] for m in matches])) if matches else None})
    for index,row in enumerate(validation):
        for threshold in (.01,.4):
            image=cv2.imread(str(ROOT/row['image_path']))
            for x,y,w,h in row['boxes']:cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            for name,color in [('pilot',(0,200,255)),('round2',(0,0,255))]:
                record=next(r for r in predictions if r['image_id']==row['image_id'] and r['model']==name)
                for p in record['thresholds'][str(threshold)]['predictions'][:10]:
                    x,y,x2,y2=map(int,p['xyxy']);cv2.rectangle(image,(x,y),(x2,y2),color,2)
                    cv2.putText(image,f"{name} {p['confidence']:.3f}",(x,max(12,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
            cv2.imwrite(str(output/f'{index:02d}_{Path(row["image_path"]).stem}_conf{threshold}.jpg'),image)
    report={'evaluation_scope':'12 common validation images, no image/group in either train; source/pHash checks PASS',
            'limitations':'Validation used for round2 early stopping/checkpoint selection, NOT an unbiased final test. Prior development inspection exists. Three pilot test images not evaluated.',
            'same_config':{'imgsz':640,'letterbox':'square','nms_iou':.7,'max_det':300,'thresholds':THRESHOLDS},
            'ultralytics_val_metrics':metrics,'threshold_summary':summary,'per_image':predictions,
            'test_used':False,'product_threshold_modified':False,'promoted_to_default':False}
    (output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (output/'thresholds.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    for name in nets:
        maxima=[r['raw_score_max'] for r in predictions if r['model']==name]
        plt.hist(maxima,bins=np.linspace(0,1,21),alpha=.5,label=name)
    plt.legend();plt.xlabel('Validation per-image maximum raw class score');plt.savefig(output/'confidence.png');plt.close()
    print(json.dumps({'metrics':metrics,'thresholds':summary},indent=2))


if __name__=='__main__':main()