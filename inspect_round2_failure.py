"""Failure evidence from existing validation predictions; no threshold tuning."""
import json
from pathlib import Path

import cv2

from localization import _box_iou

ROOT=Path(__file__).resolve().parent


def main():
    experiment=ROOT/'experiments/yolo_round2_splitB_run01'
    comparison=json.loads((experiment/'validation_comparison/comparison.json').read_text(encoding='utf-8'))
    record=next(r for r in comparison['per_image'] if r['model']=='round2' and 'AQUA2_20449' in r['image'])
    manifest=json.loads((ROOT/'reports/round2_gate_20261124T154153249700Z/dataset/manifest.json').read_text(encoding='utf-8'))['rows']
    gt=next(r for r in manifest if r['image_id']==record['image_id'])
    image=cv2.imread(str(ROOT/record['image']))
    output=ROOT/'reports/round2_streamlit';output.mkdir(exist_ok=True)
    for x,y,w,h in gt['boxes']:cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    predictions=[]
    for index,p in enumerate(record['thresholds']['0.1']['predictions']):
        x,y,x2,y2=p['xyxy'];box=[x,y,x2-x,y2-y]
        overlap=max(_box_iou(box,b) for b in gt['boxes'])
        predictions.append({**p,'gt_iou':overlap,'accepted_at_04':p['confidence']>=.4})
        color=(0,0,255) if p['confidence']>=.4 else (0,180,255)
        cv2.rectangle(image,(int(x),int(y)),(int(x2),int(y2)),color,2)
        cv2.putText(image,f"P{index} {p['confidence']:.3f} IoU {overlap:.3f}",(int(x),max(15,int(y)-5-index*14)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
    cv2.imwrite(str(output/'failure_gt_predictions.jpg'),image)
    x,y,w,h=gt['boxes'][0];cv2.imwrite(str(output/'failure_gt_crop.png'),cv2.imread(str(ROOT/record['image']))[y:y+h,x:x+w])
    report={'image':record['image'],'gt_xywh':gt['boxes'],'predictions':predictions,
            'evidence':'High-score box is not IoU>=.5 match; lower score box has greater GT overlap. At .4 high-score wrong geometry survives, correct-overlap box is below threshold.',
            'hypothesis':'partial plate-row localization or background/text confusion; inspect crop overlay before causal claim',
            'threshold_changed':False,'split':'validation_not_test'}
    (output/'failure.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()