"""Posttraining inference evidence at frozen conf=.4; low-score diagnostic only."""
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parent
EXPERIMENT=ROOT/'experiments/pilot21_20261117_run01'
DATASET=ROOT/'reports/annotation_review_20261117T152948733793Z/yolo_trial'


def main():
    model=YOLO(str(EXPERIMENT/'weights/best.pt'))
    assert model.names=={0:'license_plate'},model.names
    folder=EXPERIMENT/'inference_audit';folder.mkdir(exist_ok=False)
    manifest=json.loads((DATASET/'manifest.json').read_text(encoding='utf-8'))['rows']
    sources=[(ROOT/row['image_path'],row['split']) for row in manifest if row['split'] in ('val','test')]
    sources.append((ROOT/'dataset/license_plate_dataset/0/images/0407.jpg','train_development_not_independent'))
    sources.append((ROOT/'dataset/license_plate_dataset/0/images/0402.jpg','development_not_independent'))
    records=[]
    for index,(path,split) in enumerate(sources):
        image=cv2.imread(str(path))
        result=model.predict(image,device=0,conf=.001,verbose=False)[0]
        boxes=result.boxes.xyxy.cpu().tolist();scores=result.boxes.conf.cpu().tolist()
        accepted=[(b,s) for b,s in zip(boxes,scores) if s>=.4]
        overlay=image.copy()
        for box,score in accepted:
            x1,y1,x2,y2=map(int,box)
            cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(overlay,f'plate {score:.4f}',(x1,max(15,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
        cv2.imwrite(str(folder/f'{index:02d}_{path.stem}_conf04.jpg'),overlay)
        # Diagnostic top5 only, not promoted to operational detections.
        diagnostic=image.copy()
        for box,score in sorted(zip(boxes,scores),key=lambda p:p[1],reverse=True)[:5]:
            x1,y1,x2,y2=map(int,box);cv2.rectangle(diagnostic,(x1,y1),(x2,y2),(0,180,255),1)
            cv2.putText(diagnostic,f'{score:.5f}',(x1,max(12,y1-2)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
        cv2.imwrite(str(folder/f'{index:02d}_{path.stem}_diagnostic_top5.jpg'),diagnostic)
        records.append({'image':str(path),'split':split,'threshold':.4,'accepted_count':len(accepted),
                        'accepted':[{'xyxy':b,'confidence':s} for b,s in accepted],
                        'diagnostic_conf001_count':len(boxes),'maximum_score':max(scores,default=None),
                        'all_diagnostic_boxes':[{'xyxy':b,'confidence':s} for b,s in zip(boxes,scores)]})
        print(path.name,split,'accepted@.4',len(accepted),'max_score',max(scores,default=None))
    (folder/'predictions.json').write_text(json.dumps({'checkpoint':str(EXPERIMENT/'weights/best.pt'),
        'classes':model.names,'results':records,'warning':'No threshold tuning on test; diagnostics at .001 are NOT production detections.'},indent=2),encoding='utf-8')


if __name__=='__main__':main()