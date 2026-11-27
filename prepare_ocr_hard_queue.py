"""Train-only unlabeled OCR review suggestions, metadata proxies NOT GT."""
from collections import Counter
from datetime import datetime,timezone
import json

import cv2
import numpy as np

from annotation_store import AnnotationStore
from glyph_review import ROOT,plates_from_detection,normalized_text,CLASSES


def main():
    detection=AnnotationStore(ROOT/'data/annotations.sqlite3');ocr=AnnotationStore(ROOT/'data/ocr_glyphs.sqlite3')
    done={r['image_id'] for r in ocr.all_latest().values() if r.get('review_status')=='reviewed'}
    train=[r for r in ocr.all_latest().values() if r.get('review_status')=='reviewed' and r['split']=='train']
    counts=Counter(c['label'] for r in train for c in r['characters'])
    candidates=[]
    for row in plates_from_detection(detection):
        if row['split']!='train' or row['image_id'] in done:continue
        image=cv2.imread(str(ROOT/row['image_path']));x,y,w,h=row['plate_bbox'];crop=image[y:y+h,x:x+w]
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        edges=cv2.Canny(gray,50,150)
        points=cv2.findNonZero(edges)
        angle=None
        if points is not None and len(points)>=5:
            _,(rw,rh),degrees=cv2.minAreaRect(points)
            angle=float(min(abs(degrees),abs(90-degrees)))
        from segmentation import preprocess_plate
        binary=preprocess_plate(crop)
        border=np.concatenate([binary[0],binary[-1],binary[:,0],binary[:,-1]])
        labels=set(normalized_text(row['plate_text'])) if row['text_confirmed'] else set()
        missing=[c for c in labels if not counts[c]];sparse=[c for c in labels if 0<counts[c]<3]
        candidates.append({**row,'status':'UNREVIEWED_OCR_CANDIDATE','pixel_area':w*h,'laplacian_variance':sharpness,
                           'edge_rectangle_angle_proxy':angle,'border_foreground_fraction':float(np.mean(border>0)),
                           'confirmed_text_missing_train_classes':missing,'confirmed_text_sparse_train_classes':sparse,
                           'proxy_warning':'Smallness/blur/angle/frame are unverified priority heuristics, not reviewed metadata or labels'})
    # Rank by resource-independent relative size, blur and border contact; use
    # confirmed text only, never guessed class from OCR or filename.
    for metric,reverse in [('pixel_area',False),('laplacian_variance',False),('border_foreground_fraction',True)]:
        ordered=sorted(candidates,key=lambda r:r[metric],reverse=reverse)
        for rank,row in enumerate(ordered):row[metric+'_priority_rank']=rank
    for row in candidates:
        row['priority_score']=sum(row[m+'_priority_rank'] for m in ('pixel_area','laplacian_variance','border_foreground_fraction'))-10*len(row['confirmed_text_missing_train_classes'])-3*len(row['confirmed_text_sparse_train_classes'])
    candidates.sort(key=lambda r:(r['priority_score'],r['image_path']))
    folder=ROOT/'reports'/('ocr_hard_queue_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));folder.mkdir()
    for i,row in enumerate(candidates):
        row['rank']=i+1
        image=cv2.imread(str(ROOT/row['image_path']));x,y,w,h=row['plate_bbox']
        cv2.imwrite(str(folder/f'{i+1:02d}_{row["image_id"][:10]}_crop.png'),image[y:y+h,x:x+w])
    (folder/'queue.json').write_text(json.dumps({'candidates':candidates,'train_class_counts':dict(counts),
        'missing_train_classes':[c for c in CLASSES if not counts[c]],'reviewed_or_split_changes':False},ensure_ascii=False,indent=2),encoding='utf-8')
    import csv
    fields=['rank','image_path','source_group','split','text_confirmed','plate_text','pixel_area','laplacian_variance',
            'edge_rectangle_angle_proxy','border_foreground_fraction','confirmed_text_missing_train_classes','priority_score']
    with (folder/'queue.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(candidates)
    print('TRAIN ONLY review candidates',len(candidates),folder)


if __name__=='__main__':main()