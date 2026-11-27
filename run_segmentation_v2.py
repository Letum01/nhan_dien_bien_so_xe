"""33 TRAIN development sources; no old validation or pending hard13 access."""
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import hashlib
import time
import csv
import cv2

from annotation_store import AnnotationStore
from glyph_review import ROOT,normalized_text,validate_glyph_plate,plates_from_detection
from segmentation_v2 import CONFIG,VARIANTS,segment_v2
from segmentation_benchmark import match_boxes
from benchmark_crop_padding import padded_box
from evaluate_labeled import edit_distance
from dataset_validation import check_split_leakage


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def aggregate(rows):
    tp=sum(r['matching']['tp'] for r in rows);fp=sum(r['matching']['fp'] for r in rows);fn=sum(r['matching']['fn'] for r in rows)
    n=sum(len(r['truth']) for r in rows)
    return {'plates':len(rows),'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0,
            'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
            'exact':sum(r['prediction']==r['truth'] for r in rows),
            'cer':sum(edit_distance(r['truth'],r['prediction']) for r in rows)/n,
            'same_count':sum(r['matching']['same_count'] for r in rows),
            'full_order':sum(r['matching']['complete_order_correct'] for r in rows),
            'mask_hits_reviewed_ink_pixels':sum(r['mask_hits_reviewed_ink_pixels'] for r in rows)}


def main():
    d=AnnotationStore(ROOT/'data/annotations.sqlite3');o=AnnotationStore(ROOT/'data/ocr_glyphs.sqlite3')
    allrows=[r for r in o.all_latest().values() if r.get('review_status')=='reviewed']
    rows=[r for r in allrows if r['split']=='train']
    if len(rows)!=33:raise ValueError('Expected33trainplates')
    check_split_leakage(allrows,ROOT)
    parents={r['image_id']:r for r in plates_from_detection(d)}
    for r in rows:
        if any(r[k]!=parents[r['image_id']][k] for k in ('detection_revision','plate_bbox','source_group','split')):raise ValueError('Parent changed')
        validate_glyph_plate(r,r['plate_bbox'][2],r['plate_bbox'][3])
    folder=Path(os.environ['ALPR_V2_RESUME']) if os.environ.get('ALPR_V2_RESUME') else ROOT/'reports'/('segmentation_v2_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    folder.mkdir(exist_ok=bool(os.environ.get('ALPR_V2_RESUME')))
    checkpoint=ROOT/'experiments/ocr_mix_experiment02_20261125T080339560263Z/cnn_mix/cnn_model.keras'
    from yolo_ui_bridge import detect,CHECKPOINT
    protocol={'config':CONFIG,'variants':VARIANTS,'selection':'YOLOcrop train exact descending,CER ascending,recall descending,F1 descending; no val',
              'crop':'GT no padding;YOLO top score.4 padded15%each side',
              'cnn_sha256':sha(checkpoint),'yolo_sha256':sha(CHECKPOINT),'train_ids':[r['image_id'] for r in rows]}
    protocol_file=folder/'protocol_BEFORE_RUN.json'
    if protocol_file.exists():
        if json.loads(protocol_file.read_text())!=json.loads(json.dumps(protocol)):raise ValueError('Resume protocol mismatch')
    else:protocol_file.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    import tensorflow as tf
    from main import recognize_characters
    cnn=tf.keras.models.load_model(checkpoint,compile=False)
    records=[]
    for r in rows:
        completed=folder/(r['image_id'][:12]+'_records.json')
        if completed.exists():
            records.extend(json.loads(completed.read_text(encoding='utf-8')));continue
        current=[]
        image=cv2.imread(str(ROOT/r['image_path']));px,py,_,_=r['plate_bbox']
        gt=[[px+c['bbox'][0],py+c['bbox'][1],c['bbox'][2],c['bbox'][3]] for c in r['characters']]
        detections=detect(image,str(CHECKPOINT));best=max(detections,key=lambda p:p[2]) if detections else None
        for kind in ('gt','yolo'):
            box=r['plate_bbox'] if kind=='gt' else padded_box(best[1],.15,image.shape) if best else None
            for variant in VARIANTS:
                start=time.perf_counter();pred='';boxes=[];hits=0
                if box:
                    x,y,w,h=box;crop=image[y:y+h,x:x+w]
                    seg,trace=segment_v2(crop,variant);pred=recognize_characters([c for c,_ in seg],'cnn',{'cnn':cnn})
                    boxes=[[x+b[0],y+b[1],b[2],b[3]] for _,b in seg]
                    # Scoring mask/GT intersection after inference, never protection input.
                    original=trace.get('original_binary',trace['binary'])
                    mask=trace['frame_mask']
                    for gx,gy,gw,gh in gt:
                        lx,ty=max(0,gx-x),max(0,gy-y);rx,by=min(w,gx+gw-x),min(h,gy+gh-y)
                        if rx>lx and by>ty:hits+=int(((original[ty:by,lx:rx]>0)&(mask[ty:by,lx:rx]>0)).sum())
                    out=folder/r['image_id'][:12];out.mkdir(exist_ok=True)
                    for name in ('binary','frame_mask','protected_mask'):
                        cv2.imwrite(str(out/f'{kind}_{variant}_{name}.png'),trace[name])
                    overlay=crop.copy()
                    for gx,gy,gw,gh in gt:cv2.rectangle(overlay,(gx-x,gy-y),(gx+gw-x,gy+gh-y),(0,255,0),1)
                    for i,(_,b) in enumerate(seg):
                        cx,cy,cw,ch=b;cv2.rectangle(overlay,(cx,cy),(cx+cw,cy+ch),(0,0,255),1)
                    cv2.imwrite(str(out/f'{kind}_{variant}_overlay.png'),overlay)
                item={'plate_id':r['image_id'],'image':r['image_path'],'group':r['source_group'],'crop':kind,'variant':variant,
                                'prediction':pred,'truth':normalized_text(r['plate_text']),'matching':match_boxes(boxes,gt),
                                'mask_hits_reviewed_ink_pixels':hits,'seconds':time.perf_counter()-start}
                records.append(item);current.append(item)
        completed.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
        print('COMPLETED',r['image_path'],flush=True)
    summary={kind:{v:aggregate([r for r in records if r['crop']==kind and r['variant']==v]) for v in VARIANTS} for kind in ('gt','yolo')}
    selected=max(VARIANTS,key=lambda v:(summary['yolo'][v]['exact'],-summary['yolo'][v]['cer'],summary['yolo'][v]['recall'],summary['yolo'][v]['f1']))
    result={'protocol':protocol,'summary':summary,'selected_train_only':selected,'records':records,'production_changed':False}
    (folder/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    table=[{'crop':kind,'variant':v,**m} for kind,variants in summary.items() for v,m in variants.items()]
    with (folder/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print('OUTPUT',folder,'SELECTED',selected);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()