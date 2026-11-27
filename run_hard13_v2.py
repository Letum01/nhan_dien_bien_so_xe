"""Milestone and codehash gate, refuses partial hard13 evaluation."""
import hashlib
import json
from pathlib import Path

from annotation_store import AnnotationStore
from glyph_review import ROOT


def check_gate(protocol,reviewed):
    rows={k:r for k,r in reviewed.items() if r.get('review_status')=='reviewed'}
    missing=[r['image_path'] for r in protocol['plates'] if r['image_id'] not in rows]
    if missing:raise ValueError('BLOCKED: hard13 still missing reviewed glyphs: '+str(missing))
    for expected in protocol['plates']:
        current=rows[expected['image_id']]
        for key in ('source_image_id','plate_bbox','source_group','split','detection_revision'):
            if current[key]!=expected[key]:raise ValueError('Hard13parentchanged:'+expected['image_path'])
    return [rows[r['image_id']] for r in protocol['plates']]


def main():
    raw=(ROOT/'reports/hard13_v2_protocol.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!='d810a90581579f0b62f0816493f447ff023c1e7d265e0b5e7beface343a6f363':
        raise ValueError('Frozen hard13 protocol changed')
    protocol=json.loads(raw)
    for key in ('code_sha256','checkpoints_sha256'):
        for name,digest in protocol[key].items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Frozenhashchanged:'+name)
    selected=check_gate(protocol,AnnotationStore(ROOT/'data/ocr_glyphs.sqlite3').all_latest())
    from audit_glyph_reviews import main as audit
    from dataset_validation import check_split_leakage
    from glyph_review import validate_glyph_plate,normalized_text
    from yolo_ui_bridge import detect,CHECKPOINT
    from benchmark_crop_padding import padded_box
    from segmentation_v2 import segment_v2
    from segmentation_benchmark import match_boxes
    from run_segmentation_v2 import aggregate
    import cv2
    import tensorflow as tf
    from main import recognize_characters
    check_split_leakage(selected,ROOT)
    folder,report=audit()
    if report['export_validator'].get('status')!='PASS':raise ValueError('Auditfailed')
    audited={r['image_id']:r for r in json.loads((folder/'dataset/ocr_revisions.json').read_text(encoding='utf-8'))}
    if any(audited.get(r['image_id'])!=r for r in selected):raise ValueError('Review changed during audit')
    cnn_path=next(p for p in protocol['checkpoints_sha256'] if p.endswith('.keras'))
    model=tf.keras.models.load_model(cnn_path,compile=False)
    records=[]
    for row in selected:
        validate_glyph_plate(row,row['plate_bbox'][2],row['plate_bbox'][3])
        image=cv2.imread(str(ROOT/row['image_path']));px,py,_,_=row['plate_bbox']
        truth=[[px+c['bbox'][0],py+c['bbox'][1],c['bbox'][2],c['bbox'][3]] for c in row['characters']]
        detections=detect(image,str(CHECKPOINT));best=max(detections,key=lambda r:r[2]) if detections else None
        for kind in ('gt','yolo'):
            box=row['plate_bbox'] if kind=='gt' else padded_box(best[1],.15,image.shape) if best else None
            for variant in protocol['variants']:
                text='';boxes=[]
                if box:
                    x,y,w,h=box;crop=image[y:y+h,x:x+w]
                    seg,trace=segment_v2(crop,variant)
                    text=recognize_characters([char for char,_ in seg],'cnn',{'cnn':model})
                    boxes=[[x+b[0],y+b[1],b[2],b[3]] for _,b in seg]
                    out=folder/'hard13'/row['image_id'][:12];out.mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(out/f'{kind}_{variant}_binary.png'),trace['binary'])
                    overlay=crop.copy()
                    for gx,gy,gw,gh in truth:cv2.rectangle(overlay,(gx-x,gy-y),(gx+gw-x,gy+gh-y),(0,255,0),1)
                    for _,(cx,cy,cw,ch) in seg:cv2.rectangle(overlay,(cx,cy),(cx+cw,cy+ch),(0,0,255),1)
                    cv2.imwrite(str(out/f'{kind}_{variant}_overlay.png'),overlay)
                matching=match_boxes(boxes,truth,.5)
                covered=sum(bool(box and box[0]<=gx and box[1]<=gy and gx+gw<=box[0]+box[2] and gy+gh<=box[1]+box[3]) for gx,gy,gw,gh in truth)
                records.append({'image':row['image_path'],'plate_id':row['image_id'],'group':row['source_group'],'crop':kind,'variant':variant,
                                'crop_box':box,'covered':covered,'gt_boxes':truth,'prediction_boxes':boxes,
                                'matched_correct':sum(text[m['pred']]==row['characters'][m['gt']]['label'] for m in matching['matches']),
                                'truth':normalized_text(row['plate_text']),'prediction':text,'matching':matching,
                                'mask_hits_reviewed_ink_pixels':0})
    summary={kind:{v:aggregate([r for r in records if r['crop']==kind and r['variant']==v]) for v in protocol['variants']} for kind in ('gt','yolo')}
    (folder/'hard13_v2_results.json').write_text(json.dumps({'protocol':protocol,'summary':summary,'records':records,
        'scope':'TRAIN hard13 evaluation, no retuning, not independent final test'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(folder,json.dumps(summary,indent=2))


if __name__=='__main__':main()