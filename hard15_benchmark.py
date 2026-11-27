"""Milestone-gated hard-case benchmark; protocol frozen before new evaluation.

prepare: pin configuration and weights without evaluation.
run: requires the first 15 frozen hard queue identities human-reviewed. Audits
all reviewed data in new snapshot. Never tunes parameters, writes labels or fits.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import cv2
import numpy as np

from annotation_store import AnnotationStore
from benchmark_crop_padding import padded_box
from glyph_review import ROOT, normalized_text, validate_glyph_plate, plates_from_detection
from ocr_hard_review import load_hard_queue, QUEUE
from segmentation_benchmark import experimental_segment, match_boxes

PROTOCOL=ROOT/'reports/hard15_protocol.json'
YOLO=ROOT/'experiments/yolo_round2_splitB_run01/weights/best.pt'
CNN=ROOT/'experiments/ocr_mix_experiment02_20261125T080339560263Z/cnn_mix/cnn_model.keras'
PREVIOUS=ROOT/'reports/glyph_audit_20261125T095817713548Z/dataset/ocr_revisions.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    if PROTOCOL.exists():raise FileExistsError('Protocol already frozen; never overwrite after viewing results')
    queue=json.loads(QUEUE.read_text(encoding='utf-8'))['candidates']
    protocol={'frozen_utc':datetime.now(timezone.utc).isoformat(),'queue':str(QUEUE),'queue_sha256':sha(QUEUE),
              'plate_ids':[r['image_id'] for r in sorted(queue,key=lambda r:r['rank'])[:15]],
              'crop_policies':{'gt':'Reviewed detection bbox, no padding',
                               'yolo':'highest score >=.4, padding15% per side of original xywh, clip image; GT never used to select detection'},
              'padding_fraction':.15,'detector_threshold':.4,'matching_iou':.5,
              'segmentation_variants':['baseline','border_components'],
              'matching':'maximum-cardinality valid one-to-one at IoU>=.5 then total IoU',
              'weights_sha256':{str(YOLO):sha(YOLO),str(CNN):sha(CNN)},
              'code_sha256':{name:sha(ROOT/name) for name in ('segmentation.py','segmentation_benchmark.py','benchmark_crop_padding.py','yolo_detector.py','yolo_ui_worker.py','yolo_ui_bridge.py')},
              'previous35_glyph_revisions':str(PREVIOUS),'previous35_sha256':sha(PREVIOUS),
              'selection':'NO algorithm selection or threshold tuning in this benchmark',
              'scope':'TRAIN hard-case development; not final independent accuracy. Previously ranked/visually inspected examples are not unseen.'}
    if len(set(protocol['plate_ids']))!=15:raise ValueError('Need 15 unique frozen IDs')
    PROTOCOL.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FROZEN',PROTOCOL)


def check_milestone(protocol,detection_store,ocr_store):
    hard=load_hard_queue(detection_store,ocr_store)
    if hard['errors']:raise ValueError('Hard queue parent errors: '+str(hard['errors']))
    reviewed={r['image_id']:r for r in hard['completed']}
    missing=[i for i in protocol['plate_ids'] if i not in reviewed]
    if missing:raise ValueError(f'Milestone BLOCKED: {len(missing)} of frozen first15 hard plates still not Glyph reviewed')
    return [reviewed[i] for i in protocol['plate_ids']]


def aggregate(rows):
    tp=sum(r['segmentation']['tp'] for r in rows);fp=sum(r['segmentation']['fp'] for r in rows);fn=sum(r['segmentation']['fn'] for r in rows)
    matches=[m for r in rows for m in r['segmentation']['matches']]
    return {'plates':len(rows),'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'mean_matched_iou':float(np.mean([m['iou'] for m in matches])) if matches else None,
            'exact_text':sum(r['exact_text'] for r in rows),
            'full_match_order':sum(r['segmentation']['complete_order_correct'] for r in rows),
            'matched_classifier_correct':sum(r['matched_classifier_correct'] for r in rows),
            'matched_classifier_samples':sum(len(r['segmentation']['matches']) for r in rows),
            'full_crop_coverage':sum(r['covered_glyphs']==r['gt_glyph_count'] for r in rows)}


def execute():
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    for path,digest in protocol['weights_sha256'].items():
        if sha(path)!=digest:raise ValueError('Checkpoint changed: '+path)
    for name,digest in protocol['code_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen preprocessing/detector implementation changed: '+name)
    if sha(QUEUE)!=protocol['queue_sha256'] or sha(PREVIOUS)!=protocol['previous35_sha256']:
        raise ValueError('Frozen queue/reference snapshot changed')
    detections=AnnotationStore(ROOT/'data/annotations.sqlite3');ocr=AnnotationStore(ROOT/'data/ocr_glyphs.sqlite3')
    selected=check_milestone(protocol,detections,ocr) # BEFORE audit or model inference
    from dataset_validation import check_split_leakage
    corpus=[r for r in ocr.all_latest().values() if r.get('review_status')=='reviewed']
    parents={r['image_id']:r for r in plates_from_detection(detections)}
    for row in corpus:
        parent=parents.get(row['image_id'])
        if not parent or any(row[k]!=parent[k] for k in ('source_image_id','detection_revision','plate_bbox','source_group','split')):
            raise ValueError('Parent changed')
        validate_glyph_plate(row,row['plate_bbox'][2],row['plate_bbox'][3])
    check_split_leakage(corpus,ROOT)
    # Verify no plate was in the completed segmentation policy search datasets.
    evidence=[]
    for pattern in ('segmentation_benchmark_*/benchmark.json','crop_padding_*/results.json','segmentation_train_ablation_*/results.json'):
        for path in (ROOT/'reports').glob(pattern):
            old=json.loads(path.read_text(encoding='utf-8'))
            records=old.get('per_plate',old.get('records',[]))
            ids={r.get('plate_id') for r in records}
            overlap=ids&set(protocol['plate_ids'])
            evidence.append({'artifact':str(path),'overlap':sorted(overlap)})
            if overlap:raise ValueError('Hard set already used in policy experiments; independence needs review: '+str(path))
    from audit_glyph_reviews import main as audit
    folder,audit_report=audit(ocr.path,detections.path)
    if audit_report['export_validator'].get('status')!='PASS':raise ValueError('Full corpus export validator did not pass')
    previous=json.loads(PREVIOUS.read_text(encoding='utf-8'));old_ids={r['image_id'] for r in previous}
    new5=[r for r in selected if r['image_id'] not in old_ids]
    create_new_contact(selected,new5,folder)
    (folder/'hard15_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    import tensorflow as tf
    from main import recognize_characters
    from yolo_ui_bridge import detect
    cnn=tf.keras.models.load_model(CNN,compile=False)
    if cnn.input_shape!=(None,32,32,1) or cnn.output_shape!=(None,36):raise ValueError('CNN shape mismatch')
    records=[]
    for row in selected:
        image=cv2.imread(str(ROOT/row['image_path']));px,py,pw,ph=row['plate_bbox']
        truth=[[px+c['bbox'][0],py+c['bbox'][1],c['bbox'][2],c['bbox'][3]] for c in row['characters']]
        source=folder/'hard15'/row['image_id'][:12];source.mkdir(parents=True)
        proposals=detect(image,str(YOLO))
        best=max(proposals,key=lambda p:p[2]) if proposals else None
        for crop_type in ('gt','yolo'):
            cropbox=row['plate_bbox'] if crop_type=='gt' else padded_box(best[1],.15,image.shape) if best else None
            for variant in protocol['segmentation_variants']:
                start=time.perf_counter();segments=[];predicted='';trace={};boxes=[]
                if cropbox:
                    x,y,w,h=cropbox;crop=image[y:y+h,x:x+w]
                    segments,trace=experimental_segment(crop,variant)
                    predicted=recognize_characters([c for c,_ in segments],'cnn',{'cnn':cnn})
                    boxes=[[x+b[0],y+b[1],b[2],b[3]] for _,b in segments]
                    cv2.imwrite(str(source/f'{crop_type}_{variant}_crop.png'),crop)
                    cv2.imwrite(str(source/f'{crop_type}_{variant}_binary.png'),trace.pop('binary'))
                score=match_boxes(boxes,truth,.5)
                covered=sum(cropbox is not None and cropbox[0]<=x and cropbox[1]<=y and cropbox[0]+cropbox[2]>=x+w and cropbox[1]+cropbox[3]>=y+h for x,y,w,h in truth)
                record={'image':row['image_path'],'plate_id':row['image_id'],'source_group':row['source_group'],'split':row['split'],
                        'crop':crop_type,'variant':variant,'crop_box':cropbox,'original_yolo_box':best[1] if best else None,
                        'original_yolo_score':best[2] if best else None,'covered_glyphs':covered,'gt_glyph_count':len(truth),
                        'segmentation':score,'prediction':predicted,'reviewed_text':normalized_text(row['plate_text']),
                        'exact_text':predicted==normalized_text(row['plate_text']),
                        'matched_classifier_correct':sum(predicted[m['pred']]==row['characters'][m['gt']]['label'] for m in score['matches']),
                        'classification_errors':[{'pred_index':m['pred'],'gt_index':m['gt'],'predicted':predicted[m['pred']],
                           'expected':row['characters'][m['gt']]['label']} for m in score['matches'] if predicted[m['pred']]!=row['characters'][m['gt']]['label']],
                        'trace':trace,'seconds_crop_segmentation_cnn':time.perf_counter()-start}
                records.append(record)
                overlay=image.copy()
                for x,y,w,h in truth:cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,0),1)
                for i,(x,y,w,h) in enumerate(boxes):
                    cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,0,255),1)
                    cv2.putText(overlay,str(i),(x,max(10,y-2)),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,255),1)
                cv2.imwrite(str(source/f'{crop_type}_{variant}_overlay.jpg'),overlay)
    summary={f'{crop}_{variant}':aggregate([r for r in records if r['crop']==crop and r['variant']==variant])
             for crop in ('gt','yolo') for variant in protocol['segmentation_variants']}
    groups={group:{key:aggregate([r for r in records if r['source_group']==group and f"{r['crop']}_{r['variant']}"==key])
                   for key in summary} for group in sorted({r['source_group'] for r in selected})}
    result={'status':'BENCHMARK_EXECUTED_NO_TUNING','reviewed_corpus':len(corpus),'glyphs':sum(len(r['characters']) for r in corpus),
            'hard_plates':len(selected),'new_since35':len(new5),'protocol':protocol,'policy_history_scan':evidence,
            'summary':summary,'per_group':groups,'records':records,'production_modified':False,'training':False}
    (folder/'hard15_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    table=[{'crop_variant':key,**value} for key,value in summary.items()]
    with (folder/'hard15_summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# Hard15 segmentation benchmark',f'\nActual reviewed corpus{len(corpus)},glyphs{result["glyphs"]};hardplates15,new since35={len(new5)}.',
           '\nFixed protocol, no retuning. GTcrop vs YOLOpad15 separate;CNNMix fixed;IoU.5one-to-one.',
           '\n|Scope|TP|FP|FN|P|R|F1|Exact|Full order|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,m in summary.items():lines.append(f"|{key}|{m['tp']}|{m['fp']}|{m['fn']}|{m['precision']}|{m['recall']}|{m['f1']}|{m['exact_text']}/15|{m['full_match_order']}/15|")
    lines+=['\nThese TRAINsources were ranked/seen during annotation; scan found no prior completed segmentation policy search overlap, '
            'but NOT a blind or final independent test. No class labels supplied to prediction. '
            'Match accuracy must include segmentation FN/FP, not count missing glyphs as correct. '
            'Review per-plate classification errors, split/merge hypotheses and overlays. No production promotion.']
    (folder/'HARD15_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('REPORT',folder);print(json.dumps(summary,indent=2))


def create_new_contact(selected,new_rows,folder):
    tiles=[];index_rows=[]
    for row in new_rows:
        image=cv2.imread(str(ROOT/row['image_path']));px,py,_,_=row['plate_bbox']
        for i,c in enumerate(row['characters']):
            x,y,w,h=c['bbox'];glyph=image[py+y:py+y+h,px+x:px+x+w]
            tile=np.full((120,108,3),240,np.uint8);scale=min(100/w,85/h)
            glyph=cv2.resize(glyph,(max(1,int(w*scale)),max(1,int(h*scale))))
            tile[:glyph.shape[0],:glyph.shape[1]]=glyph
            cv2.putText(tile,f"{len(tiles)}:{c['label']}",(3,105),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
            index_rows.append({'sheet_index':len(tiles),'image':row['image_path'],'glyph_index':i,'label':c['label'],'bbox':c['bbox']});tiles.append(tile)
    if tiles:
        sheet=np.full((((len(tiles)+8)//9)*120,9*108,3),240,np.uint8)
        for i,tile in enumerate(tiles):sheet[i//9*120:(i//9+1)*120,i%9*108:(i%9+1)*108]=tile
        cv2.imwrite(str(folder/'new_since35_contact.jpg'),sheet)
    (folder/'new_since35_contact_index.json').write_text(json.dumps(index_rows,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.action=='prepare' else execute()