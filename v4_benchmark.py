"""Evaluate one change at a time on reviewed development images only."""
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from benchmark_yolo_character import score, summarize
from dual_yolo_client import Worker, PYTHON, CHECKPOINT, CHARACTER
from dual_yolo_pipeline import MixClassifiers
from final_ocr_dataset import ROOT, check_protocol, read
from glyph_review import CLASSES
from v4_geometry import rectify, mapped_box, decode
from yolo_character_pipeline import normalize_glyph, order_boxes
from yolo_character_dataset import validate, sha


def main():
    check_protocol(); root=ROOT/'datasets/yolo_character_round1'; validate(root)
    out=ROOT/'experiments/alpr_v4'; output=out/'benchmark.json'
    if output.exists():raise FileExistsError(output)
    refs={(r['id'],r['scope']):r for r in read(ROOT/'experiments/yolo_character_round1/comparison/results.json')['records'] if r['method']=='character_yolo'}
    samples=[r['review'] for r in read(root/'manifest.json')['rows'] if r['review']['split']=='val']
    models=MixClassifiers(); results={}
    for arm in ('baseline','mixed','finetune','deskew','perspective'):
        weights=CHARACTER if arm in ('baseline','deskew','perspective') else out/arm/'weights/best.pt'
        if not weights.is_file():continue
        if arm in ('mixed','finetune') and sha(weights)!=read(out/arm/'receipt.json')['best_sha256']:raise ValueError('Weight hash mismatch')
        records=[]
        with Worker(command=[str(PYTHON),'-u',str(ROOT/'dual_yolo_worker.py'),str(CHECKPOINT),str(weights)]) as worker:
            for row in samples:
                im=cv2.imread(str(ROOT/row['image_path']))
                for scope in ('gt','yolo'):
                    ref=refs[row['image_id'],scope]; box=ref['crop_box']; start=perf_counter(); reason='no_crop'; H=np.eye(3); pred=[]; glyphs=[]; peak=None
                    if box:
                        x,y,w,h=box; roi=im[y:y+h,x:x+w]
                        roi,H,reason=rectify(roi,arm if arm in ('deskew','perspective') else 'raw')
                        response=worker.request('character',roi);peak=response['peak_reserved'];pred=order_boxes(response['rows'])
                        glyphs=[normalize_glyph(roi[b:b+d,a:a+c]) for a,b,c,d in [r['bbox'] for r in pred]]
                    p=np.asarray(models.cnn(np.asarray(glyphs)[...,None],training=False)) if glyphs else np.empty((0,36))
                    text=''.join(CLASSES[i] for i in p.argmax(1)); decoded=decode(p,CLASSES)
                    # Match in original image coordinates after inverse mapping (AABB).
                    boxes=[mapped_box(r['bbox'],np.linalg.inv(H)) for r in pred]
                    record=score({'glyph_boxes':boxes,'text':dict.fromkeys(('cnn','svm','rf','ensemble'),text)},box,ref['gt_boxes'],row)
                    record['ocr']={k:v for k,v in record['ocr'].items() if k=='cnn'}
                    record.update(image=row['image_path'],scope=scope,seconds=perf_counter()-start,rectification=reason,
                                  decoded=decoded,peak_reserved=peak,baseline_boxes_equal=record['pred_boxes']==ref['pred_boxes'])
                    records.append(record)
        summary={}
        for scope in ('gt','yolo'):
            rs=[r for r in records if r['scope']==scope]
            summary[scope]=summarize([{**r,'ocr':{**dict.fromkeys(('svm','rf','ensemble'),{'edits':0,'exact':False,'matched_correct':0}),**r['ocr']}} for r in rs]);summary[scope]['ocr']={'cnn':summary[scope]['ocr']['cnn']}
        results[arm]={'weights_sha256':sha(weights),'records':records,'summary':summary}
        (out/'benchmark_progress.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(arm,summary,flush=True)
    output.write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()