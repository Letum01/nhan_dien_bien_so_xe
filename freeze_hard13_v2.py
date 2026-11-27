"""Freeze remaining hard13 AFTER development; no pending image pixels opened."""
import hashlib
import json
from pathlib import Path

from annotation_store import AnnotationStore
from glyph_review import ROOT
from ocr_hard_review import load_hard_queue


def main():
    folders=sorted((ROOT/'reports').glob('segmentation_v2_*/results.json'))
    if not folders:raise ValueError('Development must complete before freezing candidate')
    result=json.loads(folders[-1].read_text(encoding='utf-8'))
    d=AnnotationStore(ROOT/'data/annotations.sqlite3');o=AnnotationStore(ROOT/'data/ocr_glyphs.sqlite3')
    hard=load_hard_queue(d,o)
    if hard['errors'] or len(hard['pending'])!=13:raise ValueError('Expected13validpending')
    ids={r['image_id'] for r in hard['pending']}
    cnn_manifest=json.loads((ROOT/'experiments/ocr_mix_bundle_20261125/manifest.json').read_text(encoding='utf-8'))
    cnn_sources={r['source_image_id'] for r in cnn_manifest['manifest'] if r['origin']=='reviewed_real' and r['split']=='train'}
    yolo_rows=json.loads((ROOT/'reports/round2_gate_20261124T154153249700Z/dataset/manifest.json').read_text(encoding='utf-8'))['rows']
    yolo_train={r['image_id'] for r in yolo_rows if r['split']=='train'}
    sources=[]
    for row in hard['pending']:
        sources.append({k:row[k] for k in ('image_id','source_image_id','image_path','plate_bbox','detection_revision','source_group','split')})
        sources[-1].update(cnn_mix_real_training_source=row['source_image_id'] in cnn_sources,
                           yolo_round2_training_source=row['source_image_id'] in yolo_train,
                           development_history='Hardness metrics/crop proposals inspected; no glyph GT available during V2, no pending crop tuning in V2 loop')
    candidate=result['selected_train_only']
    if candidate in ('baseline','border_components'):
        status='NO_V2_CANDIDATE_SELECTED_keep_frozen_control'
    else:status='EXPERIMENTAL_CANDIDATE_ONLY'
    files=['segmentation_v2.py','segmentation.py','segmentation_benchmark.py','benchmark_crop_padding.py']
    protocol={'status':status,'development_results':str(folders[-1]),'candidate':candidate,
              'variants':list(dict.fromkeys(['baseline','border_components',candidate])),
              'config':result['protocol']['config'],'code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
              'checkpoints_sha256':{str(ROOT/'experiments/ocr_mix_experiment02_20261125T080339560263Z/cnn_mix/cnn_model.keras'):result['protocol']['cnn_sha256'],
                                    str(ROOT/'experiments/yolo_round2_splitB_run01/weights/best.pt'):result['protocol']['yolo_sha256']},
              'iou':.5,'crop':'GT no padding, YOLO top score>=.4 pad15%eachside clipped',
              'selection':'No retuning on hard13','plates':sources,
              'not_independent_test':'TRAIN sources and YOLO development exposure; classifier exposure per-row explicit. No broad generalization claims.'}
    path=ROOT/'reports/hard13_v2_protocol.json'
    if path.exists():raise FileExistsError('Never overwrite frozen protocol')
    path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FROZEN',path,candidate)


if __name__=='__main__':main()