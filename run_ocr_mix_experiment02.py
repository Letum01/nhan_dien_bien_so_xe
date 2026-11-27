"""Approved paired CNN exploration, no production changes or test fitting."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3

import cv2
import joblib
import numpy as np
import tensorflow as tf

from run_ocr_mix_experiment01 import preflight,metrics
from ocr_mix_experiment import fit,sha
from dataset import CHARACTER_CLASSES
from ensemble_recognition import vote
from main import recognize_characters
from segmentation import segment_characters

ROOT=Path(__file__).resolve().parent
BUNDLE=ROOT/'experiments/ocr_mix_bundle_20261125'
FIRST=ROOT/'experiments/ocr_mix_experiment01_20261125T072726619780Z'


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--resume-experiment',type=Path);args=parser.parse_args()
    meta,arrays,gate=preflight()
    synthetic=json.loads((ROOT/'configs/ocr_synthetic_only.json').read_text())
    mixed=json.loads((ROOT/'configs/ocr_mix_cnn.json').read_text())
    fields=('seed','epochs','batch','learning_rate')
    if any(synthetic[k]!=mixed[k] for k in fields):raise ValueError('CNN paired training budget mismatch')
    folder=args.resume_experiment.resolve() if args.resume_experiment else ROOT/'experiments'/('ocr_mix_experiment02_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if not args.resume_experiment:
        folder.mkdir()
        (folder/'preflight.json').write_text(json.dumps(gate,indent=2),encoding='utf-8')
        for filename in ('annotations.sqlite3','ocr_glyphs.sqlite3'):
            with sqlite3.connect(ROOT/'data'/filename) as source,sqlite3.connect(folder/filename) as target:source.backup(target)
    elif json.loads((folder/'preflight.json').read_text())['manifest_sha256']!=gate['manifest_sha256']:
        raise ValueError('Resume dataset changed')
    valrows=[r for r in meta['manifest'] if r['origin']=='reviewed_real' and r['split']=='validation']
    chars=[x.squeeze() for x in arrays['validation_x']]
    expected=[CHARACTER_CLASSES[int(x)] for x in arrays['validation_y']]
    results={};models={};contracts={}
    for name,config in [('cnn_synthetic',synthetic),('cnn_mix',mixed)]:
        run=folder/name
        if run.exists():
            saved_config=json.loads((run/'run_config.json').read_text())
            if saved_config['config']!=config:raise ValueError('Existing run config mismatch')
            trained=json.loads((run/'validation.json').read_text())
        else:trained=fit(BUNDLE,run,'cnn',config)
        loaded=tf.keras.models.load_model(run/'cnn_model.keras',compile=False)
        if loaded.input_shape!=(None,32,32,1) or loaded.output_shape!=(None,36):raise ValueError('Reloaded CNN shape mismatch')
        predicted=recognize_characters(chars,'cnn',{'cnn':loaded})
        if list(predicted)!=trained['predictions']:raise ValueError('Checkpoint reload changed predictions')
        results[name]={**metrics(expected,predicted,valrows),'run_dir':str(run),'total_seconds':trained['seconds'],
                       'unseen_real_train_classes':trained['classes_without_real_train']}
        models[name]=loaded;contracts[name]=json.loads((run/'cnn_contract.json').read_text())
        print('RESULT',name,results[name]['correct'],'/57',flush=True)
    if contracts['cnn_synthetic']['initial_weight_sha256']!=contracts['cnn_mix']['initial_weight_sha256']:
        raise ValueError('Paired CNN initial weights differ')
    baseline=tf.keras.models.load_model(ROOT/'saved_models/cnn_model.keras',compile=False)
    results['cnn_old_baseline']=metrics(expected,recognize_characters(chars,'cnn',{'cnn':baseline}),valrows)
    candidates={'svm':joblib.load(FIRST/'svm_mix/svm_model.joblib'),
                'rf':joblib.load(FIRST/'rf_mix/rf_model.joblib'),
                'scaler':joblib.load(FIRST/'svm_mix/hog_scaler.joblib'),'cnn':models['cnn_mix']}
    members={name:recognize_characters(chars,name,candidates) for name in ('svm','rf','cnn')}
    for name in ('svm','rf'):
        if candidates[name].classes_.tolist()!=list(range(36)):raise ValueError('Class order mismatch')
        results[name+'_mix']=metrics(expected,members[name],valrows)
    ensemble=vote(members)
    results['majority_candidates']={**metrics(expected,ensemble.raw_text,valrows),'uncertain_positions':ensemble.uncertain_positions}
    originals=json.loads((Path(meta['real_source'])/'ocr_revisions.json').read_text(encoding='utf-8'))
    plate_results=[]
    for row in originals:
        if row['split']!='val':continue
        image=cv2.imread(str(ROOT/row['image_path']));x,y,w,h=row['plate_bbox'];crop=image[y:y+h,x:x+w]
        trace={};seg=segment_characters(crop,trace=trace)
        predicted={name:recognize_characters([ch for ch,_ in seg],name,candidates) for name in ('svm','rf','cnn')}
        voted=vote(predicted)
        # Exact text agreement is diagnostic for a fixed REVIEWED GT crop, not
        # YOLO end-to-end or independent final plate accuracy.
        from glyph_review import normalized_text
        truth=normalized_text(row['plate_text'])
        matches=[]
        from localization import _box_iou
        for i,(_,box) in enumerate(seg):
            ranked=sorted([(_box_iou(box,c['bbox']),j) for j,c in enumerate(row['characters'])],reverse=True)
            matches.append({'segment_index':i,'bbox':box,'best_reviewed_glyph_iou':ranked[0][0] if ranked else 0,
                            'reviewed_index':ranked[0][1] if ranked else None})
        item={'image':row['image_path'],'plate_id':row['image_id'],'source_group':row['source_group'],
              'reviewed_text':truth,'reviewed_glyphs':len(row['characters']),'automatic_glyphs':len(seg),
              'automatic_predictions':predicted,'ensemble_raw':voted.raw_text,'ensemble_status':voted.status,
              'text_agreement_on_gt_crop':{name:text==truth for name,text in predicted.items()},
              'segmentation_matches':matches,'crop_scope':'Reviewed GT crop only, NOT YOLO end-to-end accuracy'}
        plate_results.append(item)
        output=folder/'segmentation_vs_reviewed';output.mkdir(exist_ok=True)
        overlay=crop.copy()
        for c in row['characters']:
            cx,cy,cw,ch=c['bbox'];cv2.rectangle(overlay,(cx,cy),(cx+cw,cy+ch),(0,255,0),1)
        for i,(_,box) in enumerate(seg):
            cx,cy,cw,ch=box;cv2.rectangle(overlay,(cx,cy),(cx+cw,cy+ch),(0,0,255),1)
            cv2.putText(overlay,str(i),(cx,max(10,cy-2)),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,255),1)
        cv2.imwrite(str(output/(row['image_id'][:12]+'.png')),overlay)
    integrity={}
    for filename in ('annotations.sqlite3','ocr_glyphs.sqlite3'):
        with sqlite3.connect(folder/filename) as old,sqlite3.connect(ROOT/'data'/filename) as now:
            integrity[filename]={t:old.execute('SELECT * FROM '+t+' ORDER BY 1').fetchall()==now.execute('SELECT * FROM '+t+' ORDER BY 1').fetchall() for t in ('revisions','drafts')}
            if not all(integrity[filename].values()):raise ValueError('Source annotations changed during experiment')
    for name,digest in meta['baseline_sha256'].items():
        if sha(ROOT/'saved_models'/name)!=digest:raise ValueError('Baseline checkpoint changed')
    report={'experiment':str(folder),'preflight':gate,'contracts':contracts,'results':results,
            'segmentation_vs_reviewed':plate_results,'source_integrity':integrity,
            'members':members,'expected':expected,'validation_rows':valrows,
            'devices':[str(d) for d in tf.config.list_physical_devices()], 'tensorflow':tf.__version__,
            'test':'NONE_NOT_EVALUATED','scope':'57 reviewed development glyphs from7plates, not independent final test. Val used for stopping/selection only.',
            'products_or_threshold_modified':False,'soft_weighted_voting':'NOT_ENABLED'}
    (folder/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'reports/OCR_MIX_EXPERIMENT_02.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('EXPERIMENT',folder,flush=True)


if __name__=='__main__':main()