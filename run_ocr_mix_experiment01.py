"""Approved SVM/RF 2x2 experiment with strict frozen-bundle provenance gate."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import cv2
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from annotation_store import AnnotationStore
from dataset import CHARACTER_CLASSES, extract_hog_features, generate_synthetic_character_dataset
from dataset_validation import check_split_leakage
from glyph_review import plates_from_detection
from main import recognize_characters
from ocr_mix_experiment import fit, load_bundle, sha
from validate_glyph_dataset import validate

ROOT=Path(__file__).resolve().parent
BUNDLE=ROOT/'experiments/ocr_mix_bundle_20261125'


def preflight():
    meta,arrays=load_bundle(BUNDLE)
    if [len(arrays[k]) for k in ('synthetic_y','train_y','validation_y')]!=[3600,147,57]:
        raise ValueError('Approved dataset sizes changed')
    if sha(ROOT/'dataset.py')!=meta['generator_source_sha256']:
        raise ValueError('Synthetic generator source changed')
    real=Path(meta['real_source'])
    if sha(real/'characters.csv')!=meta['real_manifest_sha256']:raise ValueError('Real manifest changed')
    validated=validate(real)
    rows=list(csv.DictReader((real/'characters.csv').open(encoding='utf-8-sig',newline='')))
    originals=json.loads((real/'ocr_revisions.json').read_text(encoding='utf-8'))
    ocr=AnnotationStore(ROOT/'data/ocr_glyphs.sqlite3').all_latest()
    parents={r['image_id']:r for r in plates_from_detection(AnnotationStore(ROOT/'data/annotations.sqlite3'))}
    for row in originals:
        if ocr.get(row['image_id'])!=row:raise ValueError('Live OCR reviewed revision differs from approved export')
        parent=parents.get(row['image_id'])
        if not parent or any(parent[k]!=row[k] for k in ('detection_revision','plate_bbox','source_group','split','source_image_id')):
            raise ValueError('Detection parent differs')
        if sha(ROOT/row['image_path'])!=row['source_image_id']:raise ValueError('Source image changed')
    check_split_leakage(originals,ROOT)
    group_splits={};pixel_splits={}
    for sample in meta['manifest']:
        prefix='synthetic' if sample['origin']=='generated_synthetic' else sample['split']
        image=arrays[prefix+'_x'][sample['array_index']]
        label=int(arrays[prefix+'_y'][sample['array_index']])
        if hashlib.sha256(image.tobytes()).hexdigest()!=sample['pixel_sha256'] or CHARACTER_CLASSES[label]!=sample['class']:
            raise ValueError('Per-sample pixels/labels differ from manifest')
        digest=sample['pixel_sha256']
        if digest in pixel_splits and pixel_splits[digest]!=sample['split']:raise ValueError('Pixel leakage')
        pixel_splits[digest]=sample['split']
        if sample['origin']=='reviewed_real':
            if sha(sample['source_path'])!=sample['source_file_sha256']:raise ValueError('Exported glyph changed')
            if sample['group'] in group_splits and group_splits[sample['group']]!=sample['split']:raise ValueError('Group leakage')
            group_splits[sample['group']]=sample['split']
        elif sample.get('label_source')!='rendered character instruction, not OCR output':raise ValueError('Synthetic label provenance invalid')
    state=np.random.get_state()
    try:sx,sy=generate_synthetic_character_dataset(100)
    finally:np.random.set_state(state)
    if not np.array_equal(sx,arrays['synthetic_x']) or not np.array_equal(sy,arrays['synthetic_y']):
        raise ValueError('Synthetic regeneration not identical')
    for name,digest in meta['baseline_sha256'].items():
        if sha(ROOT/'saved_models'/name)!=digest:raise ValueError('Baseline weight changed')
    return meta,arrays,{'status':'PASS','real_validator':validated,'generator_regeneration':'pixel/label identical',
                        'live_reviewed_parent_check':'PASS','source_group_hash_phash_leakage':'PASS',
                        'manifest_sha256':sha(BUNDLE/'manifest.json'),'synthetic':3600,'real_train':147,'real_validation':57,
                        'validation_sources':7,'historical_baseline_provenance':'UNVERIFIED; descriptive development comparison only'}


def metrics(expected,predicted,valrows):
    expected=np.array(list(expected));predicted=np.array(list(predicted))
    if len(expected)!=len(predicted):raise ValueError('Different glyph counts in metric input')
    report=classification_report(expected,predicted,labels=CHARACTER_CLASSES,output_dict=True,zero_division=0)
    supported=[c for c in CHARACTER_CLASSES if report[c]['support']>0]
    per_plate=defaultdict(list);per_group=defaultdict(list)
    for i,r in enumerate(valrows):per_plate[r['plate_id']].append(i);per_group[r['group']].append(i)
    def subset(indices):return {'correct':int((expected[indices]==predicted[indices]).sum()),'samples':len(indices),'accuracy':float(np.mean(expected[indices]==predicted[indices]))}
    return {'correct':int((expected==predicted).sum()),'samples':len(expected),
            'accuracy':float(np.mean(expected==predicted)),
            'macro_recall_supported':float(np.mean([report[c]['recall'] for c in supported])),
            'supported_class_count':len(supported),'I_predictions':int((predicted=='I').sum()),
            'histogram':dict(Counter(predicted.tolist())),'per_class':report,
            'confusion_matrix':confusion_matrix(expected,predicted,labels=CHARACTER_CLASSES).tolist(),
            'matrix_class_order':CHARACTER_CLASSES,
            'per_plate':{k:subset(v) for k,v in per_plate.items()},'per_group':{k:subset(v) for k,v in per_group.items()},
            'predictions':predicted.tolist()}


def main():
    meta,arrays,gate=preflight()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=ROOT/'experiments'/('ocr_mix_experiment01_'+stamp);folder.mkdir(exist_ok=False)
    (folder/'preflight.json').write_text(json.dumps(gate,indent=2),encoding='utf-8')
    for database in ('annotations.sqlite3','ocr_glyphs.sqlite3'):
        source=ROOT/'data'/database
        with sqlite3.connect(f'file:{source.as_posix()}?mode=ro',uri=True) as c,sqlite3.connect(folder/database) as target:c.backup(target)
    configs={'svm_synthetic':('svm','ocr_synthetic_only.json'),'svm_mix':('svm','ocr_mix_svm.json'),
             'rf_synthetic':('rf','ocr_synthetic_only.json'),'rf_mix':('rf','ocr_mix_rf.json')}
    valrows=[r for r in meta['manifest'] if r['origin']=='reviewed_real' and r['split']=='validation']
    chars=[image.squeeze() for image in arrays['validation_x']]
    features=np.array([extract_hog_features(c) for c in chars],np.float32)
    expected=[CHARACTER_CLASSES[int(label)] for label in arrays['validation_y']]
    results={};loaded_checks={}
    for run,(name,filename) in configs.items():
        config=json.loads((ROOT/'configs'/filename).read_text(encoding='utf-8'))
        run_dir=folder/run
        trained=fit(BUNDLE,run_dir,name,config)
        model=joblib.load(run_dir/(name+'_model.joblib'))
        assert model.classes_.tolist()==list(range(36))
        if name=='svm':
            scaler=joblib.load(run_dir/'hog_scaler.joblib')
            assert scaler.n_features_in_==324 and scaler.n_samples_seen_==(3600 if run.endswith('synthetic') else 3747)
            predicted=model.predict(scaler.transform(features))
        else:predicted=model.predict(features)
        labels=[CHARACTER_CLASSES[int(i)] for i in predicted]
        if labels!=trained['predictions']:raise ValueError('Reloaded checkpoint predictions changed')
        results[run]={**metrics(expected,labels,valrows),'run_dir':str(run_dir),'config':config,
                      'fit_timing':json.loads((run_dir/'training_log.json').read_text()),
                      'total_run_seconds':trained['seconds'],'checkpoint_sha256':sha(run_dir/(name+'_model.joblib'))}
        loaded_checks[run]='PASS reload predictions, class mapping, HOG/scaler'
        print(run,results[run]['correct'],'/',len(expected),results[run]['accuracy'],flush=True)
    for name in ('svm','rf'):
        model=joblib.load(ROOT/'saved_models'/(name+'_model.joblib'))
        values=features
        if name=='svm':values=joblib.load(ROOT/'saved_models/hog_scaler.joblib').transform(features)
        predicted=[CHARACTER_CLASSES[int(i)] for i in model.predict(values)]
        results[name+'_old_baseline']={**metrics(expected,predicted,valrows),'checkpoint':str(ROOT/'saved_models'/(name+'_model.joblib')),
                                      'provenance':'Historical training membership not verified'}
    changes={}
    for name in ('svm','rf'):
        synthetic=results[name+'_synthetic'];mix=results[name+'_mix']
        changes[name]={'improved_recall_classes':[c for c in CHARACTER_CLASSES if mix['per_class'][c]['support'] and mix['per_class'][c]['recall']>synthetic['per_class'][c]['recall']],
                       'decreased_recall_classes':[c for c in CHARACTER_CLASSES if mix['per_class'][c]['support'] and mix['per_class'][c]['recall']<synthetic['per_class'][c]['recall']]}
    for database in ('annotations.sqlite3','ocr_glyphs.sqlite3'):
        with sqlite3.connect(folder/database) as before,sqlite3.connect(ROOT/'data'/database) as now:
            for table in ('revisions','drafts'):
                assert before.execute('SELECT * FROM '+table+' ORDER BY 1').fetchall()==now.execute('SELECT * FROM '+table+' ORDER BY 1').fetchall()
    for name,digest in meta['baseline_sha256'].items():assert sha(ROOT/'saved_models'/name)==digest
    report={'experiment':str(folder),'preflight':gate,'results':results,'recall_changes_mix_vs_synthetic':changes,
            'reload_checks':loaded_checks,'validation_rows':valrows,'expected':expected,
            'data_or_baseline_changed':False,'cnn_training':'NOT_REQUESTED_NOT_RUN','test':'ABSENT_NOT_EVALUATED',
            'scope':'57 development glyphs from 7 plates; not independent test accuracy. Synthetic+real comparison conditional on preprocessing/seed/domain weights.'}
    (folder/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'reports/OCR_MIX_EXPERIMENT_01.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('EXPERIMENT',folder)


if __name__=='__main__':main()