"""Reproducible train-only synthetic/real experiment preparation and opt-in fitting.

prepare never trains. fit writes a NEW run directory; validation is evaluation
and early stopping only. No production artifact loading or promotion changes.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import joblib
import numpy as np

from dataset import CHARACTER_CLASSES, extract_hog_features, generate_synthetic_character_dataset
from validate_glyph_dataset import validate

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contact(images,labels,path):
    indices=[]
    for label in range(36):indices.extend(np.flatnonzero(labels==label)[:3].tolist())
    if not indices:return
    sheet=np.full((((len(indices)+11)//12)*60,12*50,3),235,np.uint8)
    for index,i in enumerate(indices):
        x=index%12*50;y=index//12*60
        sheet[y:y+32,x:x+32]=cv2.cvtColor(images[i].squeeze(),cv2.COLOR_GRAY2BGR)
        cv2.putText(sheet,CHARACTER_CLASSES[int(labels[i])],(x+5,y+53),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
    cv2.imwrite(str(path),sheet)


def prepare(real_dataset,output,samples_per_class=100):
    if samples_per_class<1:raise ValueError('At least one generated sample per class required')
    real_dataset=Path(real_dataset).resolve();output=Path(output).resolve()
    validation=validate(real_dataset)
    if not (real_dataset/'ocr_revisions.json').exists():raise ValueError('Reviewed source provenance required')
    revisions=json.loads((real_dataset/'ocr_revisions.json').read_text(encoding='utf-8'))
    from dataset_validation import check_split_leakage
    check_split_leakage(revisions,ROOT)
    rows=list(csv.DictReader((real_dataset/'characters.csv').open(encoding='utf-8-sig',newline='')))
    buckets={split:[] for split in ('train','validation')}
    for row in rows:
        if row['split'] not in buckets:continue  # test never loaded into this experiment
        image=cv2.imread(str(real_dataset/row['image_path']),0)
        buckets[row['split']].append((image,CHARACTER_CLASSES.index(row['class_label']),row))
    if not all(buckets.values()):raise ValueError('Reviewed real train AND validation required')
    # Generator currently hardcodes seed42. Preserve/restore caller RNG; record source hash.
    state=np.random.get_state()
    try:synthetic,synthetic_labels=generate_synthetic_character_dataset(samples_per_class)
    finally:np.random.set_state(state)
    output.mkdir(parents=True,exist_ok=False)
    arrays={'synthetic_x':synthetic,'synthetic_y':synthetic_labels}
    manifest=[]
    for index,(image,label) in enumerate(zip(synthetic,synthetic_labels)):
        manifest.append({'origin':'generated_synthetic','split':'train','array_index':index,'class':CHARACTER_CLASSES[int(label)],
                         'pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'group':f'generator42_{index}',
                         'label_source':'rendered character instruction, not OCR output','independent_source':False})
    for split,entries in buckets.items():
        arrays[split+'_x']=np.stack([entry[0] for entry in entries])[...,None]
        arrays[split+'_y']=np.array([entry[1] for entry in entries],np.int32)
        for index,(image,label,row) in enumerate(entries):
            manifest.append({'origin':'reviewed_real','split':split,'array_index':index,'class':CHARACTER_CLASSES[label],
                             'pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'group':row['source_group'],
                             'plate_id':row['plate_id'],'source_image_id':row['source_image_id'],
                             'source_path':str(real_dataset/row['image_path']), 'source_file_sha256':sha(real_dataset/row['image_path'])})
    train_pixels={r['pixel_sha256'] for r in manifest if r['split']=='train'}
    overlaps=[r for r in manifest if r['split']=='validation' and r['pixel_sha256'] in train_pixels]
    if overlaps:raise ValueError('Exact pixel overlap train/validation; bundle incomplete, do not use')
    np.savez_compressed(output/'arrays.npz',**arrays)
    metadata={'status':'PREPARED_NOT_TRAINED','seed':42,'generator_seed':42,'samples_per_synthetic_class':samples_per_class,
              'generator_source_sha256':sha(ROOT/'dataset.py'),'class_order':CHARACTER_CLASSES,
              'real_source':str(real_dataset),'real_manifest_sha256':sha(real_dataset/'characters.csv'),
              'source_validator':validation,'array_file_sha256':sha(output/'arrays.npz'),
              'manifest':manifest,'preprocessing':{'shape':[32,32,1],'range':[0,255],'dtype':'uint8',
                  'polarity':'white ink black background','hog':{'orientations':9,'pixels_per_cell':[8,8],'cells_per_block':[2,2],'dimension':324},
                  'cnn_rescaling':'internal 1/255 once','synthetic_threshold':60,'reviewed_normalized_threshold':127,
                  'note':'Known threshold/domain mismatch retained, not assumed equivalent; ablation needs separate experiment'},
              'historical_weights_provenance':'UNVERIFIED; current synthetic generator is reproducible but cannot reconstruct original model train membership without original manifest',
              'versions':{'opencv':cv2.__version__,'numpy':np.__version__},
              'baseline_sha256':{p.name:sha(p) for p in (ROOT/'saved_models').iterdir() if p.suffix in ('.joblib','.keras')}}
    import sklearn,tensorflow as tf
    metadata['versions'].update(sklearn=sklearn.__version__,tensorflow=tf.__version__)
    (output/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    contact(synthetic,synthetic_labels,output/'synthetic_contact.png')
    for split in buckets:contact(arrays[split+'_x'],arrays[split+'_y'],output/(split+'_contact.png'))
    print('PREPARED',output, {k:len(v) for k,v in arrays.items() if k.endswith('_y')})
    return metadata


def load_bundle(bundle):
    bundle=Path(bundle)
    meta=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    if meta['class_order']!=CHARACTER_CLASSES or sha(bundle/'arrays.npz')!=meta['array_file_sha256']:
        raise ValueError('Bundle hash/class mapping mismatch')
    arrays=dict(np.load(bundle/'arrays.npz',allow_pickle=False))
    for prefix in ('synthetic','train','validation'):
        image,labels=arrays[prefix+'_x'],arrays[prefix+'_y']
        if image.dtype!=np.uint8 or image.shape[1:]!=(32,32,1) or len(image)!=len(labels):raise ValueError('Invalid pixel/label contract')
        if not set(np.unique(labels))<=set(range(36)):raise ValueError('Invalid classes')
    return meta,arrays


def fit(bundle,output,model_name,config):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    meta,arrays=load_bundle(bundle)
    config=dict(config)
    output=Path(output).resolve()
    protected=ROOT/'saved_models'
    if output==protected or protected in output.parents or output.exists():raise ValueError('Use NEW experiment outside saved_models')
    variant=config['variant']
    if variant not in ('synthetic_only','synthetic_plus_real'):raise ValueError('Invalid mixture')
    x=arrays['synthetic_x'];y=arrays['synthetic_y']
    weights=np.ones(len(y),np.float32)
    if variant=='synthetic_plus_real':
        # Equal domain TOTAL weight, not fake new independent samples.
        real_weight=float(config.get('real_domain_fraction',.5))
        if not 0<real_weight<1:raise ValueError('Invalid domain fraction')
        weights=np.concatenate([np.full(len(y),(1-real_weight)/len(y)),np.full(len(arrays['train_y']),real_weight/len(arrays['train_y']))])
        weights=weights/weights.mean()
        x=np.concatenate([x,arrays['train_x']]);y=np.concatenate([y,arrays['train_y']])
    vx,vy=arrays['validation_x'],arrays['validation_y']
    output.mkdir(parents=True,exist_ok=False)
    (output/'run_config.json').write_text(json.dumps({'model':model_name,'config':config,'bundle':str(Path(bundle).resolve())},indent=2),encoding='utf-8')
    from sklearn.metrics import classification_report,confusion_matrix
    started=__import__('time').perf_counter()
    if model_name in ('svm','rf'):
        features=np.array([extract_hog_features(image.squeeze()) for image in x],np.float32)
        valfeatures=np.array([extract_hog_features(image.squeeze()) for image in vx],np.float32)
        if model_name=='svm':
            scaler=StandardScaler().fit(features) # train only; no validation/scaler fitting
            joblib.dump(scaler,output/'hog_scaler.joblib')
            model=SVC(C=config.get('C',10),gamma=config.get('gamma','scale'),probability=True,random_state=config['seed'])
            fit_started=time.perf_counter()
            model.fit(scaler.transform(features),y,sample_weight=weights)
            estimator_fit_seconds=time.perf_counter()-fit_started
            predicted=model.predict(scaler.transform(valfeatures))
        else:
            model=RandomForestClassifier(n_estimators=config.get('n_estimators',100),max_depth=config.get('max_depth',20),random_state=config['seed'],n_jobs=-1)
            fit_started=time.perf_counter()
            model.fit(features,y,sample_weight=weights)
            estimator_fit_seconds=time.perf_counter()-fit_started
            predicted=model.predict(valfeatures)
        joblib.dump(model,output/(model_name+'_model.joblib'))
    elif model_name=='cnn':
        import tensorflow as tf
        from models import build_cnn_model
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(config['seed'])
        if config.get('finetune_baseline'):
            model=tf.keras.models.load_model(ROOT/'saved_models/cnn_model.keras',compile=False)
            if model.input_shape!=(None,32,32,1) or model.output_shape!=(None,36):raise ValueError('CNN contract mismatch')
        else:model=build_cnn_model()
        # Train-only augmentation integrated into saved inference-compatible model.
        model=tf.keras.Sequential([tf.keras.layers.Input((32,32,1)),
            tf.keras.layers.RandomRotation(5/360,fill_mode='constant',fill_value=0,seed=config['seed']),
            tf.keras.layers.RandomTranslation(.03,.03,fill_mode='constant',fill_value=0,seed=config['seed']),model])
        model.compile(optimizer=tf.keras.optimizers.Adam(config.get('learning_rate',.001)),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        def rescaling_count(layer):
            if isinstance(layer,tf.keras.layers.Rescaling):
                if not np.isclose(layer.scale,1/255):raise ValueError('Unexpected CNN normalization scale')
                return 1
            return sum(rescaling_count(child) for child in getattr(layer,'layers',[]))
        if rescaling_count(model)!=1:raise ValueError('CNN must normalize exactly once')
        (output/'cnn_contract.json').write_text(json.dumps({'input_shape':list(model.input_shape),'output_shape':list(model.output_shape),
            'rescaling_count':1,'class_order':CHARACTER_CLASSES,'physical_devices':[str(d) for d in tf.config.list_physical_devices()],
            'weight_effect':'sample_weight multiplies per-example training loss; unweighted accuracy metric, not domain-balanced accuracy',
            'initial_weight_sha256':hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()},indent=2),encoding='utf-8')
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True),
                   tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2,factor=.5),
                   tf.keras.callbacks.ModelCheckpoint(str(output/'cnn_best.keras'),monitor='val_loss',save_best_only=True),
                   tf.keras.callbacks.CSVLogger(str(output/'training.csv'))]
        model.fit(x,y,sample_weight=weights,validation_data=(vx,vy),epochs=config.get('epochs',30),batch_size=config.get('batch',16),callbacks=callbacks,verbose=2)
        model.save(output/'cnn_model.keras');predicted=model.predict(vx,verbose=0).argmax(axis=1)
    else:raise ValueError('Unsupported model')
    expected=[CHARACTER_CLASSES[int(i)] for i in vy];observed=[CHARACTER_CLASSES[int(i)] for i in predicted]
    valrows=[r for r in meta['manifest'] if r['origin']=='reviewed_real' and r['split']=='validation']
    per_plate=defaultdict(list);per_group=defaultdict(list)
    train_real_classes=set(arrays['train_y'].tolist())
    for index,row in enumerate(valrows):
        per_plate[row['plate_id']].append(index);per_group[row['group']].append(index)
    def subset(indices):
        return {'samples':len(indices),'correct':sum(int(predicted[i])==int(vy[i]) for i in indices),
                'accuracy':float(np.mean(predicted[indices]==vy[indices])) if indices else None}
    report={'model':model_name,'run_id':output.name,'config':config,'dataset_manifest':str(Path(bundle).resolve()/'manifest.json'),
            'dataset_manifest_sha256':sha(Path(bundle)/'manifest.json'),'seed':config['seed'],
            'seconds':__import__('time').perf_counter()-started,'class_order':CHARACTER_CLASSES,
            'validation':subset(list(range(len(vy)))),'predictions':observed,'expected':expected,
            'per_class':classification_report(expected,observed,labels=CHARACTER_CLASSES,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(expected,observed,labels=CHARACTER_CLASSES).tolist(),
            'I_predictions':observed.count('I'),'per_plate':{k:subset(v) for k,v in per_plate.items()},
            'per_source_group':{k:subset(v) for k,v in per_group.items()},
            'classes_without_real_train':subset([i for i,label in enumerate(vy) if int(label) not in train_real_classes]),
            'scope':'Development validation, not independent final test. Validation never fitted. Clustered glyphs from plates.'}
    import sklearn,tensorflow as tf
    report['runtime_versions']={'numpy':np.__version__,'opencv':cv2.__version__,'sklearn':sklearn.__version__,'tensorflow':tf.__version__}
    (output/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    domain_weights={'synthetic':float(weights[:len(arrays['synthetic_y'])].sum()),
                    'real':float(weights[len(arrays['synthetic_y']):].sum())}
    (output/'training_log.json').write_text(json.dumps({'model':model_name,'train_samples':len(y),
        'estimator_fit_seconds':estimator_fit_seconds if model_name in ('svm','rf') else None,
        'domain_sample_weight_sums':domain_weights,
        'validation_samples':len(vy),'fit_seconds':report['seconds'],'seed':config['seed'],
        'sample_weight_sum':float(weights.sum()),'class_counts':dict(Counter(map(int,y)))},indent=2),encoding='utf-8')
    (output/'preprocessing.json').write_text(json.dumps(meta['preprocessing'],indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    prepare_parser=sub.add_parser('prepare');prepare_parser.add_argument('dataset',type=Path);prepare_parser.add_argument('output',type=Path);prepare_parser.add_argument('--samples-per-class',type=int,default=100)
    train=sub.add_parser('fit');train.add_argument('bundle',type=Path);train.add_argument('output',type=Path);train.add_argument('--model',choices=['svm','rf','cnn'],required=True);train.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.dataset,args.output,args.samples_per_class)
    else:print(fit(args.bundle,args.output,args.model,json.loads(args.config.read_text(encoding='utf-8'))))


if __name__=='__main__':main()