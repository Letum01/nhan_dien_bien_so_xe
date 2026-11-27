import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ocr_mix_experiment import fit,load_bundle,sha


def bundle(tmp_path):
    path=tmp_path/'bundle';path.mkdir()
    rng=np.random.default_rng(42)
    synthetic=rng.choice([0,255],(36,32,32,1)).astype(np.uint8)
    real=synthetic[:2].copy()
    np.savez_compressed(path/'arrays.npz',synthetic_x=synthetic,synthetic_y=np.arange(36,dtype=np.int32),
                        train_x=real,train_y=np.array([0,1]),validation_x=real,validation_y=np.array([0,1]))
    # Deliberately synthetic fixture only; not source labels or accuracy evidence.
    from dataset import CHARACTER_CLASSES
    meta={'class_order':CHARACTER_CLASSES,'array_file_sha256':sha(path/'arrays.npz'),
          'preprocessing':{'shape':[32,32,1],'scope':'synthetic fixture'},
          'manifest':[dict(origin='reviewed_real',split='validation',plate_id='fixture',group='fixture') for _ in range(2)]}
    (path/'manifest.json').write_text(json.dumps(meta),encoding='utf-8')
    return path


@pytest.mark.parametrize('model',['svm','rf'])
def test_fixture_fit_does_not_change_bundle_or_fit_scaler_on_validation(tmp_path,model):
    path=bundle(tmp_path);before=(path/'arrays.npz').read_bytes()
    report=fit(path,tmp_path/model,model,dict(seed=42,variant='synthetic_plus_real',real_domain_fraction=.5,n_estimators=3))
    assert report['validation']['samples']==2
    assert (tmp_path/model/'validation.json').exists()
    assert before==(path/'arrays.npz').read_bytes()
    if model=='svm':
        import joblib
        scaler=joblib.load(tmp_path/model/'hog_scaler.joblib')
        assert scaler.n_samples_seen_==38
    log=json.loads((tmp_path/model/'training_log.json').read_text(encoding='utf-8'))
    assert log['estimator_fit_seconds']>=0
    assert log['domain_sample_weight_sums']['synthetic']==pytest.approx(log['domain_sample_weight_sums']['real'])


def test_bundle_tamper_is_blocked(tmp_path):
    path=bundle(tmp_path)
    with (path/'arrays.npz').open('ab') as stream:stream.write(b'change')
    with pytest.raises(ValueError,match='hash'):load_bundle(path)


def test_cnn_fixture_one_epoch_path(tmp_path):
    path=bundle(tmp_path)
    report=fit(path,tmp_path/'cnn','cnn',dict(seed=42,variant='synthetic_only',epochs=1,batch=16))
    assert report['validation']['samples']==2
    assert (tmp_path/'cnn/cnn_model.keras').exists()
    assert (tmp_path/'cnn/training.csv').exists()