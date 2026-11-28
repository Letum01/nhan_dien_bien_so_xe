import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime_validation import validate_runtime, validate_fusion_head
from training_artifacts import resolve_artifact


def fixture_manifest(root):
    contract = root/'contract.json'
    contract.write_text(json.dumps({'class_order': list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')}))
    return {'classes': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'features': {'hog': 324, 'embedding': 256, 'fusion': 580},
            'cnn_contract': 'contract.json', 'artifacts': {
                'contract.json': {'bytes': contract.stat().st_size,
                                  'sha256': hashlib.sha256(contract.read_bytes()).hexdigest()}}}


def test_runtime_subset_and_hash_rejection(tmp_path):
    spec = fixture_manifest(tmp_path)
    assert validate_runtime(tmp_path, spec) == spec  # no legacy models present
    (tmp_path/'contract.json').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash/size'):
        validate_runtime(tmp_path, spec)
    spec = fixture_manifest(tmp_path)
    changed = {'class_order': list(spec['classes'])[::-1]}
    path=tmp_path/'contract.json';path.write_text(json.dumps(changed))
    spec['artifacts']['contract.json']={'bytes':path.stat().st_size,
                                      'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match='CNN class'):
        validate_runtime(tmp_path, spec)


def test_class_mapping_and_fusion_contract_rejection(tmp_path):
    spec = fixture_manifest(tmp_path)
    wrong = copy.deepcopy(spec); wrong['classes'] = wrong['classes'][::-1]
    with pytest.raises(ValueError, match='mapping'):
        validate_runtime(tmp_path, wrong)
    head = {'classes': list(spec['classes']),
            'classifier': SimpleNamespace(classes_=list(range(36)), n_features_in_=579),
            'scaler': SimpleNamespace(n_features_in_=580)}
    with pytest.raises(ValueError, match='contract'):
        validate_fusion_head(head, 'svm_fusion')


def test_historical_resolver_checks_rebased_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv('ALPR_TRAINING_ROOT', str(tmp_path))
    target=tmp_path/'experiments/model.pt';target.parent.mkdir();target.write_bytes(b'original')
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    assert resolve_artifact(r'C:\old\project\experiments\model.pt', digest)==target
    with pytest.raises(ValueError, match='hash mismatch'):
        resolve_artifact(r'C:\old\project\experiments\model.pt', '0'*64)
    with pytest.raises(ValueError):resolve_artifact('../escape')


def test_restored_training_dependencies_are_exact():
    root=Path(__file__).resolve().parents[1]
    expected={'configs/ocr_synthetic_only.json': '999b9a9a9d22a32dc1fb8be81c40b413c5ad4df564743e46c0b5a25d4f51db79',
              'segmentation_v2.py': 'bb6c2440cd7f9104e8b1ef09b8917f6c7e33521b883caf1199fdda1512b92d98'}
    for name,digest in expected.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest


def test_research_gate_is_not_runtime_gate(monkeypatch, tmp_path):
    import final_ocr_dataset as research
    monkeypatch.setattr(research, 'ROOT', tmp_path)
    # Unmodified historical protocol still rejects absent artifacts.
    with pytest.raises(FileNotFoundError):research.check_protocol()
    artifact=tmp_path/'history.bin';artifact.write_bytes(b'changed')
    monkeypatch.setattr(research, 'read', lambda _: {'hashes': {'history.bin': '0'*64}})
    with pytest.raises(ValueError, match='Frozen dependency changed'):
        research.check_protocol()


def test_evaluation_artifact_is_measured_and_missing_is_explicit(tmp_path):
    from evaluation_summary import load_summary
    with pytest.raises(FileNotFoundError):load_summary(tmp_path/'absent.json')
    data=load_summary()
    assert data['images']==12 and data['reference_characters']==99
    assert len(data['evidence_sha256'])==64
    assert all(len(v['checkpoint_sha256'])==64 for v in data['models'].values())


def test_evaluation_rejects_tampered_evidence(tmp_path):
    from evaluation_summary import load_summary, ROOT
    data=json.loads((ROOT/'configs/development-regression.json').read_text())
    path=tmp_path/'summary.json';path.write_text(json.dumps(data))
    (tmp_path/data['evidence_file']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='evidence hash'):
        load_summary(path)