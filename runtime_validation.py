"""Runtime-only, hash-locked artifact validation; research gates stay separate."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASSES = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_runtime(root: Path = ROOT, manifest: dict | None = None) -> dict:
    """Verify registered bytes, never learn expected hashes from candidate files."""
    root = Path(root).resolve()
    spec = manifest or json.loads((ROOT / 'configs/runtime-inference.json').read_text(encoding='utf-8'))
    if spec['classes'] != CLASSES or spec['features'] != {'hog': 324, 'embedding': 256, 'fusion': 580}:
        raise ValueError('Runtime class/feature mapping mismatch')
    for name, record in spec['artifacts'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing/unsafe runtime artifact: {name}')
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError(f'Runtime artifact hash/size mismatch: {name}')
    contract = json.loads((root / spec['cnn_contract']).read_text(encoding='utf-8'))
    if contract['class_order'] != list(CLASSES):
        raise ValueError('CNN class contract mismatch')
    return spec


def validate_fusion_head(head: dict, name: str) -> None:
    if (head['classes'] != list(CLASSES)
            or list(head['classifier'].classes_) != list(range(36))
            or head['classifier'].n_features_in_ != 580
            or (name == 'svm_fusion' and head['scaler'].n_features_in_ != 580)):
        raise ValueError(f'Fusion feature/class contract mismatch: {name}')


if __name__ == '__main__':
    validate_runtime()
    print('Runtime artifact validation PASS')