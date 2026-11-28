"""Resolve historical artifact paths without modifying immutable receipts."""
import hashlib
import os
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parent
MARKERS = ('experiments', 'reports', 'dataset', 'datasets', 'data', 'pretrained', 'saved_models')


def artifact_root() -> Path:
    return Path(os.environ.get('ALPR_TRAINING_ROOT', ROOT)).resolve()


def resolve_artifact(value: str | Path, expected_sha256: str | None = None) -> Path:
    """Rebase only known project-relative paths; require exact hash when supplied."""
    raw = str(value)
    parts = PureWindowsPath(raw).parts
    if PureWindowsPath(raw).is_absolute() or Path(raw).is_absolute():
        indices = [i for i, part in enumerate(parts) if part in MARKERS]
        if not indices:
            raise ValueError(f'Historical path has no recognized artifact scope: {raw}')
        relative = Path(*parts[min(indices):])
    else:
        relative = Path(*parts)
    root = artifact_root()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Artifact path escapes configured root: {value}')
    if expected_sha256 is not None:
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected_sha256:
            raise ValueError(f'Historical artifact hash mismatch: {value} -> {path}')
    return path