"""Verify/copy an approved local asset pack. Never downloads or overwrites files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inside(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or path == root or not path.is_relative_to(root):
        raise ValueError(f'Unsafe manifest path: {name}')
    return path


def restore(manifest, source, destination, *, apply=False):
    entries = json.loads(manifest.read_text(encoding='utf-8-sig'))
    if not isinstance(entries, dict) or not entries:
        raise ValueError('Expected a non-empty path-to-hash manifest')
    source, destination = source.resolve(), destination.resolve()
    plan = []
    seen = set()
    for name, record in entries.items():
        src, dst = inside(source, name), inside(destination, name)
        if dst in seen:
            raise ValueError(f'Duplicate destination: {name}')
        seen.add(dst)
        for path in (src, dst) if dst.exists() else (src,):
            if not path.is_file() or path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
                raise ValueError(f'Asset missing, pointer-only, or hash mismatch: {path}')
        if not dst.exists():
            plan.append((src, dst))
    if apply:
        for src, dst in plan:
            dst.parent.mkdir(parents=True, exist_ok=True)
            with src.open('rb') as a, dst.open('xb') as b:
                shutil.copyfileobj(a, b)
            if digest(src) != digest(dst):
                raise OSError(f'Copy verification failed: {dst}')
    return len(entries), len(plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'source', 'destination'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    count, missing = restore(args.manifest, args.source, args.destination, apply=args.apply)
    print(f'Verified {count} asset(s); {missing} initially missing; copy enabled={args.apply}')