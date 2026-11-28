"""Load measured development results, never substitute constants for evidence."""
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_summary(path: Path = ROOT/'configs/development-regression.json') -> dict:
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['scope'] != 'development regression; not independent or blind test':
        raise ValueError('Unknown evaluation scope')
    evidence_path = (path.parent / data['evidence_file']).resolve()
    if evidence_path.parent != path.parent.resolve():
        raise ValueError('Evaluation evidence must be adjacent to the summary')
    evidence_bytes = evidence_path.read_bytes()
    if hashlib.sha256(evidence_bytes).hexdigest() != data['evidence_sha256']:
        raise ValueError('Evaluation evidence hash mismatch')
    records = json.loads(evidence_bytes)['records']
    if (len(records) != data['images']
            or sum(record['reference_characters'] for record in records) != data['reference_characters']):
        raise ValueError('Evaluation cohort count mismatch')
    for name, metric in data['models'].items():
        if not (0 <= metric['exact'] <= data['images']
                and 0 <= metric['matched_correct'] <= metric['matched_total']
                and metric['edits'] >= 0):
            raise ValueError(f'Invalid measured counts: {name}')
        for field in ('exact', 'edits', 'matched_correct'):
            if sum(record['models'][name][field] for record in records) != metric[field]:
                raise ValueError(f'Evaluation aggregate mismatch: {name}/{field}')
        if sum(record['matched_total'] for record in records) != metric['matched_total']:
            raise ValueError(f'Evaluation matched count mismatch: {name}')
    return data