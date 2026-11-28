"""Non-fitting checks of retained recipes and their external input pack."""
import argparse
import json
from pathlib import Path

from training_artifacts import artifact_root, resolve_artifact

ROOT = Path(__file__).resolve().parent


def check(recipe: str) -> dict:
    if recipe == 'plate':
        from validate_yolo_dataset import validate_dataset
        result = validate_dataset(resolve_artifact('reports/round2_gate_20261124T154153249700Z/dataset'), artifact_root())
        resolve_artifact('pretrained/yolov8n.pt', 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36')
        return result
    if recipe == 'character':
        from train_yolo_character import readiness
        receipt = resolve_artifact('reports/glyph_audit_20261125T142710898675Z/hard13_v2_results.json')
        inputs = json.loads(resolve_artifact('experiments/yolo_character_round1.inputs.json').read_text())
        resolve_artifact(receipt, inputs['hard13_result_sha256'])
        resolve_artifact('pretrained/yolov8n.pt', inputs['pretrained_sha256'])
        return readiness(resolve_artifact('datasets/yolo_character_round1'), receipt)
    if recipe == 'cnn':
        from run_ocr_mix_experiment01 import preflight
        result = preflight()[2]
        for name in ['ocr_synthetic_only.json', 'ocr_mix_cnn.json']:
            json.loads((ROOT/'configs'/name).read_text())
        return result
    if recipe in ('svm', 'rf'):
        from fusion_dataset import prepare
        from hybrid_features import V2
        rows, _ = prepare()
        expected = json.loads((V2/'synthetic_manifest.json').read_text())['arrays_sha256']
        resolve_artifact(V2/'synthetic.npz', expected)
        for name, digest in json.loads((V2/'checkpoint_hashes.json').read_text()).items():
            resolve_artifact(V2/name, digest)
        return {'status': 'PASS', 'train': len(rows['train']), 'validation': len(rows['validation']),
                'scope': 'Inputs/preprocessing only; feature extraction, CV and fitting NOT executed'}
    raise ValueError(f'Unknown recipe: {recipe}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', choices=['plate', 'character', 'cnn', 'svm', 'rf'])
    args = parser.parse_args()
    print(json.dumps(check(args.recipe), indent=2))