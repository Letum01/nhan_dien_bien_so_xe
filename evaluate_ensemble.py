"""Validation-frozen hard-vote experiment on reviewed real character datasets.

No calibration artifacts currently exist: soft voting is explicitly blocked,
not approximated with raw classifier scores. Models must not have trained on
validation/test rows; verify provenance externally before using these metrics.
"""
import argparse
import hashlib
import json
from pathlib import Path

from ensemble_recognition import BACKENDS, vote
from main import load_trained_models, recognize_characters
from train_real_characters import load_reviewed_characters


def metrics(expected, predicted):
    if len(expected) != len(predicted):
        raise ValueError("Different glyph counts in evaluation")
    pairs = list(zip(expected, predicted))
    total = len(pairs)
    return {"samples": total, "character_accuracy": sum(a == b for a, b in pairs)/total if total else None,
            "unknown_rate": sum(b == "?" for _, b in pairs)/total if total else None}


def run(manifest: Path, models_dir: Path):
    arrays = load_reviewed_characters(manifest)
    models = load_trained_models(str(models_dir))
    if models is None or any(models.get(name) is None for name in BACKENDS):
        raise ValueError("All three models required")
    from dataset import IDX_TO_CLASS
    provenance = hashlib.sha256(manifest.read_bytes()).hexdigest()
    reports, weights = {}, None
    for split in ("validation", "test"):
        x, y = arrays[split]
        expected = "".join(IDX_TO_CLASS[int(label)] for label in y)
        chars = [image.squeeze() for image in x]
        predictions = {name: recognize_characters(chars, name, models) for name in BACKENDS}
        singles = {name: metrics(expected, text) for name, text in predictions.items()}
        if split == "validation":
            # Predeclared rule, no model gets a privileged weight. Persist before
            # test inference; floor only prevents an undefined zero-weight model.
            weights = {name: max(singles[name]["character_accuracy"], 1e-6) for name in BACKENDS}
        majority = vote(predictions)
        weighted = vote(predictions, "weighted", weights, provenance)
        reports[split] = {"single_models": singles, "majority": metrics(expected, majority.raw_text),
                          "weighted": metrics(expected, weighted.raw_text),
                          "predictions": predictions, "expected": expected,
                          "majority_raw": majority.raw_text, "weighted_raw": weighted.raw_text}
    candidates = {name: value["character_accuracy"] for name, value in reports["validation"]["single_models"].items()}
    candidates.update({method: reports["validation"][method]["character_accuracy"] for method in ("majority", "weighted")})
    return {"weights_rule": "validation character accuracy, floor 1e-6", "weights": weights,
            "validation_manifest_sha256": provenance, "models_dir": str(models_dir.resolve()),
            "selected_on_validation": max(candidates, key=candidates.get), "reports": reports,
            "soft_voting": {"status": "BLOCKED", "reason": "No independently fitted/calibrated model artifacts"},
            "warning": "Character-level experiment only. Does not measure plate exact match or detection quality. Verify model training provenance."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("models_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new report path; do not overwrite experiment evidence")
    args.output.write_text(json.dumps(run(args.manifest, args.models_dir), indent=2), encoding="utf-8")