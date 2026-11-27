"""Explicit abstaining ensemble policies, not automatically promoted models.

Majority voting is an experimental baseline. Weighted/soft policies must be
frozen on validation before independent test evaluation. No default weights,
uncalibrated margins, random tie breakers or syntactic label substitutions.
"""
from collections import Counter
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

BACKENDS = ("svm", "rf", "cnn")
CLASSES = tuple("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass
class VoteResult:
    raw_text: str
    status: str
    uncertain_positions: list
    method: str
    members: dict


def _validate_predictions(predictions: Mapping[str, str]):
    if set(predictions) != set(BACKENDS):
        raise ValueError("All three SVM/RF/CNN predictions required")
    lengths = {len(text) for text in predictions.values()}
    if len(lengths) != 1:
        raise ValueError("Predictions must share the exact same segmented glyphs")
    if any(ch not in CLASSES for text in predictions.values() for ch in text):
        raise ValueError("Prediction outside the frozen class mapping")


def vote(predictions: Mapping[str, str], method="majority", weights=None,
         validation_provenance=None) -> VoteResult:
    _validate_predictions(predictions)
    if method not in ("majority", "weighted"):
        raise ValueError("Use soft_vote for calibrated probability matrices")
    if method == "weighted":
        if not validation_provenance or not weights or set(weights) != set(BACKENDS):
            raise ValueError("Weighted voting requires validation provenance and three frozen weights")
        if any(not math.isfinite(float(w)) or float(w) <= 0 for w in weights.values()):
            raise ValueError("Weights must be finite and positive")
    characters, uncertain = [], []
    for index, labels in enumerate(zip(*(predictions[name] for name in BACKENDS))):
        if method == "majority":
            counts = Counter(labels)
            best, count = counts.most_common(1)[0]
            chosen = best if count >= 2 else "?"
        else:
            totals = {}
            for name, label in zip(BACKENDS, labels):
                totals[label] = totals.get(label, 0.0) + float(weights[name])
            ordered = sorted(totals.items(), key=lambda entry: entry[1], reverse=True)
            chosen = ordered[0][0] if len(ordered) == 1 or not math.isclose(ordered[0][1], ordered[1][1]) else "?"
        if chosen == "?":
            uncertain.append(index)
        characters.append(chosen)
    raw = "".join(characters)
    return VoteResult(raw, "uncertain" if uncertain else "prediction" if raw else "unreadable",
                      uncertain, method, dict(predictions))


def soft_vote(probabilities: Mapping[str, np.ndarray], class_orders: Mapping[str, Sequence[str]],
              weights, calibration_provenance, validation_provenance) -> VoteResult:
    """Combine ALREADY calibrated arrays, aligned explicitly to class labels.

    This function does not calibrate models or assert metadata proves good
    calibration. Calibration must be fitted on separate data by the caller.
    Production does not enable this policy without a measured experiment.
    """
    if not calibration_provenance or not validation_provenance:
        raise ValueError("Calibration and validation provenance are mandatory")
    if set(probabilities) != set(BACKENDS) or set(class_orders) != set(BACKENDS):
        raise ValueError("All three calibrated distributions/class orders required")
    if set(weights) != set(BACKENDS) or any(not math.isfinite(float(w)) or w <= 0 for w in weights.values()):
        raise ValueError("Three finite positive frozen weights required")
    aligned, members, rows = [], {}, None
    for name in BACKENDS:
        order = list(class_orders[name])
        array = np.asarray(probabilities[name], dtype=float)
        if len(order) != 36 or set(order) != set(CLASSES) or array.ndim != 2 or array.shape[1] != 36:
            raise ValueError("Expected unique 36-class probability distribution")
        if not np.isfinite(array).all() or (array < 0).any() or not np.allclose(array.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Not a normalized probability matrix")
        if rows is not None and len(array) != rows:
            raise ValueError("Different glyph counts")
        rows = len(array)
        matrix = array[:, [order.index(label) for label in CLASSES]]
        aligned.append(matrix * weights[name])
        members[name] = "".join(CLASSES[i] for i in matrix.argmax(axis=1))
    combined = sum(aligned) / sum(weights.values())
    chars, uncertain = [], []
    for index, row in enumerate(combined):
        winners = np.flatnonzero(np.isclose(row, row.max(), atol=1e-10, rtol=1e-9))
        if len(winners) != 1:
            chars.append("?")
            uncertain.append(index)
        else:
            chars.append(CLASSES[winners[0]])
    raw = "".join(chars)
    return VoteResult(raw, "uncertain" if uncertain else "prediction" if raw else "unreadable",
                      uncertain, "soft", members)


def recognize_ensemble(char_images, models):
    from main import recognize_characters
    predictions = {name: recognize_characters(char_images, name, models) for name in BACKENDS}
    return vote(predictions)