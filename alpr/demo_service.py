"""Streamlit-independent recognition routing and result assembly."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypedDict

import numpy as np

from demo_fusion import FUSION, FusionHeads, run_fusion
from dual_yolo_client import shared_worker
from dual_yolo_pipeline import MixClassifiers, run
from final_ocr_dataset import ROOT, check_protocol, read


class DemoRow(TypedDict):
    filename: str
    prediction: str
    algorithm: str
    seconds: float | None
    status: str


def infer(image: np.ndarray, method: str, backend: str, models: MixClassifiers,
          get_fusion: Callable[[], FusionHeads]) -> dict:
    """Dispatch without changing detector, pixel, or classifier contracts."""
    if backend in (*FUSION, 'compare'):
        return run_fusion(image, shared_worker, models, get_fusion, backend, method)
    return run(image, shared_worker, models, backend, method)


def request_backend(backend: str, *, compare: bool, collect: bool) -> str:
    if compare:
        return 'compare'
    if collect and backend not in FUSION:
        return 'all'
    return backend


def result_row(name: str, method: str, backend: str, result: Mapping) -> DemoRow:
    text = result['text'][backend]
    status = 'Có dự đoán'
    if not result['plate']:
        status = 'Không thấy biển'
    elif not text or '?' in text:
        status = 'Không chắc chắn'
    return DemoRow(filename=Path(name).name, prediction=text, algorithm=f'{method} / {backend}',
                   seconds=round(result['timings']['total'], 4), status=status)


def error_row(name: str, method: str, backend: str, exc: Exception) -> DemoRow:
    return DemoRow(filename=Path(name).name, prediction='', algorithm=f'{method} / {backend}',
                   seconds=None, status=f'Lỗi: {exc}')


def save_research_case(root: Path, name: str, payload: bytes, image: np.ndarray,
                       result: dict, method: str, expected: str) -> Path:
    """Save only explicitly requested research cases, with their model hashes."""
    from v4_failure_store import save_case

    check_protocol()
    hashes = dict(read(ROOT / 'reports/final_ocr_protocol.json')['hashes'])
    if any(key in result['text'] for key in FUSION):
        heads = read(ROOT / 'reports/adaptive_score_probe.json')['head_hashes']
        hashes.update({f'experiments/hybrid_features_01/{key}.joblib': heads[key] for key in FUSION})
    return save_case(root, name, payload, image, result, method, hashes, expected)