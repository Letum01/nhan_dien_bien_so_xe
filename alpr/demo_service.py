"""Streamlit-independent recognition routing and result assembly."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypedDict

import numpy as np

from demo_fusion import FUSION, FusionHeads, run_fusion
from dual_yolo_client import shared_worker
from dual_yolo_pipeline import run
from types import SimpleNamespace


class DemoRow(TypedDict):
    filename: str
    prediction: str
    algorithm: str
    seconds: float | None
    status: str


def infer(image: np.ndarray, backend: str, cnn: Callable,
          get_fusion: Callable[[], FusionHeads]) -> dict:
    """Dispatch without changing detector, pixel, or classifier contracts."""
    if backend in (*FUSION, 'compare'):
        return run_fusion(image, shared_worker, cnn, get_fusion, backend)
    if backend != 'cnn':
        raise ValueError('Unknown classifier')
    return run(image, shared_worker, SimpleNamespace(cnn=cnn), 'cnn')


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
