import numpy as np

from ocr_v2_synthetic import generate
from train_ocr_v2 import weights


def test_synthetic_contract_and_seed():
    images, labels = generate(2)
    again, _ = generate(2)
    assert images.shape == (72, 32, 32) and images.dtype == np.uint8
    assert np.array_equal(images, again)
    assert set(np.unique(images)) == {0, 255}
    assert np.all(np.any(images, axis=(1, 2)))
    assert np.array_equal(np.bincount(labels), np.full(36, 2))


def test_domain_weight_sums_are_equal():
    w = weights(7200, 370)
    assert len(w) == 7570
    assert np.isclose(w[:7200].sum(), w[7200:].sum())