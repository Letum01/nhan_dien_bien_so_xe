"""P0 Unit tests verifying HOG extraction across multiple image types and edge cases."""
import numpy as np
import cv2
import pytest
from dataset import extract_hog_features
from glyph_review import ROOT

def test_hog_empty_blank_image():
    """Blank image should yield all-zero HOG vector."""
    blank = np.zeros((32, 32), dtype=np.uint8)
    feats = extract_hog_features(blank)
    assert feats.shape == (324,)
    assert feats.dtype == np.float32
    assert np.all(feats == 0.0)
    assert not np.any(np.isnan(feats))
    assert not np.any(np.isinf(feats))

def test_hog_vertical_line():
    """Vertical line should produce non-zero features concentrated in vertical gradient bins."""
    img = np.zeros((32, 32), dtype=np.uint8)
    img[:, 15:17] = 255
    feats = extract_hog_features(img)
    assert feats.shape == (324,)
    assert np.count_nonzero(feats) > 0
    assert not np.any(np.isnan(feats))
    assert np.max(feats) > 0.1

def test_hog_horizontal_line():
    """Horizontal line should produce non-zero features."""
    img = np.zeros((32, 32), dtype=np.uint8)
    img[15:17, :] = 255
    feats = extract_hog_features(img)
    assert feats.shape == (324,)
    assert np.count_nonzero(feats) > 0
    assert not np.any(np.isnan(feats))

def test_hog_diagonal_line():
    """Diagonal line should produce non-zero features in diagonal gradient bins."""
    img = np.zeros((32, 32), dtype=np.uint8)
    cv2.line(img, (0, 0), (31, 31), 255, 2)
    feats = extract_hog_features(img)
    assert feats.shape == (324,)
    assert np.count_nonzero(feats) > 0
    assert not np.any(np.isnan(feats))

def test_hog_synthetic_glyph():
    """A rendered synthetic letter should produce standard valid HOG."""
    img = np.zeros((32, 32), dtype=np.uint8)
    cv2.putText(img, 'B', (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    feats = extract_hog_features(img)
    assert feats.shape == (324,)
    assert np.count_nonzero(feats) >= 30
    assert np.max(feats) <= 1.0
    assert np.min(feats) >= 0.0

def test_hog_real_reviewed_glyphs():
    """Real reviewed glyphs from characters.csv must have non-zero, valid HOG."""
    import csv
    real_dir = ROOT / 'reports/glyph_audit_20261125T145221295978Z/dataset'
    real_csv = real_dir / 'characters.csv'
    if not real_csv.exists():
        pytest.skip("Real dataset characters.csv not found")
    
    with real_csv.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    
    # Test first 10 rows
    tested = 0
    for r in rows[:10]:
        img_p = real_dir / r['image_path']
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p), 0)
        assert img is not None
        feats = extract_hog_features(img)
        assert feats.shape == (324,)
        assert np.count_nonzero(feats) >= 20, f"Real glyph {r['image_path']} had too few non-zero features"
        assert not np.any(np.isnan(feats))
        assert not np.any(np.isinf(feats))
        tested += 1
    assert tested >= 5