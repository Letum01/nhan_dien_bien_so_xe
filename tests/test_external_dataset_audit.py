import io
import zipfile

import pytest

from external_dataset_audit import bands, parse_yolo, safe_members, vn_box


@pytest.mark.parametrize('name', ['../escape.txt', 'C:/test.txt', 'x/CON.txt', 'x/a.txt:stream'])
def test_reject_unsafe_zip_paths(name):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr(name, 'fixture')
    with zipfile.ZipFile(stream) as archive, pytest.raises(ValueError, match='Unsafe'):
        safe_members(archive)


def test_reject_case_collision():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('a/X.txt', '')
        archive.writestr('A/y.txt', '')
    with zipfile.ZipFile(stream) as archive, pytest.raises(ValueError, match='collision'):
        safe_members(archive)


def test_label_and_hash_band_controls():
    boxes, errors = parse_yolo('29 .5 .5 .1 .2\n29 .5 .5 .1 .2\n30 .5 .5 .1 .2\n0 .5 .5 nan .2', 30, 100, 100)
    assert boxes == [[29, 45., 40., 10., 20.]]
    assert errors == {'duplicate_box': 1, 'class_out_of_range': 1, 'invalid_box': 1}
    assert vn_box((10, 20, 90, 90), 100, 100, False) is None
    assert vn_box((10, 20, 90, 90), 100, 100, True) == [0, 10, 20, 80, 70]
    assert set(bands(0)) & set(bands(sum(1 << i for i in (0, 10, 19, 28, 37, 46))))