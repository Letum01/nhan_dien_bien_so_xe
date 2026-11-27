import pytest

from dataset import post_process_plate_text


@pytest.mark.parametrize("raw,expected", [
    ("51A00530", "51A00530"),
    ("5OA0053O", "5OA0053O"),
    ("51A0B530", "51A0B530"),
    ("INVALID", "INVALID"),
    ("", ""),
    ("51A00S3", "51A00S3"),
])
def test_post_process_is_conservative(raw, expected):
    assert post_process_plate_text(raw) == expected


def test_low_confidence_glyphs_are_corrected_only_in_valid_pattern():
    assert post_process_plate_text("5OA0053O", [0.1] * 8) == "50A00530"


def test_high_confidence_glyph_is_preserved():
    assert post_process_plate_text("5OA00530", [0.1, 1.0, 1, 1, 1, 1, 1, 1]) == "5OA00530"


def test_confidence_outside_probability_range_rejected():
    with pytest.raises(ValueError):
        post_process_plate_text("51A00530", [2.0] * 8)


def test_confidence_length_must_match():
    with pytest.raises(ValueError):
        post_process_plate_text("51A00530", [0.5])