import csv

import pytest

from train_real_characters import load_reviewed_characters
from collect_real_characters import export_characters


def test_training_rejects_unreviewed_data(tmp_path):
    manifest = tmp_path / "characters.csv"
    manifest.write_text("image_path,class_label,annotation_status,source_group,split\na.png,A,unlabeled,car1,train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed glyphs"):
        load_reviewed_characters(manifest)


def test_character_export_never_overwrites_directory(tmp_path):
    with pytest.raises(FileExistsError):
        export_characters(tmp_path / "absent.csv", tmp_path)