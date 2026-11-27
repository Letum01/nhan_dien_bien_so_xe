"""Render real upload branches in AppTest via injected UploadedFile payloads.

This is server-side interaction coverage, not a browser/file-picker test.
"""
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.proto.Common_pb2 import FileURLs
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def uploaded(data, name):
    return UploadedFile(UploadedFileRec("test", name, "image/png", data), FileURLs())


def test_render_uploaded_png_and_corrupt_file():
    ok, buffer = cv2.imencode(".png", np.zeros((360, 640, 3), np.uint8))
    assert ok
    with patch("streamlit.delta_generator.DeltaGenerator.file_uploader", return_value=uploaded(buffer.tobytes(), "blank.png")):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        assert not app.exception
        assert any("UNKNOWN" in item.value for item in app.success)
        assert len(app.get("imgs")) >= 2
    with patch("streamlit.delta_generator.DeltaGenerator.file_uploader", return_value=uploaded(b"corrupt", "bad.jpg")):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        assert not app.exception
        assert any("Không đọc được ảnh" in item.value for item in app.error)


def test_real_jpeg_upload_inference_for_three_legacy_models():
    data = (ROOT / "dataset/license_plate_dataset/0/images/0402.jpg").read_bytes()
    with patch("streamlit.delta_generator.DeltaGenerator.file_uploader", return_value=uploaded(data, "0402.jpg")):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        for choice in ("CNN (Deep Learning)", "SVM (HOG + Support Vector)", "Random Forest (HOG + Ensembles)", "Ensemble (Majority — thử nghiệm)"):
            app.selectbox[0].select(choice).run()
            assert not app.exception
            assert not app.error
            assert len(app.get("imgs")) >= 3
        comparison = next(button for button in app.button if button.label.startswith("So sánh SVM"))
        comparison.click().run()
        assert not app.exception
        assert len(app.dataframe) >= 1
        frame = app.dataframe[0].value
        assert set(frame["model"]) == {"svm", "rf", "cnn", "ensemble"}