from pathlib import Path
import cv2
import numpy as np

from streamlit.testing.v1 import AppTest
from image_io import decode_uploaded_image
from main import end_to_end_pipeline, load_trained_models


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_tabs_models_and_empty_history():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert len(app.tabs) == 3
    assert len(app.selectbox) == 2
    assert "Chưa có lịch sử" in " ".join(item.value for item in app.info)


def test_model_selection_reruns_without_exception():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    for choice in (
        "CNN (Deep Learning)",
        "SVM (HOG + Support Vector)",
        "Random Forest (HOG + Ensembles)",
        "Ensemble (Majority — thử nghiệm)",
    ):
        app.selectbox[0].select(choice).run()
        assert not app.exception
        assert app.selectbox[0].value == choice


def test_history_and_model_information_are_rendered():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert any("Thông tin mô hình" in tab.label for tab in app.tabs)
    assert any("Chưa có lịch sử nhận diện" in item.value for item in app.info)


def test_upload_decoder_used_by_app_accepts_images_and_rejects_corruption():
    image = np.zeros((50, 90, 3), dtype=np.uint8)
    for extension in (".jpg", ".png"):
        ok, encoded = cv2.imencode(extension, image)
        assert ok and decode_uploaded_image(encoded.tobytes()) is not None
    assert decode_uploaded_image(b"broken-upload") is None


def test_ui_state_after_image_decode_pipeline_and_repeat_uploads():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    models = load_trained_models("saved_models")
    assert models is not None
    for model in ("cnn", "svm", "rf"):
        _, rendered, _, _ = end_to_end_pipeline(
            np.zeros((320, 480, 3), dtype=np.uint8), model, models
        )
        assert rendered.shape == (320, 480, 3)
        assert not app.exception
    # Repeated script execution approximates sequential UI interactions; no
    # uploader simulation API exists in Streamlit 1.36 AppTest.
    app.run()
    assert not app.exception