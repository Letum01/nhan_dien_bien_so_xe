"""Explicit optional-model smoke check; may download official model weights."""
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from ocr_backends import recognize_crop


def main():
    root = Path(__file__).resolve().parent
    image = np.full((100, 420, 3), 240, np.uint8)
    cv2.putText(image, "51A12345", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (10, 10, 10), 3)
    result = recognize_crop(image, "paddle")
    print("one-line synthetic:", result.to_dict())
    assert result.status == "prediction", result.error
    from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
    from streamlit.proto.Common_pb2 import FileURLs
    data = (root / "dataset/license_plate_dataset/0/images/0402.jpg").read_bytes()
    payload = UploadedFile(UploadedFileRec("paddle-test", "0402.jpg", "image/jpeg", data), FileURLs())
    with patch("streamlit.delta_generator.DeltaGenerator.file_uploader", return_value=payload):
        app = AppTest.from_file(str(root / "app.py"), default_timeout=120).run()
        app.selectbox[0].select("PaddleOCR (chuỗi trực tiếp)").run(timeout=120)
        assert not app.exception
        assert not app.error
        assert any("65L100780" in item.value for item in app.success)
        print("Paddle JPEG upload/server-side AppTest PASS")


if __name__ == "__main__":
    main()