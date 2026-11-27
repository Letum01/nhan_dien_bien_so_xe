"""Presentation only: rendering helpers shared by the existing Streamlit pages."""

from collections.abc import Sequence

import cv2
import numpy as np
import streamlit as st


def show(image: np.ndarray, caption: str) -> None:
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=caption)


def draw_ordered_boxes(image: np.ndarray, boxes: Sequence[Sequence[int]], *,
                       font_scale: float = .4, min_text_y: int = 10) -> None:
    """Annotate a presentation copy in its own coordinate space, in place."""
    for index, (x, y, width, height) in enumerate(boxes, 1):
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 1)
        cv2.putText(image, str(index), (x, max(min_text_y, y)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)