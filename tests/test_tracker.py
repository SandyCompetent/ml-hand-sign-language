"""Unit tests for MediaPipe 3D HandTracker."""

import os
import cv2
import numpy as np
import pytest
from sign_language.vision import HandTracker


@pytest.fixture(scope="module")
def tracker():
    t = HandTracker(max_num_hands=1)
    yield t
    t.close()


def test_tracker_blank_image(tracker):
    """Tracker should process blank image without errors and report 0 hands."""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, hands = tracker.find_hands(blank, draw=True)
    assert annotated.shape == (480, 640, 3)
    assert len(hands) == 0


def test_tracker_real_image(tracker):
    """Tracker should detect hand landmarks on real sample image if available."""
    sample_path = os.path.join(os.path.dirname(__file__), "..", "Asset", "Hand Sign", "1.jpeg")
    if not os.path.exists(sample_path):
        pytest.skip("Asset/Hand Sign/1.jpeg not found")

    bgr = cv2.imread(sample_path)
    annotated, hands = tracker.find_hands(bgr, draw=True)
    assert len(hands) >= 1

    hand = hands[0]
    assert "raw_landmarks" in hand
    assert len(hand["raw_landmarks"]) == 21
    assert "pixel_landmarks" in hand
    assert len(hand["pixel_landmarks"]) == 21
    assert "features" in hand
    assert isinstance(hand["features"], np.ndarray)
    assert hand["features"].size > 21
    assert "bbox" in hand
    x1, y1, x2, y2 = hand["bbox"]
    assert x2 > x1 and y2 > y1
