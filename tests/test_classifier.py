"""Unit tests for gesture analysis and LandmarkClassifier."""

import pytest
from sign_language.models import GestureAnalyzer, LandmarkClassifier


def make_landmarks_from_states(thumb=False, index=False, middle=False, ring=False, pinky=False):
    """Helper creating synthetic 21 landmark set with specified finger extensions."""
    # Wrist at (0.5, 0.9, 0.0)
    lms = [(0.5, 0.9, 0.0)]

    # Thumb: 1, 2, 3, 4
    if thumb:
        lms.extend([(0.45, 0.85, 0.0), (0.40, 0.80, 0.0), (0.35, 0.75, 0.0), (0.28, 0.70, 0.0)])
    else:
        lms.extend([(0.48, 0.85, 0.0), (0.47, 0.80, 0.0), (0.48, 0.78, 0.0), (0.49, 0.77, 0.0)])

    # Index: 5, 6, 7, 8
    if index:
        lms.extend([(0.45, 0.70, 0.0), (0.45, 0.55, 0.0), (0.45, 0.45, 0.0), (0.45, 0.35, 0.0)])
    else:
        lms.extend([(0.45, 0.70, 0.0), (0.45, 0.65, 0.0), (0.45, 0.75, 0.0), (0.45, 0.80, 0.0)])

    # Middle: 9, 10, 11, 12
    if middle:
        lms.extend([(0.50, 0.68, 0.0), (0.50, 0.52, 0.0), (0.50, 0.42, 0.0), (0.50, 0.32, 0.0)])
    else:
        lms.extend([(0.50, 0.68, 0.0), (0.50, 0.64, 0.0), (0.50, 0.74, 0.0), (0.50, 0.79, 0.0)])

    # Ring: 13, 14, 15, 16
    if ring:
        lms.extend([(0.55, 0.70, 0.0), (0.55, 0.55, 0.0), (0.55, 0.45, 0.0), (0.55, 0.36, 0.0)])
    else:
        lms.extend([(0.55, 0.70, 0.0), (0.55, 0.65, 0.0), (0.55, 0.75, 0.0), (0.55, 0.80, 0.0)])

    # Pinky: 17, 18, 19, 20
    if pinky:
        lms.extend([(0.60, 0.72, 0.0), (0.60, 0.58, 0.0), (0.60, 0.48, 0.0), (0.60, 0.40, 0.0)])
    else:
        lms.extend([(0.60, 0.72, 0.0), (0.60, 0.68, 0.0), (0.60, 0.77, 0.0), (0.60, 0.82, 0.0)])

    return lms


def test_gesture_analyzer_hello():
    lms = make_landmarks_from_states(thumb=True, index=True, middle=True, ring=True, pinky=True)
    label, conf = GestureAnalyzer.classify_gesture(lms)
    assert label == "Hello"
    assert conf > 0.8


def test_gesture_analyzer_l():
    lms = make_landmarks_from_states(thumb=True, index=True, middle=False, ring=False, pinky=False)
    label, conf = GestureAnalyzer.classify_gesture(lms)
    assert label == "L"
    assert conf > 0.8


def test_gesture_analyzer_ily():
    lms = make_landmarks_from_states(thumb=True, index=True, middle=False, ring=False, pinky=True)
    label, conf = GestureAnalyzer.classify_gesture(lms)
    assert label == "ILY"
    assert conf > 0.8


def test_gesture_analyzer_y():
    lms = make_landmarks_from_states(thumb=True, index=False, middle=False, ring=False, pinky=True)
    label, conf = GestureAnalyzer.classify_gesture(lms)
    assert label == "Y"
    assert conf > 0.8


def test_classifier_temporal_smoothing():
    clf = LandmarkClassifier(history_size=5, stability_threshold=3)

    # Empty inputs
    label, conf, dist = clf.predict(None)
    assert label == "blank"

    # Send 4 frames of "Hello"
    lms = make_landmarks_from_states(thumb=True, index=True, middle=True, ring=True, pinky=True)
    for _ in range(4):
        label, conf, dist = clf.predict(lms)

    assert label == "Hello"
    assert conf > 0.7
    assert dist["Hello"] >= 0.6
