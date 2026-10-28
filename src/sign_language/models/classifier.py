"""Sign Language Gesture Classifiers.

Provides landmark-based classification with temporal debouncing as well as
backward-compatible adapters for legacy CNN models.
"""

from __future__ import annotations
import os
import collections
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

from .gestures import GestureAnalyzer


class BaseClassifier(ABC):
    """Abstract Base Class for Sign Language Gesture Classifiers."""

    @abstractmethod
    def predict(
        self,
        raw_landmarks: Optional[List[Tuple[float, float, float]]] = None,
        normalized_features: Optional[np.ndarray] = None,
        hand_image_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[str, float, Dict[str, float]]:
        """Predict the sign label, confidence score, and distribution.

        Returns:
            label: Detected sign string (e.g. 'A', 'Hello', 'blank').
            confidence: Float confidence between 0.0 and 1.0.
            distribution: Dictionary of class scores.
        """
        pass


class LandmarkClassifier(BaseClassifier):
    """Real-time 3D landmark gesture classifier with temporal smoothing."""

    def __init__(self, history_size: int = 5, stability_threshold: int = 4):
        self.history_size = history_size
        self.stability_threshold = stability_threshold
        self._history = collections.deque(maxlen=history_size)
        self._last_stable_label = "blank"

    def predict(
        self,
        raw_landmarks: Optional[List[Tuple[float, float, float]]] = None,
        normalized_features: Optional[np.ndarray] = None,
        hand_image_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[str, float, Dict[str, float]]:
        if raw_landmarks is None or len(raw_landmarks) < 21:
            self._history.append("blank")
            return "blank", 0.0, {"blank": 1.0}

        # 1. Instantaneous classification via GestureAnalyzer
        raw_label, conf = GestureAnalyzer.classify_gesture(raw_landmarks)

        # 2. Append to temporal smoothing window
        self._history.append(raw_label)

        # 3. Vote across recent history to eliminate single-frame glitches
        counts = collections.Counter(self._history)
        most_common_label, count = counts.most_common(1)[0]

        distribution = {k: v / len(self._history) for k, v in counts.items()}

        if count >= self.stability_threshold:
            self._last_stable_label = most_common_label
            return most_common_label, conf, distribution

        # Fallback to current smoothed or raw prediction
        return most_common_label, conf * (count / len(self._history)), distribution

    def reset_history(self) -> None:
        """Clear temporal smoothing history."""
        self._history.clear()
        self._last_stable_label = "blank"


class LegacyCNNClassifier(BaseClassifier):
    """Adapter for legacy Keras CNN models trained on thresholded binary images."""

    def __init__(self, model_json_path: str, model_weights_path: str):
        self.model_json_path = model_json_path
        self.model_weights_path = model_weights_path
        self.model = None

        try:
            from tensorflow.keras.models import model_from_json
            if os.path.exists(model_json_path) and os.path.exists(model_weights_path):
                with open(model_json_path, "r") as f:
                    self.model = model_from_json(f.read())
                self.model.load_weights(model_weights_path)
                print(f"[LegacyCNNClassifier] Loaded model from {model_json_path}")
            else:
                print(f"[LegacyCNNClassifier] Model files not found: {model_json_path}")
        except Exception as e:
            print(f"[LegacyCNNClassifier] Could not load Keras model: {e}")

    def predict(
        self,
        raw_landmarks: Optional[List[Tuple[float, float, float]]] = None,
        normalized_features: Optional[np.ndarray] = None,
        hand_image_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[str, float, Dict[str, float]]:
        if self.model is None or hand_image_bgr is None:
            return "blank", 0.0, {"blank": 1.0}

        try:
            # Otsu thresholding preprocessing matching CodeV3/image_processing.py
            gray = cv2.cvtColor(hand_image_bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            _, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            res = cv2.resize(res, (128, 128))
            tensor = res.reshape(1, 128, 128, 1) / 255.0

            probs = self.model.predict(tensor, verbose=0)[0]
            max_idx = int(np.argmax(probs))
            conf = float(probs[max_idx])

            # CodeV3 classes: 0 = blank, 1..26 = A..Z
            if max_idx == 0:
                label = "blank"
            else:
                label = chr(ord("A") + max_idx - 1)

            return label, conf, {label: conf}
        except Exception as e:
            return "blank", 0.0, {"error": str(e)}


def create_classifier(mode: str = "landmark") -> BaseClassifier:
    """Factory function to instantiate the requested classifier."""
    if mode == "landmark":
        return LandmarkClassifier()
    elif mode == "legacy":
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        repo_root = os.path.dirname(os.path.dirname(pkg_root))
        
        # Check modern legacy/ location first, then fallback to CodeV3/
        json_path = os.path.join(repo_root, "legacy", "phase3_asl_threshold", "model", "model-bw.json")
        weights_path = os.path.join(repo_root, "legacy", "phase3_asl_threshold", "model", "model-bw.h5")
        if not os.path.exists(json_path):
            json_path = os.path.join(repo_root, "CodeV3", "model", "model-bw.json")
            weights_path = os.path.join(repo_root, "CodeV3", "model", "model-bw.h5")
        return LegacyCNNClassifier(json_path, weights_path)
    else:
        raise ValueError(f"Unknown classifier mode: {mode}")
