"""Hand tracking and 3D landmark extraction module using Google MediaPipe Tasks."""

from __future__ import annotations
import os
import math
import urllib.request
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarksConnections
except ImportError:
    mp = None
    vision = None
    BaseOptions = None
    HandLandmarksConnections = None

DEFAULT_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Standard MediaPipe 21 landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


class HandTracker:
    """Robust 3D Hand Tracker using MediaPipe Tasks HandLandmarker."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_num_hands: int = 1,
        min_hand_detection_confidence: float = 0.6,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.max_num_hands = max_num_hands
        self.min_detection_conf = min_hand_detection_confidence
        self.min_presence_conf = min_hand_presence_confidence
        self.min_tracking_conf = min_tracking_confidence

        if model_path is None:
            # Default to bundled models directory
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            pkg_root = os.path.dirname(curr_dir)
            model_dir = os.path.join(pkg_root, "models", "assets")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "hand_landmarker.task")

        self.model_path = model_path
        self._ensure_model_file()

        if vision is None or BaseOptions is None:
            raise RuntimeError("MediaPipe is not installed or failed to import.")

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_conf,
            min_hand_presence_confidence=self.min_presence_conf,
            min_tracking_confidence=self.min_tracking_conf,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def _ensure_model_file(self) -> None:
        """Download hand_landmarker.task model if missing."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f"[HandTracker] Downloading MediaPipe model to {self.model_path}...")
            urllib.request.urlretrieve(DEFAULT_TASK_URL, self.model_path)
            print("[HandTracker] Download complete.")

    def find_hands(
        self, frame_bgr: np.ndarray, draw: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Process a BGR video frame and detect hand landmarks.

        Returns:
            annotated_frame: Frame with skeletal drawing overlay (if draw=True).
            hands: List of dicts containing landmark data, bounding boxes, and normalized features.
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.landmarker.detect(mp_image)
        annotated_frame = frame_bgr.copy()
        hands_data: List[Dict[str, Any]] = []

        if not detection_result.hand_landmarks:
            return annotated_frame, hands_data

        for hand_idx, landmarks in enumerate(detection_result.hand_landmarks):
            raw_pts = []
            pixel_pts = []

            for lm in landmarks:
                raw_pts.append((lm.x, lm.y, lm.z))
                px = int(np.clip(lm.x * w, 0, w - 1))
                py = int(np.clip(lm.y * h, 0, h - 1))
                pixel_pts.append((px, py))

            # Compute bounding box
            xs = [p[0] for p in pixel_pts]
            ys = [p[1] for p in pixel_pts]
            x_min, x_max = max(0, min(xs) - 20), min(w - 1, max(xs) + 20)
            y_min, y_max = max(0, min(ys) - 20), min(h - 1, max(ys) + 20)

            # Handedness label
            handedness = "Right"
            hand_score = 1.0
            if (
                detection_result.handedness
                and len(detection_result.handedness) > hand_idx
                and detection_result.handedness[hand_idx]
            ):
                cat = detection_result.handedness[hand_idx][0]
                handedness = cat.category_name
                hand_score = cat.score

            # Scale- and Translation-Invariant feature vector
            normalized_features = self._extract_normalized_features(raw_pts)

            hand_info = {
                "raw_landmarks": raw_pts,
                "pixel_landmarks": pixel_pts,
                "bbox": (x_min, y_min, x_max, y_max),
                "handedness": handedness,
                "score": hand_score,
                "features": normalized_features,
            }
            hands_data.append(hand_info)

            if draw:
                self.draw_hand_skeleton(annotated_frame, pixel_pts, (x_min, y_min, x_max, y_max), handedness)

        return annotated_frame, hands_data

    @staticmethod
    def _extract_normalized_features(raw_pts: List[Tuple[float, float, float]]) -> np.ndarray:
        """Extract scale- and translation-invariant normalized coordinates and joint distances."""
        # 1. Translate relative to wrist (point 0)
        wrist = np.array(raw_pts[0], dtype=np.float32)
        translated = np.array(raw_pts, dtype=np.float32) - wrist

        # 2. Scale by Euclidean distance from wrist to middle finger MCP (point 9)
        ref_dist = float(np.linalg.norm(translated[MIDDLE_MCP]))
        if ref_dist < 1e-4:
            ref_dist = 1.0
        scaled = translated / ref_dist

        # 3. Compute joint angles and fingertip relative distances
        features = scaled.flatten().tolist()

        # Inter-fingertip distances (scale-invariant)
        for i in range(len(FINGERTIPS)):
            for j in range(i + 1, len(FINGERTIPS)):
                f1 = scaled[FINGERTIPS[i]]
                f2 = scaled[FINGERTIPS[j]]
                features.append(float(np.linalg.norm(f1 - f2)))

        # Fingertip to wrist distances
        for tip in FINGERTIPS:
            features.append(float(np.linalg.norm(scaled[tip])))

        return np.array(features, dtype=np.float32)

    def draw_hand_skeleton(
        self,
        frame: np.ndarray,
        pixel_pts: List[Tuple[int, int]],
        bbox: Tuple[int, int, int, int],
        handedness: str,
    ) -> None:
        """Draw sleek glowing hand landmarks and connections onto frame."""
        x_min, y_min, x_max, y_max = bbox

        # Draw semi-transparent bounding box with corner accents
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (50, 205, 50), 1, cv2.LINE_AA)
        corner_len = 15
        cv2.line(frame, (x_min, y_min), (x_min + corner_len, y_min), (0, 255, 128), 2)
        cv2.line(frame, (x_min, y_min), (x_min, y_min + corner_len), (0, 255, 128), 2)
        cv2.line(frame, (x_max, y_min), (x_max - corner_len, y_min), (0, 255, 128), 2)
        cv2.line(frame, (x_max, y_min), (x_max, y_min + corner_len), (0, 255, 128), 2)

        # Label
        cv2.putText(
            frame,
            f"{handedness} Hand",
            (x_min, max(20, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 128),
            1,
            cv2.LINE_AA,
        )

        # Connections
        if HandLandmarksConnections is not None:
            for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                p1 = pixel_pts[conn.start]
                p2 = pixel_pts[conn.end]
                cv2.line(frame, p1, p2, (0, 220, 255), 2, cv2.LINE_AA)

        # Draw joint nodes
        for idx, (px, py) in enumerate(pixel_pts):
            if idx in FINGERTIPS:
                # Fingertips: larger accent circles
                cv2.circle(frame, (px, py), 6, (0, 100, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)
            elif idx == WRIST:
                cv2.circle(frame, (px, py), 7, (255, 0, 128), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), 4, (0, 255, 200), -1, cv2.LINE_AA)

    def close(self) -> None:
        """Release MediaPipe landmarker resources."""
        if hasattr(self, "landmarker") and self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def __del__(self) -> None:
        self.close()
