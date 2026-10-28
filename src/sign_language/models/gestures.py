"""Kinematic and geometric gesture analyzer for American Sign Language (ASL) and hand signs."""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# Landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


def euclidean_dist(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Euclidean distance in 3D coordinate space."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


class GestureAnalyzer:
    """Analyzes 21 3D hand landmarks to extract finger states, pinch contacts, and gesture semantics."""

    @classmethod
    def analyze_hand(cls, raw_landmarks: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        """Extract geometric characteristics from 21 landmarks."""
        if len(raw_landmarks) < 21:
            return {}

        wrist = raw_landmarks[WRIST]
        ref_scale = euclidean_dist(wrist, raw_landmarks[MIDDLE_MCP])
        if ref_scale < 1e-4:
            ref_scale = 1.0

        # Finger extension check: compare distance from tip to wrist vs pip to wrist
        def is_finger_extended(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
            d_tip_wrist = euclidean_dist(raw_landmarks[tip_idx], wrist)
            d_pip_wrist = euclidean_dist(raw_landmarks[pip_idx], wrist)
            d_tip_mcp = euclidean_dist(raw_landmarks[tip_idx], raw_landmarks[mcp_idx])
            d_pip_mcp = euclidean_dist(raw_landmarks[pip_idx], raw_landmarks[mcp_idx])
            return d_tip_wrist > d_pip_wrist and d_tip_mcp > d_pip_mcp

        # Thumb extension: check distance of thumb tip to pinky MCP and index MCP
        d_thumb_pinky_mcp = euclidean_dist(raw_landmarks[THUMB_TIP], raw_landmarks[PINKY_MCP]) / ref_scale
        d_thumb_index_mcp = euclidean_dist(raw_landmarks[THUMB_TIP], raw_landmarks[INDEX_MCP]) / ref_scale
        d_thumb_wrist = euclidean_dist(raw_landmarks[THUMB_TIP], wrist) / ref_scale
        thumb_extended = (d_thumb_pinky_mcp > 1.1 and d_thumb_index_mcp > 0.6) or (d_thumb_wrist > 1.3)

        index_extended = is_finger_extended(INDEX_TIP, INDEX_PIP, INDEX_MCP)
        middle_extended = is_finger_extended(MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
        ring_extended = is_finger_extended(RING_TIP, RING_PIP, RING_MCP)
        pinky_extended = is_finger_extended(PINKY_TIP, PINKY_PIP, PINKY_MCP)

        # Pairwise fingertip distances normalized by hand scale
        thumb_tip = raw_landmarks[THUMB_TIP]
        index_tip = raw_landmarks[INDEX_TIP]
        middle_tip = raw_landmarks[MIDDLE_TIP]
        ring_tip = raw_landmarks[RING_TIP]
        pinky_tip = raw_landmarks[PINKY_TIP]

        d_thumb_index = euclidean_dist(thumb_tip, index_tip) / ref_scale
        d_thumb_middle = euclidean_dist(thumb_tip, middle_tip) / ref_scale
        d_thumb_ring = euclidean_dist(thumb_tip, ring_tip) / ref_scale
        d_thumb_pinky = euclidean_dist(thumb_tip, pinky_tip) / ref_scale

        d_index_middle = euclidean_dist(index_tip, middle_tip) / ref_scale
        d_middle_ring = euclidean_dist(middle_tip, ring_tip) / ref_scale
        d_ring_pinky = euclidean_dist(ring_tip, pinky_tip) / ref_scale

        # Pinch detection: threshold ~ 0.35 of hand reference scale
        pinching_index = d_thumb_index < 0.35
        pinching_middle = d_thumb_middle < 0.35

        return {
            "scale": ref_scale,
            "thumb_extended": thumb_extended,
            "index_extended": index_extended,
            "middle_extended": middle_extended,
            "ring_extended": ring_extended,
            "pinky_extended": pinky_extended,
            "d_thumb_index": d_thumb_index,
            "d_thumb_middle": d_thumb_middle,
            "d_thumb_ring": d_thumb_ring,
            "d_thumb_pinky": d_thumb_pinky,
            "d_index_middle": d_index_middle,
            "d_middle_ring": d_middle_ring,
            "d_ring_pinky": d_ring_pinky,
            "pinching_index": pinching_index,
            "pinching_middle": pinching_middle,
        }

    @classmethod
    def classify_gesture(cls, raw_landmarks: List[Tuple[float, float, float]]) -> Tuple[str, float]:
        """Classify hand gesture into ASL alphabet letters, numbers, or phrases with confidence score."""
        geom = cls.analyze_hand(raw_landmarks)
        if not geom:
            return "blank", 0.0

        t = geom["thumb_extended"]
        i = geom["index_extended"]
        m = geom["middle_extended"]
        r = geom["ring_extended"]
        p = geom["pinky_extended"]

        d_ti = geom["d_thumb_index"]
        d_tm = geom["d_thumb_middle"]
        d_im = geom["d_index_middle"]
        pinch_index = geom["pinching_index"]
        pinch_middle = geom["pinching_middle"]

        extended_count = sum([i, m, r, p])

        # --- High-Priority Specific ASL Configurations ---

        # "I Love You" (ILY): Thumb, Index, Pinky extended; Middle and Ring folded
        if t and i and not m and not r and p:
            return "ILY", 0.95

        # "Rock / Horns": Index and Pinky extended, Middle and Ring folded, Thumb folded/clamped
        if not t and i and not m and not r and p:
            return "Rock", 0.92

        # "Y" (Hang Loose): Thumb and Pinky extended, Index, Middle, Ring folded
        if t and not i and not m and not r and p:
            return "Y", 0.95

        # "L": Thumb and Index extended at angle, others folded
        if t and i and not m and not r and not p and d_ti > 0.6:
            return "L", 0.95

        # "OK" sign / "F": Thumb and Index touching to form loop, Middle, Ring, Pinky extended
        if pinch_index and m and r and p:
            return "OK", 0.94

        # "D": Index extended straight up, Thumb touches Middle tip, Ring & Pinky folded
        if i and not m and not r and not p and (d_tm < 0.45 or pinch_middle):
            return "D", 0.92

        # "I": Pinky extended straight up, all other fingers (and thumb) folded into fist
        if not t and not i and not m and not r and p:
            return "I", 0.95

        # "V" / "2" vs "U": Index and Middle extended
        if not t and i and m and not r and not p:
            if d_im > 0.4:
                return "V", 0.93  # Separated = V
            else:
                return "U", 0.91  # Together = U

        # "W" / "3" / "6": Index, Middle, Ring extended, Pinky folded
        if i and m and r and not p:
            if t:
                return "W", 0.92
            return "6", 0.88

        # "B" / "4": 4 fingers extended upright (Index, Middle, Ring, Pinky), Thumb folded across palm
        if not t and i and m and r and p:
            return "B", 0.94

        # "5" / "Open Hand" / "Hello": All 5 fingers extended and spread
        if t and i and m and r and p:
            return "Hello", 0.96

        # "1": Only Index extended straight up
        if not t and i and not m and not r and not p:
            return "1", 0.93

        # "C" / "O": Curved fingers forming an arc or circle
        if not i and not m and not r and not p:
            # Fist variants: A, S, E, T, M, N
            wrist = raw_landmarks[WRIST]
            thumb_tip = raw_landmarks[THUMB_TIP]
            index_mcp = raw_landmarks[INDEX_MCP]
            ref_scale = geom["scale"]

            # If thumb tip is high alongside index MCP -> "A"
            if thumb_tip[1] < index_mcp[1] and d_ti < 0.6:
                return "A", 0.90

            # If thumb is folded across the front of fingers -> "S"
            if d_tm < 0.4:
                return "S", 0.88

            # Default closed fist -> "A" or "E"
            return "A", 0.85

        # "C": Partially curled fingers
        if d_ti > 0.35 and d_ti < 0.75 and not r and not p:
            return "C", 0.86

        # Generic fallback
        if extended_count == 0:
            return "A", 0.80

        return "blank", 0.0
