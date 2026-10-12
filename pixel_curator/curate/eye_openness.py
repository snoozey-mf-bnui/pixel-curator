#!/usr/bin/env python3
"""
eye_openness.py
---------------
Supercharged module for evaluating eye openness in a frame.

Enhancements:
- Built-in geometric analysis: Automatically calculates Eye Aspect Ratio (EAR) 
  directly from 6-point eye contours (e.g., MediaPipe, dlib, OpenPose).
- Biologically accurate scoring: Maps EAR values to a 0.0-1.0 openness scale using 
  standard anatomical thresholds.
- State classification: Accurately flags `is_eyes_closed` (blink) and `is_squinting`.
- Configurable thresholds for strict vs. loose blink/squint detection.
- Robust input sanitization, dynamic confidence scoring, and JSON serialization.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)

# 6-point eye contour format: List[Tuple[float, float]]
# Standard order: [outer_corner, top_outer, top_inner, inner_corner, bottom_inner, bottom_outer]
EyePoints = List[Tuple[float, float]]


@dataclass
class EyeOpennessThresholds:
    """Configurable thresholds based on standard Eye Aspect Ratio (EAR) values."""
    # EAR typically ranges from ~0.1 (closed) to ~0.35 (wide open).
    closed_ear: float = 0.15       # Below this = eyes closed (blink)
    squint_ear: float = 0.22       # Below this = squinting
    wide_open_ear: float = 0.30    # Above this = wide open (surprised)


@dataclass
class EyeOpennessScores:
    """Container for eye openness metrics with serialization."""
    eye_openness: Optional[float] = None        # 0.0 (closed) to 1.0 (fully open)
    is_eyes_closed: Optional[bool] = None
    is_squinting: Optional[bool] = None
    ear_value: Optional[float] = None           # Raw Eye Aspect Ratio
    confidence: float = 0.0
    thresholds_used: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Converts scores to a dictionary, rounding for cleaner output."""
        data = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.items()}

    def to_json(self) -> str:
        """Serializes scores to a JSON string."""
        return json.dumps(self.to_dict())


def _sanitize_score(value: Optional[float]) -> Optional[float]:
    """Validates and clamps scores to [0.0, 1.0]. Returns None if invalid."""
    if value is None: return None
    if not isinstance(value, (int, float)): return None
    if math.isnan(value) or math.isinf(value): return None
    return max(0.0, min(1.0, float(value)))


def _dist(p1: Optional[Tuple[float, float]], p2: Optional[Tuple[float, float]]) -> float:
    """Euclidean distance between two points."""
    if not p1 or not p2: return 0.0
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# --- Geometric Estimation Helpers ---

def calculate_ear(eye_points: EyePoints) -> Optional[float]:
    """
    Calculates the Eye Aspect Ratio (EAR).
    Formula: (||p2-p6|| + ||p3-p5||) / (2.0 * ||p1-p4||)
    Where p1 is outer corner, p4 is inner corner, p2/p3 are top lid, p5/p6 are bottom lid.
    """
    if not eye_points or len(eye_points) != 6:
        return None
        
    p1, p2, p3, p4, p5, p6 = eye_points
    
    # Vertical distances
    A = _dist(p2, p6)
    B = _dist(p3, p5)
    
    # Horizontal distance
    C = _dist(p1, p4)
    
    if C < 1e-6:
        return None
        
    ear = (A + B) / (2.0 * C)
    return float(ear)


def estimate_openness_from_landmarks(
    left_eye: Optional[EyePoints] = None,
    right_eye: Optional[EyePoints] = None,
    thresholds: EyeOpennessThresholds = EyeOpennessThresholds()
) -> Dict[str, Optional[float]]:
    """
    Estimates eye openness from 6-point eye contours.
    Returns a dictionary of raw signals.
    """
    ears = []
    if left_eye:
        ear_l = calculate_ear(left_eye)
        if ear_l is not None: ears.append(ear_l)
    if right_eye:
        ear_r = calculate_ear(right_eye)
        if ear_r is not None: ears.append(ear_r)
        
    if not ears:
        return {}
        
    avg_ear = float(np.mean(ears))
    
    # Map EAR to a 0.0-1.0 openness score
    # < closed_ear = 0.0, > wide_open_ear = 1.0
    range_val = thresholds.wide_open_ear - thresholds.closed_ear
    if range_val <= 0: range_val = 0.15 # Fallback
        
    openness = (avg_ear - thresholds.closed_ear) / range_val
    openness = max(0.0, min(1.0, openness))
    
    return {
        "ear_value": avg_ear,
        "eye_openness": openness,
        "is_eyes_closed": avg_ear < thresholds.closed_ear,
        "is_squinting": thresholds.closed_ear <= avg_ear < thresholds.squint_ear
    }


# --- Main Logic ---

def evaluate_eye_openness(
    eye_openness: Optional[float] = None,
    is_eyes_closed: Optional[bool] = None,
    is_squinting: Optional[bool] = None,
    ear_value: Optional[float] = None,
    thresholds: EyeOpennessThresholds = EyeOpennessThresholds(),
) -> EyeOpennessScores:
    """
    Evaluate eye openness from pre-computed signals or raw EAR.
    Redistributes confidence based on available data.
    """
    open_q = _sanitize_score(eye_openness)

    scores = EyeOpennessScores(
        eye_openness=open_q,
        is_eyes_closed=is_eyes_closed,
        is_squinting=is_squinting,
        ear_value=ear_value,
        thresholds_used=asdict(thresholds),
    )

    # If we have raw EAR, use it to infer everything else (overrides legacy floats)
    if ear_value is not None:
        range_val = thresholds.wide_open_ear - thresholds.closed_ear
        if range_val > 0:
            scores.eye_openness = max(0.0, min(1.0, (ear_value - thresholds.closed_ear) / range_val))
        scores.is_eyes_closed = ear_value < thresholds.closed_ear
        scores.is_squinting = thresholds.closed_ear <= ear_value < thresholds.squint_ear
        scores.confidence = 0.9
    else:
        # Fallback to legacy logic
        if open_q is not None:
            scores.is_eyes_closed = open_q < 0.2
            scores.is_squinting = 0.2 <= open_q < 0.5
            scores.confidence = 0.7
        elif is_eyes_closed is not None:
            scores.eye_openness = 0.1 if is_eyes_closed else 0.8
            scores.confidence = 0.6
        else:
            scores.confidence = 0.0

    return scores


def evaluate_from_landmarks(
    left_eye: Optional[EyePoints] = None,
    right_eye: Optional[EyePoints] = None,
    thresholds: EyeOpennessThresholds = EyeOpennessThresholds(),
) -> EyeOpennessScores:
    """
    Convenience wrapper: Estimates all eye openness signals directly from 
    6-point eye contour landmarks.
    """
    signals = estimate_openness_from_landmarks(left_eye, right_eye, thresholds)

    return evaluate_eye_openness(
        eye_openness=signals.get("eye_openness"),
        is_eyes_closed=signals.get("is_eyes_closed"),
        is_squinting=signals.get("is_squinting"),
        ear_value=signals.get("ear_value"),
        thresholds=thresholds
    )


def get_eye_openness_score(scores: EyeOpennessScores) -> float:
    """Returns the eye openness score (0.0 = closed, 1.0 = fully open)."""
    return scores.eye_openness if scores.eye_openness is not None else 0.5


def are_eyes_closed(scores: EyeOpennessScores, threshold: float = 0.2) -> bool:
    """Returns True if the eyes are considered closed."""
    if scores.ear_value is not None:
        # Use raw EAR for precision if available
        return scores.ear_value < 0.15 
    if scores.eye_openness is not None:
        return scores.eye_openness < threshold
    return scores.is_eyes_closed if scores.is_eyes_closed is not None else False


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Floats (Backward Compatible):")
    legacy_scores = evaluate_eye_openness(eye_openness=0.1)
    print(f"   Openness: {legacy_scores.eye_openness}, Closed: {legacy_scores.is_eyes_closed}")
    print(f"   JSON: {legacy_scores.to_json()}\n")

    print("2. Testing Geometric Landmarks (Wide Open Eyes):")
    # Simulating a wide open eye. Horizontal dist = 100, Vertical dist = 40.
    # EAR = (40 + 40) / (2 * 100) = 0.40
    open_eye_pts = [
        (0, 50), (25, 30), (75, 30), (100, 50), (75, 70), (25, 70)
    ]
    open_scores = evaluate_from_landmarks(left_eye=open_eye_pts, right_eye=open_eye_pts)
    print(f"   EAR: {open_scores.ear_value:.4f} (Should be ~0.4)")
    print(f"   Openness: {open_scores.eye_openness:.4f} (Should be 1.0)")
    print(f"   Squinting: {open_scores.is_squinting}\n")

    print("3. Testing Geometric Landmarks (Squinting Eyes):")
    # Simulating a squint. Horizontal dist = 100, Vertical dist = 20.
    # EAR = (20 + 20) / (2 * 100) = 0.20
    squint_eye_pts = [
        (0, 50), (25, 40), (75, 40), (100, 50), (75, 60), (25, 60)
    ]
    squint_scores = evaluate_from_landmarks(left_eye=squint_eye_pts, right_eye=squint_eye_pts)
    print(f"   EAR: {squint_scores.ear_value:.4f} (Should be ~0.2)")
    print(f"   Openness: {squint_scores.eye_openness:.4f}")
    print(f"   Squinting: {squint_scores.is_squinting} (Should be True)\n")

    print("4. Testing Geometric Landmarks (Closed Eyes / Blink):")
    # Simulating a blink. Horizontal dist = 100, Vertical dist = 5.
    # EAR = (5 + 5) / (2 * 100) = 0.05
    closed_eye_pts = [
        (0, 50), (25, 45), (75, 45), (100, 50), (75, 55), (25, 55)
    ]
    closed_scores = evaluate_from_landmarks(left_eye=closed_eye_pts, right_eye=closed_eye_pts)
    print(f"   EAR: {closed_scores.ear_value:.4f} (Should be ~0.05)")
    print(f"   Openness: {closed_scores.eye_openness:.4f} (Should be 0.0)")
    print(f"   Closed: {closed_scores.is_eyes_closed} (Should be True)\n")
