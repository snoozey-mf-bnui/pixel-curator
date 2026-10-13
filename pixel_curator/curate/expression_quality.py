#!/usr/bin/env python3
"""
expression_quality.py
---------------------
Supercharged module for evaluating facial expression quality and aesthetic presentation.

Enhancements:
- Built-in geometric analysis: Automatically calculates expression intensity (mouth/eyebrow 
  displacement) and aesthetic presentation (facial symmetry) directly from landmarks.
- Built-in CV analysis: Calculates expression clarity via local face sharpness (Laplacian).
- Robust input sanitization (handles NaN, Infinity, and out-of-bounds values).
- Weight tracking and redistribution (if a metric is missing, its weight is redistributed).
- JSON serialization for pipeline integration.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field

import numpy as np

# Attempt to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

# Bbox format: Tuple[int, int, int, int] -> (x_min, y_min, x_max, y_max)
BBox = tuple[int, int, int, int]
# Point format: Tuple[float, float] -> (x, y)
Point = tuple[float, float]

# --- Default Configuration ---
DEFAULT_WEIGHTS: dict[str, float] = {
    "expression_intensity": 0.35,
    "expression_clarity": 0.30,
    "expression_aesthetic": 0.35,
}


@dataclass
class ExpressionScores:
    """Container for expression quality metrics with serialization capabilities."""
    expression_intensity: float | None = None
    expression_clarity: float | None = None
    expression_aesthetic: float | None = None
    overall_expression_score: float | None = None
    weights_used: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Converts scores to a dictionary, rounding for cleaner output."""
        data = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.items()}

    def to_json(self) -> str:
        """Serializes scores to a JSON string."""
        return json.dumps(self.to_dict())


def _sanitize_score(value: float | None) -> float | None:
    """Validates and clamps scores to [0.0, 1.0]. Returns None if invalid."""
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return max(0.0, min(1.0, float(value)))


def _calculate_distance(p1: Point | None, p2: Point | None) -> float:
    """Euclidean distance between two points."""
    if not p1 or not p2: return 0.0
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# --- Geometric & CV Estimation Helpers ---

def estimate_expression_intensity(
    left_eye: Point | None = None,
    right_eye: Point | None = None,
    mouth_top: Point | None = None,
    mouth_bottom: Point | None = None,
    mouth_left: Point | None = None,
    mouth_right: Point | None = None,
    left_brow: Point | None = None,
    right_brow: Point | None = None,
) -> float | None:
    """
    Estimates expression intensity based on facial landmark displacement.
    Normalized by interocular distance to be scale-invariant.
    """
    if not left_eye or not right_eye:
        logger.warning("Missing eye landmarks for expression intensity calculation.")
        return None

    iod = _calculate_distance(left_eye, right_eye)  # Interocular distance
    if iod < 1e-6:
        return 0.0

    intensity_signals = []

    # 1. Mouth Openness (Surprise, Joy, Anger)
    if mouth_top and mouth_bottom:
        mouth_open = _calculate_distance(mouth_top, mouth_bottom) / iod
        # Typical mouth open ratio is 0.1 to 0.5 for strong expressions
        intensity_signals.append(min(1.0, mouth_open / 0.4))

    # 2. Mouth Stretch (Smile, Frown)
    if mouth_left and mouth_right:
        mouth_width = _calculate_distance(mouth_left, mouth_right) / iod
        # Typical mouth width ratio is 0.4 (neutral) to 0.7 (wide smile)
        intensity_signals.append(max(0.0, min(1.0, (mouth_width - 0.4) / 0.3)))

    # 3. Eyebrow Raise (Surprise, Fear)
    if left_brow and right_brow:
        left_raise = _calculate_distance(left_brow, left_eye) / iod
        right_raise = _calculate_distance(right_brow, right_eye) / iod
        avg_raise = (left_raise + right_raise) / 2.0
        # Typical brow raise ratio is 0.1 (neutral) to 0.3 (raised)
        intensity_signals.append(max(0.0, min(1.0, (avg_raise - 0.1) / 0.2)))

    if not intensity_signals:
        return None

    # Average of available signals
    return float(np.mean(intensity_signals))


def estimate_expression_aesthetic(
    left_eye: Point | None = None,
    right_eye: Point | None = None,
    mouth_left: Point | None = None,
    mouth_right: Point | None = None,
    left_brow: Point | None = None,
    right_brow: Point | None = None,
) -> float | None:
    """
    Estimates expression aesthetic based on facial symmetry.
    Higher symmetry often correlates with perceived aesthetic balance in expressions.
    """
    if not left_eye or not right_eye or not mouth_left or not mouth_right:
        logger.warning("Missing landmarks for expression aesthetic calculation.")
        return None

    iod = _calculate_distance(left_eye, right_eye)
    if iod < 1e-6:
        return 0.0

    # Find midline x-coordinate between the eyes
    mid_x = (left_eye[0] + right_eye[0]) / 2.0

    # Calculate horizontal asymmetry from the midline (normalized by IOD)
    mouth_asym = abs((mouth_left[0] - mid_x) + (mid_x - mouth_right[0])) / iod
    brow_asym = abs((left_brow[0] - mid_x) + (mid_x - right_brow[0])) / iod

    # Also check vertical asymmetry (e.g., one eyebrow raised higher)
    brow_vert_asym = abs(left_brow[1] - right_brow[1]) / iod

    # Total asymmetry (lower is better)
    total_asymmetry = (mouth_asym + brow_asym + brow_vert_asym) / 3.0

    # Convert to aesthetic score: 1.0 (perfectly symmetrical) to 0.0 (highly asymmetrical)
    # 0.2 asymmetry is noticeable, so we scale against that
    return float(max(0.0, 1.0 - (total_asymmetry / 0.2)))


def estimate_expression_clarity(image: np.ndarray, face_bbox: BBox | None) -> float | None:
    """
    Estimates expression clarity via local face sharpness (Laplacian variance).
    Blurry faces have unreadable expressions.
    """
    if not CV2_AVAILABLE or image is None or face_bbox is None:
        return None

    x1, y1, x2, y2 = face_bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: 1000 variance is extremely sharp
    return float(min(1.0, sharpness / (sharpness + 1000.0)))


# --- Main Logic ---

def compute_overall_expression(
    expression_intensity: float | None = None,
    expression_clarity: float | None = None,
    expression_aesthetic: float | None = None,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> ExpressionScores:
    """
    Main function to compute expression quality from pre-computed signals.
    Redistributes weights if certain metrics are missing (None).
    """
    # Sanitize inputs
    intensity_q = _sanitize_score(expression_intensity)
    clarity_q = _sanitize_score(expression_clarity)
    aesthetic_q = _sanitize_score(expression_aesthetic)

    scores = ExpressionScores(
        expression_intensity=intensity_q,
        expression_clarity=clarity_q,
        expression_aesthetic=aesthetic_q,
        weights_used=weights
    )

    components = []
    used_weights = []

    if intensity_q is not None:
        components.append(intensity_q)
        used_weights.append(weights.get("expression_intensity", 0.0))
    if clarity_q is not None:
        components.append(clarity_q)
        used_weights.append(weights.get("expression_clarity", 0.0))
    if aesthetic_q is not None:
        components.append(aesthetic_q)
        used_weights.append(weights.get("expression_aesthetic", 0.0))

    if not components or sum(used_weights) == 0:
        scores.overall_expression_score = 0.0
    else:
        weighted_sum = sum(c * w for c, w in zip(components, used_weights))
        total_weight = sum(used_weights)
        scores.overall_expression_score = float(weighted_sum / total_weight)

    return scores


def compute_expression_from_signals(
    image: np.ndarray | None = None,
    face_bbox: BBox | None = None,
    landmarks: dict[str, Point] | None = None,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> ExpressionScores:
    """
    Convenience wrapper: Estimates all expression signals directly from image and landmarks.
    Expects 'landmarks' dict to contain keys like 'left_eye', 'mouth_top', etc.
    """
    intensity = None
    aesthetic = None
    clarity = None

    if landmarks:
        intensity = estimate_expression_intensity(
            left_eye=landmarks.get("left_eye"),
            right_eye=landmarks.get("right_eye"),
            mouth_top=landmarks.get("mouth_top"),
            mouth_bottom=landmarks.get("mouth_bottom"),
            mouth_left=landmarks.get("mouth_left"),
            mouth_right=landmarks.get("mouth_right"),
            left_brow=landmarks.get("left_brow"),
            right_brow=landmarks.get("right_brow"),
        )
        aesthetic = estimate_expression_aesthetic(
            left_eye=landmarks.get("left_eye"),
            right_eye=landmarks.get("right_eye"),
            mouth_left=landmarks.get("mouth_left"),
            mouth_right=landmarks.get("mouth_right"),
            left_brow=landmarks.get("left_brow"),
            right_brow=landmarks.get("right_brow"),
        )

    if image is not None and face_bbox is not None:
        clarity = estimate_expression_clarity(image, face_bbox)

    return compute_overall_expression(
        expression_intensity=intensity,
        expression_clarity=clarity,
        expression_aesthetic=aesthetic,
        weights=weights
    )


def get_expression_score(scores: ExpressionScores) -> float:
    """Returns the overall expression quality score."""
    return scores.overall_expression_score if scores.overall_expression_score is not None else 0.0


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Floats (Backward Compatible):")
    basic_scores = compute_overall_expression(
        expression_intensity=0.8,
        expression_clarity=0.9,
        expression_aesthetic=0.7
    )
    print(f"   Overall Expression: {basic_scores.overall_expression_score:.4f}")
    print(f"   JSON: {basic_scores.to_json()}\n")

    print("2. Testing Missing Data (Weight Redistribution):")
    partial_scores = compute_overall_expression(
        expression_intensity=0.8,
        expression_clarity=0.9
    )
    print(f"   Overall Expression: {partial_scores.overall_expression_score:.4f}\n")

    print("3. Testing Geometric Landmark Calculation (Surprise Expression):")
    # Simulating landmarks for a surprised face (wide eyes, open mouth, raised brows)
    # IOD = 100px. Mouth open = 60px (0.6 ratio -> maxed at 1.0)
    mock_landmarks = {
        "left_eye": (100, 100),
        "right_eye": (200, 100),
        "mouth_top": (150, 160),
        "mouth_bottom": (150, 220),
        "mouth_left": (120, 190),
        "mouth_right": (180, 190),
        "left_brow": (100, 60),
        "right_brow": (200, 60)
    }
    geo_scores = compute_expression_from_signals(landmarks=mock_landmarks)
    print(f"   Intensity: {geo_scores.expression_intensity:.4f} (Should be high)")
    print(f"   Aesthetic: {geo_scores.expression_aesthetic:.4f} (Should be symmetrical ~1.0)")
    print(f"   Overall: {geo_scores.overall_expression_score:.4f}\n")

    if CV2_AVAILABLE:
        print("4. Testing Full CV + Landmark Pipeline:")
        # Simulated 300x300 face crop
        np.random.seed(42)
        mock_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        full_scores = compute_expression_from_signals(
            image=mock_img,
            face_bbox=(0, 0, 300, 300),
            landmarks=mock_landmarks
        )
        print(f"   Clarity: {full_scores.expression_clarity:.4f}")
        print(f"   Overall: {full_scores.overall_expression_score:.4f}")
