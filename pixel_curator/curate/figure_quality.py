#!/usr/bin/env python3
"""
figure_quality.py
-----------------
Supercharged module for computing subject integrity and figure quality metrics.

Enhancements:
- Weighted integrity scoring (configurable).
- Robust input validation (handles NaN, Infinity, and out-of-bounds values).
- JSON serialization for easy pipeline integration (save to DB or JSONL).
- Built-in geometry & CV helpers to compute metrics directly from bounding boxes/images.
- Standardized logging for debugging.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field

import numpy as np

# Attempt to import OpenCV (optional, only needed for raw image sharpness calculation)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Default Configuration ---
DEFAULT_WEIGHTS: dict[str, float] = {
    "face_sharpness": 0.35,
    "body_proportion": 0.20,
    "occlusion": 0.20,
    "centering": 0.15,
    "headroom": 0.10,
}

DEFAULT_MAX_SHARPNESS = 120.0  # Adjust based on your specific camera/sensor setup


@dataclass
class FigureQualityScores:
    """Container for figure quality metrics with serialization capabilities."""
    face_sharpness: float | None = None
    body_proportion: float | None = None
    occlusion: float | None = None
    centering: float | None = None
    headroom: float | None = None
    integrity: float | None = None
    weights_used: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Converts scores to a dictionary, rounding for cleaner output."""
        data = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.items()}

    def to_json(self) -> str:
        """Serializes scores to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> FigureQualityScores:
        """Creates a FigureQualityScores instance from a dictionary."""
        valid_keys = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_keys)


def _sanitize_score(value: float | None) -> float | None:
    """Validates and clamps scores to [0.0, 1.0]. Returns None if invalid."""
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        logger.warning(f"Invalid score type ({type(value)}). Expected float. Ignoring.")
        return None
    if math.isnan(value) or math.isinf(value):
        logger.warning("NaN or Infinity encountered in score. Ignoring.")
        return None
    return max(0.0, min(1.0, float(value)))


def normalize_face_sharpness(value: float, max_reasonable: float = DEFAULT_MAX_SHARPNESS) -> float:
    """
    Normalize face sharpness into [0, 1] range using an asymptotic function.
    """
    if value <= 0:
        return 0.0
    return min(1.0, value / (value + max_reasonable))


def compute_integrity(
    scores: FigureQualityScores,
    weights: dict[str, float] = DEFAULT_WEIGHTS
) -> float:
    """
    Calculate overall subject integrity score using weighted averaging.
    Redistributes weights if certain metrics are missing (None).
    """
    components = []
    used_weights = []

    # Face Sharpness
    if scores.face_sharpness is not None:
        components.append(normalize_face_sharpness(scores.face_sharpness))
        used_weights.append(weights.get("face_sharpness", 0.0))

    # Body Proportion
    if scores.body_proportion is not None:
        components.append(scores.body_proportion)
        used_weights.append(weights.get("body_proportion", 0.0))

    # Occlusion
    if scores.occlusion is not None:
        components.append(scores.occlusion)
        used_weights.append(weights.get("occlusion", 0.0))

    # Centering
    if scores.centering is not None:
        components.append(scores.centering)
        used_weights.append(weights.get("centering", 0.0))

    # Headroom
    if scores.headroom is not None:
        components.append(scores.headroom)
        used_weights.append(weights.get("headroom", 0.0))

    if not components or sum(used_weights) == 0:
        return 0.0

    # Calculate weighted average
    weighted_sum = sum(c * w for c, w in zip(components, used_weights))
    total_weight = sum(used_weights)

    return float(weighted_sum / total_weight)


# --- Geometric & CV Computation Helpers ---

def calculate_centering(
    face_bbox: tuple[int, int, int, int],
    img_width: int,
    img_height: int
) -> float:
    """
    Calculates centering score based on distance of face center from image center.
    Bbox format: (x_min, y_min, x_max, y_max)
    """
    x1, y1, x2, y2 = face_bbox
    face_cx = (x1 + x2) / 2.0
    face_cy = (y1 + y2) / 2.0

    img_cx = img_width / 2.0
    img_cy = img_height / 2.0

    # Euclidean distance between centers
    distance = math.hypot(face_cx - img_cx, face_cy - img_cy)

    # Max possible distance from center is half the diagonal
    max_distance = math.hypot(img_cx, img_cy)
    if max_distance == 0:
        return 0.0

    # Closer to center = higher score
    return 1.0 - min(1.0, (distance / max_distance))


def calculate_headroom(
    face_bbox: tuple[int, int, int, int],
    img_height: int,
    ideal_ratio: float = 0.15
) -> float:
    """
    Calculates headroom score. Penalizes too much or too little headroom.
    Bbox format: (x_min, y_min, x_max, y_max)
    """
    y1 = face_bbox[1]
    headroom_px = y1
    total_height = img_height

    if total_height == 0:
        return 0.0

    headroom_ratio = headroom_px / total_height

    # Using a simple gaussian-like falloff. 1.0 at ideal_ratio, drops off as it moves away.
    # Tune the spread (0.1) based on strictness requirements
    spread = 0.1
    score = math.exp(-((headroom_ratio - ideal_ratio) ** 2) / (2 * spread ** 2))
    return float(score)


def calculate_face_sharpness(image_array: np.ndarray) -> float:
    """
    Computes face sharpness using the variance of the Laplacian.
    Requires OpenCV. Expects a cropped image array of the face (BGR or Grayscale).
    """
    if not CV2_AVAILABLE:
        logger.error("OpenCV (cv2) is required to calculate face sharpness but is not installed.")
        return 0.0

    if image_array is None or image_array.size == 0:
        return 0.0

    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array

    # Variance of Laplacian (standard blur detection)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(sharpness)


# --- Main Entry Point ---

def compute_figure_quality(
    face_sharpness: float | None = None,
    body_proportion: float | None = None,
    occlusion: float | None = None,
    centering: float | None = None,
    headroom: float | None = None,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> FigureQualityScores:
    """
    Main entry point. Computes all figure quality metrics and returns
    a FigureQualityScores object with individual scores + overall integrity.
    
    Note: Inputs should ideally be in [0.0, 1.0] range, except face_sharpness
    which is normalized internally.
    """
    # Sanitize all inputs
    scores = FigureQualityScores(
        face_sharpness=_sanitize_score(face_sharpness),
        body_proportion=_sanitize_score(body_proportion),
        occlusion=_sanitize_score(occlusion),
        centering=_sanitize_score(centering),
        headroom=_sanitize_score(headroom),
        weights_used=weights
    )

    scores.integrity = compute_integrity(scores, weights)
    return scores


def get_integrity_score(scores: FigureQualityScores) -> float:
    """Convenience function to get just the integrity score."""
    if scores.integrity is not None:
        return scores.integrity
    return compute_integrity(scores, scores.weights_used or DEFAULT_WEIGHTS)


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing basic weighted computation (Backward Compatible):")
    basic_scores = compute_figure_quality(
        face_sharpness=85.5,  # Raw variance score
        body_proportion=0.9,
        occlusion=0.8,
        centering=0.95,
        headroom=0.85
    )
    print(f"   Integrity: {basic_scores.integrity:.4f}")
    print(f"   JSON Output: {basic_scores.to_json()}\n")

    print("2. Testing missing data (weights redistribute automatically):")
    partial_scores = compute_figure_quality(
        face_sharpness=150.0,
        centering=0.8,
        # Occlusion, headroom, body_proportion missing
    )
    print(f"   Integrity: {partial_scores.integrity:.4f}")
    print(f"   Scores Dict: {partial_scores.to_dict()}\n")

    print("3. Testing bad data (NaN / Out of bounds):")
    bad_scores = compute_figure_quality(
        face_sharpness=float('nan'),  # Will be ignored
        body_proportion=1.5,          # Will be clamped to 1.0
        centering=-0.2                # Will be clamped to 0.0
    )
    print(f"   Integrity: {bad_scores.integrity:.4f}")
    print(f"   Cleaned Scores: {bad_scores.to_dict()}\n")

    print("4. Testing Geometry Helpers:")
    # Simulated image: 1920x1080. Face located at top-left-ish
    mock_bbox = (800, 150, 1100, 450)
    centering = calculate_centering(mock_bbox, 1920, 1080)
    headroom = calculate_headroom(mock_bbox, 1080)
    print(f"   Centering Score: {centering:.4f}")
    print(f"   Headroom Score: {headroom:.4f}")
