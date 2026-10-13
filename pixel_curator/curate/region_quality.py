#!/usr/bin/env python3
"""
region_quality.py
-----------------
Supercharged module for computing region-specific quality scores for image frames.

Enhancements:
- Built-in CV helpers to automatically crop and compute regional sharpness 
  directly from raw images and bounding boxes.
- Robust input sanitization (handles NaN, Infinity, and out-of-bounds values).
- Weight tracking: Saves the weights used in the dataclass for full auditability.
- JSON serialization for pipeline integration (save to DB or JSONL).
- Flexible threshold checks (e.g., require ALL regions to pass, or ANY region).
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

# Bbox format: Tuple[int, int, int, int] -> (x_min, y_min, x_max, y_max)
BBox = tuple[int, int, int, int]


@dataclass
class RegionQualityScores:
    """Container for region-specific quality metrics with serialization capabilities."""
    face_quality: float | None = None
    body_quality: float | None = None
    overall_quality: float | None = None
    combined_regional_score: float | None = None
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


def normalize_sharpness(value: float, max_reasonable: float = 120.0) -> float:
    """
    Normalize sharpness (Laplacian variance) into [0, 1] range.
    """
    if value <= 0:
        return 0.0
    return min(1.0, value / (value + max_reasonable))


def compute_region_sharpness(
    image: np.ndarray,
    bbox: BBox | None,
    max_reasonable: float = 120.0
) -> float | None:
    """
    Crops an image to the provided bounding box and computes normalized sharpness.
    """
    if not CV2_AVAILABLE:
        logger.error("OpenCV (cv2) is required to compute region sharpness but is not installed.")
        return None

    if image is None or bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # Clamp coordinates to image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if x2 <= x1 or y2 <= y1:
        logger.warning(f"Invalid bounding box after clamping: {bbox}")
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Convert to grayscale if needed
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    # Variance of Laplacian
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return normalize_sharpness(sharpness, max_reasonable)


def compute_regional_score(
    face_quality: float | None = None,
    body_quality: float | None = None,
    overall_quality: float | None = None,
    face_weight: float = 0.4,
    body_weight: float = 0.4,
    overall_weight: float = 0.2,
) -> RegionQualityScores:
    """
    Calculate region-specific quality scores and a weighted combined score.
    Redistributes weights automatically if certain regions are missing (None).
    """
    # Sanitize inputs
    face_q = _sanitize_score(face_quality)
    body_q = _sanitize_score(body_quality)
    overall_q = _sanitize_score(overall_quality)

    weights_map = {
        "face_quality": face_weight,
        "body_quality": body_weight,
        "overall_quality": overall_weight,
    }

    scores = RegionQualityScores(
        face_quality=face_q,
        body_quality=body_q,
        overall_quality=overall_q,
        weights_used=weights_map,
    )

    components = []
    used_weights = []

    if face_q is not None:
        components.append(face_q)
        used_weights.append(face_weight)

    if body_q is not None:
        components.append(body_q)
        used_weights.append(body_weight)

    if overall_q is not None:
        components.append(overall_q)
        used_weights.append(overall_weight)

    if not components or sum(used_weights) == 0:
        scores.combined_regional_score = 0.0
    else:
        weighted_sum = sum(c * w for c, w in zip(components, used_weights))
        total_weight = sum(used_weights)
        scores.combined_regional_score = float(weighted_sum / total_weight)

    return scores


def compute_from_bboxes(
    image: np.ndarray,
    face_bbox: BBox | None = None,
    body_bbox: BBox | None = None,
    face_weight: float = 0.4,
    body_weight: float = 0.4,
    overall_weight: float = 0.2,
    max_sharpness: float = 120.0
) -> RegionQualityScores:
    """
    Convenience wrapper: Auto-calculates regional sharpness directly from 
    an image and bounding boxes.
    """
    if image is None:
        return compute_regional_score(
            face_weight=face_weight,
            body_weight=body_weight,
            overall_weight=overall_weight
        )

    # Compute overall image sharpness
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    overall_raw = cv2.Laplacian(gray, cv2.CV_64F).var()
    overall_q = normalize_sharpness(overall_raw, max_sharpness)

    # Compute regional sharpness
    face_q = compute_region_sharpness(image, face_bbox, max_sharpness)
    body_q = compute_region_sharpness(image, body_bbox, max_sharpness)

    return compute_regional_score(
        face_quality=face_q,
        body_quality=body_q,
        overall_quality=overall_q,
        face_weight=face_weight,
        body_weight=body_weight,
        overall_weight=overall_weight,
    )


def get_combined_regional_score(scores: RegionQualityScores) -> float:
    """Returns the combined regional quality score."""
    return scores.combined_regional_score if scores.combined_regional_score is not None else 0.0


def has_strong_face_and_body(scores: RegionQualityScores, threshold: float = 0.6) -> bool:
    """
    Returns True if both face and body quality meet or exceed the threshold.
    Useful for strict selection criteria.
    """
    face_ok = scores.face_quality is not None and scores.face_quality >= threshold
    body_ok = scores.body_quality is not None and scores.body_quality >= threshold
    return face_ok and body_ok


def meets_regional_criteria(
    scores: RegionQualityScores,
    threshold: float = 0.6,
    require_all: bool = True
) -> bool:
    """
    Flexible criteria check. 
    If require_all=True, all available region scores must pass.
    If require_all=False, any available region score must pass.
    """
    available_scores = [
        s for s in [scores.face_quality, scores.body_quality, scores.overall_quality]
        if s is not None
    ]

    if not available_scores:
        return False

    if require_all:
        return all(s >= threshold for s in available_scores)
    else:
        return any(s >= threshold for s in available_scores)


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Scores (Backward Compatible):")
    # Face is sharp, body is blurry, overall is okay
    basic_scores = compute_regional_score(
        face_quality=0.9,
        body_quality=0.2,
        overall_quality=0.6
    )
    print(f"   Combined Score: {basic_scores.combined_regional_score:.4f}")
    print(f"   Strong Face & Body? {has_strong_face_and_body(basic_scores)}")
    print(f"   JSON: {basic_scores.to_json()}\n")

    print("2. Testing Missing Data (Weight Redistribution):")
    # Only face is provided, so face_weight (0.4) becomes 100% of the calculation
    partial_scores = compute_regional_score(face_quality=0.8)
    print(f"   Combined Score: {partial_scores.combined_regional_score:.4f} (Should be 0.8)")
    print(f"   Weights Used: {partial_scores.weights_used}\n")

    print("3. Testing Bad Data (NaN / Out of Bounds):")
    bad_scores = compute_regional_score(
        face_quality=float('nan'),  # Ignored
        body_quality=1.5,           # Clamped to 1.0
        overall_quality=-0.5        # Clamped to 0.0
    )
    print(f"   Face: {bad_scores.face_quality}, Body: {bad_scores.body_quality}, Overall: {bad_scores.overall_quality}")
    print(f"   Combined: {bad_scores.combined_regional_score:.4f}\n")

    if CV2_AVAILABLE:
        print("4. Testing CV Pipeline (Auto-crop from Bounding Boxes):")
        # Simulate a 500x500 image with random noise (high sharpness)
        np.random.seed(42)
        mock_image = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)

        # Face bbox (top left), Body bbox (center)
        face_bbox = (100, 100, 200, 200)
        body_bbox = (150, 150, 350, 400)

        cv_scores = compute_from_bboxes(mock_image, face_bbox, body_bbox)
        print(f"   Face Quality: {cv_scores.face_quality:.4f}")
        print(f"   Body Quality: {cv_scores.body_quality:.4f}")
        print(f"   Overall Quality: {cv_scores.overall_quality:.4f}")
        print(f"   All Pass 0.5 Threshold? {meets_regional_criteria(cv_scores, 0.5, require_all=True)}\n")
