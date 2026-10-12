#!/usr/bin/env python3
"""
background_quality.py
---------------------
Supercharged module for evaluating the quality and cleanliness of the background.

Enhancements:
- Built-in Computer Vision algorithms to estimate background metrics directly from images:
  - Background Cleanliness: Evaluated via Canny edge density in the background region.
  - Subject Contrast: Evaluated via mean luminance difference between subject and background.
  - Distraction Heuristic: Flags distracting backgrounds if edge density and color variance are high.
- Robust input sanitization (handles NaN, Infinity, and out-of-bounds values).
- Hard penalty logic: Explicitly flags and caps scores for highly distracting backgrounds.
- Dynamic confidence scoring and JSON serialization for pipeline integration.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Tuple

import numpy as np

# Attempt to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

# Bbox format: Tuple[int, int, int, int] -> (x_min, y_min, x_max, y_max)
BBox = Tuple[int, int, int, int]


@dataclass
class BackgroundThresholds:
    """Configurable thresholds for background classification."""
    distracting_edge_density: float = 0.15  # >15% edges in BG = busy/distracting
    good_contrast: float = 40.0             # Luminance diff > 40 = good contrast


@dataclass
class BackgroundQualityScores:
    """Container for background quality metrics with serialization capabilities."""
    background_cleanliness: Optional[float] = None
    subject_background_contrast: Optional[float] = None
    has_distracting_background: Optional[bool] = None
    overall_background_score: Optional[float] = None
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


# --- CV Estimation Helpers ---

def estimate_background_signals(
    image: np.ndarray, 
    subject_bbox: Optional[BBox],
    thresholds: BackgroundThresholds = BackgroundThresholds()
) -> Dict[str, Optional[float]]:
    """
    Analyzes the image outside the subject bbox to determine cleanliness and contrast.
    """
    if not CV2_AVAILABLE or image is None or subject_bbox is None:
        return {}

    h, w = image.shape[:2]
    x1, y1, x2, y2 = subject_bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    # cv2.rectangle requires integer point tuples; detection bboxes are floats.
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x2 <= x1 or y2 <= y1:
        return {}

    # Create a mask for the background (everything outside the subject bbox)
    bg_mask = np.ones((h, w), dtype=np.uint8) * 255
    cv2.rectangle(bg_mask, (x1, y1), (x2, y2), 0, -1)

    bg_pixels = np.count_nonzero(bg_mask)
    if bg_pixels < 10:
        return {} # Not enough background to analyze

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 1. Background Cleanliness (Edge Density)
    edges = cv2.Canny(gray, 50, 150)
    bg_edges = cv2.bitwise_and(edges, edges, mask=bg_mask)
    edge_density = np.count_nonzero(bg_edges) / bg_pixels
    
    # Map edge density to cleanliness score (0.15 density -> ~0.0 cleanliness)
    cleanliness = max(0.0, 1.0 - (edge_density / thresholds.distracting_edge_density))

    # 2. Subject-Background Contrast (Luminance Difference)
    subject_crop = gray[y1:y2, x1:x2]
    if subject_crop.size == 0:
        contrast = 0.0
    else:
        subject_mean = np.mean(subject_crop)
        bg_mean = np.mean(gray[bg_mask > 0])
        lum_diff = abs(subject_mean - bg_mean)
        # Normalize: diff of 100+ is maximum contrast
        contrast = min(1.0, lum_diff / 100.0)

    # 3. Distraction Heuristic
    # If the background has high texture (edges) AND high color variance, it's distracting.
    bg_color = image[bg_mask > 0]
    color_std = np.std(bg_color) if bg_color.size > 0 else 0
    
    is_distracting = bool(edge_density > thresholds.distracting_edge_density and color_std > 40.0)

    return {
        "cleanliness": float(cleanliness),
        "contrast": float(contrast),
        "is_distracting": is_distracting
    }


# --- Main Logic ---

def evaluate_background_quality(
    background_cleanliness: Optional[float] = None,
    subject_background_contrast: Optional[float] = None,
    has_distracting_background: Optional[bool] = None,
    thresholds: BackgroundThresholds = BackgroundThresholds(),
) -> BackgroundQualityScores:
    """
    Evaluate background quality from pre-computed signals.
    Redistributes weights if certain metrics are missing (None).
    """
    clean_q = _sanitize_score(background_cleanliness)
    contrast_q = _sanitize_score(subject_background_contrast)

    scores = BackgroundQualityScores(
        background_cleanliness=clean_q,
        subject_background_contrast=contrast_q,
        has_distracting_background=has_distracting_background,
        thresholds_used=asdict(thresholds),
    )

    components = []
    used_weights = []

    if clean_q is not None:
        components.append(clean_q)
        used_weights.append(0.6)
    if contrast_q is not None:
        components.append(contrast_q)
        used_weights.append(0.4)

    # Hard fail: If explicitly flagged as distracting, cap score
    if has_distracting_background is True:
        if not components:
            scores.overall_background_score = 0.2
        else:
            raw_score = sum(c * w for c, w in zip(components, used_weights)) / sum(used_weights)
            scores.overall_background_score = min(0.3, raw_score)
        scores.confidence = 0.8
        return scores

    if not components or sum(used_weights) == 0:
        scores.overall_background_score = 0.0
        scores.confidence = 0.0
    else:
        weighted_sum = sum(c * w for c, w in zip(components, used_weights))
        total_weight = sum(used_weights)
        scores.overall_background_score = float(weighted_sum / total_weight)
        # Confidence scales with how many metrics we evaluated
        scores.confidence = 0.4 + (0.2 * len(components))

    return scores


def evaluate_background_from_image(
    image: np.ndarray,
    subject_bbox: Optional[BBox] = None,
    thresholds: BackgroundThresholds = BackgroundThresholds(),
) -> BackgroundQualityScores:
    """
    Convenience wrapper: Estimates all background signals directly from a raw image,
    then evaluates the background quality.
    """
    if image is None or not CV2_AVAILABLE or subject_bbox is None:
        return evaluate_background_quality(thresholds=thresholds)

    signals = estimate_background_signals(image, subject_bbox, thresholds)

    return evaluate_background_quality(
        background_cleanliness=signals.get("cleanliness"),
        subject_background_contrast=signals.get("contrast"),
        has_distracting_background=signals.get("is_distracting"),
        thresholds=thresholds
    )


def get_background_score(scores: BackgroundQualityScores) -> float:
    """Returns the overall background quality score."""
    return scores.overall_background_score if scores.overall_background_score is not None else 0.0


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Floats (Backward Compatible):")
    # Clean (0.9), High Contrast (0.8), Not Distracting
    good_scores = evaluate_background_quality(0.9, 0.8, False)
    print(f"   Score: {good_scores.overall_background_score:.4f} (Should be high)")
    print(f"   JSON: {good_scores.to_json()}\n")

    print("2. Testing Distracting Hard Penalty:")
    # Clean (0.8), Contrast (0.8), BUT Distracting = True
    bad_scores = evaluate_background_quality(0.8, 0.8, True)
    print(f"   Score: {bad_scores.overall_background_score:.4f} (Should be capped at 0.3)\n")

    if CV2_AVAILABLE:
        print("3. Testing CV Pipeline - Clean Studio Background:")
        # 500x500 image. Gray background (128). White subject box (255) in center.
        mock_clean_img = np.ones((500, 500, 3), dtype=np.uint8) * 128
        mock_clean_img[200:300, 200:300] = 255
        
        clean_cv_scores = evaluate_background_from_image(mock_clean_img, subject_bbox=(200, 200, 300, 300))
        print(f"   Cleanliness: {clean_cv_scores.background_cleanliness:.4f} (Should be ~1.0)")
        print(f"   Contrast: {clean_cv_scores.subject_background_contrast:.4f} (Should be high)")
        print(f"   Distracting: {clean_cv_scores.has_distracting_background}\n")

        print("4. Testing CV Pipeline - Busy/Distracting Background:")
        # 500x500 image. Random noise background. Solid subject box in center.
        mock_busy_img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        mock_busy_img[200:300, 200:300] = (200, 50, 50) # Solid red subject
        
        busy_cv_scores = evaluate_background_from_image(mock_busy_img, subject_bbox=(200, 200, 300, 300))
        print(f"   Cleanliness: {busy_cv_scores.background_cleanliness:.4f} (Should be near 0)")
        print(f"   Distracting: {busy_cv_scores.has_distracting_background} (Should be True)")
        print(f"   Overall Score: {busy_cv_scores.overall_background_score:.4f} (Should be capped at 0.3)")
