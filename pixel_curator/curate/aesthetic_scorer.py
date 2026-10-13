#!/usr/bin/env python3
"""
aesthetic_scorer.py
-------------------
Supercharged module for computing overall aesthetic and composition quality.

Enhancements:
- Built-in Computer Vision algorithms to estimate aesthetics directly from images:
  - Lighting Quality: Evaluated via dynamic range, exposure level, and contrast.
  - Color Harmony: Evaluated via HSV hue histogram entropy (lower entropy = more harmonious).
  - Composition Balance: Evaluated via the center of mass of the image's edges/saliency.
  - Subject Presentation: Evaluated via local sharpness (requires bounding box).
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

# --- Default Configuration ---
DEFAULT_WEIGHTS: dict[str, float] = {
    "lighting_quality": 0.30,
    "color_harmony": 0.25,
    "composition_balance": 0.20,
    "subject_presentation": 0.25,
}


@dataclass
class AestheticScores:
    """Container for aesthetic quality metrics with serialization capabilities."""
    lighting_quality: float | None = None
    color_harmony: float | None = None
    composition_balance: float | None = None
    subject_presentation: float | None = None
    overall_aesthetic_score: float | None = None
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


# --- CV Estimation Helpers ---

def estimate_lighting_quality(image: np.ndarray) -> float | None:
    """
    Estimates lighting quality based on exposure and contrast.
    Penalizes extreme over/underexposure and flat lighting.
    """
    if not CV2_AVAILABLE or image is None: return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mean = np.mean(gray)
    std = np.std(gray)

    # Exposure score: 1.0 at 127.5, drops off symmetrically towards 0 or 255
    exposure_score = 1.0 - (abs(mean - 127.5) / 127.5)

    # Contrast score: ~1.0 at std 60+, drops off for flat images
    contrast_score = min(1.0, std / 60.0)

    return float((exposure_score * 0.6) + (contrast_score * 0.4))


def estimate_color_harmony(image: np.ndarray) -> float | None:
    """
    Estimates color harmony using the entropy of the Hue channel.
    Lower entropy (dominant color palette) generally indicates higher harmony.
    """
    if not CV2_AVAILABLE or image is None: return None
    if len(image.shape) != 3: return 0.5 # Grayscale images are neutrally harmonious

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate hue histogram
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    hist = hist / (hist.sum() + 1e-6)

    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    max_entropy = math.log2(180) # Max possible entropy

    # Score: 1.0 (low entropy/harmonious) to 0.0 (high entropy/chaotic)
    return float(1.0 - (entropy / max_entropy))


def estimate_composition_balance(image: np.ndarray) -> float | None:
    """
    Estimates composition balance by finding the center of mass of the edges.
    A centered mass generally indicates a balanced (though perhaps classical) composition.
    """
    if not CV2_AVAILABLE or image is None: return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape

    # Use Canny edges to find visual weight
    edges = cv2.Canny(gray, 50, 150)

    # Calculate moments
    M = cv2.moments(edges)
    if M["m00"] == 0:
        return 0.5 # No edges found, neutral score

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Distance from absolute center
    dist = math.hypot(cx - w/2, cy - h/2)
    max_dist = math.hypot(w/2, h/2)

    # 1.0 if perfectly centered, drops off as it moves away
    return float(1.0 - (dist / max_dist))


def estimate_subject_presentation(image: np.ndarray, subject_bbox: BBox | None = None) -> float | None:
    """
    Estimates subject presentation via local sharpness (Laplacian variance).
    Requires a bounding box. If none provided, returns None.
    """
    if not CV2_AVAILABLE or image is None or subject_bbox is None: return None

    x1, y1, x2, y2 = subject_bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if x2 <= x1 or y2 <= y1: return None

    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Normalize: 1500 variance is extremely sharp
    return float(min(1.0, sharpness / (sharpness + 1500.0)))


# --- Main Logic ---

def compute_overall_aesthetic(
    lighting_quality: float | None = None,
    color_harmony: float | None = None,
    composition_balance: float | None = None,
    subject_presentation: float | None = None,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> AestheticScores:
    """
    Main function to compute aesthetic quality from pre-computed signals.
    Redistributes weights if certain metrics are missing (None).
    """
    # Sanitize inputs
    light_q = _sanitize_score(lighting_quality)
    color_q = _sanitize_score(color_harmony)
    comp_q = _sanitize_score(composition_balance)
    sub_q = _sanitize_score(subject_presentation)

    scores = AestheticScores(
        lighting_quality=light_q,
        color_harmony=color_q,
        composition_balance=comp_q,
        subject_presentation=sub_q,
        weights_used=weights
    )

    components = []
    used_weights = []

    if light_q is not None:
        components.append(light_q)
        used_weights.append(weights.get("lighting_quality", 0.0))
    if color_q is not None:
        components.append(color_q)
        used_weights.append(weights.get("color_harmony", 0.0))
    if comp_q is not None:
        components.append(comp_q)
        used_weights.append(weights.get("composition_balance", 0.0))
    if sub_q is not None:
        components.append(sub_q)
        used_weights.append(weights.get("subject_presentation", 0.0))

    if not components or sum(used_weights) == 0:
        scores.overall_aesthetic_score = 0.0
    else:
        weighted_sum = sum(c * w for c, w in zip(components, used_weights))
        total_weight = sum(used_weights)
        scores.overall_aesthetic_score = float(weighted_sum / total_weight)

    return scores


def compute_aesthetic_from_image(
    image: np.ndarray,
    subject_bbox: BBox | None = None,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> AestheticScores:
    """
    Convenience wrapper: Estimates all aesthetic signals directly from a raw image.
    """
    if image is None or not CV2_AVAILABLE:
        return compute_overall_aesthetic(weights=weights)

    light = estimate_lighting_quality(image)
    color = estimate_color_harmony(image)
    comp = estimate_composition_balance(image)
    sub = estimate_subject_presentation(image, subject_bbox)

    return compute_overall_aesthetic(
        lighting_quality=light,
        color_harmony=color,
        composition_balance=comp,
        subject_presentation=sub,
        weights=weights
    )


def get_aesthetic_score(scores: AestheticScores) -> float:
    """Returns the overall aesthetic score."""
    return scores.overall_aesthetic_score if scores.overall_aesthetic_score is not None else 0.0


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Floats (Backward Compatible):")
    basic_scores = compute_overall_aesthetic(
        lighting_quality=0.8,
        color_harmony=0.9,
        composition_balance=0.6,
        subject_presentation=0.7
    )
    print(f"   Overall Aesthetic: {basic_scores.overall_aesthetic_score:.4f}")
    print(f"   JSON: {basic_scores.to_json()}\n")

    print("2. Testing Missing Data (Weight Redistribution):")
    # If subject presentation is missing, its 0.25 weight is redistributed
    partial_scores = compute_overall_aesthetic(
        lighting_quality=0.8,
        color_harmony=0.9,
        composition_balance=0.6
    )
    print(f"   Overall Aesthetic: {partial_scores.overall_aesthetic_score:.4f} (Should be higher than basic)")
    print(f"   Weights Used: {partial_scores.weights_used}\n")

    if CV2_AVAILABLE:
        print("3. Testing Full CV Pipeline (Estimating from Image Array):")
        # Create a synthetic image: 500x500, blue background (harmonious), bright center (good lighting)
        mock_img = np.zeros((500, 500, 3), dtype=np.uint8)
        mock_img[:] = (200, 100, 50) # Blue-ish background
        cv2.circle(mock_img, (250, 250), 100, (255, 255, 255), -1) # White circle in center

        # Simulate subject bbox exactly over the white circle
        cv_scores = compute_aesthetic_from_image(mock_img, subject_bbox=(150, 150, 350, 350))
        print(f"   Lighting Quality: {cv_scores.lighting_quality:.4f}")
        print(f"   Color Harmony: {cv_scores.color_harmony:.4f}")
        print(f"   Composition Balance: {cv_scores.composition_balance:.4f}")
        print(f"   Subject Presentation: {cv_scores.subject_presentation:.4f}")
        print(f"   Overall Aesthetic: {cv_scores.overall_aesthetic_score:.4f}\n")
