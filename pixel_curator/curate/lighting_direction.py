#!/usr/bin/env python3
"""
lighting_direction.py
---------------------
Supercharged module for evaluating lighting direction and character.

Enhancements:
- Built-in Computer Vision algorithms to estimate lighting directly from images:
  - Shadow Strength: Evaluated via local dynamic range (99th vs 1st percentile).
  - Side Lighting: Evaluated via left vs. right luminance gradient.
  - Overhead Lighting: Evaluated via top vs. bottom luminance gradient.
  - Rim Lighting: Evaluated by comparing border brightness to the subject's center.
- Configurable thresholds for categorization (dramatic, side, overhead, rim, soft).
- Dynamic confidence scoring based on distance from category boundaries.
- Robust input sanitization and JSON serialization for pipeline integration.
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
class LightingThresholds:
    """Configurable thresholds for lighting classification."""
    dramatic_shadow: float = 0.60   # 0-1.0 dynamic range required for dramatic lighting
    soft_shadow: float = 0.30       # Below this is considered flat/soft lighting
    side_gradient: float = 0.15     # 15% luminance difference between left/right
    overhead_gradient: float = 0.15 # 15% luminance difference between top/bottom
    rim_brightness: float = 0.80    # Border must be 80% as bright as max luminance to be rim light
    rim_margin: float = 0.20        # Border must be 20% brighter than the center to be rim light


@dataclass
class LightingDirectionScores:
    """Container for lighting direction classification results with serialization."""
    lighting_type: str = "unknown"
    confidence: float = 0.0
    has_rim_light: Optional[bool] = None
    has_strong_side_light: Optional[bool] = None
    is_overhead: Optional[bool] = None
    shadow_strength: Optional[float] = None
    left_right_gradient: Optional[float] = None
    top_bottom_gradient: Optional[float] = None
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

def estimate_lighting_signals(image: np.ndarray, bbox: Optional[BBox]) -> Dict[str, Optional[float]]:
    """
    Analyzes a subject crop to determine luminance gradients and shadow strength.
    """
    if not CV2_AVAILABLE or image is None or bbox is None:
        return {}

    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if x2 <= x1 or y2 <= y1: return {}

    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    gray_f = gray.astype(np.float32) / 255.0

    # 1. Shadow Strength (Local Dynamic Range ignoring sensor noise)
    p1, p99 = np.percentile(gray_f, 1), np.percentile(gray_f, 99)
    shadow_strength = float(max(0.0, min(1.0, p99 - p1)))

    # 2. Left vs Right Gradient (Side Lighting)
    mid_x = gray_f.shape[1] // 2
    left_mean = np.mean(gray_f[:, :mid_x])
    right_mean = np.mean(gray_f[:, mid_x:])
    # Positive = Left is brighter, Negative = Right is brighter
    lr_gradient = float((left_mean - right_mean))

    # 3. Top vs Bottom Gradient (Overhead Lighting)
    mid_y = gray_f.shape[0] // 2
    top_mean = np.mean(gray_f[:mid_y, :])
    bottom_mean = np.mean(gray_f[mid_y:, :])
    # Positive = Top is brighter (Overhead), Negative = Bottom is brighter (Underlighting)
    tb_gradient = float((top_mean - bottom_mean))

    # 4. Rim Lighting Heuristic
    # Compare the brightness of the outer 10% border to the inner 50% center
    border_h = int(gray_f.shape[0] * 0.1)
    border_w = int(gray_f.shape[1] * 0.1)
    
    if border_h > 0 and border_w > 0:
        # Create a mask for the center
        mask = np.zeros_like(gray_f, dtype=bool)
        mask[border_h:-border_h, border_w:-border_w] = True
        
        border_mean = np.mean(gray_f[~mask])
        center_mean = np.mean(gray_f[mask])
        max_lum = np.max(gray_f)
        
        # Rim light if border is significantly brighter than center, and close to max lum
        is_rim = bool(border_mean > center_mean * 1.2 and border_mean > max_lum * 0.8)
    else:
        is_rim = False

    return {
        "shadow_strength": shadow_strength,
        "left_right_gradient": lr_gradient,
        "top_bottom_gradient": tb_gradient,
        "has_rim_light": is_rim
    }


# --- Main Logic ---

def classify_lighting_direction(
    shadow_strength: Optional[float] = None,
    has_rim_light: Optional[bool] = None,
    has_strong_side_light: Optional[bool] = None,
    is_overhead: Optional[bool] = None,
    left_right_gradient: Optional[float] = None,
    top_bottom_gradient: Optional[float] = None,
    thresholds: LightingThresholds = LightingThresholds(),
) -> LightingDirectionScores:
    """
    Classify the direction and character of lighting dynamically.
    Supports both pre-computed booleans and geometric signals.
    """
    thresh_dict = asdict(thresholds)
    
    # Infer booleans from gradients if gradients are provided
    if left_right_gradient is not None and has_strong_side_light is None:
        has_strong_side_light = abs(left_right_gradient) > thresholds.side_gradient
    if top_bottom_gradient is not None and is_overhead is None:
        is_overhead = top_bottom_gradient > thresholds.overhead_gradient

    scores = LightingDirectionScores(
        shadow_strength=_sanitize_score(shadow_strength),
        has_rim_light=has_rim_light,
        has_strong_side_light=has_strong_side_light,
        is_overhead=is_overhead,
        left_right_gradient=left_right_gradient,
        top_bottom_gradient=top_bottom_gradient,
        thresholds_used=thresh_dict,
    )

    if scores.shadow_strength is None and not has_rim_light and not has_strong_side_light and not is_overhead:
        scores.lighting_type = "unknown"
        scores.confidence = 0.0
        return scores

    # 1. Rim Lighting (High priority aesthetic trait)
    if has_rim_light:
        scores.lighting_type = "rim"
        scores.confidence = 0.75
        if scores.shadow_strength and scores.shadow_strength > thresholds.dramatic_shadow:
            scores.lighting_type = "dramatic"
            scores.confidence = 0.85
        return scores

    # 2. Dramatic Lighting (High shadow strength)
    if scores.shadow_strength is not None and scores.shadow_strength >= thresholds.dramatic_shadow:
        scores.lighting_type = "dramatic"
        scores.confidence = 0.5 + (0.5 * (scores.shadow_strength - thresholds.dramatic_shadow) / (1.0 - thresholds.dramatic_shadow))
        # If it also has strong side light, it's likely split lighting
        if has_strong_side_light:
            scores.lighting_type = "side"
            scores.confidence = min(1.0, scores.confidence + 0.1)
        return scores

    # 3. Side Lighting
    if has_strong_side_light:
        scores.lighting_type = "side"
        # Confidence based on gradient magnitude
        if left_right_gradient is not None:
            scores.confidence = min(1.0, abs(left_right_gradient) / 0.4) 
        else:
            scores.confidence = 0.65
        return scores

    # 4. Overhead Lighting
    if is_overhead:
        scores.lighting_type = "overhead"
        if top_bottom_gradient is not None:
            scores.confidence = min(1.0, top_bottom_gradient / 0.4)
        else:
            scores.confidence = 0.6
        return scores

    # 5. Soft / Flat Lighting
    if scores.shadow_strength is not None and scores.shadow_strength < thresholds.soft_shadow:
        scores.lighting_type = "soft"
        scores.confidence = 1.0 - (scores.shadow_strength / thresholds.soft_shadow)
        return scores

    # 6. Ambiguous Fallback
    scores.lighting_type = "ambiguous"
    scores.confidence = 0.4
    return scores


def classify_lighting_from_image(
    image: np.ndarray,
    bbox: Optional[BBox] = None,
    thresholds: LightingThresholds = LightingThresholds(),
) -> LightingDirectionScores:
    """
    Convenience wrapper: Estimates all lighting signals directly from a raw image,
    then classifies the lighting direction.
    """
    if image is None or not CV2_AVAILABLE or bbox is None:
        return classify_lighting_direction(thresholds=thresholds)

    signals = estimate_lighting_signals(image, bbox)

    return classify_lighting_direction(
        shadow_strength=signals.get("shadow_strength"),
        has_rim_light=signals.get("has_rim_light"),
        left_right_gradient=signals.get("left_right_gradient"),
        top_bottom_gradient=signals.get("top_bottom_gradient"),
        thresholds=thresholds
    )


def get_lighting_type(scores: LightingDirectionScores) -> str:
    """Returns the classified lighting type."""
    return scores.lighting_type


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Signals (Backward Compatible):")
    legacy_scores = classify_lighting_direction(shadow_strength=0.8, has_rim_light=True)
    print(f"   Type: {legacy_scores.lighting_type}, Confidence: {legacy_scores.confidence:.2f}\n")

    if CV2_AVAILABLE:
        print("2. Testing CV Pipeline - Side Lighting (Left to Right gradient):")
        # 300x600 crop. Left half is bright (200), right half is dark (50)
        mock_side_img = np.zeros((600, 300, 3), dtype=np.uint8)
        mock_side_img[:, :150] = (200, 200, 200)
        mock_side_img[:, 150:] = (50, 50, 50)
        
        side_scores = classify_lighting_from_image(mock_side_img, bbox=(0, 0, 300, 600))
        print(f"   Type: {side_scores.lighting_type} (Should be side)")
        print(f"   Shadow Strength: {side_scores.shadow_strength:.4f}")
        print(f"   L/R Gradient: {side_scores.left_right_gradient:.4f}\n")

        print("3. Testing CV Pipeline - Overhead Lighting:")
        # 600x300 crop. Top half is bright, bottom half is dark
        mock_overhead_img = np.zeros((300, 600, 3), dtype=np.uint8)
        mock_overhead_img[:150, :] = (200, 200, 200)
        mock_overhead_img[150:, :] = (80, 80, 80)
        
        overhead_scores = classify_lighting_from_image(mock_overhead_img, bbox=(0, 0, 600, 300))
        print(f"   Type: {overhead_scores.lighting_type} (Should be overhead)")
        print(f"   T/B Gradient: {overhead_scores.top_bottom_gradient:.4f}\n")

        print("4. Testing CV Pipeline - Soft/Flat Lighting:")
        # Uniformly lit, low contrast image
        mock_soft_img = np.ones((500, 500, 3), dtype=np.uint8) * 128
        # Add slight noise so it's not perfectly 0 variance
        np.random.seed(42)
        mock_soft_img += np.random.randint(-10, 10, (500, 500, 3), dtype=np.int8)
        
        soft_scores = classify_lighting_from_image(mock_soft_img, bbox=(0, 0, 500, 500))
        print(f"   Type: {soft_scores.lighting_type} (Should be soft)")
        print(f"   Shadow Strength: {soft_scores.shadow_strength:.4f}\n")

        print("5. Testing CV Pipeline - Rim Lighting:")
        # Dark center, bright borders
        mock_rim_img = np.zeros((500, 500, 3), dtype=np.uint8)
        mock_rim_img[:] = 20 # Dark center
        mock_rim_img[:50, :] = 240 # Top border
        mock_rim_img[-50:, :] = 240 # Bottom border
        mock_rim_img[:, :50] = 240 # Left border
        mock_rim_img[:, -50:] = 240 # Right border
        
        rim_scores = classify_lighting_from_image(mock_rim_img, bbox=(0, 0, 500, 500))
        print(f"   Type: {rim_scores.lighting_type} (Should be rim)")
        print(f"   Has Rim Light: {rim_scores.has_rim_light}")
