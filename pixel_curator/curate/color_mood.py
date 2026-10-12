#!/usr/bin/env python3
"""
color_mood.py
-------------
Supercharged module for evaluating the overall color mood and temperature of a frame.

Enhancements:
- Built-in Computer Vision algorithms to estimate color metrics directly from images:
  - Color Warmth: Evaluated via the ratio of Red to Blue channels (LAB color space).
  - Color Vibrancy: Evaluated via mean Saturation (HSV color space).
  - Dominant Hue: Evaluated via histogram analysis of the Hue channel (0-360 degrees).
- Mood Categorization: Maps warmth and vibrancy to aesthetic categories 
  (Energetic, Nostalgic, Calm, Electric, Neutral).
- Intelligent Intensity Score: Scores how strongly the mood is expressed, rather than 
  naively averaging warmth and vibrancy.
- Robust input sanitization and JSON serialization for pipeline integration.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict

import numpy as np

# Attempt to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ColorMoodScores:
    """Container for color mood metrics with serialization capabilities."""
    color_warmth: Optional[float] = None          # 0.0 (cool) to 1.0 (warm)
    color_vibrancy: Optional[float] = None        # 0.0 (muted) to 1.0 (vibrant)
    overall_mood_score: Optional[float] = None    # Intensity of the detected mood
    dominant_hue: Optional[float] = None          # 0–360 degrees
    mood_category: str = "unknown"
    confidence: float = 0.0

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

def estimate_color_warmth(image: np.ndarray) -> Optional[float]:
    """
    Estimates color warmth using the CIELAB color space.
    Compares the 'a' channel (Green-Red) and 'b' channel (Blue-Yellow).
    Returns a normalized score where 0.0 is cool and 1.0 is warm.
    """
    if not CV2_AVAILABLE or image is None: return None
    
    # Convert BGR to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 'a' channel: 0 (green) to 255 (red). Center is 128.
    # 'b' channel: 0 (blue) to 255 (yellow). Center is 128.
    a_chan = lab[:, :, 1].astype(np.float32)
    b_chan = lab[:, :, 2].astype(np.float32)
    
    # Deviation from neutral gray. 
    # Positive 'a' = Red, Positive 'b' = Yellow (both contribute to warmth)
    warmth_map = (a_chan - 128.0) + (b_chan - 128.0)
    mean_warmth = float(np.mean(warmth_map))
    
    # Map roughly from [-100 (very cool) to 100 (very warm)] -> [0.0, 1.0]
    return max(0.0, min(1.0, (mean_warmth + 50.0) / 100.0))


def estimate_color_vibrancy(image: np.ndarray) -> Optional[float]:
    """
    Estimates color vibrancy using the mean Saturation in HSV color space.
    """
    if not CV2_AVAILABLE or image is None: return None
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    
    # Mean saturation (0-255) normalized to 0.0-1.0
    return float(np.mean(sat) / 255.0)


def estimate_dominant_hue(image: np.ndarray) -> Optional[float]:
    """
    Estimates the dominant hue in degrees (0-360).
    Ignores pixels with very low saturation (grays/whites/blacks) to find the true color.
    """
    if not CV2_AVAILABLE or image is None: return None
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    
    # Mask out low saturation pixels (e.g., sat < 40)
    color_mask = sat > 40
    if np.count_nonzero(color_mask) < 10:
        return None # Image is essentially grayscale
        
    # Extract hues of colored pixels
    valid_hues = hue[color_mask]
    
    # Histogram of hues (OpenCV hue is 0-179)
    hist = cv2.calcHist([valid_hues], [0], None, [180], [0, 180]).flatten()
    dominant_bin = np.argmax(hist)
    
    # Convert back to 0-360 degrees
    return float(dominant_bin * 2.0)


# --- Main Logic ---

def evaluate_color_mood(
    color_warmth: Optional[float] = None,
    color_vibrancy: Optional[float] = None,
    dominant_hue: Optional[float] = None,
) -> ColorMoodScores:
    """
    Evaluate the color mood of a frame from pre-computed signals.
    Categorizes the mood and calculates an intensity score.
    """
    # Sanitize inputs
    warmth_q = _sanitize_score(color_warmth)
    vibrancy_q = _sanitize_score(color_vibrancy)

    scores = ColorMoodScores(
        color_warmth=warmth_q,
        color_vibrancy=vibrancy_q,
        dominant_hue=dominant_hue,
    )

    if warmth_q is None and vibrancy_q is None:
        scores.confidence = 0.0
        return scores

    warmth = warmth_q if warmth_q is not None else 0.5
    vibrancy = vibrancy_q if vibrancy_q is not None else 0.5

    # 1. Categorize Mood
    is_warm = warmth > 0.55
    is_cool = warmth < 0.45
    is_vibrant = vibrancy > 0.45
    is_muted = vibrancy < 0.25

    if is_warm and is_vibrant:
        scores.mood_category = "Energetic"
    elif is_warm and is_muted:
        scores.mood_category = "Nostalgic"
    elif is_cool and is_vibrant:
        scores.mood_category = "Electric"
    elif is_cool and is_muted:
        scores.mood_category = "Calm"
    elif is_muted:
        scores.mood_category = "Muted"
    elif is_vibrant:
        scores.mood_category = "Vibrant"
    else:
        scores.mood_category = "Neutral"

    # 2. Calculate Mood Intensity Score
    # Mood intensity is high when colors strongly deviate from the "boring" center
    # (Neutral warmth = 0.5, Muted vibrancy = 0.0).
    # We calculate how far the image is from that boring center.
    warmth_deviation = abs(warmth - 0.5) * 2.0  # Scale to 0.0 - 1.0
    vibrancy_level = vibrancy
    
    # Overall intensity is a combination of how vibrant it is AND how far warmth is from neutral
    scores.overall_mood_score = float((warmth_deviation * 0.5) + (vibrancy_level * 0.5))
    scores.confidence = 0.8 if (warmth_q is not None and vibrancy_q is not None) else 0.5

    return scores


def evaluate_color_mood_from_image(
    image: np.ndarray,
) -> ColorMoodScores:
    """
    Convenience wrapper: Estimates all color signals directly from a raw image array.
    """
    if image is None or not CV2_AVAILABLE:
        return evaluate_color_mood()

    warmth = estimate_color_warmth(image)
    vibrancy = estimate_color_vibrancy(image)
    hue = estimate_dominant_hue(image)

    return evaluate_color_mood(
        color_warmth=warmth,
        color_vibrancy=vibrancy,
        dominant_hue=hue
    )


def get_color_mood_score(scores: ColorMoodScores) -> float:
    """Returns the overall color mood intensity score."""
    return scores.overall_mood_score if scores.overall_mood_score is not None else 0.0


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Floats (Backward Compatible):")
    # Warm (0.8) and Vibrant (0.8) -> Energetic
    scores_1 = evaluate_color_mood(color_warmth=0.8, color_vibrancy=0.8)
    print(f"   Mood: {scores_1.mood_category}, Intensity: {scores_1.overall_mood_score:.4f}")
    
    # Cool (0.2) and Muted (0.2) -> Calm
    scores_2 = evaluate_color_mood(color_warmth=0.2, color_vibrancy=0.2)
    print(f"   Mood: {scores_2.mood_category}, Intensity: {scores_2.overall_mood_score:.4f}\n")

    if CV2_AVAILABLE:
        print("2. Testing CV Pipeline - Warm & Vibrant Image (Orange/Yellow):")
        # 100x100 image, highly saturated orange (BGR: 0, 165, 255)
        warm_vibrant_img = np.zeros((100, 100, 3), dtype=np.uint8)
        warm_vibrant_img[:] = (0, 165, 255)
        cv_scores_1 = evaluate_color_mood_from_image(warm_vibrant_img)
        print(f"   Mood: {cv_scores_1.mood_category} (Should be Energetic)")
        print(f"   Warmth: {cv_scores_1.color_warmth:.4f}, Vibrancy: {cv_scores_1.color_vibrancy:.4f}")
        print(f"   Dominant Hue: {cv_scores_1.dominant_hue:.2f} degrees\n")

        print("3. Testing CV Pipeline - Cool & Muted Image (Dark Grayish Blue):")
        # 100x100 image, muted blue (BGR: 100, 80, 50)
        cool_muted_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cool_muted_img[:] = (100, 80, 50)
        cv_scores_2 = evaluate_color_mood_from_image(cool_muted_img)
        print(f"   Mood: {cv_scores_2.mood_category} (Should be Calm/Muted)")
        print(f"   Warmth: {cv_scores_2.color_warmth:.4f}, Vibrancy: {cv_scores_2.color_vibrancy:.4f}\n")

        print("4. Testing CV Pipeline - Neutral/Gray Image:")
        # 100x100 image, solid gray
        gray_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        cv_scores_3 = evaluate_color_mood_from_image(gray_img)
        print(f"   Mood: {cv_scores_3.mood_category} (Should be Neutral/Muted)")
        print(f"   Intensity: {cv_scores_3.overall_mood_score:.4f} (Should be near 0)")
        print(f"   JSON: {cv_scores_3.to_json()}\n")
