#!/usr/bin/env python3
"""
color_cast_detector.py
----------------------
Supercharged module for detecting strong color casts in frames.

Enhancements:
- Built-in Computer Vision algorithms to estimate color cast directly from images:
  - CIELAB Color Space Analysis: Isolates 'a' (Green-Red) and 'b' (Blue-Yellow) channels.
  - Cast Strength: Evaluated via asymptotic deviation from neutral gray (128).
  - Dominant Cast Classification: Accurately labels the cast (yellow, blue, green, red, none).
- Configurable thresholds for strict vs. loose cast detection.
- Dynamic confidence scoring based on the severity of the deviation.
- Robust input sanitization and JSON serialization for pipeline integration.
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


@dataclass
class ColorCastThresholds:
    """Configurable thresholds for color cast classification."""
    # Minimum deviation from 128 in LAB space to be considered a cast
    cast_detection_limit: float = 15.0
    # Threshold for strong cast (0.0 to 1.0 normalized strength)
    strong_cast_threshold: float = 0.35


@dataclass
class ColorCastScores:
    """Container for color cast detection results with serialization."""
    cast_strength: float | None = None       # 0.0 (neutral) to 1.0 (strong cast)
    dominant_cast: str | None = None         # yellow, blue, green, red, none
    is_cast_detected: bool | None = None
    confidence: float = 0.0
    thresholds_used: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Converts scores to a dictionary, rounding for cleaner output."""
        data = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.items()}

    def to_json(self) -> str:
        """Serializes scores to a JSON string."""
        return json.dumps(self.to_dict())


def _sanitize_score(value: float | None) -> float | None:
    """Validates and clamps scores to [0.0, 1.0]. Returns None if invalid."""
    if value is None: return None
    if not isinstance(value, (int, float)): return None
    if math.isnan(value) or math.isinf(value): return None
    return max(0.0, min(1.0, float(value)))


# --- CV Estimation Helpers ---

def estimate_color_cast_signals(
    image: np.ndarray,
    thresholds: ColorCastThresholds = ColorCastThresholds()
) -> dict[str, float | None]:
    """
    Analyzes image in CIELAB color space to determine color cast.
    The 'a' channel maps Green(0) to Red(255). The 'b' maps Blue(0) to Yellow(255).
    Neutral white balance is 128 on both.
    """
    if not CV2_AVAILABLE or image is None:
        return {}

    # Convert to LAB color space
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    a_chan = lab[:, :, 1].astype(np.float32)
    b_chan = lab[:, :, 2].astype(np.float32)

    # Calculate mean deviation from neutral (128)
    a_dev = float(np.mean(a_chan) - 128.0)
    b_dev = float(np.mean(b_chan) - 128.0)

    # The maximum absolute deviation dictates the cast strength
    max_dev = max(abs(a_dev), abs(b_dev))

    # Normalize strength asymptotically: reaches ~0.5 at 40 dev, ~0.8 at 120 dev
    strength = max_dev / (max_dev + 40.0)

    # Determine dominant cast direction
    dominant_cast = "none"
    is_cast = False
    if max_dev > thresholds.cast_detection_limit:
        is_cast = True
        if abs(a_dev) > abs(b_dev):
            # 'a' channel deviation is stronger
            dominant_cast = "red" if a_dev > 0 else "green"
        else:
            # 'b' channel deviation is stronger
            dominant_cast = "yellow" if b_dev > 0 else "blue"

    return {
        "cast_strength": float(strength),
        "dominant_cast": dominant_cast,
        "is_cast_detected": is_cast
    }


# --- Main Logic ---

def detect_color_cast(
    cast_strength: float | None = None,
    dominant_cast: str | None = None,
    is_cast_detected: bool | None = None,
    thresholds: ColorCastThresholds = ColorCastThresholds(),
) -> ColorCastScores:
    """
    Evaluate color cast from pre-computed signals.
    Redistributes confidence based on available data.
    """
    str_q = _sanitize_score(cast_strength)

    scores = ColorCastScores(
        cast_strength=str_q,
        dominant_cast=dominant_cast,
        is_cast_detected=is_cast_detected,
        thresholds_used=asdict(thresholds),
    )

    # Infer is_cast_detected from strength if not provided
    if scores.is_cast_detected is None and str_q is not None:
        scores.is_cast_detected = str_q > thresholds.strong_cast_threshold

    # Infer cast_strength from boolean if not provided
    if str_q is None and scores.is_cast_detected is not None:
        scores.cast_strength = 0.6 if scores.is_cast_detected else 0.15

    # Set dominant_cast to none if explicitly not a cast
    if scores.dominant_cast is None:
        scores.dominant_cast = "none" if not scores.is_cast_detected else "unknown"

    # Calculate dynamic confidence
    if scores.cast_strength is not None and scores.dominant_cast not in [None, "unknown", "none"]:
        scores.confidence = 0.9
    elif scores.cast_strength is not None:
        scores.confidence = 0.7
    elif scores.is_cast_detected is not None:
        scores.confidence = 0.5
    else:
        scores.confidence = 0.0

    return scores


def detect_color_cast_from_image(
    image: np.ndarray,
    thresholds: ColorCastThresholds = ColorCastThresholds(),
) -> ColorCastScores:
    """
    Convenience wrapper: Estimates all color cast signals directly from a raw image array.
    """
    if image is None or not CV2_AVAILABLE:
        return detect_color_cast(thresholds=thresholds)

    signals = estimate_color_cast_signals(image, thresholds)

    return detect_color_cast(
        cast_strength=signals.get("cast_strength"),
        dominant_cast=signals.get("dominant_cast"),
        is_cast_detected=signals.get("is_cast_detected"),
        thresholds=thresholds
    )


def get_cast_strength(scores: ColorCastScores) -> float:
    """Returns the color cast strength (0.0 = neutral, 1.0 = strong cast)."""
    return scores.cast_strength if scores.cast_strength is not None else 0.0


def has_strong_color_cast(scores: ColorCastScores, threshold: float = 0.35) -> bool:
    """Returns True if a strong color cast is detected."""
    if scores.cast_strength is not None:
        return scores.cast_strength >= threshold
    return scores.is_cast_detected if scores.is_cast_detected is not None else False


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Signals (Backward Compatible):")
    legacy_scores = detect_color_cast(cast_strength=0.8, dominant_cast="yellow")
    print(f"   Strength: {legacy_scores.cast_strength}, Cast: {legacy_scores.dominant_cast}")
    print(f"   JSON: {legacy_scores.to_json()}\n")

    if CV2_AVAILABLE:
        print("2. Testing CV Pipeline - Neutral/White Balanced Image:")
        # Standard 200x200 gray image (128, 128, 128)
        neutral_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        neutral_scores = detect_color_cast_from_image(neutral_img)
        print(f"   Strength: {neutral_scores.cast_strength:.4f} (Should be ~0.0)")
        print(f"   Dominant Cast: {neutral_scores.dominant_cast} (Should be none)\n")

        print("3. Testing CV Pipeline - Heavy Yellow/Orange Cast:")
        # Warm indoor lighting simulation
        yellow_img = np.ones((200, 200, 3), dtype=np.uint8)
        yellow_img[:, :] = (50, 120, 200) # High Red, Low Blue -> Yellow/Orange
        yellow_scores = detect_color_cast_from_image(yellow_img)
        print(f"   Strength: {yellow_scores.cast_strength:.4f} (Should be high)")
        print(f"   Dominant Cast: {yellow_scores.dominant_cast} (Should be yellow)\n")

        print("4. Testing CV Pipeline - Heavy Blue Cast:")
        # Cold lighting simulation
        blue_img = np.ones((200, 200, 3), dtype=np.uint8)
        blue_img[:, :] = (200, 120, 50) # High Blue, Low Red -> Blue
        blue_scores = detect_color_cast_from_image(blue_img)
        print(f"   Strength: {blue_scores.cast_strength:.4f} (Should be high)")
        print(f"   Dominant Cast: {blue_scores.dominant_cast} (Should be blue)\n")

        print("5. Testing CV Pipeline - Heavy Green Cast:")
        # Fluorescent lighting simulation
        green_img = np.ones((200, 200, 3), dtype=np.uint8)
        green_img[:, :] = (50, 200, 50) # High Green -> Green
        green_scores = detect_color_cast_from_image(green_img)
        print(f"   Strength: {green_scores.cast_strength:.4f} (Should be high)")
        print(f"   Dominant Cast: {green_scores.dominant_cast} (Should be green)")
        print(f"   Is Strong Cast: {has_strong_color_cast(green_scores)}\n")
