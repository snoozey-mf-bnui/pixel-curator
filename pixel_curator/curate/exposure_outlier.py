#!/usr/bin/env python3
"""
exposure_outlier.py
-------------------
Supercharged module for detecting overexposed or underexposed frames.

Enhancements:
- Built-in Computer Vision histogram analysis: Automatically calculates exposure 
  level and clipping ratios directly from image arrays.
- Clipping Detection: Distinguishes between artistic high-key/low-key lighting 
  and truly damaged overexposed/underexposed frames by measuring blown-out highlights 
  and crushed blacks.
- Configurable thresholds for clipping percentages and mean brightness.
- Dynamic confidence scoring based on the severity of the clipping.
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
class ExposureThresholds:
    """Configurable thresholds for exposure classification."""
    # Percentage of pixels that must be pure white/black to be considered clipped
    max_white_clip: float = 0.05  # 5% blown-out whites
    max_black_clip: float = 0.05  # 5% crushed blacks

    # Absolute mean brightness limits (0.0 to 1.0) regardless of clipping
    extreme_bright_limit: float = 0.95
    extreme_dark_limit: float = 0.05


@dataclass
class ExposureScores:
    """Container for exposure outlier metrics with serialization."""
    exposure_level: float | None = None      # 0.0 (very dark) to 1.0 (very bright)
    is_overexposed: bool | None = None
    is_underexposed: bool | None = None
    white_clip_ratio: float | None = None    # % of pixels at 255
    black_clip_ratio: float | None = None    # % of pixels at 0
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

def estimate_exposure_signals(
    image: np.ndarray,
    thresholds: ExposureThresholds = ExposureThresholds()
) -> dict[str, float | None]:
    """
    Analyzes image histogram to determine mean exposure and clipping ratios.
    """
    if not CV2_AVAILABLE or image is None:
        return {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    total_pixels = float(gray.size)

    if total_pixels == 0:
        return {}

    # 1. Calculate mean exposure level (0.0 to 1.0)
    mean_lum = float(np.mean(gray))
    exposure_level = mean_lum / 255.0

    # 2. Calculate Clipping Ratios
    # White clip: pixels >= 250 (avoids sensor noise at exactly 255)
    white_clip = float(np.count_nonzero(gray >= 250)) / total_pixels
    # Black clip: pixels <= 5
    black_clip = float(np.count_nonzero(gray <= 5)) / total_pixels

    # 3. Determine Outlier Status
    is_overexposed = (white_clip > thresholds.max_white_clip) or (exposure_level > thresholds.extreme_bright_limit)
    is_underexposed = (black_clip > thresholds.max_black_clip) or (exposure_level < thresholds.extreme_dark_limit)

    return {
        "exposure_level": exposure_level,
        "white_clip_ratio": white_clip,
        "black_clip_ratio": black_clip,
        "is_overexposed": is_overexposed,
        "is_underexposed": is_underexposed
    }


# --- Main Logic ---

def evaluate_exposure(
    exposure_level: float | None = None,
    is_overexposed: bool | None = None,
    is_underexposed: bool | None = None,
    white_clip_ratio: float | None = None,
    black_clip_ratio: float | None = None,
    thresholds: ExposureThresholds = ExposureThresholds(),
) -> ExposureScores:
    """
    Evaluate exposure level of a frame from pre-computed signals.
    Uses clipping ratios to accurately flag outliers without penalizing high-key/low-key art.
    """
    exp_q = _sanitize_score(exposure_level)
    w_clip_q = _sanitize_score(white_clip_ratio)
    b_clip_q = _sanitize_score(black_clip_ratio)

    scores = ExposureScores(
        exposure_level=exp_q,
        is_overexposed=is_overexposed,
        is_underexposed=is_underexposed,
        white_clip_ratio=w_clip_q,
        black_clip_ratio=b_clip_q,
        thresholds_used=asdict(thresholds),
    )

    # If clipping data is provided, use it to infer/override outlier flags
    has_clip_data = w_clip_q is not None or b_clip_q is not None

    if has_clip_data:
        if w_clip_q is not None and w_clip_q > thresholds.max_white_clip:
            scores.is_overexposed = True
        if b_clip_q is not None and b_clip_q > thresholds.max_black_clip:
            scores.is_underexposed = True

    # Fallback to mean exposure limits if clipping data is missing
    if exp_q is not None:
        if scores.is_overexposed is None:
            scores.is_overexposed = exp_q > thresholds.extreme_bright_limit
        if scores.is_underexposed is None:
            scores.is_underexposed = exp_q < thresholds.extreme_dark_limit

    # Infer exposure level from booleans if missing
    if exp_q is None:
        if scores.is_overexposed:
            scores.exposure_level = 0.95
        elif scores.is_underexposed:
            scores.exposure_level = 0.05

    # Calculate dynamic confidence
    if exp_q is not None and has_clip_data:
        scores.confidence = 0.9  # Highest confidence when we have both mean and clipping
    elif has_clip_data:
        scores.confidence = 0.8  # Clipping is a very strong signal on its own
    elif exp_q is not None:
        scores.confidence = 0.6  # Mean alone is a weak signal for outliers
    else:
        scores.confidence = 0.4

    return scores


def evaluate_exposure_from_image(
    image: np.ndarray,
    thresholds: ExposureThresholds = ExposureThresholds(),
) -> ExposureScores:
    """
    Convenience wrapper: Estimates all exposure signals directly from a raw image array.
    """
    if image is None or not CV2_AVAILABLE:
        return evaluate_exposure(thresholds=thresholds)

    signals = estimate_exposure_signals(image, thresholds)

    return evaluate_exposure(
        exposure_level=signals.get("exposure_level"),
        is_overexposed=signals.get("is_overexposed"),
        is_underexposed=signals.get("is_underexposed"),
        white_clip_ratio=signals.get("white_clip_ratio"),
        black_clip_ratio=signals.get("black_clip_ratio"),
        thresholds=thresholds
    )


def get_exposure_level(scores: ExposureScores) -> float:
    """Returns the exposure level (0.0 = dark, 1.0 = bright)."""
    return scores.exposure_level if scores.exposure_level is not None else 0.5


def is_exposure_outlier(scores: ExposureScores) -> bool:
    """Returns True if the frame has problematic exposure."""
    return bool(scores.is_overexposed or scores.is_underexposed)


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Signals (Backward Compatible):")
    legacy_scores = evaluate_exposure(exposure_level=0.9)
    print(f"   Level: {legacy_scores.exposure_level}, Overexposed: {legacy_scores.is_overexposed}")
    print(f"   Confidence: {legacy_scores.confidence:.2f}\n")

    if CV2_AVAILABLE:
        print("2. Testing CV Pipeline - High-Key Image (Bright but NOT overexposed):")
        # 500x500 image, mostly light gray (200), no blown whites
        high_key_img = np.ones((500, 500, 3), dtype=np.uint8) * 200
        high_key_scores = evaluate_exposure_from_image(high_key_img)
        print(f"   Exposure Level: {high_key_scores.exposure_level:.4f} (High)")
        print(f"   White Clip: {high_key_scores.white_clip_ratio:.4f} (0.0)")
        print(f"   Is Outlier: {is_exposure_outlier(high_key_scores)} (Should be False)\n")

        print("3. Testing CV Pipeline - Blown Out Image (Overexposed):")
        # 500x500 image, entirely pure white (255)
        blown_img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        blown_scores = evaluate_exposure_from_image(blown_img)
        print(f"   Exposure Level: {blown_scores.exposure_level:.4f}")
        print(f"   White Clip: {blown_scores.white_clip_ratio:.4f} (1.0)")
        print(f"   Is Overexposed: {blown_scores.is_overexposed} (Should be True)\n")

        print("4. Testing CV Pipeline - Crushed Blacks (Underexposed):")
        # 500x500 image, entirely pure black (0)
        dark_img = np.zeros((500, 500, 3), dtype=np.uint8)
        dark_scores = evaluate_exposure_from_image(dark_img)
        print(f"   Exposure Level: {dark_scores.exposure_level:.4f}")
        print(f"   Black Clip: {dark_scores.black_clip_ratio:.4f} (1.0)")
        print(f"   Is Underexposed: {dark_scores.is_underexposed} (Should be True)")
        print(f"   JSON: {dark_scores.to_json()}\n")
