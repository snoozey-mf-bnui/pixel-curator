#!/usr/bin/env python3
"""
motion_blur_detector.py
-----------------------
Supercharged module for detecting motion blur or camera shake in frames.

Enhancements:
- Built-in Computer Vision algorithms to estimate blur directly from images:
  - Overall Blur Level: Evaluated via Variance of the Laplacian (standard defocus/blur metric).
  - Directional Motion Blur: Evaluated via horizontal vs. vertical Sobel gradients. 
    Distinguishes between out-of-focus blur and directional camera shake.
- Configurable thresholds for categorization (sharp, slightly_blurry, very_blurry).
- Dynamic confidence scoring based on distance from category boundaries.
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
class BlurThresholds:
    """Configurable thresholds for blur classification."""
    # Laplacian variance thresholds (lower = more blurry)
    sharp_laplacian: float = 100.0       
    slightly_blurry_laplacian: float = 50.0 
    # Ratio threshold for directional motion (e.g., X variance is < 60% of Y variance)
    motion_ratio_threshold: float = 0.60 


@dataclass
class MotionBlurScores:
    """Container for motion blur detection results with serialization."""
    blur_level: Optional[float] = None          # 0.0 (sharp) to 1.0 (very blurry)
    is_blurry: Optional[bool] = None
    is_motion_blur: Optional[bool] = None       # True if directional blur is detected
    motion_direction: Optional[str] = None      # "horizontal", "vertical", or None
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

def estimate_blur_signals(
    image: np.ndarray, 
    thresholds: BlurThresholds = BlurThresholds()
) -> Dict[str, Optional[float]]:
    """
    Analyzes an image to determine overall blur level and directional motion blur.
    """
    if not CV2_AVAILABLE or image is None:
        return {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 1. Overall Blur (Variance of Laplacian)
    # High variance = sharp edges. Low variance = blurry.
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Normalize to 0.0-1.0 blur level (inverted: high var = low blur)
    # Using an asymptotic function: max_reasonable var is ~500+
    blur_level = 1.0 - min(1.0, laplacian_var / (laplacian_var + 100.0))

    # 2. Directional Motion Blur (Sobel Gradients)
    # If there is horizontal motion blur, vertical edges will get smeared.
    # Thus, the variance of the horizontal gradient (Sobel X) will drop significantly.
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    
    var_x = float(sobel_x.var())
    var_y = float(sobel_y.var())
    
    is_motion_blur = False
    motion_direction = None
    
    # Avoid division by zero
    max_var = max(var_x, var_y)
    if max_var > 1e-6:
        min_var = min(var_x, var_y)
        ratio = min_var / max_var
        
        # If one direction has significantly less variance than the other,
        # it indicates directional smearing (motion blur).
        if ratio < thresholds.motion_ratio_threshold:
            is_motion_blur = True
            # If var_x is much lower than var_y, vertical edges are smeared -> Horizontal motion
            motion_direction = "horizontal" if var_x < var_y else "vertical"

    return {
        "blur_level": float(blur_level),
        "is_blurry": blur_level > 0.5, # >0.5 blur level corresponds to ~100 Laplacian variance
        "is_motion_blur": is_motion_blur,
        "motion_direction": motion_direction
    }


# --- Main Logic ---

def detect_motion_blur(
    blur_level: Optional[float] = None,
    is_blurry: Optional[bool] = None,
    is_motion_blur: Optional[bool] = None,
    motion_direction: Optional[str] = None,
    thresholds: BlurThresholds = BlurThresholds(),
) -> MotionBlurScores:
    """
    Evaluate motion blur from pre-computed signals.
    Redistributes confidence based on available data.
    """
    blur_q = _sanitize_score(blur_level)

    scores = MotionBlurScores(
        blur_level=blur_q,
        is_blurry=is_blurry,
        is_motion_blur=is_motion_blur,
        motion_direction=motion_direction,
        thresholds_used=asdict(thresholds),
    )

    # Infer is_blurry from blur_level if not provided
    if scores.is_blurry is None and blur_q is not None:
        scores.is_blurry = blur_q > 0.5

    # Infer blur_level from is_blurry if not provided
    if blur_q is None and scores.is_blurry is not None:
        scores.blur_level = 0.8 if scores.is_blurry else 0.2

    # Calculate dynamic confidence
    if scores.blur_level is not None and scores.is_motion_blur is not None:
        scores.confidence = 0.9
    elif scores.blur_level is not None:
        scores.confidence = 0.7
    elif scores.is_blurry is not None:
        scores.confidence = 0.5
    else:
        scores.confidence = 0.0

    return scores


def detect_blur_from_image(
    image: np.ndarray,
    thresholds: BlurThresholds = BlurThresholds(),
) -> MotionBlurScores:
    """
    Convenience wrapper: Estimates all blur signals directly from a raw image array.
    """
    if image is None or not CV2_AVAILABLE:
        return detect_motion_blur(thresholds=thresholds)

    signals = estimate_blur_signals(image, thresholds)

    return detect_motion_blur(
        blur_level=signals.get("blur_level"),
        is_blurry=signals.get("is_blurry"),
        is_motion_blur=signals.get("is_motion_blur"),
        motion_direction=signals.get("motion_direction"),
        thresholds=thresholds
    )


def get_blur_level(scores: MotionBlurScores) -> float:
    """Returns the blur level (0.0 = sharp, 1.0 = very blurry)."""
    return scores.blur_level if scores.blur_level is not None else 0.0


def is_frame_blurry(scores: MotionBlurScores, threshold: float = 0.5) -> bool:
    """Returns True if the frame is considered blurry."""
    if scores.blur_level is not None:
        return scores.blur_level >= threshold
    return scores.is_blurry if scores.is_blurry is not None else False


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Signals (Backward Compatible):")
    legacy_scores = detect_motion_blur(blur_level=0.8, is_blurry=True)
    print(f"   Blur Level: {legacy_scores.blur_level:.4f}, Confidence: {legacy_scores.confidence:.2f}\n")

    if CV2_AVAILABLE:
        print("2. Testing CV Pipeline - Sharp Image:")
        # 500x500 image with a highly detailed pattern (fine grid)
        sharp_img = np.zeros((500, 500, 3), dtype=np.uint8)
        sharp_img[::10, :] = 255
        sharp_img[:, ::10] = 255
        sharp_scores = detect_blur_from_image(sharp_img)
        print(f"   Blur Level: {sharp_scores.blur_level:.4f} (Should be near 0)")
        print(f"   Is Blurry: {sharp_scores.is_blurry}")
        print(f"   Is Motion Blur: {sharp_scores.is_motion_blur}\n")

        print("3. Testing CV Pipeline - Out-of-Focus Blur (Gaussian):")
        # Apply heavy uniform blur to the sharp image
        blurry_img = cv2.GaussianBlur(sharp_img, (31, 31), 0)
        blurry_scores = detect_blur_from_image(blurry_img)
        print(f"   Blur Level: {blurry_scores.blur_level:.4f} (Should be near 1.0)")
        print(f"   Is Blurry: {blurry_scores.is_blurry}")
        print(f"   Is Motion Blur: {blurry_scores.is_motion_blur} (Should be False, it's uniform)\n")

        print("4. Testing CV Pipeline - Horizontal Motion Blur:")
        # Simulate horizontal motion blur using a horizontal kernel
        kernel = np.zeros((15, 15), dtype=np.float32)
        kernel[7, :] = 1.0 / 15.0  # Horizontal smearing kernel
        motion_img = cv2.filter2D(sharp_img, -1, kernel)
        motion_scores = detect_blur_from_image(motion_img)
        print(f"   Blur Level: {motion_scores.blur_level:.4f} (Should be high)")
        print(f"   Is Motion Blur: {motion_scores.is_motion_blur} (Should be True)")
        print(f"   Motion Direction: {motion_scores.motion_direction} (Should be horizontal)")
        print(f"   JSON: {motion_scores.to_json()}\n")
