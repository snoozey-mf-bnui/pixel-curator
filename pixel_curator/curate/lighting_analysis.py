#!/usr/bin/env python3
"""
skin_rendering.py
-----------------
Supercharged module for evaluating skin rendering quality and texture fidelity.

Enhancements:
- Built-in Computer Vision algorithms to estimate skin quality directly from image patches:
  - Skin Texture Detail: Evaluated via Laplacian variance (rewards fine pores/micro-texture).
  - Skin Smoothness: Evaluated via local standard deviation (penalizes blemishes/noise but 
    heavily penalizes "plastic" zero-variance skin).
  - Skin Naturalness: Evaluated via histogram entropy (rewards organic texture distribution, 
    penalizes over-smoothed AI skin or excessive noise).
- Robust input sanitization (handles NaN, Infinity, and out-of-bounds values).
- Weight tracking and redistribution (if a metric is missing, its weight is redistributed).
- JSON serialization for pipeline integration.
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

# --- Default Configuration ---
DEFAULT_WEIGHTS: Dict[str, float] = {
    "skin_texture_detail": 0.40,
    "skin_smoothness": 0.30,
    "skin_naturalness": 0.30,
}


@dataclass
class SkinRenderingScores:
    """Container for skin rendering metrics with serialization capabilities."""
    skin_texture_detail: Optional[float] = None
    skin_smoothness: Optional[float] = None
    skin_naturalness: Optional[float] = None
    overall_skin_score: Optional[float] = None
    weights_used: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Converts scores to a dictionary, rounding for cleaner output."""
        data = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.items()}

    def to_json(self) -> str:
        """Serializes scores to a JSON string."""
        return json.dumps(self.to_dict())


def _sanitize_score(value: Optional[float]) -> Optional[float]:
    """Validates and clamps scores to [0.0, 1.0]. Returns None if invalid."""
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return max(0.0, min(1.0, float(value)))


# --- CV Estimation Helpers ---

def estimate_skin_texture_detail(image_patch: np.ndarray) -> Optional[float]:
    """
    Estimates skin texture detail (pores, micro-wrinkles) via Laplacian variance.
    High variance indicates well-preserved fine detail.
    """
    if not CV2_AVAILABLE or image_patch is None or image_patch.size == 0:
        return None
        
    gray = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY) if len(image_patch.shape) == 3 else image_patch
    
    # Variance of Laplacian
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Normalize: Skin patches usually have a variance between 0 (plastic) and 300 (very porous).
    # We use an asymptotic function reaching 1.0 around 150+ variance.
    return float(var / (var + 150.0))


def estimate_skin_smoothness(image_patch: np.ndarray, window_size: int = 16) -> Optional[float]:
    """
    Estimates skin smoothness using local standard deviation.
    Penalizes blemishes/noise (high local std) but ALSO penalizes "plastic" skin (zero local std).
    Uses a Gaussian curve to reward a natural, smooth-but-realistic micro-texture.
    """
    if not CV2_AVAILABLE or image_patch is None or image_patch.size == 0:
        return None
        
    gray = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY) if len(image_patch.shape) == 3 else image_patch
    gray_f = gray.astype(np.float32)
    
    # Vectorized local standard deviation
    mean = cv2.boxFilter(gray_f, cv2.CV_32F, (window_size, window_size), normalize=True)
    sq_mean = cv2.boxFilter(gray_f * gray_f, cv2.CV_32F, (window_size, window_size), normalize=True)
    local_std = np.sqrt(np.maximum(sq_mean - (mean * mean), 0.0))
    
    mean_local_std = float(np.mean(local_std))
    
    # Ideal smooth skin has a micro-texture local std around 5.0 to 10.0.
    # Blemished skin > 20.0. Plastic skin < 2.0.
    ideal_std = 7.0
    spread = 6.0
    
    # Gaussian curve: 1.0 at ideal_std, drops off if too smooth (plastic) or too rough (acne/noise)
    smoothness_score = math.exp(-((mean_local_std - ideal_std) ** 2) / (2 * spread ** 2))
    
    return float(smoothness_score)


def estimate_skin_naturalness(image_patch: np.ndarray) -> Optional[float]:
    """
    Estimates skin naturalness via histogram entropy.
    Over-smoothed (AI-generated) skin has ultra-low entropy. Noisy skin has ultra-high entropy.
    Natural skin falls in a healthy middle ground.
    """
    if not CV2_AVAILABLE or image_patch is None or image_patch.size == 0:
        return None
        
    gray = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY) if len(image_patch.shape) == 3 else image_patch
    
    # Calculate histogram entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # Max entropy for 256 bins is 8.0. 
    # Natural skin usually sits around 5.0 to 6.5.
    # Plastic skin sits < 4.0. Noisy skin sits > 7.0.
    ideal_entropy = 5.75
    spread = 1.5
    
    # Gaussian curve: 1.0 at ideal_entropy, drops off for plastic or noisy skin
    naturalness_score = math.exp(-((entropy - ideal_entropy) ** 2) / (2 * spread ** 2))
    
    return float(naturalness_score)


# --- Main Logic ---

def compute_overall_skin_rendering(
    skin_texture_detail: Optional[float] = None,
    skin_smoothness: Optional[float] = None,
    skin_naturalness: Optional[float] = None,
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
) -> SkinRenderingScores:
    """
    Main function to compute skin rendering quality from pre-computed signals.
    Redistributes weights if certain metrics are missing (None).
    """
    # Sanitize inputs
    tex_q = _sanitize_score(skin_texture_detail)
    smth_q = _sanitize_score(skin_smoothness)
    nat_q = _sanitize_score(skin_naturalness)

    scores = SkinRenderingScores(
        skin_texture_detail=tex_q,
        skin_smoothness=smth_q,
        skin_naturalness=nat_q,
        weights_used=weights
    )

    components = []
    used_weights = []

    if tex_q is not None:
        components.append(tex_q)
        used_weights.append(weights.get("skin_texture_detail", 0.0))
    if smth_q is not None:
        components.append(smth_q)
        used_weights.append(weights.get("skin_smoothness", 0.0))
    if nat_q is not None:
        components.append(nat_q)
        used_weights.append(weights.get("skin_naturalness", 0.0))

    if not components or sum(used_weights) == 0:
        scores.overall_skin_score = 0.0
    else:
        weighted_sum = sum(c * w for c, w in zip(components, used_weights))
        total_weight = sum(used_weights)
        scores.overall_skin_score = float(weighted_sum / total_weight)

    return scores


def compute_skin_rendering_from_image(
    image_patch: np.ndarray,
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
) -> SkinRenderingScores:
    """
    Convenience wrapper: Estimates all skin rendering signals directly from a raw image patch.
    """
    if image_patch is None or not CV2_AVAILABLE:
        return compute_overall_skin_rendering(weights=weights)

    texture = estimate_skin_texture_detail(image_patch)
    smoothness = estimate_skin_smoothness(image_patch)
    naturalness = estimate_skin_naturalness(image_patch)

    return compute_overall_skin_rendering(
        skin_texture_detail=texture,
        skin_smoothness=smoothness,
        skin_naturalness=naturalness,
        weights=weights
    )


def get_skin_rendering_score(scores: SkinRenderingScores) -> float:
    """Returns the overall skin rendering quality score."""
    return scores.overall_skin_score if scores.overall_skin_score is not None else 0.0


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Floats (Backward Compatible):")
    basic_scores = compute_overall_skin_rendering(
        skin_texture_detail=0.8,
        skin_smoothness=0.7,
        skin_naturalness=0.9
    )
    print(f"   Overall Skin Score: {basic_scores.overall_skin_score:.4f}")
    print(f"   JSON: {basic_scores.to_json()}\n")

    print("2. Testing Missing Data (Weight Redistribution):")
    partial_scores = compute_overall_skin_rendering(
        skin_texture_detail=0.8,
        skin_naturalness=0.9
    )
    print(f"   Overall Skin Score: {partial_scores.overall_skin_score:.4f}\n")

    if CV2_AVAILABLE:
        print("3. Testing CV Pipeline - Natural Skin (Simulated):")
        # 200x200 patch. Base tone 120, add organic noise for pores, smooth slightly.
        np.random.seed(42)
        natural_skin = np.ones((200, 200), dtype=np.uint8) * 120
        natural_skin += np.random.normal(0, 8, (200, 200)).astype(np.uint8)
        natural_skin = cv2.GaussianBlur(natural_skin, (3, 3), 0)
        nat_scores = compute_skin_rendering_from_image(natural_skin)
        print(f"   Texture: {nat_scores.skin_texture_detail:.4f}, Smoothness: {nat_scores.skin_smoothness:.4f}, Naturalness: {nat_scores.skin_naturalness:.4f}")
        print(f"   Overall: {nat_scores.overall_skin_score:.4f}\n")

        print("4. Testing CV Pipeline - Plastic Skin (Over-smoothed):")
        # 200x200 patch. Completely flat color.
        plastic_skin = np.ones((200, 200), dtype=np.uint8) * 120
        plastic_scores = compute_skin_rendering_from_image(plastic_skin)
        print(f"   Texture: {plastic_scores.skin_texture_detail:.4f} (Should be 0)")
        print(f"   Smoothness: {plastic_scores.skin_smoothness:.4f} (Should be low, penalized for plastic)")
        print(f"   Overall: {plastic_scores.overall_skin_score:.4f}\n")

        print("5. Testing CV Pipeline - Noisy Skin (Bad Sensor/ISO):")
        # 200x200 patch. Extreme noise.
        noisy_skin = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        noisy_scores = compute_skin_rendering_from_image(noisy_skin)
        print(f"   Texture: {noisy_scores.skin_texture_detail:.4f} (High, but fake)")
        print(f"   Smoothness: {noisy_scores.skin_smoothness:.4f} (Should be near 0)")
        print(f"   Overall: {noisy_scores.overall_skin_score:.4f}\n")
