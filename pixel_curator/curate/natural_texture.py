#!/usr/bin/env python3
"""
natural_texture.py
------------------
Supercharged module for evaluating natural surface detail and texture retention.

Enhancements:
- Vectorized local standard deviation calculation using Integral Images / Box Filters
  (O(1) complexity, removes slow Python for-loops).
- Asymptotic normalization for unbounded metrics (Laplacian variance).
- Auto-Canny edge detection: dynamically calculates thresholds instead of hardcoded 50/150.
- Accepts both BGR/RGB and Grayscale images automatically.
- Robust input sanitization and JSON serialization.
- Configurable weights for the final combined detail score.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# --- Default Configuration ---
DEFAULT_WEIGHTS: Dict[str, float] = {
    "high_freq_energy": 0.40,
    "texture_consistency": 0.40,
    "edge_density": 0.20,
}

DEFAULT_MAX_LAPLACIAN_VAR = 1500.0  # Tunable: represents "extremely high detail"


@dataclass
class NaturalTextureScores:
    """Container for natural texture metrics with serialization capabilities."""
    high_freq_energy: Optional[float] = None
    texture_consistency: Optional[float] = None
    edge_density: Optional[float] = None
    natural_detail_score: Optional[float] = None
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


def _to_grayscale(image: np.ndarray) -> Optional[np.ndarray]:
    """Converts any image array (BGR/RGB/Gray) to grayscale."""
    if image is None or image.size == 0:
        return None
    
    # If it's already 2D, it's grayscale
    if len(image.shape) == 2:
        return image
        
    # If it's 3D, convert (assuming BGR or RGB, same result for grayscale conversion)
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            
    logger.warning(f"Unexpected image shape: {image.shape}")
    return None


def normalize_high_freq(value: float, max_reasonable: float = DEFAULT_MAX_LAPLACIAN_VAR) -> float:
    """
    Normalize Laplacian variance into [0, 1] range using an asymptotic function.
    """
    if value <= 0:
        return 0.0
    return min(1.0, value / (value + max_reasonable))


def compute_high_frequency_energy(gray: np.ndarray) -> float:
    """
    Estimate high-frequency content using Laplacian variance.
    Returns RAW variance (use normalize_high_freq to scale it 0-1).
    """
    if gray is None or gray.size == 0:
        return 0.0
        
    # Ensure uint8 for standard Laplacian
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
        
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))


def compute_texture_consistency(gray: np.ndarray, window_size: int = 32) -> float:
    """
    Measures how consistent local texture is across the image using Vectorized Box Filters.
    Lower variance in local standard deviation = more consistent texture.
    Returns score in [0, 1].
    """
    if gray is None or gray.size == 0:
        return 0.0

    # Convert to float32 for precise local variance calculation
    gray_f = gray.astype(np.float32)
    
    # Use OpenCV's boxFilter for O(1) performance instead of nested for-loops
    ksize = (window_size, window_size)
    mean = cv2.boxFilter(gray_f, cv2.CV_32F, ksize, normalize=True)
    sq_mean = cv2.boxFilter(gray_f * gray_f, cv2.CV_32F, ksize, normalize=True)
    
    # Variance = E[X^2] - (E[X])^2
    variance = sq_mean - (mean * mean)
    local_std = np.sqrt(np.maximum(variance, 0.0))
    
    # Global consistency metric
    mean_local_std = float(np.mean(local_std))
    std_local_std = float(np.std(local_std))
    
    if mean_local_std < 1e-6:
        return 0.0  # Completely flat image
        
    # Consistency = 1.0 when std is 0, drops as std increases relative to mean
    consistency = 1.0 - min(1.0, std_local_std / (mean_local_std + 1e-6))
    return consistency


def compute_edge_density(gray: np.ndarray) -> float:
    """
    Measures edge density using Auto-Canny edge detection.
    Dynamically calculates thresholds based on image median.
    Returns score in [0, 1].
    """
    if gray is None or gray.size == 0:
        return 0.0

    # Auto-Canny implementation
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lower, upper)
    
    return float(np.count_nonzero(edges) / edges.size)


def compute_natural_texture(
    image: np.ndarray,
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
    window_size: int = 32
) -> NaturalTextureScores:
    """
    Main function. Analyzes an image (BGR or Grayscale) and returns texture-related scores.
    """
    gray = _to_grayscale(image)
    if gray is None:
        return NaturalTextureScores(weights_used=weights)

    # Compute raw metrics
    raw_high_freq = compute_high_frequency_energy(gray)
    texture_cons = compute_texture_consistency(gray, window_size)
    edge_dens = compute_edge_density(gray)

    # Normalize all metrics to [0, 1]
    norm_high_freq = normalize_high_freq(raw_high_freq)
    norm_texture_cons = _sanitize_score(texture_cons)
    norm_edge_dens = _sanitize_score(edge_dens)

    # Weighted average for final score
    components = []
    used_weights = []

    if norm_high_freq is not None:
        components.append(norm_high_freq)
        used_weights.append(weights.get("high_freq_energy", 0.0))

    if norm_texture_cons is not None:
        components.append(norm_texture_cons)
        used_weights.append(weights.get("texture_consistency", 0.0))

    if norm_edge_dens is not None:
        components.append(norm_edge_dens)
        used_weights.append(weights.get("edge_density", 0.0))

    if not components or sum(used_weights) == 0:
        final_score = 0.0
    else:
        weighted_sum = sum(c * w for c, w in zip(components, used_weights))
        total_weight = sum(used_weights)
        final_score = float(weighted_sum / total_weight)

    return NaturalTextureScores(
        high_freq_energy=round(norm_high_freq, 4),  # Store the normalized version for consistency
        texture_consistency=round(norm_texture_cons, 4) if norm_texture_cons is not None else None,
        edge_density=round(norm_edge_dens, 4) if norm_edge_dens is not None else None,
        natural_detail_score=round(final_score, 4),
        weights_used=weights
    )


def get_natural_detail_score(scores: NaturalTextureScores) -> float:
    """Returns the combined natural detail score."""
    return scores.natural_detail_score if scores.natural_detail_score is not None else 0.0


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Synthetic Flat Image (Low Texture):")
    # Create a 500x500 solid gray image
    flat_img = np.ones((500, 500, 3), dtype=np.uint8) * 128
    flat_scores = compute_natural_texture(flat_img)
    print(f"   Detail Score: {flat_scores.natural_detail_score} (Should be ~0.0)")
    print(f"   JSON: {flat_scores.to_json()}\n")

    print("2. Testing Synthetic Noisy Image (High Freq, Low Consistency):")
    # Create a 500x500 random noise image
    np.random.seed(42)
    noise_img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    noise_scores = compute_natural_texture(noise_img)
    print(f"   Detail Score: {noise_scores.natural_detail_score}")
    print(f"   High Freq: {noise_scores.high_freq_energy}, Consistency: {noise_scores.texture_consistency}\n")

    print("3. Testing Synthetic Pattern Image (High Freq, High Consistency):")
    # Create a checkerboard pattern
    pattern = np.zeros((500, 500), dtype=np.uint8)
    pattern[::20, :] = 255
    pattern[:, ::20] = 255
    pattern_scores = compute_natural_texture(pattern)
    print(f"   Detail Score: {pattern_scores.natural_detail_score}")
    print(f"   High Freq: {pattern_scores.high_freq_energy}, Consistency: {pattern_scores.texture_consistency}\n")
