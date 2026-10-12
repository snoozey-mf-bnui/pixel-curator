#!/usr/bin/env python3
"""
fabric_analysis.py
------------------
Supercharged module for classifying frames based on clothing coverage level.

Enhancements:
- Built-in Computer Vision skin detection: Automatically calculates skin vs. clothing 
  coverage ratio directly from an image and body bounding box using YCrCb color space.
- Configurable thresholds for categorization (full, partial, minimal).
- Dynamic confidence scoring based on distance from category boundaries.
- Robust input sanitization (handles NaN, Infinity, and out-of-bounds values).
- Sheer fabric heuristic: Flags potential sheer clothing if high edge density is found 
  over detected skin regions.
- JSON serialization for pipeline integration.
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
class CoverageThresholds:
    """Configurable thresholds for coverage classification."""
    full_threshold: float = 0.85       # >85% covered = full
    partial_threshold: float = 0.50    # >50% covered = partial
    minimal_threshold: float = 0.15    # >15% covered = minimal


@dataclass
class FabricAnalysisScores:
    """Container for fabric / coverage classification results with serialization."""
    coverage_level: str = "unknown"
    coverage_ratio: Optional[float] = None
    has_clothing: Optional[bool] = None
    has_sheer: Optional[bool] = None
    confidence: float = 0.0
    thresholds_used: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Converts scores to a dictionary, rounding for cleaner output."""
        data = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.items()}

    def to_json(self) -> str:
        """Serializes scores to a JSON string."""
        return json.dumps(self.to_dict())


def _sanitize_ratio(value: Optional[float]) -> Optional[float]:
    """Validates ratio scores. Clamps to [0.0, 1.0]. Returns None if invalid."""
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return max(0.0, min(1.0, float(value)))


def _calculate_dynamic_confidence(value: float, threshold: float) -> float:
    """
    Calculates confidence dynamically based on how far the value exceeds the threshold.
    E.g., if threshold is 0.85, exactly 0.85 = 0.5 confidence. 1.0 = 1.0 confidence.
    """
    if value <= threshold:
        return 0.0
    conf = (value - threshold) / (1.0 - threshold)
    return min(1.0, 0.5 + (0.5 * conf))


# --- CV Estimation Helpers ---

def compute_coverage_ratio(
    image: np.ndarray, 
    body_bbox: Optional[BBox]
) -> Optional[float]:
    """
    Computes the clothing coverage ratio by detecting skin pixels in the YCrCb color space.
    Ratio = 1.0 - (Skin Area / Body Bbox Area).
    """
    if not CV2_AVAILABLE or image is None or body_bbox is None:
        return None

    x1, y1, x2, y2 = body_bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    
    # Convert to YCrCb color space (much better for skin detection than RGB/HSV)
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCR_CB)
    
    # Standard YCrCb skin bounds
    lower_skin = np.array([0, 133, 77])
    upper_skin = np.array([255, 173, 127])
    
    # Create skin mask
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Clean up mask (remove small noise)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    skin_pixels = float(np.count_nonzero(mask))
    total_pixels = float(crop.shape[0] * crop.shape[1])
    
    if total_pixels == 0:
        return None
        
    skin_ratio = skin_pixels / total_pixels
    coverage_ratio = 1.0 - skin_ratio
    
    return _sanitize_ratio(coverage_ratio)


def detect_sheer_heuristic(
    image: np.ndarray, 
    body_bbox: Optional[BBox], 
    coverage_ratio: Optional[float]
) -> Optional[bool]:
    """
    Heuristic to detect sheer clothing.
    If the region is classified as mostly covered (>50%) but has high edge density 
    over the skin regions, it might be sheer.
    """
    if not CV2_AVAILABLE or image is None or body_bbox is None or coverage_ratio is None:
        return None
        
    if coverage_ratio < 0.5:
        return False  # Not enough clothing to be considered sheer
        
    x1, y1, x2, y2 = body_bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # High frequency edges (fabric texture over skin)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    
    # If coverage is high but edge density is very high, flag as sheer
    # (Normal skin has lower edge density, solid fabric has medium, sheer fabric over skin is high)
    if edge_density > 0.15 and coverage_ratio < 0.85:
        return True
        
    return False


# --- Main Logic ---

def classify_coverage_level(
    coverage_ratio: Optional[float] = None,
    has_clothing: Optional[bool] = None,
    has_sheer: Optional[bool] = None,
    thresholds: CoverageThresholds = CoverageThresholds(),
) -> FabricAnalysisScores:
    """
    Classify the clothing coverage level of a frame dynamically.
    """
    # Sanitize inputs
    cov_ratio = _sanitize_ratio(coverage_ratio)
    thresh_dict = asdict(thresholds)

    scores = FabricAnalysisScores(
        coverage_ratio=cov_ratio,
        has_clothing=has_clothing,
        has_sheer=has_sheer,
        thresholds_used=thresh_dict,
    )

    if cov_ratio is None and has_clothing is None:
        scores.coverage_level = "unknown"
        scores.confidence = 0.0
        return scores

    # 1. Check Sheer (Overrides ratio categories)
    if has_sheer:
        scores.coverage_level = "sheer"
        scores.confidence = 0.80  # Static confidence for explicit sheer flag
        return scores

    # 2. Classify by Ratio
    if cov_ratio is not None:
        if cov_ratio >= thresholds.full_threshold:
            scores.coverage_level = "full"
            scores.confidence = _calculate_dynamic_confidence(cov_ratio, thresholds.full_threshold)
        elif cov_ratio >= thresholds.partial_threshold:
            scores.coverage_level = "partial"
            scores.confidence = _calculate_dynamic_confidence(cov_ratio, thresholds.partial_threshold)
        elif cov_ratio >= thresholds.minimal_threshold:
            scores.coverage_level = "minimal"
            scores.confidence = _calculate_dynamic_confidence(cov_ratio, thresholds.minimal_threshold)
        else:
            scores.coverage_level = "minimal"
            # For <15% covered, confidence that it is "minimal" is very high
            scores.confidence = 1.0 - _calculate_dynamic_confidence(cov_ratio, thresholds.minimal_threshold)
            
        # Infer has_clothing if not explicitly provided
        if has_clothing is None:
            scores.has_clothing = cov_ratio > 0.05
    else:
        # 3. Fallback if ratio is missing but has_clothing is provided
        if has_clothing is False:
            scores.coverage_level = "minimal"
            scores.confidence = 0.5
        elif has_clothing is True:
            scores.coverage_level = "partial"  # Assume partial if we only know clothing exists
            scores.confidence = 0.4

    return scores


def classify_from_image(
    image: np.ndarray,
    body_bbox: Optional[BBox] = None,
    thresholds: CoverageThresholds = CoverageThresholds(),
) -> FabricAnalysisScores:
    """
    Convenience wrapper: Auto-calculates coverage ratio and sheer heuristic 
    directly from an image and body bounding box.
    """
    if image is None or not CV2_AVAILABLE or body_bbox is None:
        return classify_coverage_level(thresholds=thresholds)

    cov_ratio = compute_coverage_ratio(image, body_bbox)
    has_sheer = detect_sheer_heuristic(image, body_bbox, cov_ratio)

    return classify_coverage_level(
        coverage_ratio=cov_ratio,
        has_clothing=True if cov_ratio is not None and cov_ratio > 0.05 else None,
        has_sheer=has_sheer,
        thresholds=thresholds
    )


def get_coverage_level(scores: FabricAnalysisScores) -> str:
    """Returns the coverage level category."""
    return scores.coverage_level


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Ratios (Dynamic Confidence):")
    # 95% covered (Very solid 'full')
    scores_1 = classify_coverage_level(coverage_ratio=0.95)
    print(f"   Level: {scores_1.coverage_level}, Confidence: {scores_1.confidence:.2f}")
    
    # 86% covered (Barely 'full', lower confidence)
    scores_2 = classify_coverage_level(coverage_ratio=0.86)
    print(f"   Level: {scores_2.coverage_level}, Confidence: {scores_2.confidence:.2f}\n")

    print("2. Testing Sheer Override:")
    sheer_scores = classify_coverage_level(coverage_ratio=0.80, has_sheer=True)
    print(f"   Level: {sheer_scores.coverage_level}, Confidence: {sheer_scores.confidence:.2f}\n")

    print("3. Testing Custom Thresholds:")
    custom_thresh = CoverageThresholds(full_threshold=0.70)
    custom_scores = classify_coverage_level(coverage_ratio=0.75, thresholds=custom_thresh)
    print(f"   Level: {custom_scores.coverage_level} (Normally partial, now full)")
    print(f"   JSON: {custom_scores.to_json()}\n")

    if CV2_AVAILABLE:
        print("4. Testing CV Pipeline (Auto-Skin Detection):")
        # Simulate a 400x600 body crop.
        # Top half (0-300) is clothing (solid blue). Bottom half (300-600) is skin (YCrCb bounds).
        mock_img = np.zeros((600, 400, 3), dtype=np.uint8)
        
        # Solid clothing (Blue)
        mock_img[:300, :] = (200, 0, 0) 
        
        # Skin tone (Y=144, Cr=173, Cb=127 -> BGR approx (127, 173, 144)? OpenCV BGR for YCrCb 144,173,127)
        # Let's just use a known skin BGR: (130, 160, 190) 
        mock_img[300:, :] = (130, 160, 190) 
        
        # Body bbox covers the whole image
        body_bbox = (0, 0, 400, 600)
        
        cv_scores = classify_from_image(mock_img, body_bbox)
        print(f"   Coverage Ratio: {cv_scores.coverage_ratio:.4f} (Should be ~0.5)")
        print(f"   Level: {cv_scores.coverage_level} (Should be partial)")
        print(f"   Has Clothing: {cv_scores.has_clothing}")
        print(f"   JSON: {cv_scores.to_json()}\n")
