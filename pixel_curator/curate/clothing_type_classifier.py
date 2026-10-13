#!/usr/bin/env python3
"""
clothing_type_classifier.py
---------------------------
Supercharged module for classifying the type of clothing/outfit in a frame.

Enhancements:
- Built-in Computer Vision skin detection: Automatically infers `coverage_level` 
  directly from an image and body bounding box using YCrCb color space.
- Heuristic Clothing Classification: Infers `clothing_type` based on coverage 
  (e.g., 0% skin = nude, <20% skin = swimwear/minimal, >80% skin = fully clothed).
- Configurable thresholds for categorization.
- Dynamic confidence scoring based on whether CV or pre-computed strings were used.
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

# Bbox format: Tuple[int, int, int, int] -> (x_min, y_min, x_max, y_max)
BBox = tuple[int, int, int, int]


@dataclass
class CoverageThresholds:
    """Configurable thresholds for clothing coverage classification."""
    nude_threshold: float = 0.05       # <5% clothing = nude
    minimal_threshold: float = 0.25    # <25% clothing = minimal (swimwear/lingerie)
    medium_threshold: float = 0.60     # <60% clothing = medium (casual short sleeves)
    # >60% clothing = high/full (formal, winter wear)


@dataclass
class ClothingTypeScores:
    """Container for clothing type classification results with serialization."""
    clothing_type: str = "unknown"           # lingerie, casual, formal, swimwear, nude, etc.
    coverage_level: str | None = None     # full, high, medium, low, none
    coverage_ratio: float | None = None   # 0.0 (nude) to 1.0 (fully covered)
    confidence: float = 0.0
    thresholds_used: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Converts scores to a dictionary, rounding for cleaner output."""
        data = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.items()}

    def to_json(self) -> str:
        """Serializes scores to a JSON string."""
        return json.dumps(self.to_dict())


def _sanitize_ratio(value: float | None) -> float | None:
    """Validates ratio scores. Clamps to [0.0, 1.0]. Returns None if invalid."""
    if value is None: return None
    if not isinstance(value, (int, float)): return None
    if math.isnan(value) or math.isinf(value): return None
    return max(0.0, min(1.0, float(value)))


# --- CV Estimation Helpers ---

def compute_coverage_ratio(
    image: np.ndarray,
    body_bbox: BBox | None
) -> float | None:
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


def infer_clothing_from_coverage(
    coverage_ratio: float,
    thresholds: CoverageThresholds = CoverageThresholds()
) -> tuple[str, str]:
    """
    Infers the coverage level and clothing type heuristically from the skin coverage ratio.
    """
    if coverage_ratio < thresholds.nude_threshold:
        return "none", "nude"
    elif coverage_ratio < thresholds.minimal_threshold:
        return "low", "swimwear_or_lingerie"
    elif coverage_ratio < thresholds.medium_threshold:
        return "medium", "casual_shorts_or_tshirt"
    else:
        return "high", "formal_or_full_casual"


# --- Main Logic ---

def classify_clothing_type(
    clothing_type: str | None = None,
    coverage_level: str | None = None,
    coverage_ratio: float | None = None,
    thresholds: CoverageThresholds = CoverageThresholds(),
) -> ClothingTypeScores:
    """
    Classify clothing type and coverage from pre-computed signals.
    If coverage_ratio is provided, it will be used to infer missing types/levels.
    """
    cov_ratio = _sanitize_ratio(coverage_ratio)

    scores = ClothingTypeScores(
        clothing_type=clothing_type or "unknown",
        coverage_level=coverage_level,
        coverage_ratio=cov_ratio,
        thresholds_used=asdict(thresholds),
    )

    # If we have a coverage ratio, use it to infer missing data
    if cov_ratio is not None:
        inferred_level, inferred_type = infer_clothing_from_coverage(cov_ratio, thresholds)

        if scores.coverage_level is None:
            scores.coverage_level = inferred_level

        # Only override clothing_type if it's completely unknown
        if scores.clothing_type == "unknown":
            scores.clothing_type = inferred_type

        scores.confidence = 0.8 if scores.clothing_type != "unknown" else 0.6
    else:
        # Fallback to explicit string inputs
        if scores.clothing_type != "unknown" and scores.coverage_level is not None:
            scores.confidence = 0.9
        elif scores.clothing_type != "unknown":
            scores.confidence = 0.7
        elif scores.coverage_level is not None:
            scores.confidence = 0.5
        else:
            scores.confidence = 0.2

    return scores


def classify_from_image(
    image: np.ndarray,
    body_bbox: BBox | None = None,
    thresholds: CoverageThresholds = CoverageThresholds(),
) -> ClothingTypeScores:
    """
    Convenience wrapper: Auto-calculates coverage ratio and infers clothing type 
    directly from an image and body bounding box.
    """
    if image is None or not CV2_AVAILABLE or body_bbox is None:
        return classify_clothing_type(thresholds=thresholds)

    cov_ratio = compute_coverage_ratio(image, body_bbox)

    return classify_clothing_type(
        coverage_ratio=cov_ratio,
        thresholds=thresholds
    )


def get_clothing_type(scores: ClothingTypeScores) -> str:
    """Returns the classified clothing type."""
    return scores.clothing_type


def get_coverage_level(scores: ClothingTypeScores) -> str | None:
    """Returns the coverage level."""
    return scores.coverage_level


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Strings (Backward Compatible):")
    legacy_scores = classify_clothing_type(clothing_type="formal", coverage_level="high")
    print(f"   Type: {legacy_scores.clothing_type}, Level: {legacy_scores.coverage_level}")
    print(f"   JSON: {legacy_scores.to_json()}\n")

    print("2. Testing Coverage Ratio Inference (Swimwear):")
    # 15% clothing coverage -> Should infer swimwear/lingerie
    ratio_scores = classify_clothing_type(coverage_ratio=0.15)
    print(f"   Type: {ratio_scores.clothing_type} (Should be swimwear_or_lingerie)")
    print(f"   Level: {ratio_scores.coverage_level} (Should be low)\n")

    if CV2_AVAILABLE:
        print("3. Testing CV Pipeline - Nude/Minimal Clothing:")
        # Simulate a 400x600 body crop. 100% skin tone.
        nude_img = np.zeros((600, 400, 3), dtype=np.uint8)
        nude_img[:] = (130, 160, 190) # Skin tone BGR

        nude_scores = classify_from_image(nude_img, body_bbox=(0, 0, 400, 600))
        print(f"   Coverage Ratio: {nude_scores.coverage_ratio:.4f} (Should be ~0.0)")
        print(f"   Type: {nude_scores.clothing_type} (Should be nude)")
        print(f"   Level: {nude_scores.coverage_level}\n")

        print("4. Testing CV Pipeline - Fully Clothed:")
        # Simulate a 400x600 body crop. 0% skin (solid blue clothing).
        clothed_img = np.zeros((600, 400, 3), dtype=np.uint8)
        clothed_img[:] = (200, 0, 0) # Blue

        clothed_scores = classify_from_image(clothed_img, body_bbox=(0, 0, 400, 600))
        print(f"   Coverage Ratio: {clothed_scores.coverage_ratio:.4f} (Should be ~1.0)")
        print(f"   Type: {clothed_scores.clothing_type} (Should be formal_or_full_casual)")
        print(f"   Level: {clothed_scores.coverage_level} (Should be high)\n")
