#!/usr/bin/env python3
"""
coverage_classifier.py
----------------------
Supercharged module for classifying frames according to framing intensity 
and composition level.

Enhancements:
- Configurable thresholds via `CoverageThresholds` dataclass.
- Dynamic confidence scoring based on distance from category boundaries.
- Built-in bounding box geometry helpers to auto-calculate ratios and infer full body.
- Robust input sanitization (handles NaN, out-of-bounds, and impossible ratios).
- JSON serialization for pipeline integration.
- Handles edge cases (e.g., face larger than body).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

# Bbox format: Tuple[int, int, int, int] -> (x_min, y_min, x_max, y_max)
BBox = tuple[int, int, int, int]


@dataclass
class CoverageThresholds:
    """Configurable thresholds for coverage classification."""
    close_portrait_face: float = 0.15   # Face takes up >15% of image
    upper_body_face: float = 0.05       # Face takes up >5% of image
    full_figure_body: float = 0.45      # Body takes up >45% of image
    upper_body_fallback: float = 0.15   # Body takes up >15% (no face detected)
    full_body_vertical_span: float = 0.85 # Body bbox spans >85% of image height


@dataclass
class CoverageClassification:
    """Container for framing / coverage classification results."""
    category: str = "unknown"
    confidence: float = 0.0
    face_ratio: float | None = None
    body_ratio: float | None = None
    has_full_body: bool | None = None
    thresholds_used: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Converts classification to a dictionary, rounding for cleaner output."""
        data = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.items()}

    def to_json(self) -> str:
        """Serializes classification to a JSON string."""
        return json.dumps(self.to_dict())


def _sanitize_ratio(value: float | None) -> float | None:
    """Validates ratio scores. Clamps to [0.0, 1.0]. Returns None if invalid."""
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return max(0.0, min(1.0, float(value)))


def _calculate_dynamic_confidence(
    value: float,
    threshold: float,
    max_scale: float = 1.0
) -> float:
    """
    Calculates confidence dynamically based on how far the value exceeds the threshold.
    E.g., if threshold is 0.15, exactly 0.15 = 0.5 confidence. 0.30 = 1.0 confidence.
    """
    if value <= threshold:
        return 0.0
    # Scale confidence: reaches 1.0 at 2x the threshold value
    conf = (value - threshold) / threshold
    return min(1.0, 0.5 + (0.5 * conf)) * max_scale


def compute_area_ratio(bbox: BBox | None, img_width: int, img_height: int) -> float | None:
    """Computes the area ratio of a bounding box relative to the image."""
    if bbox is None or img_width <= 0 or img_height <= 0:
        return None

    x1, y1, x2, y2 = bbox
    bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
    img_area = img_width * img_height

    if img_area == 0:
        return None

    return _sanitize_ratio(bbox_area / img_area)


def infer_has_full_body(
    body_bbox: BBox | None,
    img_height: int,
    vertical_span_threshold: float = 0.85
) -> bool:
    """
    Infers if the full body is visible by checking if the body bbox spans
    almost the entire vertical height of the image.
    """
    if body_bbox is None or img_height <= 0:
        return False

    x1, y1, x2, y2 = body_bbox
    bbox_height = y2 - y1

    if bbox_height <= 0:
        return False

    vertical_span = bbox_height / img_height
    return vertical_span >= vertical_span_threshold


def classify_coverage(
    face_area_ratio: float | None = None,
    body_area_ratio: float | None = None,
    has_full_body: bool = False,
    is_dynamic_angle: bool = False,
    thresholds: CoverageThresholds = CoverageThresholds(),
) -> CoverageClassification:
    """
    Classify the framing level of a frame dynamically based on configurable thresholds.
    """
    # Sanitize inputs
    face_ratio = _sanitize_ratio(face_area_ratio)
    body_ratio = _sanitize_ratio(body_area_ratio)

    thresh_dict = asdict(thresholds)

    if face_ratio is None and body_ratio is None:
        return CoverageClassification(
            category="unknown",
            confidence=0.0,
            thresholds_used=thresh_dict
        )

    # Impossible geometry check: Face cannot be larger than body
    if face_ratio is not None and body_ratio is not None:
        if face_ratio > body_ratio and body_ratio > 0:
            logger.debug("Face ratio > body ratio. Adjusting body ratio to match face.")
            body_ratio = face_ratio

    # 1. Check for Full Body / Dynamic Wide first (structural override)
    if has_full_body or (body_ratio is not None and body_ratio > thresholds.full_figure_body):
        category = "dynamic_wide" if is_dynamic_angle else "full_figure"

        # Calculate confidence based on body ratio
        conf = 0.8  # Base confidence if inferred via has_full_body flag
        if body_ratio is not None and body_ratio > thresholds.full_figure_body:
            conf = _calculate_dynamic_confidence(body_ratio, thresholds.full_figure_body)

        return CoverageClassification(
            category=category,
            confidence=conf,
            face_ratio=face_ratio,
            body_ratio=body_ratio,
            has_full_body=True,
            thresholds_used=thresh_dict
        )

    # 2. Check for Close Portrait
    if face_ratio is not None and face_ratio > thresholds.close_portrait_face:
        conf = _calculate_dynamic_confidence(face_ratio, thresholds.close_portrait_face)
        return CoverageClassification(
            category="close_portrait",
            confidence=conf,
            face_ratio=face_ratio,
            body_ratio=body_ratio,
            has_full_body=False,
            thresholds_used=thresh_dict
        )

    # 3. Check for Upper Body (via face size)
    if face_ratio is not None and face_ratio > thresholds.upper_body_face:
        conf = _calculate_dynamic_confidence(face_ratio, thresholds.upper_body_face)
        return CoverageClassification(
            category="upper_body",
            confidence=conf,
            face_ratio=face_ratio,
            body_ratio=body_ratio,
            has_full_body=False,
            thresholds_used=thresh_dict
        )

    # 4. Fallback to Upper Body (via body size if face detection failed)
    if body_ratio is not None and body_ratio > thresholds.upper_body_fallback:
        conf = _calculate_dynamic_confidence(body_ratio, thresholds.upper_body_fallback)
        return CoverageClassification(
            category="upper_body",
            confidence=conf,
            face_ratio=face_ratio,
            body_ratio=body_ratio,
            has_full_body=False,
            thresholds_used=thresh_dict
        )

    # 5. Unknown Fallback
    return CoverageClassification(
        category="unknown",
        confidence=0.3,
        face_ratio=face_ratio,
        body_ratio=body_ratio,
        has_full_body=False,
        thresholds_used=thresh_dict
    )


def classify_from_bboxes(
    face_bbox: BBox | None = None,
    body_bbox: BBox | None = None,
    img_width: int = 0,
    img_height: int = 0,
    is_dynamic_angle: bool = False,
    thresholds: CoverageThresholds = CoverageThresholds(),
) -> CoverageClassification:
    """
    Convenience wrapper: Auto-calculates ratios and infers full body status 
    directly from bounding boxes and image dimensions.
    """
    if img_width <= 0 or img_height <= 0:
        logger.error("Image width and height must be provided and > 0.")
        return CoverageClassification(category="unknown", confidence=0.0)

    face_ratio = compute_area_ratio(face_bbox, img_width, img_height)
    body_ratio = compute_area_ratio(body_bbox, img_width, img_height)

    has_full_body = infer_has_full_body(
        body_bbox,
        img_height,
        thresholds.full_body_vertical_span
    )

    return classify_coverage(
        face_area_ratio=face_ratio,
        body_area_ratio=body_ratio,
        has_full_body=has_full_body,
        is_dynamic_angle=is_dynamic_angle,
        thresholds=thresholds
    )


def get_framing_category(classification: CoverageClassification) -> str:
    """Returns the framing category as a string."""
    return classification.category


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Direct Ratio Input (Dynamic Confidence):")
    # Face takes up 20% of screen (Very solid close portrait)
    scores_1 = classify_coverage(face_area_ratio=0.20, body_area_ratio=0.35)
    print(f"   Category: {scores_1.category}, Confidence: {scores_1.confidence:.2f}")

    # Face takes up 16% of screen (Barely a close portrait, lower confidence)
    scores_2 = classify_coverage(face_area_ratio=0.16, body_area_ratio=0.30)
    print(f"   Category: {scores_2.category}, Confidence: {scores_2.confidence:.2f}\n")

    print("2. Testing Bounding Box Geometry (Auto-Calculation):")
    # Simulated 1920x1080 image
    # Face bbox: 800, 200, 1100, 500 (300x300 = 90,000px)
    # Body bbox: 700, 150, 1200, 1000 (500x850 = 425,000px)
    # Image Area: 2,073,600px
    # Face Ratio ~= 0.043, Body Ratio ~= 0.205.
    # Vertical span = 850/1080 = 0.78 (Not full body).
    bbox_scores = classify_from_bboxes(
        face_bbox=(800, 200, 1100, 500),
        body_bbox=(700, 150, 1200, 1000),
        img_width=1920,
        img_height=1080
    )
    print(f"   Category: {bbox_scores.category}") # Should be upper_body (fallback via body_ratio > 0.15)
    print(f"   JSON: {bbox_scores.to_json()}\n")

    print("3. Testing Full Body Inference from Bounding Box:")
    # Body bbox spans almost entire height: y1=50, y2=1050 (1000px tall out of 1080)
    full_body_scores = classify_from_bboxes(
        face_bbox=(900, 50, 1000, 150),
        body_bbox=(600, 50, 1300, 1050),
        img_width=1920,
        img_height=1080,
        is_dynamic_angle=False
    )
    print(f"   Category: {full_body_scores.category}") # Should be full_figure
    print(f"   Has Full Body: {full_body_scores.has_full_body}\n")

    print("4. Testing Custom Thresholds:")
    custom_thresh = CoverageThresholds(close_portrait_face=0.05)
    custom_scores = classify_coverage(face_area_ratio=0.06, thresholds=custom_thresh)
    print(f"   Category: {custom_scores.category} (Normally upper_body, now close_portrait)")
    print(f"   JSON: {custom_scores.to_json()}\n")
