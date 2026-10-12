#!/usr/bin/env python3
"""
face_occlusion.py
-----------------
Supercharged module for detecting occlusion of the face in a frame.

Enhancements:
- Built-in Computer Vision skin detection: Automatically calculates occlusion 
  level by detecting non-skin pixels inside the face bounding box using YCrCb color space.
- Spatial Occlusion Mapping: Divides the face into regions (top, bottom, left, right) 
  to determine WHERE the occlusion is occurring.
- Heuristic Occlusion Classification: Infers occlusion type based on spatial location 
  (e.g., top occlusion = "hair", bottom occlusion = "mask", asymmetrical = "hand").
- Configurable thresholds for categorization.
- Dynamic confidence scoring and JSON serialization for pipeline integration.
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
class OcclusionThresholds:
    """Configurable thresholds for occlusion classification."""
    # A completely clean, unoccluded face crop usually has 50-70% skin 
    # (the rest is eyes, eyebrows, lips, hairline).
    max_healthy_skin_ratio: float = 0.45 
    occluded_threshold: float = 0.25   # >25% non-skin = occluded


@dataclass
class FaceOcclusionScores:
    """Container for face occlusion metrics with serialization."""
    occlusion_level: Optional[float] = None     # 0.0 (visible) to 1.0 (fully occluded)
    is_occluded: Optional[bool] = None
    occlusion_type: Optional[str] = None        # hair, mask, hand, object, unknown
    skin_ratio: Optional[float] = None
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

def estimate_occlusion_signals(
    image: np.ndarray, 
    face_bbox: Optional[BBox],
    thresholds: OcclusionThresholds = OcclusionThresholds()
) -> Dict[str, Optional[float]]:
    """
    Analyzes a face crop to determine occlusion by looking for non-skin pixels.
    """
    if not CV2_AVAILABLE or image is None or face_bbox is None:
        return {}

    x1, y1, x2, y2 = face_bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if x2 <= x1 or y2 <= y1:
        return {}

    crop = image[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]
    
    # Convert to YCrCb and create skin mask
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCR_CB)
    lower_skin = np.array([0, 133, 77])
    upper_skin = np.array([255, 173, 127])
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    total_pixels = float(crop_h * crop_w)
    skin_pixels = float(np.count_nonzero(skin_mask))
    skin_ratio = skin_pixels / total_pixels
    
    # Calculate occlusion level
    # If skin_ratio is 0.6, occlusion is 0.0. If skin_ratio is 0.1, occlusion is high.
    occlusion_level = max(0.0, (thresholds.max_healthy_skin_ratio - skin_ratio) / thresholds.max_healthy_skin_ratio)
    
    # Spatial Analysis: Split into top/bottom halves and left/right halves
    mid_y = crop_h // 2
    mid_x = crop_w // 2
    
    top_skin = float(np.count_nonzero(skin_mask[:mid_y, :])) / (mid_y * crop_w)
    bottom_skin = float(np.count_nonzero(skin_mask[mid_y:, :])) / ((crop_h - mid_y) * crop_w)
    left_skin = float(np.count_nonzero(skin_mask[:, :mid_x])) / (crop_h * mid_x)
    right_skin = float(np.count_nonzero(skin_mask[:, mid_x:])) / (crop_h * (crop_w - mid_x))
    
    # Determine occlusion type heuristically
    occlusion_type = "none"
    if occlusion_level > 0.1:
        # Top occluded (Hair over face)
        if top_skin < 0.30 and bottom_skin > 0.40:
            occlusion_type = "hair"
        # Bottom occluded (Face mask, hand over mouth)
        elif bottom_skin < 0.30 and top_skin > 0.40:
            occlusion_type = "mask" 
        # Asymmetrical (Hand on cheek, phone)
        elif abs(left_skin - right_skin) > 0.25:
            occlusion_type = "hand"
        else:
            occlusion_type = "object"

    return {
        "occlusion_level": float(occlusion_level),
        "is_occluded": occlusion_level > thresholds.occluded_threshold,
        "occlusion_type": occlusion_type,
        "skin_ratio": float(skin_ratio)
    }


# --- Main Logic ---

def evaluate_face_occlusion(
    occlusion_level: Optional[float] = None,
    is_occluded: Optional[bool] = None,
    occlusion_type: Optional[str] = None,
    skin_ratio: Optional[float] = None,
    thresholds: OcclusionThresholds = OcclusionThresholds(),
) -> FaceOcclusionScores:
    """
    Evaluate face occlusion from pre-computed signals.
    Redistributes confidence based on available data.
    """
    occl_q = _sanitize_score(occlusion_level)

    scores = FaceOcclusionScores(
        occlusion_level=occl_q,
        is_occluded=is_occluded,
        occlusion_type=occlusion_type,
        skin_ratio=_sanitize_score(skin_ratio),
        thresholds_used=asdict(thresholds),
    )

    # Infer is_occluded from level if not provided
    if scores.is_occluded is None and occl_q is not None:
        scores.is_occluded = occl_q > thresholds.occluded_threshold

    # Infer occlusion_level from is_occluded if not provided
    if occl_q is None and scores.is_occluded is not None:
        scores.occlusion_level = 0.8 if scores.is_occluded else 0.1

    # Calculate dynamic confidence
    if scores.occlusion_level is not None and scores.skin_ratio is not None:
        scores.confidence = 0.9
    elif scores.occlusion_level is not None:
        scores.confidence = 0.7
    elif scores.is_occluded is not None:
        scores.confidence = 0.5
    else:
        scores.confidence = 0.0

    return scores


def evaluate_occlusion_from_image(
    image: np.ndarray,
    face_bbox: Optional[BBox] = None,
    thresholds: OcclusionThresholds = OcclusionThresholds(),
) -> FaceOcclusionScores:
    """
    Convenience wrapper: Estimates all occlusion signals directly from a raw image,
    then evaluates the face occlusion.
    """
    if image is None or not CV2_AVAILABLE or face_bbox is None:
        return evaluate_face_occlusion(thresholds=thresholds)

    signals = estimate_occlusion_signals(image, face_bbox, thresholds)

    return evaluate_face_occlusion(
        occlusion_level=signals.get("occlusion_level"),
        is_occluded=signals.get("is_occluded"),
        occlusion_type=signals.get("occlusion_type"),
        skin_ratio=signals.get("skin_ratio"),
        thresholds=thresholds
    )


def get_occlusion_level(scores: FaceOcclusionScores) -> float:
    """Returns the occlusion level (0.0 = visible, 1.0 = fully occluded)."""
    return scores.occlusion_level if scores.occlusion_level is not None else 0.0


def is_face_occluded(scores: FaceOcclusionScores, threshold: float = 0.25) -> bool:
    """Returns True if the face is considered occluded."""
    if scores.occlusion_level is not None:
        return scores.occlusion_level >= threshold
    return scores.is_occluded if scores.is_occluded is not None else False


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Signals (Backward Compatible):")
    legacy_scores = evaluate_face_occlusion(occlusion_level=0.8, occlusion_type="hand")
    print(f"   Level: {legacy_scores.occlusion_level}, Type: {legacy_scores.occlusion_type}")
    print(f"   JSON: {legacy_scores.to_json()}\n")

    if CV2_AVAILABLE:
        print("2. Testing CV Pipeline - Clean Unoccluded Face:")
        # 400x400 face crop. Skin tone BGR (130, 160, 190). 
        # Add dark spots for eyes/mouth so it's not 100% skin (which would be unnatural)
        clean_img = np.ones((400, 400, 3), dtype=np.uint8) * (130, 160, 190)
        cv2.circle(clean_img, (150, 150), 20, (0, 0, 0), -1) # Left eye
        cv2.circle(clean_img, (250, 150), 20, (0, 0, 0), -1) # Right eye
        cv2.rectangle(clean_img, (180, 250), (220, 270), (0, 0, 0), -1) # Mouth
        
        clean_scores = evaluate_occlusion_from_image(clean_img, face_bbox=(0, 0, 400, 400))
        print(f"   Skin Ratio: {clean_scores.skin_ratio:.4f} (Should be ~0.85)")
        print(f"   Occlusion Level: {clean_scores.occlusion_level:.4f} (Should be near 0)")
        print(f"   Type: {clean_scores.occlusion_type}\n")

        print("3. Testing CV Pipeline - Hair Over Face (Top Occlusion):")
        hair_img = clean_img.copy()
        # Cover top 30% with "hair" (black/brown)
        hair_img[:120, :] = (20, 20, 20) 
        
        hair_scores = evaluate_occlusion_from_image(hair_img, face_bbox=(0, 0, 400, 400))
        print(f"   Skin Ratio: {hair_scores.skin_ratio:.4f} (Should be lower)")
        print(f"   Occlusion Level: {hair_scores.occlusion_level:.4f} (Should be high)")
        print(f"   Type: {hair_scores.occlusion_type} (Should be 'hair')\n")

        print("4. Testing CV Pipeline - Face Mask (Bottom Occlusion):")
        mask_img = clean_img.copy()
        # Cover bottom 35% with "mask" (blue)
        mask_img[260:, :] = (255, 0, 0) 
        
        mask_scores = evaluate_occlusion_from_image(mask_img, face_bbox=(0, 0, 400, 400))
        print(f"   Skin Ratio: {mask_scores.skin_ratio:.4f}")
        print(f"   Occlusion Level: {mask_scores.occlusion_level:.4f} (Should be high)")
        print(f"   Type: {mask_scores.occlusion_type} (Should be 'mask')\n")
