#!/usr/bin/env python3
"""
full_body_detector.py
---------------------
Supercharged module for detecting whether the full body is visible in a frame.

Enhancements:
- Built-in geometric analysis: Automatically calculates full body visibility 
  directly from 2D pose keypoints (checking for ankles/feet presence) or 
  bounding boxes (checking vertical span).
- Automated Framing Classification: Infers framing_type (full_body, upper_body, 
  portrait, close_up) based on visible joints and face-to-body ratios.
- Configurable thresholds for strict vs. loose full body detection.
- Dynamic confidence scoring based on the quality of the data provided.
- Robust input sanitization and JSON serialization for pipeline integration.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Bbox format: Tuple[int, int, int, int] -> (x_min, y_min, x_max, y_max)
BBox = Tuple[int, int, int, int]
# Point format: Tuple[float, float] -> (x, y)
Point = Tuple[float, float]


@dataclass
class FullBodyThresholds:
    """Configurable thresholds for full body classification."""
    # Vertical span of body bbox relative to image height to be considered "full body"
    full_body_span: float = 0.85       
    # Vertical span to be considered "upper body" (e.g., waist up)
    upper_body_span: float = 0.40
    # Face height to body height ratio to be considered "close up" or "portrait"
    portrait_face_ratio: float = 0.45


@dataclass
class FullBodyScores:
    """Container for full body detection results with serialization."""
    full_body_visibility: Optional[float] = None   # 0.0 (cropped) to 1.0 (full body visible)
    is_full_body: Optional[bool] = None
    framing_type: Optional[str] = None             # full_body, upper_body, portrait, close_up, unknown
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


def _is_visible(point: Optional[Point], img_w: int = 0, img_h: int = 0) -> bool:
    """Checks if a keypoint is present and within image bounds."""
    if not point: return False
    if img_w > 0 and (point[0] < 0 or point[0] > img_w): return False
    if img_h > 0 and (point[1] < 0 or point[1] > img_h): return False
    return True


# --- Geometric Estimation Helpers ---

def estimate_framing_from_geometry(
    img_width: int = 0,
    img_height: int = 0,
    face_bbox: Optional[BBox] = None,
    body_bbox: Optional[BBox] = None,
    keypoints: Optional[Dict[str, Point]] = None,
    thresholds: FullBodyThresholds = FullBodyThresholds()
) -> Dict[str, Optional[float]]:
    """
    Estimates framing type and full body visibility from bounding boxes and keypoints.
    """
    has_kps = keypoints is not None
    has_bboxes = face_bbox is not None and body_bbox is not None and img_height > 0
    
    if not has_kps and not has_bboxes:
        return {}

    visibility = 0.0
    framing_type = "unknown"

    # 1. Keypoint-based Detection (Most Accurate)
    if has_kps:
        head_visible = _is_visible(keypoints.get("nose"), img_width, img_height) or \
                       _is_visible(keypoints.get("neck"), img_width, img_height)
                       
        left_ankle = _is_visible(keypoints.get("left_ankle"), img_width, img_height)
        right_ankle = _is_visible(keypoints.get("right_ankle"), img_width, img_height)
        feet_visible = left_ankle or right_ankle
        
        knees_visible = _is_visible(keypoints.get("left_knee"), img_width, img_height) or \
                        _is_visible(keypoints.get("right_knee"), img_width, img_height)
                        
        hips_visible = _is_visible(keypoints.get("left_hip"), img_width, img_height) or \
                       _is_visible(keypoints.get("right_hip"), img_width, img_height)

        if head_visible and feet_visible:
            visibility = 1.0
            framing_type = "full_body"
        elif head_visible and knees_visible and not feet_visible:
            visibility = 0.6
            framing_type = "upper_body"
        elif head_visible and hips_visible and not knees_visible:
            visibility = 0.4
            framing_type = "portrait"
        elif head_visible and not hips_visible:
            visibility = 0.2
            framing_type = "close_up"
            
    # 2. Bounding Box Fallback (If keypoints are missing)
    elif has_bboxes:
        x1, y1, x2, y2 = body_bbox
        body_h = y2 - y1
        span = body_h / img_height if img_height > 0 else 0
        
        face_h = face_bbox[3] - face_bbox[1]
        face_ratio = face_h / body_h if body_h > 0 else 0

        if span >= thresholds.full_body_span:
            visibility = min(1.0, span)
            framing_type = "full_body"
        elif face_ratio >= thresholds.portrait_face_ratio:
            visibility = 0.2
            framing_type = "close_up"
        elif span >= thresholds.upper_body_span:
            visibility = 0.5
            framing_type = "upper_body"
        else:
            visibility = 0.3
            framing_type = "portrait"

    return {
        "full_body_visibility": float(visibility),
        "is_full_body": visibility >= 0.85,
        "framing_type": framing_type
    }


# --- Main Logic ---

def detect_full_body(
    full_body_visibility: Optional[float] = None,
    is_full_body: Optional[bool] = None,
    framing_type: Optional[str] = None,
    thresholds: FullBodyThresholds = FullBodyThresholds(),
) -> FullBodyScores:
    """
    Evaluate full body visibility from pre-computed signals.
    Redistributes confidence based on available data.
    """
    vis_q = _sanitize_score(full_body_visibility)

    scores = FullBodyScores(
        full_body_visibility=vis_q,
        is_full_body=is_full_body,
        framing_type=framing_type,
        thresholds_used=asdict(thresholds),
    )

    # Infer is_full_body from visibility if not provided
    if scores.is_full_body is None and vis_q is not None:
        scores.is_full_body = vis_q >= thresholds.full_body_span

    # Infer visibility from boolean if not provided
    if vis_q is None and scores.is_full_body is not None:
        scores.full_body_visibility = 0.9 if scores.is_full_body else 0.4

    # Infer framing_type if not provided
    if scores.framing_type is None:
        if scores.is_full_body is True:
            scores.framing_type = "full_body"
        elif scores.is_full_body is False:
            scores.framing_type = "cropped"
        else:
            scores.framing_type = "unknown"

    # Calculate dynamic confidence
    if vis_q is not None and scores.framing_type not in ["unknown", "cropped"]:
        scores.confidence = 0.9
    elif vis_q is not None:
        scores.confidence = 0.7
    elif scores.is_full_body is not None:
        scores.confidence = 0.5
    else:
        scores.confidence = 0.0

    return scores


def detect_from_geometry(
    img_width: int = 0,
    img_height: int = 0,
    face_bbox: Optional[BBox] = None,
    body_bbox: Optional[BBox] = None,
    keypoints: Optional[Dict[str, Point]] = None,
    thresholds: FullBodyThresholds = FullBodyThresholds(),
) -> FullBodyScores:
    """
    Convenience wrapper: Estimates all framing signals directly from geometry,
    then evaluates the full body detection.
    """
    signals = estimate_framing_from_geometry(
        img_width, img_height, face_bbox, body_bbox, keypoints, thresholds
    )

    return detect_full_body(
        full_body_visibility=signals.get("full_body_visibility"),
        is_full_body=signals.get("is_full_body"),
        framing_type=signals.get("framing_type"),
        thresholds=thresholds
    )


def get_full_body_visibility(scores: FullBodyScores) -> float:
    """Returns full body visibility score (0.0 = cropped, 1.0 = full body)."""
    return scores.full_body_visibility if scores.full_body_visibility is not None else 0.5


def is_full_body_visible(scores: FullBodyScores, threshold: float = 0.85) -> bool:
    """Returns True if the full body is considered visible."""
    if scores.full_body_visibility is not None:
        return scores.full_body_visibility >= threshold
    return scores.is_full_body if scores.is_full_body is not None else False


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Signals (Backward Compatible):")
    legacy_scores = detect_full_body(full_body_visibility=0.9, framing_type="full_body")
    print(f"   Visibility: {legacy_scores.full_body_visibility}, Type: {legacy_scores.framing_type}")
    print(f"   JSON: {legacy_scores.to_json()}\n")

    print("2. Testing Geometric Keypoints (Full Body):")
    # Simulating a pose where head, hips, and ankles are visible in a 1080px tall image
    full_body_kps = {
        "nose": (500, 100),
        "neck": (500, 150),
        "left_hip": (450, 600), "right_hip": (550, 600),
        "left_knee": (440, 800), "right_knee": (560, 800),
        "left_ankle": (430, 1000), "right_ankle": (570, 1000)
    }
    kp_scores = detect_from_geometry(img_width=1000, img_height=1080, keypoints=full_body_kps)
    print(f"   Visibility: {kp_scores.full_body_visibility:.4f} (Should be 1.0)")
    print(f"   Framing: {kp_scores.framing_type} (Should be full_body)\n")

    print("3. Testing Geometric Keypoints (Upper Body / Cropped Legs):")
    # Simulating a pose where ankles and knees are missing
    upper_body_kps = {
        "nose": (500, 100),
        "neck": (500, 150),
        "left_hip": (450, 600), "right_hip": (550, 600)
        # No knees or ankles
    }
    upper_scores = detect_from_geometry(img_width=1000, img_height=1080, keypoints=upper_body_kps)
    print(f"   Visibility: {upper_scores.full_body_visibility:.4f}")
    print(f"   Framing: {upper_scores.framing_type} (Should be upper_body)\n")

    print("4. Testing Geometric BBox Fallback (Full Body via Span):")
    # Simulating an image where the body bbox takes up 90% of the vertical height
    face_bbox = (400, 50, 600, 250)
    body_bbox = (300, 50, 700, 1010) # 960px tall in 1080px image
    bbox_scores = detect_from_geometry(
        img_width=1000, img_height=1080, face_bbox=face_bbox, body_bbox=body_bbox
    )
    print(f"   Visibility: {bbox_scores.full_body_visibility:.4f}")
    print(f"   Framing: {bbox_scores.framing_type} (Should be full_body)\n")

    print("5. Testing Geometric BBox Fallback (Close Up via Face Ratio):")
    # Simulating an image where the face takes up 60% of the body bbox height
    face_bbox = (400, 100, 600, 400) # 300px tall face
    body_bbox = (350, 100, 650, 600) # 500px tall body
    closeup_scores = detect_from_geometry(
        img_width=1000, img_height=1080, face_bbox=face_bbox, body_bbox=body_bbox
    )
    print(f"   Visibility: {closeup_scores.full_body_visibility:.4f}")
    print(f"   Framing: {closeup_scores.framing_type} (Should be close_up)")
