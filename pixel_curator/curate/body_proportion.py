#!/usr/bin/env python3
"""
body_proportion.py
------------------
Supercharged module for evaluating anatomical body proportions.

Enhancements:
- Built-in geometric analysis: Automatically calculates head, torso, leg, and arm 
  lengths directly from 2D pose keypoints and a face bounding box.
- Biologically accurate scoring: Uses Gaussian distribution curves instead of rigid 
  linear drop-offs to evaluate how "natural" a ratio is.
- Scale-invariant: All lengths are normalized by body height, allowing it to work 
  regardless of how close or far the subject is from the camera.
- Hard penalty logic: Explicitly flags and caps scores for impossible anatomical ratios.
- Robust input sanitization, weight tracking, and JSON serialization.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)

# Bbox format: Tuple[int, int, int, int] -> (x_min, y_min, x_max, y_max)
BBox = Tuple[int, int, int, int]
# Point format: Tuple[float, float] -> (x, y)
Point = Tuple[float, float]

# Ideal human anatomical ratios (relative to overall body height)
# Source: Standard anatomical drawing ratios (e.g., Andrew Loomis / Vitruvian)
IDEAL_HEAD_BODY = 0.135  # Head is ~1/7.5 of body height
IDEAL_TORSO_LEG = 1.0    # Torso length roughly equals leg length
IDEAL_ARM_LEG = 0.85     # Arms are slightly shorter than legs


@dataclass
class BodyProportionScores:
    """Container for body proportion metrics with serialization capabilities."""
    head_body_ratio: Optional[float] = None
    torso_leg_ratio: Optional[float] = None
    arm_leg_ratio: Optional[float] = None
    overall_proportion_score: Optional[float] = None
    has_unrealistic_proportions: Optional[bool] = None
    confidence: float = 0.0
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
    if value is None: return None
    if not isinstance(value, (int, float)): return None
    if math.isnan(value) or math.isinf(value): return None
    return max(0.0, min(1.0, float(value)))


def _dist(p1: Optional[Point], p2: Optional[Point]) -> float:
    """Euclidean distance between two points."""
    if not p1 or not p2: return 0.0
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _gaussian_score(value: float, ideal: float, spread: float) -> float:
    """
    Returns a 0.0-1.0 score based on a Gaussian curve centered at `ideal`.
    Allows for natural biological variation without failing strict linear thresholds.
    """
    return math.exp(-((value - ideal) ** 2) / (2 * spread ** 2))


# --- Geometric Calculation Helpers ---

def calculate_ratios_from_geometry(
    face_bbox: Optional[BBox] = None,
    keypoints: Optional[Dict[str, Point]] = None
) -> Dict[str, Optional[float]]:
    """
    Calculates body ratios directly from a face bounding box and 2D pose keypoints.
    Expects standard keypoint names: 'neck', 'mid_hip', 'left_ankle', 'left_shoulder', etc.
    """
    if keypoints is None or "neck" not in keypoints or "mid_hip" not in keypoints:
        return {"head_body_ratio": None, "torso_leg_ratio": None, "arm_leg_ratio": None}

    neck = keypoints["neck"]
    mid_hip = keypoints["mid_hip"]
    
    # Determine ankle (use whichever is visible)
    ankle = keypoints.get("left_ankle") or keypoints.get("right_ankle")
    
    if not ankle:
        return {"head_body_ratio": None, "torso_leg_ratio": None, "arm_leg_ratio": None}

    # 1. Head Body Ratio
    head_body_ratio = None
    if face_bbox is not None:
        head_height = face_bbox[3] - face_bbox[1]  # y_max - y_min
        body_height = _dist(neck, ankle) + _dist(face_bbox[1], neck) # Approximate full height
        if body_height > 1e-6:
            head_body_ratio = head_height / body_height

    # 2. Torso Leg Ratio
    torso_leg_ratio = None
    torso_len = _dist(neck, mid_hip)
    leg_len = _dist(mid_hip, ankle)
    if leg_len > 1e-6:
        torso_leg_ratio = torso_len / leg_len

    # 3. Arm Leg Ratio
    arm_leg_ratio = None
    # Use left arm if available, else right
    shoulder = keypoints.get("left_shoulder") or keypoints.get("right_shoulder")
    elbow = keypoints.get("left_elbow") or keypoints.get("right_elbow")
    wrist = keypoints.get("left_wrist") or keypoints.get("right_wrist")
    
    if shoulder and elbow and wrist:
        arm_len = _dist(shoulder, elbow) + _dist(elbow, wrist)
        if leg_len > 1e-6:
            arm_leg_ratio = arm_len / leg_len

    return {
        "head_body_ratio": head_body_ratio,
        "torso_leg_ratio": torso_leg_ratio,
        "arm_leg_ratio": arm_leg_ratio
    }


# --- Main Logic ---

def evaluate_body_proportions(
    head_body_ratio: Optional[float] = None,
    torso_leg_ratio: Optional[float] = None,
    arm_leg_ratio: Optional[float] = None,
    has_unrealistic_proportions: Optional[bool] = None,
    weights: Dict[str, float] = field(default_factory=lambda: {
        "head_body_ratio": 0.40,
        "torso_leg_ratio": 0.35,
        "arm_leg_ratio": 0.25,
    }),
) -> BodyProportionScores:
    """
    Evaluate body proportions from pre-computed ratios.
    Uses Gaussian curves to reward biologically natural ratios.
    Redistributes weights if certain metrics are missing (None).
    """
    scores = BodyProportionScores(
        head_body_ratio=head_body_ratio,
        torso_leg_ratio=torso_leg_ratio,
        arm_leg_ratio=arm_leg_ratio,
        has_unrealistic_proportions=has_unrealistic_proportions,
        weights_used=weights
    )

    components = []
    used_weights = []

    if head_body_ratio is not None:
        # Ideal ~0.135. Spread of 0.05 allows 0.08 to 0.19 to score decently.
        score = _gaussian_score(head_body_ratio, IDEAL_HEAD_BODY, 0.05)
        components.append(score)
        used_weights.append(weights.get("head_body_ratio", 0.0))

    if torso_leg_ratio is not None:
        # Ideal ~1.0. Spread of 0.3 allows 0.7 to 1.3 to score decently.
        score = _gaussian_score(torso_leg_ratio, IDEAL_TORSO_LEG, 0.3)
        components.append(score)
        used_weights.append(weights.get("torso_leg_ratio", 0.0))

    if arm_leg_ratio is not None:
        # Ideal ~0.85. Spread of 0.2 allows 0.65 to 1.05 to score decently.
        score = _gaussian_score(arm_leg_ratio, IDEAL_ARM_LEG, 0.2)
        components.append(score)
        used_weights.append(weights.get("arm_leg_ratio", 0.0))

    # Hard fail: If explicitly flagged as unrealistic, cap score
    if has_unrealistic_proportions is True:
        scores.overall_proportion_score = 0.2
        scores.confidence = 0.9
        return scores

    if not components or sum(used_weights) == 0:
        scores.overall_proportion_score = 0.0
        scores.confidence = 0.0
    else:
        weighted_sum = sum(c * w for c, w in zip(components, used_weights))
        total_weight = sum(used_weights)
        raw_score = float(weighted_sum / total_weight)
        
        # Check for severe deformities (e.g., ratio outside 3x the ideal)
        if (head_body_ratio and head_body_ratio > 0.4) or \
           (torso_leg_ratio and torso_leg_ratio > 2.5) or \
           (arm_leg_ratio and arm_leg_ratio > 1.8):
            scores.has_unrealistic_proportions = True
            scores.overall_proportion_score = min(0.2, raw_score)
        else:
            scores.overall_proportion_score = raw_score
            
        # Confidence scales with how many metrics we evaluated
        scores.confidence = 0.4 + (0.2 * len(components))

    return scores


def evaluate_from_geometry(
    face_bbox: Optional[BBox] = None,
    keypoints: Optional[Dict[str, Point]] = None,
    weights: Dict[str, float] = field(default_factory=lambda: {
        "head_body_ratio": 0.40,
        "torso_leg_ratio": 0.35,
        "arm_leg_ratio": 0.25,
    }),
) -> BodyProportionScores:
    """
    Convenience wrapper: Auto-calculates ratios directly from a face bbox and keypoints,
    then evaluates the proportions.
    """
    ratios = calculate_ratios_from_geometry(face_bbox, keypoints)
    
    return evaluate_body_proportions(
        head_body_ratio=ratios.get("head_body_ratio"),
        torso_leg_ratio=ratios.get("torso_leg_ratio"),
        arm_leg_ratio=ratios.get("arm_leg_ratio"),
        weights=weights
    )


def get_proportion_score(scores: BodyProportionScores) -> float:
    """Returns the overall body proportion score."""
    return scores.overall_proportion_score if scores.overall_proportion_score is not None else 0.0


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Floats (Backward Compatible):")
    # Perfect ratios
    perfect_scores = evaluate_body_proportions(
        head_body_ratio=0.135,
        torso_leg_ratio=1.0,
        arm_leg_ratio=0.85
    )
    print(f"   Score: {perfect_scores.overall_proportion_score:.4f} (Should be ~1.0)")
    print(f"   JSON: {perfect_scores.to_json()}\n")

    print("2. Testing Slightly Off Proportions (Gaussian Falloff):")
    # Head slightly too small, legs slightly too long
    off_scores = evaluate_body_proportions(
        head_body_ratio=0.10,
        torso_leg_ratio=0.8,
        arm_leg_ratio=0.95
    )
    print(f"   Score: {off_scores.overall_proportion_score:.4f} (Should be decent, ~0.6-0.8)\n")

    print("3. Testing Severe AI Artifacts (Auto-Detection):")
    # Head is 50% of body height, legs are 3x torso
    bad_scores = evaluate_body_proportions(
        head_body_ratio=0.50,
        torso_leg_ratio=3.0,
        arm_leg_ratio=0.85
    )
    print(f"   Score: {bad_scores.overall_proportion_score:.4f} (Should be capped at 0.2)")
    print(f"   Unrealistic Flagged: {bad_scores.has_unrealistic_proportions}\n")

    print("4. Testing Geometric Auto-Calculation (Perfect Pose):")
    # Simulating a 500px tall subject.
    # Neck at y=100, MidHip at y=250, Ankle at y=500. Face from 50 to 100.
    mock_face = (100, 50, 150, 100)
    mock_keypoints = {
        "neck": (125, 100),
        "mid_hip": (125, 250),
        "left_ankle": (125, 500),
        "left_shoulder": (100, 110),
        "left_elbow": (90, 180),
        "left_wrist": (85, 250)
    }
    geo_scores = evaluate_from_geometry(face_bbox=mock_face, keypoints=mock_keypoints)
    print(f"   Head/Body Ratio: {geo_scores.head_body_ratio:.4f}")
    print(f"   Torso/Leg Ratio: {geo_scores.torso_leg_ratio:.4f}")
    print(f"   Arm/Leg Ratio: {geo_scores.arm_leg_ratio:.4f}")
    print(f"   Overall Score: {geo_scores.overall_proportion_score:.4f}\n")
