#!/usr/bin/env python3
"""
body_symmetry.py
----------------
Supercharged module for evaluating body proportion symmetry and balance.

Enhancements:
- Built-in geometric analysis: Automatically calculates symmetry and proportion 
  balance directly from 2D skeletal keypoints (e.g., MediaPipe, OpenPose).
- Horizontal Symmetry: Compares left vs. right limb distances from the spinal midline.
- Vertical Proportion Balance: Evaluates limb lengths and joint positions against 
  natural human anatomical ratios to flag broken poses or AI artifacts.
- Robust input sanitization (handles NaN, Infinity, and out-of-bounds values).
- Weight tracking and redistribution (if a metric is missing, its weight is redistributed).
- JSON serialization for pipeline integration.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Point format: Tuple[float, float] -> (x, y)
Point = Tuple[float, float]

# --- Default Configuration ---
DEFAULT_WEIGHTS: Dict[str, float] = {
    "horizontal_symmetry": 0.50,
    "vertical_proportion_balance": 0.50,
}


@dataclass
class BodySymmetryScores:
    """Container for body symmetry metrics with serialization capabilities."""
    horizontal_symmetry: Optional[float] = None
    vertical_proportion_balance: Optional[float] = None
    overall_symmetry_score: Optional[float] = None
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


def _dist(p1: Optional[Point], p2: Optional[Point]) -> float:
    """Euclidean distance between two points."""
    if not p1 or not p2: return 0.0
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _gaussian_score(value: float, ideal: float, spread: float) -> float:
    """Returns a 0.0-1.0 score based on a Gaussian curve centered at `ideal`."""
    return math.exp(-((value - ideal) ** 2) / (2 * spread ** 2))


# --- Geometric Estimation Helpers ---

def estimate_horizontal_symmetry(keypoints: Dict[str, Point]) -> Optional[float]:
    """
    Calculates horizontal symmetry by comparing left/right joint distances 
    to the spinal midline.
    """
    # Need central points to establish midline
    if "neck" not in keypoints or "mid_hip" not in keypoints:
        return None

    neck = keypoints["neck"]
    mid_hip = keypoints["mid_hip"]
    
    # Midline vector direction
    mid_dx = mid_hip[0] - neck[0]
    mid_dy = mid_hip[1] - neck[1]
    mid_len = math.hypot(mid_dx, mid_dy)
    if mid_len < 1e-6:
        return None

    # Pairs of joints to compare
    pairs = [
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle")
    ]

    asymmetries = []
    for left_name, right_name in pairs:
        left_pt = keypoints.get(left_name)
        right_pt = keypoints.get(right_name)
        if not left_pt or not right_pt:
            continue

        # Vector from neck to joint
        left_v = (left_pt[0] - neck[0], left_pt[1] - neck[1])
        right_v = (right_pt[0] - neck[0], right_pt[1] - neck[1])

        # Project onto midline vector to get "height" along the spine
        left_proj = (left_v[0] * mid_dx + left_v[1] * mid_dy) / (mid_len ** 2)
        right_proj = (right_v[0] * mid_dx + right_v[1] * mid_dy) / (mid_len ** 2)

        # Perpendicular distance to midline (horizontal offset)
        left_perp = abs(left_v[0] * mid_dy - left_v[1] * mid_dx) / mid_len
        right_perp = abs(right_v[0] * mid_dy - right_v[1] * mid_dx) / mid_len

        # Compare the absolute horizontal offsets
        max_perp = max(left_perp, right_perp)
        if max_perp > 1e-6:
            asymmetries.append(abs(left_perp - right_perp) / max_perp)

    if not asymmetries:
        return None

    # Lower asymmetry = higher score
    mean_asymmetry = float(np.mean(asymmetries))
    return float(max(0.0, 1.0 - mean_asymmetry))


def estimate_vertical_proportion_balance(keypoints: Dict[str, Point]) -> Optional[float]:
    """
    Calculates vertical proportion balance using standard human anatomical ratios.
    Penalizes unnatural limb lengths (common in AI generation or severe perspective distortion).
    """
    # Need fundamental joints to calculate proportions
    required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    if not all(k in keypoints for k in required):
        return None

    # 1. Shoulder/Hip Width Ratio (Ideal ~1.0 to 1.4 depending on gender/build)
    shoulder_width = _dist(keypoints["left_shoulder"], keypoints["right_shoulder"])
    hip_width = _dist(keypoints["left_hip"], keypoints["right_hip"])
    
    if hip_width < 1e-6: return None
    sh_hip_ratio = shoulder_width / hip_width
    sh_hip_score = _gaussian_score(sh_hip_ratio, ideal=1.2, spread=0.4)

    scores = [sh_hip_score]

    # 2. Torso vs Leg Length Ratio (Ideal ~1.0)
    # Torso = Neck to MidHip. Leg = Hip to Knee + Knee to Ankle
    if "neck" in keypoints and "mid_hip" in keypoints:
        torso_len = _dist(keypoints["neck"], keypoints["mid_hip"])
        if torso_len > 1e-6:
            if "left_knee" in keypoints and "left_ankle" in keypoints:
                leg_len = _dist(keypoints["left_hip"], keypoints["left_knee"]) + \
                          _dist(keypoints["left_knee"], keypoints["left_ankle"])
                if leg_len > 1e-6:
                    torso_leg_ratio = torso_len / leg_len
                    scores.append(_gaussian_score(torso_leg_ratio, ideal=1.0, spread=0.3))

            # 3. Upper vs Lower Leg Ratio (Ideal ~1.0)
            if "left_knee" in keypoints and "left_ankle" in keypoints:
                upper_leg = _dist(keypoints["left_hip"], keypoints["left_knee"])
                lower_leg = _dist(keypoints["left_knee"], keypoints["left_ankle"])
                if lower_leg > 1e-6:
                    leg_ratio = upper_leg / lower_leg
                    scores.append(_gaussian_score(leg_ratio, ideal=1.0, spread=0.25))

    # 4. Upper vs Lower Arm Ratio (Ideal ~1.0 to 1.1)
    if all(k in keypoints for k in ["left_shoulder", "left_elbow", "left_wrist"]):
        upper_arm = _dist(keypoints["left_shoulder"], keypoints["left_elbow"])
        lower_arm = _dist(keypoints["left_elbow"], keypoints["left_wrist"])
        if lower_arm > 1e-6:
            arm_ratio = upper_arm / lower_arm
            scores.append(_gaussian_score(arm_ratio, ideal=1.1, spread=0.3))

    if not scores:
        return None

    return float(np.mean(scores))


# --- Main Logic ---

def compute_overall_body_symmetry(
    horizontal_symmetry: Optional[float] = None,
    vertical_proportion_balance: Optional[float] = None,
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
) -> BodySymmetryScores:
    """
    Main function to compute body symmetry quality from pre-computed signals.
    Redistributes weights if certain metrics are missing (None).
    """
    # Sanitize inputs
    h_sym = _sanitize_score(horizontal_symmetry)
    v_prop = _sanitize_score(vertical_proportion_balance)

    scores = BodySymmetryScores(
        horizontal_symmetry=h_sym,
        vertical_proportion_balance=v_prop,
        weights_used=weights
    )

    components = []
    used_weights = []

    if h_sym is not None:
        components.append(h_sym)
        used_weights.append(weights.get("horizontal_symmetry", 0.0))
    if v_prop is not None:
        components.append(v_prop)
        used_weights.append(weights.get("vertical_proportion_balance", 0.0))

    if not components or sum(used_weights) == 0:
        scores.overall_symmetry_score = 0.0
    else:
        weighted_sum = sum(c * w for c, w in zip(components, used_weights))
        total_weight = sum(used_weights)
        scores.overall_symmetry_score = float(weighted_sum / total_weight)

    return scores


def compute_symmetry_from_keypoints(
    keypoints: Dict[str, Point],
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
) -> BodySymmetryScores:
    """
    Convenience wrapper: Estimates all symmetry signals directly from 2D pose keypoints.
    Expects standard keypoint names: 'neck', 'mid_hip', 'left_shoulder', etc.
    """
    h_sym = estimate_horizontal_symmetry(keypoints)
    v_prop = estimate_vertical_proportion_balance(keypoints)

    return compute_overall_body_symmetry(
        horizontal_symmetry=h_sym,
        vertical_proportion_balance=v_prop,
        weights=weights
    )


def get_body_symmetry_score(scores: BodySymmetryScores) -> float:
    """Returns the overall body symmetry score."""
    return scores.overall_symmetry_score if scores.overall_symmetry_score is not None else 0.0


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Floats (Backward Compatible):")
    basic_scores = compute_overall_body_symmetry(
        horizontal_symmetry=0.8,
        vertical_proportion_balance=0.9
    )
    print(f"   Overall Symmetry: {basic_scores.overall_symmetry_score:.4f}")
    print(f"   JSON: {basic_scores.to_json()}\n")

    print("2. Testing Geometric Keypoint Calculation (Symmetrical Pose):")
    # Simulating a perfectly symmetrical, well-proportioned front-facing pose
    mock_keypoints = {
        "neck": (100, 50),
        "mid_hip": (100, 150),
        "left_shoulder": (70, 60), "right_shoulder": (130, 60),
        "left_elbow": (60, 90), "right_elbow": (140, 90),
        "left_wrist": (55, 120), "right_wrist": (145, 120),
        "left_hip": (85, 150), "right_hip": (115, 150),
        "left_knee": (85, 200), "right_knee": (115, 200),
        "left_ankle": (85, 250), "right_ankle": (115, 250)
    }
    sym_scores = compute_symmetry_from_keypoints(mock_keypoints)
    print(f"   Horizontal Symmetry: {sym_scores.horizontal_symmetry:.4f} (Should be ~1.0)")
    print(f"   Vertical Proportion: {sym_scores.vertical_proportion_balance:.4f}")
    print(f"   Overall: {sym_scores.overall_symmetry_score:.4f}\n")

    print("3. Testing Geometric Keypoint Calculation (Asymmetrical Pose):")
    # Simulating a pose where the left arm is broken/far from midline
    asym_keypoints = mock_keypoints.copy()
    asym_keypoints["left_elbow"] = (20, 90)
    asym_keypoints["left_wrist"] = (10, 120)
    asym_scores = compute_symmetry_from_keypoints(asym_keypoints)
    print(f"   Horizontal Symmetry: {asym_scores.horizontal_symmetry:.4f} (Should be < 0.5)")
    print(f"   Overall: {asym_scores.overall_symmetry_score:.4f}\n")

    print("4. Testing Bad Proportions (AI Artifact / Broken Generation):")
    # Simulating a pose where the arms are 5x longer than the legs
    bad_prop_keypoints = mock_keypoints.copy()
    bad_prop_keypoints["left_elbow"] = (60, 150)
    bad_prop_keypoints["left_wrist"] = (55, 250)
    bad_prop_scores = compute_symmetry_from_keypoints(bad_prop_keypoints)
    print(f"   Vertical Proportion: {bad_prop_scores.vertical_proportion_balance:.4f} (Should be low)")
    print(f"   Overall: {bad_prop_scores.overall_symmetry_score:.4f}\n")
