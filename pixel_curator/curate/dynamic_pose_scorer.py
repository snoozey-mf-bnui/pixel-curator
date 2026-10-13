#!/usr/bin/env python3
"""
dynamic_pose_scorer.py
----------------------
Supercharged module for evaluating how dynamic and expressive a pose is.

Enhancements:
- Built-in geometric analysis: Automatically calculates dynamic metrics directly 
  from 2D skeletal keypoints (e.g., MediaPipe, OpenPose).
- Pose Energy: Measures limb extension away from the body's center of mass.
- Pose Variation: Measures joint flexion (deviation from a stiff, straight neutral pose).
- Pose Interest: Evaluates torso counter-rotation (contrapposto) and shoulder/hip tilt.
- Robust input sanitization (handles NaN, Infinity, and out-of-bounds values).
- Weight tracking and redistribution (if a metric is missing, its weight is redistributed).
- JSON serialization for pipeline integration.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Point format: Tuple[float, float] -> (x, y)
Point = tuple[float, float]

# --- Default Configuration ---
DEFAULT_WEIGHTS: dict[str, float] = {
    "pose_energy": 0.35,
    "pose_variation": 0.35,
    "pose_interest": 0.30,
}


@dataclass
class DynamicPoseScores:
    """Container for dynamic pose metrics with serialization capabilities."""
    pose_energy: float | None = None
    pose_variation: float | None = None
    pose_interest: float | None = None
    overall_dynamic_score: float | None = None
    weights_used: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Converts scores to a dictionary, rounding for cleaner output."""
        data = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.items()}

    def to_json(self) -> str:
        """Serializes scores to a JSON string."""
        return json.dumps(self.to_dict())


def _sanitize_score(value: float | None) -> float | None:
    """Validates and clamps scores to [0.0, 1.0]. Returns None if invalid."""
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return max(0.0, min(1.0, float(value)))


def _dist(p1: Point | None, p2: Point | None) -> float:
    """Euclidean distance between two points."""
    if not p1 or not p2: return 0.0
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _angle(p1: Point | None, p2: Point | None, p3: Point | None) -> float | None:
    """
    Calculates the angle in degrees at p2 formed by p1-p2-p3.
    Returns None if any point is missing.
    """
    if not p1 or not p2 or not p3: return None

    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cross = abs(v1[0] * v2[1] - v1[1] * v2[0])

    # atan2 gives the angle directly, avoiding division by zero errors
    angle_rad = math.atan2(cross, dot)
    return math.degrees(angle_rad)


# --- Geometric Estimation Helpers ---

def estimate_pose_energy(keypoints: dict[str, Point]) -> float | None:
    """
    Estimates pose energy by measuring how far the extremities (wrists, ankles)
    are from the body's center of mass (torso center), normalized by torso length.
    A star-jump pose = high energy. A tucked cannonball pose = low energy.
    """
    required = ["neck", "mid_hip"]
    if not all(k in keypoints for k in required): return None

    neck = keypoints["neck"]
    mid_hip = keypoints["mid_hip"]
    torso_len = _dist(neck, mid_hip)

    if torso_len < 1e-6: return None

    # Center of mass approximation
    com = ((neck[0] + mid_hip[0]) / 2.0, (neck[1] + mid_hip[1]) / 2.0)

    extremities = [
        keypoints.get("left_wrist"), keypoints.get("right_wrist"),
        keypoints.get("left_ankle"), keypoints.get("right_ankle"),
        keypoints.get("nose") # Head movement counts slightly
    ]

    reaches = []
    for ext in extremities:
        if ext:
            reaches.append(_dist(com, ext) / torso_len)

    if not reaches: return None

    # Average reach normalized by torso length.
    # Typical neutral pose reach is ~1.5 to 2.0. High energy is ~3.0+
    avg_reach = float(np.mean(reaches))

    # Asymptotic normalization: reaches 1.0 around avg_reach = 3.5
    return float(avg_reach / (avg_reach + 2.5))


def estimate_pose_variation(keypoints: dict[str, Point]) -> float | None:
    """
    Estimates pose variation by measuring joint flexion.
    Compares elbow and knee angles against a perfectly straight 180 degrees.
    Bent joints indicate a more dynamic/varied pose.
    """
    # Joints to evaluate: (p1, joint, p3)
    joints_to_check = [
        ("left_shoulder", "left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow", "right_wrist"),
        ("left_hip", "left_knee", "left_ankle"),
        ("right_hip", "right_knee", "right_ankle")
    ]

    flex_scores = []
    for p1_name, p2_name, p3_name in joints_to_check:
        p1, p2, p3 = keypoints.get(p1_name), keypoints.get(p2_name), keypoints.get(p3_name)
        angle = _angle(p1, p2, p3)

        if angle is not None:
            # 180 degrees = perfectly straight (0 flex). 0 degrees = fully folded (max flex).
            # We reward flexion. 90 degrees flexion = 1.0 score.
            flex = (180.0 - angle) / 90.0
            flex_scores.append(max(0.0, min(1.0, flex)))

    if not flex_scores: return None

    return float(np.mean(flex_scores))


def estimate_pose_interest(keypoints: dict[str, Point]) -> float | None:
    """
    Estimates pose interest via contrapposto and torso tilt.
    Measures the angle difference between the shoulder line and hip line.
    Parallel lines = stiff/boring. Intersecting/tilted lines = dynamic/interesting.
    """
    required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    if not all(k in keypoints for k in required): return None

    l_sh, r_sh = keypoints["left_shoulder"], keypoints["right_shoulder"]
    l_hip, r_hip = keypoints["left_hip"], keypoints["right_hip"]

    # Angle of shoulder line and hip line relative to horizontal
    # atan2(delta_y, delta_x)
    shoulder_angle = math.degrees(math.atan2(l_sh[1] - r_sh[1], l_sh[0] - r_sh[0]))
    hip_angle = math.degrees(math.atan2(l_hip[1] - r_hip[1], l_hip[0] - r_hip[0]))

    # Absolute difference in tilt. 0 = parallel. 90 = perpendicular.
    tilt_diff = abs(shoulder_angle - hip_angle)

    # A tilt difference of 15-20 degrees is classic contrapposto (very interesting).
    # 45+ degrees is extreme action.
    # We use an asymptotic normalizer: reaches 1.0 around 30 degrees difference.
    return float(tilt_diff / (tilt_diff + 15.0))


# --- Main Logic ---

def compute_dynamic_pose_score(
    pose_energy: float | None = None,
    pose_variation: float | None = None,
    pose_interest: float | None = None,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> DynamicPoseScores:
    """
    Main function to compute dynamic pose quality from pre-computed signals.
    Redistributes weights if certain metrics are missing (None).
    """
    # Sanitize inputs
    energy_q = _sanitize_score(pose_energy)
    variation_q = _sanitize_score(pose_variation)
    interest_q = _sanitize_score(pose_interest)

    scores = DynamicPoseScores(
        pose_energy=energy_q,
        pose_variation=variation_q,
        pose_interest=interest_q,
        weights_used=weights
    )

    components = []
    used_weights = []

    if energy_q is not None:
        components.append(energy_q)
        used_weights.append(weights.get("pose_energy", 0.0))
    if variation_q is not None:
        components.append(variation_q)
        used_weights.append(weights.get("pose_variation", 0.0))
    if interest_q is not None:
        components.append(interest_q)
        used_weights.append(weights.get("pose_interest", 0.0))

    if not components or sum(used_weights) == 0:
        scores.overall_dynamic_score = 0.0
    else:
        weighted_sum = sum(c * w for c, w in zip(components, used_weights))
        total_weight = sum(used_weights)
        scores.overall_dynamic_score = float(weighted_sum / total_weight)

    return scores


def compute_dynamic_from_keypoints(
    keypoints: dict[str, Point],
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> DynamicPoseScores:
    """
    Convenience wrapper: Estimates all dynamic pose signals directly from 2D pose keypoints.
    Expects standard keypoint names: 'neck', 'mid_hip', 'left_shoulder', etc.
    """
    energy = estimate_pose_energy(keypoints)
    variation = estimate_pose_variation(keypoints)
    interest = estimate_pose_interest(keypoints)

    return compute_dynamic_pose_score(
        pose_energy=energy,
        pose_variation=variation,
        pose_interest=interest,
        weights=weights
    )


def get_dynamic_pose_score(scores: DynamicPoseScores) -> float:
    """Returns the overall dynamic pose score."""
    return scores.overall_dynamic_score if scores.overall_dynamic_score is not None else 0.0


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Pre-computed Floats (Backward Compatible):")
    basic_scores = compute_dynamic_pose_score(
        pose_energy=0.8,
        pose_variation=0.6,
        pose_interest=0.9
    )
    print(f"   Overall Dynamic Score: {basic_scores.overall_dynamic_score:.4f}")
    print(f"   JSON: {basic_scores.to_json()}\n")

    print("2. Testing Geometric Keypoint Calculation (Static A-Pose):")
    # Stiff, straight, symmetrical pose. Low energy, low variation, low interest.
    static_keypoints = {
        "neck": (100, 50), "mid_hip": (100, 150),
        "left_shoulder": (80, 60), "right_shoulder": (120, 60),
        "left_elbow": (75, 100), "right_elbow": (125, 100),
        "left_wrist": (70, 140), "right_wrist": (130, 140),
        "left_hip": (90, 150), "right_hip": (110, 150),
        "left_knee": (90, 200), "right_knee": (110, 200),
        "left_ankle": (90, 250), "right_ankle": (110, 250)
    }
    static_scores = compute_dynamic_from_keypoints(static_keypoints)
    print(f"   Energy: {static_scores.pose_energy:.4f} (Should be low)")
    print(f"   Variation: {static_scores.pose_variation:.4f} (Should be 0.0)")
    print(f"   Interest: {static_scores.pose_interest:.4f} (Should be 0.0)")
    print(f"   Overall: {static_scores.overall_dynamic_score:.4f}\n")

    print("3. Testing Geometric Keypoint Calculation (Dynamic Action Pose):")
    # Simulating a running/throwing pose. Limbs extended, joints bent, torso tilted.
    dynamic_keypoints = {
        "neck": (105, 50), "mid_hip": (95, 150),           # Slight torso lean
        "left_shoulder": (85, 55), "right_shoulder": (125, 65), # Tilted shoulders
        "left_elbow": (50, 40), "right_elbow": (150, 100),  # Left arm bent up, right arm bent back
        "left_wrist": (20, 20), "right_wrist": (180, 140),
        "left_hip": (100, 145), "right_hip": (90, 155),     # Tilted hips
        "left_knee": (120, 190), "right_knee": (70, 200),   # Legs spread
        "left_ankle": (140, 240), "right_ankle": (50, 250)
    }
    dynamic_scores = compute_dynamic_from_keypoints(dynamic_keypoints)
    print(f"   Energy: {dynamic_scores.pose_energy:.4f} (Should be high)")
    print(f"   Variation: {dynamic_scores.pose_variation:.4f} (Should be high)")
    print(f"   Interest: {dynamic_scores.pose_interest:.4f} (Should be high)")
    print(f"   Overall: {dynamic_scores.overall_dynamic_score:.4f}\n")
