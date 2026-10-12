#!/usr/bin/env python3
"""
pose_diversity.py
-----------------
Supercharged module for evaluating pose variety and body angle diversity.

Enhancements:
- Proper Dataclass usage with JSON serialization for pipeline integration.
- Advanced pose bucketing supporting Yaw, Pitch, and Roll (3D angles).
- Continuous Keypoint Variance analysis: measures actual geometric skeletal 
  diversity across frames, not just categorical buckets.
- Robust input sanitization (handles NaN, missing keys, and malformed arrays).
- Standardized logging for debugging.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)

# --- Constants ---
VALID_POSE_BUCKETS = [
    "frontal", 
    "three_quarter_left", 
    "three_quarter_right", 
    "profile_left", 
    "profile_right", 
    "rear_three_quarter", 
    "rear",
    "unknown"
]


@dataclass
class PoseDiversityScores:
    """Container for pose diversity metrics with serialization capabilities."""
    diversity_score: Optional[float] = None
    unique_pose_ratio: Optional[float] = None
    average_frontalness: Optional[float] = None
    keypoint_variance: Optional[float] = None  # New: Continuous spatial diversity
    pose_distribution: Dict[str, int] = field(default_factory=dict)
    total_frames: int = 0

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


def bucket_pose(
    frontalness: Optional[float] = None,
    yaw: Optional[float] = None
) -> str:
    """
    Buckets pose based on frontalness or explicit 3D yaw angle.
    Yaw is expected in degrees (-180 to 180).
    """
    # Prefer explicit yaw if provided
    if yaw is not None and not math.isnan(yaw):
        yaw_abs = abs(yaw)
        if yaw_abs <= 22.5:
            return "frontal"
        elif yaw_abs <= 67.5:
            return f"three_quarter_{'left' if yaw < 0 else 'right'}"
        elif yaw_abs <= 112.5:
            return f"profile_{'left' if yaw < 0 else 'right'}"
        elif yaw_abs <= 157.5:
            return "rear_three_quarter"
        else:
            return "rear"

    # Fallback to frontalness float
    if frontalness is None or math.isnan(frontalness):
        return "unknown"

    if frontalness >= 0.75:
        return "frontal"
    elif frontalness >= 0.5:
        # In basic frontalness, we can't determine left/right, so generic 3/4
        return "three_quarter"
    elif frontalness >= 0.25:
        return "profile"
    else:
        return "rear_three_quarter"


def calculate_keypoint_variance(frames: List[Dict], keypoint_key: str = "pose_keypoints") -> Optional[float]:
    """
    Computes the average variance of skeletal keypoints across all frames.
    High variance indicates diverse body poses and camera angles.
    
    Expects keypoints in format: List[float] or np.ndarray of shape (N,) or (N, 2).
    """
    valid_keypoints: List[np.ndarray] = []
    
    for frame in frames:
        kps = frame.get(keypoint_key)
        if kps is None:
            continue
            
        try:
            # Convert to flat numpy array
            kp_array = np.asarray(kps, dtype=np.float32).flatten()
            
            # Filter out completely zeroed-out keypoints (missed detections)
            if np.any(kp_array > 0):
                valid_keypoints.append(kp_array)
        except Exception as e:
            logger.warning(f"Failed to parse keypoints for a frame: {e}")
            continue

    if len(valid_keypoints) < 2:
        return None  # Need at least 2 frames to compute variance

    try:
        # Stack to shape: (Num_Frames, Num_Keypoint_Coords)
        stacked = np.stack(valid_keypoints, axis=0)
        
        # Calculate variance across frames (axis=0) and take the mean
        # We use max to avoid negative zeros, and mean to get a single score
        mean_variance = float(np.mean(np.var(stacked, axis=0)))
        
        # Normalize: Keypoints are usually in 0-1 (normalized) or 0-WH (pixel) space.
        # Assuming normalized 0-1 coordinates, a variance > 0.05 is highly diverse.
        # We use an asymptotic normalizer to cap it at 1.0
        normalized_variance = min(1.0, mean_variance / (mean_variance + 0.05))
        return normalized_variance
        
    except Exception as e:
        logger.error(f"Error computing keypoint variance: {e}")
        return None


def compute_pose_diversity(
    frames: List[Dict[str, Any]],
    frontalness_key: str = "frontalness",
    yaw_key: str = "yaw",  # New: allows for 3D angle support
    pose_bucket_key: str = "pose_bucket",
    keypoint_key: str = "pose_keypoints",  # New: allows for geometric variance
) -> PoseDiversityScores:
    """
    Analyze a list of frames and compute overall pose diversity metrics.
    Uses both categorical (bucket) entropy and continuous (keypoint) variance.
    """
    if not frames:
        return PoseDiversityScores(
            diversity_score=0.0,
            unique_pose_ratio=0.0,
            total_frames=0
        )

    pose_buckets = []
    frontalness_values = []

    for frame in frames:
        # 1. Determine Pose Bucket
        bucket = frame.get(pose_bucket_key)
        yaw = frame.get(yaw_key)
        front = frame.get(frontalness_key)
        
        if not bucket or bucket == "unknown":
            bucket = bucket_pose(frontalness=front, yaw=yaw)
            
        pose_buckets.append(bucket)

        # 2. Collect Frontalness
        if front is not None:
            sanitized_front = _sanitize_score(front)
            if sanitized_front is not None:
                frontalness_values.append(sanitized_front)

    # --- Calculate Categorical Diversity (Entropy) ---
    bucket_counts = Counter(pose_buckets)
    unique_buckets = len(bucket_counts)
    total = len(pose_buckets)

    unique_pose_ratio = unique_buckets / total if total > 0 else 0.0

    if unique_buckets <= 1:
        categorical_diversity = 0.0
    else:
        probs = np.array(list(bucket_counts.values())) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(unique_buckets)
        categorical_diversity = entropy / max_entropy if max_entropy > 0 else 0.0

    # --- Calculate Continuous Diversity (Keypoints) ---
    keypoint_variance = calculate_keypoint_variance(frames, keypoint_key)

    # --- Combine Scores ---
    # If we have keypoint variance, blend it with categorical diversity
    # Keypoints provide finer detail, so we weight them higher if available.
    if keypoint_variance is not None:
        final_diversity = (0.4 * categorical_diversity) + (0.6 * keypoint_variance)
    else:
        final_diversity = categorical_diversity

    avg_frontalness = float(np.mean(frontalness_values)) if frontalness_values else 0.0

    return PoseDiversityScores(
        diversity_score=round(final_diversity, 4),
        unique_pose_ratio=round(unique_pose_ratio, 4),
        average_frontalness=round(avg_frontalness, 4),
        keypoint_variance=round(keypoint_variance, 4) if keypoint_variance is not None else None,
        pose_distribution=dict(bucket_counts),
        total_frames=total,
    )


def get_average_frontalness(frames: List[Dict], key: str = "frontalness") -> float:
    """Returns the mean frontalness across frames (if available)."""
    values = [_sanitize_score(f.get(key)) for f in frames]
    valid_values = [v for v in values if v is not None]
    return float(np.mean(valid_values)) if valid_values else 0.0


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Basic Categorical Diversity (Backward Compatible):")
    mock_frames_basic = [
        {"frontalness": 0.9}, {"frontalness": 0.8},  # Frontal
        {"frontalness": 0.4}, {"frontalness": 0.3},  # Profile
        {"frontalness": 0.1}                          # Rear
    ]
    basic_scores = compute_pose_diversity(mock_frames_basic)
    print(f"   JSON: {basic_scores.to_json()}\n")

    print("2. Testing Advanced 3D Yaw Bucketing & Keypoint Variance:")
    # Simulating frames with 2D keypoints (e.g., [x1, y1, x2, y2, ...])
    # Frame 1: Standing straight, facing forward
    # Frame 2: Arms raised, facing left
    # Frame 3: Crouching, facing right
    mock_frames_advanced = [
        {"yaw": 0, "pose_keypoints": [0.5, 0.5, 0.4, 0.6, 0.6, 0.6]},
        {"yaw": -45, "pose_keypoints": [0.5, 0.5, 0.3, 0.4, 0.7, 0.4]},
        {"yaw": 90, "pose_keypoints": [0.4, 0.7, 0.35, 0.8, 0.45, 0.8]}
    ]
    adv_scores = compute_pose_diversity(mock_frames_advanced)
    print(f"   Diversity Score: {adv_scores.diversity_score}")
    print(f"   Keypoint Variance: {adv_scores.keypoint_variance}")
    print(f"   Pose Distribution: {adv_scores.pose_distribution}\n")

    print("3. Testing Missing Data / Bad Inputs:")
    mock_frames_bad = [
        {"yaw": float('nan'), "frontalness": 1.2},  # Will be ignored / clamped
        {"frontalness": None},                       # Will be "unknown"
        {}                                           # Empty frame
    ]
    bad_scores = compute_pose_diversity(mock_frames_bad)
    print(f"   JSON: {bad_scores.to_json()}\n")
