#!/usr/bin/env python3
"""
gaze_direction.py
-----------------
Supercharged module for evaluating the direction of the subject’s gaze.

Enhancements:
- Built-in geometric analysis: Automatically calculates head yaw (left/right) and 
  pitch (up/down) directly from facial landmarks.
- Iris tracking heuristic: Calculates gaze deviation using iris landmarks relative 
  to the eye center (detects looking away without turning the head).
- Configurable thresholds for categorization (at_camera, left, right, up, down, away).
- Dynamic confidence scoring based on distance from category boundaries.
- Robust input sanitization and JSON serialization for pipeline integration.
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


@dataclass
class GazeThresholds:
    """Configurable thresholds for gaze classification (in approximate degrees)."""
    at_camera_yaw: float = 12.0       # Max yaw deviation for "at_camera"
    at_camera_pitch: float = 12.0     # Max pitch deviation for "at_camera"
    looking_away_yaw: float = 20.0    # Yaw deviation to be classified as "left" or "right"
    looking_down_pitch: float = -15.0 # Pitch deviation to be classified as "down"
    looking_up_pitch: float = 15.0    # Pitch deviation to be classified as "up"
    iris_offset_limit: float = 0.25   # Max normalized iris offset for "at_camera"


@dataclass
class GazeDirectionScores:
    """Container for gaze direction classification results with serialization."""
    gaze_direction: str = "unknown"
    confidence: float = 0.0
    eye_contact_score: float | None = None
    head_yaw: float | None = None       # Negative = Left, Positive = Right
    head_pitch: float | None = None     # Negative = Down, Positive = Up
    iris_offset: float | None = None    # 0.0 (centered) to 1.0 (fully looking left/right)
    thresholds_used: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Converts scores to a dictionary, rounding for cleaner output."""
        data = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.items()}

    def to_json(self) -> str:
        """Serializes scores to a JSON string."""
        return json.dumps(self.to_dict())


def _sanitize_score(value: float | None) -> float | None:
    """Validates and clamps scores to [0.0, 1.0]. Returns None if invalid."""
    if value is None: return None
    if not isinstance(value, (int, float)): return None
    if math.isnan(value) or math.isinf(value): return None
    return max(0.0, min(1.0, float(value)))


def _dist(p1: Point | None, p2: Point | None) -> float:
    if not p1 or not p2: return 0.0
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# --- Geometric Estimation Helpers ---

def estimate_head_pose(landmarks: dict[str, Point]) -> tuple[float | None, float | None]:
    """
    Estimates head yaw (left/right) and pitch (up/down) in approximate degrees.
    """
    required = ["left_eye", "right_eye", "nose_tip", "mouth_left", "mouth_right"]
    if not all(k in landmarks for k in required):
        return None, None

    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    nose_tip = landmarks["nose_tip"]
    mouth_left = landmarks["mouth_left"]
    mouth_right = landmarks["mouth_right"]

    # 1. Yaw Calculation (Left/Right head turn)
    # Compare horizontal distance from nose to left eye vs nose to right eye.
    dist_left = abs(nose_tip[0] - left_eye[0])
    dist_right = abs(right_eye[0] - nose_tip[0])

    if dist_left + dist_right < 1e-6:
        yaw = 0.0
    else:
        # Ratio: 0.5 = straight, > 0.5 = turned right, < 0.5 = turned left
        ratio = dist_left / (dist_left + dist_right)
        # Map ratio 0.0-1.0 to -90 to +90 degrees
        yaw = (ratio - 0.5) * 180.0

    # 2. Pitch Calculation (Up/Down head tilt)
    # Compare vertical distance from eyes to nose vs nose to mouth.
    eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
    mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2.0

    dist_eye_nose = abs(eye_center_y - nose_tip[1])
    dist_nose_mouth = abs(mouth_center_y - nose_tip[1])

    if dist_nose_mouth < 1e-6:
        pitch = 0.0
    else:
        # Typical neutral ratio is ~0.8.
        # Looking down: nose gets closer to mouth (ratio drops).
        # Looking up: nose gets further from mouth (ratio rises).
        ratio = dist_eye_nose / dist_nose_mouth
        # Map ratio deviation to degrees (heuristic mapping)
        pitch = (ratio - 0.8) * 90.0

    return float(yaw), float(pitch)


def estimate_iris_offset(landmarks: dict[str, Point]) -> float | None:
    """
    Estimates how far the iris is offset from the center of the eye.
    Useful for detecting gaze aversion without head movement.
    Returns a normalized score from 0.0 (centered) to 1.0 (max offset).
    """
    required = ["left_eye", "right_eye", "left_iris", "right_iris"]
    if not all(k in landmarks for k in required):
        return None

    offsets = []
    for side in ["left", "right"]:
        eye = landmarks[f"{side}_eye"]
        iris = landmarks[f"{side}_iris"]

        # We need eye width to normalize the offset.
        # If we don't have eye corners, we can approximate eye width using the eye point itself,
        # but ideally we use left/right eye corners. Let's fallback to standard distance.
        # For this heuristic, we assume `left_eye` and `right_eye` are the centers of the eyes.
        # If you have `left_eye_outer` and `left_eye_inner`, this would be more accurate.
        # We will estimate eye width as the distance between the two eyes * 0.3.
        eye_dist = _dist(landmarks["left_eye"], landmarks["right_eye"])
        if eye_dist < 1e-6: continue

        eye_width_approx = eye_dist * 0.3

        offset = _dist(eye, iris) / eye_width_approx
        offsets.append(offset)

    if not offsets: return None
    return float(np.mean(offsets))


# --- Main Logic ---

def classify_gaze_direction(
    eye_contact_score: float | None = None,
    head_tilt: float | None = None,
    looking_down: bool | None = None,
    head_yaw: float | None = None,
    head_pitch: float | None = None,
    iris_offset: float | None = None,
    thresholds: GazeThresholds = GazeThresholds(),
) -> GazeDirectionScores:
    """
    Classify the direction of the subject’s gaze dynamically.
    Supports both pre-computed floats and geometric signals.
    """
    thresh_dict = asdict(thresholds)

    scores = GazeDirectionScores(
        eye_contact_score=_sanitize_score(eye_contact_score),
        head_pitch=head_pitch if head_pitch is not None else head_tilt,
        head_yaw=head_yaw,
        iris_offset=iris_offset,
        thresholds_used=thresh_dict,
    )

    # If we have geometric data, use it for superior classification
    has_geo = head_yaw is not None or head_pitch is not None or iris_offset is not None

    if has_geo:
        yaw = head_yaw or 0.0
        pitch = head_pitch or 0.0
        iris = iris_offset or 0.0

        # 1. Check Down
        if pitch < thresholds.looking_down_pitch:
            scores.gaze_direction = "down"
            scores.confidence = min(1.0, abs(pitch) / 30.0)

        # 2. Check Up
        elif pitch > thresholds.looking_up_pitch:
            scores.gaze_direction = "up"
            scores.confidence = min(1.0, pitch / 30.0)

        # 3. Check Left/Right (Yaw)
        elif yaw < -thresholds.looking_away_yaw:
            scores.gaze_direction = "left"
            scores.confidence = min(1.0, abs(yaw) / 45.0)

        elif yaw > thresholds.looking_away_yaw:
            scores.gaze_direction = "right"
            scores.confidence = min(1.0, yaw / 45.0)

        # 4. Check Iris offset (Looking away without turning head)
        elif iris > thresholds.iris_offset_limit:
            scores.gaze_direction = "away"
            scores.confidence = min(1.0, (iris - thresholds.iris_offset_limit) * 4.0)

        # 5. At Camera
        elif abs(yaw) < thresholds.at_camera_yaw and abs(pitch) < thresholds.at_camera_pitch and iris < thresholds.iris_offset_limit:
            scores.gaze_direction = "at_camera"
            # Confidence is 1.0 minus the normalized deviations
            yaw_dev = abs(yaw) / thresholds.at_camera_yaw
            pitch_dev = abs(pitch) / thresholds.at_camera_pitch
            iris_dev = iris / thresholds.iris_offset_limit
            max_dev = max(yaw_dev, pitch_dev, iris_dev)
            scores.confidence = 1.0 - (max_dev * 0.5) # Ranges from 0.5 to 1.0
            scores.eye_contact_score = scores.confidence

        else:
            scores.gaze_direction = "ambiguous"
            scores.confidence = 0.4

        return scores

    # Fallback to Legacy Logic (if only floats/booleans are provided)
    if eye_contact_score is None and looking_down is None and head_tilt is None:
        scores.gaze_direction = "unknown"
        scores.confidence = 0.0
        return scores

    if eye_contact_score is not None and eye_contact_score >= 0.75:
        scores.gaze_direction = "at_camera"
        scores.confidence = eye_contact_score
    elif looking_down or (head_tilt is not None and head_tilt < -0.3):
        scores.gaze_direction = "down"
        scores.confidence = 0.7 if looking_down else 0.6
    elif eye_contact_score is not None and eye_contact_score < 0.4:
        scores.gaze_direction = "away"
        scores.confidence = 1.0 - eye_contact_score
    else:
        scores.gaze_direction = "ambiguous"
        scores.confidence = 0.5

    return scores


def classify_gaze_from_landmarks(
    landmarks: dict[str, Point],
    thresholds: GazeThresholds = GazeThresholds(),
) -> GazeDirectionScores:
    """
    Convenience wrapper: Estimates head pose and iris offset directly from landmarks,
    then classifies the gaze direction.
    """
    yaw, pitch = estimate_head_pose(landmarks)
    iris = estimate_iris_offset(landmarks)

    return classify_gaze_direction(
        head_yaw=yaw,
        head_pitch=pitch,
        iris_offset=iris,
        thresholds=thresholds
    )


def get_gaze_direction(scores: GazeDirectionScores) -> str:
    """Returns the classified gaze direction."""
    return scores.gaze_direction


def get_eye_contact_score(scores: GazeDirectionScores) -> float:
    """Returns the eye contact score (if available)."""
    return scores.eye_contact_score if scores.eye_contact_score is not None else 0.0


# --- Example Usage / Tests ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("1. Testing Legacy Floats (Backward Compatible):")
    legacy_scores = classify_gaze_direction(eye_contact_score=0.9)
    print(f"   Direction: {legacy_scores.gaze_direction}, Confidence: {legacy_scores.confidence:.2f}\n")

    print("2. Testing Geometric Landmarks (Looking at Camera):")
    # Symmetrical face, eyes looking straight
    straight_landmarks = {
        "left_eye": (100, 100), "right_eye": (200, 100),
        "nose_tip": (150, 140),
        "mouth_left": (120, 180), "mouth_right": (180, 180),
        "left_iris": (100, 100), "right_iris": (200, 100)
    }
    straight_scores = classify_gaze_from_landmarks(straight_landmarks)
    print(f"   Direction: {straight_scores.gaze_direction}, Yaw: {straight_scores.head_yaw:.2f}, Pitch: {straight_scores.head_pitch:.2f}")
    print(f"   JSON: {straight_scores.to_json()}\n")

    print("3. Testing Geometric Landmarks (Looking Right):")
    # Nose shifted to the right
    right_landmarks = straight_landmarks.copy()
    right_landmarks["nose_tip"] = (175, 140)
    right_scores = classify_gaze_from_landmarks(right_landmarks)
    print(f"   Direction: {right_scores.gaze_direction}, Confidence: {right_scores.confidence:.2f}\n")

    print("4. Testing Geometric Landmarks (Looking Down):")
    # Nose shifted down (closer to mouth)
    down_landmarks = straight_landmarks.copy()
    down_landmarks["nose_tip"] = (150, 165)
    down_scores = classify_gaze_from_landmarks(down_landmarks)
    print(f"   Direction: {down_scores.gaze_direction}, Confidence: {down_scores.confidence:.2f}\n")

    print("5. Testing Geometric Landmarks (Iris Aversion):")
    # Head straight, but iris shifted
    aversion_landmarks = straight_landmarks.copy()
    aversion_landmarks["left_iris"] = (115, 100)
    aversion_landmarks["right_iris"] = (215, 100)
    aversion_scores = classify_gaze_from_landmarks(aversion_landmarks)
    print(f"   Direction: {aversion_scores.gaze_direction}, Confidence: {aversion_scores.confidence:.2f}\n")
