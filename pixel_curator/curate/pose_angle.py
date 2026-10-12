"""Priority 5 - pose diversity & body-angle normalization.

Turns whatever pose signals happen to be available (pose buckets, frontalness,
dynamic-pose score, full-body visibility, keypoint variance) into one small,
manifest-safe ``pose`` block using conservative buckets. It never invents
precise labels: when a signal is absent the field stays ``unknown`` / ``None``.

Also provides dataset-level pose/body-angle distribution stats and
warning-only dominance checks for the Priority-4 card / validator.
"""
from __future__ import annotations

from collections import Counter

# Conservative, closed vocabularies (no invented fine-grained labels).
BODY_ANGLE_BUCKETS = ("unknown", "front", "three_quarter", "side", "back")
POSE_LABELS = ("unknown", "standing", "seated", "reclined", "dynamic",
               "closeup", "partial", "full_body")

# Map the richer pose_diversity / yaw buckets down to coarse body-angle buckets.
_ANGLE_FROM_BUCKET = {
    "frontal": "front", "front": "front",
    "three_quarter": "three_quarter", "three_quarter_left": "three_quarter",
    "three_quarter_right": "three_quarter",
    "profile": "side", "profile_left": "side", "profile_right": "side",
    "side": "side",
    "rear_three_quarter": "back", "rear": "back", "back": "back",
}
_LABEL_FROM_FRAMING = {
    "full_body": "full_body", "upper_body": "partial", "half_body": "partial",
    "portrait": "partial", "close_up": "closeup", "closeup": "closeup",
}
_LABEL_FROM_POSE = {
    "standing": "standing", "sitting": "seated", "seated": "seated",
    "reclined": "reclined", "lying": "reclined", "dynamic": "dynamic",
    "walking": "dynamic", "action": "dynamic", "model": "dynamic",
}


def _flat(sample: dict) -> dict:
    out = dict(sample or {})
    for nest in ("signals", "metadata", "pose", "card_fields"):
        blk = (sample or {}).get(nest)
        if isinstance(blk, dict):
            for k, v in blk.items():
                out.setdefault(k, v)
    return out


def _first(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _num(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _angle_from_yaw(yaw):
    y = _num(yaw)
    if y is None:
        return "unknown", None
    ya = abs(y)
    if ya <= 22.5:
        bucket = "front"
    elif ya <= 67.5:
        bucket = "three_quarter"
    elif ya <= 112.5:
        bucket = "side"
    else:
        bucket = "back"
    return bucket, round(y, 2)


def normalize_pose_signals(sample: dict | None) -> dict:
    """Normalize available pose signals into the standard ``pose`` block.

    Always returns a dict shaped like::

        {"pose_label", "body_angle_bucket", "angle_degrees",
         "dynamic_pose_score", "keypoint_variance", "full_body_visible",
         "confidence"}

    Missing signals yield conservative ``unknown`` / ``None`` values. Never
    raises.
    """
    f = _flat(sample or {})
    signals_present = 0

    # --- body angle ---
    angle_bucket, angle_deg = "unknown", None
    yaw = _first(f, "yaw", "angle_degrees", "body_angle_degrees")
    if yaw is not None:
        angle_bucket, angle_deg = _angle_from_yaw(yaw)
        if angle_bucket != "unknown":
            signals_present += 1
    if angle_bucket == "unknown":
        raw_bucket = _first(f, "body_angle_bucket", "pose_bucket")
        if raw_bucket is not None:
            mapped = _ANGLE_FROM_BUCKET.get(str(raw_bucket).lower())
            if mapped:
                angle_bucket = mapped
                signals_present += 1
    if angle_bucket == "unknown":
        front = _num(_first(f, "frontalness", "average_frontalness"))
        if front is not None:
            if front >= 0.75:
                angle_bucket = "front"
            elif front >= 0.5:
                angle_bucket = "three_quarter"
            elif front >= 0.25:
                angle_bucket = "side"
            else:
                angle_bucket = "back"
            signals_present += 1

    # --- pose label ---
    pose_label = "unknown"
    explicit = _first(f, "pose_label", "pose_type")
    if explicit is not None:
        pose_label = _LABEL_FROM_POSE.get(str(explicit).lower(), "unknown")
        if pose_label != "unknown":
            signals_present += 1
    if pose_label == "unknown":
        framing = _first(f, "framing_type", "framing_label", "framing")
        if framing is not None:
            pose_label = _LABEL_FROM_FRAMING.get(str(framing).lower(), "unknown")
            if pose_label != "unknown":
                signals_present += 1

    # --- continuous signals ---
    dyn = _num(_first(f, "dynamic_pose_score", "overall_dynamic_score"))
    if dyn is None:
        dp = (sample or {}).get("dynamic_pose")
        if isinstance(dp, dict):
            dyn = _num(dp.get("overall_dynamic_score"))
    if dyn is not None:
        signals_present += 1
        if pose_label == "unknown" and dyn >= 0.6:
            pose_label = "dynamic"

    kv = _num(_first(f, "keypoint_variance"))
    if kv is not None:
        signals_present += 1

    # --- full body visibility ---
    full_vis = None
    fbv = _num(_first(f, "full_body_visibility", "full_body_score"))
    if fbv is not None:
        full_vis = fbv >= 0.85
        signals_present += 1
    else:
        flag = _first(f, "is_full_body", "full_body_visible")
        if isinstance(flag, bool):
            full_vis = flag
            signals_present += 1
        elif pose_label == "full_body":
            full_vis = True

    explicit_conf = _num(_first(f, "pose_confidence", "confidence")) \
        if isinstance(sample, dict) and "pose" in sample else None
    confidence = explicit_conf if explicit_conf is not None else round(
        min(1.0, 0.2 * signals_present), 4)

    return {
        "pose_label": pose_label,
        "body_angle_bucket": angle_bucket,
        "angle_degrees": angle_deg,
        "dynamic_pose_score": dyn,
        "keypoint_variance": kv,
        "full_body_visible": full_vis,
        "confidence": confidence,
    }


def _ratio(counter: Counter, total: int):
    if not counter or total <= 0:
        return None, 0.0
    label, cnt = counter.most_common(1)[0]
    return label, round(cnt / total, 4)


def compute_pose_angle_stats(samples: list) -> dict:
    """Dataset-level pose/body-angle distribution + dominance ratios."""
    poses = [normalize_pose_signals(s) for s in (samples or [])]
    n = len(poses)
    label_counts = Counter(p["pose_label"] for p in poses
                           if p["pose_label"] != "unknown")
    angle_counts = Counter(p["body_angle_bucket"] for p in poses
                           if p["body_angle_bucket"] != "unknown")
    full = [p for p in poses if p["full_body_visible"] is True]
    dyn = [p["dynamic_pose_score"] for p in poses
           if p["dynamic_pose_score"] is not None]
    kv = [p["keypoint_variance"] for p in poses
          if p["keypoint_variance"] is not None]
    labelled = sum(label_counts.values())
    angled = sum(angle_counts.values())
    dom_pose_label, dom_pose_ratio = _ratio(label_counts, labelled or n)
    dom_angle_label, dom_angle_ratio = _ratio(angle_counts, angled or n)
    return {
        "pose_label_distribution": dict(label_counts.most_common()),
        "body_angle_distribution": dict(angle_counts.most_common()),
        "full_body_ratio": round(len(full) / n, 4) if n else 0.0,
        "dynamic_pose_average": round(sum(dyn) / len(dyn), 4) if dyn else None,
        "keypoint_variance_average": round(sum(kv) / len(kv), 4) if kv else None,
        "dominant_pose_label": dom_pose_label,
        "dominant_pose_ratio": dom_pose_ratio,
        "dominant_angle_label": dom_angle_label,
        "dominant_angle_ratio": dom_angle_ratio,
        "pose_signal_coverage": round((labelled + angled) / (2 * n), 4)
        if n else 0.0,
    }


def pose_validation_warnings(stats: dict, *, max_dominant_pose_ratio: float = 0.45,
                             max_dominant_angle_ratio: float = 0.50,
                             min_variety: int = 2,
                             require_signal: bool = False,
                             sample_count: int = 0) -> list:
    """Return ``(code, message, count)`` tuples for pose/angle dominance. Pure
    diagnostics - the caller decides severity and never blocks export."""
    out = []
    pose_dist = stats.get("pose_label_distribution") or {}
    angle_dist = stats.get("body_angle_distribution") or {}
    if stats.get("dominant_pose_ratio", 0.0) and \
            stats["dominant_pose_ratio"] > max_dominant_pose_ratio:
        out.append(("pose_dominance_high",
                    f"Pose '{stats.get('dominant_pose_label')}' is "
                    f"{stats['dominant_pose_ratio']:.0%} of labelled samples "
                    f"(> {max_dominant_pose_ratio:.0%}).",
                    int(stats["dominant_pose_ratio"] * max(1, sample_count))))
    if stats.get("dominant_angle_ratio", 0.0) and \
            stats["dominant_angle_ratio"] > max_dominant_angle_ratio:
        out.append(("body_angle_dominance_high",
                    f"Body angle '{stats.get('dominant_angle_label')}' is "
                    f"{stats['dominant_angle_ratio']:.0%} of angled samples "
                    f"(> {max_dominant_angle_ratio:.0%}).",
                    int(stats["dominant_angle_ratio"] * max(1, sample_count))))
    have_signal = bool(pose_dist) or bool(angle_dist)
    if have_signal and len(pose_dist) < min_variety and len(angle_dist) < min_variety:
        out.append(("pose_variety_low",
                    "Pose/body-angle variety is low; dataset may overrepresent "
                    "one pose/angle pattern.", sample_count))
    if not have_signal and require_signal and sample_count:
        out.append(("pose_signal_missing_for_enabled_check",
                    "Pose/body-angle tracking requested but no pose signals are "
                    "present on any sample.", sample_count))
    return out


__all__ = [
    "BODY_ANGLE_BUCKETS", "POSE_LABELS", "normalize_pose_signals",
    "compute_pose_angle_stats", "pose_validation_warnings",
]
