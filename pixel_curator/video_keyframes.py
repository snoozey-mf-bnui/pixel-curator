"""Studio expansion - smart keyframe selection for training.

Plans video keyframe extraction and selects the best training-ready frames
from candidate frame metadata - without requiring ffmpeg or decoding pixels in
this module. Two complementary capabilities:

  1. ``plan_keyframe_extraction(video_meta, ...)`` - produce a non-destructive
     extraction plan (sampling stride / target frame count / ffmpeg-friendly
     filter hints) from lightweight video metadata. Heavy extraction itself is
     delegated to ``pixel_curator.extract`` when the user runs it for real.

  2. ``select_keyframes(candidates, ...)`` - choose the most useful frames from
     already-extracted candidates using deterministic quality + temporal
     diversity scoring (sharpness, exposure, face visibility, spacing), so a
     long clip collapses to a compact, varied, high-quality training subset.

Local-first, non-destructive, dry-run friendly. Pure stdlib.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("redresser.video_keyframes")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")


@dataclass
class KeyframePlan:
    video_path: str
    duration_s: float = 0.0
    source_fps: float = 0.0
    target_fps: float = 0.5            # conservative sampling for catalog use
    max_frames: int = 200
    planned_frame_count: int = 0
    sampling_stride_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "duration_s": round(self.duration_s, 2),
            "source_fps": round(self.source_fps, 3),
            "target_fps": self.target_fps,
            "max_frames": self.max_frames,
            "planned_frame_count": self.planned_frame_count,
            "sampling_stride_s": round(self.sampling_stride_s, 3),
            "notes": self.notes,
        }


def plan_keyframe_extraction(video_meta: dict, target_fps: float = 0.5,
                             max_frames: int = 200) -> dict:
    """Plan a non-destructive keyframe extraction from video metadata.

    ``video_meta`` may include ``path``/``video_path``, ``duration``/
    ``duration_s``, and ``fps``/``source_fps``. Never raises; missing fields
    yield a conservative plan with a warning note.
    """
    vm = video_meta or {}
    path = str(vm.get("path") or vm.get("video_path") or "")
    try:
        duration = float(vm.get("duration_s", vm.get("duration", 0)) or 0)
    except (TypeError, ValueError):
        duration = 0.0
    try:
        source_fps = float(vm.get("source_fps", vm.get("fps", 0)) or 0)
    except (TypeError, ValueError):
        source_fps = 0.0

    plan = KeyframePlan(video_path=path, duration_s=duration,
                        source_fps=source_fps, target_fps=target_fps,
                        max_frames=max_frames)
    if target_fps > 0:
        plan.sampling_stride_s = round(1.0 / target_fps, 3)
    if duration > 0 and target_fps > 0:
        n = int(duration * target_fps)
        plan.planned_frame_count = min(n, max_frames)
        if n > max_frames:
            plan.notes.append(
                f"Sampling capped at {max_frames} frames "
                f"(would be {n} at {target_fps} fps).")
    else:
        plan.notes.append(
            "Duration/fps unavailable; run pixel_curator.extract to probe the "
            "video, then re-plan.")
    # ffmpeg-friendly hint (used by pixel_curator.extract.build_filter)
    plan.notes.append(f"ffmpeg filter hint: fps={target_fps}")
    return plan.to_dict()


def _f(meta: dict, key: str, default: float = 0.0) -> float:
    try:
        v = meta.get(key)
        return float(v) if v is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _frame_quality(c: dict) -> float:
    """Deterministic 0..1 quality from sharpness/exposure/face visibility."""
    sharp = _f(c, "sharpness", _f(c, "quality_score", 0.5))
    bright = _f(c, "brightness", _f(c, "exposure", 0.5))
    quality = _f(c, "quality_score", sharp)
    face_raw = c.get("face_visibility")
    if isinstance(face_raw, str):
        face = {"clear": 0.9, "visible": 0.8, "partial": 0.45,
                "none": 0.0}.get(face_raw.strip().lower(), 0.5)
    else:
        face = _f(c, "face_visibility", 0.5)
    # penalise poor exposure (ideal ~0.5)
    exposure_pen = 1.0 - min(abs(bright - 0.5) * 1.6, 1.0)
    return round(0.45 * quality + 0.25 * sharp + 0.15 * face
                 + 0.15 * exposure_pen, 4)


def select_keyframes(candidates: list[dict], max_keyframes: int = 40,
                     min_quality: float = 0.35,
                     min_spacing_s: float = 1.0) -> dict:
    """Select the best, most diverse keyframes from candidate frames.

    Each candidate may include ``path``/``sample_path``, ``timestamp_s`` (or
    ``frame_index``), ``sharpness``, ``brightness``/``exposure``,
    ``quality_score``, ``face_visibility``. Selection is greedy by quality with
    a temporal-spacing constraint so frames are varied, not near-duplicates.

    Returns ``{selected, rejected, selected_count, candidate_count,
    warnings}``. Non-destructive: no files are read or moved.
    """
    warnings: list[str] = []
    scored: list[dict] = []
    for i, c in enumerate(candidates or []):
        c = dict(c)
        ts = c.get("timestamp_s")
        if ts is None and c.get("frame_index") is not None:
            try:
                ts = float(c["frame_index"]) / float(c.get("fps", 1) or 1)
            except (TypeError, ValueError):
                ts = float(i)
        c["_ts"] = float(ts) if ts is not None else float(i)
        c["_quality"] = _frame_quality(c)
        c["_path"] = c.get("path") or c.get("sample_path") or f"frame_{i:05d}"
        scored.append(c)

    # quality gate
    eligible = [c for c in scored if c["_quality"] >= min_quality]
    gated = len(scored) - len(eligible)
    if gated:
        warnings.append(f"{gated} candidate frame(s) below quality "
                        f"threshold {min_quality}.")

    # greedy selection: highest quality first, enforce temporal spacing
    eligible.sort(key=lambda c: (-c["_quality"], c["_ts"]))
    selected: list[dict] = []
    chosen_ts: list[float] = []
    for c in eligible:
        if len(selected) >= max_keyframes:
            break
        if any(abs(c["_ts"] - t) < min_spacing_s for t in chosen_ts):
            continue
        selected.append(c)
        chosen_ts.append(c["_ts"])

    selected.sort(key=lambda c: c["_ts"])            # restore temporal order
    selected_rows = [{
        "path": c["_path"], "timestamp_s": round(c["_ts"], 3),
        "quality": c["_quality"],
        "reason_codes": ["keyframe_quality_high", "temporal_diversity"],
    } for c in selected]
    selected_paths = {c["_path"] for c in selected}
    rejected_rows = [{
        "path": c["_path"], "timestamp_s": round(c["_ts"], 3),
        "quality": c["_quality"],
        "reason_codes": (["below_quality_threshold"]
                         if c["_quality"] < min_quality
                         else ["too_close_to_selected_keyframe"]),
    } for c in scored if c["_path"] not in selected_paths]

    if not selected_rows and scored:
        warnings.append("No keyframes met the selection criteria; relax "
                        "min_quality or min_spacing_s.")
    return {
        "selected": selected_rows,
        "rejected": rejected_rows,
        "selected_count": len(selected_rows),
        "candidate_count": len(scored),
        "max_keyframes": max_keyframes,
        "min_quality": min_quality,
        "min_spacing_s": min_spacing_s,
        "warnings": warnings,
    }


def keyframes_to_samples(selection: dict, lora_target: str | None = None
                         ) -> list[dict]:
    """Convert selected keyframes into DatasetSplitter-ready sample dicts."""
    rows: list[dict] = []
    for kf in (selection or {}).get("selected", []):
        meta = {"source_type": "video_frame",
                "quality_score": kf.get("quality"),
                "timestamp_s": kf.get("timestamp_s")}
        if lora_target:
            meta["lora_target"] = lora_target
        rows.append({"sample_path": kf["path"], "source_type": "image",
                     "metadata": meta, "warnings": []})
    return rows


__all__ = [
    "VIDEO_EXTENSIONS", "KeyframePlan", "plan_keyframe_extraction",
    "select_keyframes", "keyframes_to_samples",
]
