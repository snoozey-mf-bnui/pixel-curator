"""Framing / coverage-level taxonomy (Priority 3) - one shared definition.

Framing ("how much of the figure is in frame") was already derived inside smart
curation; this factors that single rule into one place so nothing duplicates it,
adds a clean taxonomy + a dataset report, and lets you optionally surface the
framing level as a caption token for controllability (prompt "full body" etc.).

  closeup     face fills the frame (>=40% face-height ratio)
  portrait    head-and-shoulders (25-40%)
  upper_body  waist-up (12-25%)
  full_body   whole figure (<12%)

SFW: this is purely a *coverage / composition* level, the standard
portrait-vs-full-figure distinction - nothing else.
"""
from __future__ import annotations

from collections import Counter

FRAMING_LEVELS = ("closeup", "portrait", "upper_body", "full_body")

# caption-friendly phrasing for each level (for optional controllability tags)
FRAMING_PHRASE = {
    "closeup": "close-up portrait",
    "portrait": "portrait",
    "upper_body": "upper body",
    "full_body": "full body",
}


def classify_framing(face_h_ratio: float) -> str:
    """Coverage level from the face-height / image-height ratio. This is THE
    definition - smart curation imports it so there's exactly one rule."""
    if face_h_ratio >= 0.40:
        return "closeup"
    if face_h_ratio >= 0.25:
        return "portrait"
    if face_h_ratio >= 0.12:
        return "upper_body"
    return "full_body"


def framing_phrase(level: str) -> str:
    return FRAMING_PHRASE.get(level, level or "")


def framing_distribution(conn, statuses=("selected", "packaged")) -> dict:
    """Count selected frames per framing level (reads detections.framing)."""
    qs = ",".join("?" for _ in statuses)
    try:
        rows = conn.execute(
            f"SELECT d.framing, COUNT(*) FROM detections d "
            f"JOIN frames f ON f.frame_id = d.frame_id "
            f"WHERE f.status IN ({qs}) GROUP BY d.framing", statuses).fetchall()
    except Exception:
        return {"counts": {}, "total": 0}
    counts = {k: v for k, v in rows if k}
    return {"counts": counts, "total": sum(counts.values())}


def render_framing(dist: dict) -> str:
    counts, total = dist["counts"], dist["total"]
    lines = ["", "Framing / coverage levels", "=" * 44]
    if not total:
        lines.append("no framing data - run `curate --smart` first.")
        return "\n".join(lines)
    for lvl in FRAMING_LEVELS:
        n = counts.get(lvl, 0)
        bar = "#" * int(30 * n / total) if total else ""
        lines.append(f"  {lvl:<12} {n:>6} ({100*n/total:4.0f}%) {bar}")
    other = {k: v for k, v in counts.items() if k not in FRAMING_LEVELS}
    if other:
        lines.append(f"  other        {other}")
    lines.append("\nUse framing in recipes to build portrait-only vs full-figure "
                 "sets, e.g.  build --recipe pose_v1  (framing=\"full_body\").")
    return "\n".join(lines)
