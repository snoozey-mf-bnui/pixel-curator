"""Video keyframe pruning — collapse long-video frame floods to a diverse subset.

A 7-minute clip sampled at a fixed fps produces hundreds of near-identical
frames from every static shot. That floods the dataset, wastes curation time,
and fights diversity (one shot dominates). This connects the previously-orphaned
``video_keyframes.select_keyframes`` into the pipeline: for each VIDEO source it
keeps only the best, temporally-spaced frames and marks the rest
``rejected_redundant`` (non-destructive status change; photos are untouched).

Run it as a curation step after scoring:  ``pixel-curator keyframes [--apply]``.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict

from ..video_keyframes import select_keyframes

KEEP_STATUSES = ("selected", "packaged", "scored")
REDUNDANT_STATUS = "rejected_redundant"


def plan_keyframe_prune(by_source: dict, *, max_per_video: int = 80,
                        min_spacing: float = 2.0, min_quality: float = 0.0) -> dict:
    """Pure: given {source_id: [frame dicts]}, decide keep vs drop per source
    using temporal-diversity keyframe selection. Frame dicts carry frame_id,
    path, sharpness, brightness. Returns {keep_ids, drop_ids, per_source}."""
    keep_ids, drop_ids, per_source = [], [], []
    for src, items in by_source.items():
        items = sorted(items, key=lambda c: str(c.get("path") or ""))
        cands = []
        for i, c in enumerate(items):
            cands.append({"frame_id": c["frame_id"], "path": c.get("path"),
                          "timestamp_s": float(i),          # ordinal proxy spacing
                          "sharpness": c.get("sharpness") or 0.0,
                          "brightness": c.get("brightness") or 0.0})
        res = select_keyframes(cands, max_keyframes=max_per_video,
                               min_quality=min_quality, min_spacing_s=min_spacing)
        # select_keyframes returns rows keyed by path; map back to frame_id
        kept_paths = {r.get("path") for r in res.get("selected", [])}
        n_kept = 0
        for c in items:
            if c.get("path") in kept_paths:
                keep_ids.append(c["frame_id"]); n_kept += 1
            else:
                drop_ids.append(c["frame_id"])
        per_source.append({"source_id": src, "frames": len(items),
                           "kept": n_kept, "dropped": len(items) - n_kept})
    return {"keep_ids": keep_ids, "drop_ids": drop_ids, "per_source": per_source}


def prune_video_keyframes(conn: sqlite3.Connection, *, max_per_video: int = 80,
                          min_spacing: float = 2.0, min_quality: float = 0.0,
                          apply: bool = False) -> dict:
    """Fetch video-source frames, plan the keyframe prune, and (when apply)
    mark the redundant ones. Dry-run by default. Reversible (status only)."""
    qs = ",".join("?" for _ in KEEP_STATUSES)
    try:
        rows = conn.execute(
            f"SELECT f.frame_id, f.path, f.source_id, f.sharpness, f.brightness "
            f"FROM frames f JOIN sources s ON s.source_id = f.source_id "
            f"WHERE s.kind = 'video' AND f.status IN ({qs})",
            KEEP_STATUSES).fetchall()
    except Exception as exc:
        return {"summary": {"videos": 0, "scanned": 0, "kept": 0, "dropped": 0},
                "applied": False, "warnings": [f"query failed: {exc}"]}

    by_source: dict = defaultdict(list)
    for fid, path, src, sharp, bright in rows:
        by_source[src].append({"frame_id": fid, "path": path,
                               "sharpness": sharp, "brightness": bright})

    plan = plan_keyframe_prune(by_source, max_per_video=max_per_video,
                               min_spacing=min_spacing, min_quality=min_quality)
    drop = plan["drop_ids"]
    if apply and drop:
        conn.executemany(
            f"UPDATE frames SET status='{REDUNDANT_STATUS}' WHERE frame_id=?",
            [(d,) for d in drop])
        conn.commit()
    return {
        "summary": {"videos": len(by_source), "scanned": len(rows),
                    "kept": len(plan["keep_ids"]), "dropped": len(drop)},
        "per_source": plan["per_source"], "applied": bool(apply and drop),
        "warnings": [],
    }


def render_keyframes(res: dict) -> str:
    s = res["summary"]
    mode = "APPLIED" if res["applied"] else "DRY-RUN"
    lines = ["", f"Video keyframe prune [{mode}]", "=" * 56,
             f"video sources : {s['videos']}",
             f"frames scanned: {s['scanned']}",
             f"keyframes kept: {s['kept']}",
             f"redundant     : {s['dropped']}"]
    busiest = sorted(res.get("per_source", []),
                     key=lambda r: r["dropped"], reverse=True)[:5]
    for r in busiest:
        if r["dropped"]:
            lines.append(f"   {r['source_id'][:16]:<18} {r['frames']} -> "
                         f"{r['kept']} kept ({r['dropped']} redundant)")
    for w in res.get("warnings", []):
        lines.append(f"  ! {w}")
    if not res["applied"]:
        lines.append("\nDry-run. Re-run with --apply to mark redundant frames "
                     "'rejected_redundant' (reversible; nothing is deleted).")
    return "\n".join(lines)
