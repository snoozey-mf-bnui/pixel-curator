"""Phase 2: annotation & tagging layer (additive, manifest v13).

Free-form tags, custom metadata fields and caption history for the
Review workspace.  Bulk operations are the unit the UI endpoints call -
they are plain functions over the manifest connection so they test
without HTTP.  Decisions reuse the existing manual-verdict statuses
(selected / rejected_manual / scored) - nothing destructive, captions
are snapshotted to history before every overwrite.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from .util import now_iso

DECISIONS = {"selected": "selected",
             "rejected": "rejected_manual",
             "needs_review": "scored"}


def _split_tags(raw: str | None) -> list[str]:
    return [t for t in (raw or "").split(",") if t]


def get_annotations(conn, frame_ids: Iterable[str]) -> dict[str, dict]:
    ids = list(frame_ids)
    if not ids:
        return {}
    marks = ",".join("?" for _ in ids)
    out = {}
    for fid, tags, custom, ts in conn.execute(
            f"SELECT frame_id, tags, custom_json, updated_at "
            f"FROM annotations WHERE frame_id IN ({marks})", ids):
        try:
            custom_d = json.loads(custom) if custom else {}
        except Exception:
            custom_d = {}
        out[fid] = {"tags": _split_tags(tags), "custom": custom_d,
                    "updated_at": ts}
    return out


def bulk_tag(conn, frame_ids: list[str], add: list[str] | None = None,
             remove: list[str] | None = None) -> dict:
    """Add/remove tags on many frames; returns per-frame tag state."""
    add = [t.strip() for t in (add or []) if t and t.strip()]
    remove = {t.strip() for t in (remove or []) if t and t.strip()}
    existing = get_annotations(conn, frame_ids)
    result = {}
    for fid in frame_ids:
        cur = existing.get(fid, {"tags": [], "custom": {}})
        tags = [t for t in cur["tags"] if t not in remove]
        for t in add:
            if t not in tags:
                tags.append(t)
        conn.execute(
            "INSERT INTO annotations(frame_id, tags, custom_json, "
            "updated_at) VALUES (?,?,?,?) ON CONFLICT(frame_id) DO UPDATE "
            "SET tags=excluded.tags, updated_at=excluded.updated_at",
            (fid, ",".join(tags), json.dumps(cur["custom"]), now_iso()))
        result[fid] = tags
    conn.commit()
    return {"ok": True, "updated": len(result), "tags": result}


def bulk_decision(conn, frame_ids: list[str], decision: str) -> dict:
    """Bulk manual verdict: selected / rejected / needs_review."""
    if decision not in DECISIONS:
        raise ValueError(
            f"unknown decision '{decision}' "
            f"(options: {', '.join(sorted(DECISIONS))})")
    status = DECISIONS[decision]
    conn.executemany("UPDATE frames SET status=? WHERE frame_id=?",
                     [(status, fid) for fid in frame_ids])
    conn.commit()
    return {"ok": True, "updated": len(frame_ids), "status": status}


def set_custom_metadata(conn, frame_ids: list[str],
                        custom: dict) -> dict:
    """Merge custom metadata fields (pose_label, lighting_label,
    expression_label, curation_note, ...) into each frame's annotation.
    Setting a key to null/empty removes it."""
    existing = get_annotations(conn, frame_ids)
    for fid in frame_ids:
        cur = existing.get(fid, {"tags": [], "custom": {}})
        merged = dict(cur["custom"])
        for k, v in (custom or {}).items():
            if v in (None, ""):
                merged.pop(k, None)
            else:
                merged[str(k)] = v
        conn.execute(
            "INSERT INTO annotations(frame_id, tags, custom_json, "
            "updated_at) VALUES (?,?,?,?) ON CONFLICT(frame_id) DO UPDATE "
            "SET custom_json=excluded.custom_json, "
            "updated_at=excluded.updated_at",
            (fid, ",".join(cur["tags"]), json.dumps(merged), now_iso()))
    conn.commit()
    return {"ok": True, "updated": len(frame_ids)}


def save_caption(conn, frame_id: str, caption: str) -> dict:
    """Caption edit with history: the previous caption is snapshotted to
    caption_history before the overwrite - never lost."""
    prev = conn.execute(
        "SELECT caption_text FROM captions WHERE frame_id=?",
        (frame_id,)).fetchone()
    if prev and (prev[0] or "") != caption:
        conn.execute(
            "INSERT INTO caption_history(frame_id, caption_text, saved_at) "
            "VALUES (?,?,?)", (frame_id, prev[0] or "", now_iso()))
    conn.execute(
        "INSERT INTO captions(frame_id, caption_text, edited, updated_at) "
        "VALUES (?,?,1,?) ON CONFLICT(frame_id) DO UPDATE SET "
        "caption_text=excluded.caption_text, edited=1, "
        "updated_at=excluded.updated_at",
        (frame_id, caption, now_iso()))
    conn.commit()
    history_n = conn.execute(
        "SELECT COUNT(*) FROM caption_history WHERE frame_id=?",
        (frame_id,)).fetchone()[0]
    return {"ok": True, "history_entries": history_n}


def caption_history(conn, frame_id: str, limit: int = 20) -> list[dict]:
    return [{"caption": r[0], "saved_at": r[1]} for r in conn.execute(
        "SELECT caption_text, saved_at FROM caption_history "
        "WHERE frame_id=? ORDER BY id DESC LIMIT ?", (frame_id, limit))]


def all_tags(conn, limit: int = 200) -> list[dict]:
    """Distinct annotation tags with usage counts (for the tag picker)."""
    counts: dict[str, int] = {}
    for (raw,) in conn.execute(
            "SELECT tags FROM annotations WHERE tags != ''"):
        for t in _split_tags(raw):
            counts[t] = counts.get(t, 0) + 1
    return [{"tag": t, "count": n} for t, n in
            sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]
