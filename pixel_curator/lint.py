"""Phase 4: Active Linting - detection + safe, journaled, reversible fixes.

Extends the report-only lint (`insights lint`) with an opt-in fix engine.
Strict separation: detectors only *find* issues; the fix engine only fixes
issues explicitly marked safe, writes a JSON journal of every change to
``<output_base>/lint_fixes/``, and `undo_fixes()` restores the exact prior
state (captions, statuses, deleted rows, counts).

Issue codes
-----------
AUTO-FIXABLE (DB-only, journaled, reversible):
  caption_contradiction   'solo' with 2+ faces / 'no_humans' with a face ->
                          contradicted tag removed (old caption snapshotted
                          to caption_history AND the journal)
  caption_duplicate_tags  the same tag repeated in one caption -> deduped
  orphan_rows             captions/detections/dataset_frames rows whose
                          frame no longer exists -> rows backed up to the
                          journal, then removed
  missing_file            selected/packaged frame whose image file is gone
                          -> status flipped to 'file_missing' (excluded
                          from builds; original status journaled)
  dataset_count_drift     datasets.image_count disagrees with actual
                          dataset_frames train rows -> count recomputed

MANUAL REVIEW (never auto-fixed):
  missing_caption         selected frames without captions (run `caption`)
  low_identity_outlier    kept frames far below the identity threshold

Nothing here touches image files. Detection works on a read-only manifest;
fixes require a writable one.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .util import now_iso

log = logging.getLogger("pixel_curator.lint")

FIXABLE_CODES = ("caption_contradiction", "caption_duplicate_tags",
                 "orphan_rows", "missing_file", "dataset_count_drift")
MANUAL_CODES = ("missing_caption", "low_identity_outlier")

_CHILD_TABLES = ("captions", "detections", "dataset_frames")


@dataclass
class LintIssue:
    code: str
    subject: str                       # frame_id / table / dataset version
    detail: str
    fixable: bool = False
    fix_description: str = ""
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"code": self.code, "subject": self.subject,
                "detail": self.detail, "fixable": self.fixable,
                "fix_description": self.fix_description}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _q(conn, sql: str, params=()) -> list:
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []


def _split_tags(caption: str) -> list[str]:
    return [t.strip() for t in str(caption or "").split(",") if t.strip()]


def _norm(tag: str) -> str:
    return tag.strip().lower().replace("_", " ")


def _remove_tags(caption: str, drop: set[str]) -> str:
    kept = [t for t in _split_tags(caption) if _norm(t) not in drop]
    return ", ".join(kept)


def _dedupe_tags(caption: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for t in _split_tags(caption):
        k = _norm(t)
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return ", ".join(out)


# ---------------------------------------------------------------------------
# Detectors (read-only)
# ---------------------------------------------------------------------------

def detect_caption_contradictions(conn, limit: int = 500) -> list[LintIssue]:
    out: list[LintIssue] = []
    for fid, cap, faces in _q(
            conn,
            "SELECT f.frame_id, c.caption_text, d.face_count FROM frames f "
            "JOIN captions c ON c.frame_id=f.frame_id "
            "JOIN detections d ON d.frame_id=f.frame_id "
            "WHERE f.status IN ('selected','packaged')"):
        cap = cap or ""
        tags = {_norm(t) for t in _split_tags(cap)}
        drop: set[str] = set()
        details = []
        if "solo" in tags and (faces or 0) >= 2:
            drop.add("solo")
            details.append("tagged solo but 2+ faces detected")
        if "no humans" in tags and (faces or 0) >= 1:
            drop.add("no humans")
            details.append("tagged no_humans but a face was detected")
        if drop:
            out.append(LintIssue(
                code="caption_contradiction", subject=fid,
                detail="; ".join(details), fixable=True,
                fix_description=f"remove contradicted tag(s): "
                                f"{', '.join(sorted(drop))}",
                payload={"drop": sorted(drop), "caption": cap}))
            if len(out) >= limit:
                break
    return out


def detect_caption_duplicate_tags(conn, limit: int = 500) -> list[LintIssue]:
    out: list[LintIssue] = []
    for fid, cap in _q(
            conn,
            "SELECT f.frame_id, c.caption_text FROM frames f "
            "JOIN captions c ON c.frame_id=f.frame_id "
            "WHERE f.status IN ('selected','packaged') "
            "AND c.caption_text IS NOT NULL AND c.caption_text != ''"):
        deduped = _dedupe_tags(cap)
        if deduped != ", ".join(_split_tags(cap)):
            n = len(_split_tags(cap)) - len(_split_tags(deduped))
            out.append(LintIssue(
                code="caption_duplicate_tags", subject=fid,
                detail=f"{n} duplicated tag(s) in caption", fixable=True,
                fix_description="remove repeated tags (order preserved)",
                payload={"caption": cap, "deduped": deduped}))
            if len(out) >= limit:
                break
    return out


def detect_orphan_rows(conn) -> list[LintIssue]:
    out: list[LintIssue] = []
    for table in _CHILD_TABLES:
        rows = _q(conn, f"SELECT COUNT(*) FROM {table} t WHERE NOT EXISTS "
                        f"(SELECT 1 FROM frames f WHERE "
                        f"f.frame_id = t.frame_id)")
        n = rows[0][0] if rows else 0
        if n:
            out.append(LintIssue(
                code="orphan_rows", subject=table,
                detail=f"{n} row(s) reference frames that no longer exist",
                fixable=True,
                fix_description="back up the rows to the journal, then "
                                "remove them",
                payload={"table": table, "count": n}))
    return out


def detect_missing_files(conn, path_exists: Optional[Callable] = None,
                         limit: int = 2000) -> list[LintIssue]:
    exists = path_exists or (lambda p: Path(p).exists())
    out: list[LintIssue] = []
    for fid, path, status in _q(
            conn, "SELECT frame_id, path, status FROM frames "
                  "WHERE status IN ('selected','packaged')"):
        try:
            ok = exists(path)
        except Exception:
            ok = True                       # unreadable FS: don't flag
        if not ok:
            out.append(LintIssue(
                code="missing_file", subject=fid,
                detail=f"image file not found: {path}", fixable=True,
                fix_description="set status='file_missing' so builds skip "
                                "it (original status journaled)",
                payload={"path": path, "status": status}))
            if len(out) >= limit:
                break
    return out


def detect_dataset_count_drift(conn) -> list[LintIssue]:
    out: list[LintIssue] = []
    for version, recorded in _q(
            conn, "SELECT version, image_count FROM datasets"):
        rows = _q(conn, "SELECT COUNT(*) FROM dataset_frames "
                        "WHERE version = ? AND split = 'train'", (version,))
        actual = rows[0][0] if rows else 0
        if actual and int(recorded or 0) != actual:
            out.append(LintIssue(
                code="dataset_count_drift", subject=f"v{version:03d}",
                detail=f"image_count={recorded} but {actual} train rows",
                fixable=True,
                fix_description=f"recompute image_count -> {actual}",
                payload={"version": version, "recorded": recorded,
                         "actual": actual}))
    return out


def detect_missing_captions(conn, limit: int = 2000) -> list[LintIssue]:
    rows = _q(conn,
              "SELECT f.frame_id FROM frames f LEFT JOIN captions c "
              "ON c.frame_id = f.frame_id "
              "WHERE f.status IN ('selected','packaged') "
              "AND (c.caption_text IS NULL OR c.caption_text = '') "
              "LIMIT ?", (limit,))
    if not rows:
        return []
    return [LintIssue(
        code="missing_caption", subject=f"{len(rows)} frame(s)",
        detail=f"{len(rows)} selected frame(s) have no caption",
        fixable=False,
        fix_description="run `pixel-curator caption` (auto-fix would "
                        "invent captions - manual step by design)")]


def detect_low_identity_outliers(conn, threshold: float = 0.30,
                                 limit: int = 50) -> list[LintIssue]:
    rows = _q(conn,
              "SELECT f.frame_id, d.identity_sim FROM frames f "
              "JOIN detections d ON d.frame_id = f.frame_id "
              "WHERE f.status IN ('selected','packaged') "
              "AND d.identity_sim IS NOT NULL AND d.identity_sim < ? "
              "ORDER BY d.identity_sim LIMIT ?", (threshold, limit))
    return [LintIssue(
        code="low_identity_outlier", subject=r[0],
        detail=f"kept frame with identity similarity {r[1]:.2f} "
               f"(< {threshold})", fixable=False,
        fix_description="review manually - identity decisions are never "
                        "auto-fixed") for r in rows]


def run_lint(conn, path_exists: Optional[Callable] = None) -> list[LintIssue]:
    """All detectors; each degrades independently on older schemas."""
    issues: list[LintIssue] = []
    for det in (detect_caption_contradictions, detect_caption_duplicate_tags,
                detect_orphan_rows,
                lambda c: detect_missing_files(c, path_exists),
                detect_dataset_count_drift, detect_missing_captions,
                detect_low_identity_outliers):
        try:
            issues.extend(det(conn))
        except Exception as exc:                    # pragma: no cover
            log.warning("lint detector failed: %s", exc)
    return issues


# ---------------------------------------------------------------------------
# Fix engine (opt-in, journaled, reversible)
# ---------------------------------------------------------------------------

def _snapshot_caption(conn, frame_id: str, caption: str) -> None:
    """Preserve the pre-fix caption in caption_history (existing pattern)."""
    try:
        conn.execute("INSERT INTO caption_history(frame_id, caption_text, "
                     "saved_at) VALUES (?,?,?)",
                     (frame_id, caption, now_iso()))
    except sqlite3.OperationalError:
        pass                                        # pre-v13 manifest


def apply_fixes(conn, issues: list[LintIssue], output_base: Path | str,
                only: Optional[set[str]] = None) -> dict:
    """Apply the safe fixes for AUTO-FIXABLE issues. Returns a summary with
    the journal path. Every change is recorded with its prior state so
    `undo_fixes(journal)` restores it exactly."""
    entries: list[dict] = []
    counts: dict[str, int] = {}
    skipped = 0
    for issue in issues:
        if not issue.fixable:
            continue
        if only and issue.code not in only:
            skipped += 1
            continue
        try:
            entry = _apply_one(conn, issue)
        except Exception as exc:
            entry = None
            log.warning("fix failed for %s/%s: %s", issue.code,
                        issue.subject, exc)
        if entry:
            entries.append(entry)
            counts[issue.code] = counts.get(issue.code, 0) + 1
    conn.commit()

    journal_path = None
    if entries:
        jd = Path(output_base) / "lint_fixes"
        try:
            jd.mkdir(parents=True, exist_ok=True)
            journal_path = jd / (
                f"lint_fix_{datetime.now():%Y%m%d_%H%M%S}.json")
            journal_path.write_text(json.dumps({
                "tool": "lint", "version": 1, "created_at": now_iso(),
                "counts": counts, "entries": entries,
            }, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:                    # pragma: no cover
            log.warning("lint journal not written: %s", exc)
            journal_path = None
    return {"fixed": sum(counts.values()), "counts": counts,
            "skipped": skipped,
            "journal": str(journal_path) if journal_path else None}


def _current_caption(conn, frame_id: str) -> Optional[str]:
    row = conn.execute("SELECT caption_text FROM captions WHERE frame_id=?",
                       (frame_id,)).fetchone()
    return row[0] if row else None


def _apply_one(conn, issue: LintIssue) -> Optional[dict]:
    p = issue.payload
    # Caption fixes re-read the CURRENT caption at apply time (not the
    # detection-time payload) so multiple fixers touching the same frame
    # compose instead of clobbering each other; both are idempotent.
    if issue.code == "caption_contradiction":
        cap = _current_caption(conn, issue.subject)
        if cap is None:
            return None
        new_cap = _remove_tags(cap, set(p["drop"]))
        if new_cap == cap:
            return None                             # already fixed
        _snapshot_caption(conn, issue.subject, cap)
        conn.execute("UPDATE captions SET caption_text=?, updated_at=? "
                     "WHERE frame_id=?",
                     (new_cap, now_iso(), issue.subject))
        return {"code": issue.code, "frame_id": issue.subject,
                "before": cap, "after": new_cap}

    if issue.code == "caption_duplicate_tags":
        cap = _current_caption(conn, issue.subject)
        if cap is None:
            return None
        deduped = _dedupe_tags(cap)
        if deduped == ", ".join(_split_tags(cap)):
            return None                             # already fixed
        _snapshot_caption(conn, issue.subject, cap)
        conn.execute("UPDATE captions SET caption_text=?, updated_at=? "
                     "WHERE frame_id=?",
                     (deduped, now_iso(), issue.subject))
        return {"code": issue.code, "frame_id": issue.subject,
                "before": cap, "after": deduped}

    if issue.code == "orphan_rows":
        table = p["table"]
        if table not in _CHILD_TABLES:              # defense in depth
            return None
        cur = conn.execute(
            f"SELECT * FROM {table} t WHERE NOT EXISTS "
            f"(SELECT 1 FROM frames f WHERE f.frame_id = t.frame_id)")
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        conn.execute(
            f"DELETE FROM {table} WHERE frame_id IN (SELECT t.frame_id "
            f"FROM {table} t WHERE NOT EXISTS (SELECT 1 FROM frames f "
            f"WHERE f.frame_id = t.frame_id))")
        return {"code": issue.code, "table": table, "columns": cols,
                "removed_rows": rows}

    if issue.code == "missing_file":
        conn.execute("UPDATE frames SET status='file_missing' "
                     "WHERE frame_id=?", (issue.subject,))
        return {"code": issue.code, "frame_id": issue.subject,
                "before": p["status"], "after": "file_missing"}

    if issue.code == "dataset_count_drift":
        conn.execute("UPDATE datasets SET image_count=? WHERE version=?",
                     (p["actual"], p["version"]))
        return {"code": issue.code, "version": p["version"],
                "before": p["recorded"], "after": p["actual"]}
    return None


def undo_fixes(conn, journal_path: Path | str) -> dict:
    """Exactly reverse a fix journal: captions, statuses, deleted rows and
    counts are restored to their recorded prior state."""
    data = json.loads(Path(journal_path).read_text(encoding="utf-8"))
    restored = 0
    errors: list[str] = []
    # reverse order: composed fixes (e.g. two caption fixes on one frame)
    # unwind newest-first so the oldest 'before' state wins
    for e in reversed(data.get("entries", [])):
        try:
            code = e.get("code")
            if code in ("caption_contradiction", "caption_duplicate_tags"):
                conn.execute("UPDATE captions SET caption_text=?, "
                             "updated_at=? WHERE frame_id=?",
                             (e["before"], now_iso(), e["frame_id"]))
            elif code == "missing_file":
                conn.execute("UPDATE frames SET status=? WHERE frame_id=?",
                             (e["before"], e["frame_id"]))
            elif code == "orphan_rows":
                table = e["table"]
                if table not in _CHILD_TABLES:
                    raise ValueError(f"unknown table {table}")
                cols = e["columns"]
                marks = ",".join("?" * len(cols))
                conn.executemany(
                    f"INSERT OR REPLACE INTO {table}({','.join(cols)}) "
                    f"VALUES ({marks})",
                    [[r.get(c) for c in cols] for r in e["removed_rows"]])
            elif code == "dataset_count_drift":
                conn.execute("UPDATE datasets SET image_count=? "
                             "WHERE version=?", (e["before"], e["version"]))
            else:
                continue
            restored += 1
        except Exception as exc:
            errors.append(f"{e.get('code')}: {exc}")
    conn.commit()
    return {"restored": restored, "errors": errors,
            "from": str(journal_path)}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_report(issues: list[LintIssue], fix_mode: bool = False) -> str:
    if not issues:
        return ("Lint clean - no issues found.\n"
                "Next: pixel-curator suggest-preset --dataset <N> for "
                "training recommendations.")
    fixable = [i for i in issues if i.fixable]
    manual = [i for i in issues if not i.fixable]
    lines: list[str] = [f"Lint: {len(issues)} issue(s) - "
                        f"{len(fixable)} auto-fixable, "
                        f"{len(manual)} need manual review", ""]
    if fixable:
        lines.append("AUTO-FIXABLE" + ("" if fix_mode
                                       else "  (run with --fix to apply)"))
        for i in fixable[:40]:
            lines.append(f"  [{i.code}] {i.subject}: {i.detail}")
            lines.append(f"      fix: {i.fix_description}")
        if len(fixable) > 40:
            lines.append(f"  ... and {len(fixable) - 40} more")
        lines.append("")
    if manual:
        lines.append("MANUAL REVIEW  (never auto-fixed)")
        for i in manual[:40]:
            lines.append(f"  [{i.code}] {i.subject}: {i.detail}")
            lines.append(f"      next: {i.fix_description}")
        if len(manual) > 40:
            lines.append(f"  ... and {len(manual) - 40} more")
    return "\n".join(lines)


__all__ = [
    "LintIssue", "FIXABLE_CODES", "MANUAL_CODES",
    "run_lint", "apply_fixes", "undo_fixes", "render_report",
    "detect_caption_contradictions", "detect_caption_duplicate_tags",
    "detect_orphan_rows", "detect_missing_files",
    "detect_dataset_count_drift", "detect_missing_captions",
    "detect_low_identity_outliers",
]
