"""Dataset / manifest integrity validator ("validate" command).

A read-only "doctor for your data": it inspects the manifest and dataset
builds for the silent inconsistencies that don't raise tracebacks but quietly
corrupt training - the class of bug where a report reads one column while
training reads another.

Every check is defensive: a single failing check is captured and reported as a
FAIL row, never raised, so one bad table can't hide the rest of the report.
Nothing here writes, moves, or deletes - it only reads and reports.

Checks
------
  db_integrity        sqlite PRAGMA integrity_check / foreign_key sanity
  caption_drift       caption_text vs tags_json agreement; empty/!edited rows
  orphans             captions / detections / dataset_frames pointing at
                      frame_ids that no longer exist in frames
  missing_files       selected/packaged frames whose image file is gone
  dataset_manifest    datasets.image_count vs actual dataset_frames rows
  broken_symlinks     dead symlinks inside built dataset directories
  run_refs            runs pointing at a dataset_version that doesn't exist

Public API
----------
  run_validation(conn, *, base=None, path_exists=None, sample=8) -> dict
  render_report(report) -> str
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Optional

# status constants
OK = "ok"
WARN = "warn"
FAIL = "fail"

_RANK = {OK: 0, WARN: 1, FAIL: 2}


def _quality_metric_version() -> Optional[int]:
    try:
        from .curate.quality_signals import METRIC_VERSION
        return int(METRIC_VERSION)
    except Exception:
        return None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,)).fetchone()
    return row is not None


def _check(name: str, status: str, summary: str,
           count: int = 0, items: Optional[list] = None) -> dict:
    return {"name": name, "status": status, "summary": summary,
            "count": int(count), "items": list(items or [])}


def _guard(fn: Callable[[], dict], name: str) -> dict:
    """Run a check, converting any exception into a FAIL row."""
    try:
        return fn()
    except Exception as exc:                       # never let one check abort
        return _check(name, FAIL, f"check raised: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_db_integrity(conn: sqlite3.Connection) -> dict:
    def run() -> dict:
        rows = conn.execute("PRAGMA integrity_check").fetchall()
        msgs = [r[0] for r in rows]
        if msgs == ["ok"]:
            return _check("db_integrity", OK, "sqlite integrity_check: ok")
        return _check("db_integrity", FAIL,
                      "sqlite reports corruption", len(msgs), msgs[:8])
    return _guard(run, "db_integrity")


def check_caption_drift(conn: sqlite3.Connection) -> dict:
    """caption_text is what trains; tags_json is the raw record. Flag rows
    where they disagree in ways that bite: tagged but no usable caption."""
    def run() -> dict:
        if not _table_exists(conn, "captions"):
            return _check("caption_drift", OK, "no captions table yet")
        # tags present but caption_text empty -> nothing will train for that frame
        empty = conn.execute(
            "SELECT frame_id FROM captions "
            "WHERE tags_json IS NOT NULL AND TRIM(tags_json) NOT IN ('','[]') "
            "AND (caption_text IS NULL OR TRIM(caption_text)='')"
        ).fetchall()
        # caption_text present but no tags_json (hand-written / orphaned record)
        notags = conn.execute(
            "SELECT COUNT(*) FROM captions "
            "WHERE caption_text IS NOT NULL AND TRIM(caption_text)<>'' "
            "AND (tags_json IS NULL OR TRIM(tags_json)='')"
        ).fetchone()[0]
        n_empty = len(empty)
        if n_empty:
            return _check("caption_drift", WARN,
                          f"{n_empty} frames tagged but caption_text is empty "
                          f"(won't train); {notags} caption-only rows",
                          n_empty, [r[0] for r in empty[:8]])
        return _check("caption_drift", OK,
                      f"caption_text/tags_json consistent ({notags} "
                      "caption-only rows, informational)")
    return _guard(run, "caption_drift")


def check_orphans(conn: sqlite3.Connection) -> dict:
    """Child rows referencing a frame_id that is gone from frames."""
    def run() -> dict:
        if not _table_exists(conn, "frames"):
            return _check("orphans", OK, "no frames table yet")
        total = 0
        detail = []
        for tbl in ("captions", "detections", "clip_embeds", "study_labels",
                    "dataset_frames"):
            if not _table_exists(conn, tbl):
                continue
            n = conn.execute(
                f"SELECT COUNT(*) FROM {tbl} t "
                "WHERE NOT EXISTS (SELECT 1 FROM frames f "
                "WHERE f.frame_id = t.frame_id)").fetchone()[0]
            if n:
                total += n
                detail.append(f"{tbl}:{n}")
        if total:
            return _check("orphans", WARN,
                          f"{total} child rows reference missing frames "
                          f"({', '.join(detail)})", total, detail)
        return _check("orphans", OK, "no orphaned child rows")
    return _guard(run, "orphans")


def check_missing_files(conn: sqlite3.Connection,
                        path_exists: Callable[[str], bool],
                        sample: int = 8) -> dict:
    """selected/packaged frames whose image file no longer exists on disk."""
    def run() -> dict:
        if not _table_exists(conn, "frames"):
            return _check("missing_files", OK, "no frames table yet")
        rows = conn.execute(
            "SELECT frame_id, path FROM frames "
            "WHERE status IN ('selected','packaged')").fetchall()
        missing = [fid for fid, p in rows if not path_exists(p)]
        checked = len(rows)
        if missing:
            return _check("missing_files", WARN,
                          f"{len(missing)}/{checked} selected/packaged frames "
                          "have no file on disk", len(missing), missing[:sample])
        return _check("missing_files", OK,
                      f"all {checked} selected/packaged frame files present")
    return _guard(run, "missing_files")


def check_dataset_manifest(conn: sqlite3.Connection) -> dict:
    """datasets.image_count should match the train rows in dataset_frames."""
    def run() -> dict:
        if not (_table_exists(conn, "datasets")
                and _table_exists(conn, "dataset_frames")):
            return _check("dataset_manifest", OK, "no dataset builds yet")
        rows = conn.execute(
            "SELECT version, recipe_name, image_count FROM datasets "
            "ORDER BY version").fetchall()
        problems = []
        for ver, recipe, declared in rows:
            actual = conn.execute(
                "SELECT COUNT(*) FROM dataset_frames "
                "WHERE version=? AND split='train'", (ver,)).fetchone()[0]
            if declared is not None and int(declared) != actual:
                problems.append(
                    f"v{ver} '{recipe}': declared {declared} != {actual} rows")
        if problems:
            return _check("dataset_manifest", WARN,
                          f"{len(problems)} dataset(s) disagree with their "
                          "frame rows", len(problems), problems[:8])
        return _check("dataset_manifest", OK,
                      f"{len(rows)} dataset build(s) match their frame rows")
    return _guard(run, "dataset_manifest")


def check_broken_symlinks(conn: sqlite3.Connection, sample: int = 8) -> dict:
    """Dead symlinks inside built dataset directories (read-only walk)."""
    def run() -> dict:
        if not _table_exists(conn, "datasets"):
            return _check("broken_symlinks", OK, "no dataset builds yet")
        dirs = [r[0] for r in conn.execute(
            "SELECT dir FROM datasets WHERE dir IS NOT NULL").fetchall()]
        broken = []
        scanned = 0
        for d in dirs:
            p = Path(d).expanduser()
            if not p.exists():
                continue
            scanned += 1
            for f in p.rglob("*"):
                if f.is_symlink() and not f.exists():
                    broken.append(str(f))
                    if len(broken) >= 200:
                        break
        if broken:
            return _check("broken_symlinks", WARN,
                          f"{len(broken)} dead symlinks across {scanned} "
                          "dataset dir(s)", len(broken), broken[:sample])
        return _check("broken_symlinks", OK,
                      f"no dead symlinks ({scanned} dataset dir(s) scanned)")
    return _guard(run, "broken_symlinks")


def check_quality_coverage(conn: sqlite3.Connection,
                           metric_version: Optional[int] = None) -> dict:
    """Warn if the Priority-2 quality pass hasn't been run (or is stale) on the
    frames you'd export. Quality-aware selection silently degrades to
    sharpness-only without it, so this nudges you to run `quality` first."""
    def run() -> dict:
        if not _table_exists(conn, "frames"):
            return _check("quality_coverage", OK, "no frames yet")
        selected = conn.execute(
            "SELECT COUNT(*) FROM frames WHERE status IN ('selected','packaged')"
        ).fetchone()[0]
        if not selected:
            return _check("quality_coverage", OK, "no selected frames yet")
        if not _table_exists(conn, "frame_quality"):
            return _check("quality_coverage", WARN,
                          f"{selected} selected frames but no quality scores - "
                          "run `quality` to enable integrity-aware selection",
                          selected)
        scored = conn.execute(
            "SELECT COUNT(*) FROM frames f JOIN frame_quality q "
            "ON q.frame_id = f.frame_id "
            "WHERE f.status IN ('selected','packaged')").fetchone()[0]
        missing = selected - scored
        # stale: scored at an older metric_version
        stale = 0
        if metric_version is not None:
            stale = conn.execute(
                "SELECT COUNT(*) FROM frames f JOIN frame_quality q "
                "ON q.frame_id = f.frame_id WHERE f.status IN "
                "('selected','packaged') AND COALESCE(q.metric_version,0) < ?",
                (metric_version,)).fetchone()[0]
        if missing > 0:
            return _check("quality_coverage", WARN,
                          f"{missing}/{selected} selected frames have NO quality "
                          "score - run `quality` before export for integrity-aware "
                          "selection", missing)
        if stale > 0:
            return _check("quality_coverage", WARN,
                          f"{stale}/{selected} frames scored at an older "
                          f"metric_version (< {metric_version}) - re-run `quality`",
                          stale)
        return _check("quality_coverage", OK,
                      f"all {selected} selected frames have current quality scores")
    return _guard(run, "quality_coverage")


def check_advanced_quality(conn: sqlite3.Connection,
                           low_texture: float = 0.25) -> dict:
    """Warnings-only diagnostics over computed advanced-quality signals (never
    blocks export). No-op/OK when the optional signals were never run."""
    def run() -> dict:
        if not _table_exists(conn, "frame_advanced_quality"):
            return _check("advanced_quality", OK,
                          "advanced signals not computed (optional)")
        rows = conn.execute(
            "SELECT signals_json FROM frame_advanced_quality").fetchall()
        if not rows:
            return _check("advanced_quality", OK, "no advanced signals yet (optional)")
        import json as _json
        low_tex = missing_cov = 0
        low = {"aesthetic_score": 0, "skin_rendering_score": 0,
               "expression_score": 0, "body_symmetry_score": 0,
               "lighting_score": 0, "body_proportion_score": 0,
               "background_quality_score": 0, "full_body_score": 0,
               "eye_openness_score": 0}
        LOW_T = 0.30
        # batch-3/4 directional / confidence diagnostics (warnings only)
        HIGH_T = 0.60
        b3 = {"motion_blur_high": 0, "face_occlusion_high": 0,
              "color_mood_low_confidence": 0, "color_cast_high": 0,
              "clothing_type_unknown": 0, "duplicate_detection_positive": 0,
              "exposure_outlier_high": 0}
        # priority-6 region readiness (per-region thresholds, warnings only)
        REGION_T = {"face_region_score": 0.60, "torso_region_score": 0.55,
                    "full_body_region_score": 0.50, "region_balance_score": 0.45}
        REGION_CODE = {"face_region_score": "face_region_quality_low",
                       "torso_region_score": "torso_region_quality_low",
                       "full_body_region_score": "full_body_region_quality_low",
                       "region_balance_score": "region_balance_low"}
        region = {k: 0 for k in REGION_T}
        region_partial = 0
        # priority-7 coverage / framing intensity (dataset-level ratios)
        from collections import Counter as _Counter
        cov_signal = 0
        cov_unknown = framing_unknown = cov_needs_review = 0
        cov_missing = framing_missing = 0
        framing_counter: _Counter = _Counter()
        pose_sig_present = pose_sig_partial = 0
        for (sj,) in rows:
            try:
                s = _json.loads(sj) if sj else {}
            except Exception:
                continue
            tex = s.get("texture_preservation_score")
            if tex is not None and tex < low_texture:
                low_tex += 1
            if "coverage" in s and not s.get("coverage_label"):
                missing_cov += 1
            for key in low:
                v = s.get(key)
                if v is not None and v < LOW_T:
                    low[key] += 1
            mb = s.get("motion_blur_score")
            if mb is not None and mb > HIGH_T:
                b3["motion_blur_high"] += 1
            fo = s.get("face_occlusion_score")
            if fo is not None and fo > HIGH_T:
                b3["face_occlusion_high"] += 1
            if "color_mood" in s:
                conf = (s.get("color_mood") or {}).get("confidence")
                if conf is not None and conf < LOW_T:
                    b3["color_mood_low_confidence"] += 1
            cc = s.get("color_cast_score")
            if cc is not None and cc > HIGH_T:
                b3["color_cast_high"] += 1
            if "clothing_type" in s:
                lbl = s.get("clothing_type_label") or \
                    (s.get("clothing_type") or {}).get("clothing_type")
                if lbl == "unknown":
                    b3["clothing_type_unknown"] += 1
            if (s.get("duplicate_detection") or {}).get("is_duplicate") is True:
                b3["duplicate_detection_positive"] += 1
            exo = s.get("exposure_outlier") or {}
            if exo.get("is_overexposed") is True or exo.get("is_underexposed") is True:
                b3["exposure_outlier_high"] += 1
            for key, thr in REGION_T.items():
                v = s.get(key)
                if v is not None and v < thr:
                    region[key] += 1
            rr = s.get("region_readiness")
            if isinstance(rr, dict) and any(
                    "partial" in str(w) for w in (rr.get("warnings") or [])):
                region_partial += 1
            # coverage / framing intensity (only on rows that carry the signal)
            if "coverage_intensity" in s or "framing_label" in s:
                cov_signal += 1
                cl = s.get("coverage_label")
                fl = s.get("framing_label")
                if cl is None:
                    cov_missing += 1
                elif cl == "unknown":
                    cov_unknown += 1
                elif cl == "needs_review":
                    cov_needs_review += 1
                if fl is None:
                    framing_missing += 1
                elif fl == "unknown":
                    framing_unknown += 1
                elif fl not in ("needs_review",):
                    framing_counter[fl] += 1
                if s.get("coverage_review_recommended") is True:
                    cov_needs_review += 1
            # pose signature presence (priority 8)
            if "pose_signature" in s or "pose_signature_id" in s:
                pose_sig_present += 1
                psig = s.get("pose_signature")
                if isinstance(psig, dict) and any(
                        "partial" in str(w) for w in (psig.get("warnings") or [])):
                    pose_sig_partial += 1
        notes = []
        if low_tex:
            notes.append(f"{low_tex} frames texture_preservation < {low_texture} "
                         "(possible over-smoothing)")
        if missing_cov:
            notes.append(f"{missing_cov} frames missing coverage_label")
        for key, n in low.items():
            if n:
                notes.append(f"{n} frames {key} < {LOW_T}")
        if b3["motion_blur_high"]:
            notes.append(f"{b3['motion_blur_high']} frames motion_blur_score > {HIGH_T}")
        if b3["face_occlusion_high"]:
            notes.append(f"{b3['face_occlusion_high']} frames face_occlusion_score > {HIGH_T}")
        if b3["color_mood_low_confidence"]:
            notes.append(f"{b3['color_mood_low_confidence']} frames color_mood confidence < {LOW_T}")
        if b3["color_cast_high"]:
            notes.append(f"{b3['color_cast_high']} frames color_cast_score > {HIGH_T}")
        if b3["clothing_type_unknown"]:
            notes.append(f"{b3['clothing_type_unknown']} frames clothing_type unknown")
        if b3["duplicate_detection_positive"]:
            notes.append(f"{b3['duplicate_detection_positive']} frames flagged as duplicate (awareness only)")
        if b3["exposure_outlier_high"]:
            notes.append(f"{b3['exposure_outlier_high']} frames exposure outlier (over/under-exposed)")
        for key, n in region.items():
            if n:
                notes.append(f"{n} frames {REGION_CODE[key]} (< {REGION_T[key]})")
        if region_partial:
            notes.append(f"{region_partial} frames region_quality_partial "
                         "(global signal fallback)")
        # priority-7 coverage / framing intensity warnings (ratios over rows
        # that actually carry the coverage signal; never blocks export)
        if cov_signal:
            if cov_missing:
                notes.append(f"{cov_missing} frames coverage_label_missing")
            if framing_missing:
                notes.append(f"{framing_missing} frames framing_label_missing")
            if cov_unknown / cov_signal > 0.25:
                notes.append(f"coverage_unknown_high: {cov_unknown}/{cov_signal} "
                             "frames have unknown coverage (> 25%)")
            if framing_unknown / cov_signal > 0.25:
                notes.append(f"framing_unknown_high: {framing_unknown}/{cov_signal}"
                             " frames have unknown framing (> 25%)")
            if cov_needs_review / cov_signal > 0.20:
                notes.append(f"coverage_needs_review_high: {cov_needs_review}/"
                             f"{cov_signal} frames need review (> 20%)")
            if framing_counter:
                _dom, _c = framing_counter.most_common(1)[0]
                _framed = sum(framing_counter.values())
                if _framed and _c / _framed > 0.60:
                    notes.append(f"framing_dominance_high: '{_dom}' is "
                                 f"{_c}/{_framed} framed samples (> 60%)")
                if len(framing_counter) <= 1 and _framed >= 4:
                    notes.append("coverage_distribution_unbalanced: only one "
                                 "framing category present")
        if pose_sig_present and pose_sig_partial / pose_sig_present > 0.50:
            notes.append(f"pose_duplicate_detection_partial: {pose_sig_partial}/"
                         f"{pose_sig_present} pose signatures lack keypoints/"
                         "body angle")
        if notes:
            return _check("advanced_quality", WARN, "; ".join(notes),
                          low_tex + missing_cov, notes)
        return _check("advanced_quality", OK,
                      f"{len(rows)} frames with advanced signals, no issues")
    return _guard(run, "advanced_quality")


def check_run_refs(conn: sqlite3.Connection) -> dict:
    """Training runs pointing at a dataset_version that no longer exists."""
    def run() -> dict:
        if not (_table_exists(conn, "runs") and _table_exists(conn, "datasets")):
            return _check("run_refs", OK, "no training runs yet")
        bad = conn.execute(
            "SELECT run_id, name, dataset_version FROM runs r "
            "WHERE r.dataset_version IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM datasets d WHERE d.version = r.dataset_version)"
        ).fetchall()
        if bad:
            items = [f"run {rid} '{nm}' -> missing dataset v{dv}"
                     for rid, nm, dv in bad]
            return _check("run_refs", WARN,
                          f"{len(bad)} run(s) reference a missing dataset "
                          "version", len(bad), items[:8])
        return _check("run_refs", OK, "all runs reference existing datasets")
    return _guard(run, "run_refs")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_validation(conn: sqlite3.Connection, *,
                   path_exists: Optional[Callable[[str], bool]] = None,
                   sample: int = 8) -> dict:
    """Run every check and return a structured report. Read-only."""
    if path_exists is None:
        def path_exists(p: str) -> bool:
            try:
                return Path(p).exists()
            except Exception:
                return False

    checks = [
        check_db_integrity(conn),
        check_caption_drift(conn),
        check_orphans(conn),
        check_missing_files(conn, path_exists, sample),
        check_dataset_manifest(conn),
        check_broken_symlinks(conn, sample),
        check_run_refs(conn),
        check_quality_coverage(conn, _quality_metric_version()),
        check_advanced_quality(conn),
    ]
    counts = {OK: 0, WARN: 0, FAIL: 0}
    for c in checks:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    worst = max((c["status"] for c in checks), key=lambda s: _RANK[s],
                default=OK)
    return {"checks": checks, "summary": counts, "overall": worst,
            "healthy": worst == OK}


def render_report(report: dict) -> str:
    glyph = {OK: "OK  ", WARN: "WARN", FAIL: "FAIL"}
    lines = ["", "Dataset / manifest validation", "=" * 60]
    for c in report["checks"]:
        lines.append(f"[{glyph[c['status']]}] {c['name']:<18} {c['summary']}")
        for it in c["items"][:8]:
            lines.append(f"           - {it}")
    s = report["summary"]
    lines.append("-" * 60)
    lines.append(f"overall: {report['overall'].upper()}  "
                 f"(ok={s.get(OK,0)} warn={s.get(WARN,0)} fail={s.get(FAIL,0)})")
    if report["healthy"]:
        lines.append("No integrity problems found.")
    else:
        lines.append("Review the WARN/FAIL rows above. This command is "
                     "read-only; it changes nothing.")
    return "\n".join(lines)
