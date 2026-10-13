"""General-purpose export: manifest dump + selection export.

Decouples curation results from any specific training framework. Two
targets:

* ``export_manifest``  - one flat record per frame (path, status, optical
  signals, identity/face signals, quality metrics, caption) as CSV, JSONL,
  or Parquet (when pandas/pyarrow is installed). Open it in a spreadsheet,
  filter, done.
* ``export_selection`` - materialize a chosen slice ("top 10%", "score >=
  0.8") into a flat folder as symlinks (zero-copy culling of huge
  libraries) or real copies, with optional caption sidecars.

Scoring: ``frame_quality.integrity`` (0..1) when present, otherwise the
percentile rank of ``sharpness`` across the exported set - a unified 0..1
``score`` column is always emitted so downstream filtering never has to
care which signal was available.

Stdlib-only by default; never raises for ordinary input problems.
"""
from __future__ import annotations

import csv
import json
import logging
import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path

log = logging.getLogger("pixel_curator.export.general")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
FORMATS = ("csv", "jsonl", "parquet")

# Frames with these statuses are excluded from selection exports.
EXCLUDED_STATUS_PREFIXES = ("drop", "quarantine", "reject")


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
def iter_frame_records(conn: sqlite3.Connection) -> Iterator[dict]:
    """Yield one flat dict per frame, joining every per-frame signal table
    that exists in this manifest (older DBs simply lack the newer tables)."""
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    joins, cols = [], []
    if "detections" in tables:
        joins.append("LEFT JOIN detections d ON d.frame_id = f.frame_id")
        cols += ["d.face_count", "d.face_area", "d.det_conf",
                 "d.identity_sim", "d.framing"]
    if "captions" in tables:
        joins.append("LEFT JOIN captions c ON c.frame_id = f.frame_id")
        cols += ["c.caption_text", "c.tags_json", "c.vlm_text"]
    if "frame_quality" in tables:
        joins.append("LEFT JOIN frame_quality q ON q.frame_id = f.frame_id")
        cols += ["q.integrity", "q.face_pass", "q.pose_bucket",
                 "q.face_sharpness", "q.centering", "q.occlusion",
                 "q.frontalness", "q.eyes_open"]
    if "frame_outfit" in tables:
        joins.append("LEFT JOIN frame_outfit o ON o.frame_id = f.frame_id")
        cols += ["o.dominant_colors", "o.garments"]
    fcols = {r[1] for r in conn.execute("PRAGMA table_info(frames)")}
    extra_f = ["f.cluster_id"] if "cluster_id" in fcols else []
    sql = ("SELECT f.frame_id, f.path, f.status, f.source_id, f.dhash, "
           "f.sharpness, f.brightness, f.scored_at"
           + (", " + ", ".join(extra_f + cols) if (cols or extra_f) else "")
           + " FROM frames f " + " ".join(joins))
    conn.row_factory = sqlite3.Row
    for row in conn.execute(sql):
        rec = dict(row)
        tags = rec.pop("tags_json", None)
        if tags:
            try:
                rec["tags"] = ", ".join(json.loads(tags)) \
                    if isinstance(json.loads(tags), list) else str(tags)
            except Exception:
                rec["tags"] = str(tags)
        yield rec


def _unified_scores(records: list[dict]) -> None:
    """Attach a 0..1 ``score`` to each record: integrity when available,
    else percentile rank of sharpness. Mutates records in place."""
    have_integrity = [r for r in records if r.get("integrity") is not None]
    sharp_vals = sorted(
        float(r.get("sharpness") or 0.0) for r in records)
    n = max(len(sharp_vals), 1)

    def pct_rank(v: float) -> float:
        # fraction of frames with sharpness <= v
        import bisect
        return bisect.bisect_right(sharp_vals, v) / n

    for r in records:
        if r.get("integrity") is not None:
            r["score"] = round(float(r["integrity"]), 6)
            r["score_source"] = "integrity"
        else:
            r["score"] = round(pct_rank(float(r.get("sharpness") or 0.0)), 6)
            r["score_source"] = "sharpness_percentile" if sharp_vals \
                else "none"


# ---------------------------------------------------------------------------
# manifest export
# ---------------------------------------------------------------------------
def export_manifest(conn: sqlite3.Connection, out_path: Path | str,
                    fmt: str = "csv") -> dict:
    """Write one record per frame to ``out_path``. Returns a report dict."""
    fmt = (fmt or "csv").strip().lower()
    if fmt not in FORMATS:
        return {"ok": False, "error": f"unknown format '{fmt}'; "
                f"choose from {', '.join(FORMATS)}"}
    records = list(iter_frame_records(conn))
    _unified_scores(records)
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        try:
            import pandas as pd
        except Exception:
            return {"ok": False, "error": "parquet export requires pandas "
                    "+ pyarrow (pip install pandas pyarrow); "
                    "use --format csv or jsonl instead"}
        pd.DataFrame(records).to_parquet(out, index=False)
    elif fmt == "jsonl":
        with open(out, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        fieldnames = sorted({k for r in records for k in r}) or ["empty"]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for rec in records:
                w.writerow(rec)
    return {"ok": True, "format": fmt, "path": str(out),
            "records": len(records)}


# ---------------------------------------------------------------------------
# selection export
# ---------------------------------------------------------------------------
def select_records(records: list[dict], top_pct: float | None = None,
                   min_score: float | None = None) -> list[dict]:
    """Filter records by score. ``top_pct`` (0-100) keeps the best slice by
    rank; ``min_score`` (0-1) keeps everything at or above the unified
    score. If both are given, a frame must satisfy both."""
    eligible = [r for r in records
                if not str(r.get("status", "")).lower().startswith(
                    EXCLUDED_STATUS_PREFIXES)]
    if min_score is not None:
        eligible = [r for r in eligible if r.get("score", 0.0) >= min_score]
    eligible.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    if top_pct is not None:
        keep = max(1, round(len(eligible) * min(max(top_pct, 0.0), 100.0)
                          / 100.0)) if eligible else 0
        eligible = eligible[:keep]
    return eligible


def select_records_balanced(records: list[dict], n: int,
                            top_pct: float | None = None,
                            min_score: float | None = None) -> list[dict]:
    """Evenly-spread selection across CLIP clusters: round-robin over
    ``cluster_id`` groups, best-scored frame first within each group.
    Frames without a cluster assignment fall back to the global score
    ranking. Requires ``cluster`` to have been run for full effect."""
    eligible = select_records(records, top_pct=top_pct, min_score=min_score)
    if n <= 0 or not eligible:
        return []
    groups: dict[object, list[dict]] = {}
    for r in eligible:
        groups.setdefault(r.get("cluster_id"), []).append(r)
    # biggest groups first so round-robin drains them fairly
    ordered = sorted(groups.values(), key=len, reverse=True)
    picked: list[dict] = []
    while len(picked) < n and any(ordered):
        for g in ordered:
            if g and len(picked) < n:
                picked.append(g.pop(0))
    return picked


def export_selection(conn: sqlite3.Connection, out_dir: Path | str,
                     top_pct: float | None = None,
                     min_score: float | None = None,
                     mode: str = "symlink",
                     with_captions: bool = True,
                     balanced: int = 0) -> dict:
    """Materialize the selected frames into ``out_dir`` as symlinks or
    copies, named sequentially, plus a ``selection_manifest.csv`` index.
    ``balanced`` > 0 spreads the pick evenly across CLIP clusters."""
    mode = (mode or "symlink").strip().lower()
    if mode not in ("symlink", "copy"):
        return {"ok": False, "error": "mode must be 'symlink' or 'copy'"}
    if top_pct is None and min_score is None and not balanced:
        return {"ok": False, "error": "nothing selected: pass --top-pct, "
                "--min-score and/or --balanced"}

    records = list(iter_frame_records(conn))
    _unified_scores(records)
    if balanced:
        chosen = select_records_balanced(records, balanced, top_pct,
                                         min_score)
    else:
        chosen = select_records(records, top_pct, min_score)

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    written, missing, errors = 0, 0, []
    index_rows = []
    for i, rec in enumerate(chosen, 1):
        src = Path(rec.get("path") or "")
        if not src.is_file():
            missing += 1
            continue
        name = f"{i:06d}{src.suffix.lower()}"
        dst = out / name
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if mode == "symlink":
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
            if with_captions and rec.get("caption_text"):
                dst.with_suffix(".txt").write_text(
                    str(rec["caption_text"]), encoding="utf-8")
            written += 1
            index_rows.append({"file": name, "source": str(src),
                               "score": rec.get("score"),
                               "score_source": rec.get("score_source"),
                               "status": rec.get("status"),
                               "frame_id": rec.get("frame_id")})
        except Exception as exc:
            errors.append(f"{src}: {exc}")

    index_path = out / "selection_manifest.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "source", "score",
                                           "score_source", "status",
                                           "frame_id"])
        w.writeheader()
        w.writerows(index_rows)

    return {"ok": not errors, "out_dir": str(out), "mode": mode,
            "selected": len(chosen), "written": written,
            "missing_source": missing, "index": str(index_path),
            "errors": errors[:20],
            "total_frames": len(records)}
