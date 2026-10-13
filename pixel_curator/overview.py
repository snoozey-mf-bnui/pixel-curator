"""Phase 9: ``pixel-curator overview`` - one-screen dataset + process status.

Read-only. Answers "where is this dataset and what should I run next?"
without needing to know four different commands: frame funnel, caption /
quality coverage, pending work with concrete next commands, registry
counts, and any paused pipeline checkpoint.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from . import manifest
from .config import DB_NAME, Project

logger = logging.getLogger(__name__)

ACTIVE = ("selected", "packaged")


def _one(conn, sql: str, params=()) -> int:
    try:
        row = conn.execute(sql, params).fetchone()
        return int(row[0] or 0)
    except sqlite3.Error:
        return 0


def overview_report(prj: Project, base: Path) -> dict:
    out: dict = {"output_base": str(base), "manifest": False}
    if not (base / DB_NAME).exists():
        return out
    conn = manifest.connect(base, readonly=True)
    out["manifest"] = True
    out["statuses"] = dict(conn.execute(
        "SELECT status, COUNT(*) FROM frames GROUP BY status"))
    total = sum(out["statuses"].values())
    out["total"] = total
    active = sum(out["statuses"].get(s, 0) for s in ACTIVE)
    out["active"] = active

    out["captioned"] = _one(
        conn, "SELECT COUNT(*) FROM frames f JOIN captions c "
              "ON c.frame_id = f.frame_id "
              "WHERE f.status IN ('selected','packaged')")
    out["quality_scored"] = _one(
        conn, "SELECT COUNT(*) FROM frames f JOIN frame_quality q "
              "ON q.frame_id = f.frame_id "
              "WHERE f.status IN ('selected','packaged')")

    out["dataset_versions"] = _one(conn, "SELECT COUNT(*) FROM datasets")
    row = conn.execute(
        "SELECT version, recipe_name, kind FROM datasets "
        "ORDER BY version DESC LIMIT 1").fetchone()
    out["latest_version"] = (
        {"version": row[0], "name": row[1], "kind": row[2]} if row else None)
    out["experiments"] = _one(conn, "SELECT COUNT(*) FROM experiments")
    out["runs"] = _one(conn, "SELECT COUNT(*) FROM runs")

    # paused pipelines (Phase 7 recipe engine takes priority over legacy)
    out["pipeline"] = None
    try:
        from .pipeline_recipes import load_recipe_state
        st = load_recipe_state(conn)
        if st and st.get("next_stage"):
            out["pipeline"] = {"engine": "recipe",
                               "pipeline": st.get("pipeline"),
                               "next_stage": st.get("next_stage"),
                               "error": st.get("error")}
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    if out["pipeline"] is None:
        try:
            from .pipeline_dag import load_state
            st = load_state(conn)
            if st:
                out["pipeline"] = {"engine": "legacy",
                                   "next_stage": st.get("next_stage")
                                   or st.get("stage")}
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass
    conn.close()

    # pending work -> concrete next commands
    pending: list[tuple[str, str]] = []
    s = out["statuses"]
    if s.get("new"):
        pending.append((f"{s['new']} frame(s) not yet curated",
                        "pixel-curator curate"))
    uncap = active - out["captioned"]
    if uncap > 0:
        pending.append((f"{uncap} active frame(s) without captions",
                        "pixel-curator caption"))
    unq = active - out["quality_scored"]
    if unq > 0:
        pending.append((f"{unq} active frame(s) without quality scores",
                        "pixel-curator quality --resume"))
    if out["pipeline"]:
        pending.append(
            (f"pipeline '{out['pipeline'].get('pipeline', 'full')}' paused "
             f"before '{out['pipeline']['next_stage']}'",
             "pixel-curator pipeline resume"))
    out["pending"] = pending
    return out


def render_overview(rep: dict) -> str:
    lines = ["", "DATASET OVERVIEW", "=" * 60,
             f"output: {rep['output_base']}"]
    if not rep.get("manifest"):
        lines.append("No manifest yet - start with: pixel-curator extract "
                     "(or photos)")
        return "\n".join(lines)
    s = rep["statuses"]
    lines.append(f"frames : {rep['total']} total | {rep['active']} active "
                 "(selected/packaged)")
    if s:
        top = sorted(s.items(), key=lambda kv: -kv[1])
        lines.append("status : " + ", ".join(f"{k}: {v}" for k, v in top))
    lines.append(f"coverage: captions {rep['captioned']}/{rep['active']} | "
                 f"quality {rep['quality_scored']}/{rep['active']}")
    lv = rep.get("latest_version")
    lines.append(
        f"registry: {rep['dataset_versions']} dataset version(s)"
        + (f" (latest v{lv['version']:03d} '{lv['name']}' [{lv['kind']}])"
           if lv else "")
        + f" | {rep['experiments']} experiment(s) | {rep['runs']} run(s)")
    lines.append("-" * 60)
    if rep["pending"]:
        lines.append("pending work:")
        for what, cmd in rep["pending"]:
            lines.append(f"  * {what}")
            lines.append(f"      -> {cmd}")
    else:
        lines.append("pending work: none - dataset is fully processed.")
    return "\n".join(lines)


__all__ = ["overview_report", "render_overview"]
