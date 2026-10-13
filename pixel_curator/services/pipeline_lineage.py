"""Dataset Pipeline Explorer - image lineage (Phase X).

Reconstructs, for a single image, where it entered the pipeline, which stages it
is present in / missing from, which dataset versions include it, and (when it
dropped out after curation) the most likely removal reason. Read-only; depends
only on the standard library and ``dataset_explorer`` (no UI imports). Transparent
about uncertainty: an unknown reason is reported as "unknown" rather than guessed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import dataset_explorer as dx

logger = logging.getLogger(__name__)


@dataclass
class ImageLineage:
    image_id: str
    path: str
    stages_present: list = field(default_factory=list)
    stages_missing: list = field(default_factory=list)
    version_membership: list = field(default_factory=list)
    events: list = field(default_factory=list)
    likely_removal_reason: str | None = None
    quality_signals: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    file_exists: bool | None = None
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"image_id": self.image_id, "path": self.path,
                "stages_present": self.stages_present,
                "stages_missing": self.stages_missing,
                "version_membership": self.version_membership,
                "events": self.events,
                "likely_removal_reason": self.likely_removal_reason,
                "quality_signals": self.quality_signals, "flags": self.flags,
                "file_exists": self.file_exists, "warnings": self.warnings}


def _resolve_frame_id(conn, image_path_or_id: str) -> str | None:
    """Accept a frame_id or an absolute/partial path and return the frame_id."""
    s = str(image_path_or_id)
    try:
        row = conn.execute("SELECT frame_id FROM frames WHERE frame_id = ?",
                           (s,)).fetchone()
        if row:
            return row[0]
        row = conn.execute("SELECT frame_id FROM frames WHERE path = ?",
                           (s,)).fetchone()
        if row:
            return row[0]
        row = conn.execute(
            "SELECT frame_id FROM frames WHERE path LIKE ? ORDER BY frame_id LIMIT 1",
            (f"%{Path(s).name}",)).fetchone()
        if row:
            return row[0]
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return None
    return None


def _events_for(conn, frame_id: str) -> list:
    """Per-frame history from the events table (source_id-keyed), best effort."""
    out = []
    if not dx._table_exists(conn, "events"):
        return out
    try:
        src = conn.execute("SELECT source_id FROM frames WHERE frame_id = ?",
                           (frame_id,)).fetchone()
        sid = src[0] if src else None
        if sid is None:
            return out
        for level, message, ts in conn.execute(
                "SELECT level, message, ts FROM events WHERE source_id = ? "
                "ORDER BY ts LIMIT 50", (sid,)):
            out.append({"level": level, "message": message, "ts": ts})
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    return out


def get_image_lineage(output_base, image_path_or_id: str) -> ImageLineage:
    """Build the lineage record for one image. Never raises for ordinary input."""
    conn = dx.connect_ro(output_base)
    lin = ImageLineage(image_id=str(image_path_or_id), path="")
    if conn is None:
        lin.warnings.append("manifest unavailable")
        return lin
    try:
        fid = _resolve_frame_id(conn, image_path_or_id)
        if fid is None:
            lin.warnings.append("image not found in manifest")
            return lin
        # pull the single record through the explorer normalizer
        flt = dx.ExplorerFilter(limit=1)
        cluster_expr = "f.cluster_id" if dx._has_column(conn, "frames", "cluster_id") else "NULL"
        sql = dx._BASE_SELECT.format(cluster=cluster_expr) + " WHERE f.frame_id = ?"
        row = conn.execute(sql, (fid,)).fetchone()
        if row is None:
            lin.warnings.append("image row vanished")
            return lin
        vmap = dx._version_map(conn)
        rec = dx._row_to_record(row, vmap)
        lin.image_id = rec.image_id
        lin.path = rec.path
        lin.version_membership = rec.version_membership
        lin.quality_signals = rec.quality_signals
        lin.flags = rec.flags
        lin.stages_present = list(rec.stage_keys)
        # post_keyframes / package / plan stages aren't frame-derivable from the
        # manifest, so they are neither "present" nor reported as "missing".
        _underived = ("post_keyframes", "training_package_ready",
                      "training_plan_ready")
        lin.stages_missing = [k for k in dx.STAGE_KEYS
                              if k not in rec.stage_keys and k not in _underived]
        lin.events = _events_for(conn, fid)
        try:
            lin.file_exists = bool(rec.path) and Path(rec.path).exists()
        except Exception:
            lin.file_exists = None
        # likely removal reason: dropped out of curation after being analyzed
        status = rec.metadata.get("status")
        if "post_smart_curation" not in rec.stage_keys and "post_quality" in rec.stage_keys:
            reason = "unknown"
            for flag, label in dx._REMOVAL_RULES:
                if flag in rec.flags:
                    reason = label
                    break
            if status == "rejected" and reason == "unknown":
                reason = "manual or rule-based exclusion"
            lin.likely_removal_reason = reason
        return lin
    finally:
        conn.close()


__all__ = ["ImageLineage", "get_image_lineage"]
