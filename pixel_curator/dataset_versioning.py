"""Phase 2: Dataset Versioning - lineage, curation snapshots, unified history.

Extends the existing versioned build registry (manifest v4 ``datasets`` /
``dataset_frames``) rather than replacing it:

* every version may record an explicit parent (``--parent N``) so the
  evolution of a dataset ("v5 was refined from v3") is queryable
* a curation snapshot (frame status counts, identity-anchor state,
  duplicate-scan coverage, caption/quality coverage, active curation
  performance settings) is stored with each version at creation time
* folder-based exports (dataset_splitter lora-target exports, kohya
  packages) register in the same registry with path-based membership
  (``dataset_version_files``), so ``datasets``, ``history`` and diffs
  see every dataset artifact, not just recipe builds
* every registered version gets a ``version.json`` sidecar inside its
  dataset folder - human-readable provenance that travels with the files

Additive and non-destructive: nothing here deletes or rewrites earlier
versions; registration failures degrade to structured warnings so export
workflows never break because of versioning.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from . import manifest
from .util import now_iso

log = logging.getLogger("pixel_curator.dataset_versioning")

SIDECAR_NAME = "version.json"

#: Registry kinds. NULL/'' in older rows means 'build'.
KIND_BUILD = "build"
KIND_LORA_TARGET_EXPORT = "lora_target_export"
KIND_KOHYA_PACKAGE = "kohya_package"
KIND_MULTI_RECIPE_PACKAGE = "multi_recipe_package"

#: meta-table keys maintained by the registry (cross-dataset queries)
META_LATEST_VERSION = "dataset_latest_version"
META_LATEST_INFO = "dataset_latest_info"
META_LATEST_BY_KIND = "dataset_latest_by_kind"


# ---------------------------------------------------------------------------
# Curation snapshot
# ---------------------------------------------------------------------------

def _distribution(conn: sqlite3.Connection, table: str, column: str,
                  where: str = "") -> Optional[dict]:
    """count/mean/min/median/max for a numeric column. None when empty
    or the table/column does not exist on this schema."""
    clause = (f"{column} IS NOT NULL"
              + (f" AND ({where})" if where else ""))
    try:
        n, mean, lo, hi = conn.execute(
            f"SELECT COUNT(*), AVG({column}), MIN({column}), MAX({column}) "
            f"FROM {table} WHERE {clause}").fetchone()
        if not n:
            return None
        median = conn.execute(
            f"SELECT {column} FROM {table} WHERE {clause} "
            f"ORDER BY {column} LIMIT 1 OFFSET ?", (int(n) // 2,)
        ).fetchone()[0]
        rnd = lambda v: round(float(v), 4) if v is not None else None
        return {"count": int(n), "mean": rnd(mean), "min": rnd(lo),
                "median": rnd(median), "max": rnd(hi)}
    except sqlite3.OperationalError:
        return None


def capture_curation_snapshot(conn: sqlite3.Connection) -> dict:
    """Snapshot the curation state that shaped the current eligible pool.

    Read-only; every section degrades independently on older schemas or
    partial pipelines (missing tables/columns simply omit that section).
    """
    snap: dict = {"captured_at": now_iso()}

    def _try(key: str, sql: str, shape=None) -> None:
        try:
            rows = conn.execute(sql).fetchall()
            snap[key] = shape(rows) if shape else rows
        except sqlite3.OperationalError:
            pass

    _try("frame_status_counts",
         "SELECT status, COUNT(*) FROM frames GROUP BY status",
         lambda rows: {r[0]: r[1] for r in rows})
    _try("caption_coverage",
         "SELECT COUNT(*), SUM(CASE WHEN edited THEN 1 ELSE 0 END) "
         "FROM captions",
         lambda rows: {"captioned": rows[0][0] or 0,
                       "manually_edited": rows[0][1] or 0})
    _try("duplicate_scan",
         "SELECT COUNT(DISTINCT dup_group_id), "
         "SUM(CASE WHEN dup_role='candidate' THEN 1 ELSE 0 END) "
         "FROM frames WHERE dup_group_id IS NOT NULL",
         lambda rows: {"groups": rows[0][0] or 0,
                       "candidates": rows[0][1] or 0})
    _try("quality_metrics",
         "SELECT COUNT(*), MAX(metric_version) FROM frame_quality",
         lambda rows: {"scored_frames": rows[0][0] or 0,
                       "metric_version": rows[0][1]})
    _try("study_labels",
         "SELECT COUNT(*), SUM(study_export_eligible) FROM study_labels",
         lambda rows: {"classified": rows[0][0] or 0,
                       "export_eligible": rows[0][1] or 0})

    # per-signal quality distributions of the current eligible pool
    dist = {
        "sharpness": _distribution(
            conn, "frames", "sharpness",
            "status IN ('selected','packaged')"),
        "identity_sim": _distribution(conn, "detections", "identity_sim"),
        "integrity": _distribution(conn, "frame_quality", "integrity"),
    }
    dist = {k: v for k, v in dist.items() if v}
    if dist:
        snap["quality_distribution"] = dist

    # which curation passes have run (latest activity timestamps)
    passes = {}
    for key, sql in (
            ("quality_scored_at", "SELECT MAX(scored_at) FROM frames"),
            ("deep_quality_at",
             "SELECT MAX(calculated_at) FROM frame_quality"),
            ("captions_updated_at", "SELECT MAX(updated_at) FROM captions"),
            ("clustered_frames",
             "SELECT COUNT(*) FROM frames WHERE cluster_id IS NOT NULL")):
        try:
            val = conn.execute(sql).fetchone()[0]
        except sqlite3.OperationalError:
            val = None
        if val:
            passes[key] = val
    if passes:
        snap["curation_passes"] = passes

    # meta-table settings recorded by curation passes
    for key, meta_key in (("identity_anchor", "identity_anchor_info"),
                          ("curation_performance", "curation_performance"),
                          ("study_classified_at", "study_classified_at")):
        try:
            value = manifest.meta_get(conn, meta_key)
        except sqlite3.OperationalError:
            value = None
        if value is not None:
            snap[key] = value
    return snap


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def next_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(version),0) FROM datasets"
                       ).fetchone()
    return int(row[0]) + 1


def validate_parent(conn: sqlite3.Connection,
                    parent_version: Optional[int]) -> Optional[str]:
    """Return a warning string when the requested parent does not exist."""
    if parent_version is None:
        return None
    row = conn.execute("SELECT 1 FROM datasets WHERE version = ?",
                       (int(parent_version),)).fetchone()
    if row:
        return None
    return (f"Parent version v{int(parent_version):03d} not found in the "
            "registry; lineage not recorded.")


def register_version(conn: sqlite3.Connection, *,
                     kind: str,
                     name: str,
                     directory: Path | str,
                     note: str = "",
                     parent_version: Optional[int] = None,
                     params: Optional[dict] = None,
                     image_count: int = 0,
                     val_count: int = 0,
                     files: Optional[list[dict]] = None,
                     curation: Optional[dict] = None,
                     version: Optional[int] = None,
                     content_hash: Optional[str] = None) -> dict:
    """Insert a version row (+ optional path membership). Returns the record.

    ``files`` entries are dicts with at least ``rel_path``; ``source_path``
    and ``caption`` are optional. ``version`` may be pre-reserved by the
    caller (the recipe builder names its folder after the version before
    placing files); when omitted the next free number is used.
    Never destructive: versions only append.
    """
    warnings: list[str] = []
    parent_warning = validate_parent(conn, parent_version)
    if parent_warning:
        warnings.append(parent_warning)
        parent_version = None

    version = int(version) if version is not None else next_version(conn)
    curation = curation if curation is not None \
        else capture_curation_snapshot(conn)
    conn.execute(
        "INSERT INTO datasets(version, recipe_name, recipe_json, "
        "content_hash, built_at, image_count, val_count, dir, note, "
        "parent_version, kind, curation_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (version, name, json.dumps(params or {}, ensure_ascii=False),
         content_hash, now_iso(), int(image_count), int(val_count),
         str(directory), note, parent_version, kind,
         json.dumps(curation, ensure_ascii=False)))
    if files:
        conn.executemany(
            "INSERT OR REPLACE INTO dataset_version_files"
            "(version, rel_path, source_path, caption) VALUES (?,?,?,?)",
            [(version, str(f.get("rel_path")),
              str(f["source_path"]) if f.get("source_path") else None,
              f.get("caption"))
             for f in files if f.get("rel_path")])
    conn.commit()
    record = {
        "version": version, "kind": kind, "name": name,
        "dir": str(directory), "note": note,
        "parent_version": parent_version,
        "created_at": now_iso(),
        "image_count": int(image_count), "val_count": int(val_count),
        "content_hash": content_hash,
        "params": params or {}, "file_count": len(files or []),
        "curation_snapshot": curation, "warnings": warnings,
    }
    _update_latest_meta(conn, record)
    return record


def _update_latest_meta(conn: sqlite3.Connection, record: dict) -> None:
    """Maintain cross-dataset pointers in the ``meta`` table.

    ``dataset_latest_version`` / ``dataset_latest_info`` always track the
    newest registered version; ``dataset_latest_by_kind`` maps each kind to
    its newest version. Best-effort: failures never break registration.
    """
    try:
        manifest.meta_set(conn, META_LATEST_VERSION, record["version"])
        manifest.meta_set(conn, META_LATEST_INFO, {
            "version": record["version"], "kind": record["kind"],
            "name": record["name"], "dir": record["dir"],
            "created_at": record["created_at"],
            "image_count": record["image_count"],
        })
        by_kind = manifest.meta_get(conn, META_LATEST_BY_KIND) or {}
        if not isinstance(by_kind, dict):           # defensive
            by_kind = {}
        by_kind[record["kind"] or KIND_BUILD] = record["version"]
        manifest.meta_set(conn, META_LATEST_BY_KIND, by_kind)
        conn.commit()
    except Exception as exc:                        # pragma: no cover
        log.warning("latest-version meta not updated: %s", exc)


def write_version_sidecar(version_dir: Path | str,
                          record: dict) -> Optional[Path]:
    """Write ``version.json`` inside the dataset folder. Never raises."""
    try:
        d = Path(version_dir)
        d.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in record.items() if k != "warnings"}
        path = d / SIDECAR_NAME
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path
    except Exception as exc:                        # pragma: no cover
        log.warning("version sidecar not written for %s: %s",
                    version_dir, exc)
        return None


def register_folder_export(output_base: Path | str, *,
                           kind: str,
                           name: str,
                           export_dir: Path | str,
                           files: Optional[list[dict]] = None,
                           image_count: int = 0,
                           params: Optional[dict] = None,
                           note: str = "",
                           parent_version: Optional[int] = None) -> dict:
    """Register a folder-based export into the shared registry.

    Safe entry point for export pipelines: opens its own manifest
    connection and converts every failure into a warning, so exports
    never fail because of version registration.
    """
    try:
        conn = manifest.connect(Path(output_base))
    except Exception as exc:
        return {"version": None,
                "warnings": [f"Version registry unavailable ({exc}); "
                             "export completed unversioned."]}
    try:
        record = register_version(
            conn, kind=kind, name=name, directory=export_dir, note=note,
            parent_version=parent_version, params=params,
            image_count=image_count, files=files)
        sidecar = write_version_sidecar(export_dir, record)
        if sidecar:
            record["sidecar"] = str(sidecar)
        return record
    except Exception as exc:
        return {"version": None,
                "warnings": [f"Version registration failed ({exc}); "
                             "export completed unversioned."]}
    finally:
        try:
            conn.close()
        except Exception:                           # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Queries: lineage / history / file diff
# ---------------------------------------------------------------------------

_ROW_KEYS = ("version", "name", "kind", "built_at", "image_count",
             "val_count", "dir", "note", "parent_version", "content_hash")


def get_version(conn: sqlite3.Connection, version: int) -> Optional[dict]:
    try:
        row = conn.execute(
            "SELECT version, recipe_name, kind, built_at, image_count, "
            "val_count, dir, note, parent_version, content_hash "
            "FROM datasets WHERE version = ?", (int(version),)).fetchone()
    except sqlite3.OperationalError:    # pre-v18 manifest opened read-only
        row = conn.execute(
            "SELECT version, recipe_name, NULL, built_at, image_count, "
            "val_count, dir, note, NULL, content_hash "
            "FROM datasets WHERE version = ?", (int(version),)).fetchone()
    if not row:
        return None
    d = dict(zip(_ROW_KEYS, row))
    d["kind"] = d["kind"] or KIND_BUILD
    return d


def lineage_chain(conn: sqlite3.Connection, version: int) -> list[dict]:
    """Walk parents from ``version`` back to the root (cycle-safe).

    Every step carries ``changes`` (membership diff vs its parent) and
    ``curation_changes`` (curation-snapshot delta vs its parent) so the
    chain reads as an evolution story, not just a list of ancestors.
    """
    chain: list[dict] = []
    seen: set[int] = set()
    current: Optional[int] = int(version)
    while current is not None and current not in seen:
        seen.add(current)
        rec = get_version(conn, current)
        if rec is None:
            break
        rec["changes"] = _change_summary(conn, rec)
        rec["curation_changes"] = _curation_delta_summary(conn, rec)
        chain.append(rec)
        current = rec.get("parent_version")
    return chain


def _membership_counts(conn: sqlite3.Connection, version: int) -> dict:
    out = {"frames": 0, "files": 0}
    try:
        out["frames"] = conn.execute(
            "SELECT COUNT(*) FROM dataset_frames WHERE version = ?",
            (version,)).fetchone()[0]
    except sqlite3.OperationalError:
        pass
    try:
        out["files"] = conn.execute(
            "SELECT COUNT(*) FROM dataset_version_files WHERE version = ?",
            (version,)).fetchone()[0]
    except sqlite3.OperationalError:
        pass
    return out


def diff_version_files(conn: sqlite3.Connection, va: int, vb: int) -> dict:
    """Path-based membership diff for folder-export versions."""
    def load(v: int) -> set[str]:
        return {r[0] for r in conn.execute(
            "SELECT rel_path FROM dataset_version_files WHERE version = ?",
            (int(v),))}
    a, b = load(va), load(vb)
    return {"a": va, "b": vb,
            "added": sorted(b - a), "removed": sorted(a - b),
            "summary": (f"v{va} -> v{vb}: +{len(b - a)} added, "
                        f"-{len(a - b)} removed")}


def _change_summary(conn: sqlite3.Connection, rec: dict) -> str:
    """One-line change summary of a version against its parent."""
    parent = rec.get("parent_version")
    if parent is None:
        return ""
    counts = _membership_counts(conn, rec["version"])
    try:
        if counts["frames"] > 0:
            from .builder import diff_datasets
            d = diff_datasets(conn, int(parent), int(rec["version"]))
            return (f"+{len(d['added'])} -{len(d['removed'])} frames, "
                    f"{len(d['caption_changed'])} captions, "
                    f"{len(d['split_changed'])} splits vs v{parent:03d}")
        if counts["files"] > 0:
            d = diff_version_files(conn, int(parent), int(rec["version"]))
            return (f"+{len(d['added'])} -{len(d['removed'])} files "
                    f"vs v{parent:03d}")
    except Exception:                               # pragma: no cover
        pass
    return f"derived from v{int(parent):03d}"


def curation_delta(conn: sqlite3.Connection, version: int,
                   parent_version: Optional[int]) -> Optional[dict]:
    """Structured delta between the curation snapshots of a version and
    its parent (how the curated pool evolved between the two exports).

    Returns None when either snapshot is missing (pre-versioning rows)."""
    if parent_version is None:
        return None
    a = get_curation_snapshot(conn, int(parent_version))
    b = get_curation_snapshot(conn, int(version))
    if not a or not b:
        return None

    def _num(d: Optional[dict], *keys) -> int:
        cur = d or {}
        for k in keys:
            cur = (cur or {}).get(k) if isinstance(cur, dict) else None
        try:
            return int(cur or 0)
        except (TypeError, ValueError):
            return 0

    delta: dict = {"parent": int(parent_version), "version": int(version)}
    # pool status movement (curation-relevant statuses)
    sa = a.get("frame_status_counts") or {}
    sb = b.get("frame_status_counts") or {}
    status: dict = {}
    for st in sorted(set(sa) | set(sb)):
        d = int(sb.get(st, 0)) - int(sa.get(st, 0))
        if d:
            status[st] = {"from": int(sa.get(st, 0)),
                          "to": int(sb.get(st, 0)), "delta": d}
    if status:
        delta["frame_status"] = status
    for key, path in (("captioned", ("caption_coverage", "captioned")),
                      ("manually_edited",
                       ("caption_coverage", "manually_edited")),
                      ("quality_scored",
                       ("quality_metrics", "scored_frames")),
                      ("study_classified", ("study_labels", "classified"))):
        d = _num(b, *path) - _num(a, *path)
        if d:
            delta[key] = {"from": _num(a, *path), "to": _num(b, *path),
                          "delta": d}
    return delta if len(delta) > 2 else None


def _curation_delta_summary(conn: sqlite3.Connection, rec: dict) -> str:
    """Human one-liner for :func:`curation_delta` (empty when no delta)."""
    try:
        d = curation_delta(conn, rec["version"], rec.get("parent_version"))
    except Exception:                               # pragma: no cover
        return ""
    if not d:
        return ""
    bits: list[str] = []
    for st, v in (d.get("frame_status") or {}).items():
        if st in ("selected", "packaged") or len(d.get("frame_status", {})) <= 3:
            bits.append(f"{st} {v['from']}→{v['to']}")
    for key, label in (("captioned", "captions"),
                       ("manually_edited", "caption edits"),
                       ("quality_scored", "quality-scored"),
                       ("study_classified", "study-classified")):
        if key in d:
            bits.append(f"{label} {d[key]['delta']:+d}")
    return ", ".join(bits)


def version_history(conn: sqlite3.Connection,
                    name: Optional[str] = None) -> list[dict]:
    """All registered versions (optionally filtered by recipe/export name),
    each with membership counts and a change summary vs its parent."""
    try:
        rows = conn.execute(
            "SELECT version, recipe_name, kind, built_at, image_count, "
            "val_count, dir, note, parent_version, content_hash "
            "FROM datasets ORDER BY version").fetchall()
    except sqlite3.OperationalError:    # pre-v18 manifest opened read-only
        rows = conn.execute(
            "SELECT version, recipe_name, NULL, built_at, image_count, "
            "val_count, dir, note, NULL, content_hash "
            "FROM datasets ORDER BY version").fetchall()
    out: list[dict] = []
    for row in rows:
        rec = dict(zip(_ROW_KEYS, row))
        rec["kind"] = rec["kind"] or KIND_BUILD
        if name and rec["name"] != name:
            continue
        rec["membership"] = _membership_counts(conn, rec["version"])
        rec["changes"] = _change_summary(conn, rec)
        rec["curation_changes"] = _curation_delta_summary(conn, rec)
        out.append(rec)
    return out


def get_curation_snapshot(conn: sqlite3.Connection,
                          version: int) -> Optional[dict]:
    try:
        row = conn.execute(
            "SELECT curation_json FROM datasets WHERE version = ?",
            (int(version),)).fetchone()
    except sqlite3.OperationalError:
        return None
    if not row or not row[0]:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_history(entries: list[dict]) -> str:
    if not entries:
        return ("No dataset versions registered yet. Run "
                "`pixel-curator build --recipe <name>` or a versioned export.")
    lines = [f"{'ver':>5}  {'kind':<19} {'name':<20} {'imgs':>5} "
             f"{'parent':>7}  {'built at':<20} changes / note",
             "-" * 100]
    for e in entries:
        parent = (f"v{e['parent_version']:03d}"
                  if e.get("parent_version") is not None else "-")
        detail = e.get("changes") or ""
        if e.get("note"):
            detail = f"{detail}  [{e['note']}]" if detail else e["note"]
        lines.append(
            f"v{e['version']:03d}  {e['kind']:<19} "
            f"{(e['name'] or '')[:20]:<20} {e['image_count'] or 0:>5} "
            f"{parent:>7}  {(e['built_at'] or '')[:19]:<20} {detail}")
        if e.get("curation_changes"):
            lines.append(f"{'':>13}curation: {e['curation_changes']}")
    return "\n".join(lines)


def render_lineage(chain: list[dict]) -> str:
    if not chain:
        return "Version not found in the registry."
    lines = [f"Lineage of v{chain[0]['version']:03d} "
             f"({chain[0]['name'] or 'unnamed'}):"]
    for i, rec in enumerate(chain):
        arrow = "   " if i == 0 else "-> "
        lines.append(
            f"  {arrow}v{rec['version']:03d}  {rec['kind']:<19} "
            f"{(rec['name'] or ''):<20} imgs={rec['image_count'] or 0:<5} "
            f"{(rec['built_at'] or '')[:19]}"
            + (f"  [{rec['note']}]" if rec.get("note") else ""))
        detail = []
        if rec.get("changes"):
            detail.append(rec["changes"])
        if rec.get("curation_changes"):
            detail.append(f"curation: {rec['curation_changes']}")
        for line in detail:
            lines.append(f"        {line}")
    if chain[-1].get("parent_version") is None and len(chain) == 1:
        lines.append("  (no parent recorded - use `build --parent N` or "
                     "export parenting to link versions)")
    return "\n".join(lines)


__all__ = [
    "KIND_BUILD", "KIND_LORA_TARGET_EXPORT", "KIND_KOHYA_PACKAGE",
    "KIND_MULTI_RECIPE_PACKAGE", "SIDECAR_NAME",
    "META_LATEST_VERSION", "META_LATEST_INFO", "META_LATEST_BY_KIND",
    "capture_curation_snapshot", "next_version", "validate_parent",
    "register_version", "write_version_sidecar", "register_folder_export",
    "get_version", "lineage_chain", "diff_version_files", "version_history",
    "curation_delta", "get_curation_snapshot",
    "render_history", "render_lineage",
]
