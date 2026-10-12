"""Dataset Pipeline Explorer - shared service (Phase X).

The single source of truth for a read-focused, visual data-lineage view of how
personal media moves through the dataset pipeline. Consumed by the CLI, the
Gradio GUI, and (via status JSON snapshots) the legacy UI - none of which are
imported here. This module depends only on the standard library plus an optional
lazy import of ``media_support`` for on-demand thumbnails.

Design guarantees:
  * Read-only: opens the manifest in SQLite ``mode=ro`` (never locks a running
    pass, never writes to source data).
  * Non-destructive: never moves/copies/edits source media. Thumbnails are
    written only to an explicit, separate cache dir, on demand, capped.
  * Safe on partial projects: every stage / table / field is optional; missing
    data yields warnings + unavailable stages, never exceptions.
  * Bounded memory: listing paginates at the SQL layer and never decodes image
    pixels; derived-filter scans are capped.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

DB_FILENAME = "extractor_manifest.sqlite3"
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 500
_MAX_DERIVED_SCAN = 50_000        # cap rows scanned when derived filters are active
HEIF_EXTS = (".heic", ".heif")
SURVIVOR_STATUSES = ("selected", "packaged", "kept")

# --- canonical pipeline stage model (missing stages are allowed) -------------
STAGE_DEFS = [
    ("raw_import", "Raw Import", "manifest",
     "All imported frames recorded in the manifest."),
    ("post_quality", "Post Quality Analysis", "manifest",
     "Frames that have been scored by the quality pass."),
    ("post_smart_curation", "Post Smart Curation", "manifest",
     "Frames that survived smart curation (selected / packaged / kept)."),
    ("post_keyframes", "Post Keyframe Pruning", "derived",
     "Keyframe-pruned survivors (tracked implicitly via surviving frames)."),
    ("post_clustering", "Post Clustering / Diversity", "manifest",
     "Frames assigned a diversity cluster id."),
    ("curated_versions", "Curated Dataset Versions", "dataset",
     "Frames included in one or more built dataset versions."),
    ("captioned", "Captioned", "manifest",
     "Frames with a stored training caption."),
    ("training_package_ready", "Training Package Ready", "package",
     "Frames packaged into a Kohya/Colab training bundle."),
    ("training_plan_ready", "Training Plan Ready", "workflow",
     "A training plan handoff exists for the project."),
]
STAGE_KEYS = [s[0] for s in STAGE_DEFS]
STAGE_LABEL = {s[0]: s[1] for s in STAGE_DEFS}
STAGE_ORDER = {s[0]: i for i, s in enumerate(STAGE_DEFS)}


# ============================ dataclasses ===================================
@dataclass
class PipelineStage:
    key: str
    label: str
    order: int
    description: str = ""
    source_kind: str = "unknown"     # manifest | dataset | package | workflow | derived
    source_path: Optional[str] = None
    available: bool = False
    count: int = 0
    previous_stage_key: Optional[str] = None
    retention_rate: Optional[float] = None
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "order": self.order,
                "description": self.description, "source_kind": self.source_kind,
                "source_path": self.source_path, "available": self.available,
                "count": self.count, "previous_stage_key": self.previous_stage_key,
                "retention_rate": self.retention_rate, "warnings": self.warnings}


@dataclass
class ExplorerImageRecord:
    image_id: str
    path: str
    filename: str
    thumbnail_path: Optional[str] = None
    stage_keys: list = field(default_factory=list)
    version_membership: list = field(default_factory=list)
    architecture_target: str = "unknown"     # sdxl | unknown
    overall_quality_score: Optional[float] = None
    quality_signals: dict = field(default_factory=dict)
    caption: str = ""
    source_folder: str = ""
    source_batch: str = ""
    flags: list = field(default_factory=list)
    duplicate_group: Optional[str] = None
    cluster_id: Optional[str] = None
    framing: Optional[str] = None
    coverage: Optional[str] = None
    target_label: Optional[str] = None
    last_analyzed: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "image_id", "path", "filename", "thumbnail_path", "stage_keys",
            "version_membership", "architecture_target", "overall_quality_score",
            "quality_signals", "caption", "source_folder", "source_batch",
            "flags", "duplicate_group", "cluster_id", "framing", "coverage",
            "target_label", "last_analyzed", "metadata")}


@dataclass
class ExplorerFilter:
    stage_key: Optional[str] = None
    min_quality: Optional[float] = None
    max_quality: Optional[float] = None
    architecture_target: Optional[str] = None
    target_label: Optional[str] = None
    required_flags: list = field(default_factory=list)
    excluded_flags: list = field(default_factory=list)
    caption_query: str = ""
    source_query: str = ""
    version: Optional[str] = None
    duplicate_status: Optional[str] = None    # duplicate | unique
    cluster_id: Optional[str] = None
    framing: Optional[str] = None
    coverage: Optional[str] = None
    # Visual Focus Classifier filters (applied when results are present)
    vf_focus: Optional[str] = None
    vf_detail: Optional[str] = None
    vf_pose: Optional[str] = None
    vf_target: Optional[str] = None
    vf_min_score: Optional[float] = None
    # Florence-2 caption filters (applied when results are present)
    fl_processed: Optional[bool] = None
    fl_focus: Optional[str] = None
    fl_routing_tag: Optional[str] = None
    caption_natural_query: str = ""
    # Visual focus routing filters (applied when routing results are present)
    rt_focus: Optional[str] = None
    rt_detail: Optional[str] = None
    rt_pose: Optional[str] = None
    rt_tag: Optional[str] = None
    rt_target: Optional[str] = None
    rt_recipe: Optional[str] = None
    rt_min_confidence: Optional[float] = None
    rt_min_score: Optional[float] = None
    # Specialized recipe plan filters (applied when a plan is present)
    rp_recipe: Optional[str] = None
    rp_primary: Optional[str] = None
    rp_arch: Optional[str] = None
    rp_version: Optional[str] = None
    rp_min_confidence: Optional[float] = None
    # Training package filters (applied when a package plan is present)
    pkg_assigned: Optional[str] = None
    pkg_primary: Optional[str] = None
    pkg_type: Optional[str] = None
    limit: int = DEFAULT_PAGE_SIZE
    offset: int = 0

    def clamped(self) -> "ExplorerFilter":
        self.limit = max(1, min(int(self.limit or DEFAULT_PAGE_SIZE), MAX_PAGE_SIZE))
        self.offset = max(0, int(self.offset or 0))
        return self

    def _derived_active(self) -> bool:
        return bool(self.architecture_target or self.target_label or self.coverage
                    or self.required_flags or self.excluded_flags
                    or self.duplicate_status or self.vf_focus or self.vf_detail
                    or self.vf_pose or self.vf_target
                    or self.vf_min_score is not None
                    or self.fl_processed is not None or self.fl_focus
                    or self.fl_routing_tag or self.caption_natural_query
                    or self.rt_focus or self.rt_detail or self.rt_pose
                    or self.rt_tag or self.rt_target
                    or self.rt_recipe or self.rt_min_confidence is not None
                    or self.rt_min_score is not None
                    or self.rp_recipe or self.rp_primary or self.rp_arch
                    or self.rp_version or self.rp_min_confidence is not None
                    or self.pkg_assigned or self.pkg_primary or self.pkg_type)


@dataclass
class StageComparison:
    stage_a: PipelineStage
    stage_b: PipelineStage
    kept_count: int = 0
    removed_count: int = 0
    added_count: int = 0
    kept_ids: list = field(default_factory=list)
    removed_ids: list = field(default_factory=list)
    added_ids: list = field(default_factory=list)
    retention_rate: Optional[float] = None
    removal_reasons: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"stage_a": self.stage_a.to_dict(), "stage_b": self.stage_b.to_dict(),
                "kept_count": self.kept_count, "removed_count": self.removed_count,
                "added_count": self.added_count, "kept_ids": self.kept_ids,
                "removed_ids": self.removed_ids, "added_ids": self.added_ids,
                "retention_rate": self.retention_rate,
                "removal_reasons": self.removal_reasons, "warnings": self.warnings}


# ============================ low-level helpers =============================
def manifest_path(output_base) -> Path:
    return Path(output_base).expanduser() / DB_FILENAME


def connect_ro(output_base) -> Optional[sqlite3.Connection]:
    """Open the manifest read-only. Returns None if absent/unopenable."""
    db = manifest_path(output_base)
    if not db.exists():
        return None
    try:
        return sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    except Exception:
        return None


def _table_exists(conn, name: str) -> bool:
    try:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,)).fetchone() is not None
    except Exception:
        return False


def _scalar(conn, sql: str, params: tuple = ()) -> Optional[int]:
    try:
        row = conn.execute(sql, params).fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except Exception:
        return None


def _has_column(conn, table: str, col: str) -> bool:
    try:
        return any(r[1] == col for r in conn.execute(f"PRAGMA table_info({table})"))
    except Exception:
        return False


# ============================ source discovery =============================
_ARTIFACT_NAMES = {
    "dataset_manifest": "dataset_manifest.json",
    "dataset_card": "dataset_card.json",
    "package_manifest": "package_manifest.json",
    "training_plan": "training_plan.json",
    "workflow_summary": "workflow_summary.md",
}


def discover_explorer_sources(output_base) -> dict:
    """Layered discovery of every data source the explorer can read. Never
    raises; reports what exists."""
    base = Path(output_base).expanduser()
    out: dict = {"output_base": str(base), "manifest_db": None, "tables": [],
                 "artifacts": {}, "warnings": []}
    if not base.exists():
        out["warnings"].append(f"output base does not exist: {base}")
        return out
    db = base / DB_FILENAME
    if db.exists():
        out["manifest_db"] = str(db)
        conn = connect_ro(base)
        if conn is not None:
            try:
                out["tables"] = sorted(
                    r[0] for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"))
            except Exception as exc:
                out["warnings"].append(f"manifest unreadable: {exc}")
            finally:
                conn.close()
    else:
        out["warnings"].append("no extractor_manifest.sqlite3 under output base")
    # on-disk artifacts (search shallowly - base + one level of subdirs)
    for key, name in _ARTIFACT_NAMES.items():
        hits = []
        if (base / name).is_file():
            hits.append(str(base / name))
        try:
            hits += [str(p) for p in base.glob(f"*/{name}")][:20]
        except Exception:
            pass
        if hits:
            out["artifacts"][key] = hits
    return out


# ============================ stage membership =============================
def _stage_frame_ids(conn, stage_key: str) -> Optional[set]:
    """Return the set of frame_ids in a manifest-backed stage, or None if the
    stage's membership cannot be enumerated from the manifest."""
    if conn is None:
        return None
    try:
        if stage_key == "raw_import":
            return {r[0] for r in conn.execute("SELECT frame_id FROM frames")}
        # post_keyframes is not separately tracked in the manifest (pruning
        # happens at extraction) -> report it as unavailable rather than fabricate
        # membership.
        if stage_key == "post_quality" and _table_exists(conn, "frame_quality"):
            return {r[0] for r in conn.execute(
                "SELECT frame_id FROM frame_quality")}
        if stage_key == "post_smart_curation":
            qs = ",".join("?" for _ in SURVIVOR_STATUSES)
            return {r[0] for r in conn.execute(
                f"SELECT frame_id FROM frames WHERE status IN ({qs})",
                SURVIVOR_STATUSES)}
        if stage_key == "post_clustering" and _has_column(conn, "frames", "cluster_id"):
            return {r[0] for r in conn.execute(
                "SELECT frame_id FROM frames WHERE cluster_id IS NOT NULL")}
        if stage_key == "curated_versions" and _table_exists(conn, "dataset_frames"):
            return {r[0] for r in conn.execute(
                "SELECT DISTINCT frame_id FROM dataset_frames")}
        if stage_key == "captioned" and _table_exists(conn, "captions"):
            return {r[0] for r in conn.execute(
                "SELECT frame_id FROM captions WHERE caption_text IS NOT NULL "
                "AND TRIM(caption_text) <> ''")}
    except Exception:
        return None
    return None   # package / plan stages aren't frame-enumerable from the manifest


def _package_frame_count(sources: dict) -> Optional[int]:
    for path in sources.get("artifacts", {}).get("package_manifest", []):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for key in ("image_count", "count", "total"):
                if isinstance(data.get(key), int):
                    return data[key]
            imgs = data.get("images") or data.get("frames")
            if isinstance(imgs, list):
                return len(imgs)
        except Exception:
            continue
    return None


def load_pipeline_stages(output_base) -> list:
    """Compute every stage's availability, count, and retention rate. Missing
    stages are returned with ``available=False`` and a warning, never omitted."""
    base = Path(output_base).expanduser()
    sources = discover_explorer_sources(base)
    conn = connect_ro(base)
    stages: list = []
    try:
        for key, label, kind, desc in STAGE_DEFS:
            st = PipelineStage(key=key, label=label, order=STAGE_ORDER[key],
                               description=desc, source_kind=kind)
            count = None
            if kind in ("manifest", "dataset", "derived"):
                if conn is None:
                    st.warnings.append("manifest unavailable")
                else:
                    ids = _stage_frame_ids(conn, key)
                    count = len(ids) if ids is not None else None
                    if key == "curated_versions" and _table_exists(conn, "datasets"):
                        st.source_path = sources.get("manifest_db")
                    if count is None:
                        st.warnings.append("stage not tracked in this manifest")
            elif kind == "package":
                count = _package_frame_count(sources)
                hits = sources.get("artifacts", {}).get("package_manifest")
                if hits:
                    st.source_path = hits[0]
                else:
                    st.warnings.append("no package_manifest.json found")
            elif kind == "workflow":
                hits = sources.get("artifacts", {}).get("training_plan")
                if hits:
                    st.source_path = hits[0]
                    count = 1
                else:
                    st.warnings.append("no training_plan.json found")
            st.available = count is not None
            st.count = int(count) if count is not None else 0
            stages.append(st)
    finally:
        if conn is not None:
            conn.close()
    # retention vs previous *available* stage
    prev = None
    for st in stages:
        if prev is not None:
            st.previous_stage_key = prev.key
            if st.available and prev.available and prev.count > 0:
                st.retention_rate = round(st.count / prev.count, 4)
        if st.available:
            prev = st
    return stages


# ============================ record normalization =========================
def _parse_signals(signals_json) -> dict:
    try:
        return json.loads(signals_json) if signals_json else {}
    except Exception:
        return {}


def _coverage_label(sig: dict, framing: Optional[str]) -> Optional[str]:
    fl = sig.get("framing_label")
    if not fl and isinstance(sig.get("coverage_intensity"), dict):
        fl = sig["coverage_intensity"].get("framing_label")
    return fl or framing


def _target_label(sig: dict, framing: Optional[str]) -> Optional[str]:
    hint = sig.get("training_use_hint")
    if hint:
        return str(hint)
    f = (framing or "").lower()
    if f in ("full_body", "full body"):
        return "pose"
    if f in ("closeup", "close_crop", "detail_crop"):
        return "detail"
    if f in ("upper_body", "torso_focus", "half_body"):
        return "outfit"
    if f in ("portrait", "face", "face_closeup", "headshot"):
        return "identity"
    return None


def _architecture_target(caption: str, sig: dict) -> str:
    meta = sig.get("architecture_target")
    if meta == "sdxl":
        return meta
    if caption and "score_" in caption.lower():
        return "sdxl"
    return "unknown"


def _derive_flags(status, caption, integrity, occlusion, face_count,
                  identity_sim, sig: dict, ext: str) -> list:
    flags = []
    if integrity is not None and integrity < 0.55:
        flags.append("low_quality")
    if (occlusion is not None and occlusion > 0.5) or \
            (_num(sig.get("face_occlusion_score")) or 0) > 0.5:
        flags.append("occluded")
    if (_num(sig.get("motion_blur_score")) or 0) > 0.5:
        flags.append("motion_blur")
    pd = sig.get("pose_duplicate")
    if isinstance(pd, dict) and pd.get("near_duplicate"):
        flags.append("pose_duplicate")
    if not (caption or "").strip():
        flags.append("no_caption")
    if (face_count in (0, None)) and identity_sim is None:
        flags.append("no_face")
    if status == "rejected":
        flags.append("rejected")
    if ext in HEIF_EXTS:
        flags.append("heic")
    return flags


def _num(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _stage_keys_for(status, has_quality, cluster_id, in_version, has_caption) -> list:
    keys = ["raw_import"]
    if has_quality:
        keys.append("post_quality")
    if status in SURVIVOR_STATUSES:
        keys.append("post_smart_curation")
    if cluster_id is not None:
        keys.append("post_clustering")
    if in_version:
        keys.append("curated_versions")
    if has_caption:
        keys.append("captioned")
    return sorted(keys, key=lambda k: STAGE_ORDER[k])


def _row_to_record(row, version_map: dict) -> ExplorerImageRecord:
    (fid, path, source_id, dhash, status, cluster_id, sharpness,
     framing, identity_sim, face_count, integrity, detail_retention,
     occlusion, signals_json, caption_text) = row
    sig = _parse_signals(signals_json)
    caption = caption_text or ""
    p = Path(path) if path else Path(fid)
    ext = p.suffix.lower()
    versions = version_map.get(fid, [])
    rec = ExplorerImageRecord(
        image_id=str(fid), path=str(path or ""), filename=p.name,
        cluster_id=str(cluster_id) if cluster_id is not None else None,
        framing=framing,
        coverage=_coverage_label(sig, framing),
        target_label=_target_label(sig, framing),
        architecture_target=_architecture_target(caption, sig),
        overall_quality_score=round(integrity, 4) if integrity is not None else None,
        caption=caption,
        source_folder=str(p.parent) if path else "",
        source_batch=str(source_id or ""),
        duplicate_group=str(dhash) if dhash else None,
        version_membership=[v for v, _ in versions],
        last_analyzed=None,
    )
    rec.stage_keys = _stage_keys_for(
        status, integrity is not None, cluster_id, bool(versions),
        bool(caption.strip()))
    rec.flags = _derive_flags(status, caption, integrity, occlusion, face_count,
                              identity_sim, sig, ext)
    rec.quality_signals = {
        "integrity": _num(integrity), "detail_retention": _num(detail_retention),
        "occlusion": _num(occlusion), "identity_sim": _num(identity_sim),
        "sharpness": _num(sharpness),
        "aesthetic_score": _num(sig.get("aesthetic_score")),
        "region_balance_score": _num(sig.get("region_balance_score")),
        "texture_preservation_score": _num(sig.get("texture_preservation_score")),
        "clothing_type": sig.get("clothing_type_label"),
    }
    rec.metadata = {"status": status, "concepts": [c for _, c in versions]}
    return rec


_BASE_SELECT = (
    "SELECT f.frame_id, f.path, f.source_id, f.dhash, f.status, "
    "{cluster}, f.sharpness, d.framing, d.identity_sim, d.face_count, "
    "q.integrity, q.detail_retention, q.occlusion, a.signals_json, c.caption_text "
    "FROM frames f "
    "LEFT JOIN detections d ON d.frame_id=f.frame_id "
    "LEFT JOIN frame_quality q ON q.frame_id=f.frame_id "
    "LEFT JOIN frame_advanced_quality a ON a.frame_id=f.frame_id "
    "LEFT JOIN captions c ON c.frame_id=f.frame_id ")


def _version_map(conn) -> dict:
    """frame_id -> [(version_label, concept)]."""
    out: dict = {}
    if not _table_exists(conn, "dataset_frames"):
        return out
    try:
        for fid, ver, concept in conn.execute(
                "SELECT frame_id, version, concept FROM dataset_frames"):
            out.setdefault(fid, []).append((f"v{int(ver):03d}", concept))
    except Exception:
        pass
    return out


def _load_visual_focus_map(output_base) -> dict:
    """Optional merge source: Visual Focus Classifier results (basename -> fields).
    Returns {} if results are absent or the service is unavailable."""
    try:
        from . import visual_focus_service as vfs
    except Exception:
        return {}
    base = Path(output_base).expanduser()
    for cand in (base / vfs.RESULTS_NAME, base / "visual_focus" / vfs.RESULTS_NAME):
        if cand.exists():
            try:
                return vfs.results_by_filename(cand)
            except Exception:
                return {}
    return {}


def _load_florence_map(output_base) -> dict:
    """Optional merge source: Florence-2 caption results (basename -> fields)."""
    try:
        from . import florence_captioning as flc
    except Exception:
        return {}
    base = Path(output_base).expanduser()
    for cand in (base / flc.RESULTS_NAME, base / "florence" / flc.RESULTS_NAME):
        if cand.exists():
            try:
                return flc.results_by_filename(cand)
            except Exception:
                return {}
    return {}


def _load_routing_map(output_base) -> dict:
    """Optional merge source: visual-focus routing results (basename -> fields)."""
    try:
        from . import visual_focus_classifier as vc
    except Exception:
        return {}
    base = Path(output_base).expanduser()
    for cand in (base / vc.RESULTS_NAME, base / "visual_focus" / vc.RESULTS_NAME):
        if cand.exists():
            try:
                return vc.results_by_filename(cand)
            except Exception:
                return {}
    return {}


def _load_recipe_plan_map(output_base) -> dict:
    """Optional merge source: specialized recipe assignments (basename -> fields)."""
    try:
        from . import specialized_recipe_builder as rb
    except Exception:
        return {}
    base = Path(output_base).expanduser()
    for cand in (base / rb.OUTPUT_FILES["assignments"],
                 base / "specialized_recipes" / rb.OUTPUT_FILES["assignments"]):
        if cand.exists():
            try:
                return rb.results_by_filename(cand)
            except Exception:
                return {}
    return {}


def _load_package_map(output_base) -> dict:
    """Optional merge source: training-package assignments (basename -> fields)."""
    try:
        from . import multi_recipe_packager as mp
    except Exception:
        return {}
    base = Path(output_base).expanduser()
    for cand in (base / "training_packages" / mp.OUTPUT_FILES["plan"],
                 base / mp.OUTPUT_FILES["plan"]):
        if cand.exists():
            try:
                plan = mp.load_multi_recipe_package_plan(cand)
                # load_* doesn't restore packages; read raw for image assignments
                import json as _json
                raw = _json.loads(cand.read_text(encoding="utf-8"))
                out = {}
                for spec in raw.get("packages", []):
                    for rec in spec.get("image_records", []):
                        nm = Path(rec.get("image_path", "")).name
                        if not nm:
                            continue
                        e = out.setdefault(nm, {
                            "assigned_training_packages": [],
                            "primary_training_package": "", "package_types": [],
                            "multi_recipe_package_plan_id": raw.get("plan_id")})
                        e["assigned_training_packages"].append(spec["package_name"])
                        e["package_types"].append(spec["package_type"])
                        if spec["package_type"] == "dreambooth" or \
                                not e["primary_training_package"]:
                            e["primary_training_package"] = spec["package_name"]
                return out
            except Exception:
                return {}
    return {}


def _build_where(conn, flt: ExplorerFilter):
    where, params = [], []
    if flt.stage_key == "post_smart_curation":
        qs = ",".join("?" for _ in SURVIVOR_STATUSES)
        where.append(f"f.status IN ({qs})")
        params += list(SURVIVOR_STATUSES)
    elif flt.stage_key == "post_quality":
        where.append("q.integrity IS NOT NULL")
    elif flt.stage_key == "post_clustering" and _has_column(conn, "frames", "cluster_id"):
        where.append("f.cluster_id IS NOT NULL")
    elif flt.stage_key == "captioned":
        where.append("c.caption_text IS NOT NULL AND TRIM(c.caption_text) <> ''")
    if flt.min_quality is not None:
        where.append("q.integrity >= ?")
        params.append(float(flt.min_quality))
    if flt.max_quality is not None:
        where.append("q.integrity <= ?")
        params.append(float(flt.max_quality))
    if flt.caption_query:
        where.append("LOWER(c.caption_text) LIKE ?")
        params.append(f"%{flt.caption_query.lower()}%")
    if flt.source_query:
        where.append("(LOWER(f.source_id) LIKE ? OR LOWER(f.path) LIKE ?)")
        params += [f"%{flt.source_query.lower()}%"] * 2
    if flt.cluster_id not in (None, ""):
        where.append("f.cluster_id = ?")
        params.append(flt.cluster_id)
    if flt.framing:
        where.append("LOWER(d.framing) = ?")
        params.append(flt.framing.lower())
    return (" WHERE " + " AND ".join(where)) if where else "", params


def load_image_records(output_base, filters: Optional[ExplorerFilter] = None) -> list:
    """Paginated, filtered image records. SQL filters paginate at the DB layer;
    derived filters (architecture/target/coverage/flags/duplicate) iterate a
    bounded, ordered window so memory stays flat even on 300k-frame projects."""
    flt = (filters or ExplorerFilter()).clamped()
    conn = connect_ro(output_base)
    if conn is None:
        return []
    try:
        cluster_expr = "f.cluster_id" if _has_column(conn, "frames", "cluster_id") else "NULL"
        sql = _BASE_SELECT.format(cluster=cluster_expr)
        where, params = _build_where(conn, flt)
        sql += where + " ORDER BY f.frame_id"
        vmap = _version_map(conn)
        dup_set = None
        if flt.duplicate_status:
            dup_set = {r[0] for r in conn.execute(
                "SELECT dhash FROM frames WHERE dhash IS NOT NULL "
                "GROUP BY dhash HAVING COUNT(*) > 1")}
        version_filter = flt.version
        vf_map = _load_visual_focus_map(output_base)
        fl_map = _load_florence_map(output_base)
        rt_map = _load_routing_map(output_base)
        rp_map = _load_recipe_plan_map(output_base)
        pkg_map = _load_package_map(output_base)

        def _passes_derived(rec: ExplorerImageRecord) -> bool:
            vf = vf_map.get(rec.filename) if vf_map else None
            if vf:
                rec.metadata["visual_focus"] = vf
            fl = fl_map.get(rec.filename) if fl_map else None
            if fl:
                rec.metadata["florence"] = fl
            rt = rt_map.get(rec.filename) if rt_map else None
            if rt:
                rec.metadata["routing"] = rt
            if flt.rt_focus and (not rt or rt.get("primary_focus") != flt.rt_focus):
                return False
            if flt.rt_detail and (not rt or rt.get("detail_level") != flt.rt_detail):
                return False
            if flt.rt_pose and (not rt or rt.get("pose_type") != flt.rt_pose):
                return False
            if flt.rt_tag and (not rt or flt.rt_tag not in (rt.get("routing_tags") or [])):
                return False
            if flt.rt_target and (not rt or flt.rt_target not in
                                  (rt.get("recommended_targets") or [])):
                return False
            if flt.rt_recipe and (not rt or rt.get("suggested_recipe_type") != flt.rt_recipe):
                return False
            if flt.rt_min_confidence is not None and (
                    not rt or (rt.get("routing_confidence") or 0) < flt.rt_min_confidence):
                return False
            if flt.rt_min_score is not None and (
                    not rt or (rt.get("suitability_score") or 0) < flt.rt_min_score):
                return False
            rp = rp_map.get(rec.filename) if rp_map else None
            if rp:
                rec.metadata["recipe_plan"] = rp
            if flt.rp_recipe and (not rp or flt.rp_recipe not in
                                  (rp.get("recommended_recipes") or [])):
                return False
            if flt.rp_primary and (not rp or rp.get("primary_recipe") != flt.rp_primary):
                return False
            if flt.rp_arch and (not rp or rp.get("architecture") != flt.rp_arch):
                return False
            if flt.rp_version and (not rp or rp.get("suggested_version_name") != flt.rp_version):
                return False
            if flt.rp_min_confidence is not None and (
                    not rp or (rp.get("recipe_confidence") or 0) < flt.rp_min_confidence):
                return False
            pkg = pkg_map.get(rec.filename) if pkg_map else None
            if pkg:
                rec.metadata["training_packages"] = pkg
            if flt.pkg_assigned and (not pkg or flt.pkg_assigned not in
                                     (pkg.get("assigned_training_packages") or [])):
                return False
            if flt.pkg_primary and (not pkg or
                                    pkg.get("primary_training_package") != flt.pkg_primary):
                return False
            if flt.pkg_type and (not pkg or flt.pkg_type not in
                                 (pkg.get("package_types") or [])):
                return False
            if flt.fl_processed is not None and bool(
                    fl and fl.get("florence_processed")) != flt.fl_processed:
                return False
            if flt.fl_focus and (not fl or fl.get("primary_focus") != flt.fl_focus):
                return False
            if flt.fl_routing_tag and (not fl or flt.fl_routing_tag not in
                                       (fl.get("routing_tags") or [])):
                return False
            if flt.caption_natural_query and (
                    not fl or flt.caption_natural_query.lower()
                    not in (fl.get("caption_natural") or "").lower()):
                return False
            if flt.vf_focus and (not vf or vf.get("primary_focus") != flt.vf_focus):
                return False
            if flt.vf_detail and (not vf or vf.get("detail_tier") != flt.vf_detail):
                return False
            if flt.vf_pose and (not vf or vf.get("pose_expression") != flt.vf_pose):
                return False
            if flt.vf_target and (not vf or flt.vf_target not in
                                  (vf.get("recommended_targets") or [])):
                return False
            if flt.vf_min_score is not None and (
                    not vf or (vf.get("visual_focus_score") or 0)
                    < flt.vf_min_score):
                return False
            if flt.architecture_target and rec.architecture_target != flt.architecture_target:
                return False
            if flt.target_label and rec.target_label != flt.target_label:
                return False
            if flt.coverage and rec.coverage != flt.coverage:
                return False
            if version_filter and version_filter not in rec.version_membership:
                return False
            if flt.required_flags and not all(f in rec.flags for f in flt.required_flags):
                return False
            if flt.excluded_flags and any(f in rec.flags for f in flt.excluded_flags):
                return False
            if dup_set is not None:
                is_dup = rec.duplicate_group in dup_set
                if flt.duplicate_status == "duplicate" and not is_dup:
                    return False
                if flt.duplicate_status == "unique" and is_dup:
                    return False
            return True

        derived = flt._derived_active() or bool(version_filter)
        out: list = []
        if not derived:
            sql += " LIMIT ? OFFSET ?"
            cur = conn.execute(sql, (*params, flt.limit, flt.offset))
            for row in cur:
                rec = _row_to_record(row, vmap)
                if vf_map:
                    vf = vf_map.get(rec.filename)
                    if vf:
                        rec.metadata["visual_focus"] = vf
                if fl_map:
                    fl = fl_map.get(rec.filename)
                    if fl:
                        rec.metadata["florence"] = fl
                if rt_map:
                    rt = rt_map.get(rec.filename)
                    if rt:
                        rec.metadata["routing"] = rt
                if rp_map:
                    rp = rp_map.get(rec.filename)
                    if rp:
                        rec.metadata["recipe_plan"] = rp
                if pkg_map:
                    pkg = pkg_map.get(rec.filename)
                    if pkg:
                        rec.metadata["training_packages"] = pkg
                out.append(rec)
        else:
            cur = conn.execute(sql, params)
            skipped = scanned = 0
            for row in cur:
                scanned += 1
                if scanned > _MAX_DERIVED_SCAN:
                    break
                rec = _row_to_record(row, vmap)
                if not _passes_derived(rec):
                    continue
                if skipped < flt.offset:
                    skipped += 1
                    continue
                out.append(rec)
                if len(out) >= flt.limit:
                    break
        return out
    finally:
        conn.close()


# ============================ summaries ====================================
def get_stage_summary(output_base) -> dict:
    stages = load_pipeline_stages(output_base)
    sources = discover_explorer_sources(output_base)
    avail = [s for s in stages if s.available]
    return {
        "output_base": str(Path(output_base).expanduser()),
        "manifest_db": sources.get("manifest_db"),
        "stages": [s.to_dict() for s in stages],
        "available_stages": [s.key for s in avail],
        "total_imported": next((s.count for s in stages if s.key == "raw_import"), 0),
        "artifacts": sources.get("artifacts", {}),
        "warnings": sources.get("warnings", []),
    }


# ============================ comparison ===================================
_REMOVAL_RULES = [
    ("rejected", "manual or rule-based exclusion"),
    ("low_quality", "quality threshold / low subject integrity"),
    ("pose_duplicate", "pose near-duplicate"),
    ("occluded", "poor framing / occlusion"),
    ("no_caption", "low caption readiness"),
]


def _removal_reasons(output_base, removed_ids: Iterable[str]) -> dict:
    ids = list(removed_ids)
    if not ids:
        return {}
    flt = ExplorerFilter(limit=MAX_PAGE_SIZE)
    sample = set(ids[:MAX_PAGE_SIZE])
    conn = connect_ro(output_base)
    reasons: dict = {}
    if conn is None:
        return {}
    try:
        cluster_expr = "f.cluster_id" if _has_column(conn, "frames", "cluster_id") else "NULL"
        sql = _BASE_SELECT.format(cluster=cluster_expr) + " ORDER BY f.frame_id"
        vmap = {}
        for row in conn.execute(sql):
            if row[0] not in sample:
                continue
            rec = _row_to_record(row, vmap)
            reason = "unknown"
            for flag, label in _REMOVAL_RULES:
                if flag in rec.flags:
                    reason = label
                    break
            reasons[reason] = reasons.get(reason, 0) + 1
            sample.discard(row[0])
            if not sample:
                break
    except Exception:
        pass
    finally:
        conn.close()
    return reasons


def compare_stages(output_base, stage_a: str, stage_b: str,
                   limit_ids: int = 1000) -> StageComparison:
    stages = {s.key: s for s in load_pipeline_stages(output_base)}
    sa = stages.get(stage_a) or PipelineStage(stage_a, stage_a, 0)
    sb = stages.get(stage_b) or PipelineStage(stage_b, stage_b, 0)
    cmp = StageComparison(stage_a=sa, stage_b=sb)
    conn = connect_ro(output_base)
    if conn is None:
        cmp.warnings.append("manifest unavailable")
        return cmp
    try:
        ids_a = _stage_frame_ids(conn, stage_a)
        ids_b = _stage_frame_ids(conn, stage_b)
    finally:
        conn.close()
    if ids_a is None or ids_b is None:
        miss = stage_a if ids_a is None else stage_b
        cmp.warnings.append(
            f"stage '{miss}' membership is not frame-enumerable from the manifest "
            f"(package/plan stages list no frame ids); counts only")
        return cmp
    kept = ids_a & ids_b
    removed = ids_a - ids_b
    added = ids_b - ids_a
    cmp.kept_count, cmp.removed_count, cmp.added_count = len(kept), len(removed), len(added)
    cmp.kept_ids = sorted(kept)[:limit_ids]
    cmp.removed_ids = sorted(removed)[:limit_ids]
    cmp.added_ids = sorted(added)[:limit_ids]
    cmp.retention_rate = round(len(kept) / len(ids_a), 4) if ids_a else None
    cmp.removal_reasons = _removal_reasons(output_base, sorted(removed))
    return cmp


# ============================ thumbnails (on demand) =======================
def ensure_thumbnails(records: list, output_base, max_n: int = DEFAULT_PAGE_SIZE,
                      max_side: int = 384) -> dict:
    """Generate thumbnails for up to ``max_n`` records into a separate cache dir
    (never touches sources). Sets ``record.thumbnail_path`` in place. Degrades
    safely when Pillow/pillow-heif are unavailable."""
    res = {"made": 0, "reused": 0, "skipped": 0, "warnings": []}
    try:
        from . import media_support as ms
    except Exception as exc:
        res["warnings"].append(f"media_support unavailable: {exc}")
        return res
    cache = Path(output_base).expanduser() / "explorer_thumbs"
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        res["warnings"].append(f"cannot create thumb cache: {exc}")
        return res
    ms.register_heif_opener_if_available()
    for rec in records[:max_n]:
        src = Path(rec.path)
        if not rec.path or not src.exists():
            res["skipped"] += 1
            continue
        dest = cache / f"{rec.image_id}.jpg"
        if dest.exists():
            rec.thumbnail_path = str(dest)
            res["reused"] += 1
            continue
        out = ms.build_thumbnail_safe(src, dest, max_side=max_side)
        if out.get("ok"):
            rec.thumbnail_path = out["thumbnail"]
            res["made"] += 1
        else:
            res["skipped"] += 1
            res["warnings"].extend(out.get("warnings", [])[:1])
    return res


def render_summary_markdown(summary: dict) -> str:
    lines = ["## Dataset Pipeline Explorer", "",
             f"Manifest: `{summary.get('manifest_db') or '(none found)'}`",
             f"Imported frames: **{summary.get('total_imported', 0)}**", "",
             "| Stage | Status | Count | Retention |",
             "|---|---|---:|---:|"]
    for s in summary.get("stages", []):
        status = "available" if s["available"] else "missing"
        ret = f"{s['retention_rate'] * 100:.1f}%" if s.get("retention_rate") is not None else "—"
        lines.append(f"| {s['label']} | {status} | "
                     f"{s['count'] if s['available'] else '—'} | {ret} |")
    for w in summary.get("warnings", []):
        lines.append(f"\n> ⚠ {w}")
    return "\n".join(lines)


def load_all_image_records(output_base, filters: Optional[ExplorerFilter] = None,
                           hard_cap: int = 1_000_000) -> list:
    """Return every matching record by walking pages past ``MAX_PAGE_SIZE``.

    ``load_image_records`` clamps any single page to ``MAX_PAGE_SIZE`` (a UI
    safety cap). For batch/offline consumers (recipe routing, packaging) that
    need the whole manifest, this paginates by offset until a short page is
    returned or ``hard_cap`` is reached. The per-page cap is left untouched, so
    UI payloads are unaffected.
    """
    from dataclasses import replace
    base = filters or ExplorerFilter()
    out: list = []
    offset = base.offset
    while True:
        page = replace(base, limit=MAX_PAGE_SIZE, offset=offset)
        recs = load_image_records(output_base, page)
        out.extend(recs)
        if len(recs) < MAX_PAGE_SIZE or len(out) >= hard_cap:
            break
        offset += MAX_PAGE_SIZE
    return out[:hard_cap]


__all__ = [
    "PipelineStage", "ExplorerImageRecord", "ExplorerFilter", "StageComparison",
    "STAGE_DEFS", "STAGE_KEYS", "STAGE_LABEL", "DEFAULT_PAGE_SIZE", "MAX_PAGE_SIZE",
    "manifest_path", "connect_ro", "discover_explorer_sources",
    "load_pipeline_stages", "load_image_records", "load_all_image_records",
    "get_stage_summary", "compare_stages", "ensure_thumbnails",
    "render_summary_markdown",
]
