"""SQLite manifest: connection handling, versioned schema migrations, helpers.

Schema version lives in PRAGMA user_version. Migrations are additive and
idempotent-safe; a v1.0 single-file manifest upgrades in place, no data loss.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from .config import DB_NAME
from .util import human_bytes, now_iso, stable_id

# -----------------------------
# Migrations
# -----------------------------

MIGRATIONS: dict[int, str] = {
    # v1 - core pipeline (matches legacy single-file schema)
    1: """
        CREATE TABLE IF NOT EXISTS sources (
            source_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            kind TEXT NOT NULL,
            size_bytes INTEGER,
            mtime REAL,
            discovered_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'discovered',
            output_dir TEXT,
            frames_extracted INTEGER DEFAULT 0,
            error TEXT,
            meta_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_sources_kind ON sources(kind);
        CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status);

        CREATE TABLE IF NOT EXISTS frames (
            frame_id TEXT PRIMARY KEY,
            source_id TEXT,
            path TEXT NOT NULL,
            dhash TEXT,
            sharpness REAL,
            brightness REAL,
            status TEXT NOT NULL DEFAULT 'new',
            scored_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_frames_source ON frames(source_id);
        CREATE INDEX IF NOT EXISTS idx_frames_status ON frames(status);

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT,
            ts TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL
        );
    """,
    # v2 - smart curation (face/identity) + future captioning + key/value meta
    2: """
        CREATE TABLE IF NOT EXISTS detections (
            frame_id TEXT PRIMARY KEY,
            face_count INTEGER,
            face_area REAL,
            det_conf REAL,
            identity_sim REAL,
            framing TEXT,
            embed BLOB,
            detected_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_det_sim ON detections(identity_sim);

        CREATE TABLE IF NOT EXISTS captions (
            frame_id TEXT PRIMARY KEY,
            tags_json TEXT,
            caption_text TEXT,
            edited INTEGER DEFAULT 0,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """,
    # v3 - CLIP diversity clustering (Phase 2.4)
    3: """
        ALTER TABLE frames ADD COLUMN cluster_id INTEGER;

        CREATE TABLE IF NOT EXISTS clip_embeds (
            frame_id TEXT PRIMARY KEY,
            embed BLOB,
            embedded_at TEXT
        );
    """,
    # v4 - versioned dataset builds (Phase 4)
    4: """
        CREATE TABLE IF NOT EXISTS datasets (
            version INTEGER PRIMARY KEY,
            recipe_name TEXT NOT NULL,
            recipe_json TEXT,
            content_hash TEXT,
            built_at TEXT,
            image_count INTEGER,
            val_count INTEGER,
            dir TEXT,
            note TEXT
        );

        CREATE TABLE IF NOT EXISTS dataset_frames (
            version INTEGER NOT NULL,
            frame_id TEXT NOT NULL,
            concept TEXT NOT NULL,
            split TEXT NOT NULL DEFAULT 'train',
            caption TEXT,
            PRIMARY KEY (version, frame_id, concept)
        );
        CREATE INDEX IF NOT EXISTS idx_df_version ON dataset_frames(version);
    """,
    # v5 - subject bbox for smart-crop + training runs (Phase 5)
    5: """
        ALTER TABLE detections ADD COLUMN bbox TEXT;

        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            dataset_version INTEGER,
            preset TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            config_json TEXT,
            command TEXT,
            output_dir TEXT,
            started_at TEXT,
            finished_at TEXT,
            current_step INTEGER DEFAULT 0,
            total_steps INTEGER DEFAULT 0,
            current_epoch INTEGER DEFAULT 0,
            total_epochs INTEGER DEFAULT 0,
            last_loss REAL,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS run_metrics (
            run_id INTEGER NOT NULL,
            step INTEGER NOT NULL,
            loss REAL,
            ts TEXT,
            PRIMARY KEY (run_id, step)
        );
    """,
    # v6 - evaluation / test generation (Phase 6)
    6: """
        CREATE TABLE IF NOT EXISTS evals (
            eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
            lora TEXT NOT NULL,
            label TEXT,
            category TEXT,
            prompt TEXT,
            seed INTEGER,
            backend TEXT,
            image_path TEXT,
            likeness REAL,
            created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_evals_lora ON evals(lora);
    """,
    # v7 - Study Intelligence Layer (professional study classification)
    7: """
        CREATE TABLE IF NOT EXISTS study_labels (
            frame_id TEXT PRIMARY KEY,
            study_primary TEXT,
            study_tags TEXT,
            study_confidence REAL,
            study_reason_codes TEXT,
            figure_study_score REAL,
            fashion_study_score REAL,
            lingerie_fashion_score REAL,
            pose_clarity_score REAL,
            silhouette_clarity_score REAL,
            garment_visibility_score REAL,
            identity_lock_score REAL,
            study_review_status TEXT DEFAULT 'auto',
            study_export_eligible INTEGER DEFAULT 0,
            manual_override INTEGER DEFAULT 0,
            classified_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_study_primary
            ON study_labels(study_primary);
        CREATE INDEX IF NOT EXISTS idx_study_conf
            ON study_labels(study_confidence);
    """,
    # v8 - Concept Control Layer (visual LoRA explorer + controlled variation)
    8: """
        CREATE TABLE IF NOT EXISTS lora_influence_profiles (
            lora_id TEXT PRIMARY KEY,
            path TEXT,
            family TEXT,
            influence_tags TEXT,
            weight_min REAL,
            weight_max REAL,
            weight_default REAL,
            identity_risk TEXT,
            known_conflicts TEXT,
            notes TEXT,
            preview TEXT,
            updated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_lip_family
            ON lora_influence_profiles(family);

        CREATE TABLE IF NOT EXISTS concept_control_presets (
            name TEXT PRIMARY KEY,
            kind TEXT,
            payload TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS variation_batches (
            batch_id TEXT PRIMARY KEY,
            spec TEXT,
            base_model TEXT,
            job_count INTEGER,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS variation_jobs (
            batch_id TEXT,
            variation_id TEXT,
            prompt TEXT,
            negative TEXT,
            seed INTEGER,
            loras TEXT,
            slider_state TEXT,
            warnings TEXT,
            output_path TEXT,
            created_at TEXT,
            PRIMARY KEY (batch_id, variation_id)
        );
    """,
    # v9 - Batch Variation Controller (modes, scoring, resumability)
    9: """
        ALTER TABLE variation_batches ADD COLUMN mode TEXT;
        ALTER TABLE variation_batches ADD COLUMN source_state TEXT;
        ALTER TABLE variation_batches ADD COLUMN identity_anchor TEXT;
        ALTER TABLE variation_batches ADD COLUMN hard_cap INTEGER;
        ALTER TABLE variation_jobs ADD COLUMN preservation_score REAL;
        ALTER TABLE variation_jobs ADD COLUMN risk_level TEXT;
        ALTER TABLE variation_jobs ADD COLUMN status TEXT DEFAULT 'planned';
    """,
    # v10 - Wardrobe Variation & Selective Region Editing
    10: """
        CREATE TABLE IF NOT EXISTS wardrobe_edits (
            edit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            mask_path TEXT,
            region_id TEXT,
            edit_mode TEXT,
            prompt TEXT,
            negative TEXT,
            loras TEXT,
            denoise REAL,
            seed INTEGER,
            preserve_background INTEGER,
            preserve_pose INTEGER,
            identity_score REAL,
            risk_level TEXT,
            readiness TEXT,
            output_path TEXT,
            created_at TEXT
        );
    """,
    # v11 - measured QA: face similarity on generated variations
    11: """
        ALTER TABLE variation_jobs ADD COLUMN measured_face_sim REAL;
    """,
    # v12 - Advanced Curation: duplicate / near-duplicate grouping
    # (additive; fields populated only when a duplicates scan runs and
    # never alter curation status - flag/group/recommend only)
    12: """
        ALTER TABLE frames ADD COLUMN dup_group_id TEXT;
        ALTER TABLE frames ADD COLUMN dup_score REAL;
        ALTER TABLE frames ADD COLUMN dup_method TEXT;
        ALTER TABLE frames ADD COLUMN dup_role TEXT;
        ALTER TABLE frames ADD COLUMN dup_reasons TEXT;
        CREATE INDEX IF NOT EXISTS idx_frames_dupgroup
            ON frames(dup_group_id);
    """,
    # v13 - Phase 2 annotation layer: free-form tags, custom metadata
    # fields and caption history (captions are versioned, never lost)
    13: """
        CREATE TABLE IF NOT EXISTS annotations (
            frame_id TEXT PRIMARY KEY,
            tags TEXT,
            custom_json TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS caption_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id TEXT NOT NULL,
            caption_text TEXT,
            saved_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_caption_history_frame
            ON caption_history(frame_id);
    """,
    # v14 - generation feedback (self-improvement loop): one row per
    # generated image with its full settings, auto scores and your rating
    14: """
        CREATE TABLE IF NOT EXISTS generation_feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            prompt TEXT,
            negative_prompt TEXT,
            lora_stack TEXT,
            seed INTEGER,
            recipe TEXT,
            batch_id TEXT,
            identity_score REAL,
            aesthetic_score REAL,
            rating INTEGER DEFAULT 0,
            tags TEXT,
            settings_json TEXT,
            created_at TEXT,
            rated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_rating
            ON generation_feedback(rating);
        CREATE INDEX IF NOT EXISTS idx_feedback_image
            ON generation_feedback(image_path);
    """,
    # v15 - deep quality signals (Priority 2): per-frame subject-integrity,
    # composition and artifact metrics computed during curation.
    15: """
        CREATE TABLE IF NOT EXISTS frame_quality (
            frame_id TEXT PRIMARY KEY,
            face_sharpness REAL,
            micro_detail REAL,
            detail_retention REAL,
            blockiness REAL,
            centering REAL,
            headroom REAL,
            occlusion REAL,
            frontalness REAL,
            eyes_open REAL,
            pose_bucket TEXT,
            integrity REAL,
            metric_version INTEGER DEFAULT 1,
            face_pass INTEGER DEFAULT 0,
            calculated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_fq_integrity
            ON frame_quality(integrity);
        CREATE INDEX IF NOT EXISTS idx_fq_version
            ON frame_quality(metric_version);
    """,
    # v16 - Priority 3 controllability: optional VLM (Florence-2) rich captions
    # alongside WD14 tags, and an outfit/accessory separation layer.
    16: """
        ALTER TABLE captions ADD COLUMN vlm_text TEXT;
        ALTER TABLE captions ADD COLUMN vlm_model TEXT;

        CREATE TABLE IF NOT EXISTS frame_outfit (
            frame_id TEXT PRIMARY KEY,
            dominant_colors TEXT,
            garments TEXT,
            signature TEXT,
            calculated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_outfit_sig
            ON frame_outfit(signature);
    """,
    # v17 - optional advanced quality signals (figure/region/coverage/texture)
    # stored additively as JSON; old manifests + readers are unaffected.
    17: """
        CREATE TABLE IF NOT EXISTS frame_advanced_quality (
            frame_id TEXT PRIMARY KEY,
            signals_json TEXT,
            signal_version INTEGER DEFAULT 1,
            calculated_at TEXT
        );
    """,
    # v18 - Dataset Versioning (Phase 2): explicit lineage between versions,
    # a curation-state snapshot captured at version time, a `kind` so folder
    # exports (lora-target splits, kohya packages) share the same registry as
    # recipe builds, and path-based membership for versions without frame_ids.
    # Existing rows keep NULLs (readers treat NULL kind as 'build').
    18: """
        ALTER TABLE datasets ADD COLUMN parent_version INTEGER;
        ALTER TABLE datasets ADD COLUMN kind TEXT;
        ALTER TABLE datasets ADD COLUMN curation_json TEXT;

        CREATE TABLE IF NOT EXISTS dataset_version_files (
            version INTEGER NOT NULL,
            rel_path TEXT NOT NULL,
            source_path TEXT,
            caption TEXT,
            PRIMARY KEY (version, rel_path)
        );
        CREATE INDEX IF NOT EXISTS idx_dvf_version
            ON dataset_version_files(version);
    """,

    # v19 - Early Pre-filtering (Phase 5): frames rejected by the fast
    # ingestion battery get status 'rejected_prefilter' plus a machine-
    # readable reason here (tiny_file / low_resolution / heavy_compression /
    # truncated / text_overlay). Restore clears both. NULL for every frame
    # the prefilter never touched.
    19: """
        ALTER TABLE frames ADD COLUMN prefilter_reason TEXT;
    """,

    # v20 - Training Experiment Tracking (Phase 8): artifact-lineage records
    # linking trained models (.safetensors) to the dataset versions they
    # were trained on. Distinct from `runs` (live process tracking of the
    # internal launcher): experiments also cover external trainings
    # (Colab kohya / DreamBooth) and carry human notes + outcomes.
    20: """
        CREATE TABLE IF NOT EXISTS experiments (
            exp_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            model_path TEXT,
            model_hash TEXT,
            model_type TEXT DEFAULT 'lora',
            base_model TEXT,
            dataset_version INTEGER,
            dataset_dir TEXT,
            run_id INTEGER,
            params_json TEXT,
            outcome TEXT,
            rating INTEGER,
            notes TEXT,
            source TEXT DEFAULT 'manual',
            trained_at TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_exp_dataset
            ON experiments(dataset_version);
        CREATE INDEX IF NOT EXISTS idx_exp_name ON experiments(name);
    """,
}

SCHEMA_VERSION = max(MIGRATIONS)

DEFAULT_TIMEOUT = 30.0      # seconds a connection waits on a busy manifest


class ManifestLocked(RuntimeError):
    """The manifest is held by another process (typically an active
    training run) and a write/migration could not be applied in time."""

    def __init__(self, detail: str = ""):
        super().__init__(
            "The manifest database is temporarily locked - an active "
            "process (typically a training run) is writing to it"
            + (f" ({detail})" if detail else "") + ".\n"
            "Read-only commands (runs, status, insights) work during "
            "training; retry write commands in a moment or after the "
            "training run finishes.")


def connect(output_base: Path, readonly: bool = False,
            timeout: float = DEFAULT_TIMEOUT,
            run_migrations: bool = True) -> sqlite3.Connection:
    """Manifest connection with lock robustness.

    readonly=True opens the database in SQLite URI read-only mode: no
    journal-mode changes, no migrations, no writes - safe to run while a
    training process holds write locks.  Reads of columns added by newer
    schema versions are the callers' responsibility (all studio readers
    already degrade for older schemas).
    """
    output_base = Path(output_base)
    db = output_base / DB_NAME

    if readonly and db.exists():
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True,
                               timeout=timeout, check_same_thread=False)
        conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)};")
        return conn
    # readonly on a missing manifest falls through: nothing can be locked,
    # and creating the empty schema preserves old behavior.

    output_base.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db, timeout=timeout, check_same_thread=False)
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)};")
    try:
        # journal_mode is a write when it actually changes - only touch it
        # when needed so hot-path connects never contend with the trainer
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        if str(mode).lower() != "wal":
            conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("SELECT 1;").fetchone()
    except sqlite3.OperationalError:
        # WAL unsupported on some network/exFAT volumes
        try:
            conn.close()
            conn = sqlite3.connect(db, timeout=timeout,
                                   check_same_thread=False)
            conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)};")
            conn.execute("PRAGMA journal_mode=DELETE;")
            conn.execute("PRAGMA synchronous=FULL;")
            logging.info("SQLite WAL unavailable on this volume; "
                         "using DELETE journal")
        except sqlite3.OperationalError:
            pass        # busy volume: proceed; busy_timeout covers reads
    if run_migrations:
        _migrate_with_retry(conn, timeout)
    return conn


def _migrate_with_retry(conn: sqlite3.Connection,
                        timeout: float = DEFAULT_TIMEOUT,
                        attempts: int = 3) -> None:
    """Apply pending migrations with backoff; no-op (zero writes) when the
    schema is already current, so routine connects never take locks."""
    current = conn.execute("PRAGMA user_version;").fetchone()[0]
    if current >= SCHEMA_VERSION:
        return
    import time as _time
    delay = min(1.0, max(0.05, timeout / 10))
    # cap per-attempt blocking so a long-held trainer lock surfaces as a
    # clear message in seconds, not minutes; restored after the attempts
    per_attempt = max(0.05, min(timeout, 2.0))
    conn.execute(f"PRAGMA busy_timeout = {int(per_attempt * 1000)};")
    last: Exception | None = None
    try:
        for attempt in range(attempts):
            try:
                migrate(conn)
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                last = exc
                _time.sleep(delay * (attempt + 1))
    finally:
        conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)};")
    raise ManifestLocked(f"pending schema update to v{SCHEMA_VERSION}: "
                         f"{last}")


def migrate(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version;").fetchone()[0]
    for version in sorted(MIGRATIONS):
        if version > current:
            # statement-wise so re-applied ALTERs (e.g. after a restored
            # backup reset user_version) are tolerated, true to the
            # "idempotent-safe" contract; CREATEs use IF NOT EXISTS
            for stmt in MIGRATIONS[version].split(";"):
                stmt = stmt.strip()
                if not stmt:
                    continue
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as exc:
                    msg = str(exc).lower()
                    if ("duplicate column" in msg
                            or "already exists" in msg):
                        continue
                    raise
            conn.execute(f"PRAGMA user_version = {version};")
            conn.commit()
            logging.info("Manifest migrated to schema v%d", version)


# -----------------------------
# Meta key/value
# -----------------------------

def meta_set(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value)),
    )
    conn.commit()


def meta_get(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


# -----------------------------
# Sources / frames / events
# -----------------------------

def event(conn: sqlite3.Connection, source_id: str, level: str, message: str) -> None:
    conn.execute(
        "INSERT INTO events(source_id, ts, level, message) VALUES (?, ?, ?, ?)",
        (source_id, now_iso(), level, message),
    )
    conn.commit()


def upsert_source(
    conn: sqlite3.Connection,
    path: Path,
    kind: str,
    status: str = "discovered",
    meta: dict | None = None,
) -> str:
    sid = stable_id(path)
    st = path.stat()
    conn.execute(
        """
        INSERT INTO sources(
            source_id, path, kind, size_bytes, mtime, discovered_at, status, meta_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            path=excluded.path, kind=excluded.kind,
            size_bytes=excluded.size_bytes, mtime=excluded.mtime
        """,
        (sid, str(path), kind, st.st_size, st.st_mtime, now_iso(), status,
         json.dumps(meta or {}, ensure_ascii=False)),
    )
    conn.commit()
    return sid


def get_status(conn: sqlite3.Connection, source_id: str) -> str | None:
    row = conn.execute(
        "SELECT status FROM sources WHERE source_id = ?", (source_id,)
    ).fetchone()
    return row[0] if row else None


def mark(
    conn: sqlite3.Connection,
    source_id: str,
    status: str,
    *,
    output_dir: Path | None = None,
    frames_extracted: int | None = None,
    error: str | None = None,
    meta: dict | None = None,
) -> None:
    updates = ["status = ?"]
    values: list[object] = [status]
    if output_dir is not None:
        updates.append("output_dir = ?")
        values.append(str(output_dir))
    if frames_extracted is not None:
        updates.append("frames_extracted = ?")
        values.append(frames_extracted)
    if error is not None:
        updates.append("error = ?")
        values.append(error)
    if meta is not None:
        updates.append("meta_json = ?")
        values.append(json.dumps(meta, ensure_ascii=False))
    values.append(source_id)
    conn.execute(f"UPDATE sources SET {', '.join(updates)} WHERE source_id = ?", values)
    conn.commit()


def stats(conn: sqlite3.Connection) -> dict:
    out: dict = {"sources": {}, "frames": {}, "totals": {}}
    for kind, status, n, size in conn.execute(
        "SELECT kind, status, COUNT(*), COALESCE(SUM(size_bytes),0) "
        "FROM sources GROUP BY kind, status"
    ):
        out["sources"].setdefault(kind, {})[status] = {"count": n, "bytes": size}
    for status, n in conn.execute("SELECT status, COUNT(*) FROM frames GROUP BY status"):
        out["frames"][status] = n
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(frames_extracted),0), COALESCE(SUM(size_bytes),0) "
        "FROM sources"
    ).fetchone()
    out["totals"] = {
        "sources": row[0],
        "frames_extracted": row[1],
        "source_bytes": row[2],
        "source_bytes_h": human_bytes(row[2]),
    }
    try:
        out["totals"]["faces_scanned"] = conn.execute(
            "SELECT COUNT(*) FROM detections"
        ).fetchone()[0]
    except sqlite3.OperationalError:
        pass
    return out
