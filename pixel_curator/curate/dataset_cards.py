"""Priority 4 - dataset validation, rich dataset cards & structured manifests.

A local-first, stdlib-only reproducibility layer that makes every exported
training set auditable before LoRA training. Two cooperating surfaces:

  * ``validate_exported_dataset`` - a modular, warning-by-default validator
    (orientation, caption richness, session dominance, subject integrity, pose/
    framing outliers). Warnings never block export unless ``strict`` is set.
  * ``generate_dataset_card`` - writes ``dataset_card.md``, ``dataset_card.json``
    and a companion ``dataset_manifest.structured.json`` (schema
    ``dataset_manifest_v2``). It is purely additive: it reads the existing
    ``dataset_manifest.json`` and never modifies or overwrites it.

Everything is deterministic (sorted inputs, stable hashes), non-destructive (it
only writes small metadata files), and degrades gracefully - a failure in any
sub-check is captured and reported, never raised.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

GENERATOR = "pixel_curator.curate.dataset_cards"
STRUCTURED_SCHEMA = "dataset_manifest_v2"
CARD_VERSION = 1


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DatasetValidationConfig:
    check_orientation: bool = True
    check_caption_richness: bool = True
    check_session_dominance: bool = True
    check_subject_integrity: bool = True
    check_pose_framing_outliers: bool = True
    check_pose_angle: bool = True
    check_pose_duplicates: bool = True
    strict: bool = False
    require_advanced_signals: bool = False
    require_pose_signal: bool = False
    max_dominant_pose_ratio: float = 0.45
    max_dominant_angle_ratio: float = 0.50
    max_pose_near_duplicate_ratio: float = 0.25
    max_single_pose_cluster_ratio: float = 0.15
    # conservative defaults
    min_caption_tokens: int = 5
    min_caption_concepts: int = 3
    max_duplicate_caption_ratio: float = 0.20
    min_caption_richness_score: float = 0.40
    max_single_session_ratio: float = 0.35
    max_single_source_folder_ratio: float = 0.40
    min_subject_integrity_score: float = 0.55
    pose_outlier_min_samples: int = 8
    max_examples: int = 5

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetValidationWarning:
    code: str
    message: str
    severity: str = "warning"
    count: int = 0
    examples: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetValidationReport:
    dataset_path: str
    sample_count: int
    warnings: list  # list[DatasetValidationWarning]
    metrics: dict
    passed: bool

    def to_dict(self) -> dict:
        return {
            "dataset_path": self.dataset_path,
            "sample_count": self.sample_count,
            "passed": self.passed,
            "warning_count": len(self.warnings),
            "warnings": [w.to_dict() if isinstance(w, DatasetValidationWarning)
                         else w for w in self.warnings],
            "metrics": self.metrics,
        }


@dataclass
class DatasetCardConfig:
    include_frame_list: bool = True
    max_frame_rows_markdown: int = 500
    include_validation: bool = True
    include_reproducibility: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Manifest / sample normalisation
# ---------------------------------------------------------------------------
def _load_manifest(dataset_dir: Path, manifest_path: Path | None,
                   manifest: dict | None) -> dict:
    if manifest is not None:
        return manifest
    candidates = []
    if manifest_path is not None:
        candidates.append(Path(manifest_path))
    candidates.append(Path(dataset_dir) / "dataset_manifest.json")
    for c in candidates:
        try:
            if c.is_file():
                data = json.loads(c.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            continue
    return {}


def _first(d: dict, *keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


def _nested(sample: dict, *paths):
    """Look up the first present dotted path within a sample's nested dicts."""
    for path in paths:
        cur = sample
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur and cur[part] is not None:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and not isinstance(cur, dict):
            return cur
    return None


def _flat(sample: dict) -> dict:
    """A merged read-only view: top-level entry keys plus any nested ``signals``
    / ``metadata`` block, so a field present in either place is visible."""
    out = dict(sample)
    for nest in ("signals", "metadata", "card_fields"):
        blk = sample.get(nest)
        if isinstance(blk, dict):
            for k, v in blk.items():
                out.setdefault(k, v)
    return out


def _derive_session(sample: dict) -> str:
    f = _flat(sample)
    explicit = _first(f, "source_session", "session_id", "session",
                      "source_id", "source_folder", "video_id", "capture_date")
    if explicit:
        return str(explicit)
    src = _first(f, "source_path", "path", default="")
    if src:
        return Path(str(src)).parent.name or "unknown"
    return "unknown"


def _source_folder(sample: dict) -> str:
    f = _flat(sample)
    src = _first(f, "source_path", "path", default="")
    if src:
        return str(Path(str(src)).parent)
    return "unknown"


def _subject_integrity(sample: dict):
    f = _flat(sample)
    val = _first(f, "subject_integrity_score", "figure_integrity_score",
                 "integrity_score", "subject_integrity")
    if val is None:
        val = _nested(sample,
                      "signals.figure_integrity_score",
                      "signals.figure_quality.integrity",
                      "advanced_quality.figure_quality.integrity",
                      "advanced_quality.subject_integrity",
                      "figure_quality.integrity")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _pose_label(sample: dict):
    f = _flat(sample)
    return _first(f, "pose_bucket", "pose_type", "dynamic_pose_label", "pose")


def _framing_label(sample: dict):
    f = _flat(sample)
    return _first(f, "framing_label", "framing", "coverage_label",
                  "full_body_framing")


def _coverage_label(sample: dict):
    f = _flat(sample)
    return _first(f, "coverage_label", "coverage_level")


def _orientation_label(sample: dict) -> str:
    f = _flat(sample)
    o = _first(f, "orientation", "canonical_orientation", "orientation_state")
    if o is not None:
        return str(o)
    exif = _first(f, "exif_orientation")
    if exif is not None:
        return f"exif:{exif}"
    return "unknown"


def _caption_of(sample: dict, dataset_dir: Path) -> str:
    cap = sample.get("caption")
    if isinstance(cap, str):
        return cap
    rel = sample.get("caption_target") or sample.get("caption_path")
    if rel:
        try:
            p = Path(dataset_dir) / rel
            if p.is_file():
                return p.read_text(encoding="utf-8")
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass
    return ""


def _image_rel(sample: dict) -> str:
    return str(sample.get("image_target") or sample.get("image")
              or sample.get("image_path") or "")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _caption_concepts(caption: str) -> int:
    return len([seg for seg in caption.split(",") if seg.strip()])


def compute_caption_stats(captions: list) -> dict:
    n = len(captions)
    if n == 0:
        return {"sample_count": 0, "empty_captions": 0,
                "avg_token_count": 0.0, "avg_concept_count": 0.0,
                "unique_ratio": 0.0, "duplicate_count": 0,
                "duplicate_ratio": 0.0, "richness_score": 0.0}
    tokens = [len(c.split()) for c in captions]
    concepts = [_caption_concepts(c) for c in captions]
    empty = sum(1 for c in captions if not c.strip())
    uniq = len(set(c.strip() for c in captions))
    unique_ratio = uniq / n
    duplicate_count = n - uniq
    duplicate_ratio = duplicate_count / n
    avg_tok = sum(tokens) / n
    avg_con = sum(concepts) / n
    # richness: blend normalised token depth, concept depth and uniqueness
    tok_norm = min(1.0, avg_tok / 20.0)
    con_norm = min(1.0, avg_con / 6.0)
    richness = round((tok_norm * 0.4 + con_norm * 0.3 + unique_ratio * 0.3), 4)
    return {
        "sample_count": n,
        "empty_captions": empty,
        "avg_token_count": round(avg_tok, 3),
        "avg_concept_count": round(avg_con, 3),
        "unique_ratio": round(unique_ratio, 4),
        "duplicate_count": duplicate_count,
        "duplicate_ratio": round(duplicate_ratio, 4),
        "richness_score": richness,
    }


def _distribution(values: list) -> dict:
    c = Counter(str(v) for v in values if v is not None and str(v) != "")
    return dict(c.most_common())


def compute_diversity_stats(samples: list, dataset_dir: Path) -> dict:
    orientation = _distribution([_orientation_label(s) for s in samples])
    framing = _distribution([_framing_label(s) for s in samples])
    coverage = _distribution([_coverage_label(s) for s in samples])
    pose = _distribution([_pose_label(s) for s in samples])
    sessions = _distribution([_derive_session(s) for s in samples])
    folders = _distribution([_source_folder(s) for s in samples])
    out = {
        "orientation_distribution": orientation,
        "framing_distribution": framing,
        "coverage_distribution": coverage,
        "pose_distribution": pose,
        "session_distribution": sessions,
        "source_folder_distribution": folders,
    }
    try:
        from .pose_angle import compute_pose_angle_stats
        out["pose_angle_stats"] = compute_pose_angle_stats(samples)
    except Exception:                                    # never break the card
        pass
    return out


def compute_quality_summary(samples: list) -> dict:
    def _avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def _grab(s, *keys):
        f = _flat(s)
        return _first(f, *keys)

    integ = [_subject_integrity(s) for s in samples]
    integ_present = sum(1 for v in integ if v is not None)
    return {
        "global_quality_avg": _avg([_to_float(_grab(s, "quality_score"))
                                    for s in samples]),
        "subject_integrity_avg": _avg(integ),
        "regional_quality_avg": _avg([_to_float(_grab(s, "regional_quality_score"))
                                      for s in samples]),
        "natural_texture_avg": _avg([_to_float(_grab(s, "texture_preservation_score"))
                                     for s in samples]),
        "advanced_signal_coverage": round(integ_present / len(samples), 4)
        if samples else 0.0,
    }


def _to_float(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def compute_orientation_stats(samples: list) -> dict:
    return {"orientation_distribution":
            _distribution([_orientation_label(s) for s in samples])}


def compute_session_stats(samples: list) -> dict:
    sessions = _distribution([_derive_session(s) for s in samples])
    folders = _distribution([_source_folder(s) for s in samples])
    n = max(1, len(samples))
    top_session = next(iter(sessions.items()), (None, 0))
    top_folder = next(iter(folders.items()), (None, 0))
    return {
        "session_distribution": sessions,
        "source_folder_distribution": folders,
        "top_session": top_session[0],
        "top_session_ratio": round(top_session[1] / n, 4),
        "top_source_folder": top_folder[0],
        "top_source_folder_ratio": round(top_folder[1] / n, 4),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _cap(examples: list, limit: int) -> list:
    return [str(e) for e in examples[:limit]]


def _check_orientation(samples, cfg, warnings):
    mismatch, legacy, missing = [], [], 0
    for s in samples:
        f = _flat(s)
        exif = _first(f, "exif_orientation")
        normalized = _first(f, "exif_normalized", "orientation_normalized")
        legacy_flag = _first(f, "orientation_legacy", "legacy_orientation_path",
                             "legacy_orientation")
        has_state = (exif is not None or normalized is not None
                     or _first(f, "orientation", "canonical_orientation",
                               "orientation_state") is not None)
        if not has_state:
            missing += 1
        try:
            exif_rot = exif is not None and int(exif) not in (0, 1)
        except (TypeError, ValueError):
            exif_rot = False
        if exif_rot and bool(normalized):
            mismatch.append(_image_rel(s) or _derive_session(s))
        if legacy_flag:
            legacy.append(_image_rel(s) or _derive_session(s))
    if mismatch:
        warnings.append(DatasetValidationWarning(
            "orientation_exif_pixel_mismatch",
            "EXIF orientation indicates rotation but pixels are marked already "
            "normalized (conflicting orientation state).",
            count=len(mismatch), examples=_cap(mismatch, cfg.max_examples)))
    if legacy:
        warnings.append(DatasetValidationWarning(
            "orientation_legacy_path_detected",
            "Samples carry legacy orientation metadata from a pre-canonical "
            "path; verify pixels are upright.",
            count=len(legacy), examples=_cap(legacy, cfg.max_examples)))
    if samples and missing == len(samples):
        warnings.append(DatasetValidationWarning(
            "orientation_state_missing",
            "No orientation state recorded for any sample; orientation could "
            "not be audited.",
            count=missing, severity="info"))


def _check_captions(samples, captions, stats, cfg, warnings):
    missing = [_image_rel(s) for s, c in zip(samples, captions) if not c.strip()]
    short = [_image_rel(s) for s, c in zip(samples, captions)
             if c.strip() and len(c.split()) < cfg.min_caption_tokens]
    low_concept = [_image_rel(s) for s, c in zip(samples, captions)
                   if c.strip() and _caption_concepts(c) < cfg.min_caption_concepts]
    if missing:
        warnings.append(DatasetValidationWarning(
            "caption_missing", "Samples have empty captions.",
            count=len(missing), examples=_cap(missing, cfg.max_examples)))
    if short:
        warnings.append(DatasetValidationWarning(
            "caption_too_short",
            f"Captions below {cfg.min_caption_tokens} tokens.",
            count=len(short), examples=_cap(short, cfg.max_examples)))
    if low_concept:
        warnings.append(DatasetValidationWarning(
            "caption_low_concept_count",
            f"Captions below {cfg.min_caption_concepts} comma-separated concepts.",
            count=len(low_concept),
            examples=_cap(low_concept, cfg.max_examples)))
    if stats["duplicate_ratio"] > cfg.max_duplicate_caption_ratio:
        warnings.append(DatasetValidationWarning(
            "caption_duplicate_dominance",
            f"Duplicate caption ratio {stats['duplicate_ratio']:.2f} exceeds "
            f"{cfg.max_duplicate_caption_ratio:.2f}.",
            count=stats["duplicate_count"]))
    if samples and stats["richness_score"] < cfg.min_caption_richness_score:
        warnings.append(DatasetValidationWarning(
            "caption_richness_low",
            f"Caption richness score {stats['richness_score']:.2f} is below "
            f"{cfg.min_caption_richness_score:.2f}.",
            count=len(samples)))


def _check_sessions(samples, sess_stats, cfg, warnings):
    n = len(samples)
    if n == 0:
        return
    if sess_stats["top_session_ratio"] > cfg.max_single_session_ratio:
        warnings.append(DatasetValidationWarning(
            "session_dominance_high",
            f"Session '{sess_stats['top_session']}' is "
            f"{sess_stats['top_session_ratio']:.0%} of the dataset "
            f"(> {cfg.max_single_session_ratio:.0%}).",
            count=int(sess_stats['top_session_ratio'] * n),
            examples=[sess_stats['top_session']]))
    if sess_stats["top_source_folder_ratio"] > cfg.max_single_source_folder_ratio:
        warnings.append(DatasetValidationWarning(
            "source_folder_dominance_high",
            f"Source folder '{sess_stats['top_source_folder']}' is "
            f"{sess_stats['top_source_folder_ratio']:.0%} of the dataset "
            f"(> {cfg.max_single_source_folder_ratio:.0%}).",
            count=int(sess_stats['top_source_folder_ratio'] * n),
            examples=[sess_stats['top_source_folder']]))
    # explicit video / capture-batch dominance when those fields exist
    for key, code, label in (("video_id", "single_video_dominance_high", "video"),
                             ("capture_date", "capture_batch_dominance_high",
                              "capture batch")):
        vals = [_first(_flat(s), key) for s in samples]
        dist = _distribution(vals)
        if dist:
            top, cnt = next(iter(dist.items()))
            if cnt / n > cfg.max_single_session_ratio:
                warnings.append(DatasetValidationWarning(
                    code, f"Single {label} '{top}' is {cnt / n:.0%} of the "
                    f"dataset (> {cfg.max_single_session_ratio:.0%}).",
                    count=cnt, examples=[str(top)]))


def _check_subject_integrity(samples, cfg, warnings):
    low, present = [], 0
    for s in samples:
        val = _subject_integrity(s)
        if val is None:
            continue
        present += 1
        if val < cfg.min_subject_integrity_score:
            low.append(_image_rel(s))
    if low:
        warnings.append(DatasetValidationWarning(
            "subject_integrity_low",
            f"Subject integrity below {cfg.min_subject_integrity_score:.2f}.",
            count=len(low), examples=_cap(low, cfg.max_examples)))
    if samples and present == 0 and cfg.require_advanced_signals:
        warnings.append(DatasetValidationWarning(
            "subject_integrity_missing_for_enabled_signal",
            "Subject integrity signal required but absent on all samples.",
            count=len(samples), severity="info"))


def _check_pose_framing(samples, cfg, warnings):
    n = len(samples)
    if n < cfg.pose_outlier_min_samples:
        if cfg.require_advanced_signals and not any(
                _pose_label(s) for s in samples):
            warnings.append(DatasetValidationWarning(
                "pose_signal_missing_for_enabled_signal",
                "Pose/framing signal required but absent on all samples.",
                count=n, severity="info"))
        return
    for getter, code, msg in (
            (_pose_label, "pose_outlier_extreme", "pose"),
            (_framing_label, "framing_outlier_extreme", "framing"),
            (_coverage_label, "coverage_outlier", "coverage")):
        by_label: dict = {}
        for s in samples:
            lbl = getter(s)
            if lbl is None or str(lbl) == "":
                continue
            by_label.setdefault(str(lbl), []).append(_image_rel(s))
        singletons = [imgs[0] for lbl, imgs in by_label.items()
                      if len(imgs) == 1]
        if singletons and len(by_label) > 1:
            warnings.append(DatasetValidationWarning(
                code, f"Rare singleton {msg} categories may hurt training "
                "consistency (review for outliers).",
                count=len(singletons),
                examples=_cap(singletons, cfg.max_examples)))


def _check_pose_angle(samples, cfg, warnings):
    from .pose_angle import compute_pose_angle_stats, pose_validation_warnings
    stats = compute_pose_angle_stats(samples)
    for code, msg, count in pose_validation_warnings(
            stats, max_dominant_pose_ratio=cfg.max_dominant_pose_ratio,
            max_dominant_angle_ratio=cfg.max_dominant_angle_ratio,
            require_signal=cfg.require_pose_signal, sample_count=len(samples)):
        sev = "info" if code == "pose_signal_missing_for_enabled_check" \
            else "warning"
        warnings.append(DatasetValidationWarning(code, msg, severity=sev,
                                                 count=count))


def _check_pose_duplicates(samples, cfg, warnings):
    n = len(samples)
    if not n:
        return
    blocks = [s.get("pose_duplicate") for s in samples
              if isinstance(s.get("pose_duplicate"), dict)]
    if not blocks:
        return
    near = [s for s in samples
            if isinstance(s.get("pose_duplicate"), dict)
            and s["pose_duplicate"].get("near_duplicate")]
    if len(near) / n > cfg.max_pose_near_duplicate_ratio:
        warnings.append(DatasetValidationWarning(
            "pose_near_duplicate_ratio_high",
            f"{len(near)}/{n} samples are pose near-duplicates "
            f"(> {cfg.max_pose_near_duplicate_ratio:.0%}).",
            count=len(near),
            examples=_cap([_image_rel(s) for s in near], cfg.max_examples)))
    # largest cluster dominance + representative integrity
    sizes: dict = {}
    rep_by_cluster: dict = {}
    for s in samples:
        b = s.get("pose_duplicate")
        if not isinstance(b, dict):
            continue
        cid = b.get("duplicate_cluster_id")
        if cid:
            sizes[cid] = sizes.get(cid, 0) + 1
            if b.get("representative"):
                rep_by_cluster[cid] = rep_by_cluster.get(cid, 0) + 1
    if sizes:
        biggest = max(sizes.values())
        if biggest / n > cfg.max_single_pose_cluster_ratio:
            warnings.append(DatasetValidationWarning(
                "pose_duplicate_cluster_high",
                f"Largest pose-duplicate cluster is {biggest}/{n} samples "
                f"(> {cfg.max_single_pose_cluster_ratio:.0%}).", count=biggest))
        missing_rep = [cid for cid in sizes if rep_by_cluster.get(cid, 0) == 0]
        if missing_rep:
            warnings.append(DatasetValidationWarning(
                "pose_duplicate_representative_missing",
                f"{len(missing_rep)} pose-duplicate cluster(s) have no "
                "representative.", count=len(missing_rep),
                examples=_cap(missing_rep, cfg.max_examples)))


def validate_exported_dataset(dataset_dir, config: DatasetValidationConfig | None
                              = None, manifest: dict | None = None,
                              manifest_path=None) -> DatasetValidationReport:
    """Validate an exported (or planned) training set. Warnings by default;
    blocks only when ``config.strict`` is set. Never raises for ordinary input.
    """
    cfg = config or DatasetValidationConfig()
    dataset_dir = Path(dataset_dir)
    man = _load_manifest(dataset_dir, manifest_path, manifest)
    samples = man.get("samples") or []
    captions = [_caption_of(s, dataset_dir) for s in samples]

    cap_stats = compute_caption_stats(captions)
    sess_stats = compute_session_stats(samples)
    warnings: list = []
    for name, fn in (
        ("orientation", lambda: _check_orientation(samples, cfg, warnings)),
        ("captions", lambda: _check_captions(samples, captions, cap_stats,
                                             cfg, warnings)),
        ("sessions", lambda: _check_sessions(samples, sess_stats, cfg, warnings)),
        ("integrity", lambda: _check_subject_integrity(samples, cfg, warnings)),
        ("pose", lambda: _check_pose_framing(samples, cfg, warnings)),
        ("pose_angle", lambda: _check_pose_angle(samples, cfg, warnings)),
        ("pose_dup", lambda: _check_pose_duplicates(samples, cfg, warnings)),
    ):
        enabled = getattr(cfg, {
            "orientation": "check_orientation",
            "captions": "check_caption_richness",
            "sessions": "check_session_dominance",
            "integrity": "check_subject_integrity",
            "pose": "check_pose_framing_outliers",
            "pose_angle": "check_pose_angle",
            "pose_dup": "check_pose_duplicates"}[name])
        if not enabled:
            continue
        try:
            fn()
        except Exception as exc:                         # never crash
            warnings.append(DatasetValidationWarning(
                f"{name}_check_error", f"validation sub-check failed: {exc}",
                severity="info"))

    metrics = {
        "caption_stats": cap_stats,
        "session_stats": sess_stats,
        "quality_summary": compute_quality_summary(samples),
    }
    blocking = [w for w in warnings if w.severity == "warning"]
    passed = True if not cfg.strict else (len(blocking) == 0)
    return DatasetValidationReport(
        dataset_path=str(dataset_dir), sample_count=len(samples),
        warnings=warnings, metrics=metrics, passed=passed)


# ---------------------------------------------------------------------------
# Reproducibility hashes
# ---------------------------------------------------------------------------
def _config_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_dataset_hash(samples: list, dataset_dir: Path, target: str,
                         version: str) -> str:
    """Deterministic hash of sorted (rel image path, file size, caption text)
    plus target and version. File bytes are never hashed (cheap for large
    libraries); a missing file contributes size 0 deterministically."""
    rows = []
    for s in samples:
        rel = _image_rel(s)
        cap = s.get("caption")
        if not isinstance(cap, str):
            cap = _caption_of(s, dataset_dir)
        size = 0
        try:
            p = Path(dataset_dir) / rel
            if p.is_file():
                size = p.stat().st_size
        except Exception:
            size = 0
        rows.append((rel, size, cap))
    rows.sort(key=lambda r: r[0])
    h = hashlib.sha256()
    for rel, size, cap in rows:
        h.update(rel.encode("utf-8")); h.update(b"\0")
        h.update(str(size).encode("utf-8")); h.update(b"\0")
        h.update(cap.encode("utf-8")); h.update(b"\0")
    h.update(f"{target}\0{version}".encode())
    return h.hexdigest()


def _git_commit(start: Path):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(start),
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return None
    return None


def _source_roots(samples: list) -> list:
    roots = set()
    for s in samples:
        src = _first(_flat(s), "source_path", "path")
        if src:
            roots.add(str(Path(str(src)).parent))
    return sorted(roots)


def build_reproducibility(samples, dataset_dir, target, version, cfg_payload):
    return {
        "generator": GENERATOR,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(Path(dataset_dir)),
        "config_hash": _config_hash(cfg_payload),
        "dataset_hash": compute_dataset_hash(samples, Path(dataset_dir),
                                             target, version),
        "source_roots": _source_roots(samples),
    }


# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------
def _frame_rows(samples, dataset_dir):
    rows = []
    for i, s in enumerate(samples):
        cap = s.get("caption")
        if not isinstance(cap, str):
            cap = _caption_of(s, dataset_dir)
        rows.append({
            "index": i,
            "image": _image_rel(s),
            "caption": cap,
            "source": str(_first(_flat(s), "source_path", "path", default="")),
            "session": _derive_session(s),
            "orientation": _orientation_label(s),
            "quality": _subject_integrity(s),
        })
    return rows


def _render_markdown(card: dict, cfg: DatasetCardConfig) -> str:
    s = card["summary"]
    rep = card.get("reproducibility", {})
    cap = card["caption_summary"]
    div = card["diversity_summary"]
    q = card["quality_summary"]
    L = []
    L.append(f"# Dataset Card: {s['target']}/{s['version']}")
    L.append("")
    L.append("## Summary")
    L.append(f"- Target: {s['target']}")
    L.append(f"- Version: {s['version']}")
    L.append(f"- Created: {s['created_at']}")
    L.append(f"- Sample count: {s['sample_count']}")
    L.append(f"- Dry run: {s['dry_run']}")
    L.append(f"- Source root(s): {', '.join(rep.get('source_roots') or []) or 'n/a'}")
    L.append("")
    if cfg.include_reproducibility:
        L.append("## Reproducibility")
        L.append(f"- Generator: {rep.get('generator')}")
        L.append(f"- Git commit: {rep.get('git_commit') or 'n/a'}")
        L.append(f"- Manifest path: {card['summary'].get('manifest_path') or 'n/a'}")
        L.append(f"- Config hash: {rep.get('config_hash')}")
        L.append(f"- Dataset hash: {rep.get('dataset_hash')}")
        L.append("")
    L.append("## Caption Summary")
    L.append(f"- Empty captions: {cap['empty_captions']}")
    L.append(f"- Average token count: {cap['avg_token_count']}")
    L.append(f"- Average concept count: {cap['avg_concept_count']}")
    L.append(f"- Duplicate caption ratio: {cap['duplicate_ratio']}")
    L.append(f"- Caption richness score: {cap['richness_score']}")
    L.append("")
    L.append("## Diversity Summary")
    L.append(f"- Orientation distribution: {div['orientation_distribution']}")
    L.append(f"- Coverage/framing distribution: {div['framing_distribution']}")
    L.append(f"- Pose distribution: {div['pose_distribution']}")
    L.append(f"- Session/source distribution: {div['session_distribution']}")
    L.append("")
    L.append("## Quality Summary")
    L.append(f"- Global quality average: {q['global_quality_avg']}")
    L.append(f"- Subject integrity average: {q['subject_integrity_avg']}")
    L.append(f"- Regional quality average: {q['regional_quality_avg']}")
    L.append(f"- Natural texture average: {q['natural_texture_avg']}")
    L.append(f"- Advanced quality signal coverage: {q['advanced_signal_coverage']}")
    L.append("")
    rq = card.get("region_quality_summary") or {}
    if any(rq.get(k) is not None for k in ("avg_face_region_score",
            "avg_torso_region_score", "avg_full_body_region_score")):
        L.append("## Region Quality Summary")
        L.append("| Metric | Value |")
        L.append("|---|---:|")
        L.append(f"| Avg face region score | {rq.get('avg_face_region_score')} |")
        L.append(f"| Avg torso region score | {rq.get('avg_torso_region_score')} |")
        L.append(f"| Avg full-body region score | {rq.get('avg_full_body_region_score')} |")
        L.append(f"| Avg region balance score | {rq.get('avg_region_balance_score')} |")
        bud = rq.get("best_training_use_distribution") or {}
        if bud:
            L.append(f"| Best training use distribution | {bud} |")
        L.append("")
    cv = card.get("coverage_framing_summary") or {}
    if cv.get("samples_with_coverage_signal"):
        L.append("## Coverage & Framing Summary")
        L.append("| Metric | Value |")
        L.append("|---|---:|")
        L.append(f"| Unknown coverage count | {cv.get('unknown_coverage_count')} |")
        L.append(f"| Needs-review count | {cv.get('needs_review_count')} |")
        L.append(f"| Dominant framing label | {cv.get('dominant_framing_label')} |")
        L.append(f"| Dominant framing ratio | {cv.get('dominant_framing_ratio')} |")
        if cv.get("framing_distribution"):
            L.append(f"| Framing distribution | {cv['framing_distribution']} |")
        L.append("")
    pd = card.get("pose_duplicate_summary") or {}
    if pd.get("pose_signature_count"):
        L.append("## Pose Duplicate Summary")
        L.append("| Metric | Value |")
        L.append("|---|---:|")
        L.append(f"| Pose signature coverage | {pd.get('pose_signature_coverage_ratio')} |")
        L.append(f"| Near-duplicate count | {pd.get('near_duplicate_count')} |")
        L.append(f"| Near-duplicate ratio | {pd.get('near_duplicate_ratio')} |")
        L.append(f"| Duplicate cluster count | {pd.get('duplicate_cluster_count')} |")
        L.append(f"| Largest cluster size | {pd.get('largest_cluster_size')} |")
        L.append("")
    if cfg.include_validation and card.get("validation"):
        L.append("## Validation Warnings")
        vw = card["validation"].get("warnings", [])
        if not vw:
            L.append("- None")
        for w in vw:
            L.append(f"- {w['code']} (count={w['count']}, "
                     f"severity={w['severity']}): {w['message']}")
            if w.get("examples"):
                L.append(f"  - examples: {', '.join(map(str, w['examples']))}")
        L.append("")
    if cfg.include_frame_list:
        L.append("## Frame List")
        L.append("| # | Image | Caption | Source | Session | Orientation | Quality |")
        L.append("|---|-------|---------|--------|---------|-------------|---------|")
        rows = card.get("frames", [])
        shown = rows[:cfg.max_frame_rows_markdown]
        for r in shown:
            cap_txt = (r["caption"] or "").replace("\n", " ").replace("|", "/")
            if len(cap_txt) > 80:
                cap_txt = cap_txt[:77] + "..."
            src = Path(r["source"]).name if r["source"] else ""
            L.append(f"| {r['index']} | {r['image']} | {cap_txt} | {src} | "
                     f"{r['session']} | {r['orientation']} | "
                     f"{r['quality'] if r['quality'] is not None else ''} |")
        if len(rows) > len(shown):
            L.append("")
            L.append(f"_Frame list truncated to {len(shown)} of {len(rows)} "
                     "rows; see dataset_card.json for the full list._")
        L.append("")
    L.append("## Training Notes")
    L.append(f"- Recommended use: {s['target']} LoRA training set.")
    L.append("- Review notes: address any validation warnings above before training.")
    L.append("")
    return "\n".join(L)


def _build_card(samples, dataset_dir, manifest, cfg, validation_report,
                manifest_path):
    target = manifest.get("target", "unknown")
    version = manifest.get("version", "v000")
    dry_run = bool(manifest.get("dry_run", False))
    captions = [s.get("caption") if isinstance(s.get("caption"), str)
                else _caption_of(s, dataset_dir) for s in samples]
    cap_stats = compute_caption_stats(captions)
    diversity = compute_diversity_stats(samples, dataset_dir)
    quality = compute_quality_summary(samples)
    repro = build_reproducibility(
        samples, dataset_dir, target, version,
        {"target": target, "version": version, "card_version": CARD_VERSION})
    summary = {
        "target": target, "version": version,
        "created_at": manifest.get("created_at")
        or datetime.now().isoformat(timespec="seconds"),
        "sample_count": len(samples), "dry_run": dry_run,
        "source_root": (repro["source_roots"] or [None])[0],
        "manifest_path": str(manifest_path) if manifest_path else
        manifest.get("manifest_path"),
    }
    try:
        from .region_readiness import summarize_region_readiness
        region_summary = summarize_region_readiness(samples)
    except Exception:                                    # never break the card
        region_summary = {}
    try:
        from .coverage_intensity import summarize_coverage_intensity
        coverage_summary = summarize_coverage_intensity(samples)
    except Exception:                                    # never break the card
        coverage_summary = {}
    try:
        from .pose_duplicate_detector import summarize_pose_duplicates
        pd_rows = []
        for s in samples:
            pd = s.get("pose_duplicate") or {}
            sig_id = (s.get("signals") or {}).get("pose_signature_id") \
                or s.get("pose_signature_id")
            pd_rows.append({
                "pose_signature_id": sig_id,
                "near_duplicate": pd.get("near_duplicate"),
                "duplicate_cluster_id": pd.get("duplicate_cluster_id"),
                "representative": pd.get("representative", True)})
        pose_dup_summary = summarize_pose_duplicates(
            pd_rows, sample_count=len(samples))
    except Exception:                                    # never break the card
        pose_dup_summary = {}
    card = {
        "card_version": CARD_VERSION,
        "summary": summary,
        "reproducibility": repro,
        "caption_summary": cap_stats,
        "diversity_summary": diversity,
        "quality_summary": quality,
        "region_quality_summary": region_summary,
        "coverage_framing_summary": coverage_summary,
        "pose_duplicate_summary": pose_dup_summary,
        "frames": _frame_rows(samples, dataset_dir) if cfg.include_frame_list
        else [],
    }
    if cfg.include_validation:
        if validation_report is None:
            validation_report = validate_exported_dataset(
                dataset_dir, manifest=manifest)
        card["validation"] = (validation_report.to_dict()
                              if isinstance(validation_report,
                                            DatasetValidationReport)
                              else validation_report)
    return card


def _structured_manifest(card, manifest, samples) -> dict:
    return {
        "schema_version": STRUCTURED_SCHEMA,
        "target": card["summary"]["target"],
        "version": card["summary"]["version"],
        "created_at": card["summary"]["created_at"],
        "generator": GENERATOR,
        "dry_run": card["summary"]["dry_run"],
        "sample_count": card["summary"]["sample_count"],
        "samples": samples,
        "caption_stats": card["caption_summary"],
        "diversity_stats": card["diversity_summary"],
        "orientation_stats": compute_orientation_stats(samples),
        "session_stats": compute_session_stats(samples),
        "pose_stats": card["diversity_summary"].get("pose_angle_stats", {}),
        "quality_summary": card["quality_summary"],
        "region_quality_summary": card.get("region_quality_summary", {}),
        "coverage_framing_summary": card.get("coverage_framing_summary", {}),
        "pose_duplicate_summary": card.get("pose_duplicate_summary", {}),
        "validation_report": card.get("validation", {}),
        "reproducibility": card["reproducibility"],
    }


def generate_dataset_card(dataset_dir, manifest_path=None,
                          config: DatasetCardConfig | None = None,
                          validation_report=None, *, manifest: dict | None = None,
                          out_dir=None, name_prefix: str = "",
                          write_files: bool = True) -> dict:
    """Generate ``dataset_card.md`` / ``dataset_card.json`` and a companion
    ``dataset_manifest.structured.json`` for an exported (or planned) dataset.

    Additive and non-destructive: it reads ``dataset_manifest.json`` but never
    modifies it. ``out_dir``/``name_prefix`` let dry-run callers write companion
    files into a separate planning directory. Returns the card dict (and, under
    ``"_paths"``, any files written). Never raises for ordinary input.
    """
    cfg = config or DatasetCardConfig()
    dataset_dir = Path(dataset_dir)
    man = _load_manifest(dataset_dir, manifest_path, manifest)
    samples = man.get("samples") or []
    card = _build_card(samples, dataset_dir, man, cfg, validation_report,
                       manifest_path)
    structured = _structured_manifest(card, man, samples)
    card["_structured_manifest"] = structured

    if not write_files:
        return card

    target_dir = Path(out_dir) if out_dir else dataset_dir
    paths = {}
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        pre = f"{name_prefix}_" if name_prefix else ""
        json_p = target_dir / f"{pre}dataset_card.json"
        md_p = target_dir / f"{pre}dataset_card.md"
        struct_p = target_dir / f"{pre}dataset_manifest.structured.json"
        card_out = {k: v for k, v in card.items() if k != "_structured_manifest"}
        json_p.write_text(json.dumps(card_out, indent=2, default=str),
                          encoding="utf-8")
        md_p.write_text(_render_markdown(card, cfg), encoding="utf-8")
        struct_p.write_text(json.dumps(structured, indent=2, default=str),
                            encoding="utf-8")
        paths = {"dataset_card_json": str(json_p),
                 "dataset_card_md": str(md_p),
                 "dataset_manifest_structured_json": str(struct_p)}
    except Exception as exc:                              # never crash export
        paths = {"error": f"card write failed: {exc}"}
    card["_paths"] = paths
    return card


__all__ = [
    "DatasetValidationConfig", "DatasetValidationWarning",
    "DatasetValidationReport", "DatasetCardConfig",
    "validate_exported_dataset", "generate_dataset_card",
    "compute_dataset_hash", "compute_caption_stats", "compute_diversity_stats",
    "compute_quality_summary", "build_reproducibility",
]
