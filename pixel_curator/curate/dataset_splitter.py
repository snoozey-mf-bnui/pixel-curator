"""Deliverable 1 - standalone dataset splitter and classifier.

Turns a large local personal media archive (Google Photos Takeout-style) into
versioned, training-ready dataset splits for the private self-modeling
workflow. This module is UI-free and self-contained: it discovers samples,
classifies them into professional training targets, generates professional
catalog captions, and plans non-destructive versioned exports.

Design principles:
  * local-first, privacy-preserving, non-destructive
  * dry-run / planning by default; never copies, moves, or modifies source media
  * resume-safe: never overwrites an existing export or source file
  * graceful: corrupt sidecars, missing files, and videos warn and continue
  * no heavy computer-vision dependencies; deterministic metadata classification
  * if an existing project classifier is importable it is used defensively as a
    first automatic pass, then normalized into this module's target space

Pure stdlib.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .personal_content_analyzer import (PersonalAnalyzerConfig, coerce_config,
                                        enrich_sample)

log = logging.getLogger("redresser.dataset_splitter")

# ---------------------------------------------------------------------------
# Canonical training targets + professional aliases.
# ---------------------------------------------------------------------------
TARGETS: tuple[str, ...] = (
    "core_identity",
    "outfit_clothing",
    "style_aesthetic",
    "pose_action",
    "garment_detail",
    "catalog_reference",
    "needs_review",
    "excluded",
)

#: Targets that receive a versioned export folder.
EXPORTABLE_TARGETS: tuple[str, ...] = (
    "core_identity", "outfit_clothing", "style_aesthetic", "pose_action",
    "garment_detail", "catalog_reference",
)

TARGET_ALIASES: dict[str, str] = {
    "identity": "core_identity",
    "core_identity": "core_identity",
    "outfit": "outfit_clothing",
    "clothing": "outfit_clothing",
    "outfit_clothing": "outfit_clothing",
    "style": "style_aesthetic",
    "aesthetic": "style_aesthetic",
    "style_aesthetic": "style_aesthetic",
    "pose": "pose_action",
    "action": "pose_action",
    "pose_action": "pose_action",
    "specialized": "garment_detail",
    "garment": "garment_detail",
    "garment_detail": "garment_detail",
    "catalog": "catalog_reference",
    "catalog_reference": "catalog_reference",
    "review": "needs_review",
    "needs_review": "needs_review",
    "excluded": "excluded",
}

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".heic")
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".m4v")

# ---------------------------------------------------------------------------
# Professional caption defaults (SFW, catalog terminology).
# ---------------------------------------------------------------------------
CAPTION_DEFAULTS: dict[str, str] = {
    "core_identity": (
        "personal_identity_anchor, professional catalog reference, clear "
        "facial structure, consistent identity, studio-ready framing"),
    "outfit_clothing": (
        "personal_identity_anchor, wardrobe variation, fashion catalog "
        "styling, refined garment rendering, clean fabric boundaries"),
    "style_aesthetic": (
        "personal_identity_anchor, catalog photography style, controlled "
        "studio lighting, polished editorial composition"),
    "pose_action": (
        "personal_identity_anchor, fashion model pose, balanced body "
        "proportions, catalog-ready composition"),
    "garment_detail": (
        "personal_identity_anchor, garment detail study, refined textile "
        "structure, fabric interaction, clean garment edges"),
    "catalog_reference": (
        "personal_identity_anchor, fashion catalog reference, polished studio "
        "presentation, catalog-ready image quality"),
    "needs_review": (
        "personal_identity_anchor, review candidate, catalog dataset review "
        "required"),
    "excluded": "excluded dataset candidate",
}

# ---------------------------------------------------------------------------
# Optional existing classifier (used defensively as a first automatic pass).
# ---------------------------------------------------------------------------
try:                                                # pragma: no cover
    from ..train.dataset_classifier import (
        classify_training_sample as _existing_classifier)
except Exception:                                   # pragma: no cover
    _existing_classifier = None


# ---------------------------------------------------------------------------
# Standalone helpers.
# ---------------------------------------------------------------------------
def normalize_lora_target(value: str | None) -> tuple[str | None, list[str]]:
    """Normalize a target / alias to a canonical label.

    Returns ``(canonical_or_None, warnings)``. An unrecognised value yields
    ``(None, [warning])`` so callers can route it to ``needs_review``.
    """
    if value is None:
        return None, []
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if token in TARGET_ALIASES:
        return TARGET_ALIASES[token], []
    return None, [f"Unknown lora_target '{value}'; routed to needs_review."]


def load_sidecar_metadata(path: Path) -> tuple[dict, list[str]]:
    """Load ``<stem>.json`` sidecar next to an image. Never raises.

    Returns ``(metadata, warnings)``. Missing sidecar -> ``({}, [])``; corrupt
    JSON -> ``({}, [warning])``.
    """
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return {}, []
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}, [f"Sidecar metadata ignored (not an object): {sidecar}"]
        return data, []
    except Exception as exc:
        return {}, [f"Corrupt sidecar metadata ignored ({exc}): {sidecar}"]


def build_training_caption(target: str, metadata: dict | None = None) -> str:
    """Return a professional caption for ``target``.

    Uses the enhanced style-aware captioner when available (folding in any
    auto-tags / WD14 tags), falling back to the target-specific default. A
    valid manual ``caption`` in ``metadata`` is always preserved verbatim.
    """
    metadata = metadata or {}
    override = metadata.get("caption")
    if isinstance(override, str) and override.strip():
        return override.strip()
    try:
        from ..caption.enhance import build_caption as _enhanced
        return _enhanced(target, metadata,
                         wd14_tags=metadata.get("wd14_tags"),
                         style_aware=True)["caption"]
    except Exception:                                # graceful fallback
        return CAPTION_DEFAULTS.get(target, CAPTION_DEFAULTS["needs_review"])


def auto_tags_for_sample(metadata: dict | None = None) -> list[str]:
    """Professional SFW auto-tags for a sample (enhancer when available)."""
    try:
        from ..caption.enhance import auto_tag
        return auto_tag(metadata or {}, (metadata or {}).get("wd14_tags"))
    except Exception:                                # pragma: no cover
        return []


def _to_float(meta: dict, key: str, default: float = 0.0) -> float:
    try:
        val = meta.get(key)
        return float(val) if val is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _face_visibility(meta: dict) -> float:
    raw = meta.get("face_visibility")
    if isinstance(raw, str):
        return {"clear": 0.9, "visible": 0.8, "partial": 0.45,
                "none": 0.0}.get(raw.strip().lower(), 0.0)
    return _to_float(meta, "face_visibility", 0.0)


def _collect_tags(meta: dict) -> set[str]:
    tags: set[str] = set()
    for key in ("tags", "review_tags"):
        val = meta.get(key)
        if isinstance(val, (list, tuple, set)):
            tags.update(str(t).strip().lower() for t in val)
        elif isinstance(val, str):
            tags.update(t.strip().lower() for t in val.split(",") if t.strip())
    return tags


def _infer_source_type(path: str | Path) -> str:
    ext = Path(str(path)).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return "image"


# Optional metadata carried into the export manifest *only when present*, so the
# Priority-4 dataset card / validator can audit orientation, session balance,
# subject integrity and pose/framing without re-reading source media. Absent
# keys are omitted -> manifests for plain datasets are byte-for-byte unchanged.
_CARD_FIELD_KEYS = (
    "source_session", "session_id", "source_id", "source_folder", "video_id",
    "capture_date", "orientation", "canonical_orientation", "orientation_state",
    "exif_orientation", "exif_normalized", "orientation_normalized",
    "orientation_legacy", "legacy_orientation_path",
    "quality_score", "subject_integrity_score", "figure_integrity_score",
    "integrity_score", "regional_quality_score", "texture_preservation_score",
    "pose_bucket", "pose_type", "framing", "framing_label", "coverage_label",
    # priority-6 region readiness (review-only enrichment; never routes)
    "face_region_score", "torso_region_score", "full_body_region_score",
    "region_balance_score", "best_training_use",
    # priority-7 coverage / framing intensity (review-only; never routes)
    "framing_label", "coverage_score", "training_use_hint",
    "coverage_review_recommended",
)


def _carry_card_fields(meta: dict | None) -> dict:
    meta = meta or {}
    out = {k: meta[k] for k in _CARD_FIELD_KEYS
           if k in meta and meta[k] is not None}
    # carry structured payloads verbatim when present
    for nest in ("region_readiness", "coverage_intensity"):
        if isinstance(meta.get(nest), dict):
            out[nest] = meta[nest]
    return out


def _deterministic_classify(meta: dict, source_type: str
                            ) -> tuple[str, float, list[str]]:
    """Deterministic metadata-based classification. Never raises."""
    reasons: list[str] = []
    tags = _collect_tags(meta)
    quality = _to_float(meta, "quality_score", 0.5)
    sharp = _to_float(meta, "sharpness", quality)
    bright = _to_float(meta, "brightness", _to_float(meta, "exposure", 0.5))
    identity = _to_float(meta, "identity_similarity", 0.0)
    garment = _to_float(meta, "garment_visibility", 0.0)
    face = _face_visibility(meta)
    framing = str(meta.get("framing", "")).strip().lower()
    pose = str(meta.get("pose_type", "")).strip().lower()

    # 1) hard exclusions
    if (meta.get("rejected") or meta.get("unusable") or meta.get("corrupt")
            or meta.get("missing") or meta.get("duplicate")
            or meta.get("is_duplicate") or "excluded" in tags
            or "duplicate" in tags or "rejected" in tags or quality < 0.1):
        reasons.append("metadata_excluded")
        return "excluded", 0.95, reasons

    # 2) quality / completeness gate -> needs_review (holds, never deletes)
    if quality < 0.35:
        reasons.append("quality_score_low")
        return "needs_review", 0.5, reasons
    if sharp < 0.30:
        reasons.append("sharpness_low")
        return "needs_review", 0.5, reasons
    if bright < 0.15 or bright > 0.9:
        reasons.append("exposure_out_of_range")
        return "needs_review", 0.5, reasons
    if "review" in tags or "uncertain" in tags:
        reasons.append("user_review_flagged")
        return "needs_review", 0.5, reasons

    # 3) explicit detail / textile tags
    if tags & {"garment_detail", "textile", "fabric", "fabric_closeup",
               "detail_study", "garment_closeup"}:
        reasons.append("tag_garment_detail")
        return "garment_detail", 0.85, reasons

    # 4) core identity
    if identity >= 0.7 and face >= 0.6 and framing in (
            "", "face", "face_closeup", "headshot", "portrait",
            "upper_body", "clean_reference"):
        reasons += ["identity_similarity_strong", "face_visibility_clear",
                    "quality_score_high"]
        return "core_identity", round(min(0.99, 0.6 + 0.3 * identity), 2), \
            reasons

    # 5) garment detail by framing + visibility
    if garment >= 0.6 and framing in ("close_up", "closeup", "garment_closeup",
                                      "detail"):
        reasons += ["garment_visibility_high", "close_up_framing"]
        return "garment_detail", round(min(0.95, 0.55 + 0.3 * garment), 2), \
            reasons

    # 6) pose / action - an identified whole-body pose routes here BEFORE the
    #    generic outfit bucket, so full-body shots are not all swallowed as
    #    clothing. Only fires when a pose signal is present (pose_type / tag);
    #    framing-only samples fall through to outfit exactly as before.
    if pose in ("standing", "sitting", "dynamic", "walking", "model",
                "action") or tags & {"pose", "action", "pose_action",
                                      "full_body_movement"}:
        reasons.append("pose_type_identified")
        if framing in ("full_body", "half_body"):
            reasons.append(f"{framing}_framing")
        return "pose_action", 0.72, reasons

    # 7) outfit / clothing
    if garment >= 0.4 or tags & {"wardrobe", "outfit", "clothing",
                                 "outfit_clothing"}:
        reasons += ["garment_visibility_moderate"]
        if framing in ("full_body", "half_body"):
            reasons.append(f"{framing}_framing")
        return "outfit_clothing", round(min(0.92, 0.5 + 0.3 * garment), 2), \
            reasons

    # 8) style / aesthetic
    if tags & {"style", "aesthetic", "lighting", "studio", "color_grade",
               "background", "composition", "editorial"} \
            or bool(meta.get("style_reference")):
        reasons += ["style_aesthetic_signal"]
        return "style_aesthetic", 0.68, reasons

    # 9) catalog reference when quality is strong but target is unclear
    if quality >= 0.6 and (bool(meta.get("catalog_suitable"))
                           or framing in ("catalog", "full_body", "half_body")
                           or str(meta.get("source_type", "")) == "image"):
        reasons += ["quality_score_high", "catalog_reference_candidate"]
        return "catalog_reference", 0.6, reasons

    # 10) default: hold for review
    reasons.append("insufficient_signal")
    return "needs_review", 0.4, reasons


def _flatten_for_classifier(sample: dict) -> dict:
    """Merge top-level recognised keys with a nested ``metadata`` block."""
    meta = {k: v for k, v in sample.items()
            if k not in ("metadata", "warnings", "sample_path", "source_type")}
    meta.update(sample.get("metadata") or {})
    return meta


def classify_training_sample(sample: dict) -> dict:
    """Classify one sample dict into a canonical training target.

    Manual ``lora_target`` (top-level or inside ``metadata``) always wins.
    Videos route to ``needs_review`` unless a valid manual override is given.
    Returns the result shape with ``sample_path``, ``source_type``,
    ``lora_target``, ``confidence``, ``reason_codes``, ``metadata`` and
    ``warnings``. Never raises for ordinary input.
    """
    warnings: list[str] = list(sample.get("warnings", []))
    sample_path = sample.get("sample_path") or sample.get("path") or ""
    meta = _flatten_for_classifier(sample)
    source_type = (sample.get("source_type")
                   or meta.get("source_type")
                   or _infer_source_type(sample_path))

    def _result(target, confidence, reasons):
        return {
            "sample_path": str(sample_path),
            "source_type": source_type,
            "lora_target": target,
            "confidence": round(float(confidence), 2),
            "reason_codes": list(reasons),
            "metadata": sample.get("metadata") or {},
            "warnings": warnings,
        }

    # 1) manual override (highest priority)
    raw_override = meta.get("lora_target")
    if raw_override is not None:
        norm, w = normalize_lora_target(raw_override)
        warnings += w
        if norm is not None:
            return _result(norm, 1.0, ["manual_override"])
        return _result("needs_review", 0.5, ["manual_override_invalid"])

    # 2) videos are recorded but not processed here
    if source_type == "video":
        return _result("needs_review", 0.5, ["source_type_video"])

    # 3) defensive existing-classifier first pass, normalized into our targets
    if _existing_classifier is not None:
        try:
            ext = _existing_classifier(dict(meta))
            ext_norm, _w = normalize_lora_target(ext.get("lora_target"))
            if ext_norm and ext_norm not in ("needs_review", "excluded"):
                reasons = list(ext.get("reason_codes", [])) + \
                    ["existing_classifier"]
                return _result(ext_norm, ext.get("confidence", 0.7), reasons)
        except Exception as exc:                    # pragma: no cover
            warnings.append(f"Existing classifier unavailable ({exc}); "
                            "using metadata-based dataset splitting.")

    # 4) deterministic metadata classification
    target, conf, reasons = _deterministic_classify(meta, source_type)
    return _result(target, conf, reasons + ["metadata_rules"])


# ---------------------------------------------------------------------------
# Versioned export.
# ---------------------------------------------------------------------------
def _next_version(output_root: Path, target: str) -> str:
    n = 1
    while (output_root / f"v{n:03d}_{target}").exists():
        n += 1
    return f"v{n:03d}"


def annotate_pose_duplicates(sample_rows: list[dict], config=None) -> int:
    """Dataset-level pose near-duplicate pass over already-built manifest sample
    rows. Non-destructive: it only *annotates* each row with a ``pose_duplicate``
    block (and a ``pose_near_duplicate_review`` hint when flagged); it never
    removes, reorders or hard-filters samples. Returns the near-duplicate count.
    Never raises."""
    try:
        from .pose_duplicate_detector import (detect_pose_duplicates,
                                              PoseDuplicateConfig)
    except Exception:
        return 0
    cfg = config or PoseDuplicateConfig()
    # feed the detector a view exposing both top-level row fields and any
    # nested `signals` block (keypoints / pose metadata may live in either).
    views = []
    for r in sample_rows:
        v = dict(r)
        if isinstance(r.get("signals"), dict):
            for k, val in r["signals"].items():
                v.setdefault(k, val)
        views.append(v)
    near = 0
    try:
        results = detect_pose_duplicates(views, cfg)
    except Exception:
        return 0
    for row, res in zip(sample_rows, results):
        rd = res.to_dict()
        row["pose_duplicate"] = {
            "near_duplicate": rd["near_duplicate"],
            "duplicate_cluster_id": rd["duplicate_cluster_id"],
            "duplicate_rank": rd["duplicate_rank"],
            "representative": rd["representative"],
            "pose_duplicate_score": rd["pose_duplicate_score"],
            "similar_sample_ids": rd["similar_sample_ids"],
        }
        if rd["near_duplicate"]:
            near += 1
            row.setdefault("warnings", []).append("pose_near_duplicate_review")
    return near


def export_lora_target_dataset(samples: list[dict], target: str,
                               output_root, version: str | None = None,
                               dry_run: bool = True, copy_mode: str = "copy",
                               resume: bool = True,
                               write_card: bool = True,
                               pose_dedupe: bool = False,
                               manifest_base=None,
                               register_version: bool = True,
                               version_note: str = "",
                               parent_version: int | None = None) -> dict:
    """Plan (or, when ``dry_run`` is False, perform) a versioned export.

    Layout: ``<output_root>/<version>_<target>/{images,captions,
    dataset_manifest.json}``. Non-destructive: dry-run copies nothing and
    writes only a planning manifest under ``<output_root>/dry_run_manifests/``;
    a real run copies/symlinks images and writes one ``.txt`` caption each,
    never overwriting existing files. Missing files are skipped with warnings.

    Dataset Versioning (Phase 2): a real (non-dry-run) export always writes a
    ``version.json`` sidecar in the export folder, and - when
    ``manifest_base`` (the project output base holding the SQLite manifest)
    is provided and ``register_version`` is True - registers itself in the
    shared version registry so ``datasets`` / ``history`` / diffs see it.
    Registration failures degrade to warnings; the export itself never fails
    because of versioning.
    """
    output_root = Path(output_root).expanduser()
    warnings: list[str] = []
    if copy_mode not in ("copy", "symlink"):
        warnings.append(f"Unknown copy_mode '{copy_mode}'; using 'copy'.")
        copy_mode = "copy"

    if version is None:
        version = _next_version(output_root, target)
    elif not version.lower().startswith("v"):
        version = f"v{version}"

    version_dir = output_root / f"{version}_{target}"

    # resume / no-overwrite policy for an existing version
    already: set[str] = set()
    if version_dir.exists() and not dry_run:
        if resume:
            existing_manifest = version_dir / "dataset_manifest.json"
            if existing_manifest.is_file():
                try:
                    prev = json.loads(existing_manifest.read_text())
                    already = {Path(p).name for p in
                               prev.get("planned_images", [])}
                except Exception:
                    already = set()
            already |= {p.name for p in (version_dir / "images").glob("*")
                        if p.is_file()}
            warnings.append(
                f"Existing dataset version present; resume preserves prior "
                f"export and skips {len(already)} sample(s): {version_dir}")
        else:
            version = _next_version(output_root, target)
            version_dir = output_root / f"{version}_{target}"
            warnings.append(
                f"Existing version not overwritten; created new version "
                f"{version}.")

    planned_images: list[str] = []
    planned_captions: list[str] = []
    sample_rows: list[dict] = []
    skipped_missing = 0
    skipped_existing = 0

    for row in samples or []:
        src = row.get("sample_path") or row.get("path") or ""
        meta = row.get("metadata") or {}
        caption = build_training_caption(target, meta)
        src_path = Path(str(src)).expanduser()
        stem = src_path.stem or f"sample_{len(sample_rows):04d}"
        suffix = src_path.suffix or ".png"
        img_rel = f"images/{stem}{suffix}"
        cap_rel = f"captions/{stem}.txt"
        exists = bool(src) and src_path.is_file()
        if not exists:
            skipped_missing += 1
            warnings.append(f"Source file missing, skipped: {src or '<empty>'}")
            continue
        if (stem + suffix) in already:
            skipped_existing += 1
            continue
        planned_images.append(img_rel)
        planned_captions.append(cap_rel)
        entry = {
            "source_path": str(src_path),
            "image_target": img_rel,
            "caption_target": cap_rel,
            "caption": caption,
            "caption_source": ("manual" if meta.get("caption")
                               else "style_aware"),
            "auto_tags": auto_tags_for_sample(meta),
            "source_type": row.get("source_type",
                                   _infer_source_type(src_path)),
            "lora_target": target,
            "confidence": row.get("confidence"),
            "reason_codes": row.get("reason_codes", []),
        }
        # Secondary suitability metadata, carried only when the optional
        # analyzer produced it (i.e. enabled). Absent -> manifest unchanged.
        if "personal_suitability" in row:
            entry["personal_suitability"] = row["personal_suitability"]
        if isinstance(row.get("personal_analyzer"), dict):
            entry["personal_analyzer"] = row["personal_analyzer"]
        # carry optional audit signals (orientation/session/quality/pose/region)
        # when the metadata provides them; omitted entirely otherwise.
        card_fields = _carry_card_fields(meta)
        if card_fields:
            entry["signals"] = card_fields
            # review-only region hint: surfaces best_training_use without ever
            # changing lora_target. Flag (don't resolve) any disagreement.
            rr = card_fields.get("region_readiness") or {}
            hint = card_fields.get("best_training_use") or rr.get(
                "best_training_use")
            if hint:
                try:
                    from .region_readiness import training_use_hint as _tuh
                    entry["training_use_hint"] = _tuh(
                        {"best_training_use": hint})
                except Exception:
                    entry["training_use_hint"] = "review"
                _hint_target = {"identity": "core_identity",
                                "outfit": "outfit_clothing",
                                "pose": "pose_action"}.get(
                                    entry["training_use_hint"])
                if _hint_target and _hint_target != target:
                    entry.setdefault("warnings", []).append(
                        "coverage_hint_differs_from_lora_target")
        sample_rows.append(entry)

    pose_near_dupes = 0
    if pose_dedupe and sample_rows:
        pose_near_dupes = annotate_pose_duplicates(sample_rows)

    manifest = {
        "target": target,
        "version": version,
        "dry_run": bool(dry_run),
        "copy_mode": copy_mode,
        "pose_near_duplicates": pose_near_dupes,
        "sample_count": len(sample_rows),
        "samples": sample_rows,
        "planned_images": planned_images,
        "planned_captions": planned_captions,
        "source_paths": [r["source_path"] for r in sample_rows],
        "skipped_missing": skipped_missing,
        "skipped_existing": skipped_existing,
        "export_dir": str(version_dir),
        "warnings": warnings,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    if dry_run:
        manifest["status"] = "planned"
        # planning manifest goes to a separate, non-destructive location
        plan_dir = output_root / "dry_run_manifests"
        try:
            plan_dir.mkdir(parents=True, exist_ok=True)
            plan_path = plan_dir / f"{version}_{target}.json"
            plan_path.write_text(json.dumps(manifest, indent=2),
                                 encoding="utf-8")
            manifest["manifest_path"] = str(plan_path)
        except Exception as exc:                    # pragma: no cover
            warnings.append(f"Could not write dry-run manifest: {exc}")
        if write_card:
            _write_dataset_card_safe(manifest, version_dir, warnings,
                                     out_dir=plan_dir,
                                     name_prefix=f"{version}_{target}")
        return manifest

    # real export
    (version_dir / "images").mkdir(parents=True, exist_ok=True)
    (version_dir / "captions").mkdir(parents=True, exist_ok=True)
    copied = 0
    for r in sample_rows:
        dest_img = version_dir / r["image_target"]
        dest_cap = version_dir / r["caption_target"]
        if not dest_img.exists():                   # never overwrite
            try:
                if copy_mode == "symlink":
                    os.symlink(r["source_path"], dest_img)
                else:
                    shutil.copy2(r["source_path"], dest_img)
                copied += 1
            except Exception as exc:                # pragma: no cover
                warnings.append(f"Export failed for {r['source_path']}: {exc}")
        if not dest_cap.exists():
            dest_cap.write_text(r["caption"], encoding="utf-8")
    manifest["copied_images"] = copied
    manifest["status"] = "exported"
    (version_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(version_dir / "dataset_manifest.json")
    if write_card:
        _write_dataset_card_safe(manifest, version_dir, warnings)
    _register_export_version_safe(manifest, version_dir, target, version,
                                  warnings, manifest_base, register_version,
                                  version_note, parent_version)
    return manifest


def _register_export_version_safe(manifest: dict, version_dir: Path,
                                  target: str, version: str,
                                  warnings: list[str], manifest_base,
                                  register: bool, note: str,
                                  parent_version: int | None) -> None:
    """Phase 2 versioning hook for real exports. Never raises."""
    try:
        from ..dataset_versioning import (
            KIND_LORA_TARGET_EXPORT, register_folder_export,
            write_version_sidecar)
        files = [{"rel_path": r["image_target"],
                  "source_path": r.get("source_path"),
                  "caption": r.get("caption")}
                 for r in manifest.get("samples", [])]
        params = {"target": target, "folder_version": version,
                  "copy_mode": manifest.get("copy_mode"),
                  "pose_near_duplicates": manifest.get(
                      "pose_near_duplicates", 0)}
        if manifest_base and register:
            record = register_folder_export(
                manifest_base, kind=KIND_LORA_TARGET_EXPORT,
                name=f"{target}", export_dir=version_dir, files=files,
                image_count=manifest.get("sample_count", 0), params=params,
                note=note, parent_version=parent_version)
            for w in record.get("warnings", []):
                warnings.append(w)
            if record.get("version") is not None:
                manifest["registry_version"] = record["version"]
        else:
            # no registry available/desired: still leave a local record
            write_version_sidecar(version_dir, {
                "version": None, "kind": KIND_LORA_TARGET_EXPORT,
                "name": target, "dir": str(version_dir), "note": note,
                "parent_version": parent_version,
                "created_at": manifest.get("created_at"),
                "image_count": manifest.get("sample_count", 0),
                "val_count": 0, "params": params,
            })
    except Exception as exc:                        # never crash export
        warnings.append(f"Version registration skipped: {exc}")


def _write_dataset_card_safe(manifest: dict, version_dir: Path,
                             warnings: list[str], out_dir=None,
                             name_prefix: str = "") -> None:
    """Generate the rich dataset card + structured manifest, additively. Any
    failure is recorded as a warning and never interrupts the export."""
    try:
        from .dataset_cards import (generate_dataset_card,
                                    validate_exported_dataset)
        report = validate_exported_dataset(version_dir, manifest=manifest)
        card = generate_dataset_card(
            version_dir, manifest=manifest, validation_report=report,
            out_dir=out_dir, name_prefix=name_prefix)
        manifest["dataset_card_paths"] = card.get("_paths", {})
        manifest["validation_passed"] = report.passed
        for w in report.warnings:
            sev = getattr(w, "severity", "warning")
            if sev == "warning":
                warnings.append(f"dataset_validation: {w.code} "
                                f"(count={w.count})")
    except Exception as exc:                            # never crash export
        warnings.append(f"Could not write dataset card: {exc}")


# ---------------------------------------------------------------------------
# DatasetSplitter.
# ---------------------------------------------------------------------------
@dataclass
class DatasetSplitterConfig:
    output_root: Path
    dry_run: bool = True
    resume: bool = True
    copy_mode: str = "copy"                  # "copy" or "symlink"
    version: str | None = None
    allowed_extensions: tuple[str, ...] = IMAGE_EXTENSIONS
    video_extensions: tuple[str, ...] = VIDEO_EXTENSIONS
    # Phase 2 versioning: project output base holding the SQLite manifest.
    # When set, real exports register in the shared version registry.
    manifest_base: Path | None = None
    version_note: str = ""


class DatasetSplitter:
    """Discover, classify, and export versioned dataset splits.

    ``source`` may be a folder (scanned recursively for images/videos) or a
    list of file paths / metadata dicts. Everything is dry-run by default.
    """

    def __init__(self, source: Path | list[Path],
                 output_root: Path | None = None, dry_run: bool = True,
                 resume: bool = True, copy_mode: str = "copy",
                 version: str | None = None,
                 personal_analyzer: Any = None) -> None:
        self.source = source
        if output_root is None:
            base = (Path(source) if isinstance(source, (str, Path))
                    else Path.cwd())
            base = base if base.is_dir() else base.parent
            output_root = base / "outputs"
        self.config = DatasetSplitterConfig(
            output_root=Path(output_root).expanduser(), dry_run=bool(dry_run),
            resume=bool(resume), copy_mode=copy_mode, version=version)
        # Optional, disabled by default. Accepts None / bool / dict /
        # PersonalAnalyzerConfig / project-file path. When disabled the splitter
        # behaves byte-for-byte as before.
        self.personal_analyzer: PersonalAnalyzerConfig = \
            coerce_config(personal_analyzer)
        self._samples: list[dict] | None = None
        self._classified: list[dict] | None = None

    # -- properties ----------------------------------------------------------
    @property
    def splits_root(self) -> Path:
        return self.config.output_root / "dataset_splits"

    # -- discovery -----------------------------------------------------------
    def discover_samples(self) -> list[dict]:
        """Discover candidate samples without reading pixel data. No deep
        video processing; videos are recorded as ``source_type='video'``."""
        allowed = tuple(e.lower() for e in self.config.allowed_extensions)
        videos = tuple(e.lower() for e in self.config.video_extensions)
        items: list = ([self.source] if isinstance(self.source, (str, Path))
                       else list(self.source or []))
        out: list[dict] = []

        def _add_file(p: Path, base_meta: dict | None = None):
            ext = p.suffix.lower()
            if ext not in allowed and ext not in videos:
                return
            source_type = "video" if ext in videos else "image"
            meta, w = load_sidecar_metadata(p)
            if base_meta:
                merged = dict(meta)
                merged.update(base_meta)
                meta = merged
            out.append({"sample_path": str(p), "source_type": source_type,
                        "metadata": meta, "warnings": list(w)})

        for item in items:
            if isinstance(item, dict):
                sp = item.get("sample_path") or item.get("path") or ""
                out.append({
                    "sample_path": str(sp),
                    "source_type": item.get("source_type")
                    or _infer_source_type(sp),
                    "metadata": item.get("metadata") or {
                        k: v for k, v in item.items()
                        if k not in ("sample_path", "path", "source_type",
                                     "metadata", "warnings")},
                    "warnings": list(item.get("warnings", [])),
                })
                continue
            p = Path(str(item)).expanduser()
            if p.is_dir():
                for q in sorted(p.rglob("*")):
                    if q.is_file():
                        _add_file(q)
            else:
                _add_file(p)

        self._samples = out
        return out

    # -- classification ------------------------------------------------------
    def classify_samples(self, samples: list[dict] | None = None
                         ) -> list[dict]:
        if samples is None:
            samples = self._samples if self._samples is not None \
                else self.discover_samples()
        classified: list[dict] = []
        for s in samples:
            try:
                result = classify_training_sample(dict(s))
            except Exception as exc:                # pragma: no cover
                result = {
                    "sample_path": str(s.get("sample_path", "")),
                    "source_type": s.get("source_type", "image"),
                    "lora_target": "needs_review", "confidence": 0.0,
                    "reason_codes": ["classification_error"],
                    "metadata": s.get("metadata") or {},
                    "warnings": [f"classification error: {exc}"]}
            # Optional, additive suitability labelling. No-op when disabled;
            # never alters lora_target, never removes or rejects a sample.
            enrich_sample(result, self.personal_analyzer)
            classified.append(result)
        self._classified = classified
        return classified

    # -- export --------------------------------------------------------------
    def export_splits(self, classified_samples: list[dict] | None = None,
                      version: str | None = None) -> dict:
        if classified_samples is None:
            classified_samples = self._classified if self._classified \
                is not None else self.classify_samples()
        version = version or self.config.version
        if version is None:
            version = self._next_global_version()

        groups: dict[str, list[dict]] = {t: [] for t in TARGETS}
        for row in classified_samples:
            groups.setdefault(row.get("lora_target", "needs_review"),
                              []).append(row)

        exports: dict[str, dict] = {}
        warnings: list[str] = []
        for target in EXPORTABLE_TARGETS:
            rows = groups.get(target, [])
            manifest = export_lora_target_dataset(
                rows, target, self.splits_root, version=version,
                dry_run=self.config.dry_run, copy_mode=self.config.copy_mode,
                resume=self.config.resume,
                manifest_base=self.config.manifest_base,
                version_note=self.config.version_note)
            exports[target] = manifest
            warnings += [w for w in manifest.get("warnings", [])
                         if w not in warnings]

        return {"version": version, "exports": exports,
                "target_counts": {t: len(groups.get(t, [])) for t in TARGETS},
                "warnings": warnings}

    def _next_global_version(self) -> str:
        n = 1
        while any((self.splits_root / f"v{n:03d}_{t}").exists()
                  for t in EXPORTABLE_TARGETS):
            n += 1
        return f"v{n:03d}"

    # -- orchestration -------------------------------------------------------
    def run(self, version: str | None = None) -> dict:
        """Discover -> classify -> export, returning a structured result."""
        warnings: list[str] = []
        errors: list[str] = []
        try:
            samples = self.discover_samples()
            classified = self.classify_samples(samples)
            export = self.export_splits(classified, version=version)
        except Exception as exc:                    # never crash the caller
            return {
                "status": "error", "dry_run": self.config.dry_run,
                "version": version or self.config.version or "v001",
                "source": str(self.source), "discovered_count": 0,
                "classified_count": 0, "target_counts": {}, "exports": {},
                "warnings": warnings, "errors": [f"run error: {exc}"]}

        for c in classified:
            warnings += [w for w in c.get("warnings", []) if w not in warnings]
        warnings += [w for w in export.get("warnings", [])
                     if w not in warnings]
        return {
            "status": "planned" if self.config.dry_run else "exported",
            "dry_run": self.config.dry_run,
            "version": export["version"],
            "source": str(self.source),
            "discovered_count": len(samples),
            "classified_count": len(classified),
            "target_counts": export["target_counts"],
            "exports": export["exports"],
            "splits_root": str(self.splits_root),
            "warnings": warnings,
            "errors": errors,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }


__all__ = [
    "DatasetSplitter", "DatasetSplitterConfig", "TARGETS", "TARGET_ALIASES",
    "EXPORTABLE_TARGETS", "IMAGE_EXTENSIONS", "VIDEO_EXTENSIONS",
    "CAPTION_DEFAULTS", "normalize_lora_target", "load_sidecar_metadata",
    "classify_training_sample", "build_training_caption",
    "export_lora_target_dataset",
]
