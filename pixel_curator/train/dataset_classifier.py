"""Feature 11B - Dataset splitting and classification for phased training.

Classifies a personal media-archive sample into a LoRA training target using
only lightweight, deterministic metadata. No heavy vision models are required:
when an advanced classifier backend is unavailable the module falls back to a
metadata-based split and surfaces a structured warning instead of raising.

Manual override always wins: a sample may carry an explicit ``lora_target``
field (for example assigned during CLI review or the REDRESSER curation
flow) and that value takes priority over automatic classification.

Pure stdlib, local-first, non-destructive. Source media is never read, moved,
or modified here - only metadata dictionaries are inspected.
"""
from __future__ import annotations

import logging
from typing import Iterable

log = logging.getLogger("redresser.dataset_classifier")

# ---------------------------------------------------------------------------
# Allowed LoRA training targets (manual override + automatic classification).
# ---------------------------------------------------------------------------
LORA_TARGETS: tuple[str, ...] = (
    "core_identity",
    "outfit",
    "style",
    "pose",
    "garment_detail",
    "catalog_reference",
    "needs_review",
    "excluded",
)

#: Targets that feed an actual LoRA training split (the rest are holding bins).
TRAINABLE_TARGETS: tuple[str, ...] = (
    "core_identity", "outfit", "style", "pose", "garment_detail",
)

_ADVANCED_WARNING = (
    "Advanced classifier unavailable; using metadata-based dataset splitting.")


def _advanced_classifier_available() -> bool:
    """Return True only if an optional vision backend is importable.

    Tests and the default local environment do not ship one, so this returns
    False and the deterministic metadata path is used. Never raises.
    """
    try:                                            # pragma: no cover
        import open_clip  # noqa: F401
        return True
    except Exception:
        return False


def _f(sample: dict, key: str, default: float = 0.0) -> float:
    """Best-effort float read from a metadata dict (never raises)."""
    try:
        val = sample.get(key)
        return float(val) if val is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _norm_target(value: object) -> str | None:
    """Normalise a manual ``lora_target`` value to an allowed token or None."""
    if value is None:
        return None
    token = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return token if token in LORA_TARGETS else None


def classify_training_sample(sample: dict) -> dict:
    """Classify one sample dict into a LoRA training target.

    Parameters
    ----------
    sample:
        A metadata dictionary. Recognised optional keys include::

            path / sample_path / filepath   - source media path (preserved)
            lora_target                      - MANUAL override (wins)
            quality_score                    - 0..1
            sharpness                        - 0..1
            brightness / exposure            - 0..1 (0.5 is ideal)
            is_duplicate                     - bool
            framing                          - full_body|half_body|close_up
            face_visibility                  - 0..1 or none|partial|clear
            identity_similarity              - 0..1
            pose_type                        - standing|sitting|dynamic|...
            garment_visibility               - 0..1
            catalog_suitable                 - bool
            source_type                      - photo|render|frame|...
            review_tags                      - list[str]

    Returns
    -------
    dict with keys ``sample_path``, ``lora_target``, ``confidence``,
    ``reason_codes`` and ``warnings``. Never raises for ordinary input.
    """
    warnings: list[str] = []
    path = (sample.get("sample_path") or sample.get("path")
            or sample.get("filepath") or "")

    # 1) Manual override always wins.
    manual = sample.get("lora_target", None)
    if manual is not None:
        token = _norm_target(manual)
        if token is not None:
            return {
                "sample_path": str(path),
                "lora_target": token,
                "confidence": 1.0,
                "reason_codes": ["manual_override"],
                "warnings": warnings,
            }
        warnings.append(
            f"Manual lora_target '{manual}' is not an allowed value; falling "
            "back to automatic classification.")

    # 2) Automatic classification (deterministic, metadata-only).
    if not _advanced_classifier_available():
        warnings.append(_ADVANCED_WARNING)

    reason_codes: list[str] = []

    quality = _f(sample, "quality_score", 0.5)
    sharpness = _f(sample, "sharpness", quality)
    brightness = _f(sample, "brightness", _f(sample, "exposure", 0.5))
    identity_sim = _f(sample, "identity_similarity", 0.0)
    face_vis = _face_visibility(sample)
    garment_vis = _f(sample, "garment_visibility", 0.0)
    framing = str(sample.get("framing", "")).strip().lower()
    pose_type = str(sample.get("pose_type", "")).strip().lower()
    source_type = str(sample.get("source_type", "")).strip().lower()
    tags = _collect_tags(sample)

    # 2a) hard exclusions first.
    if sample.get("is_duplicate") is True or "duplicate" in tags:
        reason_codes.append("duplicate_status_flagged")
        return _result(path, "excluded", 0.95, reason_codes, warnings)
    if "exclude" in tags or "rejected" in tags:
        reason_codes.append("user_review_excluded")
        return _result(path, "excluded", 0.95, reason_codes, warnings)

    # 2b) quality gate -> needs_review (does not delete; just holds).
    poor = []
    if quality < 0.35:
        poor.append("quality_score_low")
    if sharpness < 0.30:
        poor.append("sharpness_low")
    if brightness < 0.15 or brightness > 0.9:
        poor.append("exposure_out_of_range")
    if poor:
        reason_codes.extend(poor)
        return _result(path, "needs_review", 0.55, reason_codes, warnings)

    if "review" in tags or "uncertain" in tags:
        reason_codes.append("user_review_flagged")
        return _result(path, "needs_review", 0.5, reason_codes, warnings)

    # 2c) positive routing - first strong signal wins, deterministically.
    # Core identity: strong identity similarity + clear face + quality.
    if identity_sim >= 0.7 and face_vis >= 0.6 and quality >= 0.6:
        reason_codes += ["identity_similarity_strong",
                         "face_visibility_clear", "quality_score_high"]
        conf = round(min(0.99, 0.6 + 0.3 * identity_sim + 0.1 * quality), 2)
        return _result(path, "core_identity", conf, reason_codes, warnings)

    # Garment detail: explicit texture/detail tag, or close-up + high garment
    # visibility. The tag path lets the media analyzer route crisp wardrobe
    # shots here (it emits a 'garment_detail' tag for very sharp body frames).
    if tags & {"garment_detail", "textile", "fabric", "fabric_closeup",
               "detail_study", "garment_closeup"}:
        reason_codes += ["garment_detail_tag"]
        return _result(path, "garment_detail", 0.85, reason_codes, warnings)
    if garment_vis >= 0.6 and framing == "close_up":
        reason_codes += ["garment_visibility_high", "close_up_framing"]
        conf = round(min(0.95, 0.55 + 0.3 * garment_vis), 2)
        return _result(path, "garment_detail", conf, reason_codes, warnings)

    # Pose: an identified whole-body pose routes here BEFORE the generic outfit
    # bucket so full-body shots are not all swallowed as clothing. Only fires
    # on a pose signal (pose_type / 'pose' tag); framing-only samples fall
    # through to outfit exactly as before.
    if pose_type in ("standing", "sitting", "dynamic", "walking", "model") \
            or "pose" in tags:
        reason_codes += ["pose_type_identified"]
        if framing in ("full_body", "half_body"):
            reason_codes.append(f"{framing}_framing")
        return _result(path, "pose", 0.7, reason_codes, warnings)

    # Outfit: garment visible on a half/full body framing.
    if garment_vis >= 0.4 and framing in ("full_body", "half_body"):
        reason_codes += ["garment_visibility_moderate",
                         f"{framing}_framing"]
        conf = round(min(0.92, 0.5 + 0.3 * garment_vis), 2)
        return _result(path, "outfit", conf, reason_codes, warnings)

    # Style: catalog-suitable / editorial, an explicit style tag, or the media
    # analyzer's style_reference flag (aesthetic close shot, weak identity).
    if bool(sample.get("catalog_suitable")) or bool(sample.get("style_reference")) \
            or "style" in tags or "editorial" in tags:
        reason_codes += ["style_reference"]
        return _result(path, "style", 0.65, reason_codes, warnings)

    # Catalog reference: clean render/photo with good quality but no strong cue.
    if quality >= 0.6 and source_type in ("photo", "render", "catalog"):
        reason_codes += ["quality_score_high", "catalog_reference_candidate"]
        return _result(path, "catalog_reference", 0.6, reason_codes, warnings)

    # Default: hold for review rather than guess.
    reason_codes.append("insufficient_signal")
    return _result(path, "needs_review", 0.4, reason_codes, warnings)


def _face_visibility(sample: dict) -> float:
    raw = sample.get("face_visibility")
    if isinstance(raw, str):
        return {"clear": 0.9, "visible": 0.8, "partial": 0.45,
                "none": 0.0}.get(raw.strip().lower(), 0.0)
    return _f(sample, "face_visibility", 0.0)


def _collect_tags(sample: dict) -> set[str]:
    tags: set[str] = set()
    for key in ("review_tags", "tags"):
        val = sample.get(key)
        if isinstance(val, (list, tuple, set)):
            tags.update(str(t).strip().lower() for t in val)
        elif isinstance(val, str):
            tags.update(t.strip().lower() for t in val.split(",") if t.strip())
    return tags


def _result(path, target, confidence, reason_codes, warnings) -> dict:
    return {
        "sample_path": str(path),
        "lora_target": target,
        "confidence": round(float(confidence), 2),
        "reason_codes": list(reason_codes),
        "warnings": list(warnings),
    }


def classify_samples(samples: Iterable[dict]) -> list[dict]:
    """Classify many samples; never raises for an individual bad row."""
    out: list[dict] = []
    for sample in samples or []:
        try:
            out.append(classify_training_sample(dict(sample)))
        except Exception as exc:                    # pragma: no cover
            log.warning("classification error: %s", exc)
            out.append({
                "sample_path": str((sample or {}).get("path", "")),
                "lora_target": "needs_review",
                "confidence": 0.0,
                "reason_codes": ["classification_error"],
                "warnings": [f"classification error: {exc}"],
            })
    return out


def split_counts(classified: Iterable[dict]) -> dict[str, int]:
    """Tally classified rows per target (all targets present, zero-filled)."""
    counts = {t: 0 for t in LORA_TARGETS}
    for row in classified or []:
        target = row.get("lora_target", "needs_review")
        counts[target] = counts.get(target, 0) + 1
    return counts


def group_by_target(classified: Iterable[dict]) -> dict[str, list[dict]]:
    """Group classified rows into lists keyed by target."""
    groups: dict[str, list[dict]] = {t: [] for t in LORA_TARGETS}
    for row in classified or []:
        groups.setdefault(row.get("lora_target", "needs_review"), []).append(
            row)
    return groups
