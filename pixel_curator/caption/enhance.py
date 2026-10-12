"""Studio expansion - improved, style-aware captioning + auto-tagging.

Builds higher-quality, training-ready captions for the personal self-modeling
workflow and derives professional (SFW) auto-tags from lightweight metadata. SFW,
catalog terminology only. Deterministic and dependency-free (no heavy vision
models), so it runs in tests and as a metadata fallback; when WD14 tags are
available (via ``caption.wd14``) they are folded in and cleaned.

Caption shape: ``<identity anchor>, <target booster>, <style cues>,
<garment cues>, <quality cues>`` - de-duplicated, comma-joined, length-capped.

Used by ``curate.dataset_splitter`` as the caption/tag provider.
"""
from __future__ import annotations
try:
    from pixel_curator import nsfw_vocab
except ImportError:
    nsfw_vocab = None

import logging
from pathlib import Path

log = logging.getLogger("redresser.caption_enhance")

IDENTITY_ANCHOR = "personal_identity_anchor"

#: per-target professional (SFW) caption boosters (aligned with kohya_config_builder)
TARGET_BOOSTERS: dict[str, str] = {
    "core_identity": "professional (SFW) catalog reference, clear facial structure, "
                     "consistent identity, studio-ready framing",
    "outfit_clothing": "wardrobe variation, fashion catalog styling, refined "
                       "garment rendering, clean fabric boundaries",
    "style_aesthetic": "catalog photography style, controlled studio lighting, "
                       "polished editorial composition",
    "pose_action": "fashion model pose, balanced body proportions, "
                   "catalog-ready composition",
    "garment_detail": "garment detail study, refined textile structure, "
                      "fabric interaction, clean garment edges",
    "catalog_reference": "fashion catalog reference, polished studio "
                         "presentation, catalog-ready image quality",
}

#: metadata-driven style cues (SFW)
_STYLE_CUES = {
    "lighting": {
        "studio": "controlled studio lighting",
        "natural": "soft natural light",
        "golden_hour": "warm golden-hour light",
        "high_key": "high-key lighting",
        "low_key": "low-key lighting",
    },
    "composition": {
        "portrait": "portrait composition",
        "full_body": "full-body composition",
        "half_body": "half-body composition",
        "close_up": "close-up composition",
        "editorial": "editorial composition",
    },
    "background": {
        "plain": "clean studio background",
        "seamless": "seamless backdrop",
        "outdoor": "outdoor catalog setting",
    },
}

#: SFW garment vocabulary for auto-tagging
_GARMENT_TAGS = (
    "dress", "blazer", "trousers", "skirt", "knitwear", "outerwear",
    "activewear", "swimwear", "lingerie", "innerwear", "gown", "suit",
    "blouse", "jacket", "coat", "jumpsuit",
)

#: quality cue ladder by quality score
_QUALITY_CUES = [
    (0.85, "high detail, sharp focus"),
    (0.65, "refined detail"),
    (0.0, "catalog quality"),
]

_MAX_CAPTION_CHARS = 320


def _dedup(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        t = p.strip().strip(",").strip()
        key = t.lower()
        if t and key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _style_cues(meta: dict) -> list[str]:
    cues: list[str] = []
    for field, mapping in _STYLE_CUES.items():
        val = str(meta.get(field, "")).strip().lower()
        if val in mapping:
            cues.append(mapping[val])
    framing = str(meta.get("framing", "")).strip().lower()
    if framing in _STYLE_CUES["composition"]:
        cues.append(_STYLE_CUES["composition"][framing])
    return cues


def _quality_cue(meta: dict) -> str:
    try:
        q = float(meta.get("quality_score", 0.6))
    except (TypeError, ValueError):
        q = 0.6
    for threshold, cue in _QUALITY_CUES:
        if q >= threshold:
            return cue
    return "catalog quality"


def auto_tag(metadata: dict | None = None,
             wd14_tags: list | None = None) -> list[str]:
    """Derive professional (SFW) SFW auto-tags from metadata (+ optional WD14 tags).

    Never raises. Drops anything outside the professional (SFW) vocabulary and
    de-duplicates. WD14 tags are filtered to the garment/style vocabulary so
    no unprofessional (SFW) terms leak into the caption manifest.
    """
    metadata = metadata or {}
    tags: list[str] = []

    # explicit metadata tags (already curated upstream)
    for key in ("tags", "review_tags"):
        val = metadata.get(key)
        if isinstance(val, (list, tuple, set)):
            tags += [str(t).strip().lower() for t in val]
        elif isinstance(val, str):
            tags += [t.strip().lower() for t in val.split(",") if t.strip()]

    # garment hint from metadata
    garment = str(metadata.get("garment", "")).strip().lower()
    if garment:
        tags.append(garment)

    # framing -> composition tag
    framing = str(metadata.get("framing", "")).strip().lower()
    if framing:
        tags.append(framing)

    # fold in WD14 tags, filtered to the professional (SFW) vocabulary
    allowed = set(_GARMENT_TAGS) | {
        "studio", "natural", "portrait", "full_body", "half_body",
        "close_up", "editorial", "outdoor", "lighting", "catalog"}
    for entry in (wd14_tags or []):
        name = (entry[0] if isinstance(entry, (list, tuple)) else entry)
        name = str(name).strip().lower().replace(" ", "_")
        if any(g in name for g in _GARMENT_TAGS) or name in allowed:
            tags.append(name)

    # keep only professional (SFW), de-duplicated tags
    clean = _dedup([t for t in tags if t and t in allowed
                    or any(g in t for g in _GARMENT_TAGS)])
    return clean


def build_caption(target: str, metadata: dict | None = None,
                  wd14_tags: list | None = None,
                  style_aware: bool = True) -> dict:
    """Compose a high-quality, style-aware caption for a target.

    A valid ``caption`` in ``metadata`` is preserved verbatim (manual
    override). Otherwise: identity anchor + target booster + style cues
    (when ``style_aware``) + garment cues + quality cue, de-duplicated and
    length-capped. Returns ``{caption, source, target, tags}``.
    """
    metadata = metadata or {}
    override = metadata.get("caption")
    tags = auto_tag(metadata, wd14_tags)
    if isinstance(override, str) and override.strip():
        return {"caption": override.strip(), "source": "manual_override",
                "target": target, "tags": tags}

    booster = TARGET_BOOSTERS.get(target, TARGET_BOOSTERS["catalog_reference"])
    parts = [IDENTITY_ANCHOR, booster]
    if style_aware:
        parts += _style_cues(metadata)
    # garment cue from tags (first garment-like tag)
    garment = next((t for t in tags
                    if any(g in t for g in _GARMENT_TAGS)), "")
    if garment:
        parts.append(f"{garment.replace('_', ' ')} styling")
    parts.append(_quality_cue(metadata))

    caption = ", ".join(_dedup(parts))
    if len(caption) > _MAX_CAPTION_CHARS:
        caption = caption[:_MAX_CAPTION_CHARS].rsplit(",", 1)[0].strip()
    return {"caption": caption, "source": "style_aware", "target": target,
            "tags": tags}


def caption_for_target(target: str, sample: dict | None = None) -> dict:
    """Drop-in replacement for the basic caption helper used by the splitter.

    Accepts a sample dict (with optional nested ``metadata``) and returns the
    standard ``{caption, source, target}`` shape plus ``tags``.
    """
    sample = sample or {}
    meta = dict(sample.get("metadata") or {})
    # allow flat keys too
    for k, v in sample.items():
        if k not in ("metadata", "warnings", "sample_path", "source_type"):
            meta.setdefault(k, v)
    wd14 = meta.get("wd14_tags")
    return build_caption(target, meta, wd14_tags=wd14, style_aware=True)


__all__ = [
    "IDENTITY_ANCHOR", "TARGET_BOOSTERS", "auto_tag", "build_caption",
    "caption_for_target",
]


def get_nsfw_positive_boosters() -> list:
    if nsfw_vocab:
        return nsfw_vocab.POSITIVE_BOOSTERS
    return ["explicit", "lewd", "obscene", "detailed pussy", "wet", "aroused"]

def enhance_with_nsfw_tags(tags: list) -> list:
    if not nsfw_vocab:
        return tags
    nsfw_tags = nsfw_vocab.get_nsfw_tags()
    return list(set(tags + nsfw_tags))
