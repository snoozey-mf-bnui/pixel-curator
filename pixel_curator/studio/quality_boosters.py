"""Feature 10F - catalog quality booster library.

Reusable, strictly professional positive/negative prompt fragments for
catalog-grade output. Fragments are added only once (de-duplicated against
the existing prompt), and the user's own prompt text is always preserved.
"""

from __future__ import annotations

from collections.abc import Iterable

# Each booster key carries a professional positive fragment and an
# artifact-control negative fragment.
BOOSTERS: dict[str, dict[str, str]] = {
    "catalog_base": {
        "pos": "high-end catalog photography, polished editorial composition",
        "neg": "low quality, low-detail fabric"},
    "catalog_lighting": {
        "pos": "controlled studio lighting, balanced highlights",
        "neg": "inconsistent lighting, harsh specular artifacts"},
    "garment_detail": {
        "pos": "refined garment rendering, detailed textile structure",
        "neg": "low-detail fabric, garment fusion"},
    "fabric_boundary": {
        "pos": "clean fabric boundaries, stable garment edges",
        "neg": "distorted garment edges, fabric artifacts, "
               "garment boundary artifacts"},
    "fabric_interaction": {
        "pos": "refined fabric interaction, soft draped fabric",
        "neg": "texture smearing, fabric artifacts"},
    "skin_texture": {
        "pos": "balanced skin texture highlights",
        "neg": "over-smoothed skin, waxy texture"},
    "identity_consistency": {
        "pos": "consistent model identity",
        "neg": "identity drift, face inconsistency, "
               "inconsistent facial structure"},
    "pose_stability": {
        "pos": "stable pose, natural body proportions",
        "neg": "unstable hands, distorted proportions"},
    "sequence_consistency": {
        "pos": "frame-to-frame consistency, studio lighting continuity",
        "neg": "frame-to-frame lighting shift, background flicker"},
    "detail_preservation": {
        "pos": "high detail preservation, crisp catalog detail",
        "neg": "texture smearing, blurry details"},
    "clean_background": {
        "pos": "clean studio background",
        "neg": "cluttered background, background flicker"},
}

# Preset-specific booster selections (keys into BOOSTERS).
PRESET_BOOSTERS: dict[str, list[str]] = {
    "catalog_pro": ["catalog_base", "catalog_lighting", "garment_detail",
                    "fabric_boundary", "skin_texture",
                    "identity_consistency", "detail_preservation",
                    "clean_background"],
    "figure_study": ["catalog_base", "catalog_lighting", "fabric_boundary",
                     "pose_stability", "identity_consistency",
                     "detail_preservation"],
    "sheer_fabric_detail": ["catalog_base", "garment_detail",
                            "fabric_interaction", "fabric_boundary",
                            "detail_preservation", "identity_consistency"],
    "wardrobe_sequence": ["catalog_base", "catalog_lighting",
                          "sequence_consistency", "identity_consistency",
                          "pose_stability"],
}


def _fragments(keys: Iterable[str], kind: str) -> list[str]:
    out: list[str] = []
    for k in keys:
        frag = BOOSTERS.get(k, {}).get(kind, "")
        for part in (p.strip() for p in frag.split(",")):
            if part and part not in out:
                out.append(part)
    return out


def positive_fragments(preset: str) -> list[str]:
    return _fragments(PRESET_BOOSTERS.get(preset, []), "pos")


def negative_fragments(preset: str) -> list[str]:
    return _fragments(PRESET_BOOSTERS.get(preset, []), "neg")


def identity_negative_fragments() -> list[str]:
    """The minimum identity-protection negatives the guard always ensures."""
    return _fragments(["identity_consistency", "fabric_boundary"], "neg")


def merge_prompt(base: str, additions: list[str]) -> tuple[str, list[str]]:
    """Append fragments to a prompt, skipping any already present
    (case-insensitive). Returns (merged_prompt, actually_added)."""
    base = (base or "").rstrip().rstrip(",")
    low = base.lower()
    added: list[str] = []
    for frag in additions:
        if frag and frag.lower() not in low:
            added.append(frag)
            low += ", " + frag.lower()
    if not added:
        return base, []
    joined = ", ".join(added)
    merged = f"{base}, {joined}" if base else joined
    return merged, added
