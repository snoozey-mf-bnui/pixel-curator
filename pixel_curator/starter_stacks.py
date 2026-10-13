"""Recommended starter stacks - explainable, identity-first templates.

Suggests clean production stacks from the user's actual LoRA library using
sidecar metadata first (concept family, priority hints, known conflicts),
filename inference second.  No ML required; every recommendation carries
reason codes and resolves through stack intelligence for a preservation
score before it is ever shown.
"""
from __future__ import annotations

from .lora_explorer import LoraCard
from .stack_intelligence import resolve_stack

# ---------------------------------------------------------------------------
# STARTER_TEMPLATES — professional, identity-first fashion/catalog templates
# ---------------------------------------------------------------------------
STARTER_TEMPLATES: list[tuple[str, str, str | None]] = [
    ("Identity + Studio Lighting", "lighting", None),
    ("Identity + Editorial Fashion", "fashion", None),
    ("Identity + Catalog Style", "style", None),
    ("Identity + Wardrobe Layering", "clothing", None),
    ("Identity + Fashion + Lighting", "fashion", "lighting"),
    ("Identity + Style + Studio Lighting", "style", "lighting"),
]


def _pick_identity(cards: list[LoraCard]) -> LoraCard | None:
    """Best identity anchor: anchor-hinted first, then identity family,
    most recently modified wins."""
    ids = [c for c in cards if c.profile.family in ("identity", "character")]
    if not ids:
        return None
    return sorted(ids, key=lambda c: (
        getattr(c.profile, "priority_hint", "") != "anchor",
        -(0 if not c.modified_at else 1), c.lora_id))[0]


def _pick_concept(cards: list[LoraCard], family: str,
                  chosen: list[LoraCard]) -> LoraCard | None:
    """Best concept card for a family: lowest identity risk first, sidecar
    metadata preferred; respects known conflicts against already-chosen
    LoRAs and families."""
    risk_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    chosen_ids = {c.lora_id for c in chosen}
    chosen_fams = {c.profile.family for c in chosen}
    pool = []
    for c in cards:
        if c.profile.family != family or c.lora_id in chosen_ids:
            continue
        if set(c.profile.known_conflicts) & chosen_ids:
            continue
        if set(getattr(c.profile, "conflict_families", ())) & chosen_fams:
            continue
        # reverse direction: an already-chosen LoRA vetoes this family
        if any(family in getattr(ch.profile, "conflict_families", ())
               or c.lora_id in ch.profile.known_conflicts
               for ch in chosen):
            continue
        pool.append(c)
    if not pool:
        return None
    return sorted(pool, key=lambda c: (
        risk_rank.get(c.profile.identity_risk, 2),
        c.metadata_source == "inferred",      # sidecar-backed first
        c.lora_id))[0]


def recommend_starter_stacks(cards: list[LoraCard],
                             base_model: str | None = None,
                             limit: int = 6) -> list[dict]:
    """Starter stack recommendations from the available library."""
    identity = _pick_identity(cards)
    if identity is None:
        return [{"name": "No identity anchor available",
                 "available": False, "stack": [], "reasons": [
                     "no_identity_lora_found",
                     "Train or import a character identity LoRA first - "
                     "starter stacks are built around an identity anchor."]}]
    out: list[dict] = []
    for name, fam_a, fam_b in STARTER_TEMPLATES:
        chosen = [identity]
        reasons = ["identity_anchor_selected",
                   f"identity anchor: {identity.lora_id}"]
        a = _pick_concept(cards, fam_a, chosen)
        if a is None:
            continue
        chosen.append(a)
        reasons.append(f"{fam_a}_concept_matched: {a.lora_id} "
                       f"({a.profile.identity_risk} risk, "
                       f"{'sidecar' if a.metadata_source != 'inferred' else 'inferred'} metadata)")
        if fam_b:
            b = _pick_concept(cards, fam_b, chosen)
            if b is None:
                continue
            chosen.append(b)
            reasons.append(f"{fam_b}_concept_matched: {b.lora_id}")
        # conservative weights: identity default; concepts capped at the
        # midpoint of their recommended range (studio-safe starting point)
        weights = {identity.lora_id: identity.profile.weight_default}
        for c in chosen[1:]:
            mid = round((c.profile.weight_min + c.profile.weight_max) / 2, 2)
            weights[c.lora_id] = min(c.profile.weight_default, mid)
        st = resolve_stack(chosen, weights, base_model)
        reasons.append("conservative_concept_weights")
        out.append({
            "name": name, "available": True,
            "identity": identity.lora_id,
            "stack": ([[st.identity_anchor.lora_id,
                        st.identity_anchor.weight]]
                      if st.identity_anchor else [])
            + [[i.lora_id, i.weight] for i in st.concept_loras],
            "weights": {k: round(v, 2) for k, v in weights.items()},
            "preservation_score": st.identity_preservation_score,
            "risk_level": st.risk_level,
            "reasons": reasons,
            "warnings": [w.message for w in st.warnings
                         if w.severity in ("caution", "critical")],
        })
        if len(out) >= limit:
            break
    if not out:
        out.append({"name": "Library needs concept LoRAs",
                    "available": False, "stack": [], "reasons": [
                        "no_concept_loras_matched",
                        "No suitable style/fashion/lighting concepts found "
                        "alongside the identity anchor."]})
    return out


# ---------------------------------------------------------------------------
# Artistic Figure Study Layer stack templates (additive) — professional
# fashion/catalog figure-study layering. Library hints match by substring.
# ---------------------------------------------------------------------------

# (template name, [(library-name hint, role, conservative weight)])
ARTISTIC_STACK_TEMPLATES: list[tuple[str, list[tuple[str, str, float]]]] = [
    ("Identity + Garment Form Study",
     [("lingerie_form_study", "garment_rendering", 0.45)]),
    ("Identity + Skin Texture Highlight + Studio Lighting",
     [("wet_glossy_skin", "skin_texture_highlight", 0.30),
      ("studio_lighting", "lighting", 0.25)]),
    ("Identity + Sheer Garment Rendering + Form Proportion Study",
     [("sheer_clothing", "sheer_garment_rendering", 0.35),
      ("figure_study", "form_proportion", 0.35)]),
    ("Identity + Garment Rendering + Catalog Pose",
     [("lingerie_form_study", "garment_rendering", 0.40),
      ("boudoir_pose", "pose", 0.30)]),
    ("Identity + Figure Study + Studio Lighting",
     [("figure_study", "form_proportion", 0.35),
      ("studio_lighting", "lighting", 0.25)]),
    ("Identity + Catalog Pose + Studio Lighting",
     [("boudoir_pose", "pose", 0.30),
      ("studio_lighting", "lighting", 0.25)]),
    ("Identity + Floral Garment Set (catalog look)",
     [("Lingerie_Set_Floral", "garment_rendering", 0.45)]),
    ("Identity + Designer Garment + Skin Texture Highlight",
     [("designer_lingerie", "garment_rendering", 0.45),
      ("wet_glossy_skin", "skin_texture_highlight", 0.30)]),
]


# ---------------------------------------------------------------------------
# Feature 5D - Wardrobe / fashion-catalog presets (validated on apply)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# WARDROBE_PRESETS — professional, SFW fashion-catalog presets (validated on
# apply). LoRA names are validator-friendly (no UI bracket prefix); a missing
# local asset warns and skips rather than crashing.
# ---------------------------------------------------------------------------
WARDROBE_PRESETS: list[dict] = [
    {
        "name": "Layered Sheer Editorial Set",
        "description": "Layered garment rendering for editorial catalog "
                       "styling with refined fabric interaction.",
        "loras": ["professional_identity_reference_v1.safetensors:0.78",
                  "structured_garment_layer_v2.safetensors:0.45",
                  "fabric_detail_enhancement_v3.safetensors:0.35"],
        "prompt": "high-end catalog styling, layered garment rendering, "
                  "refined fabric interaction, studio lighting continuity",
        "negative": "garment boundary artifacts, inconsistent fabric edges, "
                    "identity drift",
        "recommended_regions": ["torso", "full_body"],
        "use_case": "Personal wardrobe visualization and catalog-style "
                    "outfit testing.",
        "notes": "Validate available LoRAs before generation.",
    },
    {
        "name": "Structured Innerwear Try-On Set",
        "description": "Structured garment layer emphasis for fitted "
                       "wardrobe try-on visualization.",
        "loras": ["professional_identity_reference_v1.safetensors:0.78",
                  "structured_garment_layer_v2.safetensors:0.45",
                  "fabric_detail_enhancement_v3.safetensors:0.35"],
        "prompt": "fashion catalog styling, fitted garment structure, "
                  "clean garment boundaries, premium fabric detail",
        "negative": "garment artifacts, distorted proportions, identity drift",
        "recommended_regions": ["torso", "garment"],
        "use_case": "Personal fitted-garment try-on and fabric structure "
                    "review.",
        "notes": "Validate available LoRAs before generation.",
    },
    {
        "name": "Premium Catalog Fabric Study",
        "description": "Fine fabric and material detail study for premium "
                       "catalog presentation.",
        "loras": ["professional_identity_reference_v1.safetensors:0.78",
                  "structured_garment_layer_v2.safetensors:0.50",
                  "fabric_detail_enhancement_v3.safetensors:0.30"],
        "prompt": "premium catalog fabric study, refined textile structure, "
                  "soft studio shadows, lookbook composition",
        "negative": "texture smearing, over-smoothed fabric, identity drift",
        "recommended_regions": ["garment", "full_body"],
        "use_case": "Personal fabric and material rendering review.",
        "notes": "Validate available LoRAs before generation.",
    },
    {
        "name": "Full Body Wardrobe Transformation",
        "description": "Full-body wardrobe variation for complete catalog "
                       "look changes with identity consistency.",
        "loras": ["professional_identity_reference_v1.safetensors:0.80",
                  "structured_garment_layer_v2.safetensors:0.45",
                  "studio_lighting_catalog_v1.safetensors:0.25"],
        "prompt": "full body fashion catalog, wardrobe transformation, "
                  "consistent model identity, studio lighting continuity",
        "negative": "identity drift, garment boundary artifacts, "
                    "inconsistent lighting",
        "recommended_regions": ["full_body"],
        "use_case": "Personal full-look wardrobe variation testing.",
        "notes": "Validate available LoRAs before generation.",
    },
    {
        "name": "Minimal Studio Try-On",
        "description": "Lightweight identity-locked outfit study with a "
                       "single garment emphasis.",
        "loras": ["professional_identity_reference_v1.safetensors:0.80",
                  "structured_swimwear_v1.safetensors:0.40"],
        "prompt": "minimal studio try-on, clean catalog composition, "
                  "consistent identity, balanced studio lighting",
        "negative": "identity drift, garment artifacts, harsh specular "
                    "highlights",
        "recommended_regions": ["torso"],
        "use_case": "Personal quick single-garment try-on review.",
        "notes": "Validate available LoRAs before generation.",
    },
    {
        "name": "Identity-Locked Outfit Study",
        "description": "Identity-first outfit study prioritizing strong "
                       "personal identity consistency across garments.",
        "loras": ["professional_identity_reference_v1.safetensors:0.85",
                  "fabric_detail_enhancement_v3.safetensors:0.30"],
        "prompt": "identity-locked outfit study, consistent model identity, "
                  "refined garment rendering, editorial studio lighting",
        "negative": "identity drift, inconsistent facial structure, "
                    "garment boundary artifacts",
        "recommended_regions": ["torso", "full_body"],
        "use_case": "Personal identity-consistency review across outfits.",
        "notes": "Validate available LoRAs before generation.",
    },
    # -----------------------------------------------------------------------
    # Personal identity-anchor base/preset profiles. These pair the trained
    # core identity LoRA with a chosen base checkpoint for catalog-grade
    # personal self-modeling. LoRA names are validator-friendly (no UI bracket
    # prefixes); a missing local asset warns and skips rather than crashing.
    # -----------------------------------------------------------------------
    {
        "name": "Personalized CyberRealisticPony Base",
        "description": "Catalog-grade personal identity-anchor profile pairing "
                       "the CyberRealisticPony base with the trained core "
                       "identity LoRA for consistent fashion catalog output.",
        "base_checkpoint": "cyberrealisticPony_v180Coreshift.safetensors",
        "loras": ["pixel_curator_core_identity_cyberrealisticpony_v1.safetensors:0.80"],
        "weight_policy": {"identity_anchor": [0.70, 0.90], "default": 0.80},
        "prompt": ("personal_identity_anchor, professional catalog styling, "
                   "consistent identity, refined garment rendering, balanced "
                   "studio lighting, clean fabric boundaries, polished fashion "
                   "catalog composition"),
        "negative": ("identity drift, inconsistent facial structure, "
                     "distorted body proportions, unstable hands, garment "
                     "artifacts, fabric boundary errors, texture smearing, "
                     "over-smoothed skin, harsh lighting shift, low-detail "
                     "garment rendering"),
        "prompt_booster": ("personal_identity_anchor, professional catalog "
                           "styling, consistent identity, refined garment "
                           "rendering, balanced studio lighting, clean fabric "
                           "boundaries, polished fashion catalog composition"),
        "negative_booster": ("identity drift, inconsistent facial structure, "
                             "distorted body proportions, unstable hands, "
                             "garment artifacts, fabric boundary errors, "
                             "texture smearing, over-smoothed skin, harsh "
                             "lighting shift, low-detail garment rendering"),
        "quality_preset": "catalog_pro",
        "recommended_regions": ["torso", "full_body"],
        "controlnet": {
            "openpose": {"enabled": False, "weight": 0.75,
                         "use_case": "Enable for pose-guided catalog "
                                     "generation."},
            "depth": {"enabled": False, "weight": 0.45,
                      "use_case": "Enable for structure and garment "
                                  "consistency."},
        },
        "identity_guidance": {
            "ip_adapter": {"enabled_if_available": True, "weight": 0.75},
            "faceid": {"enabled_if_available": True, "weight": 0.75},
        },
        "post_process": {"realesrgan": True, "gfpgan": True,
                         "gfpgan_weight": 0.3, "final_sharpen": False,
                         "final_denoise": False},
        "recommended_modes": ["static_catalog", "wardrobe_variation",
                              "production_catalog"],
        "use_case": "Personal CyberRealisticPony identity-anchor catalog "
                    "generation and wardrobe variation.",
        "compatibility_note": "Optimized for the user's private "
                              "CyberRealisticPony personal identity-anchor "
                              "workflow.",
        "notes": "Validate available LoRAs before generation. Keep the "
                 "identity anchor weight within 0.70-0.90.",
    },
    {
        "name": "Personalized Pony Realism v22",
        "description": "Catalog-grade migration profile for a Pony Realism "
                       "v22 base, reusing the personal identity anchor LoRA "
                       "conservatively to preserve face and body consistency.",
        "base_checkpoint": "ponyRealism_v22.safetensors",
        "base_checkpoint_configurable": True,
        "loras": ["pixel_curator_core_identity_cyberrealisticpony_v1.safetensors:0.72"],
        "weight_policy": {"identity_anchor": [0.60, 0.82], "default": 0.72,
                          "reason": "Pony Realism-style bases may need a "
                                    "slightly lower identity LoRA weight to "
                                    "reduce over-baking while preserving "
                                    "face/body consistency."},
        "prompt": ("personal_identity_anchor, premium fashion catalog "
                   "realism, consistent identity, natural skin texture, "
                   "refined garment rendering, clean wardrobe styling, "
                   "balanced studio lighting, editorial catalog composition"),
        "negative": ("identity drift, face mismatch, inconsistent facial "
                     "structure, distorted proportions, unstable hands, "
                     "garment fusion, fabric artifacts, over-sharpened "
                     "texture, waxy skin, background flicker, harsh lighting "
                     "shift"),
        "prompt_booster": ("personal_identity_anchor, premium fashion catalog "
                           "realism, consistent identity, natural skin "
                           "texture, refined garment rendering, clean "
                           "wardrobe styling, balanced studio lighting, "
                           "editorial catalog composition"),
        "negative_booster": ("identity drift, face mismatch, inconsistent "
                             "facial structure, distorted proportions, "
                             "unstable hands, garment fusion, fabric "
                             "artifacts, over-sharpened texture, waxy skin, "
                             "background flicker, harsh lighting shift"),
        "quality_preset": "catalog_pro",
        "recommended_regions": ["torso", "full_body"],
        "controlnet": {
            "openpose": {"enabled": False, "weight": 0.70,
                         "use_case": "Enable for pose-guided catalog "
                                     "generation."},
            "depth": {"enabled": False, "weight": 0.40,
                      "use_case": "Enable for garment and structure "
                                  "consistency."},
        },
        "identity_guidance": {
            "ip_adapter": {"enabled_if_available": True, "weight": 0.70},
            "faceid": {"enabled_if_available": True, "weight": 0.70},
        },
        "post_process": {"realesrgan": True, "gfpgan": True,
                         "gfpgan_weight": 0.25, "final_sharpen": False,
                         "final_denoise": False},
        "recommended_modes": ["static_catalog", "wardrobe_variation",
                              "production_catalog"],
        "use_case": "Personal migration testing onto a Pony Realism v22-style "
                    "base while preserving identity consistency.",
        "compatibility_note": "Professional migration profile for Pony "
                              "Realism v22-style catalog workflows. Keep "
                              "identity LoRA weight moderate and validate "
                              "outputs before using in production batches.",
        "notes": "Validate available LoRAs before generation. If the local "
                 "checkpoint filename differs from "
                 "'ponyRealism_v22.safetensors', set base_checkpoint to the "
                 "actual local file; this profile keeps the checkpoint "
                 "configurable rather than hard-failing.",
    },
]


# Feature 6C - conservative, identity-preserving defaults applied to any
# preset that does not specify its own.
DEFAULT_POST_PROCESS = {"realesrgan": True, "gfpgan": True,
                        "gfpgan_weight": 0.3}
DEFAULT_ASSET_POLICY = {"download_missing": False, "verified_assets_only": True}


# ---------------------------------------------------------------------------
# Library-aware preset resolution (roles -> the user's actual LoRA files)
# ---------------------------------------------------------------------------
# Each preset entry's role is inferred (bracket tag first, then filename
# keywords) so a preset can be resolved against whatever library is present -
# the placeholder identity anchor becomes the user's real identity LoRA.
_BRACKET_ROLE = {
    "CLOTHING": "clothing", "ANATOMY": "anatomy", "DETAIL": "detail",
    "STYLE": "style", "CHARACTER": "character", "NSFW": "nsfw",
    "TEXT": "text", "INPAINT": "inpaint", "FACEID": "character",
    "GENERAL": "detail",
}
_ROLE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("identity", ("pixel-curator", "ohwx", "identity", "person_v",
                  "figure_study", "_anchor")),
    ("sheer", ("see-through", "see_through", "sheer", "transparent")),
    ("clothing", ("lingerie", "underwear", "underwire", "bikini", "lace",
                  "attire", "dress", "wedding", "garment", "innerwear",
                  "swimwear", "outfit")),
    ("detail", ("detail", "tweaker", "add-detail", "realism_enhancer",
                "fabric_detail", "textile_detail")),
    ("skin", ("realskin", "skin", "realism", "skin_texture", "complexion")),
    ("anatomy", ("proportion", "posture", "silhouette", "body_proportion",
                 "figure_proportion", "pose_proportion")),
    ("style", ("style", "cinematic", "aesthetic", "semireal", "photography",
               "editorial", "lighting")),
    ("character", ("girl", "woman", "person", "face", "nexblend",
                   "faetastic", "php")),
]


def detect_lora_role(name: str) -> str:
    """Infer a coarse role from a LoRA display name: bracket category tag
    first (e.g. '[SDXL-LORA-CLOTHING] ...'), then filename keywords."""
    up = name.upper()
    if "-LORA-" in up and "]" in up:
        tag = up.split("-LORA-")[1].split("]")[0].strip()
        if tag in _BRACKET_ROLE:
            # identity-looking stems still win even under a category tag
            low = name.lower()
            if any(k in low for k in ("pixel-curator", "figure_study", "identity")):
                return "identity"
            return _BRACKET_ROLE[tag]
    low = name.lower()
    for role, kws in _ROLE_KEYWORDS:
        if any(k in low for k in kws):
            return role
    return "general"


def build_library_role_index(lora_dir) -> dict[str, list[str]]:
    """Scan a LoRA directory once and group real file stems by inferred role
    (newest first). Returns {role: [stem, ...]}. Empty if the dir is absent."""
    from pathlib import Path
    index: dict[str, list[tuple[float, str]]] = {}
    d = Path(lora_dir).expanduser() if lora_dir else None
    if not d or not d.is_dir():
        return {}
    for p in d.rglob("*.safetensors"):
        if any(part.startswith(("_", ".")) for part in
               p.relative_to(d).parts[:-1]):
            continue
        try:
            mt = p.stat().st_mtime
        except OSError:
            mt = 0.0
        index.setdefault(detect_lora_role(p.stem), []).append((mt, p.stem))
    return {role: [s for _, s in sorted(items, reverse=True)]
            for role, items in index.items()}


# roles that can satisfy a requested role, in preference order
_ROLE_FALLBACKS = {
    "identity": ["identity", "character"],
    "clothing": ["clothing", "sheer"],
    "sheer": ["sheer", "clothing"],
    "detail": ["detail", "style"],
    "skin": ["skin", "detail"],
    "anatomy": ["anatomy"],
    "style": ["style", "detail"],
    "character": ["character", "identity"],
    "general": ["detail", "style", "clothing"],
}


def _pick_for_role(index: dict, role: str, used: set) -> str | None:
    for candidate_role in _ROLE_FALLBACKS.get(role, [role]):
        for stem in index.get(candidate_role, []):
            if stem not in used:
                used.add(stem)
                return stem
    return None


def resolve_preset_library_aware(preset: dict, lora_dir) -> dict:
    """Re-map a preset's LoRA stack onto the user's actual library by role,
    so the placeholder identity anchor becomes the real identity LoRA, etc.
    Entries whose role has no library match keep their original name (the
    validator will then warn/skip them). Returns
    {loras, substitutions, unresolved_roles}."""
    import re as _re
    index = build_library_role_index(lora_dir)
    all_stems = {s for lst in index.values() for s in lst}

    def _norm(s: str) -> str:
        s = _re.sub(r"\.safetensors$", "", s, flags=_re.I)
        return _re.sub(r"^\s*\[[^\]]*\]\s*", "", s).strip().lower()
    by_norm = {_norm(s): s for s in all_stems}

    used: set = set()
    out_loras: list[str] = []
    substitutions: list[dict] = []
    unresolved: list[str] = []
    for entry in preset.get("loras", []):
        head, sep, tail = str(entry).rpartition(":")
        name, weight = (head, tail) if head and sep else (str(entry), "")
        stem = _re.sub(r"\.safetensors$", "", name, flags=_re.I)
        role = detect_lora_role(name)
        # 1) keep the named file if it actually exists in the library
        real = (stem if stem in all_stems else by_norm.get(_norm(name)))
        if real and real not in used:
            used.add(real)
            out_loras.append(f"{real}:{weight}" if weight else real)
            continue
        # 2) otherwise resolve the role to the best available library file
        picked = _pick_for_role(index, role, used) if index else None
        if picked:
            out_loras.append(f"{picked}:{weight}" if weight else picked)
            substitutions.append({"role": role, "from": name, "to": picked})
        else:
            out_loras.append(str(entry))
            unresolved.append(role)
    return {"loras": out_loras, "substitutions": substitutions,
            "unresolved_roles": unresolved}


# Feature 10 link: map a preset to a recommended quality preset by intent.
def recommended_quality_preset(preset: dict) -> str:
    name = (preset.get("name") or "").lower()
    motion = preset.get("motion") or {}
    if motion or "sequence" in name or "turntable" in name or "motion" in name:
        return "wardrobe_sequence"
    if "sheer" in name or "fabric" in name:
        return "sheer_fabric_detail"
    if "full body" in name or "form" in name or "figure" in name:
        return "figure_study"
    return "catalog_pro"


def _with_defaults(preset: dict) -> dict:
    """Return a copy of a preset with post-processing + asset-policy defaults
    and a recommended quality preset filled in (existing values preserved)."""
    p = dict(preset)
    pp = dict(DEFAULT_POST_PROCESS)
    pp.update(p.get("post_process") or {})
    p["post_process"] = pp
    ap = dict(DEFAULT_ASSET_POLICY)
    ap.update(p.get("asset_policy") or {})
    p["asset_policy"] = ap
    p.setdefault("quality_preset", recommended_quality_preset(p))
    return p


# Feature 7H - motion-aware presets for garment transformation sequences.
MOTION_PRESETS: list[dict] = [
    {
        "name": "Studio Wardrobe Transformation Turntable",
        "description": "Professional catalog-style turntable sequence for "
                       "wardrobe variation visualization.",
        "loras": ["professional_identity_reference_v1.safetensors:0.78",
                  "structured_garment_layer_v2.safetensors:0.45",
                  "fabric_detail_enhancement_v3.safetensors:0.35"],
        "prompt": "high-end catalog styling, controlled wardrobe variation, "
                  "refined garment rendering, studio lighting continuity",
        "negative": "identity drift, garment boundary artifacts, unstable "
                    "hands, frame-to-frame lighting shift",
        "motion": {"type": "catalog_turntable", "frames": 8, "fps": 12,
                   "pose_guidance": "openpose",
                   "structure_guidance": "depth"},
        "post_process": {"realesrgan": True, "gfpgan": True,
                         "gfpgan_weight": 0.3},
        "recommended_regions": ["upper_garment", "full_body"],
    },
    {
        "name": "Dynamic Layered Editorial Sequence",
        "description": "Layered editorial garment transition sequence with "
                       "refined fabric interaction.",
        "loras": ["professional_identity_reference_v1.safetensors:0.78",
                  "structured_garment_layer_v2.safetensors:0.45"],
        "prompt": "editorial catalog styling, layered garment transition, "
                  "refined fabric interaction, studio lighting continuity",
        "negative": "identity drift, garment boundary artifacts, "
                    "inconsistent fabric edges, frame-to-frame lighting "
                    "shift",
        "motion": {"type": "soft_pose_transition", "frames": 12, "fps": 12,
                   "pose_guidance": "openpose",
                   "structure_guidance": "depth"},
        "post_process": {"realesrgan": True, "gfpgan": True,
                         "gfpgan_weight": 0.3},
        "recommended_regions": ["upper_garment", "full_body"],
    },
    {
        "name": "Fabric Interaction Catalog Flow",
        "description": "Subtle fabric interaction flow for premium catalog "
                       "fabric study.",
        "loras": ["professional_identity_reference_v1.safetensors:0.78",
                  "fabric_detail_enhancement_v3.safetensors:0.40"],
        "prompt": "premium catalog fabric study, soft fabric interaction, "
                  "refined textile structure, soft studio shadows",
        "negative": "identity drift, texture smearing, garment boundary "
                    "artifacts, frame-to-frame lighting shift",
        "motion": {"type": "garment_detail_reveal", "frames": 8, "fps": 10,
                   "pose_guidance": "openpose",
                   "structure_guidance": "depth"},
        "post_process": {"realesrgan": True, "gfpgan": True,
                         "gfpgan_weight": 0.3},
        "recommended_regions": ["upper_garment"],
    },
    {
        "name": "Full Body Catalog Motion Study",
        "description": "Full-body wardrobe variation motion study with "
                       "identity consistency.",
        "loras": ["professional_identity_reference_v1.safetensors:0.80",
                  "structured_garment_layer_v2.safetensors:0.40"],
        "prompt": "full body fashion catalog, controlled wardrobe variation, "
                  "consistent model identity, studio lighting continuity",
        "negative": "identity drift, garment boundary artifacts, distorted "
                    "proportions, frame-to-frame lighting shift",
        "motion": {"type": "full_body_wardrobe_variation", "frames": 12,
                   "fps": 12, "pose_guidance": "openpose",
                   "structure_guidance": "depth"},
        "post_process": {"realesrgan": True, "gfpgan": True,
                         "gfpgan_weight": 0.3},
        "recommended_regions": ["full_body"],
    },
    {
        "name": "Identity-Locked Wardrobe Motion Test",
        "description": "Identity-first wardrobe motion test prioritizing "
                       "strong personal identity consistency.",
        "loras": ["professional_identity_reference_v1.safetensors:0.85",
                  "fabric_detail_enhancement_v3.safetensors:0.30"],
        "prompt": "identity-locked catalog styling, consistent model "
                  "identity, refined garment rendering, editorial studio "
                  "lighting",
        "negative": "identity drift, inconsistent facial structure, garment "
                    "boundary artifacts, frame-to-frame lighting shift",
        "motion": {"type": "soft_pose_transition", "frames": 8, "fps": 12,
                   "pose_guidance": "openpose",
                   "structure_guidance": "depth"},
        "post_process": {"realesrgan": True, "gfpgan": True,
                         "gfpgan_weight": 0.3},
        "recommended_regions": ["upper_garment", "full_body"],
    },
]


def list_motion_presets() -> list[dict]:
    """Return motion-aware sequence presets (with conservative defaults)."""
    return [_with_defaults(p) for p in MOTION_PRESETS]


def get_motion_preset(name: str) -> dict | None:
    """Look up one motion preset by its UI-facing name (case-insensitive)."""
    low = (name or "").strip().lower()
    for p in MOTION_PRESETS:
        if p["name"].lower() == low:
            return _with_defaults(p)
    return None


def list_wardrobe_presets() -> list[dict]:
    """Return the wardrobe/catalog presets (names + metadata) for the UI,
    with conservative post-processing + asset-policy defaults applied."""
    return [_with_defaults(p) for p in WARDROBE_PRESETS]


def get_wardrobe_preset(name: str) -> dict | None:
    """Look up one wardrobe preset by its UI-facing name (case-insensitive)."""
    low = (name or "").strip().lower()
    for p in WARDROBE_PRESETS:
        if p["name"].lower() == low:
            return _with_defaults(p)
    return None


def resolve_wardrobe_preset(name: str, lora_dir=None,
                            library_aware: bool = False) -> dict | None:
    """Resolve a wardrobe OR motion preset's LoRA stack through the preflight
    validator so the UI can show resolved assets, total weight, the prompt
    fragment, and any warnings before generation. Never raises for ordinary
    validation problems.

    When ``library_aware`` is true the preset's roles are first re-mapped onto
    the user's actual library (so the placeholder identity anchor becomes the
    real identity LoRA), then validated."""
    preset = get_wardrobe_preset(name) or get_motion_preset(name)
    if preset is None:
        return None
    loras = preset["loras"]
    if library_aware:
        mapped = resolve_preset_library_aware(preset, lora_dir)
        loras = mapped["loras"]
        preset["library_substitutions"] = mapped["substitutions"]
        preset["unresolved_roles"] = mapped["unresolved_roles"]
    try:
        from .lora_validator import validate_lora_stack
        cleaned, warnings = validate_lora_stack(loras, lora_dir)
    except Exception as exc:                       # pragma: no cover
        cleaned, warnings = ({"items": [], "prompt_fragment": "",
                              "total_weight": 0.0,
                              "valid_pairs": []}, [f"validator unavailable: {exc}"])
    preset["resolved"] = cleaned
    preset["warnings"] = warnings
    return preset


def usable_presets(lora_dir=None, include_motion: bool = True) -> list[dict]:
    """Report which presets the current library can actually run. A preset is
    'usable' when an identity anchor resolves AND at least one concept LoRA
    resolves. Returns lightweight rows for the UI; never raises."""
    rows: list[dict] = []
    names = [p["name"] for p in WARDROBE_PRESETS]
    if include_motion:
        names += [p["name"] for p in MOTION_PRESETS]
    for name in names:
        r = resolve_wardrobe_preset(name, lora_dir, library_aware=True)
        if r is None:
            continue
        pairs = r["resolved"]["valid_pairs"]
        stems = [s for s, _ in pairs]
        has_identity = any(detect_lora_role(s) in ("identity", "character")
                           for s in stems)
        rows.append({
            "name": name,
            "quality_preset": r.get("quality_preset", "catalog_pro"),
            "resolved_count": len(pairs),
            "has_identity": has_identity,
            "usable": bool(has_identity and len(pairs) >= 2),
            "prompt_fragment": r["resolved"]["prompt_fragment"],
            "substitutions": r.get("library_substitutions", []),
        })
    return rows


def recommend_artistic_stacks(cards: list[LoraCard],
                              prj=None) -> list[dict]:
    """Artistic figure study stack recommendations: identity anchor first,
    conservative concept weights, layering assessed through the Identity
    Integration Layer.  Library hints match by substring; missing concept
    LoRAs degrade the template to the layers that exist."""
    from .identity_integration import assess_concept_layering
    identity = _pick_identity(cards)
    by_id = {c.lora_id.lower(): c.lora_id for c in cards}
    out: list[dict] = []
    for name, concept_specs in ARTISTIC_STACK_TEMPLATES:
        layers: list[dict] = []
        if identity is not None:
            layers.append({"name": identity.lora_id,
                           "role": "identity_anchor",
                           "weight": min(max(
                               identity.profile.weight_default, 0.75), 0.90)})
        missing: list[str] = []
        for hint, role, weight in concept_specs:
            match = next((real for low, real in by_id.items()
                          if hint.lower() in low), None)
            if match:
                layers.append({"name": match, "role": role,
                               "weight": weight})
            else:
                missing.append(hint)
        if len(layers) <= (1 if identity else 0):
            continue
        assessed = assess_concept_layering(layers, prj)
        out.append({
            "name": name,
            "available": identity is not None and not missing,
            "stack": [[l["name"], l["weight"]]
                      for l in assessed["adjusted_layers"]],
            "reason_codes": assessed["reason_codes"],
            "warnings": assessed["warnings"]
            + ([f"Library hint not matched: {', '.join(missing)} - "
                "template degraded to available layers."] if missing else [])
            + ([] if identity else
               ["No identity anchor in the library - train or import a "
                "character identity LoRA first."]),
            "identity_score_estimate": assessed["identity_score_estimate"],
            "weight_ranges": {
                "identity_anchor": "0.75-0.90",
                "garment_rendering": "0.35-0.60",
                "sheer_garment_rendering": "0.25-0.50",
                "skin_texture_highlight": "0.20-0.45",
                "pose": "0.25-0.50", "lighting": "0.20-0.40"},
        })
    return out
