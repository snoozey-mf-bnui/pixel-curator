"""Artistic prompt composer - identity-first multi-LoRA prompt recipes.

Composes professional artistic figure study prompts from structured
recipes: Pony-compatible score chains, subject/style/quality fragments,
negative emphasis chains, trigger-word injection (once, up front) and an
ordered LoRA stack with the identity anchor always first.

Weight philosophy mirrors Stack Intelligence: identity stays conservative
and strong (0.75-0.90); concept modifiers are clamped to per-role ranges
and the total concept influence is capped, trimming lowest-priority
layers first.  Every adjustment is explained through warnings so the
stack remains auditable.

Deterministic and dependency-free: no model files, no network, no torch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from ..stack_intelligence import (CONCEPT_STRENGTH_MODERATE,
                                  CONCEPT_STRENGTH_STRONG,
                                  IDENTITY_ANCHOR_MIN)

# Role ordering: who composes first and who yields first when the stack
# must be rebalanced (identity is never auto-reduced).
ROLE_PRIORITY: dict[str, int] = {
    "identity_anchor": 100,
    "character": 90,
    "form_proportion": 62,
    "garment_rendering": 58,
    "sheer_garment_rendering": 54,
    "pose": 50,
    "lighting": 42,
    "skin_texture_highlight": 38,
    "composition": 32,
    "style": 28,
}

# Conservative per-role weight ranges (studio weight philosophy)
ROLE_WEIGHT_RANGES: dict[str, tuple[float, float]] = {
    "identity_anchor": (0.75, 0.90),
    "character": (0.70, 0.88),
    "form_proportion": (0.25, 0.50),
    "garment_rendering": (0.35, 0.60),
    "sheer_garment_rendering": (0.25, 0.50),
    "pose": (0.25, 0.50),
    "lighting": (0.20, 0.40),
    "skin_texture_highlight": (0.20, 0.45),
    "composition": (0.20, 0.40),
    "style": (0.20, 0.45),
}

IDENTITY_ROLES = ("identity_anchor", "character")

# Default Pony-compatible quality chain (positive head)
DEFAULT_SCORE_CHAIN = ["score_9", "score_8_up", "score_7_up"]

# Identity-protective negative emphasis appended to every recipe
IDENTITY_NEGATIVES = [
    "identity drift", "inconsistent facial structure",
    "distorted proportions", "anatomical inaccuracy",
]
BASE_NEGATIVES = [
    "lowres", "bad anatomy", "bad hands", "deformed", "blurry",
    "watermark", "text", "signature", "jpeg artifacts", "worst quality",
]

# ---------------------------------------------------------------------------
# Layered negative control chains (Phase 4)
#
# Named, auditable groups of negative-prompt terms. Recipes declare which
# chains apply; callers can layer additional chains per generation. The
# identity_protection and render_quality chains alias the long-standing
# defaults so the composed negatives remain a superset of pre-Phase-4
# output (backward compatible).
# ---------------------------------------------------------------------------
NEGATIVE_CONTROL_CHAINS: dict[str, list[str]] = {
    "identity_protection": list(IDENTITY_NEGATIVES),
    "render_quality": list(BASE_NEGATIVES),
    "anatomical_coherence": [
        "anatomical inaccuracy", "distorted proportions", "merged limbs",
        "extra limbs", "broken joint articulation", "twisted torso errors",
    ],
    "hand_stability": [
        "malformed hands", "extra fingers", "missing fingers",
        "fused fingers", "hand instability",
    ],
    "skin_texture_control": [
        "plastic skin texture", "over-smoothed skin", "waxy appearance",
        "skin texture banding",
    ],
    "specular_control": [
        "uncontrolled specular highlights", "blown highlights",
        "harsh reflective hotspots", "metallic skin sheen errors",
    ],
    "garment_boundary_control": [
        "garment boundary artifacts", "fabric clipping",
        "misaligned seams", "texture banding", "fabric merge errors",
    ],
    "lighting_control": [
        "flat lighting", "harsh shadow clipping",
        "banded shadow gradients", "inconsistent light direction",
    ],
}

# Chains every figure study composes by default (in addition to the
# recipe's own chains): coherent anatomy + stable hands + clean render +
# identity protection.
DEFAULT_FIGURE_STUDY_CHAINS = [
    "anatomical_coherence", "hand_stability",
    "render_quality", "identity_protection",
]

# Chains that protect identity/structure: under "strict" intensity these
# receive webui-style emphasis weighting.
_CRITICAL_CHAINS = ("identity_protection", "anatomical_coherence",
                    "hand_stability")
NEGATIVE_INTENSITIES = ("standard", "strict")


def list_negative_chains() -> list[dict]:
    """Catalog of available negative control chains for CLI/UI display."""
    return [{"name": name, "terms": list(terms),
             "critical": name in _CRITICAL_CHAINS}
            for name, terms in NEGATIVE_CONTROL_CHAINS.items()]


def compose_negative_chains(chains: list[str],
                            extra_terms: Optional[list[str]] = None,
                            intensity: str = "standard") -> list[str]:
    """Resolve chain names to a deduplicated negative-term list.

    intensity="strict" applies emphasis weighting to terms from
    identity/anatomy/hand chains: (term:1.15).
    """
    if intensity not in NEGATIVE_INTENSITIES:
        raise KeyError(f"unknown negative intensity '{intensity}'. "
                       f"Options: {', '.join(NEGATIVE_INTENSITIES)}")
    out: list[str] = []
    for chain in chains:
        if chain not in NEGATIVE_CONTROL_CHAINS:
            raise KeyError(
                f"unknown negative control chain '{chain}'. "
                f"Options: {', '.join(sorted(NEGATIVE_CONTROL_CHAINS))}")
        for term in NEGATIVE_CONTROL_CHAINS[chain]:
            if intensity == "strict" and chain in _CRITICAL_CHAINS:
                out.append(f"({term}:1.15)")
            else:
                out.append(term)
    out.extend(extra_terms or [])
    return _dedupe(out)


@dataclass
class ArtisticLoraLayer:
    """One LoRA in an artistic stack.  `name` is a library hint - the CLI
    resolves it against the LoRA Explorer scan when available."""
    name: str
    role: str
    weight: float
    trigger_words: list[str] = field(default_factory=list)
    priority: int = 50
    enabled: bool = True
    notes: Optional[str] = None


@dataclass
class ArtisticPromptRecipe:
    name: str
    description: str
    positive_subject: list[str]
    positive_style: list[str]
    positive_quality: list[str]
    negative_terms: list[str]
    pony_score_chain: list[str] = field(
        default_factory=lambda: list(DEFAULT_SCORE_CHAIN))
    lora_layers: list[ArtisticLoraLayer] = field(default_factory=list)
    identity_strength: float = 0.85
    concept_strength: float = 0.65
    # named negative control chains layered on top of negative_terms
    # (render_quality + identity_protection are always applied)
    negative_chains: list[str] = field(default_factory=list)
    # Phase 5 production metadata (all optional, additive)
    guidance_recommendations: list[str] = field(default_factory=list)
    region_policy: str = "full"          # default figure-study region
    variation_axes_default: list[str] = field(default_factory=list)
    batch_defaults: dict = field(default_factory=dict)
    runtime_notes: list[str] = field(default_factory=list)


@dataclass
class ComposedArtisticPrompt:
    prompt: str
    negative_prompt: str
    lora_stack: list[dict]
    trigger_words: list[str]
    warnings: list[str]
    reason_codes: list[str]
    identity_score_estimate: float
    recipe_name: str

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Built-in professional recipes
# ---------------------------------------------------------------------------

def _layer(name: str, role: str, weight: Optional[float] = None,
           triggers: Optional[list[str]] = None,
           notes: Optional[str] = None) -> ArtisticLoraLayer:
    lo, hi = ROLE_WEIGHT_RANGES.get(role, (0.20, 0.45))
    return ArtisticLoraLayer(
        name=name, role=role,
        weight=weight if weight is not None else round((lo + hi) / 2, 2),
        trigger_words=list(triggers or []),
        priority=ROLE_PRIORITY.get(role, 50), notes=notes)


def _identity_layer() -> ArtisticLoraLayer:
    return _layer("identity_anchor", "identity_anchor", 0.85,
                  notes="Resolved to the project identity LoRA at compose "
                        "time; trigger phrase injected from project config.")


RECIPES: dict[str, ArtisticPromptRecipe] = {
    "artistic_figure_study": ArtisticPromptRecipe(
        name="artistic_figure_study",
        description="Identity-preserving fine-art figure study: form, "
                    "lighting treatment and studio composition.",
        positive_subject=["fine art figure study", "detailed anatomy",
                          "coherent proportions", "natural skin texture"],
        positive_style=["chiaroscuro studio lighting", "sculptural shadow",
                        "fine art photography", "studio composition"],
        positive_quality=["realistic rendering", "sharp focus",
                          "gallery print quality"],
        negative_terms=["flat lighting", "plastic skin texture",
                        "over-smoothed skin"],
        negative_chains=['anatomical_coherence', 'hand_stability', 'lighting_control', 'skin_texture_control'],
        lora_layers=[_identity_layer(),
                     _layer("figure_study", "form_proportion"),
                     _layer("studio_lighting", "lighting")],
    ),
    "form_proportion_study": ArtisticPromptRecipe(
        name="form_proportion_study",
        description="Form and proportion study: full-body structure, "
                    "balanced pose energy, neutral treatment.",
        positive_subject=["form and proportion study", "full body framing",
                          "balanced standing pose", "detailed anatomy"],
        positive_style=["neutral studio backdrop", "even soft lighting",
                        "clear silhouette"],
        positive_quality=["realistic rendering", "high detail",
                          "consistent character"],
        negative_terms=["foreshortening errors", "cropped limbs",
                        "distorted perspective"],
        negative_chains=['anatomical_coherence', 'hand_stability'],
        lora_layers=[_identity_layer(),
                     _layer("figure_study", "form_proportion", 0.45)],
        concept_strength=0.70,
    ),
    "lingerie_form_study": ArtisticPromptRecipe(
        name="lingerie_form_study",
        description="Lingerie form study: garment structure and fit on "
                    "the figure with strong identity preservation.",
        positive_subject=["lingerie form study", "garment structure and fit",
                          "fabric interaction with form",
                          "detailed anatomy"],
        positive_style=["soft directional lighting", "editorial styling",
                        "studio composition"],
        positive_quality=["realistic rendering", "fine fabric detail",
                          "consistent character"],
        negative_terms=["garment artifacts", "fabric clipping",
                        "misaligned seams"],
        negative_chains=['garment_boundary_control', 'anatomical_coherence', 'hand_stability'],
        lora_layers=[_identity_layer(),
                     _layer("lingerie_form_study", "garment_rendering"),
                     _layer("studio_lighting", "lighting", 0.25)],
    ),
    "sheer_garment_rendering": ArtisticPromptRecipe(
        name="sheer_garment_rendering",
        description="Sheer garment rendering: translucent fabric "
                    "behavior, layering and light interaction.",
        positive_subject=["sheer garment rendering", "translucent fabric",
                          "delicate fabric layering", "garment draping"],
        positive_style=["backlit rim lighting", "soft tonal transitions",
                        "editorial styling"],
        positive_quality=["realistic rendering", "fine fabric detail",
                          "consistent character"],
        negative_terms=["fabric artifacts", "opaque rendering errors",
                        "texture banding"],
        negative_chains=['garment_boundary_control', 'specular_control'],
        lora_layers=[_identity_layer(),
                     _layer("sheer_clothing", "sheer_garment_rendering"),
                     _layer("studio_lighting", "lighting", 0.30)],
        concept_strength=0.60,
    ),
    "glossy_skin_treatment": ArtisticPromptRecipe(
        name="glossy_skin_treatment",
        description="Skin texture highlight refinement: glossy skin "
                    "treatment with controlled specular highlights.",
        positive_subject=["skin texture highlight refinement",
                          "glossy skin treatment", "natural skin detail",
                          "specular highlight control"],
        positive_style=["sculptural studio lighting", "controlled rim light",
                        "fine art photography"],
        positive_quality=["realistic rendering", "pore-level detail",
                          "consistent character"],
        negative_terms=["plastic skin texture", "over-smoothed skin",
                        "blown highlights", "waxy appearance"],
        negative_chains=['skin_texture_control', 'specular_control'],
        lora_layers=[_identity_layer(),
                     _layer("wet_glossy_skin", "skin_texture_highlight"),
                     _layer("studio_lighting", "lighting", 0.30)],
        concept_strength=0.55,
    ),
    "boudoir_editorial_pose": ArtisticPromptRecipe(
        name="boudoir_editorial_pose",
        description="Boudoir editorial pose study: expressive pose "
                    "energy with intimate editorial lighting treatment.",
        positive_subject=["boudoir editorial pose", "expressive pose energy",
                          "elegant posing", "detailed anatomy"],
        positive_style=["intimate editorial lighting", "soft shadow falloff",
                        "studio composition"],
        positive_quality=["realistic rendering", "editorial quality",
                          "consistent character"],
        negative_terms=["stiff posing", "awkward limb placement",
                        "unnatural articulation"],
        negative_chains=['anatomical_coherence', 'hand_stability', 'lighting_control'],
        lora_layers=[_identity_layer(),
                     _layer("boudoir_pose", "pose"),
                     _layer("studio_lighting", "lighting", 0.25)],
        concept_strength=0.60,
    ),
    "editorial_fashion_study": ArtisticPromptRecipe(
        name="editorial_fashion_study",
        description="Editorial fashion study: garment variation, styling "
                    "language and editorial posing.",
        positive_subject=["editorial fashion study", "garment styling",
                          "wardrobe detail", "editorial posing"],
        positive_style=["high fashion lighting treatment",
                        "clean studio composition", "editorial framing"],
        positive_quality=["realistic rendering", "textile detail",
                          "consistent character"],
        negative_terms=["garment artifacts", "styling inconsistencies"],
        negative_chains=['garment_boundary_control', 'hand_stability'],
        lora_layers=[_identity_layer(),
                     _layer("fashion_editorial", "garment_rendering", 0.40),
                     _layer("boudoir_pose", "pose", 0.25)],
    ),
}


# ---------------------------------------------------------------------------
# Phase 5: Professional Figure Study Production recipes
#
# Production-oriented recipes with guidance recommendations, region
# policies, default variation axes and batch defaults. MPS note: keep
# preview batches small; CUDA note: larger batch planning is safe.
# ---------------------------------------------------------------------------
_MPS_NOTE = ("MPS: conservative batch sizes (4-6) and one refinement "
             "pass at a time.")
_CUDA_NOTE = ("CUDA: larger production batches are fine; keep active "
              "models on local storage for speed.")

RECIPES.update({
    "advanced_figure_study": ArtisticPromptRecipe(
        name="advanced_figure_study",
        description="Advanced professional figure study: pose and "
                    "composition guidance, form and proportion "
                    "stability, full layered negative protection.",
        positive_subject=["advanced figure study", "detailed anatomy",
                          "coherent proportions",
                          "pose and composition guidance"],
        positive_style=["chiaroscuro studio lighting",
                        "editorial framing", "sculptural shadow"],
        positive_quality=["realistic rendering", "gallery print quality",
                          "consistent character"],
        negative_terms=["flat lighting", "compositional imbalance"],
        negative_chains=["anatomical_coherence", "hand_stability",
                         "lighting_control", "skin_texture_control"],
        lora_layers=[_identity_layer(),
                     _layer("figure_study", "form_proportion"),
                     _layer("studio_lighting", "lighting"),
                     _layer("boudoir_pose", "pose", 0.30)],
        guidance_recommendations=["pose", "structure"],
        region_policy="full",
        variation_axes_default=["pose_energy", "lighting_treatment",
                                "camera_framing"],
        batch_defaults={"count": 6, "passes": ["composition", "review"]},
        runtime_notes=[_MPS_NOTE, _CUDA_NOTE],
    ),
    "identity_locked_figure_study": ArtisticPromptRecipe(
        name="identity_locked_figure_study",
        description="Identity-locked figure study: FaceID identity lock "
                    "recommended, maximum character consistency, "
                    "minimal concept load.",
        positive_subject=["professional figure study",
                          "identity preservation",
                          "character consistency", "detailed anatomy"],
        positive_style=["neutral studio backdrop", "even soft lighting"],
        positive_quality=["realistic rendering", "consistent character",
                          "sharp focus"],
        negative_terms=[],
        negative_chains=["anatomical_coherence", "hand_stability"],
        lora_layers=[_identity_layer(),
                     _layer("studio_lighting", "lighting", 0.25)],
        identity_strength=0.90,
        concept_strength=0.45,
        guidance_recommendations=["pose"],
        region_policy="full",
        batch_defaults={"count": 4,
                        "passes": ["composition", "identity", "review"]},
        runtime_notes=[_MPS_NOTE],
    ),
    "form_proportion_refinement": ArtisticPromptRecipe(
        name="form_proportion_refinement",
        description="Form and proportion refinement pass: depth-guided "
                    "structural stability at conservative denoise.",
        positive_subject=["form and proportion study",
                          "structural refinement pass",
                          "coherent proportions", "detailed anatomy"],
        positive_style=["clear silhouette", "even soft lighting"],
        positive_quality=["realistic rendering", "high detail",
                          "consistent character"],
        negative_terms=["foreshortening errors", "distorted perspective"],
        negative_chains=["anatomical_coherence", "hand_stability"],
        lora_layers=[_identity_layer(),
                     _layer("figure_study", "form_proportion", 0.45)],
        concept_strength=0.70,
        guidance_recommendations=["structure"],
        region_policy="full",
        batch_defaults={"count": 4, "passes": ["composition", "review"]},
        runtime_notes=[_MPS_NOTE],
    ),
    "garment_rendering_refinement": ArtisticPromptRecipe(
        name="garment_rendering_refinement",
        description="Garment rendering refinement pass: boundary "
                    "stability, seam alignment and fabric detail on the "
                    "torso region.",
        positive_subject=["garment rendering refinement",
                          "garment structure and fit",
                          "fine fabric detail", "refinement pass"],
        positive_style=["soft directional lighting", "editorial styling"],
        positive_quality=["realistic rendering", "textile detail",
                          "consistent character"],
        negative_terms=["garment artifacts"],
        negative_chains=["garment_boundary_control", "hand_stability"],
        lora_layers=[_identity_layer(),
                     _layer("lingerie_form_study", "garment_rendering",
                            0.45)],
        guidance_recommendations=["edges"],
        region_policy="torso",
        batch_defaults={"count": 4,
                        "passes": ["composition", "garment_refine",
                                   "review"]},
        runtime_notes=[_MPS_NOTE],
    ),
    "sheer_fabric_interaction": ArtisticPromptRecipe(
        name="sheer_fabric_interaction",
        description="Sheer fabric interaction study: translucent fabric "
                    "behavior, layering and backlit light interaction "
                    "with controlled specular response.",
        positive_subject=["sheer fabric interaction", "translucent fabric",
                          "delicate fabric layering",
                          "fabric interaction with form"],
        positive_style=["backlit rim lighting", "soft tonal transitions",
                        "editorial styling"],
        positive_quality=["realistic rendering", "fine fabric detail",
                          "consistent character"],
        negative_terms=["fabric artifacts", "opaque rendering errors"],
        negative_chains=["garment_boundary_control", "specular_control"],
        lora_layers=[_identity_layer(),
                     _layer("sheer_clothing", "sheer_garment_rendering"),
                     _layer("studio_lighting", "lighting", 0.30)],
        concept_strength=0.60,
        guidance_recommendations=["edges"],
        region_policy="torso",
        batch_defaults={"count": 6, "passes": ["composition",
                                               "garment_refine",
                                               "review"]},
        runtime_notes=[_MPS_NOTE],
    ),
    "skin_texture_highlight_refinement": ArtisticPromptRecipe(
        name="skin_texture_highlight_refinement",
        description="Skin texture highlight refinement pass: glossy "
                    "skin treatment with specular highlight control at "
                    "low denoise.",
        positive_subject=["skin texture highlight refinement",
                          "glossy skin treatment",
                          "specular highlight control",
                          "natural skin detail"],
        positive_style=["sculptural studio lighting",
                        "controlled rim light"],
        positive_quality=["realistic rendering", "pore-level detail",
                          "consistent character"],
        negative_terms=["blown highlights"],
        negative_chains=["skin_texture_control", "specular_control"],
        lora_layers=[_identity_layer(),
                     _layer("wet_glossy_skin", "skin_texture_highlight"),
                     _layer("studio_lighting", "lighting", 0.30)],
        concept_strength=0.55,
        guidance_recommendations=["structure"],
        region_policy="torso",
        batch_defaults={"count": 4, "passes": ["composition",
                                               "skin_refine", "review"]},
        runtime_notes=[_MPS_NOTE],
    ),
    "studio_lighting_refinement": ArtisticPromptRecipe(
        name="studio_lighting_refinement",
        description="Studio lighting treatment refinement: shadow "
                    "gradation, light direction consistency and "
                    "controlled contrast.",
        positive_subject=["studio lighting treatment", "refinement pass",
                          "controlled contrast", "shadow gradation"],
        positive_style=["dramatic single-source lighting",
                        "sculptural shadow", "fine art photography"],
        positive_quality=["realistic rendering", "gallery print quality",
                          "consistent character"],
        negative_terms=[],
        negative_chains=["lighting_control", "specular_control"],
        lora_layers=[_identity_layer(),
                     _layer("studio_lighting", "lighting", 0.35)],
        concept_strength=0.50,
        guidance_recommendations=["structure"],
        region_policy="full",
        batch_defaults={"count": 4, "passes": ["composition", "review"]},
        runtime_notes=[_MPS_NOTE],
    ),
    "editorial_pose_refinement": ArtisticPromptRecipe(
        name="editorial_pose_refinement",
        description="Editorial pose refinement: OpenPose-guided pose "
                    "and composition preservation with expressive "
                    "posing energy.",
        positive_subject=["editorial posing", "pose refinement pass",
                          "expressive pose energy",
                          "pose and composition guidance"],
        positive_style=["intimate editorial lighting",
                        "studio composition"],
        positive_quality=["realistic rendering", "editorial quality",
                          "consistent character"],
        negative_terms=["stiff posing", "awkward limb placement"],
        negative_chains=["anatomical_coherence", "hand_stability"],
        lora_layers=[_identity_layer(),
                     _layer("boudoir_pose", "pose", 0.40),
                     _layer("studio_lighting", "lighting", 0.25)],
        concept_strength=0.60,
        guidance_recommendations=["pose"],
        region_policy="full",
        variation_axes_default=["pose_energy", "camera_framing"],
        batch_defaults={"count": 6,
                        "passes": ["composition", "pose_variation",
                                   "review"]},
        runtime_notes=[_MPS_NOTE],
    ),
    "composition_guided_figure_study": ArtisticPromptRecipe(
        name="composition_guided_figure_study",
        description="Composition-guided figure study: framing, balance "
                    "and negative-space control with structural "
                    "guidance.",
        positive_subject=["composition-guided figure study",
                          "balanced framing", "negative space control",
                          "detailed anatomy"],
        positive_style=["clean studio composition", "editorial framing",
                        "fine art photography"],
        positive_quality=["realistic rendering", "gallery print quality",
                          "consistent character"],
        negative_terms=["compositional imbalance", "cropped limbs"],
        negative_chains=["anatomical_coherence", "lighting_control"],
        lora_layers=[_identity_layer(),
                     _layer("figure_study", "form_proportion", 0.35),
                     _layer("studio_lighting", "lighting", 0.25)],
        guidance_recommendations=["pose", "structure"],
        region_policy="full",
        variation_axes_default=["camera_framing", "lighting_treatment"],
        batch_defaults={"count": 6, "passes": ["composition", "review"]},
        runtime_notes=[_MPS_NOTE, _CUDA_NOTE],
    ),
    "region_refinement_study": ArtisticPromptRecipe(
        name="region_refinement_study",
        description="Selective region refinement study: region-aware "
                    "img2img passes with identity regions untouched.",
        positive_subject=["selective region editing",
                          "region-aware refinement", "refinement pass",
                          "detailed anatomy"],
        positive_style=["soft directional lighting",
                        "studio composition"],
        positive_quality=["realistic rendering", "consistent character"],
        negative_terms=[],
        negative_chains=["anatomical_coherence",
                         "garment_boundary_control"],
        lora_layers=[_identity_layer(),
                     _layer("figure_study", "form_proportion", 0.30)],
        concept_strength=0.50,
        guidance_recommendations=["structure", "edges"],
        region_policy="torso",
        batch_defaults={"count": 4,
                        "passes": ["composition", "garment_refine",
                                   "skin_refine", "review"]},
        runtime_notes=[_MPS_NOTE],
    ),
})


def list_recipes() -> list[dict]:
    return [{"name": r.name, "description": r.description,
             "layers": [{"name": l.name, "role": l.role, "weight": l.weight}
                        for l in r.lora_layers]}
            for r in RECIPES.values()]


def get_recipe(name: str) -> ArtisticPromptRecipe:
    if name not in RECIPES:
        raise KeyError(
            f"unknown artistic recipe '{name}'. "
            f"Options: {', '.join(sorted(RECIPES))}")
    return RECIPES[name]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def _dedupe(fragments: list[str]) -> list[str]:
    """Order-preserving, case-insensitive fragment de-duplication.

    Emphasis-normalized: "(term:1.15)" and "term" are the same fragment;
    the first occurrence (typically the emphasized one) wins."""
    seen: set[str] = set()
    out: list[str] = []
    for f in fragments:
        key = f.strip().lower()
        if key.startswith("(") and key.endswith(")") and ":" in key:
            key = key[1:key.rindex(":")].strip()
        if key and key not in seen:
            seen.add(key)
            out.append(f.strip())
    return out


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compose_artistic_prompt(
        recipe: ArtisticPromptRecipe,
        trigger_phrase: str = "",
        identity_strength: Optional[float] = None,
        concept_strength: Optional[float] = None,
        strength: float = 1.0,
        identity_lora_name: str = "",
        pony_prefix: bool = True,
        negative_chains: Optional[list[str]] = None,
        extra_negative_terms: Optional[list[str]] = None,
        negative_intensity: str = "standard") -> ComposedArtisticPrompt:
    """Compose prompt + negative + ordered LoRA stack from a recipe.

    `strength` scales concept-layer weights globally (the CLI --strength);
    identity is governed only by `identity_strength` and is never scaled
    down by concept controls.

    Phase 4 layered negative controls: the recipe's `negative_chains`
    plus any caller-supplied `negative_chains` resolve through
    NEGATIVE_CONTROL_CHAINS; `negative_intensity="strict"` emphasizes
    identity/anatomy/hand protection terms. The classic base + identity
    negatives are always present (backward compatible).
    """
    warnings: list[str] = []
    codes: list[str] = []

    id_strength = _clamp(
        identity_strength if identity_strength is not None
        else recipe.identity_strength, 0.0, 1.2)
    c_strength = _clamp(
        concept_strength if concept_strength is not None
        else recipe.concept_strength, 0.0, 1.2)

    # ----- layers: identity first, then by priority -----
    layers = [l for l in recipe.lora_layers if l.enabled]
    layers.sort(key=lambda l: (-l.priority, l.name))

    stack: list[dict] = []
    triggers: list[str] = []
    if trigger_phrase:
        triggers.append(trigger_phrase.strip())

    id_lo, id_hi = ROLE_WEIGHT_RANGES["identity_anchor"]
    concept_total = 0.0
    for l in layers:
        if l.role in IDENTITY_ROLES:
            w = _clamp(id_strength, id_lo, id_hi)
            if id_strength < IDENTITY_ANCHOR_MIN:
                warnings.append(
                    f"Identity strength {id_strength:.2f} below the "
                    f"{IDENTITY_ANCHOR_MIN:.2f} floor - raised to "
                    f"{w:.2f} to hold character consistency.")
                codes.append("identity_anchor_raised_to_floor")
            name = identity_lora_name or l.name
        else:
            lo, hi = ROLE_WEIGHT_RANGES.get(l.role, (0.20, 0.45))
            # concept_strength scales relative to the recipe baseline, so
            # recipe defaults land inside the recommended role ranges
            scale = (c_strength / recipe.concept_strength
                     if recipe.concept_strength else 1.0)
            w = _clamp(l.weight * scale * strength, 0.0, hi)
            if w < lo * 0.5:
                codes.append("concept_layer_near_floor")
            concept_total += w
            name = l.name
        for t in l.trigger_words:
            triggers.append(t)
        stack.append({"name": name, "role": l.role, "weight": round(w, 2),
                      "priority": l.priority,
                      "notes": l.notes or ""})

    # ----- concept influence cap: trim lowest priority first -----
    cap = CONCEPT_STRENGTH_MODERATE
    if concept_total > cap:
        excess = round(concept_total - cap, 4)
        for item in sorted((s for s in stack
                            if s["role"] not in IDENTITY_ROLES),
                           key=lambda s: s["priority"]):
            lo, _hi = ROLE_WEIGHT_RANGES.get(item["role"], (0.20, 0.45))
            floor = round(lo * 0.5, 2)
            cut = min(max(item["weight"] - floor, 0.0), excess)
            if cut > 0:
                item["weight"] = round(item["weight"] - cut, 2)
                excess = round(excess - cut, 4)
                warnings.append(
                    f"'{item['name']}' trimmed by {cut:.2f} to keep total "
                    f"concept influence within the safe range "
                    f"({cap:.2f}).")
            if excess <= 1e-9:
                break
        codes.append("concept_stack_normalized")
        concept_total = sum(s["weight"] for s in stack
                            if s["role"] not in IDENTITY_ROLES)

    # ----- identity preservation estimate -----
    has_identity = any(s["role"] in IDENTITY_ROLES for s in stack)
    score = 1.0
    if not has_identity:
        score -= 0.35
        warnings.append("No identity anchor layer in this recipe - "
                        "character consistency is not guaranteed.")
        codes.append("identity_anchor_missing")
    else:
        codes.append("identity_anchor_preserved")
    score -= 0.15 * max(0.0, concept_total - 0.6)
    score -= 0.05 * max(0, len([s for s in stack
                                if s["role"] not in IDENTITY_ROLES]) - 2)
    score = round(max(0.05, min(1.0, score)), 3)
    if concept_total > CONCEPT_STRENGTH_STRONG:
        codes.append("concept_stack_high_identity_risk")
        warnings.append(
            "Concept stack remains strong after trimming - reduce concept "
            "strength if identity preservation falls below threshold.")
    elif has_identity:
        codes.append("concept_stack_within_safe_range")

    # ----- prompt assembly -----
    triggers = _dedupe(triggers)
    head = list(recipe.pony_score_chain) if pony_prefix else []
    fragments = _dedupe(
        head + triggers + recipe.positive_subject
        + ["identity-preserving composition", "character consistency"]
        + recipe.positive_style + recipe.positive_quality)
    prompt = ", ".join(fragments)

    chain_names = list(recipe.negative_chains) + list(negative_chains or [])
    if negative_intensity == "strict":
        # strict intensity guarantees the critical protection chains
        chain_names += list(_CRITICAL_CHAINS)
    chain_names = _dedupe(chain_names)
    chain_terms: list[str] = []
    if chain_names:
        chain_terms = compose_negative_chains(
            chain_names, intensity=negative_intensity)
        codes.append(f"negative_chains_applied:{len(chain_names)}")
    # chain terms first so strict-emphasized forms win de-duplication;
    # base + identity negatives always close the chain (back-compat)
    negative = ", ".join(_dedupe(
        chain_terms + list(recipe.negative_terms)
        + list(extra_negative_terms or [])
        + BASE_NEGATIVES + IDENTITY_NEGATIVES))

    return ComposedArtisticPrompt(
        prompt=prompt, negative_prompt=negative, lora_stack=stack,
        trigger_words=triggers, warnings=warnings, reason_codes=codes,
        identity_score_estimate=score, recipe_name=recipe.name)


def lora_tag_string(composed: ComposedArtisticPrompt) -> str:
    """webui-convention <lora:name:weight> tags, identity first."""
    return "".join(f" <lora:{s['name']}:{s['weight']:g}>"
                   for s in composed.lora_stack if s["weight"] > 0)
