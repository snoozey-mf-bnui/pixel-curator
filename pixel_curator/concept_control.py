"""Concept Control - attribute sliders mapped onto LoRA stack weights.

Defines the professional attribute controls (Identity Anchor Strength,
Garment Style Intensity, Lighting Mood, ...), resolves slider state +
selected LoRAs into a concrete, intelligence-checked stack, and manages
concept control presets (manifest table `concept_control_presets`, v8).

A slider scales every selected LoRA in its linked families between the
family's recommended range: value 0.0 -> weight_min, 1.0 -> weight_max
(identity runs 0.0 -> 0.5..1.0 of its range so the anchor never collapses).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from .lora_explorer import FAMILY_WEIGHT_RANGES, LoraCard
from .stack_intelligence import ResolvedLoraStack, resolve_stack
from .util import now_iso


# ---------------------------------------------------------------------------
# Concept family registry - professional aliases -> canonical families
# ---------------------------------------------------------------------------
# The studio vocabulary (identity_anchor, garment_style, lighting_mood, ...)
# maps onto the canonical families used by profiles and weight ranges.
# Unknown families fall back to conservative style-range behavior.

FAMILY_ALIASES: dict[str, str] = {
    "identity_anchor": "identity", "character_style": "character",
    "garment_style": "wardrobe", "material_finish": "texture",
    "lighting_mood": "lighting", "pose_energy": "pose",
    "form_emphasis": "pose", "composition_style": "composition",
    "environment_style": "environment", "camera_treatment": "camera",
    "rendering_style": "style", "detail_enhancement": "detail",
}

UNKNOWN_FAMILY_RANGE = (0.10, 0.30)     # conservative fallback (studio-safe)
HARD_CAP_FACTOR = 1.15                  # absolute ceiling over recommended max

# response curves: identity is linear inside its protected band; concept
# families ease in (slow start) so small slider moves stay subtle, and
# flatten near the top so the last 20% of travel cannot overdrive.
def _ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)      # smoothstep


def canonical_family(family: str) -> str:
    f = (family or "").strip().lower()
    if f in FAMILY_WEIGHT_RANGES:
        return f
    return FAMILY_ALIASES.get(f, "unknown")


def family_range(family: str) -> tuple[float, float]:
    return FAMILY_WEIGHT_RANGES.get(canonical_family(family),
                                    UNKNOWN_FAMILY_RANGE)


@dataclass
class SliderMappingResult:
    """Explainable slider -> weight mapping (UI-stable schema)."""
    family: str
    slider_value: float
    raw_weight: float          # before caps / identity-aware scaling
    resolved_weight: float
    soft_capped: bool = False
    hard_capped: bool = False
    identity_scaled: bool = False
    curve: str = "smoothstep"
    note: str = ""


def map_slider_to_weight(slider_value: float, family: str,
                         identity_context: Optional[dict] = None
                         ) -> SliderMappingResult:
    """Intelligent mapping: family operating range + response curve +
    soft/hard caps + identity-aware damping.

    identity_context (optional): {"anchor_weight": float,
    "total_concept_strength": float} - when the stack is already carrying
    heavy concept influence, additional concept weight is damped so the
    identity anchor keeps headroom.
    """
    fam = canonical_family(family)
    lo, hi = family_range(fam)
    t = max(0.0, min(1.0, float(slider_value)))
    if fam in ("identity", "character"):
        # protected band: never below half-range; linear response
        w = raw = lo + (hi - lo) * (0.5 + t / 2.0)
        curve = "linear_protected"
    else:
        w = raw = lo + (hi - lo) * _ease(t)
        curve = "smoothstep"
    res = SliderMappingResult(family=fam, slider_value=t, raw_weight=round(raw, 3),
                              resolved_weight=round(w, 3), curve=curve)
    if fam not in ("identity", "character") and identity_context:
        total = float(identity_context.get("total_concept_strength") or 0.0)
        anchor = float(identity_context.get("anchor_weight") or 0.0)
        if total > 0.8 or (anchor and anchor < 0.65):
            damp = 0.85 if total <= 1.2 else 0.7
            w = w * damp
            res.identity_scaled = True
            res.note = ("damped to preserve identity headroom "
                        f"(x{damp:.2f})")
    if w > hi:
        w, res.soft_capped = hi, True
    hard = hi * HARD_CAP_FACTOR
    if w > hard:
        w, res.hard_capped = hard, True
    res.resolved_weight = round(w, 3)
    return res


def clamp_to_recommended_range(weight: float, family: str) -> float:
    lo, hi = family_range(family)
    return round(max(lo, min(float(weight), hi)), 3)


def sync_slider_to_numeric(slider_value: float, family: str) -> float:
    """Slider position (0-1) -> weight shown in the numeric input."""
    return map_slider_to_weight(slider_value, family).resolved_weight


def sync_numeric_to_slider(weight: float, family: str) -> float:
    """Weight typed in the numeric input -> slider position (0-1).
    Inverse of the mapping curve, clamped to the operating range."""
    fam = canonical_family(family)
    lo, hi = family_range(fam)
    if hi <= lo:
        return 0.5
    w = max(lo, min(float(weight), hi))
    frac = (w - lo) / (hi - lo)
    if fam in ("identity", "character"):
        return round(max(0.0, min(1.0, (frac - 0.5) * 2.0)), 3)
    # invert smoothstep numerically (monotonic - bisection is exact enough)
    a, b = 0.0, 1.0
    for _ in range(40):
        m = (a + b) / 2
        if _ease(m) < frac:
            a = m
        else:
            b = m
    return round((a + b) / 2, 3)


def suggest_safer_slider_value(slider_value: float, family: str) -> float:
    """Conservative pull-back for aggressive settings: returns the largest
    slider value whose mapped weight sits at/below the midpoint of the
    family's recommended range (identity is never pulled down)."""
    fam = canonical_family(family)
    if fam in ("identity", "character"):
        return max(float(slider_value), 0.5)
    lo, hi = family_range(fam)
    mid = lo + (hi - lo) * 0.5
    return min(float(slider_value), sync_numeric_to_slider(mid, fam))


@dataclass
class ConceptSlider:
    slider_id: str
    label: str
    families: list[str]
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.05
    default: float = 0.5
    explanation: str = ""
    identity_note: str = ""           # shown when raising it risks identity


CONCEPT_SLIDERS: list[ConceptSlider] = [
    ConceptSlider("identity_anchor_strength", "Identity Anchor Strength",
                  ["identity", "character"], default=0.6,
                  explanation="How firmly the character identity is held. "
                              "Maps onto the identity LoRA weight "
                              "(0.65-0.85 recommended)."),
    ConceptSlider("garment_style_intensity", "Garment Style Intensity",
                  ["wardrobe", "fashion"], default=0.5,
                  explanation="Wardrobe styling and garment structure "
                              "presence.",
                  identity_note="Above ~0.7 garment LoRAs begin to restyle "
                                "the face - verify with a consistency test."),
    ConceptSlider("form_emphasis", "Form Emphasis",
                  ["pose"], default=0.5,
                  explanation="Pose energy and form/proportion emphasis."),
    ConceptSlider("lighting_mood", "Lighting Mood",
                  ["lighting"], default=0.4,
                  explanation="Lighting atmosphere strength."),
    ConceptSlider("style_intensity", "Style Intensity",
                  ["style"], default=0.3,
                  explanation="Overall visual signature of style LoRAs.",
                  identity_note="Style LoRAs carry the highest identity "
                                "risk - keep low for character work."),
    ConceptSlider("texture_emphasis", "Texture Emphasis",
                  ["texture"], default=0.4,
                  explanation="Fabric and surface texture presence."),
    ConceptSlider("detail_refinement", "Detail Refinement",
                  ["detail", "refinement"], default=0.5,
                  explanation="Fine detail density and refinement passes."),
    ConceptSlider("composition_strength", "Composition Strength",
                  ["composition", "camera"], default=0.4,
                  explanation="Framing, camera perspective and composition "
                              "flow influence."),
    ConceptSlider("background_influence", "Background Influence",
                  ["environment"], default=0.3,
                  explanation="Scene/environment context strength."),
]

SLIDER_INDEX = {s.slider_id: s for s in CONCEPT_SLIDERS}


@dataclass
class ConceptSliderState:
    """slider_id -> 0..1 value (UI state)."""
    values: dict[str, float] = field(default_factory=dict)

    def value(self, slider_id: str) -> float:
        s = SLIDER_INDEX[slider_id]
        v = float(self.values.get(slider_id, s.default))
        return max(s.minimum, min(s.maximum, v))


def slider_specs() -> list[dict]:
    """UI-ready control specs."""
    return [asdict(s) for s in CONCEPT_SLIDERS]


def _family_slider(family: str) -> Optional[ConceptSlider]:
    for s in CONCEPT_SLIDERS:
        if family in s.families:
            return s
    return None


def resolve_concept_weights(cards: list[LoraCard],
                            state: ConceptSliderState,
                            overrides: Optional[dict[str, float]] = None
                            ) -> dict[str, float]:
    """Slider state -> per-LoRA weights via the intelligent mapping
    (response curves, caps, identity-aware damping).  `overrides` are
    pinned/manual weights (numeric input, Apply Suggested Balance) - they
    bypass slider mapping but still pass through the stack resolver's
    safety checks.  Two-pass: anchors first so the identity context is
    known before concept families are mapped."""
    overrides = overrides or {}
    weights: dict[str, float] = {}
    anchor_weight = 0.0
    concept_first_pass = 0.0
    for c in cards:                                    # pass 1: anchors
        fam = canonical_family(c.profile.family)
        if fam in ("identity", "character"):
            if c.lora_id in overrides:
                weights[c.lora_id] = round(float(overrides[c.lora_id]), 2)
            else:
                s = _family_slider(c.profile.family)
                t = state.value(s.slider_id) if s else 0.5
                weights[c.lora_id] = round(map_slider_to_weight(
                    t, fam).resolved_weight, 2)
            anchor_weight = max(anchor_weight, weights[c.lora_id])
        else:
            s = _family_slider(c.profile.family)
            t = state.value(s.slider_id) if s else 0.5
            concept_first_pass += map_slider_to_weight(
                t, fam).resolved_weight
    ctx = {"anchor_weight": anchor_weight,
           "total_concept_strength": concept_first_pass}
    for c in cards:                                    # pass 2: concepts
        if c.lora_id in weights:
            continue
        if c.lora_id in overrides:
            weights[c.lora_id] = round(float(overrides[c.lora_id]), 2)
            continue
        fam = canonical_family(c.profile.family)
        s = _family_slider(c.profile.family)
        t = state.value(s.slider_id) if s else 0.5
        weights[c.lora_id] = round(map_slider_to_weight(
            t, fam, identity_context=ctx).resolved_weight, 2)
    return weights


def resolve_controlled_stack(cards: list[LoraCard],
                             state: ConceptSliderState,
                             base_model: Optional[str] = None,
                             overrides: Optional[dict[str, float]] = None,
                             pinned: Optional[set] = None
                             ) -> ResolvedLoraStack:
    """Sliders + selection -> intelligence-checked stack (never overdrives:
    resolve_stack normalizes and warns).  Overridden LoRAs are treated as
    pinned unless stated otherwise."""
    pin = set(pinned) if pinned is not None else set(overrides or {})
    return resolve_stack(cards,
                         resolve_concept_weights(cards, state, overrides),
                         base_model=base_model, pinned=pin)


def variation_axes() -> list[dict]:
    """Sliders exposed as variation axes (Batch Variation Controller).
    Identity impact comes from the worst-risk family the slider drives."""
    from .lora_explorer import FAMILY_IDENTITY_RISK
    rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    out = []
    for s in CONCEPT_SLIDERS:
        risks = [FAMILY_IDENTITY_RISK.get(f, "medium") for f in s.families]
        impact = max(risks, key=lambda r: rank.get(r, 2)) if risks else "medium"
        if s.slider_id == "identity_anchor_strength":
            impact = "protected"     # swept only within the safe band
        out.append({"slider": s.slider_id, "label": s.label,
                    "families": list(s.families),
                    "minimum": s.minimum, "maximum": s.maximum,
                    "step": s.step, "default": s.default,
                    "identity_impact": impact,
                    "explanation": s.explanation})
    return out


# ---------------------------------------------------------------------------
# Presets (manifest v8: concept_control_presets)
# ---------------------------------------------------------------------------

def save_preset(conn, name: str, kind: str, payload: dict) -> None:
    """kind: 'slider_state' | 'lora_stack' | 'variation_grid'."""
    conn.execute(
        "INSERT INTO concept_control_presets (name, kind, payload, "
        "updated_at) VALUES (?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
        "kind=excluded.kind, payload=excluded.payload, "
        "updated_at=excluded.updated_at",
        (name, kind, json.dumps(payload), now_iso()))
    conn.commit()


def load_presets(conn, kind: str = "") -> list[dict]:
    q = "SELECT name, kind, payload, updated_at FROM concept_control_presets"
    args: tuple = ()
    if kind:
        q += " WHERE kind = ?"
        args = (kind,)
    out = []
    try:
        for name, k, payload, ts in conn.execute(q + " ORDER BY name", args):
            try:
                body = json.loads(payload)
            except Exception:
                body = {}
            out.append({"name": name, "kind": k, "payload": body,
                        "updated_at": ts})
    except Exception:
        pass        # pre-v8 manifest: no presets yet (backward compatible)
    return out
