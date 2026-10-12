"""Visual Focus Classification & Routing - shared service.

A deterministic, metadata-only routing layer that sits on top of Florence-2
captions/attributes + existing quality signals and recommends which specialized
training subset each image belongs to. It loads NO images and NO models - it
reasons purely over already-computed metadata, so it is fast, testable, and
import-safe.

Distinct from ``pixel_curator.curate.visual_focus_classifier`` (the pixel/CV
classifier): this module is the *routing* layer and writes its enrichment under
the explorer key ``metadata["routing"]``.

Professional, SFW, dataset-engineering taxonomy only. Never infers age/identity/
sensitive attributes. When ``# age safety handled below (NSFW mode - only from explicit metadata)`` is present (only from
an explicit upstream flag), specialized routing is blocked and the recipe is
forced to ``review``. Source media is never touched.
"""
from __future__ import annotations
try:
    from pixel_curator import nsfw_router
except ImportError:
    nsfw_router = None

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

CLASSIFIER_VERSION = "visual_focus_classifier_v1"

# ---- approved taxonomy ----
PRIMARY_FOCUS_LABELS = (
    "portrait_focus", "head_and_shoulders_focus", "upper_body_composition",
    "torso_composition", "lower_body_composition", "full_body_composition",
    "full_body_pose", "close_crop_study", "figure_study_composition",
    "outfit_focus", "gesture_pose", "dynamic_pose_focus", "lighting_composition",
    "environmental_composition", "texture_detail_focus", "hands_detail_focus",
    "feet_detail_focus", "unknown", "review_required")

DETAIL_LEVEL_LABELS = (
    "general_reference", "moderate_detail", "high_detail_study",
    "specialized_texture_focus", "low_detail", "unknown", "review_required")

POSE_LABELS = (
    "neutral_pose", "neutral_standing", "neutral_seated", "relaxed_pose",
    "dynamic_pose", "arched_or_extended_pose", "reclined_pose",
    "gesture_focused_pose", "walking_or_motion_pose", "close_crop_expressive",
    "portrait_expression", "symmetry_pose", "asymmetry_pose", "unknown",
    "review_required")

ROUTING_TARGETS = (
    "core_character_likeness", "portrait_reference", "upper_body_reference",
    "lower_body_reference", "full_body_reference", "full_body_pose_reference",
    "dynamic_pose_reference", "outfit_reference", "texture_detail_reference",
    "lighting_composition_reference", "captioning_reference", "dataset_review",
    "reject_low_quality")

RECIPE_TYPES = (
    "character_core", "portrait", "upper_body", "lower_body", "full_body",
    "pose_dynamic", "outfit", "detail_texture", "lighting_composition",
    "captioning", "review", "reject")

ROUTING_TAGS = (
    "core_candidate", "portrait_candidate", "upper_body_candidate",
    "lower_body_candidate", "full_body_candidate", "dynamic_pose_candidate",
    "outfit_candidate", "detail_texture_candidate", "lighting_reference_candidate",
    "composition_reference_candidate", "captioning_candidate", "review_required",
    "low_quality", "age_uncertain_review", "classification_uncertain")

SPECIALIZED_TARGETS = (
    "core_character_likeness", "portrait_reference", "upper_body_reference",
    "lower_body_reference", "full_body_reference", "full_body_pose_reference",
    "dynamic_pose_reference", "outfit_reference", "texture_detail_reference",
    "lighting_composition_reference")

TARGET_TO_RECIPE = {
    "core_character_likeness": "character_core", "portrait_reference": "portrait",
    "upper_body_reference": "upper_body", "lower_body_reference": "lower_body",
    "full_body_reference": "full_body", "full_body_pose_reference": "pose_dynamic",
    "dynamic_pose_reference": "pose_dynamic", "outfit_reference": "outfit",
    "texture_detail_reference": "detail_texture",
    "lighting_composition_reference": "lighting_composition",
    "captioning_reference": "captioning", "dataset_review": "review",
    "reject_low_quality": "reject"}

# recipe_type -> candidate project recipe names (first existing one wins)
RECIPE_TYPE_TO_PROJECT = {
    "character_core": ["character_v1"], "portrait": ["character_v1"],
    "upper_body": ["outfit_v1"], "lower_body": ["pose_v1"],
    "full_body": ["pose_v1"], "pose_dynamic": ["pose_v1"],
    "outfit": ["outfit_v1"], "detail_texture": ["detail_v1"],
    "lighting_composition": ["style_v1"], "captioning": [],
    "review": [], "reject": []}

RESULTS_NAME = "visual_focus_classifications.jsonl"
SUMMARY_NAME = "visual_focus_classification_summary.json"
CARD_NAME = "visual_focus_routing_card.md"


# ============================ dataclasses ===================================
@dataclass
class VisualFocusClassifierConfig:
    classifier_version: str = CLASSIFIER_VERSION
    min_primary_confidence: float = 0.45
    min_routing_confidence: float = 0.50
    excellent_score_threshold: float = 85.0
    good_score_threshold: float = 70.0
    review_score_threshold: float = 50.0
    block_specialized_routing_on_age_uncertain: bool = True
    use_florence_caption: bool = True
    use_florence_attributes: bool = True
    use_quality_signals: bool = True
    use_existing_dataset_metadata: bool = True
    allow_specialized_detail_routing: bool = True


@dataclass
class FocusPrediction:
    label: str
    confidence: float
    evidence: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 4),
                "evidence": list(self.evidence)}


@dataclass
class RoutingRecommendation:
    target: str
    confidence: float
    recipe_type: str = ""
    reason: str = ""
    blocked: bool = False
    block_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {"target": self.target, "confidence": round(self.confidence, 4),
                "recipe_type": self.recipe_type, "reason": self.reason,
                "blocked": self.blocked, "block_reason": self.block_reason}


@dataclass
class VisualFocusClassificationResult:
    image_path: str
    classifier_version: str
    primary_focus: FocusPrediction
    detail_level: FocusPrediction = field(
        default_factory=lambda: FocusPrediction("unknown", 0.0))
    pose_type: FocusPrediction = field(
        default_factory=lambda: FocusPrediction("unknown", 0.0))
    secondary_focuses: list = field(default_factory=list)
    recommended_targets: list = field(default_factory=list)
    suggested_recipe_type: str = "review"
    routing_confidence: float = 0.0
    suitability_score: float = 0.0
    suitability_tier: str = "review"
    routing_tags: list = field(default_factory=list)
    visual_summary: str = ""
    quality_summary: dict = field(default_factory=dict)
    florence_summary: dict = field(default_factory=dict)
    review_flags: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "classifier_version": self.classifier_version,
            "primary_focus": self.primary_focus.to_dict(),
            "detail_level": self.detail_level.to_dict(),
            "pose_type": self.pose_type.to_dict(),
            "secondary_focuses": [s.to_dict() for s in self.secondary_focuses],
            "recommended_targets": [r.to_dict() for r in self.recommended_targets],
            "suggested_recipe_type": self.suggested_recipe_type,
            "routing_confidence": round(self.routing_confidence, 4),
            "suitability_score": round(self.suitability_score, 2),
            "suitability_tier": self.suitability_tier,
            "routing_tags": list(self.routing_tags),
            "visual_summary": self.visual_summary,
            "quality_summary": self.quality_summary,
            "florence_summary": self.florence_summary,
            "review_flags": list(self.review_flags),
            "warnings": list(self.warnings), "metadata": self.metadata}


# ============================ helpers =======================================
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _extract_florence(src: Optional[dict]) -> dict:
    """Normalize varied Florence shapes into a flat dict of useful fields."""
    if not src:
        return {}
    fl = src
    # explorer record may nest under 'florence'
    if isinstance(src.get("florence"), dict):
        fl = {**src, **src["florence"]}
    attrs = fl.get("visual_attributes") if isinstance(
        fl.get("visual_attributes"), dict) else {}
    caption = (fl.get("caption") or fl.get("natural_caption")
               or fl.get("caption_natural") or attrs.get("natural_caption") or "")
    return {
        "caption": caption,
        "present": bool(caption) or bool(fl.get("florence_processed"))
        or bool(fl.get("ok")),
        "source": fl.get("caption_natural_source")
        or ("florence2" if fl.get("ok") else ""),
        "primary_focus": attrs.get("primary_focus") or fl.get("primary_focus"),
        "pose_type": attrs.get("pose_type") or fl.get("pose_type"),
        "detail_level": attrs.get("detail_level") or fl.get("detail_level"),
        "lighting_quality": attrs.get("lighting_quality")
        or fl.get("lighting_quality"),
        "composition_type": attrs.get("composition_type")
        or fl.get("composition_type"),
        "routing_tags": fl.get("routing_tags") or attrs.get("routing_tags") or [],
        "flags": fl.get("flags") or fl.get("florence_flags") or [],
        "model_variant": fl.get("model_variant") or fl.get("florence_model_variant"),
    }


def _extract_quality(src: Optional[dict]) -> dict:
    if not src:
        return {}
    q = src
    if isinstance(src.get("quality_signals"), dict):
        q = {**src, **src["quality_signals"]}
    keys = ("overall_quality_score", "integrity", "sharpness", "local_sharpness",
            "exposure_quality", "composition_quality", "region_clarity",
            "region_balance_score", "lighting_quality", "figure_quality",
            "natural_texture", "texture_preservation_score", "hand_quality",
            "body_proportion", "motion_blur", "motion_blur_score",
            "background_quality", "color_cast", "aesthetic_score",
            "caption_readiness")
    out = {k: _num(q.get(k)) for k in keys if q.get(k) is not None}
    for k in ("coverage", "framing", "duplicate_status", "pose_duplicate"):
        if q.get(k) is not None:
            out[k] = q.get(k)
    return out


# Florence focus vocab -> primary focus label (+ secondary hints)
_FOCUS_MAP = {
    "portrait": "portrait_focus", "head_and_shoulders": "head_and_shoulders_focus",
    "upper_body": "upper_body_composition", "torso": "torso_composition",
    "full_body": "full_body_composition", "outfit": "outfit_focus",
    "detail_texture": "texture_detail_focus", "composition": "environmental_composition",
}
_CAPTION_FOCUS_RULES = [
    ("full_body_composition", ("full body", "full-body", "head to toe",
                               "standing figure", "entire figure")),
    ("upper_body_composition", ("upper body", "waist up", "from the waist")),
    ("portrait_focus", ("portrait", "headshot", "head and shoulders", "close-up "
                        "of the face")),
    ("outfit_focus", ("outfit", "fashion", "garment", "dress", "clothing",
                      "jacket", "ensemble")),
    ("texture_detail_focus", ("close-up", "macro", "fine texture", "fabric detail",
                              "intricate detail")),
    ("environmental_composition", ("environment", "scene", "background", "setting",
                                   "landscape")),
    ("lighting_composition", ("studio lighting", "lighting setup", "rim light",
                              "dramatic lighting")),
]
_DYNAMIC_POSE_KEYS = ("dynamic", "walking", "running", "mid-stride", "gesturing",
                      "reaching", "in motion", "leaping")


def _infer_primary(fl: dict, q: dict, caption: str) -> FocusPrediction:
    evidence = []
    label, conf = "unknown", 0.3
    fl_focus = fl.get("primary_focus")
    if fl_focus and fl_focus in _FOCUS_MAP:
        label = _FOCUS_MAP[fl_focus]
        conf = 0.78
        evidence.append(f"Florence attributes indicate {fl_focus} composition.")
    elif caption:
        low = caption.lower()
        for lab, keys in _CAPTION_FOCUS_RULES:
            if any(k in low for k in keys):
                label, conf = lab, 0.66
                evidence.append("Florence caption describes "
                                f"{lab.replace('_', ' ')}.")
                break
    # quality framing fallback
    if label == "unknown":
        framing = (q.get("framing") or q.get("coverage") or "")
        fmap = {"full_body": "full_body_composition",
                "upper_body": "upper_body_composition",
                "portrait": "portrait_focus", "closeup": "close_crop_study",
                "torso_focus": "torso_composition"}
        if framing in fmap:
            label, conf = fmap[framing], 0.55
            evidence.append(f"Quality framing metadata indicates {framing}.")
    # dynamic-pose elevation
    pose = (fl.get("pose_type") or "").lower()
    if (pose in ("dynamic", "gesture")) or any(
            k in (caption or "").lower() for k in _DYNAMIC_POSE_KEYS):
        if label in ("full_body_composition", "unknown"):
            label = "full_body_pose"
            conf = max(conf, 0.7)
        evidence.append("Pose metadata indicates dynamic body positioning.")
    return FocusPrediction(label, conf, evidence)


def _secondary(fl: dict, q: dict, caption: str, primary: str) -> list:
    out = []
    low = (caption or "").lower()
    if primary != "outfit_focus" and any(
            k in low for k in ("outfit", "garment", "dress", "fabric", "clothing")):
        out.append(FocusPrediction("outfit_focus", 0.58,
                   ["Caption references outfit shape / visible fabric detail."]))
    if primary not in ("texture_detail_focus", "close_crop_study") and (
            (fl.get("detail_level") in ("high", "specialized_texture_study"))
            or "texture" in low):
        out.append(FocusPrediction("texture_detail_focus", 0.55,
                   ["High fine-detail / texture visibility."]))
    if primary != "lighting_composition" and (
            fl.get("lighting_quality") in ("good", "excellent")):
        out.append(FocusPrediction("lighting_composition", 0.5,
                   ["Lighting quality is good."]))
    return out[:3]


def _technical_quality(q: dict) -> Optional[float]:
    vals = []
    for k in ("overall_quality_score", "integrity", "sharpness", "local_sharpness",
              "region_clarity", "region_balance_score", "lighting_quality",
              "composition_quality", "aesthetic_score", "figure_quality"):
        v = _num(q.get(k))
        if v is not None:
            vals.append(_clamp01(v))
    return sum(vals) / len(vals) if vals else None


def _review_flags(fl: dict, q: dict, has_fl: bool, has_q: bool,
                  primary: FocusPrediction, cfg) -> list:
    flags = []
    src_flags = [str(f).lower() for f in (fl.get("flags") or [])]
    if "# age safety handled below (NSFW mode - only from explicit metadata)" in src_flags:
        flags.append("# age safety handled below (NSFW mode - only from explicit metadata)")
    # Propagate upstream age-uncertain safety flag so _route() can gate
    # specialized routing on it (set by florence_captioning when the
    # source metadata marks the image as age_uncertain).
    if "review_required_age_uncertain" in src_flags:
        flags.append("review_required_age_uncertain")
    if not has_fl:
        flags.append("florence_missing")
    if not has_q:
        flags.append("quality_signals_missing")
    if not fl.get("caption"):
        flags.append("caption_missing")
    tq = _technical_quality(q)
    if tq is not None and tq < 0.45:
        flags.append("low_quality")
    if _num(q.get("region_clarity")) is not None and q["region_clarity"] < 0.4:
        flags.append("low_region_clarity")
    mb = _num(q.get("motion_blur")) or _num(q.get("motion_blur_score"))
    if mb is not None and mb > 0.55:
        flags.append("motion_blur")
    lq = _num(q.get("lighting_quality"))
    if lq is not None and lq < 0.4:
        flags.append("poor_lighting")
    if q.get("duplicate_status") == "duplicate":
        flags.append("duplicate_candidate")
    pd = q.get("pose_duplicate")
    if isinstance(pd, dict) and pd.get("near_duplicate"):
        flags.append("pose_duplicate_candidate")
    if primary.confidence < cfg.min_primary_confidence:
        flags.append("classification_uncertain")
    return sorted(set(flags))


def _suitability(q: dict, fl: dict, primary: FocusPrediction,
                 routing_conf: float, flags: list, cfg) -> tuple:
    tq = _technical_quality(q)
    technical = tq if tq is not None else 0.45
    caption_clarity = 0.0
    cap = fl.get("caption") or ""
    if cap:
        caption_clarity = _clamp01(0.4 + min(len(cap.split()), 30) / 60.0)
    focus_spec = primary.confidence if primary.label not in (
        "unknown", "review_required") else 0.2
    pose_use = 0.6 if (fl.get("pose_type") in ("dynamic", "gesture")
                       or primary.label in ("full_body_pose", "dynamic_pose_focus",
                                            "gesture_pose")) else 0.45
    score = 100 * (0.35 * technical + 0.20 * caption_clarity
                   + 0.20 * focus_spec + 0.15 * pose_use + 0.10 * routing_conf)
    penalty = 0.0
    for f, pts in (("motion_blur", 16), ("low_quality", 18),
                   ("low_region_clarity", 12), ("poor_lighting", 10),
                   ("overexposed", 10), ("underexposed", 10),
                   ("duplicate_candidate", 10), ("pose_duplicate_candidate", 8),
                   ("classification_uncertain", 8), ("caption_missing", 4),
                   ("# age safety handled below (NSFW mode - only from explicit metadata)", 100),
                   ("corrupt_image", 100), ("unsupported_format", 100)):
        if f in flags:
            penalty += pts
    score = max(0.0, min(100.0, score - penalty))
    if "# age safety handled below (NSFW mode - only from explicit metadata)" in flags:
        return 0.0, "reject"
    if score >= cfg.excellent_score_threshold:
        tier = "excellent"
    elif score >= cfg.good_score_threshold:
        tier = "good"
    elif score >= cfg.review_score_threshold:
        tier = "review"
    elif score >= 30:
        tier = "low"
    else:
        tier = "reject"
    return score, tier


_FOCUS_TO_TARGETS = {
    "portrait_focus": [("portrait_reference", 0.8), ("core_character_likeness", 0.6)],
    "head_and_shoulders_focus": [("portrait_reference", 0.78),
                                 ("core_character_likeness", 0.6)],
    "upper_body_composition": [("upper_body_reference", 0.75)],
    "torso_composition": [("upper_body_reference", 0.6)],
    "lower_body_composition": [("lower_body_reference", 0.7)],
    "full_body_composition": [("full_body_reference", 0.8)],
    "full_body_pose": [("full_body_pose_reference", 0.82),
                       ("dynamic_pose_reference", 0.7)],
    "dynamic_pose_focus": [("dynamic_pose_reference", 0.8)],
    "gesture_pose": [("dynamic_pose_reference", 0.72)],
    "outfit_focus": [("outfit_reference", 0.78)],
    "texture_detail_focus": [("texture_detail_reference", 0.78)],
    "close_crop_study": [("texture_detail_reference", 0.66)],
    "figure_study_composition": [("full_body_reference", 0.6)],
    "hands_detail_focus": [("texture_detail_reference", 0.6)],
    "feet_detail_focus": [("texture_detail_reference", 0.6)],
    "lighting_composition": [("lighting_composition_reference", 0.72)],
    "environmental_composition": [("lighting_composition_reference", 0.62)],
}
_FOCUS_TO_TAG = {
    "portrait_focus": "portrait_candidate",
    "head_and_shoulders_focus": "portrait_candidate",
    "upper_body_composition": "upper_body_candidate",
    "torso_composition": "upper_body_candidate",
    "lower_body_composition": "lower_body_candidate",
    "full_body_composition": "full_body_candidate",
    "full_body_pose": "dynamic_pose_candidate",
    "dynamic_pose_focus": "dynamic_pose_candidate",
    "gesture_pose": "dynamic_pose_candidate",
    "outfit_focus": "outfit_candidate",
    "texture_detail_focus": "detail_texture_candidate",
    "close_crop_study": "detail_texture_candidate",
    "lighting_composition": "lighting_reference_candidate",
    "environmental_composition": "composition_reference_candidate",
}


def _route(primary, secondary, score, tier, flags, cfg):
    # `review_required_age_uncertain` is set upstream by florence_captioning.py
    # when md["age_uncertain"] is True, so checking the flag here is sufficient
    # (and avoids referencing `md`, which is not in scope for _route).
    age_block = ("review_required_age_uncertain" in flags
                 and cfg.block_specialized_routing_on_age_uncertain)
    recs, tags = [], []
    focuses = [primary] + list(secondary)
    if age_block:
        # show what the targets WOULD be, but mark them all blocked + route review
        best = {}
        for fp in focuses:
            for tgt, base in _FOCUS_TO_TARGETS.get(fp.label, []):
                conf = round(_clamp01(base * 0.6 + fp.confidence * 0.4), 4)
                best[tgt] = RoutingRecommendation(
                    target=tgt, confidence=conf,
                    recipe_type=TARGET_TO_RECIPE.get(tgt, "review"),
                    reason="specialized routing blocked pending age review",
                    blocked=True,
                    block_reason="age_uncertain - specialized routing blocked")
        recs = sorted(best.values(), key=lambda r: -r.confidence)
        return recs, ["age_uncertain_review", "review_required"], "review", 0.0
    if tier == "reject":
        recs.append(RoutingRecommendation(
            "reject_low_quality", 0.9, "reject",
            "Suitability below usable threshold."))
        tags.append("low_quality")
    else:
        for fp in focuses:
            for tgt, base in _FOCUS_TO_TARGETS.get(fp.label, []):
                conf = round(_clamp01(base * 0.6 + fp.confidence * 0.4), 4)
                blocked = age_block and tgt in SPECIALIZED_TARGETS
                recs.append(RoutingRecommendation(
                    target=tgt, confidence=conf,
                    recipe_type=TARGET_TO_RECIPE.get(tgt, "review"),
                    reason=f"{fp.label.replace('_', ' ')} with "
                           f"{'adequate' if tier in ('good', 'excellent') else 'limited'} "
                           "technical quality.",
                    blocked=blocked,
                    block_reason="age_uncertain - specialized routing blocked"
                    if blocked else None))
            tag = _FOCUS_TO_TAG.get(fp.label)
            if tag:
                tags.append(tag)
        if not age_block and tier in ("good", "excellent"):
            recs.append(RoutingRecommendation(
                "captioning_reference", round(score / 100.0, 4), "captioning",
                "Clear, training-ready composition."))
            tags.append("captioning_candidate")
    # dedupe recs by target keeping highest confidence
    best = {}
    for r in recs:
        if r.target not in best or r.confidence > best[r.target].confidence:
            best[r.target] = r
    recs = sorted(best.values(), key=lambda r: -r.confidence)
    # routing tags + review handling
    if age_block:
        tags = ["age_uncertain_review", "review_required"]
    if "classification_uncertain" in flags:
        tags.append("classification_uncertain")
    if tier == "review" or primary.label in ("unknown", "review_required"):
        tags.append("review_required")
    seen, dedup = set(), []
    for t in tags:
        if t in ROUTING_TAGS and t not in seen:
            seen.add(t); dedup.append(t)
    # suggested recipe
    if age_block or tier == "review" or not recs:
        recipe = "review"
    elif tier == "reject":
        recipe = "reject"
    else:
        recipe = recs[0].recipe_type or "review"
    unblocked = [r for r in recs if not r.blocked]
    routing_conf = max((r.confidence for r in unblocked), default=0.0)
    return recs, dedup, recipe, routing_conf


def _infer_detail(fl: dict, q: dict, caption: str, cfg) -> FocusPrediction:
    """Detail/texture tier from Florence detail_level + quality texture/sharpness."""
    ev = []
    fl_detail = (fl.get("detail_level") or "").lower()
    low = (caption or "").lower()
    tex = (_num(q.get("natural_texture")) or _num(q.get("texture_preservation_score")))
    sharp = (_num(q.get("sharpness")) or _num(q.get("local_sharpness"))
             or _num(q.get("region_clarity")))
    score = None
    parts = [p for p in (tex, sharp) if p is not None]
    if parts:
        score = sum(parts) / len(parts)
    # Florence hint
    if fl_detail in ("high", "specialized_texture_study"):
        ev.append("Florence indicates high fine-detail / texture visibility.")
        label = ("specialized_texture_focus" if "texture" in low or fl_detail ==
                 "specialized_texture_study" else "high_detail_study")
        return FocusPrediction(label, 0.74, ev)
    if any(k in low for k in ("fine detail", "fine texture", "sharp detail",
                              "intricate", "close crop", "macro")):
        ev.append("Caption emphasizes fine surface / texture detail.")
        return FocusPrediction("high_detail_study", 0.66, ev)
    if score is not None:
        if score >= 0.75:
            ev.append(f"Texture/sharpness signals strong ({score:.2f}).")
            return FocusPrediction("high_detail_study", _clamp01(score), ev)
        if score >= 0.5:
            ev.append("Texture clarity visible but not close-crop dominant.")
            return FocusPrediction("moderate_detail", _clamp01(score), ev)
        if score >= 0.3:
            ev.append("Moderate-to-low surface clarity.")
            return FocusPrediction("general_reference", _clamp01(1 - score), ev)
        ev.append("Low surface clarity / soft detail.")
        return FocusPrediction("low_detail", _clamp01(1 - score), ev)
    if fl_detail == "moderate":
        return FocusPrediction("moderate_detail", 0.55,
                               ["Florence indicates moderate detail."])
    if caption:
        return FocusPrediction("general_reference", 0.5,
                               ["General reference clarity (no strong detail cue)."])
    return FocusPrediction("unknown", 0.3, [])


_POSE_LABEL_MAP = {
    "seated": "neutral_seated", "sitting": "neutral_seated",
    "reclined": "reclined_pose", "lying": "reclined_pose",
    "dynamic": "dynamic_pose", "gesture": "gesture_focused_pose",
    "walking": "walking_or_motion_pose", "neutral": "neutral_standing",
    "close_crop": "close_crop_expressive",
}
_POSE_CAPTION_RULES = [
    ("walking_or_motion_pose", ("walking", "running", "mid-stride", "in motion",
                                "striding")),
    ("dynamic_pose", ("dynamic", "leaping", "jumping", "energetic")),
    ("arched_or_extended_pose", ("arched", "extended", "stretched", "reaching")),
    ("gesture_focused_pose", ("gesturing", "pointing", "raised arm", "hand raised")),
    ("reclined_pose", ("reclining", "lying", "lying down")),
    ("neutral_seated", ("seated", "sitting")),
    ("portrait_expression", ("expression", "smiling", "gaze", "looking at")),
    ("neutral_standing", ("standing", "upright", "neutral stance")),
]


def _infer_pose(fl: dict, q: dict, caption: str, primary: str) -> FocusPrediction:
    pose_attr = (fl.get("pose_type") or "").lower()
    low = (caption or "").lower()
    if pose_attr in _POSE_LABEL_MAP:
        return FocusPrediction(_POSE_LABEL_MAP[pose_attr], 0.72,
                               [f"Florence pose attribute: {pose_attr}."])
    for label, keys in _POSE_CAPTION_RULES:
        if any(k in low for k in keys):
            return FocusPrediction(label, 0.62,
                                   [f"Caption indicates {label.replace('_', ' ')}."])
    if primary in ("portrait_focus", "head_and_shoulders_focus"):
        return FocusPrediction("portrait_expression", 0.55,
                               ["Portrait framing implies portrait expression."])
    if primary in ("close_crop_study", "texture_detail_focus"):
        return FocusPrediction("close_crop_expressive", 0.5,
                               ["Close-crop framing."])
    if primary in ("full_body_composition", "full_body_pose"):
        return FocusPrediction("neutral_standing", 0.5,
                               ["Full-body framing, no dynamic cue detected."])
    return FocusPrediction("unknown", 0.35, [])


def _visual_summary(primary, detail, pose, q: dict) -> str:
    """Professional, SFW one-line composition summary."""
    f = primary.label.replace("_", " ")
    parts = [f]
    if pose.label not in ("unknown", "review_required"):
        parts.append(pose.label.replace("_", " "))
    if detail.label not in ("unknown", "review_required"):
        parts.append(f"{detail.label.replace('_', ' ')} clarity")
    lq = _num(q.get("lighting_quality"))
    light = ("clean lighting" if (lq is not None and lq >= 0.65)
             else "balanced lighting" if lq is not None else None)
    if light:
        parts.append(light)
    text = ", ".join(parts)
    return ("A " + text + ".").replace("A figure study composition",
                                        "A figure-study composition")


# ============================ public API ====================================
def classify_visual_focus(image_path, florence_result=None, quality_signals=None,
                          dataset_metadata=None,
                          config: Optional[VisualFocusClassifierConfig] = None
                          ) -> VisualFocusClassificationResult:
    """Classify one image from its metadata (no image/model loading)."""
    cfg = config or VisualFocusClassifierConfig()
    fl = _extract_florence(florence_result) if cfg.use_florence_attributes \
        or cfg.use_florence_caption else {}
    q = _extract_quality(quality_signals) if cfg.use_quality_signals else {}
    dm = dataset_metadata or {}
    caption = fl.get("caption", "") if cfg.use_florence_caption else ""
    warnings = []
    if not fl:
        warnings.append("no Florence output supplied")
    if not q:
        warnings.append("no quality signals supplied")

    primary = _infer_primary(fl, q, caption)
    detail = _infer_detail(fl, q, caption, cfg)
    pose = _infer_pose(fl, q, caption, primary.label)
    secondary = _secondary(fl, q, caption, primary.label)
    flags = _review_flags(fl, q, bool(fl), bool(q), primary, cfg)
    # provisional routing confidence for scoring
    _, _, _, prov_conf = _route(primary, secondary, 60.0, "good", flags, cfg)
    score, tier = _suitability(q, fl, primary, prov_conf, flags, cfg)
    recs, tags, recipe, routing_conf = _route(primary, secondary, score, tier,
                                              flags, cfg)
    summary = _visual_summary(primary, detail, pose, q)

    return VisualFocusClassificationResult(
        image_path=str(image_path), classifier_version=cfg.classifier_version,
        primary_focus=primary, detail_level=detail, pose_type=pose,
        secondary_focuses=secondary,
        recommended_targets=recs, suggested_recipe_type=recipe,
        routing_confidence=routing_conf, suitability_score=score,
        suitability_tier=tier, routing_tags=tags, visual_summary=summary,
        quality_summary={k: q[k] for k in (
            "overall_quality_score", "integrity", "sharpness", "lighting_quality",
            "region_clarity", "region_balance_score") if k in q},
        florence_summary={
            "caption_present": bool(caption),
            "caption_natural_source": fl.get("source", ""),
            "primary_focus": fl.get("primary_focus"),
            "pose_type": fl.get("pose_type"),
            "detail_level": fl.get("detail_level")},
        review_flags=flags, warnings=warnings,
        metadata={"source": "visual_focus_classifier", "processed_at": _now(),
                  "dataset": {k: dm.get(k) for k in ("recipe", "version", "stage")
                              if dm.get(k) is not None}})


def classify_visual_focus_record(record, config=None
                                 ) -> VisualFocusClassificationResult:
    """Classify from a single merged record (explorer row / florence jsonl line)."""
    rec = record or {}
    image_path = (rec.get("image_path") or rec.get("path")
                  or rec.get("filename") or "")
    florence = rec.get("florence") or rec
    quality = rec.get("quality_signals") or rec.get("quality") or rec
    dataset = {"recipe": rec.get("recipe"),
               "version": (rec.get("version_membership") or [None])[0]
               if rec.get("version_membership") else rec.get("version"),
               "stage": rec.get("stage")}
    return classify_visual_focus(image_path, florence_result=florence,
                                 quality_signals=quality, dataset_metadata=dataset,
                                 config=config)


def classify_visual_focus_batch(records, output_jsonl=None, config=None,
                                overwrite: bool = False) -> dict:
    cfg = config or VisualFocusClassifierConfig()
    out_path = Path(output_jsonl) if output_jsonl else None
    done = set()
    if out_path and out_path.exists():
        if overwrite:
            try:
                out_path.unlink()
            except Exception:
                pass
        else:
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        done.add(json.loads(line).get("image_path"))
                    except Exception:
                        pass
    results = []
    fh = None
    try:
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fh = out_path.open("a", encoding="utf-8")
        for rec in records:
            res = classify_visual_focus_record(rec, cfg)
            if res.image_path in done:
                continue
            results.append(res)
            if fh:
                fh.write(json.dumps(res.to_dict()) + "\n")
                fh.flush()
    finally:
        if fh:
            fh.close()
    summary = summarize_visual_focus_classifications(
        out_path if out_path else [r.to_dict() for r in results])
    return {"processed": len(results),
            "output_jsonl": str(out_path) if out_path else None,
            "summary": summary}


def load_visual_focus_classifications(path) -> list:
    p = Path(path)
    if p.is_dir():
        p = p / RESULTS_NAME
    out = []
    if not p.exists():
        return out
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    except Exception:
        pass
    return out


def summarize_visual_focus_classifications(path) -> dict:
    results = path if isinstance(path, list) else load_visual_focus_classifications(path)
    total = len(results)

    def dist(getter):
        d = {}
        for r in results:
            try:
                k = getter(r)
            except Exception:
                k = "unknown"
            if isinstance(k, list):
                for x in k:
                    d[x] = d.get(x, 0) + 1
            else:
                d[k] = d.get(k, 0) + 1
        return dict(sorted(d.items(), key=lambda kv: -kv[1]))
    scores = [r.get("suitability_score", 0) for r in results]
    return {
        "total": total, "generated_at": _now(),
        "mean_suitability": round(sum(scores) / total, 2) if total else 0.0,
        "primary_focus_distribution": dist(lambda r: r["primary_focus"]["label"]),
        "detail_level_distribution": dist(
            lambda r: (r.get("detail_level") or {}).get("label", "unknown")),
        "pose_distribution": dist(
            lambda r: (r.get("pose_type") or {}).get("label", "unknown")),
        "recipe_type_distribution": dist(lambda r: r["suggested_recipe_type"]),
        "routing_tag_distribution": dist(lambda r: r.get("routing_tags", [])),
        "recommended_target_distribution": dist(
            lambda r: [t["target"] for t in r.get("recommended_targets", [])]),
        "tier_distribution": dist(lambda r: r["suitability_tier"]),
        "review_flag_distribution": dist(lambda r: r.get("review_flags", []) or ["none"]),
    }


def render_routing_card(summary: dict) -> str:
    lines = ["# Visual Focus Routing Card", "",
             f"Classified: **{summary.get('total', 0)}**  |  mean suitability "
             f"**{summary.get('mean_suitability', 0)}**", ""]
    for title, key in (("Primary focus", "primary_focus_distribution"),
                       ("Detail level", "detail_level_distribution"),
                       ("Pose type", "pose_distribution"),
                       ("Suggested recipe types", "recipe_type_distribution"),
                       ("Recommended targets", "recommended_target_distribution"),
                       ("Routing tags", "routing_tag_distribution"),
                       ("Suitability tiers", "tier_distribution"),
                       ("Review flags", "review_flag_distribution")):
        lines.append(f"## {title}")
        for k, v in (summary.get(key) or {}).items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def map_recipe_to_project(recipe_type: str, known_recipes=None) -> dict:
    """Map a routing recipe_type to an existing project recipe name (or warn)."""
    known = set(known_recipes or [])
    for cand in RECIPE_TYPE_TO_PROJECT.get(recipe_type, []):
        if not known or cand in known:
            return {"recipe_type": recipe_type, "project_recipe": cand,
                    "exists": (cand in known) if known else None, "warning": None}
    return {"recipe_type": recipe_type, "project_recipe": None, "exists": False,
            "warning": f"no project recipe maps to '{recipe_type}'"}


def result_to_dataset_explorer_fields(result: VisualFocusClassificationResult) -> dict:
    return {
        "primary_focus": result.primary_focus.label,
        "detail_level": result.detail_level.label,
        "pose_type": result.pose_type.label,
        "secondary_focuses": [s.label for s in result.secondary_focuses],
        "routing_tags": list(result.routing_tags),
        "recommended_targets": [r.target for r in result.recommended_targets
                                if not r.blocked],
        "suggested_recipe_type": result.suggested_recipe_type,
        "routing_confidence": round(result.routing_confidence, 4),
        "suitability_score": round(result.suitability_score, 2),
        "suitability_tier": result.suitability_tier,
        "visual_summary": result.visual_summary,
        "review_flags": list(result.review_flags),
        "classifier_version": result.classifier_version,
    }


def result_to_caption_metadata(result: VisualFocusClassificationResult) -> dict:
    return {"visual_focus": {
        "primary_focus": result.primary_focus.label,
        "detail_level": result.detail_level.label,
        "pose_type": result.pose_type.label,
        "routing_tags": list(result.routing_tags),
        "recommended_targets": [r.target for r in result.recommended_targets
                                if not r.blocked],
        "suggested_recipe_type": result.suggested_recipe_type,
        "routing_confidence": round(result.routing_confidence, 4),
        "suitability_score": round(result.suitability_score, 2),
        "review_flags": list(result.review_flags)}}


def results_by_filename(path) -> dict:
    out = {}
    for r in load_visual_focus_classifications(path):
        name = Path(r.get("image_path", "")).name
        if name:
            out[name] = {
                "primary_focus": r["primary_focus"]["label"],
                "detail_level": (r.get("detail_level") or {}).get("label"),
                "pose_type": (r.get("pose_type") or {}).get("label"),
                "routing_tags": r.get("routing_tags", []),
                "recommended_targets": [t["target"] for t in
                                        r.get("recommended_targets", [])
                                        if not t.get("blocked")],
                "suggested_recipe_type": r.get("suggested_recipe_type"),
                "routing_confidence": r.get("routing_confidence"),
                "suitability_score": r.get("suitability_score"),
                "visual_summary": r.get("visual_summary", ""),
                "review_flags": r.get("review_flags", [])}
    return out


def write_visual_focus_summary(results_jsonl, output_dir) -> dict:
    """Write the summary JSON + routing card from a results JSONL. Returns the
    {name: Path} of written files. Never raises for ordinary input."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_visual_focus_classifications(results_jsonl)
    written = {}
    try:
        sp = out / SUMMARY_NAME
        sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        written["summary"] = sp
        cp = out / CARD_NAME
        cp.write_text(render_routing_card(summary), encoding="utf-8")
        written["card"] = cp
    except Exception:
        pass
    return written


__all__ = [
    "CLASSIFIER_VERSION", "PRIMARY_FOCUS_LABELS", "DETAIL_LEVEL_LABELS",
    "POSE_LABELS", "ROUTING_TARGETS", "RECIPE_TYPES", "ROUTING_TAGS",
    "TARGET_TO_RECIPE", "VisualFocusClassifierConfig", "FocusPrediction",
    "RoutingRecommendation", "VisualFocusClassificationResult",
    "classify_visual_focus", "classify_visual_focus_record",
    "classify_visual_focus_batch", "load_visual_focus_classifications",
    "summarize_visual_focus_classifications", "render_routing_card",
    "map_recipe_to_project", "result_to_dataset_explorer_fields",
    "result_to_caption_metadata", "results_by_filename",
    "write_visual_focus_summary", "RESULTS_NAME", "SUMMARY_NAME", "CARD_NAME"]


def get_nsfw_focus_route(tags: list) -> str:
    if nsfw_router:
        return nsfw_router.get_nsfw_category(tags)
    return "general_nsfw"
