"""Specialized Visual Focus Classifier (visual_focus_v1).

Organizes high-resolution figure / portrait / fashion / editorial / fitness
imagery into professional, training-oriented subsets by analyzing each image's
*visual composition* - primary focus, detail/texture tier, pose expressiveness,
technical quality, and training suitability. Output is structured and
explainable.

NSFW-aware version: Explicit/sexual content is no longer blocked by default.
Age safety flags still exist but only trigger from explicit human metadata
(never inferred from pixels).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CLASSIFIER_VERSION = "visual_focus_v1_nsfw"

# --- taxonomy (composition / detail / pose) -----------------
PRIMARY_FOCUS_LABELS = (
    "face_portrait", "head_and_shoulders", "upper_body_focus",
    "torso_composition", "full_body_pose", "lower_body_pose", "hands_detail",
    "feet_detail", "outfit_focus", "gesture_action", "environmental_portrait",
    "close_crop_study", "composition_reference", "lighting_reference",
    "unknown", "review_required")

DETAIL_TIER_LABELS = (
    "general_detail", "moderate_detail", "high_detail",
    "specialized_texture_study", "low_detail", "unknown", "review_required")

POSE_LABELS = (
    "neutral_standing", "neutral_seated", "relaxed_pose", "dynamic_pose",
    "gesture_forward", "arched_or_extended_pose", "walking_or_motion_pose",
    "close_crop_expressive", "portrait_expression", "reclined_pose",
    "symmetry_pose", "asymmetry_pose", "unknown", "review_required")

SUITABILITY_TIERS = ("excellent", "good", "review", "low", "reject")

TARGET_LABELS = (
    "identity_reference", "portrait_dataset", "full_body_pose_dataset",
    "outfit_dataset", "gesture_pose_dataset", "detail_texture_dataset",
    "lighting_composition_dataset", "captioning_reference", "review_required",
    "reject")

FLAG_VOCAB = (
    "no_clear_subject", "multiple_subjects", "low_light", "overexposed",
    "underexposed", "motion_blur", "low_region_clarity", "unsupported_format",
    "corrupt_image", "tiny_image", "huge_image", "classification_uncertain",
    "review_required_age_uncertain", "pixels_unavailable")

_FRAMING_TO_FOCUS = {
    "face_closeup": "face_portrait", "face": "face_portrait",
    "headshot": "head_and_shoulders", "portrait": "head_and_shoulders",
    "upper_body": "upper_body_focus", "half_body": "upper_body_focus",
    "torso_focus": "torso_composition", "full_body": "full_body_pose",
    "close_crop": "close_crop_study", "detail_crop": "close_crop_study",
}


@dataclass
class VisualFocusConfig:
    use_existing_quality_signals: bool = True
    use_pose_signals: bool = True
    use_vlm_refinement: bool = False
    min_confidence: float = 0.45
    detail_threshold_high: float = 0.75
    detail_threshold_moderate: float = 0.55
    suitability_reject_threshold: float = 35.0
    suitability_review_threshold: float = 55.0
    max_side_for_analysis: int = 1024
    classifier_version: str = CLASSIFIER_VERSION


@dataclass
class LabelScore:
    label: str
    confidence: float
    alternatives: list = field(default_factory=list)
    signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 4),
                "alternatives": [{"label": a, "confidence": round(c, 4)}
                                 for a, c in self.alternatives],
                "signals": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in self.signals.items()}}


@dataclass
class VisualFocusResult:
    image_path: str
    classifier_version: str
    primary_focus: LabelScore
    detail_tier: LabelScore
    pose_expression: LabelScore
    suitability_score: float
    suitability_tier: str
    recommended_targets: list = field(default_factory=list)
    technical_scores: dict = field(default_factory=dict)
    quality_signals: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    review_reasons: list = field(default_factory=list)
    caption_hint: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "classifier_version": self.classifier_version,
            "primary_focus": self.primary_focus.to_dict(),
            "detail_tier": self.detail_tier.to_dict(),
            "pose_expression": self.pose_expression.to_dict(),
            "training_suitability": {
                "score": round(self.suitability_score, 2),
                "tier": self.suitability_tier,
                "recommended_targets": list(self.recommended_targets)},
            "technical_scores": {k: round(v, 4) for k, v in
                                 self.technical_scores.items()},
            "quality_signals": self.quality_signals,
            "tags": list(self.tags), "flags": list(self.flags),
            "review_reasons": list(self.review_reasons),
            "caption_hint": self.caption_hint, "metadata": self.metadata}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _flatten_metadata(md: dict | None) -> dict:
    out: dict = {}
    if not md:
        return out
    out.update({k: v for k, v in md.items() if not isinstance(v, dict)})
    for nest in ("signals", "quality_signals", "advanced_signals"):
        if isinstance(md.get(nest), dict):
            out.update(md[nest])
    sj = md.get("signals_json")
    if isinstance(sj, str):
        try:
            out.update(json.loads(sj))
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass
    elif isinstance(sj, dict):
        out.update(sj)
    ci = md.get("coverage_intensity")
    if isinstance(ci, dict):
        out.setdefault("framing_label", ci.get("framing_label"))
        out.setdefault("training_use_hint", ci.get("training_use_hint"))
    return out


def _load_gray(path: Path, max_side: int):
    flags: list = []
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return None, None, None, ["pixels_unavailable"]
    try:
        from ..services import media_support as ms
        ms.register_heif_opener_if_available()
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    try:
        with Image.open(path) as im:
            im = im.convert("L")
            w, h = im.size
            if max(w, h) > max_side:
                scale = max_side / float(max(w, h))
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            arr = np.asarray(im, dtype="float32") / 255.0
        return arr, w, h, flags
    except FileNotFoundError:
        return None, None, None, ["corrupt_image"]
    except Exception as exc:
        msg = str(exc).lower()
        if "cannot identify" in msg or "unsupported" in msg or "decoder" in msg:
            return None, None, None, ["unsupported_format"]
        return None, None, None, ["corrupt_image"]


def _cv_signals(gray) -> dict:
    import numpy as np
    sig: dict = {}
    try:
        import cv2
        lap = cv2.Laplacian((gray * 255).astype("uint8"), cv2.CV_64F)
        lap_var = float(lap.var())
    except Exception:
        gy, gx = np.gradient(gray)
        lap_var = float((gx ** 2 + gy ** 2).mean()) * 5000.0
    sig["local_sharpness"] = _clamp01(lap_var / 600.0)
    gy, gx = np.gradient(gray)
    sig["texture_visibility"] = _clamp01(float(np.abs(gx).mean() +
                                               np.abs(gy).mean()) * 6.0)
    mean = float(gray.mean())
    clip_low = float((gray < 0.04).mean())
    clip_high = float((gray > 0.96).mean())
    sig["brightness_mean"] = mean
    sig["clip_low"] = clip_low
    sig["clip_high"] = clip_high
    sig["exposure_quality"] = _clamp01(1.0 - abs(mean - 0.5) * 1.6
                                       - clip_low - clip_high)
    sig["artifact_penalty"] = _clamp01((clip_low + clip_high) * 0.5)
    try:
        h, w = gray.shape
        cells = gray[:h // 3 * 3, :w // 3 * 3].reshape(3, h // 3, 3, w // 3)
        cell_means = cells.mean(axis=(1, 3))
        sig["lighting_quality"] = _clamp01(1.0 - float(cell_means.std()) * 2.0)
    except Exception:
        sig["lighting_quality"] = sig["exposure_quality"]
    return sig


def _decide_focus(md: dict, w, h) -> LabelScore:
    framing = (md.get("framing_label") or md.get("framing") or "").lower()
    hint = (md.get("training_use_hint") or "").lower()
    clothing = md.get("clothing_type_label")
    face_count = md.get("face_count")
    alts: list = []
    if framing in _FRAMING_TO_FOCUS:
        label = _FRAMING_TO_FOCUS[framing]
        conf = 0.82
        if clothing and clothing != "unknown" and label in (
                "upper_body_focus", "torso_composition", "full_body_pose"):
            alts.append(("outfit_focus", 0.45))
        if hint == "outfit":
            alts.append(("outfit_focus", 0.5))
        return LabelScore(label, conf, alts, {"framing": framing})
    if w and h:
        ar = h / float(w) if w else 1.0
        if face_count and int(face_count) > 1:
            return LabelScore("environmental_portrait", 0.5,
                              [("composition_reference", 0.4)],
                              {"aspect_ratio": round(ar, 2)})
        if ar >= 1.6:
            return LabelScore("full_body_pose", 0.55,
                              [("upper_body_focus", 0.4)],
                              {"aspect_ratio": round(ar, 2)})
        if ar <= 0.9:
            return LabelScore("composition_reference", 0.5,
                              [("environmental_portrait", 0.4)],
                              {"aspect_ratio": round(ar, 2)})
        return LabelScore("upper_body_focus", 0.5,
                          [("head_and_shoulders", 0.4)],
                          {"aspect_ratio": round(ar, 2)})
    return LabelScore("unknown", 0.3, [], {})


def _decide_detail(md: dict, cv: dict, cfg: VisualFocusConfig) -> LabelScore:
    sharp = _num(md.get("local_sharpness")) or cv.get("local_sharpness")
    if sharp is None:
        sharp = _num(md.get("detail_retention"))
    tex = (_num(md.get("texture_preservation_score"))
           or _num(md.get("texture_visibility")) or cv.get("texture_visibility"))
    artifact = (_num(md.get("motion_blur_score")) or cv.get("artifact_penalty")
                or 0.0)
    parts = [p for p in (sharp, tex) if p is not None]
    if not parts:
        return LabelScore("unknown", 0.3, [],
                          {"local_sharpness": sharp, "texture_visibility": tex})
    score = max(0.0, sum(parts) / len(parts) - 0.4 * (artifact or 0.0))
    sig = {"local_sharpness": sharp or 0.0, "texture_visibility": tex or 0.0,
           "artifact_penalty": artifact or 0.0}
    if score >= cfg.detail_threshold_high + 0.1:
        return LabelScore("specialized_texture_study", _clamp01(score), [], sig)
    if score >= cfg.detail_threshold_high:
        return LabelScore("high_detail", _clamp01(score),
                          [("specialized_texture_study", score - 0.1)], sig)
    if score >= cfg.detail_threshold_moderate:
        return LabelScore("moderate_detail", _clamp01(score),
                          [("general_detail", 0.5)], sig)
    if score >= 0.30:
        return LabelScore("general_detail", _clamp01(score), [], sig)
    return LabelScore("low_detail", _clamp01(1 - score), [], sig)


def _decide_pose(md: dict, focus: str, cfg: VisualFocusConfig) -> LabelScore:
    if not cfg.use_pose_signals:
        return LabelScore("unknown", 0.3, [], {})
    dyn = _num(md.get("dynamic_pose_score"))
    sym = _num(md.get("body_symmetry_score"))
    motion = _num(md.get("motion_blur_score")) or 0.0
    sig = {"pose_openness": dyn if dyn is not None else 0.0,
           "asymmetry": (1 - sym) if sym is not None else 0.0,
           "motion_likelihood": motion}
    if focus in ("face_portrait", "head_and_shoulders"):
        return LabelScore("portrait_expression", 0.6, [], sig)
    if focus in ("close_crop_study", "hands_detail", "feet_detail"):
        return LabelScore("close_crop_expressive", 0.55, [], sig)
    if dyn is not None:
        if dyn >= 0.65:
            return LabelScore("dynamic_pose", _clamp01(dyn),
                              [("gesture_forward", dyn - 0.1)], sig)
        if dyn <= 0.30:
            base = "neutral_standing" if focus == "full_body_pose" else "relaxed_pose"
            return LabelScore(base, 0.6, [("symmetry_pose", 0.4)], sig)
        return LabelScore("relaxed_pose", 0.55, [("dynamic_pose", dyn)], sig)
    if sym is not None:
        return LabelScore("symmetry_pose" if sym >= 0.6 else "asymmetry_pose",
                          0.5, [], sig)
    return LabelScore("unknown", 0.35, [], sig)


def _technical_scores(md: dict, cv: dict) -> dict:
    def pick(*keys, default=None):
        for k in keys:
            v = _num(md.get(k))
            if v is not None:
                return v
        return default
    sharp = pick("local_sharpness", "detail_retention", default=cv.get("local_sharpness"))
    exposure = pick("exposure_quality", default=cv.get("exposure_quality"))
    if exposure is None and _num(md.get("exposure_outlier_score")) is not None:
        exposure = 1 - _num(md["exposure_outlier_score"])
    region = pick("region_balance_score", "region_clarity", "integrity",
                  default=cv.get("local_sharpness"))
    lighting = pick("lighting_quality", "aesthetic_score",
                    default=cv.get("lighting_quality"))
    comp = pick("composition_quality", "centering", "aesthetic_score",
                default=cv.get("lighting_quality"))
    return {k: _clamp01(v) for k, v in {
        "sharpness": sharp or 0.0, "exposure_quality": exposure or 0.0,
        "composition_quality": comp or 0.0, "region_clarity": region or 0.0,
        "lighting_quality": lighting or 0.0}.items()}


def _suitability(tech: dict, focus: LabelScore, detail: LabelScore,
                 pose: LabelScore, flags: list, cfg: VisualFocusConfig):
    technical = sum(tech.values()) / max(1, len(tech))
    focus_use = focus.confidence if focus.label not in (
        "unknown", "review_required") else 0.2
    detail_use = {"specialized_texture_study": 1.0, "high_detail": 0.9,
                  "moderate_detail": 0.7, "general_detail": 0.55,
                  "low_detail": 0.25}.get(detail.label, 0.4)
    focus_detail = (focus_use + detail_use) / 2
    pose_use = pose.confidence if pose.label not in (
        "unknown", "review_required") else 0.3
    comp_light = (tech["composition_quality"] + tech["lighting_quality"]) / 2
    score = 100 * (0.40 * technical + 0.25 * focus_detail
                   + 0.20 * pose_use + 0.15 * comp_light)
    penalty = 0.0
    for f, pts in (("motion_blur", 18), ("low_light", 12), ("overexposed", 12),
                   ("underexposed", 12), ("low_region_clarity", 14),
                   ("classification_uncertain", 8), ("no_clear_subject", 25),
                   ("corrupt_image", 100), ("unsupported_format", 100),
                   ("tiny_image", 20)):
        if f in flags:
            penalty += pts
    score = max(0.0, min(100.0, score - penalty))
    if "corrupt_image" in flags or "unsupported_format" in flags:
        return 0.0, "reject"
    if score >= 90:
        tier = "excellent"
    elif score >= 75:
        tier = "good"
    elif score >= cfg.suitability_review_threshold:
        tier = "review"
    elif score >= cfg.suitability_reject_threshold:
        tier = "low"
    else:
        tier = "reject"
    return score, tier


def _targets(focus: str, detail: str, pose: str, tier: str) -> list:
    if tier == "reject":
        return ["reject"]
    t: list = []
    if focus in ("face_portrait", "head_and_shoulders"):
        t += ["identity_reference", "portrait_dataset"]
    if focus in ("full_body_pose", "lower_body_pose"):
        t.append("full_body_pose_dataset")
    if focus == "outfit_focus" or focus in ("upper_body_focus", "torso_composition"):
        t.append("outfit_dataset")
    if focus in ("close_crop_study", "hands_detail", "feet_detail") or \
            detail in ("high_detail", "specialized_texture_study"):
        t.append("detail_texture_dataset")
    if focus in ("lighting_reference", "composition_reference",
                 "environmental_portrait"):
        t.append("lighting_composition_dataset")
    if pose in ("dynamic_pose", "gesture_forward", "walking_or_motion_pose",
                "arched_or_extended_pose"):
        t.append("gesture_pose_dataset")
    if tier in ("excellent", "good"):
        t.append("captioning_reference")
    if not t:
        t.append("review_required")
    seen, out = set(), []
    for x in t:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _caption_hint(focus: str, pose: str, detail: str, tech: dict) -> str:
    f = focus.replace("_", " ")
    p = pose.replace("_", " ")
    d = detail.replace("_", " ")
    light = "clean lighting" if tech.get("lighting_quality", 0) >= 0.65 \
        else "soft lighting"
    parts = [f]
    if pose not in ("unknown", "review_required"):
        parts.append(p)
    if detail not in ("unknown", "review_required"):
        parts.append(f"{d} clarity")
    parts.append(light)
    return ", ".join(parts).capitalize() + "."


_FLORENCE_FOCUS = {"portrait": "face_portrait", "upper_body": "upper_body_focus",
                   "full_body": "full_body_pose", "outfit": "outfit_focus",
                   "detail_texture": "close_crop_study",
                   "composition": "composition_reference"}
_FLORENCE_POSE = {"seated": "neutral_seated", "reclined": "reclined_pose",
                  "dynamic": "dynamic_pose", "gesture": "gesture_forward",
                  "close_crop": "close_crop_expressive", "neutral": "neutral_standing"}
_FLORENCE_DETAIL = {"high": "high_detail", "moderate": "moderate_detail",
                    "general": "general_detail"}


def _apply_florence(focus, detail, pose, existing_metadata):
    md = existing_metadata or {}
    fl = md.get("florence")
    attrs = fl.get("visual_attributes") if isinstance(fl, dict) else None
    if attrs is None and isinstance(fl, dict):
        attrs = fl
    if attrs is None:
        attrs = {k.replace("florence_", ""): v for k, v in md.items()
                 if k.startswith("florence_")} or None
    if not isinstance(attrs, dict):
        return focus, detail, pose
    if focus.label == "unknown":
        mapped = _FLORENCE_FOCUS.get(attrs.get("primary_focus"))
        if mapped:
            focus = LabelScore(mapped, 0.6, focus.alternatives,
                               {"source": "florence"})
    if pose.label == "unknown":
        mapped = _FLORENCE_POSE.get(attrs.get("pose_type"))
        if mapped:
            pose = LabelScore(mapped, 0.55, [], {"source": "florence"})
    if detail.label == "unknown":
        mapped = _FLORENCE_DETAIL.get(attrs.get("detail_level"))
        if mapped:
            detail = LabelScore(mapped, 0.55, [], {"source": "florence"})
    return focus, detail, pose


def classify_visual_focus(image_path, existing_metadata=None,
                          config: VisualFocusConfig | None = None
                          ) -> VisualFocusResult:
    cfg = config or VisualFocusConfig()
    path = Path(image_path)
    md = _flatten_metadata(existing_metadata) if cfg.use_existing_quality_signals \
        else _flatten_metadata({k: v for k, v in (existing_metadata or {}).items()
                                if k in ("width", "height", "framing",
                                         "framing_label")})
    flags: list = []
    review: list = []

    gray, w, h, load_flags = _load_gray(path, cfg.max_side_for_analysis)
    flags += load_flags
    w = w or _num(md.get("width"))
    h = h or _num(md.get("height"))
    cv = _cv_signals(gray) if gray is not None else {}

    tech = _technical_scores(md, cv)

    if cv:
        if cv.get("brightness_mean", 0.5) < 0.22 or cv.get("clip_low", 0) > 0.5:
            flags.append("low_light")
        if cv.get("brightness_mean", 0.5) < 0.30:
            flags.append("underexposed")
        if cv.get("clip_high", 0) > 0.35:
            flags.append("overexposed")
    if (_num(md.get("motion_blur_score")) or 0) > 0.55:
        flags.append("motion_blur")
    if tech["region_clarity"] < 0.35:
        flags.append("low_region_clarity")
    fc = md.get("face_count")
    if fc is not None:
        if int(fc) == 0 and not md.get("framing_label"):
            flags.append("no_clear_subject")
        elif int(fc) > 1:
            flags.append("multiple_subjects")
    if w and h and max(int(w), int(h)) < 320:
        flags.append("tiny_image")
    if w and h and max(int(w), int(h)) > 12000:
        flags.append("huge_image")

    # AGE SAFETY: Only from explicit human metadata (never inferred)
    if md.get("age_uncertain") is True:
        flags.append("review_required_age_uncertain")
        review.append("age uncertain - flagged for human review")

    focus = _decide_focus(md, w, h)
    detail = _decide_detail(md, cv, cfg)
    pose = _decide_pose(md, focus.label, cfg)

    focus, detail, pose = _apply_florence(focus, detail, pose, existing_metadata)

    if max(focus.confidence, detail.confidence, pose.confidence) < cfg.min_confidence:
        flags.append("classification_uncertain")
        review.append("low classifier confidence")

    score, tier = _suitability(tech, focus, detail, pose, flags, cfg)

    if "review_required_age_uncertain" in flags:
        focus = LabelScore("review_required", 1.0, focus.alternatives, focus.signals)
        targets = ["review_required"]
    else:
        targets = _targets(focus.label, detail.label, pose.label, tier)

    tags = []
    if focus.label not in ("unknown", "review_required"):
        tags.append(focus.label)
    if pose.label not in ("unknown", "review_required"):
        tags.append(pose.label)
    if detail.label not in ("unknown", "review_required"):
        tags.append(detail.label)
    if tech["lighting_quality"] >= 0.65:
        tags.append("clean_lighting")
    if tier in ("excellent", "good"):
        tags.append("training_candidate")
    tags += [f for f in flags if f in ("motion_blur", "low_light", "overexposed")]

    return VisualFocusResult(
        image_path=str(path), classifier_version=cfg.classifier_version,
        primary_focus=focus, detail_tier=detail, pose_expression=pose,
        suitability_score=score, suitability_tier=tier,
        recommended_targets=targets, technical_scores=tech,
        quality_signals={**{k: v for k, v in cv.items()},
                         **{k: _num(md.get(k)) for k in (
                             "integrity", "aesthetic_score",
                             "region_balance_score", "texture_preservation_score")
                            if md.get(k) is not None}},
        tags=tags, flags=sorted(set(flags)), review_reasons=review,
        caption_hint=("" if "review_required_age_uncertain" in flags
                      else _caption_hint(focus.label, pose.label, detail.label, tech)),
        metadata={"width": int(w) if w else None, "height": int(h) if h else None,
                  "source": "manifest" if existing_metadata else "pixels",
                  "processed_at": _now()})


def classify_visual_focus_batch(image_paths, output_jsonl=None,
                                config: VisualFocusConfig | None = None,
                                resume: bool = True) -> list:
    cfg = config or VisualFocusConfig()
    done: set = set()
    out_path = Path(output_jsonl) if output_jsonl else None
    if out_path and resume and out_path.exists():
        try:
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line).get("image_path"))
        except Exception:
            done = set()
    results: list = []
    fh = None
    try:
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fh = out_path.open("a", encoding="utf-8")
        for p in image_paths:
            sp = str(p)
            if sp in done:
                continue
            res = classify_visual_focus(p, config=cfg)
            results.append(res)
            if fh:
                fh.write(json.dumps(res.to_dict()) + "\n")
                fh.flush()
    finally:
        if fh:
            fh.close()
    return results


def result_to_manifest_fields(result: VisualFocusResult) -> dict:
    return {
        "primary_focus": result.primary_focus.label,
        "detail_tier": result.detail_tier.label,
        "pose_expression": result.pose_expression.label,
        "visual_focus_score": round(result.suitability_score, 2),
        "suitability_tier": result.suitability_tier,
        "recommended_targets": list(result.recommended_targets),
        "visual_focus_flags": list(result.flags),
        "classifier_version": result.classifier_version,
    }


__all__ = ["CLASSIFIER_VERSION", "PRIMARY_FOCUS_LABELS", "DETAIL_TIER_LABELS",
           "POSE_LABELS", "SUITABILITY_TIERS", "TARGET_LABELS", "FLAG_VOCAB",
           "VisualFocusConfig", "LabelScore", "VisualFocusResult",
           "classify_visual_focus", "classify_visual_focus_batch",
           "result_to_manifest_fields"]
