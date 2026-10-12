"""Priority 7 - coverage & framing intensity classification.

Fuses the framing/coverage signals already produced elsewhere (coverage
classifier label, region readiness face/torso/full-body scores, full-body
detector visibility, face/body bounding boxes) into one stable, manifest-safe
verdict: a coverage label, a framing label, supporting scores, and a review
hint.

Contract: additive, default-OFF, non-destructive (reads signals only, never
images/files), deterministic, dependency-light (pure python), and honest -
ambiguous or conflicting inputs resolve to ``unknown`` / ``needs_review`` with
reason codes rather than an over-confident guess. Never raises for ordinary
input.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Canonical, professional vocabularies.
COVERAGE_LABELS = ("full_coverage", "partial_coverage", "minimal_coverage",
                   "unknown", "needs_review")
FRAMING_LABELS = ("full_body", "torso_focus", "upper_body", "close_crop",
                  "face_closeup", "detail_crop", "unknown", "needs_review")
TRAINING_USES = ("identity", "outfit", "pose", "catalog_reference", "detail",
                 "review")

_FRAMING_ALIASES = {
    "full body": "full_body", "full_body": "full_body", "body": "full_body",
    "torso": "torso_focus", "torso_focus": "torso_focus",
    "upper body": "upper_body", "upper_body": "upper_body",
    "half_body": "upper_body", "half body": "upper_body",
    "closeup": "close_crop", "close_up": "close_crop", "close_crop": "close_crop",
    "face": "face_closeup", "face_closeup": "face_closeup",
    "headshot": "face_closeup", "portrait": "upper_body",
    "detail": "detail_crop", "detail_crop": "detail_crop",
    "garment_closeup": "detail_crop", "review": "needs_review",
    "needs_review": "needs_review", "unknown": "unknown",
}
_COVERAGE_ALIASES = {
    "full": "full_coverage", "full_coverage": "full_coverage",
    "high": "full_coverage", "partial": "partial_coverage",
    "partial_coverage": "partial_coverage", "medium": "partial_coverage",
    "minimal": "minimal_coverage", "minimal_coverage": "minimal_coverage",
    "low": "minimal_coverage", "none": "minimal_coverage",
    "review": "needs_review", "needs_review": "needs_review",
    "unknown": "unknown",
}
# framing label -> (coverage label, training-use hint)
_FRAMING_TO_COVERAGE = {
    "full_body": ("full_coverage", "pose"),
    "torso_focus": ("partial_coverage", "outfit"),
    "upper_body": ("partial_coverage", "outfit"),
    "close_crop": ("minimal_coverage", "identity"),
    "face_closeup": ("minimal_coverage", "identity"),
    "detail_crop": ("minimal_coverage", "detail"),
}


@dataclass
class CoverageIntensityConfig:
    enabled: bool = False
    use_region_readiness: bool = True
    use_full_body_signal: bool = True
    use_face_bbox: bool = True
    use_body_bbox: bool = True
    require_subject_bbox: bool = False

    def to_dict(self) -> dict:
        return {"enabled": self.enabled,
                "use_region_readiness": self.use_region_readiness,
                "use_full_body_signal": self.use_full_body_signal,
                "use_face_bbox": self.use_face_bbox,
                "use_body_bbox": self.use_body_bbox,
                "require_subject_bbox": self.require_subject_bbox}


@dataclass
class CoverageIntensityResult:
    coverage_label: str = "unknown"
    framing_label: str = "unknown"
    coverage_score: float | None = None
    full_body_score: float | None = None
    torso_focus_score: float | None = None
    close_crop_score: float | None = None
    review_recommended: bool = False
    training_use_hint: str = "review"
    confidence: float = 0.0
    reason_codes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "coverage_label": self.coverage_label,
            "framing_label": self.framing_label,
            "coverage_score": _round(self.coverage_score),
            "full_body_score": _round(self.full_body_score),
            "torso_focus_score": _round(self.torso_focus_score),
            "close_crop_score": _round(self.close_crop_score),
            "review_recommended": self.review_recommended,
            "training_use_hint": self.training_use_hint,
            "confidence": _round(self.confidence),
            "reason_codes": list(self.reason_codes),
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def _round(v):
    return round(float(v), 4) if isinstance(v, (int, float)) else None


def normalize_framing_label(value) -> str:
    if value is None:
        return "unknown"
    return _FRAMING_ALIASES.get(str(value).strip().lower(), "unknown")


def normalize_coverage_label(value) -> str:
    if value is None:
        return "unknown"
    return _COVERAGE_ALIASES.get(str(value).strip().lower(), "unknown")


def _flatten(*sources) -> dict:
    out: dict = {}
    for src in sources:
        if isinstance(src, dict):
            for k, v in src.items():
                out.setdefault(k, v)
    for nest in ("region_readiness", "coverage", "full_body"):
        blk = out.get(nest)
        if isinstance(blk, dict):
            for k, v in blk.items():
                out.setdefault(f"{nest}.{k}", v)
    return out


def _num(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _first(d: dict, *keys):
    for k in keys:
        v = _num(d.get(k))
        if v is not None:
            return v
    return None


def _bbox_area_ratio(bbox, default=None):
    """Normalized bbox area if bbox looks normalized (0..1) coords."""
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return default
    w, h = abs(x1 - x0), abs(y1 - y0)
    if 0 <= w <= 1.0 and 0 <= h <= 1.0:
        return max(0.0, min(1.0, w * h))
    return default


def classify_coverage_intensity(
    image: Any | None = None,
    metadata: dict | None = None,
    quality_signals: dict | None = None,
    advanced_quality: dict | None = None,
    bboxes: dict | None = None,
    config: CoverageIntensityConfig | None = None,
) -> CoverageIntensityResult:
    """Classify coverage/framing intensity from already-computed signals.

    ``image`` is accepted for API symmetry but unused (classification is
    signal-driven). Never raises."""
    cfg = config or CoverageIntensityConfig()
    res = CoverageIntensityResult()
    f = _flatten(metadata or {}, quality_signals or {}, advanced_quality or {})
    bboxes = bboxes or {}

    face_bbox = bboxes.get("face") or bboxes.get("face_bbox")
    body_bbox = (bboxes.get("body") or bboxes.get("body_bbox")
                 or bboxes.get("full_body"))
    if cfg.require_subject_bbox and not (face_bbox or body_bbox):
        res.warnings.append("coverage_intensity_partial: missing subject geometry")

    full_body = _first(f, "full_body_region_score", "full_body_score",
                       "full_body.full_body_visibility",
                       "region_readiness.full_body_region_score") \
        if cfg.use_full_body_signal or cfg.use_region_readiness else None
    torso = _first(f, "torso_region_score",
                   "region_readiness.torso_region_score") \
        if cfg.use_region_readiness else None
    face = _first(f, "face_region_score",
                  "region_readiness.face_region_score") \
        if cfg.use_region_readiness else None
    res.full_body_score = full_body
    res.torso_focus_score = torso

    # 1) an explicit, already-computed framing/coverage label wins (normalized)
    explicit_framing = normalize_framing_label(
        _str(f, "framing_label", "framing_type", "full_body.framing_type",
             "framing"))
    explicit_coverage = normalize_coverage_label(
        _str(f, "coverage_label", "coverage.category", "coverage_level"))

    if explicit_framing not in ("unknown",):
        res.framing_label = explicit_framing
        cov, hint = _FRAMING_TO_COVERAGE.get(
            explicit_framing, ("unknown", "review"))
        res.coverage_label = (explicit_coverage if explicit_coverage
                              not in ("unknown",) else cov)
        res.training_use_hint = hint
        res.reason_codes.append("explicit_framing_label")
        res.confidence = 0.8
    else:
        # 2..5) derive from region readiness / full-body strength
        _derive_from_regions(res, full_body, torso, face)

    # face-bbox close-crop signal (small body, large face)
    if cfg.use_face_bbox and face_bbox is not None and res.framing_label \
            in ("unknown", "face_closeup", "close_crop"):
        fa = _bbox_area_ratio(face_bbox)
        if fa is not None:
            res.close_crop_score = round(min(1.0, fa * 4.0), 4)

    # coverage_score: prefer full-body strength, else map label
    res.coverage_score = _coverage_score(res, full_body, torso, face)

    # 6/7) ambiguity & conflict -> review
    _resolve_review(res, f)

    if res.framing_label == "unknown" and res.coverage_label == "unknown" \
            and not res.warnings:
        res.warnings.append("coverage_intensity_partial: no framing signals")
    return res


def _str(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if v is not None and str(v) != "":
            return v
    return None


def _derive_from_regions(res, full_body, torso, face):
    strengths = {k: v for k, v in (("full_body", full_body),
                 ("torso", torso), ("face", face)) if v is not None}
    if not strengths:
        res.framing_label = "unknown"
        res.coverage_label = "unknown"
        res.review_recommended = True
        res.reason_codes.append("no_region_signals")
        return
    # rule 2: strong full body
    if full_body is not None and full_body >= 0.60 and \
            full_body >= max(strengths.values()) - 1e-9:
        res.framing_label = "full_body"
        res.coverage_label = "full_coverage"
        res.training_use_hint = "pose"
        res.confidence = round(0.5 + 0.4 * full_body, 4)
        res.reason_codes.append("full_body_visibility_strong")
        return
    # rule 3: torso/upper-body strongest
    if torso is not None and torso >= max(strengths.values()) - 1e-9 \
            and torso >= 0.50:
        res.framing_label = "torso_focus"
        res.coverage_label = "partial_coverage"
        res.training_use_hint = "outfit"
        res.confidence = round(0.5 + 0.3 * torso, 4)
        res.reason_codes.append("torso_region_strongest")
        return
    # rule 4: face dominant, body weak
    if face is not None and face >= max(strengths.values()) - 1e-9 \
            and (full_body is None or full_body < 0.45):
        res.framing_label = "face_closeup"
        res.coverage_label = "minimal_coverage"
        res.training_use_hint = "identity"
        res.confidence = round(0.5 + 0.3 * face, 4)
        res.reason_codes.append("face_region_dominant")
        return
    # ambiguous
    res.framing_label = "unknown"
    res.coverage_label = "unknown"
    res.review_recommended = True
    res.reason_codes.append("ambiguous_region_strengths")


def _coverage_score(res, full_body, torso, face):
    if full_body is not None:
        return round(full_body, 4)
    label_map = {"full_coverage": 0.9, "partial_coverage": 0.6,
                 "minimal_coverage": 0.3}
    if res.coverage_label in label_map:
        return label_map[res.coverage_label]
    return None


def _resolve_review(res, f):
    # conflicting explicit signals: full-body label but a face-closeup hint
    conflict = False
    fb = _num(f.get("full_body_region_score")) or _num(f.get("full_body_score"))
    if res.framing_label == "full_body" and fb is not None and fb < 0.35:
        conflict = True
    if conflict:
        res.review_recommended = True
        res.framing_label = "needs_review"
        res.coverage_label = "needs_review"
        res.training_use_hint = "review"
        res.confidence = min(res.confidence, 0.3)
        res.reason_codes.append("conflicting_coverage_signals")
    if res.framing_label in ("unknown", "needs_review"):
        res.review_recommended = True


# ---------------------------------------------------------------------------
# Dataset-level summary
# ---------------------------------------------------------------------------
def _sample_view(sample: dict) -> dict:
    out: dict = {}
    sources = [sample]
    for nest in ("signals", "metadata"):
        blk = sample.get(nest)
        if isinstance(blk, dict):
            sources.append(blk)
    ci = None
    for src in sources:
        if ci is None and isinstance(src.get("coverage_intensity"), dict):
            ci = src["coverage_intensity"]
        for k in ("coverage_label", "framing_label", "training_use_hint",
                  "coverage_review_recommended"):
            if k not in out and src.get(k) is not None:
                out[k] = src[k]
    if ci:
        for k in ("coverage_label", "framing_label", "training_use_hint"):
            out.setdefault(k, ci.get(k))
        out.setdefault("coverage_review_recommended",
                       ci.get("review_recommended"))
    return out


def summarize_coverage_intensity(samples: list) -> dict:
    """Dataset-level coverage/framing distribution + dominance. Never raises."""
    from collections import Counter
    cov, frm, use = Counter(), Counter(), Counter()
    unknown_cov = needs_review = with_signal = 0
    n = len(samples or [])
    for s in samples or []:
        v = _sample_view(s)
        cl = v.get("coverage_label")
        fl = v.get("framing_label")
        if cl is None and fl is None:
            continue
        with_signal += 1
        if cl:
            cov[str(cl)] += 1
            if cl in ("unknown",):
                unknown_cov += 1
            if cl == "needs_review":
                needs_review += 1
        if fl:
            frm[str(fl)] += 1
            if fl == "needs_review":
                needs_review += 1
        if v.get("coverage_review_recommended") is True:
            needs_review += 1
        if v.get("training_use_hint"):
            use[str(v["training_use_hint"])] += 1
    framed = sum(frm.values())
    dom_label, dom_ratio = (None, 0.0)
    if frm:
        dom_label, c = frm.most_common(1)[0]
        dom_ratio = round(c / framed, 4) if framed else 0.0
    return {
        "coverage_distribution": dict(cov.most_common()),
        "framing_distribution": dict(frm.most_common()),
        "training_use_hint_distribution": dict(use.most_common()),
        "unknown_coverage_count": unknown_cov,
        "needs_review_count": needs_review,
        "dominant_framing_label": dom_label,
        "dominant_framing_ratio": dom_ratio,
        "samples_with_coverage_signal": with_signal,
        "sample_count": n,
    }


__all__ = [
    "COVERAGE_LABELS", "FRAMING_LABELS", "TRAINING_USES",
    "CoverageIntensityConfig", "CoverageIntensityResult",
    "classify_coverage_intensity", "summarize_coverage_intensity",
    "normalize_framing_label", "normalize_coverage_label",
]
