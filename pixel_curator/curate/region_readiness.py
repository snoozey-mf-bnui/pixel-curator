"""Priority 6 - region-based quality scoring & region-aware readiness.

Aggregates the per-region signals other modules already compute (face-region
sharpness, body/torso quality, full-body visibility, texture preservation,
skin rendering, occlusion, motion blur, exposure) into three canonical region
scores - ``face`` / ``torso`` / ``full_body`` - plus a balance score and a
best-training-use hint.

Design contract:
  * additive & default-OFF - nothing here runs unless a caller enables it
  * non-destructive - it only reads signals, never images or files
  * honest - a region backed only by a global quality signal is scored with a
    ``region_score_partial`` warning and listed in ``missing_regions``; it is
    never presented as a true regional measurement
  * dependency-light & deterministic - pure-python aggregation, no cv2/torch,
    sorted/stable output, never raises for ordinary input
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Canonical region keys + accepted aliases that normalize onto them.
CANONICAL_REGIONS = ("face", "torso", "full_body")
_REGION_ALIASES = {
    "head": "face", "face": "face",
    "torso": "torso", "upper_body": "torso", "body": "torso",
    "full_body": "full_body", "person": "full_body", "figure": "full_body",
}

# best_training_use vocabulary (review-only hints, never auto-routing)
TRAINING_USES = ("identity", "outfit", "pose", "catalog", "review")


@dataclass
class RegionReadinessConfig:
    enabled: bool = False
    face_region: bool = True
    torso_region: bool = True
    full_body_region: bool = True
    require_bboxes: bool = False
    min_face_score: float = 0.60
    min_torso_score: float = 0.55
    min_full_body_score: float = 0.50

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled, "face_region": self.face_region,
            "torso_region": self.torso_region,
            "full_body_region": self.full_body_region,
            "require_bboxes": self.require_bboxes,
            "min_face_score": self.min_face_score,
            "min_torso_score": self.min_torso_score,
            "min_full_body_score": self.min_full_body_score,
        }


@dataclass
class RegionReadinessResult:
    face_region_score: float | None = None
    torso_region_score: float | None = None
    full_body_region_score: float | None = None
    region_balance_score: float | None = None
    best_training_use: str = "review"
    available_regions: list = field(default_factory=list)
    missing_regions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "face_region_score": _round(self.face_region_score),
            "torso_region_score": _round(self.torso_region_score),
            "full_body_region_score": _round(self.full_body_region_score),
            "region_balance_score": _round(self.region_balance_score),
            "best_training_use": self.best_training_use,
            "available_regions": list(self.available_regions),
            "missing_regions": list(self.missing_regions),
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def _round(v):
    return round(float(v), 4) if isinstance(v, (int, float)) else None


def normalize_region_key(key: str) -> str | None:
    return _REGION_ALIASES.get(str(key).strip().lower())


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------
def _flatten(*sources) -> dict:
    """Merge dicts (later sources do not override earlier present keys) plus a
    shallow lift of nested signal blocks so e.g. ``region_quality.face_quality``
    is reachable as ``face_quality``."""
    out: dict = {}
    for src in sources:
        if isinstance(src, dict):
            for k, v in src.items():
                out.setdefault(k, v)
    # lift selected nested blocks
    rq = out.get("region_quality")
    if isinstance(rq, dict):
        out.setdefault("face_quality", rq.get("face_quality"))
        out.setdefault("body_quality", rq.get("body_quality"))
        out.setdefault("overall_quality", rq.get("overall_quality"))
    return out


def _num(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _first(d: dict, *keys):
    for k in keys:
        val = _num(d.get(k))
        if val is not None:
            return val
    return None


def _inverse(v):
    """Convert a 0=good..1=bad signal (blur/occlusion) into 0..1 goodness."""
    return None if v is None else max(0.0, min(1.0, 1.0 - v))


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _exposure_goodness(f: dict):
    """Goodness from exposure signals: penalize over/under exposure."""
    exo = f.get("exposure_outlier")
    if isinstance(exo, dict):
        if exo.get("is_overexposed") or exo.get("is_underexposed"):
            return 0.3
        lvl = _num(exo.get("exposure_level"))
        if lvl is not None:
            # ideal ~0.5; linear falloff to edges
            return max(0.0, 1.0 - abs(lvl - 0.5) * 2.0)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_region_readiness(
    image: Any | None = None,
    metadata: dict | None = None,
    quality_signals: dict | None = None,
    bboxes: dict | None = None,
    config: RegionReadinessConfig | None = None,
) -> RegionReadinessResult:
    """Compute face/torso/full-body region readiness from already-computed
    signals. ``image`` is accepted for API symmetry but not required - scoring
    is signal-driven so it is cheap and deterministic. Never raises."""
    cfg = config or RegionReadinessConfig()
    res = RegionReadinessResult()
    f = _flatten(metadata or {}, quality_signals or {})
    bboxes = bboxes or {}

    have_face_bbox = any(bboxes.get(k) is not None
                         for k in ("face", "face_bbox", "head"))
    have_body_bbox = any(bboxes.get(k) is not None
                         for k in ("body", "body_bbox", "torso", "full_body"))
    if cfg.require_bboxes and not (have_face_bbox or have_body_bbox):
        res.warnings.append("region_readiness_partial: missing region bboxes")

    global_q = _first(f, "quality_score", "overall_quality",
                      "global_quality_avg")
    blur_good = _inverse(_first(f, "motion_blur_score"))
    texture = _first(f, "texture_preservation_score", "natural_detail_score")
    exposure_good = _exposure_goodness(f)

    # --- face region ---
    if cfg.face_region:
        face_specific = [
            _first(f, "face_quality"),
            _first(f, "skin_rendering_score"),
            _inverse(_first(f, "face_occlusion_score")),
            _first(f, "figure_integrity_score"),
        ]
        extras = [blur_good, texture, exposure_good]
        score, partial = _region_score(face_specific, extras, global_q)
        _assign(res, "face", score, partial)
        res.face_region_score = score

    # --- torso region ---
    if cfg.torso_region:
        torso_specific = [
            _first(f, "body_quality"),
            _first(f, "body_proportion_score"),
            _first(f, "clothing_type_score"),
        ]
        extras = [texture, blur_good, exposure_good]
        score, partial = _region_score(torso_specific, extras, global_q)
        _assign(res, "torso", score, partial)
        res.torso_region_score = score

    # --- full body region ---
    if cfg.full_body_region:
        fb_specific = [
            _first(f, "full_body_score", "full_body_visibility"),
            _first(f, "body_proportion_score"),
        ]
        cov = _coverage_completeness(f)
        extras = [cov, blur_good]
        score, partial = _region_score(fb_specific, extras, global_q)
        _assign(res, "full_body", score, partial)
        res.full_body_region_score = score

    # --- balance + training use ---
    present = {k: v for k, v in (
        ("face", res.face_region_score), ("torso", res.torso_region_score),
        ("full_body", res.full_body_region_score)) if v is not None}
    if len(present) >= 2:
        res.region_balance_score = round(1.0 - (max(present.values())
                                                - min(present.values())), 4)
    res.best_training_use = _best_use(res, present, cfg)
    if not present and not res.warnings:
        res.warnings.append("region_readiness_partial: no region signals "
                            "available")
    return res


def _region_score(specific, extras, global_q):
    """Blend region-specific signals (primary) with generic extras. Returns
    ``(score|None, partial_bool)``. ``partial`` is True when no region-specific
    signal was available and a global fallback was used."""
    spec = [s for s in specific if s is not None]
    ext = [e for e in extras if e is not None]
    if spec:
        base = _mean(spec)
        # extras nudge by at most +/- via a 70/30 blend, only when present
        if ext:
            return round(0.7 * base + 0.3 * _mean(ext), 4), False
        return round(base, 4), False
    # no region-specific signal
    if global_q is not None:
        return round(global_q, 4), True
    if ext:
        return round(_mean(ext), 4), True
    return None, False


def _assign(res: RegionReadinessResult, region: str, score, partial: bool):
    if score is None:
        res.missing_regions.append(region)
        return
    if partial:
        res.missing_regions.append(region)
        msg = "region_score_partial: using global signal fallback"
        if msg not in res.warnings:
            res.warnings.append(msg)
    else:
        res.available_regions.append(region)


def _coverage_completeness(f: dict):
    cov = f.get("coverage")
    label = f.get("coverage_label") or (cov.get("category")
                                        if isinstance(cov, dict) else None)
    if label is None:
        label = f.get("framing") or f.get("framing_label")
    if label is None:
        return None
    return {
        "full_body": 1.0, "full body": 1.0, "wide": 0.9,
        "three_quarter": 0.8, "half_body": 0.7, "upper_body": 0.6,
        "portrait": 0.5, "headshot": 0.35, "close_up": 0.3,
        "closeup": 0.3, "face": 0.3,
    }.get(str(label).strip().lower(), 0.5)


def _best_use(res: RegionReadinessResult, present: dict,
              cfg: RegionReadinessConfig) -> str:
    if not present:
        return "review"
    # balanced & strong across all three -> catalog reference
    if len(present) == 3 and min(present.values()) >= 0.6 \
            and (res.region_balance_score or 0.0) >= 0.6:
        return "catalog"
    # otherwise pick the strongest region that clears its threshold
    thresholds = {"face": cfg.min_face_score, "torso": cfg.min_torso_score,
                  "full_body": cfg.min_full_body_score}
    qualified = {k: v for k, v in present.items() if v >= thresholds[k]}
    pool = qualified or present
    # deterministic: highest score, ties break by canonical region order
    order = {"face": 0, "torso": 1, "full_body": 2}
    best = sorted(pool.items(), key=lambda kv: (-kv[1], order[kv[0]]))[0][0]
    return {"face": "identity", "torso": "outfit",
            "full_body": "pose"}.get(best, "review")


def summarize_region_readiness(samples: list) -> dict:
    """Dataset-level region-quality summary over manifest sample dicts. Reads
    each sample's stored ``region_readiness`` / flat region scores. Never
    raises."""
    faces, torsos, fulls, balances = [], [], [], []
    uses: dict = {}
    missing = low_face = low_torso = low_full = 0
    for s in samples or []:
        rr, scores = _sample_region_view(s)
        if not scores:
            missing += 1
            continue
        fv = scores.get("face_region_score")
        tv = scores.get("torso_region_score")
        bv = scores.get("full_body_region_score")
        ba = scores.get("region_balance_score")
        if fv is not None:
            faces.append(fv); low_face += 1 if fv < 0.60 else 0
        if tv is not None:
            torsos.append(tv); low_torso += 1 if tv < 0.55 else 0
        if bv is not None:
            fulls.append(bv); low_full += 1 if bv < 0.50 else 0
        if ba is not None:
            balances.append(ba)
        use = scores.get("best_training_use") or (rr or {}).get(
            "best_training_use")
        if use:
            uses[use] = uses.get(use, 0) + 1

    def _avg(vals):
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "avg_face_region_score": _avg(faces),
        "avg_torso_region_score": _avg(torsos),
        "avg_full_body_region_score": _avg(fulls),
        "avg_region_balance_score": _avg(balances),
        "best_training_use_distribution": dict(sorted(uses.items())),
        "missing_region_signal_count": missing,
        "low_face_region_count": low_face,
        "low_torso_region_count": low_torso,
        "low_full_body_region_count": low_full,
        "samples_with_region_scores": len(faces or torsos or fulls) and
        (len(samples or []) - missing),
    }


def _sample_region_view(sample: dict):
    """Return ``(region_readiness_dict_or_None, flat_scores_dict)`` for a
    manifest sample, looking in the entry, its ``signals`` block, and any
    stored ``region_readiness`` payload."""
    flat: dict = {}
    sources = [sample]
    for nest in ("signals", "metadata"):
        blk = sample.get(nest)
        if isinstance(blk, dict):
            sources.append(blk)
    rr = None
    for src in sources:
        if rr is None and isinstance(src.get("region_readiness"), dict):
            rr = src["region_readiness"]
        for key in ("face_region_score", "torso_region_score",
                    "full_body_region_score", "region_balance_score",
                    "best_training_use"):
            if key not in flat and src.get(key) is not None:
                flat[key] = src[key]
    if rr:
        for key in ("face_region_score", "torso_region_score",
                    "full_body_region_score", "region_balance_score",
                    "best_training_use"):
            flat.setdefault(key, rr.get(key))
    return rr, flat


def training_use_hint(result_or_scores) -> str:
    """Map a readiness result/dict to the splitter review hint vocabulary
    (identity/outfit/pose/review). Catalog maps to ``review`` for routing
    purposes (it is a balanced reference, not a single-target sample)."""
    if isinstance(result_or_scores, RegionReadinessResult):
        use = result_or_scores.best_training_use
    elif isinstance(result_or_scores, dict):
        use = result_or_scores.get("best_training_use", "review")
    else:
        use = "review"
    return {"identity": "identity", "outfit": "outfit", "pose": "pose",
            "catalog": "review", "review": "review"}.get(use, "review")


__all__ = [
    "CANONICAL_REGIONS", "TRAINING_USES", "RegionReadinessConfig",
    "RegionReadinessResult", "compute_region_readiness",
    "summarize_region_readiness", "normalize_region_key", "training_use_hint",
]
