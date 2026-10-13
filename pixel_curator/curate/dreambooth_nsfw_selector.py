#!/usr/bin/env python3
"""
dreambooth_nsfw_selector.py
===========================
v5 — Configurable NSFW DreamBooth Dataset Selector

Selects the best frames for DreamBooth training using:
- Advanced quality signals (tunable per-signal weights)
- Strong NSFW scoring (explicitness + arousal + skin exposure)
- Quota system + diversity controls
- Selection profiles, custom rules, and per-frame explanations

Configuration (Phase 6) — three layers, later wins, provenance logged:

  1. built-in profile defaults   (--profile / profile= in TOML;
                                  'balanced' == legacy behavior)
  2. [dreambooth_selector] section of the project TOML
  3. explicit command-line flags

Example project.toml section::

    [dreambooth_selector]
    profile = "high_diversity"      # balanced | maximum_quality |
                                    # high_diversity | identity_focused |
                                    # nsfw_forward
    target = 6000
    nsfw_mode = "explicit_heavy"
    nsfw_weight = 0.5
    min_quality = 0.55
    max_source_frac = 0.10
    explain = "rejected"            # off | accepted | rejected | all
    explain_top = 10
    exclude_tags = ["watermark", "text overlay"]
    quotas = ["explicit:0.6", "semi_explicit:0.25:0.1", "covered:0.15"]

    [dreambooth_selector.quality_weights]
    identity = 0.20
    sharpness = 0.15
    pose_diversity = 0.10

The hard minor-safety gate (MINOR_TERMS) is NOT configurable and always
runs first; no profile, weight table, or rule can relax it.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pixel_curator import manifest as _manifest_mod
from pixel_curator.config import load_project as _load_project

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dreambooth_nsfw_selector")


# ============================================================
# CONFIG
# ============================================================

NSFW_MODE_WEIGHTS = {
    "balanced":       {"explicitness": 0.45, "arousal": 0.30, "skin": 0.25},
    "strong":         {"explicitness": 0.55, "arousal": 0.30, "skin": 0.25},
    "explicit_heavy": {"explicitness": 0.70, "arousal": 0.22, "skin": 0.18},
}

# Legacy quality-base weights, verbatim (sum 0.97). Used RAW - no
# renormalization - so the default score scale is bit-identical to v4 and
# the min_quality gate keeps its calibration.
QUALITY_WEIGHT_DEFAULTS = {
    "identity": 0.20,
    "sharpness": 0.15,
    "figure_integrity": 0.12,
    "anatomy_detail": 0.10,
    "natural_texture": 0.08,
    "hand_quality": 0.08,
    "body_proportion": 0.07,
    "skin_rendering": 0.06,
    "expression_quality": 0.06,
    "pose_diversity": 0.05,
}

EXPLAIN_MODES = ("off", "accepted", "rejected", "all")

# Selection profiles: documented override dicts on top of the base
# defaults. 'balanced' overrides nothing - it IS the legacy behavior.
# Every profile value remains individually overridable by TOML / CLI.
SELECTION_PROFILES: dict[str, dict] = {
    "balanced": {},
    "maximum_quality": {
        # quality first: raise the floors, halve the NSFW pull
        "min_quality": 0.70, "min_sharpness": 50.0, "nsfw_weight": 0.30,
        "quality_weights": {**QUALITY_WEIGHT_DEFAULTS,
                            "natural_texture": 0.11, "anatomy_detail": 0.12,
                            "pose_diversity": 0.02},
    },
    "high_diversity": {
        # widen the pool, tighten the caps, reward pose spread
        "min_quality": 0.50, "max_source_frac": 0.10,
        "max_cluster_frac": 0.12, "temporal_spacing": 30,
        "quality_weights": {**QUALITY_WEIGHT_DEFAULTS,
                            "pose_diversity": 0.12, "identity": 0.15,
                            "sharpness": 0.13},
    },
    "identity_focused": {
        # protect the likeness: hard identity floor, identity-heavy base,
        # stronger occlusion penalty
        "min_identity": 0.45, "occlusion_penalty": 0.70,
        "quality_weights": {**QUALITY_WEIGHT_DEFAULTS,
                            "identity": 0.32, "expression_quality": 0.08,
                            "pose_diversity": 0.03, "anatomy_detail": 0.06},
    },
    "nsfw_forward": {
        # the v4 'maximum filth' use case as a named, tunable profile
        "nsfw_mode": "explicit_heavy", "nsfw_weight": 0.65,
        "arousal_boost": 1.25, "min_quality": 0.55,
    },
}


@dataclass
class ConceptQuota:
    label: str
    fraction: float = 0.0
    min_score: float = 0.0


def parse_quota_spec(spec: str) -> ConceptQuota:
    """'label:fraction[:min_score]' -> ConceptQuota. Raises ValueError on a
    malformed spec so bad config fails loudly, not silently."""
    parts = [p.strip() for p in str(spec).split(":")]
    if len(parts) < 2 or not parts[0]:
        raise ValueError(f"Bad quota spec {spec!r} - expected "
                         "'label:fraction[:min_score]'")
    label = parts[0]
    fraction = float(parts[1])
    min_score = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Quota fraction for {label!r} must be in (0, 1], "
                         f"got {fraction}")
    return ConceptQuota(label=label, fraction=fraction, min_score=min_score)


@dataclass
class SelectionConfig:
    target_count: int = 7500
    nsfw_mode: str = "explicit_heavy"
    nsfw_weight: float = 0.50
    arousal_boost: float = 1.0
    min_quality: float = 0.60
    min_identity: float = 0.30
    min_sharpness: float = 35.0
    max_source_frac: float = 0.18
    max_cluster_frac: float = 0.22
    temporal_spacing: int = 15
    pose_dup_penalty: float = 0.08
    link_mode: str = "symlink"
    quotas: list[ConceptQuota] = field(default_factory=list)
    # Phase 6 - configurability + transparency
    profile: str = "balanced"
    quality_weights: dict = field(
        default_factory=lambda: dict(QUALITY_WEIGHT_DEFAULTS))
    nsfw_mode_weights: dict = field(default_factory=dict)  # override table
    occlusion_penalty: float = 0.85     # multiplier when face_occlusion > 0.5
    exclude_tags: list[str] = field(default_factory=list)
    require_any_tags: list[str] = field(default_factory=list)
    explain: str = "accepted"           # off | accepted | rejected | all
    explain_top: int = 10               # score breakdowns logged per side
    register_version: bool = True       # register --apply exports (Phase 2)
    training_config: bool = True        # write training_config.* (Phase 10)
    # Consolidated export extras (the selector is the defined DreamBooth
    # export path; these came from the retired standalone exporter):
    repeats: int = 1                    # kohya num_repeats in dataset_config
    reg: bool = False                   # scaffold a regularization (class) set
    reg_count: int = 300                # suggested class-image count
    token: str = "spikbnu"              # trigger token for dataset_config
    class_word: str = "person"

    def resolve_quotas(self):
        if not self.quotas:
            return [ConceptQuota(label="any", fraction=1.0, min_score=self.min_quality)]
        return list(self.quotas)        # listed order == priority order


# keys resolve_selection_config accepts from TOML/CLI (all SelectionConfig
# scalars plus the rule/quota lists). Unknown TOML keys warn, not crash.
_CONFIG_SCALARS = {
    "target": ("target_count", int),
    "target_count": ("target_count", int),
    "nsfw_mode": ("nsfw_mode", str),
    "nsfw_weight": ("nsfw_weight", float),
    "arousal_boost": ("arousal_boost", float),
    "min_quality": ("min_quality", float),
    "min_identity": ("min_identity", float),
    "min_sharpness": ("min_sharpness", float),
    "max_source_frac": ("max_source_frac", float),
    "max_cluster_frac": ("max_cluster_frac", float),
    "temporal_spacing": ("temporal_spacing", int),
    "pose_dup_penalty": ("pose_dup_penalty", float),
    "occlusion_penalty": ("occlusion_penalty", float),
    "link_mode": ("link_mode", str),
    "explain": ("explain", str),
    "explain_top": ("explain_top", int),
    "register_version": ("register_version", bool),
    "training_config": ("training_config", bool),
    "repeats": ("repeats", int),
    "reg": ("reg", bool),
    "reg_count": ("reg_count", int),
    "token": ("token", str),
    "class_word": ("class_word", str),
}


def resolve_selection_config(toml_section: dict | None = None,
                             cli: dict | None = None
                             ) -> tuple[SelectionConfig, dict, list[str]]:
    """Layered config resolution: profile defaults -> [dreambooth_selector]
    TOML section -> explicit CLI values (``cli`` holds only keys the user
    actually passed). Returns ``(config, provenance, warnings)`` where
    provenance maps each field to the layer that set it.
    """
    toml_section = dict(toml_section or {})
    cli = dict(cli or {})
    warnings: list[str] = []
    provenance: dict[str, str] = {}

    profile = str(cli.get("profile")
                  or toml_section.get("profile")
                  or "balanced")
    if profile not in SELECTION_PROFILES:
        warnings.append(f"Unknown profile {profile!r} - using 'balanced'. "
                        f"Available: {', '.join(sorted(SELECTION_PROFILES))}")
        profile = "balanced"

    cfg = SelectionConfig(profile=profile)
    for f in cfg.__dataclass_fields__:
        provenance[f] = "default"

    def apply_layer(values: dict, source: str):
        for key, val in values.items():
            if key == "profile" or val is None:
                continue
            if key in _CONFIG_SCALARS:
                attr, cast = _CONFIG_SCALARS[key]
                try:
                    setattr(cfg, attr, cast(val))
                    provenance[attr] = source
                except (TypeError, ValueError):
                    warnings.append(f"{source}: bad value for {key}: {val!r}")
            elif key == "quality_weights" and isinstance(val, dict):
                merged = dict(QUALITY_WEIGHT_DEFAULTS)
                unknown = set(val) - set(QUALITY_WEIGHT_DEFAULTS)
                if unknown:
                    warnings.append(f"{source}: unknown quality_weights "
                                    f"keys ignored: {sorted(unknown)}")
                merged.update({k: float(v) for k, v in val.items()
                               if k in QUALITY_WEIGHT_DEFAULTS})
                cfg.quality_weights = merged
                provenance["quality_weights"] = source
            elif key == "nsfw_mode_weights" and isinstance(val, dict):
                cfg.nsfw_mode_weights = {k: float(v) for k, v in val.items()}
                provenance["nsfw_mode_weights"] = source
            elif key in ("exclude_tags", "require_any_tags"):
                setattr(cfg, key, [str(t) for t in (val or [])])
                provenance[key] = source
            elif key == "quotas":
                try:
                    cfg.quotas = [q if isinstance(q, ConceptQuota)
                                  else parse_quota_spec(q) for q in val]
                    provenance["quotas"] = source
                except ValueError as exc:
                    warnings.append(f"{source}: {exc}")
            else:
                warnings.append(f"{source}: unknown option {key!r} ignored")

    apply_layer(SELECTION_PROFILES[profile], f"profile:{profile}")
    apply_layer(toml_section, "toml")
    apply_layer(cli, "cli")

    # sanity checks (transparent, non-fatal)
    if cfg.explain not in EXPLAIN_MODES:
        warnings.append(f"explain={cfg.explain!r} invalid - using "
                        "'accepted'")
        cfg.explain = "accepted"
    wsum = sum(cfg.quality_weights.values())
    if not 0.8 <= wsum <= 1.2:
        warnings.append(
            f"quality_weights sum to {wsum:.2f} (default 0.97): the quality "
            f"base tops out near {wsum:.2f} and min_quality="
            f"{cfg.min_quality} sits on that same scale - adjust one or "
            "the other deliberately.")
    if cfg.quotas:
        fsum = sum(q.fraction for q in cfg.quotas)
        if fsum > 1.001:
            warnings.append(f"quota fractions sum to {fsum:.2f} (> 1.0) - "
                            "later quotas will be starved")
    return cfg, provenance, warnings


def render_resolved_config(cfg: SelectionConfig, provenance: dict) -> str:
    """One line per non-default setting, annotated with its source."""
    lines = [f"Resolved selection config (profile '{cfg.profile}'):"]
    for f in cfg.__dataclass_fields__:
        src = provenance.get(f, "default")
        if src == "default":
            continue
        val = getattr(cfg, f)
        if f == "quality_weights":
            diff = {k: v for k, v in val.items()
                    if v != QUALITY_WEIGHT_DEFAULTS[k]}
            val = f"defaults + {diff}" if diff else "defaults"
        elif f == "quotas":
            val = ", ".join(f"{q.label}:{q.fraction}" for q in val)
        lines.append(f"  {f:<18} = {val!s:<24} ({src})")
    if len(lines) == 1:
        lines.append("  all defaults (legacy v4 behavior)")
    return "\n".join(lines)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class QualitySignals:
    frame_id: str
    path: str
    source: str
    identity_score: float = 0.0
    sharpness: float = 0.0
    figure_integrity: float = 0.0
    anatomy_detail: float = 0.0
    natural_texture: float = 0.0
    hand_quality: float = 0.0
    body_proportion: float = 0.0
    pose_diversity: float = 0.0
    skin_rendering: float = 0.0
    expression_quality: float = 0.0
    face_occlusion: float = 0.0
    pose_near_duplicate: bool = False
    cluster_id: int | None = None
    caption_tags: list[str] = field(default_factory=list)
    caption_text: str = ""
    sequence_id: str | None = None
    frame_index: int = 0


@dataclass
class NSFWSignals:
    explicitness: float = 0.0
    arousal: float = 0.0
    skin_exposure: float = 0.0
    intensity_label: str = "covered"


@dataclass
class SelectionExplanation:
    frame_id: str
    path: str
    accepted: bool
    quota_label: str | None = None
    reject_reason: str | None = None
    score: float = 0.0


# ============================================================
# NSFW + QUALITY SCORING
# ============================================================

_AROUSAL_TERMS = {"wet", "dripping", "squirting", "pussy juice", "ahegao", "orgasm", "climax"}

EXPLICIT_ROOTS = {
    "nude", "naked", "pussy", "vagina", "vulva", "labia", "clit", "cock",
    "dick", "penis", "cum", "creampie", "sex", "fuck", "anal", "blowjob",
    "titfuck", "ahegao", "orgasm", "masturbate", "nipple", "fellatio",
    "cunnilingus", "rimming", "fisting", "loli",
}

# Hard safety exclusion: any frame whose caption implies a minor is dropped
# outright — never scored, never selected. These terms are NOT explicitness
# signals; a frame matching any of them is removed from the candidate pool.
MINOR_TERMS = {
    "knife",
}


def _norm(t: str) -> str:
    return " ".join(str(t or "").strip().lower().replace("_", " ").split())


def _is_explicit(t: str) -> bool:
    t = _norm(t)
    return any(root in t for root in EXPLICIT_ROOTS)


def implies_minor(tags: list[str]) -> bool:
    """True if any tag implies a minor. Whole-tag + substring, to be strict."""
    for t in tags:
        n = _norm(t)
        if not n:
            continue
        if n in MINOR_TERMS:
            return True
        if any(term in n for term in MINOR_TERMS):
            return True
    return False


def classify_nsfw(tags: list[str]) -> NSFWSignals:
    nsfw = NSFWSignals()
    norm_tags = [_norm(t) for t in tags]

    explicit_hits = sum(1 for t in norm_tags if _is_explicit(t))
    nsfw.explicitness = min(explicit_hits / 3.5, 1.0)

    arousal_hits = sum(1 for t in norm_tags if any(term in t for term in _AROUSAL_TERMS))
    nsfw.arousal = min(arousal_hits / 3.0, 1.0)

    # Simple skin exposure proxy (can be improved with real coverage signals)
    nsfw.skin_exposure = min(explicit_hits / 2.0, 1.0)

    if explicit_hits >= 2:
        nsfw.intensity_label = "explicit"
    elif explicit_hits >= 1:
        nsfw.intensity_label = "semi_explicit"
    else:
        nsfw.intensity_label = "covered"

    return nsfw


def _quality_factors(s: QualitySignals) -> dict[str, float]:
    """Per-signal normalized values (0..1), keyed like the weight table."""
    return {
        "identity": max(0.0, min(1.0, s.identity_score)),
        "sharpness": min(s.sharpness / 100.0, 1.0),
        "figure_integrity": max(0.0, min(1.0, s.figure_integrity)),
        "anatomy_detail": max(0.0, min(1.0, s.anatomy_detail)),
        "natural_texture": max(0.0, min(1.0, s.natural_texture)),
        "hand_quality": max(0.0, min(1.0, s.hand_quality)),
        "body_proportion": max(0.0, min(1.0, s.body_proportion)),
        "skin_rendering": max(0.0, min(1.0, s.skin_rendering)),
        "expression_quality": max(0.0, min(1.0, s.expression_quality)),
        "pose_diversity": max(0.0, min(1.0, s.pose_diversity)),
    }


def compute_quality_base(s: QualitySignals,
                         weights: dict | None = None) -> float:
    w = weights if weights is not None else QUALITY_WEIGHT_DEFAULTS
    factors = _quality_factors(s)
    return sum(w.get(k, 0.0) * v for k, v in factors.items())


def compute_nsfw_total(n: NSFWSignals, mode: str, arousal_boost: float = 1.0,
                       mode_weights: dict | None = None) -> float:
    w = (mode_weights
         or NSFW_MODE_WEIGHTS.get(mode, NSFW_MODE_WEIGHTS["explicit_heavy"]))
    # arousal boost multiplies the arousal component before blending
    return (n.explicitness * w["explicitness"]
            + n.arousal * arousal_boost * w["arousal"]
            + n.skin_exposure * w["skin"])


def compute_composite(s: QualitySignals, n: NSFWSignals, config: SelectionConfig) -> float:
    base = compute_quality_base(s, config.quality_weights)
    nsfw = compute_nsfw_total(n, config.nsfw_mode, config.arousal_boost,
                              config.nsfw_mode_weights or None)
    score = base * (1.0 + nsfw * config.nsfw_weight)

    if s.face_occlusion > 0.5:
        score *= config.occlusion_penalty
    if s.pose_near_duplicate:
        score -= config.pose_dup_penalty

    return max(0.0, min(score, 2.0))


def score_breakdown(s: QualitySignals, n: NSFWSignals,
                    config: SelectionConfig) -> dict:
    """Explainability: per-factor contributions behind one composite score.
    Contributions are weight*value, sorted so the drivers read first."""
    w = config.quality_weights
    factors = _quality_factors(s)
    contrib = {k: round(w.get(k, 0.0) * v, 4) for k, v in factors.items()}
    base = sum(contrib.values())
    nsfw = compute_nsfw_total(n, config.nsfw_mode, config.arousal_boost,
                              config.nsfw_mode_weights or None)
    penalties = []
    if s.face_occlusion > 0.5:
        penalties.append(f"occlusion x{config.occlusion_penalty}")
    if s.pose_near_duplicate:
        penalties.append(f"pose_dup -{config.pose_dup_penalty}")
    return {
        "quality_base": round(base, 4),
        "nsfw_total": round(nsfw, 4),
        "nsfw_intensity": n.intensity_label,
        "composite": round(compute_composite(s, n, config), 4),
        "quality_contributions": dict(
            sorted(contrib.items(), key=lambda kv: -kv[1])),
        "penalties": penalties,
    }


def _fmt_breakdown(fid: str, bd: dict, top: int = 3) -> str:
    drivers = list(bd["quality_contributions"].items())[:top]
    dstr = ", ".join(f"{k} {v:.2f}" for k, v in drivers)
    pen = f" | penalties: {', '.join(bd['penalties'])}" if bd["penalties"] else ""
    return (f"    {fid}: composite {bd['composite']:.3f} "
            f"(base {bd['quality_base']:.3f} | nsfw {bd['nsfw_total']:.2f} "
            f"{bd['nsfw_intensity']}) top: {dstr}{pen}")


# ============================================================
# DATA LOADING
# ============================================================

def _num(v, default=0.0) -> float:
    """Coerce any signal value to a float. signals_json mixes scalars with
    nested .to_dict() sub-dicts, so a value may be a number, dict, list, or None.
    Never raises."""
    if isinstance(v, bool):
        return float(default)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for k in ("score", "overall", "value", "mean", "quality"):
            if isinstance(v.get(k), (int, float)) and not isinstance(v.get(k), bool):
                return float(v[k])
        nums = [x for x in v.values() if isinstance(x, (int, float)) and not isinstance(x, bool)]
        return float(sum(nums) / len(nums)) if nums else float(default)
    if isinstance(v, (list, tuple)):
        nums = [x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
        return float(sum(nums) / len(nums)) if nums else float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _sig(sig: dict, *keys, default=0.0) -> float:
    """First present key from signals_json, coerced to a float."""
    for k in keys:
        if k in sig:
            return _num(sig[k], default)
    return _num(default)


def _tags_from_caption(caption_text, tags_json) -> list[str]:
    """Real captions are comma-separated booru tags; tags_json is a JSON list."""
    if caption_text:
        return [t.strip() for t in str(caption_text).split(",") if t.strip()]
    if tags_json:
        try:
            parsed = json.loads(tags_json)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if str(t).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def load_candidates(conn) -> list[QualitySignals]:
    logger.info("Loading frames + quality signals from manifest...")

    # Real manifest schema:
    #   identity_sim  -> detections.identity_sim (LEFT JOIN)
    #   sharpness     -> frames.sharpness
    #   cluster_id    -> frames.cluster_id
    #   integrity / detail_retention / micro_detail / occlusion / eyes_open
    #                 -> frame_quality
    #   extra signals -> frame_advanced_quality.signals_json (optional, additive)
    rows = conn.execute("""
        SELECT
            f.frame_id, f.path, f.source_id,
            d.identity_sim,
            f.sharpness,
            COALESCE(fq.integrity, 0.0),
            COALESCE(fq.detail_retention, 0.0),
            COALESCE(fq.micro_detail, 0.0),
            COALESCE(fq.occlusion, 0.0),
            COALESCE(fq.eyes_open, 0.0),
            f.cluster_id,
            COALESCE(faq.signals_json, '{}')
        FROM frames f
        LEFT JOIN detections d           ON f.frame_id = d.frame_id
        LEFT JOIN frame_quality fq       ON f.frame_id = fq.frame_id
        LEFT JOIN frame_advanced_quality faq ON f.frame_id = faq.frame_id
        WHERE f.status IN ('selected', 'packaged')
    """).fetchall()

    # Caption tags — critical for NSFW classification (real table: captions).
    # Keep the raw caption_text too, so --apply can export .txt sidecars.
    caption_tags_map: dict[str, list[str]] = {}
    caption_text_map: dict[str, str] = {}
    for fid, cap, tj in conn.execute(
            "SELECT frame_id, caption_text, tags_json FROM captions").fetchall():
        tags = _tags_from_caption(cap, tj)
        if tags:
            caption_tags_map[fid] = tags
        if cap and cap.strip():
            caption_text_map[fid] = cap.strip()

    cands: list[QualitySignals] = []
    excluded_minor = 0
    for row in rows:
        (fid, path, source, identity, sharpness, integrity, detail_retention,
         micro_detail, occlusion, eyes_open, cluster_id, signals_json) = row

        tags = caption_tags_map.get(fid, [])

        # HARD SAFETY GATE: drop any frame implying a minor before scoring.
        if implies_minor(tags):
            excluded_minor += 1
            continue

        try:
            sig = json.loads(signals_json) if signals_json else {}
        except (json.JSONDecodeError, TypeError):
            sig = {}

        # Derive sequence info from filename for temporal spacing.
        src_stem = Path(path).stem if path else ""
        sequence_id, frame_index = src_stem, 0
        for sep in ("_", "-"):
            parts = src_stem.rsplit(sep, 1)
            if len(parts) == 2 and parts[1].isdigit():
                sequence_id, frame_index = parts[0], int(parts[1])
                break

        cands.append(QualitySignals(
            frame_id=fid,
            path=path,
            source=str(source),
            identity_score=_num(identity),
            sharpness=_num(sharpness),
            figure_integrity=_num(integrity, _sig(sig, "figure_integrity_score")),
            anatomy_detail=_num(detail_retention, _sig(sig, "anatomy_detail_score")),
            natural_texture=_sig(sig, "texture_preservation_score",
                                 "natural_detail_score", default=_num(micro_detail)),
            hand_quality=_sig(sig, "hand_quality_score", "hand_quality"),
            body_proportion=_sig(sig, "body_proportion_score"),
            pose_diversity=_sig(sig, "dynamic_pose_score", "pose_diversity_score"),
            skin_rendering=_sig(sig, "skin_rendering_score"),
            expression_quality=_sig(sig, "expression_score",
                                    "expression_quality_score", default=_num(eyes_open)),
            face_occlusion=_sig(sig, "face_occlusion_score",
                                "occlusion_level", default=_num(occlusion)),
            pose_near_duplicate=False,
            cluster_id=cluster_id,
            caption_tags=tags,
            caption_text=caption_text_map.get(fid, ", ".join(tags)),
            sequence_id=sequence_id,
            frame_index=frame_index,
        ))

    logger.info(f"Loaded {len(cands)} candidate frames "
                f"(status selected/packaged)")
    if excluded_minor:
        logger.warning(f"  - SAFETY: excluded {excluded_minor} frame(s) whose "
                       f"captions implied a minor")
    with_tags = sum(1 for c in cands if c.caption_tags)
    with_ident = sum(1 for c in cands if c.identity_score > 0)
    with_clusters = sum(1 for c in cands if c.cluster_id is not None)
    logger.info(f"  - {with_tags} have caption tags, {with_ident} have identity "
                f"scores, {with_clusters} have cluster assignments")
    if with_tags == 0:
        logger.warning("  - No caption tags found — run captioning first, or "
                       "NSFW scoring will be flat.")
    return cands


# ============================================================
# SELECTION — Full Quota + Diversity Logic
# ============================================================

def select_frames(cands: list[QualitySignals], config: SelectionConfig):
    """
    Full selection pipeline:
      1. Score every candidate (quality × nsfw boost)
      2. Hard-filter on minimum quality/identity/sharpness
      3. Greedy selection with:
           - Source fraction cap
           - Cluster fraction cap
           - Temporal spacing within sequences
           - Quota tracking by NSFW intensity
    """
    logger.info("Scoring frames...")
    explain_rejects = config.explain in ("rejected", "all")

    # custom tag rules (normalized substring matching, like _is_explicit)
    exclude_norm = [_norm(t) for t in config.exclude_tags if _norm(t)]
    require_norm = [_norm(t) for t in config.require_any_tags if _norm(t)]

    def _matched_exclude(tags: list[str]) -> str | None:
        for t in tags:
            n = _norm(t)
            for rule in exclude_norm:
                if rule in n:
                    return rule
        return None

    def _has_required(tags: list[str]) -> bool:
        if not require_norm:
            return True
        for t in tags:
            n = _norm(t)
            if any(rule in n for rule in require_norm):
                return True
        return False

    # --- Phase 1: score all (minors are hard-excluded, belt-and-suspenders) ---
    scored: list[tuple[QualitySignals, NSFWSignals, float]] = []
    for s in cands:
        if implies_minor(s.caption_tags):
            continue
        nsfw = classify_nsfw(s.caption_tags)
        score = compute_composite(s, nsfw, config)
        scored.append((s, nsfw, score))

    scored.sort(key=lambda x: x[2], reverse=True)

    explanations: list[SelectionExplanation] = []

    def _explain_reject(s: QualitySignals, reason: str, score: float):
        if explain_rejects:
            explanations.append(SelectionExplanation(
                frame_id=s.frame_id, path=s.path, accepted=False,
                reject_reason=reason, score=round(score, 4)))

    # --- Phase 2: hard gates (quality floors + custom tag rules) ---
    filtered: list[tuple[QualitySignals, NSFWSignals, float]] = []
    gate = defaultdict(int)
    gate_examples: list[tuple[QualitySignals, NSFWSignals, str]] = []
    for s, nsfw, score in scored:
        reason = None
        if s.identity_score < config.min_identity:
            reason = "below_min_identity"
        elif s.sharpness < config.min_sharpness:
            reason = "below_min_sharpness"
        elif compute_quality_base(s, config.quality_weights) < config.min_quality:
            reason = "below_min_quality"
        elif exclude_norm and (hit := _matched_exclude(s.caption_tags)):
            reason = f"excluded_tag:{hit}"
        elif not _has_required(s.caption_tags):
            reason = "missing_required_tag"
        if reason:
            gate[reason.split(":", 1)[0]] += 1
            _explain_reject(s, reason, score)
            if len(gate_examples) < max(0, config.explain_top):
                gate_examples.append((s, nsfw, reason))
            continue
        filtered.append((s, nsfw, score))

    logger.info(f"After quality gate: {len(filtered)} / {len(scored)} candidates remain")
    for reason, cnt in sorted(gate.items(), key=lambda x: -x[1]):
        logger.info(f"  gate drop — {reason}: {cnt}")
    if not filtered and scored:
        logger.warning("Quality gate removed EVERY candidate. Lower "
                       "--min-identity / --min-sharpness / --min-quality, "
                       "or run the quality/identity passes first.")
    if config.explain != "off" and gate_examples:
        logger.info(f"Gate rejects, first {len(gate_examples)} "
                    "(highest-scoring casualties):")
        for s, nsfw, reason in gate_examples:
            bd = score_breakdown(s, nsfw, config)
            logger.info(_fmt_breakdown(s.frame_id, bd) + f" -> {reason}")

    # --- Phase 3: greedy selection with diversity ---
    selected: list[QualitySignals] = []
    rejected: dict[str, int] = defaultdict(int)

    # Diversity tracking
    source_count: dict[str, int] = defaultdict(int)
    cluster_count: dict[int, int] = defaultdict(int)
    seq_last_idx: dict[str, int] = {}  # sequence_id → last selected frame_index

    max_per_source = max(1, int(config.target_count * config.max_source_frac))
    max_per_cluster = max(1, int(config.target_count * config.max_cluster_frac))

    # Quota buckets by NSFW intensity
    quotas = config.resolve_quotas()
    if len(quotas) == 1 and quotas[0].label == "any":
        # Single bucket — no quota logic needed
        quota_buckets: dict[str, dict] = {"any": {"slots": config.target_count, "filled": 0, "min_score": quotas[0].min_score}}
    else:
        remaining = config.target_count
        quota_buckets: dict[str, dict] = {}
        for i, q in enumerate(quotas):
            if i == len(quotas) - 1:
                slots = remaining
            else:
                slots = max(1, int(config.target_count * q.fraction))
                remaining -= slots
            quota_buckets[q.label] = {"slots": slots, "filled": 0, "min_score": q.min_score}

    def _find_quota_bucket(intensity_label: str) -> str | None:
        """Match a frame's intensity to a quota bucket."""
        if "any" in quota_buckets:
            return "any"
        if intensity_label in quota_buckets:
            return intensity_label
        # Fallback: try partial match
        for key in quota_buckets:
            if key in intensity_label or intensity_label in key:
                return key
        return None

    for s, nsfw, score in filtered:
        if len(selected) >= config.target_count:
            break

        # Source cap
        if source_count[s.source] >= max_per_source:
            rejected["source_cap"] += 1
            _explain_reject(s, "source_cap", score)
            continue

        # Cluster cap
        if s.cluster_id is not None and cluster_count[s.cluster_id] >= max_per_cluster:
            rejected["cluster_cap"] += 1
            _explain_reject(s, "cluster_cap", score)
            continue

        # Temporal spacing
        if s.sequence_id and s.sequence_id in seq_last_idx:
            gap = s.frame_index - seq_last_idx[s.sequence_id]
            if 0 < gap < config.temporal_spacing:
                rejected["temporal_spacing"] += 1
                _explain_reject(s, "temporal_spacing", score)
                continue

        # Quota bucket
        bucket_key = _find_quota_bucket(nsfw.intensity_label)
        if bucket_key is None:
            rejected["no_quota_bucket"] += 1
            _explain_reject(s, f"no_quota_bucket:{nsfw.intensity_label}",
                            score)
            continue
        bucket = quota_buckets[bucket_key]
        if bucket["filled"] >= bucket["slots"]:
            rejected["quota_filled"] += 1
            _explain_reject(s, f"quota_filled:{bucket_key}", score)
            continue
        if score < bucket["min_score"]:
            rejected["quota_min_score"] += 1
            _explain_reject(s, f"quota_min_score:{bucket_key}", score)
            continue

        # --- Accept ---
        selected.append(s)
        bucket["filled"] += 1
        source_count[s.source] += 1
        if s.cluster_id is not None:
            cluster_count[s.cluster_id] += 1
        if s.sequence_id:
            seq_last_idx[s.sequence_id] = s.frame_index

        if config.explain in ("accepted", "all"):
            explanations.append(SelectionExplanation(
                frame_id=s.frame_id,
                path=s.path,
                accepted=True,
                quota_label=bucket_key,
                score=round(score, 4),
            ))

    # score breakdowns for the top accepted frames (explainability)
    if config.explain != "off" and selected and config.explain_top > 0:
        logger.info(f"Top {min(config.explain_top, len(selected))} "
                    "accepted, by composite score:")
        for s in selected[:config.explain_top]:
            nsfw = classify_nsfw(s.caption_tags)
            logger.info(_fmt_breakdown(s.frame_id,
                                       score_breakdown(s, nsfw, config)))

    # --- Logging ---
    if rejected:
        logger.info("Rejection breakdown:")
        for reason, cnt in sorted(rejected.items(), key=lambda x: -x[1]):
            logger.info(f"  {reason}: {cnt}")

    # NSFW intensity distribution
    intensity_dist: dict[str, int] = defaultdict(int)
    for s in selected:
        nsfw = classify_nsfw(s.caption_tags)
        intensity_dist[nsfw.intensity_label] += 1
    logger.info("NSFW intensity in selection:")
    for label, cnt in sorted(intensity_dist.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / len(selected) if selected else 0
        logger.info(f"  {label}: {cnt} ({pct:.1f}%)")

    # Quota fulfillment
    logger.info("Quota fulfillment:")
    for key, b in quota_buckets.items():
        logger.info(f"  {key}: {b['filled']} / {b['slots']}")

    summary = {
        "selected": len(selected),
        "target": config.target_count,
        "candidates_total": len(cands),
        "after_quality_gate": len(filtered),
        "profile": config.profile,
        "nsfw_mode": config.nsfw_mode,
        "nsfw_weight": config.nsfw_weight,
        "arousal_boost": config.arousal_boost,
        "intensity_distribution": dict(intensity_dist),
        "quota_fulfillment": {k: v["filled"] for k, v in quota_buckets.items()},
        "gate_reasons": dict(gate),
        "rejection_reasons": dict(rejected),
        "unique_sources": len(source_count),
        "unique_clusters": len(cluster_count),
        "explain_mode": config.explain,
    }

    return selected, explanations, summary


# ============================================================
# OUTPUT
# ============================================================

def _write_dataset_config(output_path: Path, config: SelectionConfig) -> None:
    """kohya sd-scripts dataset config beside the export (consolidated from
    the retired standalone DreamBooth exporter)."""
    lines = [
        "# kohya sd-scripts dataset config (SDXL DreamBooth / DreamBooth-LoRA)",
        "[general]",
        "shuffle_caption = true",
        "caption_extension = '.txt'",
        "keep_tokens = 1",
        "",
        "[[datasets]]",
        "resolution = 1024",
        "batch_size = 1",
        "enable_bucket = true",
        "min_bucket_reso = 768",
        "max_bucket_reso = 1280",
        "",
        "  [[datasets.subsets]]",
        "  image_dir = 'instance'",
        f"  class_tokens = '{config.token} {config.class_word}'",
        f"  num_repeats = {config.repeats}",
    ]
    if config.reg:
        lines += [
            "",
            "  [[datasets.subsets]]",
            f"  image_dir = 'reg/1_{config.class_word}'",
            f"  class_tokens = '{config.class_word}'",
            "  is_reg = true",
            "  num_repeats = 1",
        ]
    (output_path / "dataset_config.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def _write_readme(output_path: Path, config: SelectionConfig,
                  n_present: int) -> None:
    reg_note = (
        f"A regularization folder (reg/1_{config.class_word}) was created - "
        f"drop ~{config.reg_count} class images there before training.\n"
        if config.reg else "")
    (output_path / "README.md").write_text(
        f"# DreamBooth export ({n_present} instance images)\n\n"
        f"Selected by the DreamBooth selector (profile "
        f"'{config.profile}', nsfw_mode '{config.nsfw_mode}').\n\n"
        "- `instance/` - instance images + per-image caption `.txt`\n"
        "- `dataset_config.toml` - kohya data config (repeats "
        f"{config.repeats})\n"
        "- `training_config.toml` + `train.sh` - suggested trainer args and "
        "launch command (edit freely)\n"
        "- `selection_manifest.json` / `selection_log.jsonl` - what was "
        "picked and why\n"
        "- `experiment_template.json` - after training, register the model "
        "with `pixel-curator exp register --from-file ... --model ...`\n\n"
        f"{reg_note}"
        "Zip this folder -> Google Drive -> Colab, or train locally with "
        "`bash train.sh`.\n", encoding="utf-8")


def write_outputs(
    selected: list[QualitySignals],
    explanations: list[SelectionExplanation],
    summary: dict,
    config: SelectionConfig,
    output_path: Path,
    apply: bool = False,
    manifest_base: Path | None = None,
) -> dict:
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_path / "selection_manifest.json"
    log_path = output_path / "selection_log.jsonl"

    with open(manifest_path, "w") as f:
        json.dump({
            "selected_count": len(selected),
            "config": {
                "profile": config.profile,
                "target": config.target_count,
                "nsfw_mode": config.nsfw_mode,
                "nsfw_weight": config.nsfw_weight,
                "arousal_boost": config.arousal_boost,
                "min_quality": config.min_quality,
                "min_identity": config.min_identity,
                "min_sharpness": config.min_sharpness,
                "max_source_frac": config.max_source_frac,
                "max_cluster_frac": config.max_cluster_frac,
                "temporal_spacing": config.temporal_spacing,
                "pose_dup_penalty": config.pose_dup_penalty,
                "occlusion_penalty": config.occlusion_penalty,
                "quality_weights": config.quality_weights,
                "exclude_tags": config.exclude_tags,
                "require_any_tags": config.require_any_tags,
                "quotas": [f"{q.label}:{q.fraction}:{q.min_score}"
                           for q in config.quotas],
                "explain": config.explain,
            },
            "summary": summary,
        }, f, indent=2)

    with open(log_path, "w") as f:
        for exp in explanations:
            f.write(json.dumps({
                "frame_id": exp.frame_id,
                "path": exp.path,
                "accepted": exp.accepted,
                "quota_label": exp.quota_label,
                "score": exp.score,
                "reject_reason": exp.reject_reason,
            }) + "\n")

    if apply:
        instance_dir = output_path / "instance"
        instance_dir.mkdir(exist_ok=True)

        linked = 0          # genuinely (re)written this run
        present = 0         # usable image on disk after this run
        missing = 0         # selected frames whose source file is gone
        failed = 0
        captioned = 0
        used_names: set = set()
        for i, s in enumerate(selected):
            src = Path(s.path)
            if not src.is_file():
                missing += 1
                logger.warning(f"source missing, skipped: {src.name}")
                continue
            # collision-proof: two sources sharing a basename must not
            # overwrite each other (silently dropping a frame)
            name = src.name
            if name in used_names:
                name = f"{src.stem}__{i:06d}{src.suffix}"
            used_names.add(name)
            dst = instance_dir / name
            try:
                # a stale symlink from a previous export must not block a
                # real copy
                if dst.is_symlink() and config.link_mode != "symlink":
                    try:
                        dst.unlink()
                    except OSError:
                        pass
                if not dst.exists():
                    if config.link_mode == "symlink":
                        dst.symlink_to(src.resolve())
                    else:
                        shutil.copy2(src, dst)
                    linked += 1
                present += 1
                # write the caption sidecar so kohya can train with it
                if s.caption_text:
                    txt = dst.with_suffix(".txt")
                    if not txt.exists():
                        txt.write_text(s.caption_text.strip() + "\n", encoding="utf-8")
                        captioned += 1
            except Exception as e:
                logger.warning(f"Failed to link/copy {src.name}: {e}")
                failed += 1

        logger.info(f"Instance folder: {linked} written, {present} usable on "
                    f"disk, {captioned} captions, {failed} failed"
                    + (f", {missing} missing sources" if missing else ""))
        summary["images_present"] = present
        summary["missing_sources"] = missing

        # regularization (class) set scaffold
        if config.reg:
            (output_path / "reg" / f"1_{config.class_word}").mkdir(
                parents=True, exist_ok=True)

        _write_dataset_config(output_path, config)
        _write_readme(output_path, config, present)

        # Phase 2 versioning synergy: register the applied selection in the
        # dataset registry so it joins `history`/lineage. Never fatal.
        if config.register_version and manifest_base is not None:
            try:
                from ..dataset_versioning import register_folder_export
                files = [{"rel_path": f"instance/{Path(s.path).name}",
                          "source_path": s.path,
                          "caption": s.caption_text} for s in selected]
                rec = register_folder_export(
                    Path(manifest_base),
                    kind="dreambooth_selection",
                    name=f"dreambooth_{config.profile}_{config.nsfw_mode}",
                    export_dir=output_path,
                    files=files,
                    image_count=len(selected),
                    params={"profile": config.profile,
                            "target": config.target_count,
                            "nsfw_mode": config.nsfw_mode,
                            "nsfw_weight": config.nsfw_weight,
                            "min_quality": config.min_quality},
                    note=f"selector v5 ({summary.get('selected', 0)} frames)",
                )
                for w in rec.get("warnings", []):
                    logger.warning(f"versioning: {w}")
                if rec and rec.get("version"):
                    logger.info(f"Registered as dataset version "
                                f"v{rec['version']} (see: pixel-curator "
                                "history)")
                    # Phase 8: prefilled experiment template so the
                    # Colab-trained model can be linked back afterwards.
                    from ..experiments import write_experiment_template
                    tpl = write_experiment_template(
                        output_path,
                        name=f"dreambooth_{config.profile}"
                             f"_v{rec['version']:03d}",
                        dataset_version=rec["version"],
                        dataset_dir=str(output_path),
                        model_type="dreambooth",
                        params={"profile": config.profile,
                                "nsfw_mode": config.nsfw_mode,
                                "target": config.target_count})
                    if tpl:
                        logger.info(
                            "Experiment template written: after training, "
                            "run `pixel-curator exp register --from-file "
                            f"{tpl} --model <trained.safetensors>`")
                    # Phase 10: ready-to-train config beside the export
                    if config.training_config:
                        from .. import manifest as _mf
                        from ..train.export_config import generate_for_export
                        _conn = _mf.connect(Path(manifest_base))
                        try:
                            _, msgs = generate_for_export(
                                _conn, output_path,
                                dataset_version=rec["version"],
                                frame_count=len(selected),
                                model_type="dreambooth")
                        finally:
                            _conn.close()
                        for m in msgs:
                            logger.info(m)
            except Exception as exc:
                logger.warning(f"Version registration skipped: {exc}")
        elif config.training_config:
            # unversioned export: count-based recommendation still helps
            try:
                from ..train.export_config import generate_for_export
                _, msgs = generate_for_export(
                    None, output_path, frame_count=len(selected),
                    model_type="dreambooth")
                for m in msgs:
                    logger.info(m)
            except Exception as exc:      # pragma: no cover
                logger.warning(f"Training config skipped: {exc}")

    logger.info(f"Selection manifest written to {manifest_path}")
    logger.info(f"Selection log written to {log_path}")
    return {"manifest": str(manifest_path), "log": str(log_path)}


# ============================================================
# MAIN
# ============================================================

def main():
    # CLI defaults are None sentinels: only flags the user actually passed
    # override the profile/TOML layers (provenance stays honest).
    parser = argparse.ArgumentParser(
        description="Configurable NSFW DreamBooth Dataset Selector (v5). "
                    "Config layers: profile -> [dreambooth_selector] TOML "
                    "section -> CLI flags.")
    parser.add_argument("--project", required=True, help="Path to project TOML")
    parser.add_argument("--profile", default=None,
                        choices=sorted(SELECTION_PROFILES),
                        help="selection profile (default: balanced == "
                             "legacy behavior)")
    parser.add_argument("--target", type=int, default=None,
                        help="Target frame count (default 7500)")
    parser.add_argument(
        "--nsfw-mode",
        default=None,
        choices=["balanced", "strong", "explicit_heavy"],
        help="NSFW scoring mode (default explicit_heavy)",
    )
    parser.add_argument("--nsfw-weight", type=float, default=None,
                        help="NSFW weight in composite 0-1 (default 0.50)")
    parser.add_argument("--arousal-boost", type=float, default=None,
                        help="Multiplier on arousal signals (default 1.0)")
    parser.add_argument("--min-quality", type=float, default=None,
                        help="minimum quality-base to pass the gate "
                             "(default 0.60)")
    parser.add_argument("--min-identity", type=float, default=None,
                        help="minimum identity_sim; 0 disables "
                             "(default 0.30)")
    parser.add_argument("--min-sharpness", type=float, default=None,
                        help="minimum sharpness; 0 disables (default 35)")
    parser.add_argument("--max-source-frac", type=float, default=None,
                        help="per-source cap as a fraction of target "
                             "(default 0.18)")
    parser.add_argument("--max-cluster-frac", type=float, default=None,
                        help="per-cluster cap as a fraction of target "
                             "(default 0.22)")
    parser.add_argument("--temporal-spacing", type=int, default=None,
                        help="min frame-index gap within a sequence "
                             "(default 15)")
    parser.add_argument("--exclude-tag", action="append", default=None,
                        dest="exclude_tags", metavar="TAG",
                        help="reject frames whose captions contain TAG "
                             "(repeatable)")
    parser.add_argument("--require-tag", action="append", default=None,
                        dest="require_any_tags", metavar="TAG",
                        help="keep only frames containing at least one "
                             "required TAG (repeatable)")
    parser.add_argument("--quota", action="append", default=None,
                        dest="quotas", metavar="LABEL:FRAC[:MIN]",
                        help="intensity quota bucket, e.g. explicit:0.6 "
                             "(repeatable; labels: covered/semi_explicit/"
                             "explicit)")
    parser.add_argument("--explain", default=None, choices=EXPLAIN_MODES,
                        help="per-frame explanation detail in the selection "
                             "log (default: accepted)")
    parser.add_argument("--explain-top", type=int, default=None,
                        help="score breakdowns logged for the top-N "
                             "accepted / first-N gate rejects (default 10)")
    parser.add_argument("--apply", action="store_true", help="Write files to output/instance")
    parser.add_argument("--copy", action="store_true",
                        help="Copy real image files into instance/ (default: symlinks). "
                             "Use this when the output will be uploaded/moved off-machine.")
    parser.add_argument("--no-version", dest="register_version",
                        action="store_false", default=None,
                        help="skip registering the --apply export in the "
                             "dataset version registry")
    parser.add_argument("--no-training-config", dest="training_config",
                        action="store_false", default=None,
                        help="skip the auto-generated training_config.toml/"
                             "train.sh beside the export (Phase 10)")
    parser.add_argument("--repeats", type=int, default=None,
                        help="kohya num_repeats in dataset_config.toml "
                             "(default 1)")
    parser.add_argument("--reg", action="store_true", default=None,
                        help="scaffold a regularization (class) folder")
    parser.add_argument("--reg-count", dest="reg_count", type=int,
                        default=None,
                        help="suggested class-image count for the reg note")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    prj = _load_project(args.project)
    toml_section = dict(getattr(prj, "dreambooth_selector", {}) or {})

    cli_values = {
        "profile": args.profile,
        "target": args.target,
        "nsfw_mode": args.nsfw_mode,
        "nsfw_weight": args.nsfw_weight,
        "arousal_boost": args.arousal_boost,
        "min_quality": args.min_quality,
        "min_identity": args.min_identity,
        "min_sharpness": args.min_sharpness,
        "max_source_frac": args.max_source_frac,
        "max_cluster_frac": args.max_cluster_frac,
        "temporal_spacing": args.temporal_spacing,
        "exclude_tags": args.exclude_tags,
        "require_any_tags": args.require_any_tags,
        "quotas": args.quotas,
        "explain": args.explain,
        "explain_top": args.explain_top,
        "register_version": args.register_version,
        "training_config": args.training_config,
        "repeats": args.repeats,
        "reg": args.reg,
        "reg_count": args.reg_count,
        "token": getattr(prj, "trigger_token", None) or None,
        "class_word": getattr(prj, "class_word", None) or None,
        "link_mode": ("copy" if args.copy else None),
    }
    config, provenance, cfg_warnings = resolve_selection_config(
        toml_section, cli_values)
    for w in cfg_warnings:
        logger.warning(f"config: {w}")
    logger.info(render_resolved_config(config, provenance))
    logger.info(f"Config: target={config.target_count}, nsfw_mode={config.nsfw_mode}, "
                f"nsfw_weight={config.nsfw_weight}, arousal_boost={config.arousal_boost}")

    conn = _manifest_mod.connect(prj.output_path, readonly=True)

    try:
        cands = load_candidates(conn)
        selected, explanations, summary = select_frames(cands, config)
        results = write_outputs(selected, explanations, summary, config,
                                Path(args.output), apply=args.apply,
                                manifest_base=prj.output_path)
        logger.info(f"Done. Selected {len(selected)} frames.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
