"""Optional personal-media suitability analyzer (organization only).

A purely additive, local-first helper that scores how *technically useful* a
personal training sample is likely to be for downstream review, then routes it
into one of two organizational branches:

    high_suitability       strong technical/identity signal - review first
    standard_suitability    everything else - the default

This is organization metadata, nothing else. It does NOT filter, reject,
delete, quarantine, exclude, or restrict any sample, and it never changes a
sample's ``lora_target``. When disabled (the default) it does nothing and the
DatasetSplitter output is byte-for-byte unchanged.

Scoring uses only benign, professional training-quality signals that may
already exist in a sample's metadata - sharpness, face visibility, identity
confidence, and resolution. Missing signals are treated as neutral so absent
metadata never penalizes a sample. Pure stdlib, dependency-light, fully
self-contained (no out-of-repo imports).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ANALYZER_VERSION = "1.0"

#: The only two valid organizational branches.
SUITABILITY_BRANCHES = ("high_suitability", "standard_suitability")
DEFAULT_BRANCH = "standard_suitability"
DEFAULT_HIGH_THRESHOLD = 0.75

# Reference points for normalizing raw quality signals into 0..1.
_SHARPNESS_REF = 120.0          # variance-of-Laplacian-ish "good enough" point
_RESOLUTION_REF = 1024 * 1024   # 1024x1024 target
_NEUTRAL = 0.5                  # score for a missing signal (never penalize)

# weight of each signal in the blended suitability score (sums to 1.0)
_WEIGHTS = {
    "sharpness": 0.30,
    "face_visibility": 0.25,
    "identity_confidence": 0.25,
    "resolution": 0.20,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "on")


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


@dataclass
class PersonalAnalyzerConfig:
    """Disabled by default. ``high_suitability_threshold`` is the blended-score
    cutoff for the high branch; ``write_manifest_fields`` controls whether the
    verbose analyzer payload is persisted into manifests (the lightweight branch
    label is always kept when enabled)."""
    enabled: bool = False
    high_suitability_threshold: float = DEFAULT_HIGH_THRESHOLD
    write_manifest_fields: bool = True

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "PersonalAnalyzerConfig":
        d = d or {}
        return cls(
            enabled=_as_bool(d.get("enabled", False)),
            high_suitability_threshold=_as_float(
                d.get("high_suitability_threshold", DEFAULT_HIGH_THRESHOLD),
                DEFAULT_HIGH_THRESHOLD),
            write_manifest_fields=_as_bool(d.get("write_manifest_fields", True)),
        )


def load_personal_analyzer_config(project: Any) -> PersonalAnalyzerConfig:
    """Read ``[personal_analyzer]`` from a project file (.toml/.json) or a dict.
    Defensive: any problem returns the disabled default - never raises."""
    try:
        if isinstance(project, dict):
            data = project
        elif isinstance(project, (str, Path)):
            from .. import config as _cfg
            text = Path(project).expanduser().read_text(encoding="utf-8")
            if getattr(_cfg, "HAVE_TOML", False):
                import tomllib
                data = tomllib.loads(text)
            else:
                data = _cfg._parse_flat_toml(text)
        else:
            return PersonalAnalyzerConfig()
        section = (data.get("personal_analyzer") or {}) if isinstance(data, dict) else {}
        return PersonalAnalyzerConfig.from_dict(section)
    except Exception:
        return PersonalAnalyzerConfig()


def coerce_config(value: Any) -> PersonalAnalyzerConfig:
    """Normalize the many ways a caller might pass analyzer config into a
    PersonalAnalyzerConfig. ``None`` / unknown -> disabled default."""
    if value is None:
        return PersonalAnalyzerConfig()
    if isinstance(value, PersonalAnalyzerConfig):
        return value
    if isinstance(value, bool):
        return PersonalAnalyzerConfig(enabled=value)
    if isinstance(value, dict):
        return PersonalAnalyzerConfig.from_dict(value)
    if isinstance(value, (str, Path)):
        return load_personal_analyzer_config(value)
    return PersonalAnalyzerConfig()


# ---------------------------------------------------------------------------
# Branch normalization
# ---------------------------------------------------------------------------

def normalize_suitability_branch(value: Any) -> str:
    """Map any value to a valid branch. Only an exact ``high_suitability``
    yields the high branch; everything else (including invalid input and
    ``standard_suitability`` itself) normalizes to ``standard_suitability``."""
    v = str(value or "").strip().lower()
    return "high_suitability" if v == "high_suitability" else DEFAULT_BRANCH


# ---------------------------------------------------------------------------
# Signal extraction (defensive: top-level or nested under "metadata")
# ---------------------------------------------------------------------------

def _lookup(sample: dict, keys: tuple, default: Optional[float] = None
           ) -> Optional[float]:
    meta = sample.get("metadata") if isinstance(sample, dict) else None
    for store in (sample, meta):
        if not isinstance(store, dict):
            continue
        for k in keys:
            if k in store and store[k] is not None:
                try:
                    return float(store[k])
                except (TypeError, ValueError):
                    continue
    return default


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _signal_sharpness(sample: dict) -> Optional[float]:
    s = _lookup(sample, ("sharpness", "sharpness_score", "focus", "blur_score"))
    return _clamp01(s / _SHARPNESS_REF) if s is not None else None


def _signal_face_visibility(sample: dict) -> Optional[float]:
    fv = _lookup(sample, ("face_visibility", "face_area", "face_area_ratio",
                          "face_ratio"))
    return _clamp01(fv) if fv is not None else None


def _signal_identity(sample: dict) -> Optional[float]:
    idc = _lookup(sample, ("identity_confidence", "identity_sim",
                           "identity_score", "det_conf"))
    return _clamp01(idc) if idc is not None else None


def _signal_resolution(sample: dict) -> Optional[float]:
    w = _lookup(sample, ("width", "image_width", "w"))
    h = _lookup(sample, ("height", "image_height", "h"))
    if w is not None and h is not None:
        return _clamp01((w * h) / _RESOLUTION_REF)
    mp = _lookup(sample, ("megapixels",))
    if mp is not None:
        return _clamp01((mp * 1_000_000) / _RESOLUTION_REF)
    px = _lookup(sample, ("pixels",))
    if px is not None:
        return _clamp01(px / _RESOLUTION_REF)
    return None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_personal_sample(sample: dict, *,
                            high_threshold: float = DEFAULT_HIGH_THRESHOLD
                            ) -> dict:
    """Compute a blended suitability score and organizational branch for one
    sample. Returns ``suitability_branch``, ``suitability_score``, per-signal
    values and ``reason_codes``. Read-only; never mutates ``sample``."""
    raw = {
        "sharpness": _signal_sharpness(sample),
        "face_visibility": _signal_face_visibility(sample),
        "identity_confidence": _signal_identity(sample),
        "resolution": _signal_resolution(sample),
    }
    score = sum(_WEIGHTS[k] * (v if v is not None else _NEUTRAL)
                for k, v in raw.items())
    score = round(_clamp01(score), 3)
    branch = ("high_suitability" if score >= high_threshold
              else "standard_suitability")

    reasons: list[str] = []
    for k, v in raw.items():
        if v is None:
            reasons.append(f"{k}_missing_neutral")
        elif v >= 0.75:
            reasons.append(f"{k}_strong")
        elif v <= 0.25:
            reasons.append(f"{k}_weak")

    return {
        "suitability_branch": branch,
        "suitability_score": score,
        "signals": {k: (round(v, 3) if v is not None else None)
                    for k, v in raw.items()},
        "reason_codes": reasons,
        "threshold": round(float(high_threshold), 3),
        "analyzer_version": ANALYZER_VERSION,
    }


def enrich_sample(sample: dict, config: Optional[PersonalAnalyzerConfig]
                  ) -> dict:
    """Optionally attach suitability metadata to ``sample`` in place.

    No-op when ``config`` is None/disabled (output stays unchanged). On any
    analyzer error the sample is labelled ``standard_suitability`` and a warning
    is appended - it never raises, never removes anything, never touches
    ``lora_target``. Mirrors the agreed integration contract.
    """
    cfg = config if isinstance(config, PersonalAnalyzerConfig) else coerce_config(config)
    if not cfg.enabled:
        return sample
    try:
        analysis = analyze_personal_sample(
            sample, high_threshold=cfg.high_suitability_threshold)
        sample["personal_suitability"] = normalize_suitability_branch(
            analysis.get("suitability_branch"))
        if cfg.write_manifest_fields:
            sample["personal_analyzer"] = analysis
    except Exception as exc:
        sample["personal_suitability"] = "standard_suitability"
        sample.setdefault("warnings", []).append(
            f"personal_analyzer_failed: {exc}")
    return sample


__all__ = [
    "PersonalAnalyzerConfig", "load_personal_analyzer_config", "coerce_config",
    "analyze_personal_sample", "enrich_sample", "normalize_suitability_branch",
    "SUITABILITY_BRANCHES", "DEFAULT_HIGH_THRESHOLD", "ANALYZER_VERSION",
]
