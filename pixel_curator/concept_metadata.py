"""Concept metadata normalization - the hardened sidecar foundation.

Every sidecar (v2 rich, v1 native, legacy simple, or absent) normalizes
into one `ConceptMetadata` object so the Explorer, attribute controls,
stack resolution, and future concept probing all read a single shape.
Unknown values never crash anything: they normalize to studio-safe
defaults and surface as warning reasons.  Pure stdlib, additive.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Controlled vocabularies (unknown values -> safe default + warning)
# ---------------------------------------------------------------------------

RESPONSE_CURVES = ("linear", "slow_start", "fast_start", "damped", "stepped")
PRIORITY_HINTS = ("anchor", "primary", "normal", "supporting", "experimental")
RISK_LEVELS = ("low", "medium", "high")
SENSITIVITY_LEVELS = ("low", "medium", "high")
CONFLICT_SEVERITIES = ("low", "medium", "high")

DEFAULT_FAMILY = "general_concept"
DEFAULT_WEIGHT_RANGE = (0.2, 0.7)        # conservative studio default

# priority_hint -> resolver priority adjustment (additive, conservative)
PRIORITY_HINT_DELTA = {"anchor": 40, "primary": 10, "normal": 0,
                       "supporting": -15, "experimental": -25}


def _norm_enum(value, allowed: tuple, default: str,
               field_name: str, warnings: list[str]) -> str:
    v = str(value or "").strip().lower()
    if v in allowed:
        return v
    if v:
        warnings.append(f"unknown {field_name} '{value}' -> '{default}'")
    return default


def curve_fn(name: str):
    """Response curve registry (0..1 -> 0..1, monotonic)."""
    def smooth(t):  # damped: ease in/out
        t = max(0.0, min(1.0, t))
        return t * t * (3 - 2 * t)
    table = {
        "linear": lambda t: max(0.0, min(1.0, t)),
        "slow_start": lambda t: max(0.0, min(1.0, t)) ** 2,
        "fast_start": lambda t: max(0.0, min(1.0, t)) ** 0.5,
        "damped": smooth,
        "stepped": lambda t: round(max(0.0, min(1.0, t)) * 4) / 4,
    }
    return table.get(name, table["damped"])


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ControlAxis:
    axis_id: str
    label: str = ""
    description: str = ""
    recommended_range: tuple = (0.0, 1.0)
    safe_default: float = 0.5
    response_curve: str = "damped"
    identity_sensitivity: str = "medium"


@dataclass
class KnownConflict:
    concept_family: str = ""
    lora_id: str = ""
    reason: str = ""
    severity: str = "medium"


@dataclass
class ConceptMetadata:
    """Normalized internal metadata - the single shape the app consumes."""
    schema_version: int = SCHEMA_VERSION
    display_name: str = ""
    concept_family: str = DEFAULT_FAMILY
    concept_tags: list = field(default_factory=list)
    control_axes: list = field(default_factory=list)      # [ControlAxis]
    priority_hint: str = "normal"
    identity_risk_level: str = "medium"
    recommended_weight_range: tuple = DEFAULT_WEIGHT_RANGE
    known_conflicts: list = field(default_factory=list)   # [KnownConflict]
    preview_images: dict = field(default_factory=dict)
    signal_hooks: dict = field(default_factory=dict)      # reserved
    notes: str = ""
    metadata_source: str = "inferred"
    normalization_warnings: list = field(default_factory=list)

    def to_payload(self) -> dict:
        d = asdict(self)
        d["control_axes"] = [asdict(a) if not isinstance(a, dict) else a
                             for a in self.control_axes]
        d["known_conflicts"] = [asdict(c) if not isinstance(c, dict) else c
                                for c in self.known_conflicts]
        return d


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _norm_range(value, default: tuple, field_name: str,
                warnings: list[str]) -> tuple:
    try:
        lo, hi = float(value[0]), float(value[1])
        if lo > hi:
            lo, hi = hi, lo
        return (max(0.0, lo), min(1.5, hi))
    except Exception:
        if value is not None:
            warnings.append(f"invalid {field_name} -> {default}")
        return default


def _norm_axis(raw: dict, warnings: list[str]) -> Optional[ControlAxis]:
    if not isinstance(raw, dict) or not raw.get("axis_id"):
        warnings.append("control axis without axis_id skipped")
        return None
    return ControlAxis(
        axis_id=str(raw["axis_id"]),
        label=str(raw.get("label") or raw["axis_id"]),
        description=str(raw.get("description") or ""),
        recommended_range=_norm_range(raw.get("recommended_range"),
                                      (0.0, 1.0), "recommended_range",
                                      warnings),
        safe_default=float(raw.get("safe_default") or 0.5),
        response_curve=_norm_enum(raw.get("response_curve"),
                                  RESPONSE_CURVES, "damped",
                                  "response_curve", warnings),
        identity_sensitivity=_norm_enum(raw.get("identity_sensitivity"),
                                        SENSITIVITY_LEVELS, "medium",
                                        "identity_sensitivity", warnings))


def _norm_conflict(raw, warnings: list[str]) -> Optional[KnownConflict]:
    if isinstance(raw, str):                 # v1 style: bare lora id
        return KnownConflict(lora_id=raw)
    if not isinstance(raw, dict):
        warnings.append("unrecognized known_conflict entry skipped")
        return None
    return KnownConflict(
        concept_family=str(raw.get("concept_family") or ""),
        lora_id=str(raw.get("lora_id") or raw.get("lora") or ""),
        reason=str(raw.get("reason") or ""),
        severity=_norm_enum(raw.get("severity"), CONFLICT_SEVERITIES,
                            "medium", "conflict severity", warnings))


def normalize_concept_metadata(raw: Optional[dict], stem: str,
                               source: str = "inferred") -> ConceptMetadata:
    """Any sidecar shape (or None) -> normalized ConceptMetadata.
    Tolerant by contract: never raises."""
    warnings: list[str] = []
    raw = raw if isinstance(raw, dict) else {}

    # legacy mappings (simple sidecars)
    tags = raw.get("concept_tags")
    if tags is None and "tags" in raw:
        tags = raw.get("tags")
        warnings.append("legacy 'tags' mapped to concept_tags")
    previews = raw.get("preview_images")
    if not isinstance(previews, dict):
        previews = {}
    if not previews and raw.get("preview"):
        previews = {"default": str(raw["preview"])}
        warnings.append("legacy 'preview' mapped to preview_images.default")

    family = str(raw.get("concept_family") or raw.get("category")
                 or raw.get("family") or DEFAULT_FAMILY).strip().lower()

    axes = []
    for a in (raw.get("control_axes") or []):
        ax = _norm_axis(a, warnings)
        if ax:
            axes.append(ax)
    conflicts = []
    for c in (raw.get("known_conflicts") or []):
        kc = _norm_conflict(c, warnings)
        if kc:
            conflicts.append(kc)

    risk = raw.get("identity_risk_level", raw.get("identity_risk"))
    wr = raw.get("recommended_weight_range")
    if wr is None and ("weight_min" in raw or "weight_max" in raw):
        wr = (raw.get("weight_min", DEFAULT_WEIGHT_RANGE[0]),
              raw.get("weight_max", DEFAULT_WEIGHT_RANGE[1]))

    return ConceptMetadata(
        schema_version=int(raw.get("schema_version") or 1),
        display_name=str(raw.get("display_name") or stem),
        concept_family=family,
        concept_tags=[str(t) for t in (tags or [])],
        control_axes=axes,
        priority_hint=_norm_enum(raw.get("priority_hint"), PRIORITY_HINTS,
                                 "normal", "priority_hint", warnings),
        identity_risk_level=_norm_enum(risk, RISK_LEVELS, "medium",
                                       "identity_risk_level", warnings),
        recommended_weight_range=_norm_range(
            wr, DEFAULT_WEIGHT_RANGE, "recommended_weight_range", warnings),
        known_conflicts=conflicts,
        preview_images={str(k): str(v) for k, v in previews.items()},
        signal_hooks=dict(raw.get("signal_hooks") or {}),   # reserved
        notes=str(raw.get("notes") or raw.get("description") or ""),
        metadata_source=source,
        normalization_warnings=warnings)


def priority_for_hint(base_priority: int, hint: str) -> int:
    """Conservative resolver-priority adjustment from a priority hint."""
    return max(5, min(100,
                      base_priority + PRIORITY_HINT_DELTA.get(hint, 0)))
