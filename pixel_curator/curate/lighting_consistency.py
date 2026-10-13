"""Dataset-level lighting coherence analysis (Pattern B).

Reads per-frame lighting signals already stored in the manifest
(``frame_advanced_quality.signals_json``, ``frame_quality``) and produces
a single aggregate report answering:

  * Is the lighting in my selected set coherent, or scattered?
  * What is the dominant lighting profile?
  * Which frames are outliers dragging coherence down?
  * Which lighting axes are over- / under-represented?

No new pixel work is performed.  The module is read-only, opt-in,
dependency-light (stdlib + numpy), deterministic, and never raises for
ordinary input.
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric versioning
# ---------------------------------------------------------------------------
METRIC_VERSION: int = 1

# ---------------------------------------------------------------------------
# Axis weights (must sum to 1.0 across all four axes)
# ---------------------------------------------------------------------------
W_DIRECTION: float = 0.40    # Directional consistency matters most for identity
W_TEMPERATURE: float = 0.25  # Color temperature shifts cause prompt fights
W_KEY_TO_FILL: float = 0.20  # Contrast ratio affects face modelling
W_EXPOSURE: float = 0.15     # Exposure variation is easiest for the model

_AXIS_WEIGHTS: dict[str, float] = {
    "direction": W_DIRECTION,
    "temperature": W_TEMPERATURE,
    "key_to_fill": W_KEY_TO_FILL,
    "exposure": W_EXPOSURE,
}
assert abs(sum(_AXIS_WEIGHTS.values()) - 1.0) < 1e-9, \
    f"axis weights must sum to 1.0, got {sum(_AXIS_WEIGHTS.values())}"

# ---------------------------------------------------------------------------
# Scalar RANGE constants
# ---------------------------------------------------------------------------
# Each RANGE represents the "practically meaningful spread" for that axis.
# A std-dev equal to RANGE maps to coherence 0; zero std maps to 1.
# Chosen from empirical LoRA training experience:
#   temperature 0.35 – a full warm↔cool swing in normalised units
#   key_to_fill 0.40 – flat fill ↔ dramatic Rembrandt contrast range
#   exposure    0.30 – ±1.5 EV equivalent in normalised 0..1 space
RANGE_TEMPERATURE: float = 0.35
RANGE_KEY_TO_FILL: float = 0.40
RANGE_EXPOSURE: float = 0.30

_SCALAR_RANGES: dict[str, float] = {
    "temperature": RANGE_TEMPERATURE,
    "key_to_fill": RANGE_KEY_TO_FILL,
    "exposure": RANGE_EXPOSURE,
}

# ---------------------------------------------------------------------------
# Outlier thresholds
# ---------------------------------------------------------------------------
OUTLIER_STD_THRESHOLD: float = 2.0      # scalar axes: > N std from mean
OUTLIER_CATEGORY_FRAC: float = 0.05     # categorical: < 5% population
OUTLIER_CAP: int = 200                  # max reported outlier tuples

# ---------------------------------------------------------------------------
# Minimum frame count for a meaningful report
# ---------------------------------------------------------------------------
DEFAULT_MIN_FRAME_COUNT: int = 20

# ---------------------------------------------------------------------------
# Valid direction categories
# ---------------------------------------------------------------------------
VALID_DIRECTIONS = frozenset({
    "frontal", "left", "right", "top", "back", "rim", "mixed", "unknown",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if math.isnan(v) or math.isinf(v):
        return lo
    return max(lo, min(hi, float(v)))


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalized_entropy(counts: Counter) -> float:
    """Shannon entropy normalised to [0, 1] over observed categories."""
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0
    probs = [c / total for c in counts.values()]
    h = -sum(p * math.log(p) for p in probs if p > 0)
    h_max = math.log(len(counts))
    if h_max <= 0:
        return 0.0
    return _clamp(h / h_max)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LightingConsistencyResult:
    scope: str
    frame_count: int
    coherence_overall: float
    axis_coherence: dict
    axis_weights_used: dict
    axes_available: tuple
    axes_missing: tuple
    dominant_profile: dict
    outliers: tuple
    coverage_gaps: dict
    suggested_extraction: tuple
    metric_version: int = METRIC_VERSION
    calculated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "frame_count": self.frame_count,
            "coherence_overall": round(self.coherence_overall, 6),
            "axis_coherence": {k: round(v, 6) for k, v in
                               sorted(self.axis_coherence.items())},
            "axis_weights_used": {k: round(v, 6) for k, v in
                                  sorted(self.axis_weights_used.items())},
            "axes_available": list(self.axes_available),
            "axes_missing": list(self.axes_missing),
            "dominant_profile": dict(sorted(self.dominant_profile.items())),
            "outliers": [list(o) for o in self.outliers],
            "coverage_gaps": dict(sorted(self.coverage_gaps.items())),
            "suggested_extraction": list(self.suggested_extraction),
            "metric_version": self.metric_version,
            "calculated_at": self.calculated_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Signal extraction from manifest rows
# ---------------------------------------------------------------------------

def _extract_signals(signals_json_raw, fq_row) -> dict:
    """Pull the four lighting axes from stored signals.

    Returns a dict with keys ``direction``, ``temperature``,
    ``key_to_fill``, ``exposure`` — each either a value or None.
    """
    sig: dict = {}
    if isinstance(signals_json_raw, str):
        try:
            sig = json.loads(signals_json_raw)
        except (json.JSONDecodeError, TypeError):
            sig = {}
    elif isinstance(signals_json_raw, dict):
        sig = signals_json_raw

    result: dict[str, object] = {
        "direction": None,
        "temperature": None,
        "key_to_fill": None,
        "exposure": None,
    }

    # --- direction ---
    ld = sig.get("lighting_direction")
    if isinstance(ld, dict):
        lt = ld.get("lighting_type")
        if isinstance(lt, str) and lt.strip().lower() in VALID_DIRECTIONS:
            result["direction"] = lt.strip().lower()
    elif isinstance(ld, str) and ld.strip().lower() in VALID_DIRECTIONS:
        result["direction"] = ld.strip().lower()

    # --- temperature (from color_mood.color_warmth) ---
    cm = sig.get("color_mood")
    if isinstance(cm, dict):
        warmth = _safe_float(cm.get("color_warmth"))
        if warmth is not None:
            result["temperature"] = _clamp(warmth)
    else:
        t = _safe_float(sig.get("color_temperature"))
        if t is not None:
            result["temperature"] = _clamp(t)

    # --- key_to_fill (from lighting shadow_strength or contrast proxy) ---
    if isinstance(ld, dict):
        ss = _safe_float(ld.get("shadow_strength"))
        if ss is not None:
            result["key_to_fill"] = _clamp(ss)
    ktf = _safe_float(sig.get("key_to_fill"))
    if ktf is not None:
        result["key_to_fill"] = _clamp(ktf)

    # --- exposure ---
    eo = sig.get("exposure_outlier")
    if isinstance(eo, dict):
        el = _safe_float(eo.get("exposure_level"))
        if el is not None:
            result["exposure"] = _clamp(el)
    exp = _safe_float(sig.get("exposure"))
    if exp is not None:
        result["exposure"] = _clamp(exp)
    # Fallback: frame_quality brightness normalised to 0..1
    if result["exposure"] is None and fq_row is not None:
        br = _safe_float(fq_row[0] if isinstance(fq_row, (tuple, list)) else None)
        if br is not None:
            result["exposure"] = _clamp(br / 255.0 if br > 1.0 else br)

    return result


# ---------------------------------------------------------------------------
# Core analysis (pure function on extracted per-frame dicts)
# ---------------------------------------------------------------------------

def _analyze(
    frame_signals: list[tuple[str, dict]],
    *,
    scope: str,
    min_frame_count: int,
    ts_override: str | None = None,
) -> LightingConsistencyResult:
    """Pure analysis over pre-extracted per-frame signal dicts."""

    n = len(frame_signals)
    ts = ts_override or _now_iso()

    # --- Insufficient data guard ---
    if n < min_frame_count:
        return LightingConsistencyResult(
            scope=scope, frame_count=n, coherence_overall=0.0,
            axis_coherence={}, axis_weights_used={},
            axes_available=(), axes_missing=tuple(sorted(_AXIS_WEIGHTS)),
            dominant_profile={}, outliers=(), coverage_gaps={},
            suggested_extraction=("insufficient frames for analysis",),
            metric_version=METRIC_VERSION, calculated_at=ts,
        )

    # --- Collect per-axis values ---
    axis_values: dict[str, list] = {ax: [] for ax in _AXIS_WEIGHTS}
    for fid, sig in frame_signals:
        for ax in _AXIS_WEIGHTS:
            v = sig.get(ax)
            if v is not None:
                axis_values[ax].append((fid, v))

    axes_available = tuple(sorted(ax for ax in _AXIS_WEIGHTS
                                  if len(axis_values[ax]) > 0))
    axes_missing = tuple(sorted(set(_AXIS_WEIGHTS) - set(axes_available)))

    # --- Per-axis coherence ---
    axis_coherence: dict[str, float] = {}
    dominant_profile: dict[str, object] = {}
    coverage_gaps: dict[str, list] = {}
    outlier_candidates: list[tuple[str, str, float]] = []

    for ax in axes_available:
        vals = axis_values[ax]
        if ax == "direction":
            # Categorical
            cat_counts: Counter = Counter(v for _, v in vals)
            total = sum(cat_counts.values())
            ent_norm = _normalized_entropy(cat_counts)
            axis_coherence[ax] = round(1.0 - ent_norm, 6)
            dom = cat_counts.most_common(1)[0][0] if cat_counts else "unknown"
            dominant_profile[ax] = dom
            # Outliers: rare categories
            threshold_count = max(1, int(total * OUTLIER_CATEGORY_FRAC))
            rare_cats = {cat for cat, cnt in cat_counts.items()
                         if cnt < threshold_count}
            for fid, v in vals:
                if v in rare_cats:
                    severity = 1.0 - (cat_counts[v] / total)
                    outlier_candidates.append((fid, ax, round(severity, 6)))
            # Coverage gaps: missing common directions
            present_cats = set(cat_counts.keys())
            expected = {"frontal", "left", "right", "top"}
            missing_dirs = expected - present_cats
            if missing_dirs:
                coverage_gaps[ax] = sorted(missing_dirs)
        else:
            # Scalar
            arr = np.array([v for _, v in vals], dtype=np.float64)
            std = float(np.std(arr))
            rng = _SCALAR_RANGES.get(ax, 0.35)
            coh = _clamp(1.0 - std / rng) if rng > 0 else 1.0
            axis_coherence[ax] = round(coh, 6)
            mean_val = float(np.mean(arr))
            dominant_profile[ax] = round(mean_val, 4)
            # Outliers: > N std from mean
            if std > 1e-9:
                for fid, v in vals:
                    z = abs(v - mean_val) / std
                    if z > OUTLIER_STD_THRESHOLD:
                        severity = _clamp(z / (OUTLIER_STD_THRESHOLD * 2))
                        outlier_candidates.append(
                            (fid, ax, round(severity, 6)))
            # Coverage gaps for scalars
            if ax == "temperature":
                if mean_val < 0.3:
                    coverage_gaps[ax] = ["warm-lit frames"]
                elif mean_val > 0.7:
                    coverage_gaps[ax] = ["cool-lit frames"]
            elif ax == "key_to_fill":
                if mean_val < 0.25:
                    coverage_gaps[ax] = ["high key-to-fill (dramatic)"]
                elif mean_val > 0.75:
                    coverage_gaps[ax] = ["low key-to-fill (flat fill)"]
            elif ax == "exposure":
                if mean_val < 0.3:
                    coverage_gaps[ax] = ["brighter-exposed frames"]
                elif mean_val > 0.7:
                    coverage_gaps[ax] = ["darker-exposed frames"]

    # --- Renormalize weights ---
    used_weight_sum = sum(_AXIS_WEIGHTS[ax] for ax in axes_available)
    axis_weights_used: dict[str, float] = {}
    if used_weight_sum > 0:
        for ax in axes_available:
            axis_weights_used[ax] = round(
                _AXIS_WEIGHTS[ax] / used_weight_sum, 6)

    # --- Overall coherence ---
    if axis_weights_used:
        coherence_overall = _clamp(sum(
            axis_coherence[ax] * axis_weights_used[ax]
            for ax in axes_available))
    else:
        coherence_overall = 0.0

    # --- Deduplicate & sort outliers, cap at OUTLIER_CAP ---
    # Sort by severity desc, then frame_id asc, then axis asc
    outlier_candidates.sort(key=lambda t: (-t[2], t[0], t[1]))
    seen: set[str] = set()
    outliers_deduped: list[tuple[str, str, float]] = []
    for fid, ax, sev in outlier_candidates:
        key = f"{fid}|{ax}"
        if key not in seen:
            seen.add(key)
            outliers_deduped.append((fid, ax, sev))
        if len(outliers_deduped) >= OUTLIER_CAP:
            break
    outliers_tuple = tuple(tuple(o) for o in outliers_deduped)

    # --- Suggested extraction ---
    suggestions: list[str] = []
    for ax, gaps in sorted(coverage_gaps.items()):
        for g in gaps:
            suggestions.append(g)
    if not suggestions and coherence_overall < 0.6:
        suggestions.append("consider adding varied lighting conditions")
    suggested_extraction = tuple(sorted(set(suggestions)))

    return LightingConsistencyResult(
        scope=scope,
        frame_count=n,
        coherence_overall=round(coherence_overall, 6),
        axis_coherence=axis_coherence,
        axis_weights_used=axis_weights_used,
        axes_available=axes_available,
        axes_missing=axes_missing,
        dominant_profile=dominant_profile,
        outliers=outliers_tuple,
        coverage_gaps=coverage_gaps,
        suggested_extraction=suggested_extraction,
        metric_version=METRIC_VERSION,
        calculated_at=ts,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_lighting_consistency(
    manifest,
    *,
    dataset_version: int | str | None = None,
    frame_ids: Iterable[str] | None = None,
    min_frame_count: int = DEFAULT_MIN_FRAME_COUNT,
    progress: bool = True,
) -> LightingConsistencyResult:
    """Analyze lighting coherence over the eligible frame set.

    If *dataset_version* is given, restrict to that dataset's frames.
    If *frame_ids* is given, use exactly those (highest precedence).
    If fewer than *min_frame_count* frames are in scope, return a result
    with ``coherence_overall=0.0`` and a note in ``axes_missing``.
    """
    import sqlite3

    # Determine scope label
    if frame_ids is not None:
        scope = "explicit"
    elif dataset_version is not None:
        scope = f"dataset:v{dataset_version}"
    else:
        scope = "all"

    # Build frame ID list
    explicit_ids: set[str] | None = None
    if frame_ids is not None:
        explicit_ids = set(frame_ids)

    try:
        if explicit_ids is not None:
            ids = sorted(explicit_ids)
        elif dataset_version is not None:
            rows = manifest.execute(
                "SELECT frame_id FROM dataset_frames WHERE version = ?",
                (int(dataset_version),),
            ).fetchall()
            ids = sorted(r[0] for r in rows)
        else:
            rows = manifest.execute(
                "SELECT frame_id FROM frames "
                "WHERE status IN ('scored', 'accepted', 'selected', "
                "'packaged', 'new')"
            ).fetchall()
            ids = sorted(r[0] for r in rows)
    except sqlite3.OperationalError:
        ids = []

    if not ids:
        return _analyze([], scope=scope, min_frame_count=min_frame_count)

    # Fetch signals in bulk
    frame_signals: list[tuple[str, dict]] = []
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        placeholders = ",".join("?" for _ in batch)
        try:
            faq_rows = manifest.execute(
                f"SELECT frame_id, signals_json FROM frame_advanced_quality "
                f"WHERE frame_id IN ({placeholders})", batch
            ).fetchall()
        except sqlite3.OperationalError:
            faq_rows = []

        faq_map = {r[0]: r[1] for r in faq_rows}

        # Also fetch brightness fallback from frame_quality
        try:
            fq_rows = manifest.execute(
                f"SELECT frame_id, brightness FROM frame_quality "
                f"WHERE frame_id IN ({placeholders})", batch
            ).fetchall()
        except sqlite3.OperationalError:
            try:
                fq_rows = manifest.execute(
                    f"SELECT frame_id, brightness FROM frames "
                    f"WHERE frame_id IN ({placeholders})", batch
                ).fetchall()
            except sqlite3.OperationalError:
                fq_rows = []
        fq_map = {r[0]: (r[1],) for r in fq_rows}

        for fid in batch:
            sig_raw = faq_map.get(fid)
            fq = fq_map.get(fid)
            extracted = _extract_signals(sig_raw, fq)
            # Only include if at least one axis has data
            if any(v is not None for v in extracted.values()):
                frame_signals.append((fid, extracted))

        if progress:
            logger.info("lighting_consistency: fetched %d/%d frames",
                        min(i + batch_size, len(ids)), len(ids))

    return _analyze(frame_signals, scope=scope,
                    min_frame_count=min_frame_count)


# ---------------------------------------------------------------------------
# Human-readable summary renderer
# ---------------------------------------------------------------------------

def render_lighting_report(result: LightingConsistencyResult) -> str:
    """Render a human-readable summary of the lighting consistency report."""
    lines: list[str] = []
    lines.append("Lighting Consistency Report")
    lines.append(f"  scope       : {result.scope}")
    lines.append(f"  frames      : {result.frame_count}")
    lines.append(f"  coherence   : {result.coherence_overall:.4f}")
    lines.append(f"  version     : {result.metric_version}")
    lines.append(f"  calculated  : {result.calculated_at}")
    lines.append("")

    if result.axes_available:
        lines.append("Axis coherence:")
        for ax in result.axes_available:
            coh = result.axis_coherence.get(ax, 0.0)
            w = result.axis_weights_used.get(ax, 0.0)
            dom = result.dominant_profile.get(ax, "-")
            lines.append(f"  {ax:<14} {coh:.4f}  (weight {w:.3f}, "
                         f"dominant: {dom})")
    if result.axes_missing:
        lines.append(f"  missing axes: {', '.join(result.axes_missing)}")
    lines.append("")

    if result.outliers:
        lines.append(f"Outliers: {len(result.outliers)} frame(s)")
        for fid, ax, sev in result.outliers[:10]:
            lines.append(f"  {fid}  axis={ax}  severity={sev:.4f}")
        if len(result.outliers) > 10:
            lines.append(f"  ... +{len(result.outliers) - 10} more")
    else:
        lines.append("Outliers: none")
    lines.append("")

    if result.suggested_extraction:
        lines.append("Suggested extraction:")
        for s in result.suggested_extraction:
            lines.append(f"  - {s}")
    else:
        lines.append("Suggested extraction: (none — lighting is well-balanced)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== lighting_consistency self-test ===\n")

    # Case 1: Empty frame set
    r0 = _analyze([], scope="test_empty", min_frame_count=5)
    assert r0.coherence_overall == 0.0
    assert r0.frame_count == 0
    print(f"1. Empty set: coherence={r0.coherence_overall:.4f} OK")

    # Case 2: Single frame (trivially coherent)
    single = [("f001", {
        "direction": "frontal",
        "temperature": 0.5,
        "key_to_fill": 0.3,
        "exposure": 0.6,
    })]
    r1 = _analyze(single, scope="test_single", min_frame_count=1)
    assert r1.coherence_overall == 1.0, f"expected 1.0, got {r1.coherence_overall}"
    print(f"2. Single frame: coherence={r1.coherence_overall:.4f} OK")

    # Case 3: Coherent set (all same direction, similar scalars)
    coherent_frames = [
        (f"f{i:04d}", {
            "direction": "frontal",
            "temperature": 0.50 + (i % 3) * 0.01,
            "key_to_fill": 0.30 + (i % 3) * 0.01,
            "exposure": 0.55 + (i % 3) * 0.01,
        })
        for i in range(30)
    ]
    r2 = _analyze(coherent_frames, scope="test_coherent", min_frame_count=5)
    assert r2.coherence_overall > 0.8, \
        f"coherent set should score >0.8, got {r2.coherence_overall}"
    print(f"3. Coherent set: coherence={r2.coherence_overall:.4f} OK")

    # Case 4: Scattered set (random directions, wide scalars)
    import random
    rng = random.Random(42)
    dirs = ["frontal", "left", "right", "top", "back", "rim"]
    scattered_frames = [
        (f"f{i:04d}", {
            "direction": rng.choice(dirs),
            "temperature": rng.uniform(0.0, 1.0),
            "key_to_fill": rng.uniform(0.0, 1.0),
            "exposure": rng.uniform(0.0, 1.0),
        })
        for i in range(50)
    ]
    r3 = _analyze(scattered_frames, scope="test_scattered", min_frame_count=5)
    assert r3.coherence_overall < 0.6, \
        f"scattered set should score <0.6, got {r3.coherence_overall}"
    print(f"4. Scattered set: coherence={r3.coherence_overall:.4f} OK")

    # Case 5: All frames missing direction
    no_dir_frames = [
        (f"f{i:04d}", {
            "temperature": 0.5,
            "key_to_fill": 0.3,
            "exposure": 0.6,
        })
        for i in range(25)
    ]
    r4 = _analyze(no_dir_frames, scope="test_no_dir", min_frame_count=5)
    assert "direction" in r4.axes_missing
    assert "direction" not in r4.axis_coherence
    assert abs(sum(r4.axis_weights_used.values()) - 1.0) < 1e-6
    print(f"5. Missing direction: axes_missing={r4.axes_missing}, "
          f"renorm weights sum={sum(r4.axis_weights_used.values()):.4f} OK")

    # Case 6: Determinism
    r5a = _analyze(coherent_frames, scope="det_test", min_frame_count=5,
                   ts_override="2025-01-01T00:00:00Z")
    r5b = _analyze(coherent_frames, scope="det_test", min_frame_count=5,
                   ts_override="2025-01-01T00:00:00Z")
    assert r5a.to_json() == r5b.to_json(), "determinism violated"
    print("6. Determinism: byte-identical JSON OK")

    # Case 7: Outlier cap
    many_outliers = [
        (f"f{i:04d}", {
            "direction": "frontal" if i < 48 else "rim",
            "temperature": 0.5 if i < 48 else 0.05,
            "key_to_fill": 0.3,
            "exposure": 0.5,
        })
        for i in range(50)
    ]
    r6 = _analyze(many_outliers, scope="test_outliers", min_frame_count=5)
    assert len(r6.outliers) <= OUTLIER_CAP
    print(f"7. Outlier cap: {len(r6.outliers)} outliers (<= {OUTLIER_CAP}) OK")

    # Case 8: JSON serialization
    j = r2.to_json()
    parsed = json.loads(j)
    assert parsed["scope"] == "test_coherent"
    print(f"8. JSON round-trip: OK ({len(j)} bytes)")

    # Case 9: Render
    rendered = render_lighting_report(r2)
    assert "coherence" in rendered.lower()
    print(f"9. Render: OK ({len(rendered)} chars)")

    print("\n=== all self-tests passed ===")
