"""Continuous identity confidence scoring for curated frames.

Smart curation produces a binary identity decision (accept / reject).
This module derives a continuous 0..1 ``identity_confidence`` score per
frame by combining already-stored detection and quality signals.  The
score lets downstream ``build`` recipes pack the identity layer with the
*strongest* frames rather than merely the passing ones, reinforcing the
project's identity-first invariant.

Inputs are read from the ``detections`` and ``frame_quality`` manifest
tables; no new detection work or media access is performed.  Missing
components are dropped and remaining weights renormalized so the score
always computes.  A companion ``confidence_of_confidence`` value tells
callers how much signal was actually available.

Design contract:
  * additive & opt-in - nothing runs unless explicitly invoked
  * non-destructive - reads manifest rows only, never touches media
  * dependency-light - stdlib + numpy only at top level
  * deterministic - sorted iteration, stable output, no randomness
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric versioning
# ---------------------------------------------------------------------------
METRIC_VERSION: int = 1
"""Bump when the formula, weights, or component set changes."""

# ---------------------------------------------------------------------------
# Default component weights (must sum to 1.0)
# ---------------------------------------------------------------------------
W_SIM: float = 0.45      # ArcFace similarity - the base truth
W_AREA: float = 0.20     # Face coverage in frame
W_FRONTAL: float = 0.15 # Frontalness (pose quality)
W_SHARP: float = 0.12   # Face-region sharpness
W_EYES: float = 0.08    # Eye openness (soft penalty)

_WEIGHT_NAMES = ("sim", "area", "frontal", "sharp", "eyes")
_WEIGHTS = {
    "sim": W_SIM,
    "area": W_AREA,
    "frontal": W_FRONTAL,
    "sharp": W_SHARP,
    "eyes": W_EYES,
}

# Sanity check at import time - catch drift immediately.
assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9, (
    f"identity_confidence weights must sum to 1.0, got {sum(_WEIGHTS.values())}"
)

# ---------------------------------------------------------------------------
# Smoothstep / area thresholds
# ---------------------------------------------------------------------------
AREA_FLOOR: float = 0.02   # Below ~2% face coverage the signal is weak
AREA_CEIL: float = 0.18    # ~18% coverage is "full and confident"

# ---------------------------------------------------------------------------
# Penalty constants
# ---------------------------------------------------------------------------
EYES_CLOSED_PENALTY_FACTOR: float = 0.35
"""Closed eyes reduce the eyes component by at most this fraction."""

MULTI_FACE_DECAY: float = 0.35
"""Per-extra-face decay factor in the multi-face penalty denominator."""

INTEGRITY_GATE_THRESHOLD: float = 0.6
"""Below this integrity value the gate ramps down linearly."""

TOTAL_COMPONENT_COUNT: int = 5
"""Number of weighted components (for confidence_of_confidence)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if math.isnan(v) or math.isinf(v):
        return lo
    return max(lo, min(hi, float(v)))


def _smoothstep(x: float, edge0: float, edge1: float) -> float:
    """Hermite smoothstep mapping *x* from [edge0, edge1] to [0, 1]."""
    if edge1 <= edge0:
        return 0.0
    t = _clamp((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def _safe_float(v) -> Optional[float]:
    """Coerce to float or None; NaN/Inf become None."""
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


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IdentityConfidenceResult:
    """Immutable per-frame identity confidence result."""

    frame_id: str
    identity_confidence: float
    confidence_of_confidence: float
    components: dict          # named component values, each 0..1
    components_available: tuple  # sorted names that contributed
    components_missing: tuple    # sorted names that were dropped
    penalties: dict           # named multiplicative penalties, each 0..1
    metric_version: int = METRIC_VERSION
    calculated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "identity_confidence": round(self.identity_confidence, 6),
            "confidence_of_confidence": round(self.confidence_of_confidence, 6),
            "components": {k: round(v, 6) for k, v in
                           sorted(self.components.items())},
            "components_available": list(self.components_available),
            "components_missing": list(self.components_missing),
            "penalties": {k: round(v, 6) for k, v in
                          sorted(self.penalties.items())},
            "metric_version": self.metric_version,
            "calculated_at": self.calculated_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Core scoring function (pure, no side effects)
# ---------------------------------------------------------------------------

def compute_identity_confidence(
    *,
    frame_id: str,
    identity_sim: Optional[float] = None,
    face_area: Optional[float] = None,
    face_count: Optional[int] = None,
    frontalness: Optional[float] = None,
    face_sharpness: Optional[float] = None,
    eyes_open: Optional[float] = None,
    occlusion: Optional[float] = None,
    integrity: Optional[float] = None,
) -> IdentityConfidenceResult:
    """Compute a continuous identity confidence score from stored signals.

    All inputs are optional.  Missing weighted components are dropped and
    the remaining weights renormalized.  Multiplicative penalties
    (occlusion, multi-face, integrity) always apply with safe defaults
    when their inputs are absent.
    """
    components: dict[str, float] = {}
    available_weights: dict[str, float] = {}

    # 1. sim_component (required for a meaningful score, but we still
    #    compute without it - the result will just be very low-confidence)
    sim_val = _safe_float(identity_sim)
    if sim_val is not None:
        components["sim"] = _clamp(sim_val)
        available_weights["sim"] = _WEIGHTS["sim"]

    # 2. area_component
    area_val = _safe_float(face_area)
    if area_val is not None:
        components["area"] = _smoothstep(area_val, AREA_FLOOR, AREA_CEIL)
        available_weights["area"] = _WEIGHTS["area"]

    # 3. frontal_component
    frontal_val = _safe_float(frontalness)
    if frontal_val is not None:
        components["frontal"] = _clamp(frontal_val)
        available_weights["frontal"] = _WEIGHTS["frontal"]

    # 4. sharp_component
    sharp_val = _safe_float(face_sharpness)
    if sharp_val is not None:
        components["sharp"] = _clamp(sharp_val)
        available_weights["sharp"] = _WEIGHTS["sharp"]

    # 5. eyes_component
    eyes_val = _safe_float(eyes_open)
    if eyes_val is not None:
        eyes_clamped = _clamp(eyes_val)
        components["eyes"] = 1.0 - (1.0 - eyes_clamped) * EYES_CLOSED_PENALTY_FACTOR
        available_weights["eyes"] = _WEIGHTS["eyes"]

    # --- Weighted base (renormalized) ---
    total_weight = sum(available_weights.values())
    if total_weight > 0 and components:
        base = sum(components[k] * available_weights[k] for k in components) / total_weight
    else:
        base = 0.0

    # --- Multiplicative penalties ---
    # Occlusion penalty
    occ_val = _safe_float(occlusion)
    occlusion_penalty = 1.0 - _clamp(occ_val) if occ_val is not None else 1.0

    # Multi-face penalty
    fc = face_count if isinstance(face_count, int) else None
    if fc is not None and fc > 1:
        multi_face_penalty = 1.0 / (1.0 + MULTI_FACE_DECAY * (fc - 1))
    else:
        multi_face_penalty = 1.0

    # Integrity gate
    int_val = _safe_float(integrity)
    if int_val is not None and int_val < INTEGRITY_GATE_THRESHOLD:
        integrity_gate = _clamp(int_val)
    else:
        integrity_gate = 1.0

    confidence = _clamp(base * occlusion_penalty * multi_face_penalty * integrity_gate)

    # Confidence-of-confidence: fraction of weighted components available
    n_available = len(available_weights)
    cof = _clamp(n_available / TOTAL_COMPONENT_COUNT)

    # Sorted tuples for determinism
    comp_available = tuple(sorted(components.keys()))
    all_names = set(_WEIGHT_NAMES)
    comp_missing = tuple(sorted(all_names - set(components.keys())))

    penalties = {
        "occlusion": round(occlusion_penalty, 6),
        "multi_face": round(multi_face_penalty, 6),
        "integrity_gate": round(integrity_gate, 6),
    }

    return IdentityConfidenceResult(
        frame_id=frame_id,
        identity_confidence=round(confidence, 6),
        confidence_of_confidence=round(cof, 6),
        components={k: round(v, 6) for k, v in sorted(components.items())},
        components_available=comp_available,
        components_missing=comp_missing,
        penalties=penalties,
        metric_version=METRIC_VERSION,
        calculated_at=_now_iso(),
    )


# ---------------------------------------------------------------------------
# Batch entry points
# ---------------------------------------------------------------------------

def run_identity_confidence(
    manifest,
    *,
    frame_ids: Iterable[str] | None = None,
    batch_size: int = 256,
    progress: bool = True,
) -> dict:
    """Compute identity_confidence for frames.

    If *frame_ids* is ``None``, processes every frame with a detections row
    that has a non-null ``identity_sim``.  Skips frames already computed at
    the current ``METRIC_VERSION`` unless *frame_ids* is explicit.
    Batch-commits every *batch_size* rows so Ctrl-C is resumable.

    Returns a summary dict with keys: processed, skipped, failed,
    elapsed_seconds.
    """
    import sqlite3

    t0 = time.monotonic()
    processed = 0
    skipped = 0
    failed = 0

    # Determine eligible frame IDs
    if frame_ids is not None:
        ids = sorted(set(frame_ids))
        explicit = True
    else:
        explicit = False
        try:
            rows = manifest.execute(
                "SELECT frame_id FROM detections WHERE identity_sim IS NOT NULL"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        ids = sorted(r[0] for r in rows)

    if not ids:
        return {"processed": 0, "skipped": 0, "failed": 0,
                "elapsed_seconds": round(time.monotonic() - t0, 3)}

    # Skip already-computed at current version (unless explicit)
    if not explicit:
        try:
            done_rows = manifest.execute(
                "SELECT frame_id FROM frame_identity_confidence "
                "WHERE metric_version = ?",
                (METRIC_VERSION,),
            ).fetchall()
            done = {r[0] for r in done_rows}
        except sqlite3.OperationalError:
            done = set()
        ids = [fid for fid in ids if fid not in done]
        skipped = len(done)

    if not ids:
        return {"processed": 0, "skipped": skipped, "failed": 0,
                "elapsed_seconds": round(time.monotonic() - t0, 3)}

    # Process in batches
    buf: list[tuple] = []
    for i, fid in enumerate(ids):
        try:
            det = manifest.execute(
                "SELECT identity_sim, face_area, face_count, framing, bbox "
                "FROM detections WHERE frame_id = ?", (fid,)
            ).fetchone()
            if det is None:
                failed += 1
                continue

            fq = None
            try:
                fq = manifest.execute(
                    "SELECT face_sharpness, frontalness, eyes_open, "
                    "occlusion, integrity "
                    "FROM frame_quality WHERE frame_id = ?", (fid,)
                ).fetchone()
            except sqlite3.OperationalError:
                pass  # table may not exist yet

            result = compute_identity_confidence(
                frame_id=fid,
                identity_sim=det[0],
                face_area=det[1],
                face_count=det[2],
                frontalness=fq[1] if fq else None,
                face_sharpness=fq[0] if fq else None,
                eyes_open=fq[2] if fq else None,
                occlusion=fq[3] if fq else None,
                integrity=fq[4] if fq else None,
            )
            buf.append((
                fid,
                result.identity_confidence,
                result.confidence_of_confidence,
                json.dumps(result.components, ensure_ascii=False),
                json.dumps(result.penalties, ensure_ascii=False),
                result.metric_version,
                result.calculated_at,
            ))
            processed += 1
        except Exception:
            logger.debug("identity_confidence failed for %s", fid, exc_info=True)
            failed += 1
            continue

        # Batch commit
        if len(buf) >= batch_size:
            _flush(manifest, buf)
            buf.clear()
            if progress:
                logger.info("identity_confidence: %d/%d processed",
                            processed, len(ids))

    if buf:
        _flush(manifest, buf)
        buf.clear()

    elapsed = round(time.monotonic() - t0, 3)
    return {"processed": processed, "skipped": skipped, "failed": failed,
            "elapsed_seconds": elapsed}


def _flush(conn, buf: list[tuple]) -> None:
    import sqlite3
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO frame_identity_confidence("
            "frame_id, identity_confidence, confidence_of_confidence, "
            "components_json, penalties_json, metric_version, calculated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            buf,
        )
        conn.commit()
    except sqlite3.OperationalError:
        logger.warning("identity_confidence flush failed; table may not "
                       "exist yet (run migrations first)", exc_info=True)


def select_top_by_identity(
    manifest,
    *,
    limit: int,
    min_confidence: float = 0.0,
    frame_ids: Iterable[str] | None = None,
) -> list[tuple[str, float]]:
    """Return [(frame_id, identity_confidence)] sorted desc, up to *limit*.

    Deterministic tiebreak by frame_id ascending.  Useful for build
    recipes that want to pack the identity layer with the strongest frames.
    """
    import sqlite3

    params: list = [min_confidence]
    where = "WHERE identity_confidence >= ?"
    if frame_ids is not None:
        id_list = sorted(set(frame_ids))
        if not id_list:
            return []
        placeholders = ",".join("?" for _ in id_list)
        where += f" AND frame_id IN ({placeholders})"
        params.extend(id_list)

    params.append(limit)
    try:
        rows = manifest.execute(
            f"SELECT frame_id, identity_confidence "
            f"FROM frame_identity_confidence {where} "
            f"ORDER BY identity_confidence DESC, frame_id ASC "
            f"LIMIT ?",
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [(r[0], r[1]) for r in rows]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== identity_confidence self-test ===\n")

    # Case 1: Full signals, strong identity
    r1 = compute_identity_confidence(
        frame_id="test_strong",
        identity_sim=0.82,
        face_area=0.12,
        face_count=1,
        frontalness=0.9,
        face_sharpness=0.85,
        eyes_open=0.95,
        occlusion=0.05,
        integrity=0.9,
    )
    print(f"1. Strong identity:")
    print(f"   confidence={r1.identity_confidence:.4f}  "
          f"cof={r1.confidence_of_confidence:.4f}")
    print(f"   components={r1.components}")
    print(f"   penalties={r1.penalties}")
    print(f"   available={r1.components_available}")
    assert r1.identity_confidence > 0.7, "strong case should score high"
    assert r1.confidence_of_confidence == 1.0

    # Case 2: Missing face_sharpness AND frontalness -> cof = 0.6
    r2 = compute_identity_confidence(
        frame_id="test_partial",
        identity_sim=0.65,
        face_area=0.08,
        face_count=1,
        eyes_open=0.8,
        occlusion=0.1,
        integrity=0.7,
    )
    print(f"\n2. Partial (no sharp/frontal):")
    print(f"   confidence={r2.identity_confidence:.4f}  "
          f"cof={r2.confidence_of_confidence:.4f}")
    print(f"   missing={r2.components_missing}")
    assert abs(r2.confidence_of_confidence - 0.6) < 1e-6, \
        f"expected cof=0.6, got {r2.confidence_of_confidence}"

    # Case 3: Only identity_sim -> cof = 0.2
    r3 = compute_identity_confidence(
        frame_id="test_minimal",
        identity_sim=0.55,
    )
    print(f"\n3. Minimal (sim only):")
    print(f"   confidence={r3.identity_confidence:.4f}  "
          f"cof={r3.confidence_of_confidence:.4f}")
    assert abs(r3.confidence_of_confidence - 0.2) < 1e-6, \
        f"expected cof=0.2, got {r3.confidence_of_confidence}"

    # Case 4: Multiple faces penalty
    r4 = compute_identity_confidence(
        frame_id="test_multi",
        identity_sim=0.75,
        face_area=0.10,
        face_count=3,
        frontalness=0.8,
        face_sharpness=0.7,
        eyes_open=0.9,
        occlusion=0.0,
        integrity=0.85,
    )
    print(f"\n4. Multi-face (3 faces):")
    print(f"   confidence={r4.identity_confidence:.4f}  "
          f"multi_face_penalty={r4.penalties['multi_face']:.4f}")
    assert r4.penalties["multi_face"] < 1.0

    # Case 5: Low integrity gate
    r5 = compute_identity_confidence(
        frame_id="test_low_integrity",
        identity_sim=0.80,
        face_area=0.10,
        face_count=1,
        frontalness=0.85,
        face_sharpness=0.80,
        eyes_open=0.90,
        occlusion=0.0,
        integrity=0.3,
    )
    print(f"\n5. Low integrity (0.3):")
    print(f"   confidence={r5.identity_confidence:.4f}  "
          f"integrity_gate={r5.penalties['integrity_gate']:.4f}")
    assert r5.penalties["integrity_gate"] < 0.6

    # Case 6: JSON serialization
    j = r1.to_json()
    parsed = json.loads(j)
    assert parsed["frame_id"] == "test_strong"
    print(f"\n6. JSON round-trip: OK ({len(j)} bytes)")

    print("\n=== all self-tests passed ===")
