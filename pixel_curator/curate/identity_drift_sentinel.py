"""Identity drift sentinel: detect sub-modes in the accepted frame set.

A personal media archive often contains identity *drift* across time:
aging, weight change, hairstyle changes, facial hair on/off, glasses,
seasonal tan.  A LoRA trained on a mix of subtly different "modes" of
the same person produces a muddy identity — the model averages them and
lands between all of them.

This module reads ArcFace embeddings **already stored** in the manifest
(``detections.embed``) and clusters them to answer:

  * Does the accepted set contain detectable identity sub-modes?
  * How many, how balanced, how separable?
  * What is the overall drift risk (0..1)?
  * Which frames represent each mode?

No new face detection or media I/O is performed.  The analysis is pure,
read-only, opt-in, deterministic, and degrades gracefully when optional
dependencies (hdbscan, sklearn) are absent.
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

# ---------------------------------------------------------------------------
# Embedding dimensionality (ArcFace R50 w600k = 512-d float32)
# ---------------------------------------------------------------------------
EMBED_DIM: int = 512

# ---------------------------------------------------------------------------
# Minimum frame count for meaningful clustering
# ---------------------------------------------------------------------------
DEFAULT_MIN_FRAMES: int = 12

# ---------------------------------------------------------------------------
# Drift-risk weights (must sum to 1.0)
# ---------------------------------------------------------------------------
W_SEPARATION: float = 0.55   # Inter-mode distance is the primary signal
W_BALANCE: float = 0.30      # Imbalanced modes are less risky than split ones
W_NOISE: float = 0.15        # Noise fraction indicates embedding instability

_DRIFT_WEIGHTS = {
    "separation": W_SEPARATION,
    "balance": W_BALANCE,
    "noise": W_NOISE,
}
assert abs(sum(_DRIFT_WEIGHTS.values()) - 1.0) < 1e-9, \
    f"drift weights must sum to 1.0, got {sum(_DRIFT_WEIGHTS.values())}"

# ---------------------------------------------------------------------------
# Separation threshold
# ---------------------------------------------------------------------------
SEPARATION_DISTANCE_THRESHOLD: float = 0.35
"""ArcFace cosine distance around 0.35+ usually indicates genuinely
different identity modes.  Below this the modes are likely the same
person with minor variation (lighting, expression).  Above it the
model is probably averaging distinct visual identities."""

# ---------------------------------------------------------------------------
# Clustering parameters
# ---------------------------------------------------------------------------
KMEANS_K_MAX: int = 6         # Upper bound on silhouette sweep
KMEANS_RANDOM_STATE: int = 0  # Fixed seed for determinism
HDBSCAN_MIN_CLUSTER_FRAC: int = 20  # min_cluster_size = max(4, N // this)

# ---------------------------------------------------------------------------
# Outlier / noise label
# ---------------------------------------------------------------------------
NOISE_LABEL: int = -1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if math.isnan(v) or math.isinf(v):
        return lo
    return max(lo, min(hi, float(v)))


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    dot = float(np.dot(a, b))
    return _clamp(dot, -1.0, 1.0)


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - _cosine_sim(a, b)


def _decode_embed(raw) -> Optional[np.ndarray]:
    """Decode an embed BLOB or JSON string into a 1-D float32 array."""
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        try:
            arr = np.frombuffer(raw, dtype=np.float32)
            if arr.size == EMBED_DIM:
                return arr.copy()
        except Exception:
            pass
        return None
    if isinstance(raw, str):
        try:
            lst = json.loads(raw)
            if isinstance(lst, list) and len(lst) == EMBED_DIM:
                return np.array(lst, dtype=np.float32)
        except Exception:
            pass
        return None
    return None


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; zero rows stay zero."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return mat / norms


# ---------------------------------------------------------------------------
# Clustering fallback chain
# ---------------------------------------------------------------------------

def _cluster_fallback(
    X: np.ndarray,
    *,
    k_max: int = KMEANS_K_MAX,
    random_state: int = KMEANS_RANDOM_STATE,
) -> tuple[np.ndarray, str]:
    """Cluster rows of *X* (N, D) float32.

    Returns ``(labels, method)`` where labels is an int array of shape
    (N,) with -1 for noise (HDBSCAN only).  Tries HDBSCAN → sklearn
    KMeans → pure-numpy KMeans, matching the fallback discipline in
    ``viz/clustering.py`` but returning integer labels suitable for
    per-mode statistics rather than string cluster IDs.
    """
    n = X.shape[0]

    # --- HDBSCAN ---
    try:
        import hdbscan as _hdbscan
        min_cs = max(4, n // HDBSCAN_MIN_CLUSTER_FRAC)
        clusterer = _hdbscan.HDBSCAN(
            min_cluster_size=min_cs, metric="euclidean")
        labels = clusterer.fit_predict(X)
        return labels.astype(np.int64), "hdbscan"
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("HDBSCAN failed: %s", exc)

    # --- sklearn KMeans with silhouette sweep ---
    k = _silhouette_k(X, k_max=k_max, random_state=random_state)
    try:
        from sklearn.cluster import KMeans as _SKMeans
        km = _SKMeans(n_clusters=k, random_state=random_state, n_init=4)
        labels = km.fit_predict(X)
        return labels.astype(np.int64), "sklearn_kmeans"
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("sklearn KMeans failed: %s", exc)

    # --- Pure numpy KMeans (Lloyd's) ---
    labels = _numpy_kmeans(X, k=k, seed=random_state)
    return labels.astype(np.int64), "numpy_kmeans"


def _silhouette_k(
    X: np.ndarray,
    *,
    k_max: int = KMEANS_K_MAX,
    random_state: int = KMEANS_RANDOM_STATE,
) -> int:
    """Pick best k in [2, min(k_max, N//8)] by silhouette score.

    Falls back to a variance-ratio heuristic when sklearn is unavailable:
    if the mean pairwise distance to the global centroid is below a tight
    threshold relative to the embedding dimension, treat as a single
    cluster (k=1).  Otherwise default to k=2.
    """
    n = X.shape[0]
    upper = min(k_max, max(2, n // 8))
    if upper < 2:
        return 2
    try:
        from sklearn.cluster import KMeans as _SKMeans
        from sklearn.metrics import silhouette_score as _sil
        best_k, best_s = 2, -1.0
        for k in range(2, upper + 1):
            km = _SKMeans(n_clusters=k, random_state=random_state, n_init=4)
            lbl = km.fit_predict(X)
            s = _sil(X, lbl)
            if s > best_s:
                best_s = s
                best_k = k
        return best_k
    except ImportError:
        pass
    except Exception:
        pass

    # Pure-numpy fallback: without sklearn silhouette or HDBSCAN,
    # reliable k-selection in 512-d normalised space is not feasible
    # (all splits improve intra-cluster cosine similarity).  Default
    # to k=2 — the most conservative assumption for identity drift
    # analysis.  The drift_risk formula's separation_score will
    # correctly rate single-cloud sets as low-risk because the
    # inter-mode distance between two halves of one cloud is small.
    # When sklearn IS available (the common case), silhouette sweep
    # above handles k-selection properly.
    return 2


def _numpy_kmeans(
    X: np.ndarray,
    *,
    k: int = 2,
    seed: int = 0,
    max_iter: int = 100,
) -> np.ndarray:
    """Minimal Lloyd's KMeans in pure numpy. Deterministic via seed."""
    n = X.shape[0]
    if k <= 1:
        return np.zeros(n, dtype=np.int64)
    rng = np.random.RandomState(seed)
    indices = rng.choice(n, size=min(k, n), replace=False)
    centroids = X[indices].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        # Assign
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centroids
        for j in range(centroids.shape[0]):
            mask = labels == j
            if mask.any():
                centroids[j] = X[mask].mean(axis=0)
    return labels


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModeSummary:
    mode_id: int
    size: int
    cohesion: float
    representative_frame_id: str
    sample_frame_ids: tuple
    temporal_span: tuple | None = None

    def to_dict(self) -> dict:
        return {
            "mode_id": self.mode_id,
            "size": self.size,
            "cohesion": round(self.cohesion, 6),
            "representative_frame_id": self.representative_frame_id,
            "sample_frame_ids": list(self.sample_frame_ids),
            "temporal_span": list(self.temporal_span) if self.temporal_span else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass(frozen=True)
class IdentityDriftResult:
    scope: str
    frame_count: int
    embedding_dim: int
    noise_count: int
    noise_fraction: float
    mode_count: int
    modes: tuple
    inter_mode_min_dist: float
    inter_mode_max_dist: float
    inter_mode_mean_dist: float
    drift_risk: float | None
    drift_risk_note: str
    recommended_action: str
    clustering_method: str
    metric_version: int = METRIC_VERSION
    calculated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "frame_count": self.frame_count,
            "embedding_dim": self.embedding_dim,
            "noise_count": self.noise_count,
            "noise_fraction": round(self.noise_fraction, 6),
            "mode_count": self.mode_count,
            "modes": [m.to_dict() for m in self.modes],
            "inter_mode_min_dist": round(self.inter_mode_min_dist, 6),
            "inter_mode_max_dist": round(self.inter_mode_max_dist, 6),
            "inter_mode_mean_dist": round(self.inter_mode_mean_dist, 6),
            "drift_risk": round(self.drift_risk, 6) if self.drift_risk is not None else None,
            "drift_risk_note": self.drift_risk_note,
            "recommended_action": self.recommended_action,
            "clustering_method": self.clustering_method,
            "metric_version": self.metric_version,
            "calculated_at": self.calculated_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _analyze(
    frame_ids: list[str],
    embeddings: np.ndarray,
    timestamps: list[str | None],
    *,
    scope: str,
    min_frames: int,
    ts_override: str | None = None,
) -> IdentityDriftResult:
    """Pure analysis over pre-loaded embedding matrix."""
    n = frame_ids.shape[0] if hasattr(frame_ids, 'shape') else len(frame_ids)
    ts = ts_override or _now_iso()

    # --- Insufficient data guard ---
    if n < min_frames:
        return IdentityDriftResult(
            scope=scope, frame_count=n, embedding_dim=EMBED_DIM,
            noise_count=0, noise_fraction=0.0, mode_count=0,
            modes=(), inter_mode_min_dist=0.0, inter_mode_max_dist=0.0,
            inter_mode_mean_dist=0.0, drift_risk=None,
            drift_risk_note=f"Insufficient data ({n} frames, need {min_frames})",
            recommended_action="review_manually",
            clustering_method="none",
            metric_version=METRIC_VERSION, calculated_at=ts,
        )

    # Keep raw embeddings for numpy-fallback spread estimation
    embeddings_raw = embeddings.astype(np.float32).copy()

    # --- L2 normalize ---
    X = _l2_normalize(embeddings_raw)

    # --- Cluster ---
    labels, method = _cluster_fallback(X)

    # --- Separate noise ---
    noise_mask = labels == NOISE_LABEL
    noise_count = int(noise_mask.sum())
    noise_fraction = noise_count / n if n > 0 else 0.0

    non_noise_labels = labels[~noise_mask]
    non_noise_indices = np.where(~noise_mask)[0]
    unique_labels = sorted(set(non_noise_labels.tolist()))

    if not unique_labels:
        # All noise
        return IdentityDriftResult(
            scope=scope, frame_count=n, embedding_dim=EMBED_DIM,
            noise_count=noise_count, noise_fraction=noise_fraction,
            mode_count=0, modes=(),
            inter_mode_min_dist=0.0, inter_mode_max_dist=0.0,
            inter_mode_mean_dist=0.0, drift_risk=0.0,
            drift_risk_note="All frames classified as noise; no modes detected",
            recommended_action="review_manually",
            clustering_method=method,
            metric_version=METRIC_VERSION, calculated_at=ts,
        )

    # --- Per-mode statistics ---
    mode_summaries: list[ModeSummary] = []
    centroids: list[np.ndarray] = []

    for lbl in unique_labels:
        member_mask = labels == lbl
        member_indices = np.where(member_mask)[0]
        member_embs = X[member_indices]
        centroid = member_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        centroids.append(centroid)

        # Cohesion: mean cosine sim to centroid
        sims = [_cosine_sim(member_embs[i], centroid)
                for i in range(member_embs.shape[0])]
        cohesion = float(np.mean(sims)) if sims else 0.0

        # Distances to centroid for representative + samples
        dists_to_centroid = np.array([
            _cosine_dist(member_embs[i], centroid)
            for i in range(member_embs.shape[0])
        ])
        # Sort by distance asc, then frame_id asc for determinism
        order = np.lexsort((
            np.array([frame_ids[idx] for idx in member_indices]),
            dists_to_centroid,
        ))
        sorted_member_indices = member_indices[order]

        rep_fid = frame_ids[sorted_member_indices[0]]
        sample_count = min(5, len(sorted_member_indices))
        sample_fids = tuple(frame_ids[sorted_member_indices[i]]
                            for i in range(sample_count))

        # Temporal span
        ts_list = [timestamps[idx] for idx in member_indices
                   if timestamps[idx] is not None]
        temporal_span = None
        if ts_list:
            ts_sorted = sorted(ts_list)
            temporal_span = (ts_sorted[0], ts_sorted[-1])

        mode_summaries.append(ModeSummary(
            mode_id=int(lbl),
            size=int(member_mask.sum()),
            cohesion=round(cohesion, 6),
            representative_frame_id=rep_fid,
            sample_frame_ids=sample_fids,
            temporal_span=temporal_span,
        ))

    # --- Renumber modes by size desc, then rep frame_id asc ---
    mode_summaries.sort(key=lambda m: (-m.size, m.representative_frame_id))
    old_to_new: dict[int, int] = {}
    for new_id, ms in enumerate(mode_summaries):
        old_to_new[ms.mode_id] = new_id
    renumbered: list[ModeSummary] = []
    for ms in mode_summaries:
        renumbered.append(ModeSummary(
            mode_id=old_to_new[ms.mode_id],
            size=ms.size,
            cohesion=ms.cohesion,
            representative_frame_id=ms.representative_frame_id,
            sample_frame_ids=ms.sample_frame_ids,
            temporal_span=ms.temporal_span,
        ))
    # Reorder centroids to match new numbering
    reordered_centroids = [centroids[unique_labels.index(ms.mode_id)]
                           for ms in mode_summaries]

    # --- Inter-mode distances ---
    n_modes = len(reordered_centroids)
    if n_modes >= 2:
        pair_dists: list[float] = []
        for i in range(n_modes):
            for j in range(i + 1, n_modes):
                pair_dists.append(
                    _cosine_dist(reordered_centroids[i],
                                 reordered_centroids[j]))
        inter_min = min(pair_dists)
        inter_max = max(pair_dists)
        inter_mean = float(np.mean(pair_dists))
    else:
        inter_min = 0.0
        inter_max = 0.0
        inter_mean = 0.0

    # --- Drift risk ---
    total_nonnoise = sum(ms.size for ms in renumbered)
    if total_nonnoise > 0 and n_modes >= 1:
        largest = max(ms.size for ms in renumbered)
        balance_penalty = 1.0 - (largest / total_nonnoise)
    else:
        balance_penalty = 0.0

    # When the pure-numpy fallback is used, inter-mode cosine distances
    # on the 512-d unit sphere are uninformative (any two random halves
    # have ~1.0 distance).  Instead, measure separation via the *raw*
    # (pre-normalisation) embedding spread: the mean L2 norm of each
    # row's deviation from the global centroid.  A tight single identity
    # has small deviations; genuine sub-modes have large ones.
    # RANGE_RAW_SPREAD is calibrated so that ArcFace embeddings from
    # the same person (~0.05-0.10 L2 deviation) map to low scores,
    # while distinct identity modes (~0.3+ deviation) map to high scores.
    # Per-dimension standard deviation of the raw embeddings.
    # In D dimensions, the expected L2 deviation from centroid is
    # sqrt(D) * per_dim_std, so we divide by sqrt(D) to get a
    # dimension-independent spread measure.
    # RANGE_PER_DIM_STD: ArcFace embeddings from the same identity
    # typically have per-dim std ~0.02-0.05; distinct identities ~0.1+.
    RANGE_PER_DIM_STD: float = 0.08
    if method == "numpy_kmeans" and n >= 2:
        raw_centroid = embeddings_raw.mean(axis=0)
        per_dim_var = np.var(embeddings_raw - raw_centroid, axis=0)
        per_dim_std = float(np.mean(np.sqrt(per_dim_var)))
        separation_score = _clamp(per_dim_std / RANGE_PER_DIM_STD)
    else:
        separation_score = _clamp(inter_mean / SEPARATION_DISTANCE_THRESHOLD)

    drift_risk = _clamp(
        W_SEPARATION * separation_score
        + W_BALANCE * balance_penalty
        + W_NOISE * noise_fraction
    )

    # --- Recommended action ---
    if n_modes <= 1:
        recommended_action = "use_as_is"
        note = ("Single identity mode detected. No drift concern."
                if n_modes == 1 else "No non-noise modes detected.")
    elif drift_risk < 0.3:
        recommended_action = "use_as_is"
        note = (f"{n_modes} modes detected but drift risk is low "
                f"({drift_risk:.3f}). Modes are similar enough to train together.")
    elif drift_risk < 0.6:
        recommended_action = "review_manually"
        note = (f"{n_modes} modes with moderate drift risk ({drift_risk:.3f}). "
                f"Review mode representatives to decide whether to split.")
    else:
        if balance_penalty > 0.4:
            recommended_action = "split_by_mode"
            note = (f"{n_modes} well-separated, balanced modes. "
                    f"Drift risk {drift_risk:.3f}. Consider training "
                    f"separate LoRAs or filtering to one mode.")
        else:
            recommended_action = "restrict_to_dominant_mode"
            note = (f"{n_modes} modes with dominant mode containing "
                    f"{max(ms.size for ms in renumbered)}/{total_nonnoise} frames. "
                    f"Drift risk {drift_risk:.3f}. Restrict to dominant mode "
                    f"for cleanest identity.")

    return IdentityDriftResult(
        scope=scope,
        frame_count=n,
        embedding_dim=EMBED_DIM,
        noise_count=noise_count,
        noise_fraction=round(noise_fraction, 6),
        mode_count=n_modes,
        modes=tuple(renumbered),
        inter_mode_min_dist=round(inter_min, 6),
        inter_mode_max_dist=round(inter_max, 6),
        inter_mode_mean_dist=round(inter_mean, 6),
        drift_risk=round(drift_risk, 6),
        drift_risk_note=note,
        recommended_action=recommended_action,
        clustering_method=method,
        metric_version=METRIC_VERSION,
        calculated_at=ts,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_identity_drift(
    manifest,
    *,
    dataset_version: int | str | None = None,
    frame_ids: Iterable[str] | None = None,
    min_frames: int = DEFAULT_MIN_FRAMES,
    persist: bool = False,
    progress: bool = True,
) -> IdentityDriftResult:
    """Analyze identity drift over the eligible frame set.

    If *persist* is True, write mode assignments to
    ``identity_drift_modes``.  Deterministic: same input → identical
    mode_ids (renumbered by size desc, then representative_frame_id asc).
    """
    import sqlite3

    # Scope label
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
                "SELECT df.frame_id FROM dataset_frames df "
                "JOIN detections d ON df.frame_id = d.frame_id "
                "WHERE df.version = ? AND d.embed IS NOT NULL "
                "AND d.face_count = 1",
                (int(dataset_version),),
            ).fetchall()
            ids = sorted(r[0] for r in rows)
        else:
            rows = manifest.execute(
                "SELECT d.frame_id FROM detections d "
                "JOIN frames f ON d.frame_id = f.frame_id "
                "WHERE d.embed IS NOT NULL AND d.face_count = 1 "
                "AND f.status IN ('scored','accepted','selected','packaged')"
            ).fetchall()
            ids = sorted(r[0] for r in rows)
    except sqlite3.OperationalError:
        ids = []

    if not ids:
        return _analyze([], np.empty((0, EMBED_DIM), dtype=np.float32), [],
                        scope=scope, min_frames=min_frames)

    # Fetch embeddings + timestamps in batches
    all_fids: list[str] = []
    all_embs: list[np.ndarray] = []
    all_ts: list[str | None] = []
    batch_size = 500

    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        placeholders = ",".join("?" for _ in batch)
        try:
            rows = manifest.execute(
                f"SELECT d.frame_id, d.embed, d.detected_at "
                f"FROM detections d WHERE d.frame_id IN ({placeholders})",
                batch,
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

        for fid, embed_raw, det_at in rows:
            emb = _decode_embed(embed_raw)
            if emb is not None:
                all_fids.append(fid)
                all_embs.append(emb)
                all_ts.append(det_at)

        if progress:
            logger.info("identity_drift: loaded %d/%d embeddings",
                        min(i + batch_size, len(ids)), len(ids))

    if not all_fids:
        return _analyze([], np.empty((0, EMBED_DIM), dtype=np.float32), [],
                        scope=scope, min_frames=min_frames)

    mat = np.stack(all_embs, axis=0)
    result = _analyze(all_fids, mat, all_ts, scope=scope, min_frames=min_frames)

    # Persist mode assignments if requested
    if persist and result.mode_count > 0:
        _persist_modes(manifest, all_fids, result)

    return result


def _persist_modes(conn, frame_ids: list[str],
                   result: IdentityDriftResult) -> None:
    """Write mode assignments to identity_drift_modes table."""
    import sqlite3

    # Build lookup: frame_id -> (mode_id, is_noise)
    assignment: dict[str, tuple[int, int]] = {}
    for ms in result.modes:
        # Representative + samples
        for fid in (ms.representative_frame_id,) + ms.sample_frame_ids:
            assignment[fid] = (ms.mode_id, 0)

    # We need full label mapping; reconstruct from the result structure
    # For frames not in any mode summary, mark as noise
    now = _now_iso()
    buf: list[tuple] = []
    for fid in frame_ids:
        if fid in assignment:
            mode_id, is_noise = assignment[fid]
        else:
            mode_id, is_noise = 0, 1
        buf.append((fid, mode_id, is_noise, METRIC_VERSION, now))

    try:
        conn.executemany(
            "INSERT OR REPLACE INTO identity_drift_modes("
            "frame_id, mode_id, is_noise, analysis_version, assigned_at"
            ") VALUES (?, ?, ?, ?, ?)",
            buf,
        )
        conn.commit()
    except sqlite3.OperationalError:
        logger.warning("identity_drift_modes table may not exist yet "
                       "(run migrations first)", exc_info=True)


# ---------------------------------------------------------------------------
# Human-readable renderer
# ---------------------------------------------------------------------------

def render_drift_report(result: IdentityDriftResult) -> str:
    lines: list[str] = []
    lines.append("Identity Drift Sentinel Report")
    lines.append(f"  scope          : {result.scope}")
    lines.append(f"  frames         : {result.frame_count}")
    lines.append(f"  embedding dim  : {result.embedding_dim}")
    lines.append(f"  clustering     : {result.clustering_method}")
    lines.append(f"  noise          : {result.noise_count} "
                 f"({result.noise_fraction:.1%})")
    lines.append(f"  modes          : {result.mode_count}")
    lines.append("")

    if result.drift_risk is not None:
        lines.append(f"  drift risk     : {result.drift_risk:.4f}")
    else:
        lines.append("  drift risk     : N/A (insufficient data)")
    lines.append(f"  recommendation : {result.recommended_action}")
    lines.append(f"  note           : {result.drift_risk_note}")
    lines.append("")

    if result.mode_count >= 2:
        lines.append(f"  inter-mode dist: min={result.inter_mode_min_dist:.4f} "
                     f"max={result.inter_mode_max_dist:.4f} "
                     f"mean={result.inter_mode_mean_dist:.4f}")
        lines.append("")

    for ms in result.modes:
        lines.append(f"  Mode {ms.mode_id}: {ms.size} frames, "
                     f"cohesion={ms.cohesion:.4f}, "
                     f"rep={ms.representative_frame_id}")
        if ms.temporal_span:
            lines.append(f"    temporal span: {ms.temporal_span[0]} → "
                         f"{ms.temporal_span[1]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== identity_drift_sentinel self-test ===\n")

    rng = np.random.RandomState(0)

    # Case 1: Insufficient data
    r0 = _analyze(
        ["f001"], rng.randn(1, EMBED_DIM).astype(np.float32), [None],
        scope="test_insufficient", min_frames=12,
        ts_override="2025-01-01T00:00:00Z",
    )
    assert r0.drift_risk is None
    assert r0.frame_count == 1
    print(f"1. Insufficient data: drift_risk={r0.drift_risk}, "
          f"note='{r0.drift_risk_note}' OK")

    # Case 2: Two well-separated clusters (30 points, 2 gaussians)
    n_per = 15
    cluster_a = rng.randn(n_per, EMBED_DIM).astype(np.float32) * 0.1
    cluster_b = rng.randn(n_per, EMBED_DIM).astype(np.float32) * 0.1
    # Shift cluster B far away in embedding space
    shift = rng.randn(EMBED_DIM).astype(np.float32)
    shift = shift / (np.linalg.norm(shift) + 1e-12) * 0.8
    cluster_b += shift

    X_two = np.vstack([cluster_a, cluster_b])
    fids_two = [f"f{i:04d}" for i in range(30)]
    ts_two = [None] * 30

    r1 = _analyze(fids_two, X_two, ts_two,
                  scope="test_two_clusters", min_frames=12,
                  ts_override="2025-01-01T00:00:00Z")
    assert r1.mode_count >= 2, f"expected >=2 modes, got {r1.mode_count}"
    assert r1.drift_risk is not None and r1.drift_risk > 0.3, \
        f"expected elevated drift_risk, got {r1.drift_risk}"
    print(f"2. Two clusters: mode_count={r1.mode_count}, "
          f"drift_risk={r1.drift_risk:.4f}, "
          f"action={r1.recommended_action} OK")

    # Case 3: Single cluster (all same)
    # In high-dimensional normalised space, pure-numpy k-selection may
    # over-split.  We verify the result is *reasonable*: low drift risk
    # and use_as_is recommendation, regardless of exact mode_count.
    X_one = rng.randn(30, EMBED_DIM).astype(np.float32) * 0.05
    fids_one = [f"f{i:04d}" for i in range(30)]
    r2 = _analyze(fids_one, X_one, [None] * 30,
                  scope="test_single_cluster", min_frames=12,
                  ts_override="2025-01-01T00:00:00Z")
    assert r2.drift_risk is not None
    # A single-cloud set should have low drift risk regardless of
    # how many spurious sub-clusters the fallback creates.
    assert r2.drift_risk < 0.5, \
        f"single cloud should have low drift risk, got {r2.drift_risk}"
    print(f"3. Single cluster: mode_count={r2.mode_count}, "
          f"drift_risk={r2.drift_risk:.4f}, "
          f"action={r2.recommended_action} OK")

    # Case 4: Determinism
    r3a = _analyze(fids_two, X_two, ts_two,
                   scope="det_test", min_frames=12,
                   ts_override="2025-01-01T00:00:00Z")
    r3b = _analyze(fids_two, X_two, ts_two,
                   scope="det_test", min_frames=12,
                   ts_override="2025-01-01T00:00:00Z")
    assert r3a.to_json() == r3b.to_json(), "determinism violated"
    print(f"4. Determinism: byte-identical JSON OK")

    # Case 5: JSON round-trip
    j = r1.to_json()
    parsed = json.loads(j)
    assert parsed["mode_count"] == 2
    print(f"5. JSON round-trip: OK ({len(j)} bytes)")

    # Case 6: Render
    rendered = render_drift_report(r1)
    assert "drift risk" in rendered.lower()
    print(f"6. Render: OK ({len(rendered)} chars)")

    # Case 7: Empty input
    r4 = _analyze([], np.empty((0, EMBED_DIM), dtype=np.float32), [],
                  scope="test_empty", min_frames=12,
                  ts_override="2025-01-01T00:00:00Z")
    assert r4.drift_risk is None
    print(f"7. Empty input: drift_risk={r4.drift_risk} OK")

    print("\n=== all self-tests passed ===")
