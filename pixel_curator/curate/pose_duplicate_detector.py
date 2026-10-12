"""Priority 8 - body-pose duplicate detection (pose near-duplicate awareness).

A *secondary* duplicate layer that complements - never replaces - the
perceptual-hash detector in ``duplicate_detector.py``. Two frames can be
visually distinct (different crop, lighting, time) yet capture effectively the
same body pose; this module flags those pose near-duplicates so a dataset is
not silently overfilled with one standing/front-facing posture.

Contract: additive, default-OFF, fully non-destructive - it emits signatures,
scores, cluster ids and review hints only. It never deletes, moves, reorders or
hard-filters a sample, never reads images (signature is built from pose
metadata already computed), is deterministic (stable ids, sorted inputs) and
dependency-light (pure python). Weak metadata can mark a *review candidate* but
never a hard near-duplicate.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

# Canonical keypoint ordering used to build a translation/scale-invariant
# signature vector. Missing joints are stored as the sentinel below.
KP_ORDER = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
_MISSING = 999.0  # sentinel for an absent joint in the flat vector

# Body-angle neighbor map (for metadata-only similarity).
_ANGLE_NEIGHBORS = {
    "front": {"three_quarter"}, "three_quarter": {"front", "side"},
    "side": {"three_quarter", "back"}, "back": {"side"},
}


@dataclass
class PoseDuplicateConfig:
    enabled: bool = False
    similarity_threshold: float = 0.92
    near_similarity_threshold: float = 0.86
    min_keypoints: int = 6
    use_body_angle: bool = True
    use_pose_label: bool = True
    use_keypoints: bool = True
    group_by_session: bool = False
    max_examples_per_cluster: int = 10

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "similarity_threshold": self.similarity_threshold,
            "near_similarity_threshold": self.near_similarity_threshold,
            "min_keypoints": self.min_keypoints,
            "use_body_angle": self.use_body_angle,
            "use_pose_label": self.use_pose_label,
            "use_keypoints": self.use_keypoints,
            "group_by_session": self.group_by_session,
            "max_examples_per_cluster": self.max_examples_per_cluster,
        }


@dataclass
class PoseSignature:
    signature_id: str
    pose_label: str = "unknown"
    body_angle_bucket: str = "unknown"
    normalized_keypoints: list = field(default_factory=list)
    confidence: float = 0.0
    source_fields: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def has_keypoints(self) -> bool:
        return any(v != _MISSING for v in self.normalized_keypoints) \
            if self.normalized_keypoints else False

    def to_dict(self) -> dict:
        return {
            "signature_id": self.signature_id,
            "pose_label": self.pose_label,
            "body_angle_bucket": self.body_angle_bucket,
            "normalized_keypoints": [round(float(v), 5)
                                     for v in self.normalized_keypoints],
            "confidence": round(float(self.confidence), 4),
            "source_fields": list(self.source_fields),
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass
class PoseDuplicateResult:
    pose_signature: PoseSignature | None = None
    pose_duplicate_score: float | None = None
    near_duplicate: bool = False
    duplicate_cluster_id: str | None = None
    duplicate_rank: int | None = None
    representative: bool = True
    similar_sample_ids: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pose_signature": self.pose_signature.to_dict()
            if self.pose_signature else None,
            "pose_signature_id": self.pose_signature.signature_id
            if self.pose_signature else None,
            "pose_duplicate_score": round(float(self.pose_duplicate_score), 4)
            if self.pose_duplicate_score is not None else None,
            "near_duplicate": self.near_duplicate,
            "duplicate_cluster_id": self.duplicate_cluster_id,
            "duplicate_rank": self.duplicate_rank,
            "representative": self.representative,
            "similar_sample_ids": list(self.similar_sample_ids),
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Signature construction
# ---------------------------------------------------------------------------
def _extract_keypoints(sample: dict):
    """Return a ``{name: (x, y)}`` dict from whatever pose field is present."""
    for key in ("normalized_keypoints", "keypoints", "pose_keypoints",
                "landmarks"):
        kp = sample.get(key)
        if isinstance(kp, dict) and kp:
            out = {}
            for name, pt in kp.items():
                try:
                    out[str(name)] = (float(pt[0]), float(pt[1]))
                except Exception:
                    continue
            if out:
                return out
        if isinstance(kp, (list, tuple)) and kp and \
                all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in kp):
            # positional list aligned to KP_ORDER
            out = {}
            for i, p in enumerate(kp[:len(KP_ORDER)]):
                try:
                    out[KP_ORDER[i]] = (float(p[0]), float(p[1]))
                except Exception:
                    continue
            if out:
                return out
    return {}


def _normalize_keypoints(kp: dict, min_keypoints: int):
    """Translation/scale-invariant flat vector over ``KP_ORDER`` (missing joints
    = sentinel). Returns ``(vector, n_present)``; ``([], 0)`` if too few."""
    present = {n: p for n, p in kp.items() if n in KP_ORDER}
    if len(present) < min_keypoints:
        return [], len(present)
    xs = [p[0] for p in present.values()]
    ys = [p[1] for p in present.values()]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    scale = math.sqrt(sum((x - cx) ** 2 + (y - cy) ** 2
                          for x, y in zip(xs, ys)) / len(xs)) or 1.0
    vec = []
    for name in KP_ORDER:
        if name in present:
            x, y = present[name]
            vec.extend([(x - cx) / scale, (y - cy) / scale])
        else:
            vec.extend([_MISSING, _MISSING])
    return vec, len(present)


def _sig_hash(payload: str) -> str:
    return "pose_sig_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def build_pose_signature(sample: dict,
                         config: PoseDuplicateConfig | None = None
                         ) -> PoseSignature:
    """Build a deterministic pose signature from a sample's pose metadata.
    Never raises; insufficient data yields an ``unknown`` signature + warning."""
    cfg = config or PoseDuplicateConfig()
    try:
        from .pose_angle import normalize_pose_signals
        pose = normalize_pose_signals(sample)
    except Exception:
        pose = {}
    pose_label = pose.get("pose_label", "unknown")
    angle_bucket = pose.get("body_angle_bucket", "unknown")
    conf = float(pose.get("confidence") or 0.0)
    sources, warns = [], []

    vec: list = []
    if cfg.use_keypoints:
        kp = _extract_keypoints(sample)
        if kp:
            vec, n_present = _normalize_keypoints(kp, cfg.min_keypoints)
            if vec:
                sources.append("keypoints")
                conf = max(conf, 0.85)
            elif n_present:
                warns.append("pose_signature_partial: too few keypoints "
                             f"({n_present} < {cfg.min_keypoints})")

    if cfg.use_pose_label and pose_label != "unknown":
        sources.append("pose_label")
    if cfg.use_body_angle and angle_bucket != "unknown":
        sources.append("body_angle")

    if not sources:
        warns.append("pose_signature_partial: missing keypoints/body angle")

    # signature id: keypoint geometry if present, else quantized label+angle
    if vec:
        payload = "kp:" + ",".join(f"{v:.3f}" for v in vec)
    else:
        ad = pose.get("angle_degrees")
        aq = "na" if ad is None else str(int(round(float(ad) / 15.0)) * 15)
        payload = f"meta:{pose_label}|{angle_bucket}|{aq}"
    if not sources:
        payload = "unknown:" + payload
    return PoseSignature(
        signature_id=_sig_hash(payload), pose_label=pose_label,
        body_angle_bucket=angle_bucket, normalized_keypoints=vec,
        confidence=round(conf, 4), source_fields=sources, warnings=warns)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
def compare_pose_signatures(a: PoseSignature, b: PoseSignature,
                            config: PoseDuplicateConfig | None = None) -> float:
    """Deterministic pose similarity in [0, 1]. Keypoint geometry yields the
    strongest (duplicate-capable) signal; metadata-only similarity is capped
    below the hard threshold so weak data can only ever be a review candidate.
    Never raises."""
    cfg = config or PoseDuplicateConfig()
    if a is None or b is None:
        return 0.0
    # 1) keypoint geometry (translation/scale invariant)
    if a.has_keypoints() and b.has_keypoints() and \
            len(a.normalized_keypoints) == len(b.normalized_keypoints):
        dists, overlap = [], 0
        va, vb = a.normalized_keypoints, b.normalized_keypoints
        for i in range(0, len(va), 2):
            if va[i] == _MISSING or vb[i] == _MISSING:
                continue
            overlap += 1
            dists.append(math.hypot(va[i] - vb[i], va[i + 1] - vb[i + 1]))
        if overlap >= cfg.min_keypoints and dists:
            mean_d = sum(dists) / len(dists)
            return round(math.exp(-2.0 * mean_d), 4)  # identical -> 1.0
    # 2) metadata: pose label + body angle (capped < similarity_threshold)
    cap = min(0.90, cfg.similarity_threshold - 0.02)
    lab_a, lab_b = a.pose_label, b.pose_label
    ang_a, ang_b = a.body_angle_bucket, b.body_angle_bucket
    have_lab = lab_a != "unknown" and lab_b != "unknown"
    have_ang = ang_a != "unknown" and ang_b != "unknown"
    if not have_lab and not have_ang:
        return 0.0
    same_lab = have_lab and lab_a == lab_b
    same_ang = have_ang and ang_a == ang_b
    neigh_ang = have_ang and ang_b in _ANGLE_NEIGHBORS.get(ang_a, set())
    if same_lab and same_ang:
        return cap                       # strongest metadata match (review band)
    if same_lab and neigh_ang:
        return round(min(cap, 0.78), 4)
    if same_ang and not have_lab:
        return 0.60
    if same_lab and not have_ang:
        return 0.55
    if neigh_ang:
        return 0.50
    return 0.20


# ---------------------------------------------------------------------------
# Dataset-level clustering
# ---------------------------------------------------------------------------
def _sample_id(sample: dict, idx: int) -> str:
    for key in ("frame_id", "sample_id", "image_target", "image", "path",
                "source_path"):
        v = sample.get(key)
        if v:
            return str(v)
    return f"sample_{idx:06d}"


def _quality_rank_key(sample: dict, sig: PoseSignature, sid: str):
    """Representative preference: higher quality first; stable path tie-break."""
    def g(*keys, default=0.0):
        f = sample
        for nest in (None, "signals", "metadata"):
            src = f if nest is None else (f.get(nest) if isinstance(f, dict)
                                          else None)
            if isinstance(src, dict):
                for k in keys:
                    if src.get(k) is not None:
                        try:
                            return float(src[k])
                        except (TypeError, ValueError):
                            pass
        return default
    blur = g("motion_blur_score", default=None)
    blur_good = (1.0 - blur) if blur is not None else g("sharpness", default=0.0)
    return (
        -g("quality_score", "global_quality_avg"),
        -g("subject_integrity_score", "figure_integrity_score",
           "integrity_score"),
        -g("region_balance_score"),
        -blur_good,
        sid,  # earliest stable id as final deterministic tie-break
    )


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def detect_pose_duplicates(samples: list,
                           config: PoseDuplicateConfig | None = None) -> list:
    """Dataset-level pose near-duplicate clustering. Returns one
    ``PoseDuplicateResult`` per input sample, in input order. Non-destructive:
    nothing is removed or reordered. Deterministic. Never raises."""
    cfg = config or PoseDuplicateConfig()
    n = len(samples or [])
    sigs = [build_pose_signature(s, cfg) for s in (samples or [])]
    ids = [_sample_id(s, i) for i, s in enumerate(samples or [])]
    results = [PoseDuplicateResult(pose_signature=sigs[i],
                                   warnings=list(sigs[i].warnings))
               for i in range(n)]
    if n < 2:
        return results

    sessions = ([str(_session_of(s)) for s in samples]
                if cfg.group_by_session else [""] * n)
    uf = _UnionFind(n)
    best_sim = [0.0] * n
    neighbors: dict = {i: set() for i in range(n)}
    # pairwise within (optional) session groups
    for i in range(n):
        for j in range(i + 1, n):
            if cfg.group_by_session and sessions[i] != sessions[j]:
                continue
            sim = compare_pose_signatures(sigs[i], sigs[j], cfg)
            if sim >= cfg.near_similarity_threshold:
                uf.union(i, j)
                neighbors[i].add(j); neighbors[j].add(i)
                best_sim[i] = max(best_sim[i], sim)
                best_sim[j] = max(best_sim[j], sim)

    clusters: dict = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    for members in clusters.values():
        if len(members) < 2:
            i = members[0]
            results[i].representative = True
            results[i].duplicate_rank = 1
            results[i].pose_duplicate_score = round(best_sim[i], 4)
            continue
        ordered = sorted(members,
                         key=lambda i: _quality_rank_key(samples[i], sigs[i],
                                                         ids[i]))
        rep = ordered[0]
        cluster_id = "pose_dup_" + hashlib.sha256(
            sigs[rep].signature_id.encode("utf-8")).hexdigest()[:10]
        member_ids = [ids[i] for i in members]
        for rank, i in enumerate(ordered, start=1):
            r = results[i]
            r.duplicate_cluster_id = cluster_id
            r.duplicate_rank = rank
            r.representative = (i == rep)
            r.pose_duplicate_score = round(best_sim[i], 4)
            r.similar_sample_ids = sorted(
                x for x in member_ids if x != ids[i]
            )[:cfg.max_examples_per_cluster]
            # hard near-duplicate only when geometry-grade similarity is reached
            r.near_duplicate = (not r.representative
                                and best_sim[i] >= cfg.similarity_threshold)
            if r.near_duplicate:
                r.warnings.append("pose_near_duplicate_review")
    return results


def _session_of(sample: dict):
    for key in ("source_session", "session_id", "session", "source_id",
                "video_id"):
        v = sample.get(key)
        if v is not None:
            return v
    sp = sample.get("source_path") or sample.get("path")
    if sp:
        from pathlib import Path
        return Path(str(sp)).parent.name
    return "unknown"


# ---------------------------------------------------------------------------
# Dataset-level summary
# ---------------------------------------------------------------------------
def summarize_pose_duplicates(results: list, *, sample_count: int | None = None
                              ) -> dict:
    """Summarize a list of ``PoseDuplicateResult`` (or their dicts) for a
    dataset card. Never raises."""
    rows = []
    for r in results or []:
        rows.append(r.to_dict() if isinstance(r, PoseDuplicateResult) else r)
    n = sample_count if sample_count is not None else len(rows)
    with_sig = sum(1 for r in rows if r.get("pose_signature_id"))
    near = sum(1 for r in rows if r.get("near_duplicate"))
    review = sum(1 for r in rows
                 if r.get("duplicate_cluster_id") and not r.get("near_duplicate")
                 and not r.get("representative"))
    cluster_sizes: dict = {}
    for r in rows:
        cid = r.get("duplicate_cluster_id")
        if cid:
            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1
    reps = sum(1 for r in rows if r.get("representative"))
    largest = max(cluster_sizes.values()) if cluster_sizes else 0
    largest_clusters = sorted(
        ({"cluster_id": k, "size": v} for k, v in cluster_sizes.items()),
        key=lambda d: (-d["size"], d["cluster_id"]))[:5]
    return {
        "pose_signature_count": with_sig,
        "pose_signature_coverage_ratio": round(with_sig / n, 4) if n else 0.0,
        "near_duplicate_count": near,
        "near_duplicate_ratio": round(near / n, 4) if n else 0.0,
        "duplicate_cluster_count": len(cluster_sizes),
        "largest_cluster_size": largest,
        "representative_count": reps,
        "review_candidate_count": review,
        "largest_clusters": largest_clusters,
    }


__all__ = [
    "PoseDuplicateConfig", "PoseSignature", "PoseDuplicateResult",
    "build_pose_signature", "compare_pose_signatures", "detect_pose_duplicates",
    "summarize_pose_duplicates",
]
