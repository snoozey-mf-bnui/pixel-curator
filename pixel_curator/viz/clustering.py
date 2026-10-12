"""Embedding clustering with lightweight fallbacks + cluster summaries.

Preference order: HDBSCAN -> scikit-learn KMeans -> the studio's own
pure-python KMeans (curate.diversity) - so clustering always works,
even with no optional dependencies installed.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def _auto_k(n: int) -> int:
    from ..curate.diversity import auto_k
    return auto_k(n)


def cluster_embeddings(vectors: Sequence[Sequence[float]],
                       k: int = 0) -> tuple[list[str], str, list[str]]:
    """Returns (cluster ids per item, method_used, warnings)."""
    warnings: list[str] = []
    n = len(vectors)
    if n < 3:
        return ["cluster_000"] * n, "single", ["too few items to cluster"]
    k = k or min(_auto_k(n), n)

    try:
        import hdbscan
        import numpy as np
        labels = hdbscan.HDBSCAN(
            min_cluster_size=max(2, n // 50)).fit_predict(
            np.asarray(vectors, dtype=np.float32))
        return ([f"cluster_{max(0, int(l)):03d}" if l >= 0 else "noise"
                 for l in labels], "hdbscan", warnings)
    except ImportError:
        pass
    except Exception as exc:
        warnings.append(f"HDBSCAN failed ({exc}) - falling back to KMeans")

    try:
        import numpy as np
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=k, random_state=42,
                        n_init=4).fit_predict(
            np.asarray(vectors, dtype=np.float32))
        return [f"cluster_{int(l):03d}" for l in labels], "kmeans", warnings
    except ImportError:
        pass
    except Exception as exc:
        warnings.append(f"KMeans failed ({exc}) - pure-python fallback")

    from ..curate.diversity import kmeans as pure_kmeans
    labels = pure_kmeans([list(map(float, v)) for v in vectors], k)
    return ([f"cluster_{int(l):03d}" for l in labels],
            "kmeans_pure", warnings)


def cluster_summaries(items: list[dict]) -> list[dict]:
    """Professional per-cluster summaries from projected items
    ({item_id, cluster_id, quality_score, identity_confidence, framing,
    study_tags, duplicate_group_id, ...})."""
    by_cluster: dict[str, list[dict]] = {}
    for it in items:
        by_cluster.setdefault(it.get("cluster_id") or "noise", []).append(it)

    out: list[dict] = []
    for cid in sorted(by_cluster):
        members = by_cluster[cid]
        n = len(members)
        q = [m["quality_score"] for m in members
             if m.get("quality_score") is not None]
        ident = [m["identity_confidence"] for m in members
                 if m.get("identity_confidence") is not None]
        framing = Counter(m.get("framing") or "unscored" for m in members)
        tags = Counter(t for m in members
                       for t in (m.get("study_tags") or []))
        dups = sum(1 for m in members if m.get("duplicate_group_id"))
        avg_q = round(sum(q) / len(q), 3) if q else None
        avg_id = round(sum(ident) / len(ident), 3) if ident else None
        dup_density = round(dups / n, 3)
        rep = max(members,
                  key=lambda m: (m.get("quality_score") or 0,
                                 m["item_id"]))
        priority = "low"
        if (avg_q is not None and avg_q < 0.5) or dup_density > 0.5:
            priority = "high"
        elif (avg_id is not None and avg_id < 0.6) or dup_density > 0.25:
            priority = "medium"
        out.append({
            "cluster_id": cid,
            "item_count": n,
            "representative_item": rep["item_id"],
            "quality_profile": {"avg_quality": avg_q,
                                "avg_identity_confidence": avg_id},
            "dominant_framing": framing.most_common(1)[0][0],
            "dominant_tags": [t for t, _n in tags.most_common(3)],
            "duplicate_density": dup_density,
            "review_priority": priority,
            "cluster_summary": (
                f"{n} items, framing mostly "
                f"{framing.most_common(1)[0][0]}, "
                f"avg quality {avg_q if avg_q is not None else 'n/a'}, "
                f"duplicate density {dup_density:.0%} - "
                f"{priority} review priority"),
        })
    return out
