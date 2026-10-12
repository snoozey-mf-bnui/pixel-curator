"""2D projection of embedding vectors with graceful fallbacks.

Preference order: UMAP -> t-SNE (scikit-learn) -> PCA (numpy SVD) ->
pure-python PCA (power iteration; deterministic, dependency-free, fine
for the small fixtures used in tests).  The method actually used is
always reported so the UI/CLI can say so professionally.
"""

from __future__ import annotations

import math
from typing import Sequence


def available_methods() -> dict[str, bool]:
    out = {"pca": True}                       # always (pure-python fallback)
    for name, mod in (("umap", "umap"), ("tsne", "sklearn"),
                      ("numpy", "numpy")):
        try:
            __import__(mod)
            out[name] = True
        except ImportError:
            out[name] = False
    return out


def _normalize(points: list[list[float]]) -> list[list[float]]:
    if not points:
        return points
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    rx = (max(xs) - min(xs)) or 1.0
    ry = (max(ys) - min(ys)) or 1.0
    return [[round((p[0] - min(xs)) / rx * 2 - 1, 4),
             round((p[1] - min(ys)) / ry * 2 - 1, 4)] for p in points]


# ---------------------------------------------------------------------------
# Pure-python PCA (top-2 components via power iteration) - test-safe
# ---------------------------------------------------------------------------

def _pca_pure(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    n = len(vectors)
    dim = len(vectors[0])
    mean = [sum(v[d] for v in vectors) / n for d in range(dim)]
    centered = [[v[d] - mean[d] for d in range(dim)] for v in vectors]

    def matvec(w: list[float]) -> list[float]:
        # (X^T X) w  computed as X^T (X w) - O(n*dim)
        xw = [sum(row[d] * w[d] for d in range(dim)) for row in centered]
        return [sum(centered[i][d] * xw[i] for i in range(n))
                for d in range(dim)]

    def power_component(seed: int, deflate: list[float] | None) -> list[float]:
        w = [math.sin(seed + d * 0.7) for d in range(dim)]
        for _ in range(60):
            if deflate is not None:
                dot = sum(a * b for a, b in zip(w, deflate))
                w = [a - dot * b for a, b in zip(w, deflate)]
            w = matvec(w)
            norm = math.sqrt(sum(a * a for a in w)) or 1.0
            w = [a / norm for a in w]
        return w

    c1 = power_component(1, None)
    c2 = power_component(2, c1)
    return _normalize([[sum(r[d] * c1[d] for d in range(dim)),
                        sum(r[d] * c2[d] for d in range(dim))]
                       for r in centered])


def _pca_numpy(vectors) -> list[list[float]]:
    import numpy as np
    arr = np.asarray(vectors, dtype=np.float64)
    arr = arr - arr.mean(axis=0)
    _u, _s, vt = np.linalg.svd(arr, full_matrices=False)
    pts = arr @ vt[:2].T
    return _normalize([[float(x), float(y)] for x, y in pts])


def project_2d(vectors: Sequence[Sequence[float]],
               method: str = "auto") -> tuple[list[list[float]], str,
                                              list[str]]:
    """Returns (normalized [-1,1] points, method_used, warnings)."""
    warnings: list[str] = []
    if len(vectors) < 3:
        return ([[0.0, 0.0] for _ in vectors], "identity",
                ["fewer than 3 items - projection skipped"])

    chain = ([method] if method not in ("", "auto")
             else ["umap", "tsne", "pca"])
    for m in chain:
        if m == "umap":
            try:
                import numpy as np
                import umap
                pts = umap.UMAP(
                    n_components=2, random_state=42,
                    n_neighbors=min(15, len(vectors) - 1)).fit_transform(
                    np.asarray(vectors, dtype=np.float32))
                return (_normalize([[float(x), float(y)] for x, y in pts]),
                        "umap", warnings)
            except ImportError:
                warnings.append("umap-learn not installed - falling back")
            except Exception as exc:
                warnings.append(f"UMAP failed ({exc}) - falling back")
        elif m == "tsne":
            try:
                import numpy as np
                from sklearn.manifold import TSNE
                pts = TSNE(n_components=2, random_state=42,
                           perplexity=min(30, max(2, len(vectors) // 4)),
                           init="pca").fit_transform(
                    np.asarray(vectors, dtype=np.float32))
                return (_normalize([[float(x), float(y)] for x, y in pts]),
                        "tsne", warnings)
            except ImportError:
                warnings.append("scikit-learn not installed - falling back")
            except Exception as exc:
                warnings.append(f"t-SNE failed ({exc}) - falling back")
        elif m == "pca":
            try:
                return _pca_numpy(vectors), "pca", warnings
            except ImportError:
                warnings.append("numpy not installed - pure-python PCA")
                return _pca_pure(vectors), "pca_pure", warnings
    # explicit unknown method
    warnings.append(f"unknown projection method '{method}' - using PCA")
    try:
        return _pca_numpy(vectors), "pca", warnings
    except ImportError:
        return _pca_pure(vectors), "pca_pure", warnings
