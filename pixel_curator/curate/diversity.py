"""Diversity clustering (Phase 2.4): CLIP image embeddings + k-means.

Groups selected frames by scene/outfit/lighting so packaging can pull a
balanced spread instead of 400 near-identical couch-angle frames.

Backends (first available wins):
  1. open_clip (ViT-B-32, laion2b) - ships with Forge env       [cluster] extra
  2. transformers CLIPModel (openai/clip-vit-base-patch32)
Device: MPS on Apple Silicon when torch sees it, else CPU.

k-means is implemented here (numpy, k-means++ init) - no sklearn needed.
Embeddings are cached in the manifest (clip_embeds) - embed once, recluster
freely with different k.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Generator
from pathlib import Path

from .. import manifest
from ..config import ClusterConfig
from ..util import now_iso, setup_logging

logger = logging.getLogger(__name__)

_CLIP = None
_CLIP_ERR: str | None = None


def clip_available() -> tuple[bool, str]:
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:
        return False, f"missing torch/numpy: {exc}"
    try:
        import open_clip  # noqa: F401
        return True, "open_clip"
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    try:
        import transformers  # noqa: F401
        return True, "transformers"
    except Exception as exc:
        return False, f"missing open_clip/transformers: {exc}"


def _device():
    from ..runtime import get_device_name
    return get_device_name("auto")        # cuda -> mps -> cpu


def _load_clip():
    """Returns (embed_fn(list[Path]) -> np.ndarray, backend_name)."""
    global _CLIP, _CLIP_ERR
    if _CLIP is not None:
        return _CLIP
    if _CLIP_ERR is not None:
        raise RuntimeError(_CLIP_ERR)
    try:
        import numpy as np
        import torch
        from PIL import Image
        dev = _device()
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            model = model.to(dev).eval()

            def embed(paths: list[Path]):
                imgs = []
                ok_idx = []
                for i, p in enumerate(paths):
                    try:
                        with Image.open(p) as im:
                            imgs.append(preprocess(im.convert("RGB")))
                        ok_idx.append(i)
                    except Exception:
                        logger.debug("suppressed exception", exc_info=True)
                        pass
                if not imgs:
                    return ok_idx, np.zeros((0, 512), dtype=np.float32)
                batch = torch.stack(imgs).to(dev)
                with torch.no_grad():
                    feats = model.encode_image(batch)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                return ok_idx, feats.float().cpu().numpy()

            _CLIP = (embed, f"open_clip ViT-B-32 [{dev}]")
            return _CLIP
        except ImportError:
            from transformers import CLIPModel, CLIPProcessor
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
            proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            def embed(paths: list[Path]):
                pil = []
                ok_idx = []
                for i, p in enumerate(paths):
                    try:
                        with Image.open(p) as im:
                            pil.append(im.convert("RGB"))
                        ok_idx.append(i)
                    except Exception:
                        logger.debug("suppressed exception", exc_info=True)
                        pass
                if not pil:
                    return ok_idx, np.zeros((0, 512), dtype=np.float32)
                inputs = proc(images=pil, return_tensors="pt").to(dev)
                with torch.no_grad():
                    feats = model.get_image_features(**inputs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                return ok_idx, feats.float().cpu().numpy()

            _CLIP = (embed, f"transformers CLIP ViT-B/32 [{dev}]")
            return _CLIP
    except Exception as exc:
        _CLIP_ERR = (f"CLIP unavailable: {exc}\n"
                     "Install:  pip install 'pixel-curator[cluster]'")
        raise RuntimeError(_CLIP_ERR)


# -----------------------------
# k-means (numpy optional; pure-python fallback for small inputs/tests)
# -----------------------------

def kmeans(data, k: int, max_iter: int = 30, seed: int = 7) -> list[int]:
    """data: list of equal-length float vectors (or numpy 2d array).
    Returns cluster label per row. k-means++ init, empty-cluster repair."""
    try:
        import numpy as np
        X = np.asarray(data, dtype=np.float32)
        n = len(X)
        if n == 0:
            return []
        k = max(1, min(k, n))
        rng = np.random.default_rng(seed)
        # k-means++ init
        centers = [X[rng.integers(n)]]
        for _ in range(k - 1):
            d2 = np.min(
                np.stack([((X - c) ** 2).sum(axis=1) for c in centers]), axis=0
            )
            probs = d2 / max(float(d2.sum()), 1e-12)
            centers.append(X[rng.choice(n, p=probs)])
        C = np.stack(centers)
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dists = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            new_labels = dists.argmin(axis=1)
            if (new_labels == labels).all():
                labels = new_labels
                break
            labels = new_labels
            for j in range(k):
                members = X[labels == j]
                if len(members):
                    C[j] = members.mean(axis=0)
                else:  # empty cluster -> grab farthest point
                    far = dists.min(axis=1).argmax()
                    C[j] = X[far]
        return labels.tolist()
    except ImportError:
        # pure-python fallback (small inputs only)
        n = len(data)
        if n == 0:
            return []
        k = max(1, min(k, n))
        rnd = random.Random(seed)
        centers = [list(data[i]) for i in rnd.sample(range(n), k)]

        def d2(a, b):
            return sum((x - y) ** 2 for x, y in zip(a, b))

        labels = [0] * n
        for _ in range(max_iter):
            changed = False
            for i, row in enumerate(data):
                best = min(range(k), key=lambda j: d2(row, centers[j]))
                if best != labels[i]:
                    labels[i] = best
                    changed = True
            for j in range(k):
                members = [data[i] for i in range(n) if labels[i] == j]
                if members:
                    centers[j] = [sum(col) / len(members) for col in zip(*members)]
            if not changed:
                break
        return labels


def auto_k(n: int) -> int:
    return max(2, min(40, int(math.sqrt(n / 2)) or 2))


# -----------------------------
# Generators
# -----------------------------

def cluster_generator(cfg: ClusterConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    ok, backend = clip_available()
    if not ok:
        yield f"Clustering skipped: {backend}"
        return
    conn = manifest.connect(cfg.output_base)

    # 1) embed frames missing embeddings
    if cfg.reembed:
        conn.execute("DELETE FROM clip_embeds")
        conn.commit()
    rows = conn.execute(
        "SELECT f.frame_id, f.path FROM frames f "
        "LEFT JOIN clip_embeds e ON e.frame_id = f.frame_id "
        "WHERE f.status IN ('selected','packaged') AND e.frame_id IS NULL "
        "ORDER BY f.path"
    ).fetchall()

    embed_fn = None
    if rows:
        try:
            embed_fn, backend_name = _load_clip()
        except RuntimeError as exc:
            yield str(exc)
            return
        yield f"Embedding {len(rows)} frames with {backend_name}...\n"
        import numpy as np
        t0 = time.time()
        done = 0
        for i in range(0, len(rows), cfg.batch_size):
            chunk = rows[i:i + cfg.batch_size]
            ok_idx, feats = embed_fn([Path(r[1]) for r in chunk])
            payload = [
                (chunk[j][0], feats[m].astype(np.float32).tobytes(), now_iso())
                for m, j in enumerate(ok_idx)
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO clip_embeds(frame_id, embed, embedded_at) "
                "VALUES (?,?,?)", payload,
            )
            conn.commit()
            done += len(chunk)
            if done % (cfg.batch_size * 4) == 0 or done >= len(rows):
                rate = done / max(0.001, time.time() - t0)
                yield f"Embedded {done}/{len(rows)} ({rate:.1f} img/s)"
    else:
        yield "All selected frames already embedded (cache hit)."

    # 2) cluster
    import numpy as np
    data = conn.execute(
        "SELECT f.frame_id, e.embed FROM frames f "
        "JOIN clip_embeds e ON e.frame_id = f.frame_id "
        "WHERE f.status IN ('selected','packaged')"
    ).fetchall()
    if len(data) < 4:
        yield f"Only {len(data)} embedded frames - skipping clustering."
        return

    ids = [r[0] for r in data]
    X = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in data])
    k = cfg.k or auto_k(len(ids))
    yield f"\nClustering {len(ids)} frames into k={k} groups..."
    labels = kmeans(X, k, cfg.max_iter)

    conn.executemany(
        "UPDATE frames SET cluster_id=? WHERE frame_id=?",
        list(zip(labels, ids)),
    )
    conn.commit()

    sizes: dict[int, int] = {}
    for l in labels:
        sizes[l] = sizes.get(l, 0) + 1
    dist = ", ".join(f"c{c}:{n}" for c, n in sorted(sizes.items(), key=lambda x: -x[1]))
    yield (
        "\nCLUSTERING DONE.\n"
        f"Frames: {len(ids)} | clusters: {len(sizes)}\n"
        f"Sizes: {dist}\n"
        "\nPackaging will now spread selection across clusters automatically."
    )
