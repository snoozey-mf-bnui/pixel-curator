"""Semantic text-query filtering (CLIP text encoder vs cached image embeds).

Uses the same CLIP backends as ``curate.diversity`` (open_clip preferred,
transformers fallback) but encodes *text* and scores every embedded frame
by cosine similarity. Unlocks:

  * keep-only filtering  - "keep only beach shots"
  * exclusion filtering  - "drop anything with text/watermarks" (--invert)

Dry-run by default: prints the similarity distribution and how many frames
would be dropped. ``apply=True`` sets non-matching frames' status to
``dropped`` (reversible via ``rescore`` / status resets). Frames without a
cached embedding are reported, never silently dropped - run ``cluster``
first to embed the selected set.
"""
from __future__ import annotations

import logging

log = logging.getLogger("pixel_curator.curate.semantic")

_TEXT_CACHE = None
_TEXT_ERR: str | None = None


def _load_text_encoder():
    """Returns (encode_text_fn(list[str]) -> np.ndarray, backend_name)."""
    global _TEXT_CACHE, _TEXT_ERR
    if _TEXT_CACHE is not None:
        return _TEXT_CACHE
    if _TEXT_ERR is not None:
        raise RuntimeError(_TEXT_ERR)
    try:
        import torch

        from .diversity import _device
        dev = _device()
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k")
            model = model.to(dev).eval()
            tokenizer = open_clip.get_tokenizer("ViT-B-32")

            def encode_text(texts: list[str]):
                with torch.no_grad():
                    feats = model.encode_text(tokenizer(texts).to(dev))
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                return feats.float().cpu().numpy()

            _TEXT_CACHE = (encode_text, f"open_clip ViT-B-32 [{dev}]")
            return _TEXT_CACHE
        except ImportError:
            from transformers import CLIPModel, CLIPProcessor
            model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32").to(dev).eval()
            proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            def encode_text(texts: list[str]):
                inputs = proc(text=texts, return_tensors="pt",
                              padding=True).to(dev)
                with torch.no_grad():
                    feats = model.get_text_features(**inputs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                return feats.float().cpu().numpy()

            _TEXT_CACHE = (encode_text, f"transformers CLIP [{dev}]")
            return _TEXT_CACHE
    except Exception as exc:
        _TEXT_ERR = (f"CLIP text encoder unavailable: {exc}\n"
                     "Install:  pip install 'pixel-curator[cluster]'")
        raise RuntimeError(_TEXT_ERR)


def score_frames_by_text(conn, query: str,
                         statuses=("selected", "packaged")) -> dict:
    """Cosine-similarity of ``query`` against every cached frame embedding.

    Returns {ok, query, backend, scores: [(frame_id, sim)], embedded,
    total_selected, warning?}. Never raises for missing embeddings."""
    rows = conn.execute(
        f"SELECT f.frame_id, e.embed FROM frames f "
        f"LEFT JOIN clip_embeds e ON e.frame_id = f.frame_id "
        f"WHERE f.status IN ({','.join('?' * len(statuses))})",
        tuple(statuses)).fetchall()
    total = len(rows)
    embedded = [(fid, emb) for fid, emb in rows if emb is not None]
    if not embedded:
        return {"ok": False, "query": query, "scores": [], "embedded": 0,
                "total_selected": total,
                "warning": "no cached CLIP embeddings; run "
                           "'pixel-curator cluster' first"}
    try:
        encode_text, backend = _load_text_encoder()
    except RuntimeError as exc:
        return {"ok": False, "query": query, "scores": [],
                "embedded": len(embedded), "total_selected": total,
                "warning": str(exc)}

    import numpy as np
    q = encode_text([query])[0]
    ids = [fid for fid, _ in embedded]
    X = np.stack([np.frombuffer(emb, dtype=np.float32)
                  for _, emb in embedded])
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)
    sims = (X @ q).tolist()
    scores = sorted(zip(ids, (float(s) for s in sims)),
                    key=lambda t: -t[1])
    return {"ok": True, "query": query, "backend": backend,
            "scores": scores, "embedded": len(embedded),
            "total_selected": total}


def apply_text_filter(conn, query: str, threshold: float = 0.25,
                      invert: bool = False, apply: bool = False) -> dict:
    """Keep-only (default) or drop-matching (--invert) text filter.

    threshold: minimum cosine similarity to count as a match (CLIP text
    sims typically land in 0.15-0.35; 0.25 is a sane default).
    """
    res = score_frames_by_text(conn, query)
    if not res.get("ok"):
        return res
    scores = res["scores"]
    matching = [(fid, s) for fid, s in scores if s >= threshold]
    matching_ids = {fid for fid, _ in matching}
    to_drop = [fid for fid, _ in scores
               if (fid in matching_ids) == bool(invert)]

    dist = {}
    for _, s in scores:
        b = min(9, max(0, int(s * 10)))
        dist[f"0.{b}"] = dist.get(f"0.{b}", 0) + 1

    out = {"ok": True, "query": query, "threshold": threshold,
           "invert": bool(invert), "apply": bool(apply),
           "matched": len(matching), "would_drop": len(to_drop),
           "embedded": res["embedded"],
           "total_selected": res["total_selected"],
           "similarity_histogram": dict(sorted(dist.items())),
           "top_matches": [{"frame_id": f, "similarity": round(s, 4)}
                           for f, s in matching[:10]]}
    if apply and to_drop:
        from ..util import now_iso
        conn.executemany(
            "UPDATE frames SET status='dropped', scored_at=? "
            "WHERE frame_id=?",
            [(now_iso(), fid) for fid in to_drop])
        conn.commit()
        out["dropped"] = len(to_drop)
    return out
