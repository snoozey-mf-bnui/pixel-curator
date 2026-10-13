"""Generated-output likeness scorer - closes the QA loop with a number.

The dataset already carries an ArcFace identity embedding per frame in
``detections.embed`` (written by smart curation). This module builds a single
reference identity centroid from those embeddings, then scores generated
images by ArcFace cosine similarity to that centroid and writes the result
back into ``evals.likeness``. The outcome: epoch / merge selection stops being
a subjective eyeball test and becomes a ranked number.

Design seam: every function takes an injectable ``embedder`` callable
(image_path -> 512-d float vector or None) and/or a precomputed centroid, so
the DB and math are fully testable without InsightFace installed. The real
InsightFace backend (:func:`make_insightface_embedder`) is lazy and returns a
structured "unavailable" reason instead of raising - it must never crash a UI
or a batch run.
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

Embedder = Callable[[str], Optional["np.ndarray"]]

REFERENCE_STATUSES = ("selected", "packaged")


# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        return None
    return v / n


def cosine_likeness(emb: np.ndarray, centroid: np.ndarray) -> float:
    """ArcFace cosine, clamped to 0..1 (negative similarity -> 0)."""
    a, b = _unit(np.asarray(emb, dtype=np.float32)), _unit(centroid)
    if a is None or b is None:
        return 0.0
    return round(float(max(0.0, np.dot(a, b))), 4)


# ---------------------------------------------------------------------------
# Reference identity centroid (from dataset detections)
# ---------------------------------------------------------------------------

def load_reference_centroid(conn: sqlite3.Connection, *,
                            min_identity_sim: float = 0.0,
                            statuses: tuple = REFERENCE_STATUSES,
                            embed_dim: int = 512) -> dict:
    """Average the dataset's stored ArcFace embeddings into one unit centroid.

    Returns {"centroid": np.ndarray|None, "n": int, "warning": str|None}.
    Never raises."""
    try:
        has = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='detections'"
        ).fetchone()
        if not has:
            return {"centroid": None, "n": 0,
                    "warning": "no detections table - run smart curation first"}
        placeholders = ",".join("?" for _ in statuses)
        rows = conn.execute(
            f"SELECT d.embed FROM detections d "
            f"JOIN frames f ON f.frame_id = d.frame_id "
            f"WHERE d.embed IS NOT NULL AND f.status IN ({placeholders}) "
            f"AND (d.identity_sim IS NULL OR d.identity_sim >= ?)",
            (*statuses, float(min_identity_sim))).fetchall()
        vecs = []
        for (blob,) in rows:
            try:
                v = np.frombuffer(blob, dtype=np.float32)
                if v.size and (embed_dim is None or v.size == embed_dim):
                    u = _unit(v)
                    if u is not None:
                        vecs.append(u)
            except Exception:
                logger.debug("suppressed exception", exc_info=True)
                continue
        if not vecs:
            return {"centroid": None, "n": 0,
                    "warning": "no usable identity embeddings in detections "
                               "(run smart curation on selected frames)"}
        centroid = _unit(np.mean(np.stack(vecs), axis=0))
        return {"centroid": centroid, "n": len(vecs), "warning": None}
    except Exception as exc:
        return {"centroid": None, "n": 0,
                "warning": f"centroid build failed: {type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# InsightFace backend (lazy, optional, never raises)
# ---------------------------------------------------------------------------

def make_insightface_embedder(prj=None):
    """Return (embedder, reason). embedder is a callable(path)->vec|None when
    InsightFace is available, else (None, reason)."""
    try:
        import cv2
        from insightface.app import FaceAnalysis
    except Exception as exc:
        return None, (f"InsightFace/opencv unavailable ({exc}); "
                      "install 'pixel-curator[ai]'")
    try:
        from ..runtime import onnx_providers
        providers = onnx_providers()
    except Exception:
        providers = None
    state: dict = {"app": None}

    def embed(path: str):
        try:
            if state["app"] is None:
                app = FaceAnalysis(name="buffalo_l", providers=providers) \
                    if providers else FaceAnalysis(name="buffalo_l")
                app.prepare(ctx_id=0, det_size=(640, 640))
                state["app"] = app
            img = cv2.imread(str(path))
            if img is None:
                return None
            faces = state["app"].get(img)
            if not faces:
                return None
            faces.sort(key=lambda f: getattr(f, "det_score", 0.0), reverse=True)
            return np.asarray(faces[0].normed_embedding, dtype=np.float32)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            return None

    return embed, None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_evals(conn: sqlite3.Connection, *, embedder: Embedder,
                centroid: np.ndarray, lora: str | None = None,
                only_missing: bool = True, limit: int | None = None) -> dict:
    """Score eval rows (those with an image_path) and write evals.likeness.
    Returns a summary. Never raises on a single bad image."""
    where = ["image_path IS NOT NULL", "TRIM(image_path) <> ''"]
    params: list = []
    if only_missing:
        where.append("likeness IS NULL")
    if lora:
        where.append("lora = ?")
        params.append(lora)
    sql = "SELECT eval_id, image_path FROM evals WHERE " + " AND ".join(where)
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql, params).fetchall()

    scored = skipped = 0
    values = []
    for eval_id, path in rows:
        emb = None
        try:
            if Path(path).exists():
                emb = embedder(path)
        except Exception:
            emb = None
        if emb is None:
            skipped += 1
            continue
        lk = cosine_likeness(emb, centroid)
        conn.execute("UPDATE evals SET likeness=? WHERE eval_id=?", (lk, eval_id))
        values.append(lk)
        scored += 1
    conn.commit()
    mean = round(float(np.mean(values)), 4) if values else None
    return {"scored": scored, "skipped": skipped, "candidates": len(rows),
            "mean_likeness": mean,
            "min_likeness": round(min(values), 4) if values else None,
            "max_likeness": round(max(values), 4) if values else None}


def lora_likeness_summary(conn: sqlite3.Connection) -> list[dict]:
    """Per-LoRA aggregate of scored likeness, best first. Feeds the leaderboard."""
    rows = conn.execute(
        "SELECT lora, COUNT(likeness), AVG(likeness), MIN(likeness), "
        "MAX(likeness) FROM evals WHERE likeness IS NOT NULL "
        "GROUP BY lora").fetchall()
    out = []
    for lora, n, avg, lo, hi in rows:
        out.append({"lora": lora, "n": int(n),
                    "mean_likeness": round(float(avg), 4) if avg is not None else None,
                    "min_likeness": round(float(lo), 4) if lo is not None else None,
                    "max_likeness": round(float(hi), 4) if hi is not None else None})
    out.sort(key=lambda r: (r["mean_likeness"] is not None, r["mean_likeness"]),
             reverse=True)
    return out


def run_likeness(conn: sqlite3.Connection, *, prj=None, lora: str | None = None,
                 only_missing: bool = True, min_identity_sim: float = 0.0,
                 embedder: Embedder | None = None) -> dict:
    """Top-level: build centroid, get an embedder, score, summarize.
    Returns a structured report; emits a 'warning' instead of raising when the
    identity backend or reference embeddings are unavailable."""
    ref = load_reference_centroid(conn, min_identity_sim=min_identity_sim)
    if ref["centroid"] is None:
        return {"ok": False, "warning": ref["warning"], "n_reference": 0,
                "scored": 0}
    reason = None
    if embedder is None:
        embedder, reason = make_insightface_embedder(prj)
    if embedder is None:
        return {"ok": False, "warning": reason, "n_reference": ref["n"],
                "scored": 0}
    summary = score_evals(conn, embedder=embedder, centroid=ref["centroid"],
                          lora=lora, only_missing=only_missing)
    summary.update({"ok": True, "warning": None, "n_reference": ref["n"],
                    "by_lora": lora_likeness_summary(conn)})
    return summary


def render_likeness(report: dict) -> str:
    if not report.get("ok"):
        return ("Likeness scoring unavailable: "
                f"{report.get('warning') or 'unknown reason'}")
    lines = ["", "Generated-output likeness (ArcFace cosine vs dataset identity)",
             "=" * 62,
             f"reference frames: {report['n_reference']}  |  "
             f"scored: {report['scored']}  skipped(no face/file): "
             f"{report['skipped']}"]
    if report.get("mean_likeness") is not None:
        lines.append(f"this run: mean={report['mean_likeness']}  "
                     f"min={report['min_likeness']}  max={report['max_likeness']}")
    by = report.get("by_lora") or []
    if by:
        lines.append("\nper-LoRA (best first):")
        for r in by:
            lines.append(f"  {r['mean_likeness']!s:<7} mean  "
                         f"[{r['min_likeness']}-{r['max_likeness']}]  "
                         f"n={r['n']:<4} {r['lora']}")
    lines.append("\nHigher = closer to the trained identity (1.0 = identical "
                 "embedding). 0.0 means no face detected or no match.")
    return "\n".join(lines)
