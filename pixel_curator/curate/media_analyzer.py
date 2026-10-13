"""Media analysis stage - turn a raw archive into classifiable samples.

The phased pipeline's DatasetSplitter routes images by metadata (quality,
sharpness, exposure, face visibility, framing, identity similarity). A raw
Google Photos Takeout has none of that, so without analysis every image lands
in ``needs_review``. This module computes the missing signals locally and
non-destructively, producing DatasetSplitter-ready sample dicts.

Signals (all local, privacy-preserving):
  * sharpness / brightness / quality_score  - PIL + Laplacian (always)
  * face visibility + framing                - InsightFace (when [ai] extras
    are installed); otherwise omitted with a clear warning
  * identity_similarity vs ``anchor_refs/``  - ArcFace cosine vs a mean anchor,
    calibrated to 0..1 so a solid same-person match clears the classifier's
    core_identity threshold
  * garment_visibility                       - conservative framing heuristic

Heavy: face detection over a large archive takes time, so analysis supports a
``limit`` and a resumable JSON cache. Never moves or modifies source media.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from . import basic, smart

log = logging.getLogger("redresser.media_analyzer")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tiff")

# normalization references (Laplacian variance on 96x96 gray; brightness 0-255)
_SHARP_REF = 200.0
# ArcFace cosine calibration: cos<=LO -> 0.0, cos>=HI -> 1.0
_ID_LO, _ID_HI = 0.20, 0.55


def _norm_sharpness(sharp: float) -> float:
    return max(0.0, min(float(sharp) / _SHARP_REF, 1.0))


def _norm_brightness(bright: float) -> float:
    return max(0.0, min(float(bright) / 255.0, 1.0))


def _quality_score(sharp_norm: float, bright_norm: float) -> float:
    exposure = max(0.0, 1.0 - abs(bright_norm - 0.5) * 1.6)
    return round(0.55 * sharp_norm + 0.45 * exposure, 4)


def _norm_identity(cosine: float) -> float:
    return round(max(0.0, min((cosine - _ID_LO) / (_ID_HI - _ID_LO), 1.0)), 4)


_FRAMING_MAP = {                       # smart._classify_framing -> classifier
    "closeup": ("face_closeup", "clear", 0.20),
    "portrait": ("portrait", "clear", 0.35),
    "upper_body": ("half_body", "visible", 0.55),
    "full_body": ("full_body", "partial", 0.60),
}

# Secondary routing thresholds (heuristic; framing-driven, no extra models).
_DETAIL_SHARP = 0.85                    # crisp body shots -> garment-detail study
_STYLE_QUALITY = 0.6                    # aesthetic floor for a style reference
_STYLE_IDENTITY_MAX = 0.7              # below the core_identity bar


def apply_secondary_signals(meta: dict) -> dict:
    """Derive pose_type / style_reference / garment_detail tag from the base
    framing+quality signals so the phased classifier can fill pose_action,
    style_aesthetic and garment_detail instead of dumping every body shot into
    outfit_clothing. Pure, in-place, never raises.

    Heuristics (documented, not a learned model):
      * whole-body framing            -> pose_type='standing'   (pose_action)
      * crisp body shot (sharp>=0.85) -> 'garment_detail' tag   (garment_detail)
      * strong-quality close shot that is NOT a clear identity match
                                      -> style_reference=True    (style_aesthetic)
    """
    framing = str(meta.get("framing", ""))
    sharp = float(meta.get("sharpness", 0.0) or 0.0)
    quality = float(meta.get("quality_score", 0.0) or 0.0)
    ident = float(meta.get("identity_similarity", 0.0) or 0.0)
    tags = set(meta.get("tags", []) or [])

    if framing == "full_body":
        meta.setdefault("pose_type", "standing")
    if framing in ("full_body", "half_body") and sharp >= _DETAIL_SHARP:
        tags.add("garment_detail")
    if (framing in ("face_closeup", "portrait")
            and quality >= _STYLE_QUALITY and ident < _STYLE_IDENTITY_MAX):
        meta["style_reference"] = True
    if tags:
        meta["tags"] = sorted(tags)
    return meta


def build_identity_anchor(anchor_dir, det_size: int = 640
                          ) -> tuple[object, dict, list[str]]:
    """Build a mean ArcFace identity anchor from reference images.

    Returns ``(anchor_vec_or_None, info, warnings)``. Requires the [ai] extras;
    without them returns ``(None, {}, [warning])`` so analysis still runs
    (quality-only). Never raises.
    """
    warnings: list[str] = []
    anchor_dir = Path(anchor_dir).expanduser() if anchor_dir else None
    if not anchor_dir or not anchor_dir.is_dir():
        return None, {}, [f"Anchor directory not found: {anchor_dir}; identity "
                          "similarity will be unavailable."]
    ok, reason = smart.smart_available()
    if not ok:
        return None, {}, [f"Identity backend unavailable ({reason}); install "
                          "'pixel-curator[ai]' to enable identity matching."]
    try:
        app, np, cv2 = smart._load_backend(det_size)
        embeds = []
        refs = sorted(p for p in anchor_dir.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        for ref in refs:
            img = smart._read_bgr(ref, np, cv2)
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0])
                       * (f.bbox[3] - f.bbox[1]))
            embeds.append(face.normed_embedding)
        if len(embeds) < 3:
            return None, {}, [f"Only {len(embeds)} usable anchor reference(s); "
                              "need >=3 for reliable identity matching."]
        anchor = np.mean(np.stack(embeds), axis=0)
        anchor = anchor / np.linalg.norm(anchor)
        sims = [float(np.dot(anchor, e)) for e in embeds]
        info = {"refs_used": len(embeds),
                "self_similarity_min": round(min(sims), 3),
                "self_similarity_mean": round(sum(sims) / len(sims), 3)}
        return anchor, info, warnings
    except Exception as exc:                         # pragma: no cover
        return None, {}, [f"Anchor build failed: {exc}"]


def analyze_image(path, anchor=None, det_size: int = 640,
                  backend=None) -> dict:
    """Compute classification metadata for one image. Never raises.

    Returns a metadata dict with ``quality_score``, ``sharpness``,
    ``brightness`` and (when a face backend is available) ``face_visibility``,
    ``framing``, ``identity_similarity``, ``garment_visibility``.
    """
    path = Path(str(path)).expanduser()
    meta: dict = {"source_type": "image"}
    warnings: list[str] = []

    scored = None
    try:
        scored = basic.score_frame(str(path))        # (dhash, sharp, bright)
    except Exception as exc:                         # pragma: no cover
        warnings.append(f"quality scoring failed: {exc}")
    if scored is None:
        meta["corrupt"] = True
        meta["warnings"] = warnings or ["unreadable image"]
        return meta
    _dh, sharp, bright = scored
    s_norm = _norm_sharpness(sharp)
    b_norm = _norm_brightness(bright)
    meta["sharpness"] = round(s_norm, 4)
    meta["brightness"] = round(b_norm, 4)
    meta["quality_score"] = _quality_score(s_norm, b_norm)
    meta["dhash"] = _dh

    # face / framing / identity (optional backend)
    if backend is not None:
        app, np, cv2 = backend
        try:
            img = smart._read_bgr(path, np, cv2)
            faces = app.get(img) if img is not None else []
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0])
                           * (f.bbox[3] - f.bbox[1]))
                fh = float(face.bbox[3] - face.bbox[1])
                ih = float(img.shape[0]) if img is not None else fh
                ratio = (fh / ih) if ih else 0.0
                base_framing = smart._classify_framing(ratio)
                framing, face_vis, garment_vis = _FRAMING_MAP[base_framing]
                meta["framing"] = framing
                meta["face_visibility"] = face_vis
                meta["garment_visibility"] = garment_vis
                if anchor is not None:
                    cos = float(np.dot(anchor, face.normed_embedding))
                    meta["identity_cosine"] = round(cos, 4)
                    meta["identity_similarity"] = _norm_identity(cos)
            else:
                meta["face_visibility"] = "none"
                meta["framing"] = "full_body"
                meta["garment_visibility"] = 0.6
        except Exception as exc:                     # pragma: no cover
            warnings.append(f"face analysis failed: {exc}")
    # derive pose / style / garment-detail routing signals from framing+quality
    apply_secondary_signals(meta)
    if warnings:
        meta["warnings"] = warnings
    return meta


def iter_images(raw_path: Path, limit: int = 0):
    """Yield image paths under ``raw_path`` (recursive), capped at ``limit``."""
    raw_path = Path(raw_path).expanduser()
    n = 0
    for p in sorted(raw_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p
            n += 1
            if limit and n >= limit:
                return


def analyze_archive(raw_path, anchor_dir=None, limit: int = 0,
                    cache_path=None, resume: bool = True, det_size: int = 640,
                    progress: Callable[[int, int], None] | None = None
                    ) -> dict:
    """Analyze a raw archive into DatasetSplitter-ready samples.

    Builds the identity anchor (when [ai] extras + ``anchor_dir`` are present),
    scores each image, and emits ``{sample_path, source_type, metadata}`` rows.
    Resumable via a JSON cache: previously analyzed paths are reused. Returns
    ``{samples, count, analyzed, reused, identity_available, anchor_info,
    cache_path, warnings}``. Never moves or modifies source media.
    """
    raw_path = Path(raw_path).expanduser()
    warnings: list[str] = []
    if not raw_path.is_dir():
        return {"samples": [], "count": 0, "analyzed": 0, "reused": 0,
                "identity_available": False, "anchor_info": {},
                "cache_path": "", "warnings": [f"Raw path not found: {raw_path}"]}

    anchor, anchor_info, awarn = build_identity_anchor(anchor_dir, det_size)
    warnings += awarn
    identity_available = anchor is not None
    backend = None
    if smart.smart_available()[0]:
        try:
            backend = smart._load_backend(det_size)
        except Exception as exc:                     # pragma: no cover
            warnings.append(f"Face backend unavailable: {exc}")
    else:
        warnings.append(
            "Face/identity backend unavailable; analysis is quality-only. "
            "Install 'pixel-curator[ai]' for face, framing, and identity "
            "matching (core_identity routing).")

    # resumable cache
    cache: dict[str, dict] = {}
    cache_path = Path(cache_path).expanduser() if cache_path else None
    if resume and cache_path and cache_path.is_file():
        try:
            cache = {r["sample_path"]: r for r in
                     json.loads(cache_path.read_text())}
        except Exception:
            cache = {}

    paths = list(iter_images(raw_path, limit=limit))
    total = len(paths)
    samples: list[dict] = []
    analyzed = reused = 0
    for i, p in enumerate(paths):
        key = str(p)
        if key in cache:
            samples.append(cache[key])
            reused += 1
        else:
            meta = analyze_image(p, anchor=anchor, det_size=det_size,
                                 backend=backend)
            row = {"sample_path": key, "source_type": "image",
                   "metadata": meta}
            samples.append(row)
            analyzed += 1
        if progress and (i % 200 == 0 or i + 1 == total):
            progress(i + 1, total)

    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(samples, indent=2, default=str),
                                  encoding="utf-8")
        except Exception as exc:                     # pragma: no cover
            warnings.append(f"Could not write analysis cache: {exc}")

    return {"samples": samples, "count": total, "analyzed": analyzed,
            "reused": reused, "identity_available": identity_available,
            "anchor_info": anchor_info,
            "cache_path": str(cache_path) if cache_path else "",
            "warnings": warnings}


__all__ = [
    "IMAGE_EXTENSIONS", "build_identity_anchor", "analyze_image",
    "iter_images", "analyze_archive", "apply_secondary_signals",
]
