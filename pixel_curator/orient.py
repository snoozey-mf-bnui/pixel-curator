"""Orientation tool v2 — accurate, EXIF-safe, verify-after, fully reversible.

The old `insights rotation` misfired because it (a) read raw pixels with cv2,
ignoring EXIF orientation, so already-upright phone photos looked sideways;
(b) had no confidence gate, so tilted/profile faces got "corrected"; and
(c) never re-checked the result. This rewrite fixes all three and logs every
change to a JSON manifest so it can be undone exactly.

Pipeline per image:
  1. NORMALIZE EXIF  — open with the orientation tag applied (what you *see*),
     so detection works on the displayed image, not raw sideways pixels.
  2. DETECT          — InsightFace landmarks -> eyes->mouth axis. Only act when
     the detection is CONFIDENT and the axis is UNAMBIGUOUS (guards against
     rotating images that were already fine).
  3. ROTATE          — only if confidently non-upright.
  4. VERIFY          — re-detect; if it didn't actually become upright, REVERT.
  5. LOG             — append {path, detected, applied, verified, reason} to a
     JSON manifest. `undo` reverses every applied rotation exactly.

Design seam: `process_image` takes an injected `detector(pil_rgb) -> (kps,score)
| None`, so the math and file ops are unit-testable without InsightFace. The
real backend (`make_insightface_detector`) is lazy/optional and never raises.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
EXIF_ORIENTATION_TAG = 0x0112


@dataclass
class OrientConfig:
    min_det_score: float = 0.55     # reject low-confidence detections
    min_axis_ratio: float = 1.25    # eyes->mouth axis must be clearly vertical/horizontal
    apply: bool = False             # dry-run by default
    verify: bool = True             # re-check after rotating; revert if not upright
    jpeg_quality: int = 95


# ---------------------------------------------------------------------------
# Pure orientation math (testable with synthetic landmarks)
# ---------------------------------------------------------------------------

def _eyes_mouth_axis(kps) -> tuple:
    """(dx, dy) of the eyes->mouth vector. kps = 5 InsightFace points:
    [left_eye, right_eye, nose, left_mouth, right_mouth]."""
    ex = (kps[0][0] + kps[1][0]) / 2.0
    ey = (kps[0][1] + kps[1][1]) / 2.0
    mx = (kps[3][0] + kps[4][0]) / 2.0
    my = (kps[3][1] + kps[4][1]) / 2.0
    return mx - ex, my - ey


def _upright_ccw_from_axis(dx: float, dy: float) -> int:
    """CCW degrees to make the face upright (mouth below eyes). 0 = upright."""
    if abs(dy) >= abs(dx):
        return 0 if dy > 0 else 180
    return 270 if dx < 0 else 90


def decide_rotation(kps, det_score: float, cfg: OrientConfig) -> tuple:
    """Return (ccw_degrees, reason). 0 = leave it alone. Only returns a nonzero
    rotation when the detection is confident AND the axis is unambiguous - this
    is the guard that stops already-correct images from being rotated."""
    if det_score is not None and det_score < cfg.min_det_score:
        return 0, "low_confidence"
    dx, dy = _eyes_mouth_axis(kps)
    adx, ady = abs(dx), abs(dy)
    ratio = max(adx, ady) / max(min(adx, ady), 1e-6)
    if ratio < cfg.min_axis_ratio:
        return 0, "ambiguous_axis"          # face tilted ~45deg: don't risk it
    rot = _upright_ccw_from_axis(dx, dy)
    return rot, ("upright" if rot == 0 else f"rotate_{rot}")


# ---------------------------------------------------------------------------
# File ops (EXIF normalize, rotate)
# ---------------------------------------------------------------------------

def load_exif_normalized(path):
    """Open an image with its EXIF orientation baked into the pixels (what the
    user sees). Returns (rgb_image, exif_was_nontrivial, original_orientation)."""
    from PIL import Image, ImageOps
    im = Image.open(path)
    try:
        orient = im.getexif().get(EXIF_ORIENTATION_TAG, 1) or 1
    except Exception:
        orient = 1
    fixed = ImageOps.exif_transpose(im)     # applies tag, strips it
    return fixed.convert("RGB"), (orient != 1), int(orient)


def read_exif_orientation(path) -> int:
    """Canonical EXIF orientation tag read (1 = upright/none). Never raises.

    Single source of truth so callers don't re-open / re-parse the tag inline."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.getexif().get(EXIF_ORIENTATION_TAG, 1) or 1)
    except Exception:
        return 1


def exif_normalized_rgb(path):
    """Canonical read helper: return an EXIF-normalized RGB ``PIL.Image`` (the
    orientation the user sees). Thin wrapper over ``load_exif_normalized`` so
    curation modules share one orientation path instead of duplicating
    ``ImageOps.exif_transpose`` inline. Returns ``None`` on failure (never
    raises)."""
    try:
        rgb, _, _ = load_exif_normalized(path)
        return rgb
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return None


def _save(img, path: Path, cfg: OrientConfig) -> None:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        img.save(path, format="JPEG", quality=cfg.jpeg_quality)
    elif suffix == ".png":
        img.save(path, format="PNG")
    else:
        img.save(path)


def _rotate_file(path: Path, ccw: int, cfg: OrientConfig) -> None:
    from PIL import Image
    if ccw % 360 == 0:
        return
    img = Image.open(path).convert("RGB").rotate(ccw, expand=True)
    _save(img, path, cfg)


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

Detector = Callable[[object], tuple | None]   # pil_rgb -> (kps, score) | None


def process_image(path, detector: Detector, cfg: OrientConfig) -> dict:
    """Detect orientation and (when apply=True) make it upright, EXIF-safe and
    verified. Returns a change record. Never raises on a single bad image."""
    p = Path(path)
    rec = {"path": str(p), "exif_orientation": 1, "exif_normalized": False,
           "detected": 0, "applied": 0, "verified": None, "reverted": False,
           "reason": ""}
    try:
        img, exif_changed, orient = load_exif_normalized(p)
        rec["exif_orientation"] = orient
    except Exception as exc:
        rec["reason"] = f"open_failed: {exc}"
        return rec

    det = None
    try:
        det = detector(img)
    except Exception:
        det = None
    if det is None:
        rec["reason"] = "no_face"
        # still normalize EXIF (benign, makes raw pixels match what you see)
        if cfg.apply and exif_changed:
            try:
                _save(img, p, cfg); rec["exif_normalized"] = True
            except Exception as exc:
                rec["reason"] = f"save_failed: {exc}"
        return rec

    kps, score = det
    rot, reason = decide_rotation(kps, score, cfg)
    rec["detected"] = rot
    rec["reason"] = reason

    if not cfg.apply:
        return rec                                   # dry-run: report only

    # write the EXIF-normalized (and optionally rotated) image
    try:
        out = img.rotate(rot, expand=True) if rot else img
        if exif_changed or rot:
            _save(out, p, cfg)
            rec["exif_normalized"] = exif_changed
    except Exception as exc:
        rec["reason"] = f"save_failed: {exc}"
        return rec

    if rot == 0:
        return rec

    # verify-after: re-detect; if not upright now, revert the rotation
    if cfg.verify:
        try:
            re_img, _, _ = load_exif_normalized(p)
            re_det = detector(re_img)
            ok = re_det is not None and decide_rotation(re_det[0], re_det[1], cfg)[0] == 0
        except Exception:
            ok = False
        if not ok:
            _rotate_file(p, (360 - rot) % 360, cfg)  # undo the rotation
            rec["applied"] = 0
            rec["verified"] = False
            rec["reverted"] = True
            rec["reason"] = "reverted_unverified"
            return rec
        rec["verified"] = True

    rec["applied"] = rot
    return rec


# ---------------------------------------------------------------------------
# Batch run + manifest
# ---------------------------------------------------------------------------

def run_orientation(paths, detector: Detector, cfg: OrientConfig, *,
                    manifest_dir: Path | None = None) -> dict:
    """Process a list of image paths. Writes a JSON manifest when apply=True.
    Returns a summary dict with the manifest path."""
    changes = []
    summary = {"scanned": 0, "rotated": 0, "reverted": 0, "no_face": 0,
               "ambiguous": 0, "low_confidence": 0, "exif_only": 0}
    for path in paths:
        rec = process_image(path, detector, cfg)
        changes.append(rec)
        summary["scanned"] += 1
        if rec["applied"]:
            summary["rotated"] += 1
        if rec["reverted"]:
            summary["reverted"] += 1
        if rec["reason"] == "no_face":
            summary["no_face"] += 1
        elif rec["reason"] == "ambiguous_axis":
            summary["ambiguous"] += 1
        elif rec["reason"] == "low_confidence":
            summary["low_confidence"] += 1
        if rec["exif_normalized"] and not rec["applied"]:
            summary["exif_only"] += 1

    manifest = {
        "tool": "orient", "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "apply": cfg.apply,
        "config": {"min_det_score": cfg.min_det_score,
                   "min_axis_ratio": cfg.min_axis_ratio, "verify": cfg.verify},
        "summary": summary, "changes": changes,
    }
    manifest_path = None
    if cfg.apply:
        md = Path(manifest_dir or (Path.cwd() / "orient_manifests"))
        try:
            md.mkdir(parents=True, exist_ok=True)
            manifest_path = md / f"orient_{datetime.now():%Y%m%d_%H%M%S}.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception:
            manifest_path = None
    manifest["manifest_path"] = str(manifest_path) if manifest_path else None
    return manifest


def undo_from_manifest(manifest_path, cfg: OrientConfig | None = None) -> dict:
    """Reverse every applied rotation recorded in a manifest. Exact inverse:
    a frame rotated +R is rotated back by (360-R). (EXIF normalization, being
    benign, is intentionally not reversed.)"""
    cfg = cfg or OrientConfig()
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    reverted = 0
    errors = []
    for ch in data.get("changes", []):
        applied = ch.get("applied", 0)
        if applied in (90, 180, 270):
            try:
                _rotate_file(Path(ch["path"]), (360 - applied) % 360, cfg)
                reverted += 1
            except Exception as exc:
                errors.append(f"{ch['path']}: {exc}")
    return {"reverted": reverted, "errors": errors,
            "from": str(manifest_path)}


def render_summary(manifest: dict) -> str:
    s = manifest["summary"]
    mode = "APPLIED" if manifest["apply"] else "DRY-RUN"
    lines = ["", f"Orientation pass [{mode}]", "=" * 56,
             f"scanned        : {s['scanned']}",
             f"rotated upright: {s['rotated']}",
             f"reverted (failed verify): {s['reverted']}",
             f"left alone     : ambiguous={s['ambiguous']} "
             f"low_conf={s['low_confidence']} no_face={s['no_face']}",
             f"exif-normalized only: {s['exif_only']}"]
    if not manifest["apply"]:
        would = sum(1 for c in manifest["changes"] if c["detected"])
        lines.append(f"\n{would} image(s) would be rotated. Re-run with --apply "
                     "to do it (a JSON manifest is written for exact undo).")
    elif manifest.get("manifest_path"):
        lines.append(f"\nmanifest: {manifest['manifest_path']}")
        lines.append("undo with:  ... orient --undo " + manifest["manifest_path"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# InsightFace backend (lazy, optional)
# ---------------------------------------------------------------------------

def make_insightface_detector(prj=None):
    """Return (detector, reason). detector(pil_rgb)->(kps,score)|None when
    InsightFace is available, else (None, reason)."""
    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:
        return None, f"InsightFace unavailable ({exc}); install 'pixel-curator[ai]'"
    try:
        from .runtime import onnx_providers
        providers = onnx_providers()
    except Exception:
        providers = None
    state = {"app": None}

    def detect(pil_rgb):
        try:
            if state["app"] is None:
                app = (FaceAnalysis(name="buffalo_l", providers=providers)
                       if providers else FaceAnalysis(name="buffalo_l"))
                app.prepare(ctx_id=0, det_size=(640, 640))
                state["app"] = app
            import numpy as np
            bgr = np.asarray(pil_rgb)[:, :, ::-1]
            faces = state["app"].get(bgr)
            if not faces:
                return None
            faces.sort(key=lambda f: getattr(f, "det_score", 0.0), reverse=True)
            return faces[0].kps, float(getattr(faces[0], "det_score", 1.0))
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            return None

    return detect, None
