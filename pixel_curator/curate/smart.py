"""Smart curation: face detection + identity filtering (Phase 2, first slice).

Backend: InsightFace 'buffalo_l' bundle —
  - SCRFD-10G face detector
  - ArcFace R50 (w600k) identity embedding
running on onnxruntime with CoreML EP on Apple Silicon, CPU fallback.

Install:  pip install 'pixel-curator[ai]'   (insightface, onnxruntime, opencv, numpy)
Models auto-download to ~/.insightface/models/buffalo_l on first run.

Pattern credit: framing thresholds adapted from the focal-point logic in
reForge modules/textual_inversion/autocrop.py (face-weighted POI), upgraded
from YuNet/Haar to SCRFD + identity embeddings.


Workflow:
  1. pixel-curator anchor --anchor-dir /path/to/15_reference_images
  2. pixel-curator curate --smart
Frames already 'selected' by basic curation are scanned; frames with no
subject, tiny faces, or wrong identity are demoted with explicit statuses.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from pathlib import Path

from .. import manifest
from ..config import SmartCurateConfig
from ..sidecar import build_sidecar_index, subject_present
from ..util import now_iso, setup_logging

logger = logging.getLogger(__name__)

_BACKEND = None
_BACKEND_ERR: str | None = None


def _load_backend(det_size: int = 640):
    """Lazy global InsightFace app. Returns (app, numpy, cv2_or_none) or raises."""
    global _BACKEND, _BACKEND_ERR
    if _BACKEND is not None:
        return _BACKEND
    if _BACKEND_ERR is not None:
        raise RuntimeError(_BACKEND_ERR)
    try:
        import numpy as np
        from insightface.app import FaceAnalysis
        try:
            import cv2
        except Exception:
            cv2 = None
        from ..runtime import onnx_providers
        providers = onnx_providers()       # CUDA -> CoreML -> CPU
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0, det_size=(det_size, det_size))
        _BACKEND = (app, np, cv2)
        return _BACKEND
    except Exception as exc:
        _BACKEND_ERR = (
            f"Smart curation unavailable: {exc}\n"
            "Install extras:  pip install 'pixel-curator[ai]'"
        )
        raise RuntimeError(_BACKEND_ERR)


def smart_available() -> tuple[bool, str]:
    try:
        import insightface  # noqa: F401
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        return True, "ok"
    except Exception as exc:
        return False, f"missing [ai] extras: {exc}"


def _read_bgr(path: Path, np, cv2):
    if cv2 is not None:
        img = cv2.imread(str(path))
        if img is not None:
            return img
    try:
        from PIL import Image
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        return rgb[:, :, ::-1].copy()  # RGB -> BGR
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return None


def _read_bgr_exif_aware(path: Path, np, cv2) -> tuple:
    """EXIF-normalized BGR read for the smart pass (Phase 3).

    cv2.imread ignores the EXIF orientation tag, so an upright-displayed
    phone photo used to reach the detector sideways - triggering the slow
    3-rotation rescue or a false no-face rejection. Route EXIF-tagged
    images through orient.py's normalizer (the single orientation brain)
    so detection always sees what the user sees.

    Returns ``(bgr_image_or_None, exif_was_nontrivial)``. Images without
    an EXIF rotation (the overwhelmingly common case) take one cheap tag
    read and then the exact same fast path as before.
    """
    try:
        from ..orient import exif_normalized_rgb, read_exif_orientation
        if read_exif_orientation(path) != 1:
            rgb = exif_normalized_rgb(path)
            if rgb is not None:
                return np.asarray(rgb)[:, :, ::-1].copy(), True
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    return _read_bgr(path, np, cv2), False


from .framing import classify_framing as _classify_framing  # one shared rule

# -----------------------------
# Anchor (subject identity)
# -----------------------------

def build_anchor(cfg: SmartCurateConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    if not cfg.anchor_dir or not Path(cfg.anchor_dir).exists():
        yield "FATAL: --anchor-dir must point to a folder of subject reference images."
        return
    app, np, cv2 = _load_backend(cfg.det_size)
    conn = manifest.connect(cfg.output_base)

    embeds = []
    refs = sorted(
        p for p in Path(cfg.anchor_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    yield f"Building identity anchor from {len(refs)} reference images..."
    for ref in refs:
        img = _read_bgr(ref, np, cv2)
        if img is None:
            yield f"  skip (unreadable): {ref.name}"
            continue
        faces = app.get(img)
        if not faces:
            yield f"  skip (no face): {ref.name}"
            continue
        # largest face wins
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        embeds.append(face.normed_embedding)
        yield f"  ok: {ref.name} (det {face.det_score:.2f})"

    if len(embeds) < 3:
        yield f"FATAL: only {len(embeds)} usable references; need at least 3 (5-15 ideal)."
        return

    anchor = np.mean(np.stack(embeds), axis=0)
    anchor = anchor / np.linalg.norm(anchor)
    manifest.meta_set(conn, "identity_anchor", anchor.tolist())
    manifest.meta_set(conn, "identity_anchor_info", {
        "built_at": now_iso(), "refs_used": len(embeds),
        "anchor_dir": str(cfg.anchor_dir),
    })
    # self-consistency: how tight is the reference set?
    sims = [float(np.dot(anchor, e)) for e in embeds]
    yield (
        f"\nANCHOR BUILT from {len(embeds)} refs.\n"
        f"Self-similarity: min {min(sims):.3f} / mean {sum(sims)/len(sims):.3f}\n"
        f"(threshold {cfg.identity_threshold} should sit comfortably below the min)\n"
        "Next: pixel-curator curate --smart"
    )


# -----------------------------
# Smart pass over selected frames
# -----------------------------

def _upright_rotation_from_kps(kps) -> int:
    """CCW degrees needed to make the face upright (0/90/180/270) from
    the eyes->mouth axis.  0 means already upright."""
    ex, ey = (kps[0][0] + kps[1][0]) / 2, (kps[0][1] + kps[1][1]) / 2
    mx, my = (kps[3][0] + kps[4][0]) / 2, (kps[3][1] + kps[4][1]) / 2
    dx, dy = mx - ex, my - ey
    if abs(dy) >= abs(dx):
        return 0 if dy > 0 else 180
    # mouth left of eyes -> face's down points left -> rotate 270 CCW
    return 270 if dx < 0 else 90


def _rotate_bgr(img, ccw_degrees: int, cv2):
    code = {90: cv2.ROTATE_90_COUNTERCLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_CLOCKWISE}[ccw_degrees]
    return cv2.rotate(img, code)


def _make_orient_cfg(cfg: SmartCurateConfig):
    """OrientConfig from the smart-curation gates (defaults are identical)."""
    from ..orient import OrientConfig
    return OrientConfig(
        min_det_score=getattr(cfg, "orient_min_det_score", 0.55),
        min_axis_ratio=getattr(cfg, "orient_min_axis_ratio", 1.25))


def _orient_decision(kps, det_score: float, ocfg=None) -> tuple:
    """Confidence + ambiguity-gated rotation decision, shared with orient.py.
    Returns (ccw_degrees, reason); (0, ...) = leave alone. Stops the curation
    pass from over-rotating already-correct or merely-tilted faces."""
    try:
        from ..orient import OrientConfig, decide_rotation
        return decide_rotation(kps, det_score, ocfg or OrientConfig())
    except Exception:
        return 0, "orient_unavailable"


def _write_orient_manifest_safe(cfg: SmartCurateConfig, ocfg,
                                changes: list[dict], n_rotated: int,
                                n_exif_norm: int) -> str:
    """Persist an orient.py-compatible undo manifest for rotations applied
    during the smart pass. Returns a summary line ('' on failure); never
    raises - curation results must not depend on audit logging."""
    try:
        import json as _json
        from datetime import datetime
        md = Path(cfg.output_base) / "orient_manifests"
        md.mkdir(parents=True, exist_ok=True)
        payload = {
            "tool": "orient", "version": 1, "source": "curate_smart",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "apply": True,
            "config": {
                "min_det_score": getattr(ocfg, "min_det_score", 0.55),
                "min_axis_ratio": getattr(ocfg, "min_axis_ratio", 1.25),
                "verify": True},
            "summary": {"scanned": len(changes), "rotated": n_rotated,
                        "reverted": 0, "exif_only": n_exif_norm},
            "changes": changes,
        }
        path = md / f"orient_curate_{datetime.now():%Y%m%d_%H%M%S}.json"
        path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        return (f"Orientation manifest: {path}\n"
                f"  (undo with: pixel-curator orient --undo {path})\n")
    except Exception:
        return ""


def smart_curate_generator(cfg: SmartCurateConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    ok, reason = smart_available()
    if not ok:
        yield f"Smart curation skipped: {reason}"
        return
    from .performance import PerfProfiler, detect_compute_device, validate_det_size
    det_size, det_warnings = validate_det_size(cfg.det_size)
    for w in det_warnings:
        yield f"Performance note: {w}"
    profiler = PerfProfiler(enabled=getattr(cfg, "profile_performance", False))
    device_info = detect_compute_device()

    app, np, cv2 = _load_backend(det_size)
    conn = manifest.connect(cfg.output_base)

    anchor_list = manifest.meta_get(conn, "identity_anchor")
    anchor = np.asarray(anchor_list, dtype=np.float32) if anchor_list else None
    if anchor is None:
        yield ("No identity anchor found - running face-presence filtering only.\n"
               "Build one with: pixel-curator anchor --anchor-dir <refs>  "
               "to also reject wrong-identity frames.\n")

    # Optional Google Takeout sidecar assist (subject face tags).
    sidecar_idx = None
    if cfg.metadata_dir and cfg.subject_names:
        sidecar_idx = build_sidecar_index(cfg.metadata_dir)
        for w in sidecar_idx.warnings:
            yield f"Sidecar warning: {w}"
        yield (f"Sidecar metadata: {sidecar_idx.parsed} tags parsed from "
               f"{sidecar_idx.count} JSONs | subject={list(cfg.subject_names)} | "
               f"identity relax -{cfg.sidecar_identity_relax} when subject confirmed\n")

    if cfg.rescan:
        rows = conn.execute(
            "SELECT f.frame_id, f.path, s.path FROM frames f "
            "LEFT JOIN sources s ON s.source_id = f.source_id "
            "WHERE f.status IN ('selected','rejected_noface','rejected_smallface',"
            "'rejected_identity') ORDER BY f.path"
        ).fetchall()
        conn.executemany(
            "UPDATE frames SET status='selected' WHERE frame_id=?",
            [(r[0],) for r in rows],
        )
        conn.commit()
    else:
        rows = conn.execute(
            "SELECT f.frame_id, f.path, s.path FROM frames f "
            "LEFT JOIN sources s ON s.source_id = f.source_id "
            "LEFT JOIN detections d ON d.frame_id = f.frame_id "
            "WHERE f.status='selected' AND d.frame_id IS NULL ORDER BY f.path"
        ).fetchall()

    if not rows:
        yield "Nothing to scan (no selected frames without detections). Use --rescan to redo."
        return

    yield (f"Smart scan: {len(rows)} frames | backend buffalo_l "
           f"(SCRFD det + ArcFace id) | anchor: {'yes' if anchor is not None else 'no'}\n")

    n_noface = n_small = n_wrongid = n_kept = n_rescued = 0
    n_rotated = n_exif_norm = 0
    orient_changes: list[dict] = []
    ocfg = None
    try:
        ocfg = _make_orient_cfg(cfg)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    t0 = time.time()

    for i, (frame_id, path_str, source_path) in enumerate(rows, start=1):
        profiler.frame_begin()
        with profiler.stage("image_load_ms"):
            img, exif_norm = _read_bgr_exif_aware(Path(path_str), np, cv2)
        if img is None:
            conn.execute(
                "UPDATE frames SET status='decode_failed' WHERE frame_id=?", (frame_id,)
            )
            profiler.frame_end()
            continue
        ih, iw = img.shape[:2]
        with profiler.stage("face_detection_ms"):
            faces = app.get(img)

        # orientation correction: landmark check on detected faces, and
        # 90/180/270 rescue when the detector finds nothing (SCRFD is
        # rotation-sensitive - sideways faces read as no-face). Decisions
        # are confidence + ambiguity gated via orient.decide_rotation on
        # BOTH paths, and detection runs on EXIF-normalized pixels.
        rotated_by = 0
        _orient_ctx = profiler.stage("orientation_ms")
        _orient_ctx.__enter__()
        if getattr(cfg, "auto_rotate", True) and cv2 is not None:
            if faces:
                best = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
                rot, _reason = _orient_decision(
                    best.kps, float(getattr(best, "det_score", 1.0)), ocfg)
                if rot:
                    cand = _rotate_bgr(img, rot, cv2)   # don't commit until verified
                    new_faces = app.get(cand)
                    if new_faces:
                        nb = max(new_faces, key=lambda f: getattr(f, "det_score", 0.0))
                        # verify-after: only keep the rotation when the result
                        # is confidently upright (not merely ungated 0)
                        nrot, nreason = _orient_decision(
                            nb.kps, float(getattr(nb, "det_score", 1.0)), ocfg)
                        if nrot == 0 and nreason == "upright":
                            img, faces, rotated_by = cand, new_faces, rot
                            ih, iw = img.shape[:2]
            else:
                for rot in (90, 180, 270):
                    cand = _rotate_bgr(img, rot, cv2)
                    new_faces = app.get(cand)
                    if not new_faces:
                        continue
                    nb = max(new_faces, key=lambda f: getattr(f, "det_score", 0.0))
                    # same gates as the primary path: a low-confidence or
                    # ambiguous detection on a rotated candidate must not
                    # rotate the frame (the old check was ungated)
                    nrot, nreason = _orient_decision(
                        nb.kps, float(getattr(nb, "det_score", 1.0)), ocfg)
                    if nrot == 0 and nreason == "upright":
                        img, faces, rotated_by = cand, new_faces, rot
                        ih, iw = img.shape[:2]
                        break
        # persist: rotations, and benign EXIF normalization (raw pixels now
        # match what every EXIF-unaware downstream reader sees)
        if (rotated_by or exif_norm) and cv2 is not None \
                and getattr(cfg, "auto_rotate", True):
            try:
                cv2.imwrite(path_str, img)
                if rotated_by:
                    n_rotated += 1
                if exif_norm:
                    n_exif_norm += 1
                orient_changes.append({
                    "path": path_str, "exif_normalized": bool(exif_norm),
                    "detected": rotated_by, "applied": rotated_by,
                    "verified": bool(rotated_by) or None, "reverted": False,
                    "reason": ("rotate_%d" % rotated_by if rotated_by
                               else "exif_normalized")})
            except Exception:
                rotated_by = 0
        _orient_ctx.__exit__(None, None, None)

        if not faces:
            conn.execute(
                "INSERT OR REPLACE INTO detections(frame_id, face_count, face_area, "
                "det_conf, identity_sim, framing, detected_at) VALUES (?,0,0,0,NULL,'none',?)",
                (frame_id, now_iso()),
            )
            conn.execute(
                "UPDATE frames SET status='rejected_noface' WHERE frame_id=?", (frame_id,)
            )
            n_noface += 1
        else:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            bw = float(face.bbox[2] - face.bbox[0])
            bh = float(face.bbox[3] - face.bbox[1])
            area_ratio = (bw * bh) / float(iw * ih)
            framing = _classify_framing(bh / ih)
            sim = float(np.dot(anchor, face.normed_embedding)) if anchor is not None else None

            import json as _json
            bbox_norm = [
                max(0.0, float(face.bbox[0]) / iw), max(0.0, float(face.bbox[1]) / ih),
                min(1.0, float(face.bbox[2]) / iw), min(1.0, float(face.bbox[3]) / ih),
            ]
            conn.execute(
                "INSERT OR REPLACE INTO detections(frame_id, face_count, face_area, "
                "det_conf, identity_sim, framing, embed, detected_at, bbox) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (frame_id, len(faces), area_ratio, float(face.det_score), sim,
                 framing, face.normed_embedding.astype(np.float32).tobytes(),
                 now_iso(), _json.dumps(bbox_norm)),
            )

            # Sidecar assist: if Google tagged the subject in this frame's
            # source, relax the identity threshold so a good frame at a bad
            # angle is not falsely demoted as wrong-identity.
            subject_ok = False
            if sidecar_idx is not None and source_path:
                subject_ok = subject_present(
                    sidecar_idx.lookup(source_path), cfg.subject_names)
            eff_thr = cfg.identity_threshold - (
                cfg.sidecar_identity_relax if subject_ok else 0.0)

            if area_ratio < cfg.min_face_area:
                conn.execute(
                    "UPDATE frames SET status='rejected_smallface' WHERE frame_id=?",
                    (frame_id,),
                )
                n_small += 1
            elif sim is not None and sim < eff_thr:
                conn.execute(
                    "UPDATE frames SET status='rejected_identity' WHERE frame_id=?",
                    (frame_id,),
                )
                n_wrongid += 1
            else:
                if subject_ok and sim is not None and sim < cfg.identity_threshold:
                    n_rescued += 1   # would have been cut without the face tag
                n_kept += 1

        profiler.frame_end()
        if i % cfg.batch_log_every == 0 or i == len(rows):
            with profiler.stage("manifest_write_ms"):
                conn.commit()
            rate = i / max(0.001, time.time() - t0)
            yield (f"Scanned {i}/{len(rows)} ({rate:.1f} img/s) | kept {n_kept} | "
                   f"no-face {n_noface} | small {n_small} | wrong-id {n_wrongid}")
    conn.commit()

    framing_counts = dict(conn.execute(
        "SELECT d.framing, COUNT(*) FROM detections d "
        "JOIN frames f ON f.frame_id = d.frame_id "
        "WHERE f.status='selected' GROUP BY d.framing"
    ).fetchall())

    # orientation undo manifest: same JSON format as the standalone `orient`
    # command, so `pixel-curator orient --undo <manifest>` reverses this pass
    orient_manifest_line = ""
    if orient_changes:
        orient_manifest_line = _write_orient_manifest_safe(
            cfg, ocfg, orient_changes, n_rotated, n_exif_norm)

    rescued_line = (
        f"Rescued by subject face tag: {n_rescued}\n" if n_rescued else "")
    orient_line = (
        f"Rotated upright: {n_rotated} | EXIF-normalized: {n_exif_norm}\n"
        if (n_rotated or n_exif_norm) else "")
    yield (
        "\nSMART CURATION DONE.\n"
        f"Kept (subject confirmed): {n_kept}\n"
        f"{rescued_line}"
        f"{orient_line}"
        f"Rejected no-face: {n_noface}\n"
        f"Rejected small-face: {n_small}\n"
        f"Rejected wrong-identity: {n_wrongid}\n"
        f"Framing mix of survivors: {framing_counts}\n"
        f"{orient_manifest_line}"
        "\nNext: pixel-curator package"
    )

    if profiler.enabled:
        import json as _json
        summary = profiler.summary(det_size=det_size,
                                   workers=getattr(cfg, "workers", None),
                                   batch_size=None,
                                   device=device_info.get("selected_device"))
        try:
            manifest.meta_set(conn, "curation_performance", summary)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass
        yield ("\nCURATION PERFORMANCE PROFILE\n"
               f"device: {device_info.get('selected_device')} "
               f"(mps={device_info.get('mps_available')}, "
               f"cuda={device_info.get('cuda_available')}) | det_size {det_size}\n"
               + _json.dumps(summary, indent=2))
