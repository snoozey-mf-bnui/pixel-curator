"""Deep quality signals for personal figure work (Priority 2).

Lightweight, numpy-only image-quality metrics computed during curation. They go
beyond whole-image sharpness to score *subject integrity* and *composition*,
which is what matters for high-fidelity identity training:

  Subject integrity   face-region sharpness, micro-detail, detail retention
                      (over-smoothing/waxiness), occlusion confidence, eyes-open
  Composition         subject centering, headroom, frontalness, pose bucket
  Artifacts           JPEG 8x8 blockiness (compression damage)

Plus dataset-level aggregates: subject-scale consistency (a body-proportion
proxy), skin-texture/detail consistency, and pose diversity.

All metric functions are pure (numpy in, number out) so they're unit-testable
without InsightFace or real files. Geometry metrics take a *pixel* bbox; the
curation pass denormalizes the stored detection bbox and feeds it in.
"""
from __future__ import annotations

import logging
import math
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)

METRIC_VERSION = 1          # bump when a metric formula changes -> triggers recompute
_SHARP_REF = 120.0          # face-sharpness normalization midpoint

# approximate insightface 2d106 eyelid indices (left, right). Best-effort: used
# only by the optional --with-face pass; guarded, and easy to retune here.
_EYE_106_LEFT = (35, 36, 33, 37, 39, 42)
_EYE_106_RIGHT = (89, 90, 87, 91, 93, 96)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def to_gray(rgb) -> np.ndarray:
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim == 2:
        return a
    a = a[..., :3]
    return a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _crop(img, bbox):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = int(max(0, min(x0, w - 1)))
    x1 = int(max(x0 + 1, min(x1, w)))
    y0 = int(max(0, min(y0, h - 1)))
    y1 = int(max(y0 + 1, min(y1, h)))
    return img[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# subject integrity
# ---------------------------------------------------------------------------

def laplacian_variance(gray) -> float:
    g = np.asarray(gray, dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def face_region_sharpness(gray, bbox_px) -> float | None:
    """Sharpness of the FACE region only - the whole image can be crisp while
    the face is soft, which is what actually matters for identity."""
    if bbox_px is None:
        return None
    reg = _crop(gray, bbox_px)
    return laplacian_variance(reg) if reg.size >= 16 else None


def micro_detail(gray, bbox_px=None) -> float:
    """Mean gradient magnitude = fine texture energy (skin pores, hair)."""
    g = _crop(gray, bbox_px) if bbox_px is not None else gray
    if g.size < 4:
        return 0.0
    gx = float(np.abs(np.diff(g, axis=1)).mean())
    gy = float(np.abs(np.diff(g, axis=0)).mean())
    return (gx + gy) / 2.0


def detail_retention(gray, bbox_px=None) -> float | None:
    """High-frequency energy relative to overall contrast. Low -> waxy /
    over-smoothed / beauty-filtered. 0..1, higher = more real detail kept."""
    g = _crop(gray, bbox_px) if bbox_px is not None else gray
    if g.size < 9:
        return None
    lap_std = laplacian_variance(g) ** 0.5
    contrast = float(g.std()) + 1e-6
    return float(min(1.0, lap_std / contrast))


def compression_blockiness(gray) -> float:
    """JPEG 8x8 blockiness: energy at block boundaries vs interior. 0 = clean,
    higher = more compression artifacts."""
    g = np.asarray(gray, dtype=np.float32)
    h, w = g.shape
    if w < 16 or h < 16:
        return 0.0
    cols = np.arange(8, w, 8)
    rows = np.arange(8, h, 8)
    bound = (float(np.abs(g[:, cols] - g[:, cols - 1]).mean())
             + float(np.abs(g[rows, :] - g[rows - 1, :]).mean())) / 2.0
    interior = (float(np.abs(np.diff(g, axis=1)).mean())
                + float(np.abs(np.diff(g, axis=0)).mean())) / 2.0 + 1e-6
    return float(max(0.0, bound / interior - 1.0))


def occlusion_quality(det_score, face_area_frac=None) -> float | None:
    """Cheap occlusion/framing proxy from detection confidence (+ a penalty for
    tiny faces that usually mean far/occluded framing). 0..1, higher = cleaner."""
    if det_score is None:
        return None
    s = float(det_score)
    if face_area_frac is not None and face_area_frac < 0.003:
        s *= 0.7
    return float(min(1.0, max(0.0, s)))


def eye_aspect_ratio(eye_pts) -> float:
    p = np.asarray(eye_pts, dtype=np.float32)
    width = float(p[:, 0].max() - p[:, 0].min()) + 1e-6
    height = float(p[:, 1].max() - p[:, 1].min())
    return height / width


def eyes_open_score(left_eye_pts, right_eye_pts, threshold=0.18) -> float | None:
    """1.0 ~ wide open, <1 squint/closed. Needs eyelid landmark points; returns
    None when unavailable (graceful)."""
    if left_eye_pts is None or right_eye_pts is None:
        return None
    ear = (eye_aspect_ratio(left_eye_pts) + eye_aspect_ratio(right_eye_pts)) / 2.0
    return float(min(1.5, ear / threshold))


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

def centering_score(bbox_px, w, h) -> float:
    cx = (bbox_px[0] + bbox_px[2]) / 2.0
    cy = (bbox_px[1] + bbox_px[3]) / 2.0
    dx = abs(cx - w / 2.0) / (w / 2.0)
    dy = abs(cy - h / 2.0) / (h / 2.0)
    return float(max(0.0, 1.0 - (dx + dy) / 2.0))


def headroom_score(bbox_px, w, h) -> float:
    """Space above the head. Triangular, peaking at ~10% headroom; 0 when the
    head is cropped at the top or there's too much dead space."""
    frac = bbox_px[1] / float(h)
    if frac <= 0 or frac >= 0.35:
        return 0.0
    return float(max(0.0, 1.0 - abs(frac - 0.10) / 0.15))


def frontalness(pose) -> float | None:
    if pose is None:
        return None
    pitch, yaw = float(pose[0]), float(pose[1])
    return float(max(0.0, 1.0 - (abs(yaw) + abs(pitch)) / 90.0))


def pose_bucket(pose) -> str | None:
    if pose is None:
        return None
    yaw = abs(float(pose[1]))
    return "frontal" if yaw < 15 else "three_quarter" if yaw < 45 else "profile"


# ---------------------------------------------------------------------------
# per-frame orchestrator
# ---------------------------------------------------------------------------

def _norm_sharp(v):
    return None if v is None else float(v / (v + _SHARP_REF))


def analyze_frame(rgb, *, bbox_px=None, det_score=None, face_area_frac=None,
                  eyes=None, pose=None) -> dict:
    """Compute all signals for one frame. `eyes` = (left_pts, right_pts) or None;
    `pose` = (pitch, yaw, roll) or None. Returns metrics + an `integrity` score."""
    gray = to_gray(rgb)
    h, w = gray.shape[:2]
    out = {
        "whole_sharpness": laplacian_variance(gray),
        "face_sharpness": face_region_sharpness(gray, bbox_px),
        "micro_detail": micro_detail(gray, bbox_px),
        "detail_retention": detail_retention(gray, bbox_px),
        "blockiness": compression_blockiness(gray),
        "centering": centering_score(bbox_px, w, h) if bbox_px is not None else None,
        "headroom": headroom_score(bbox_px, w, h) if bbox_px is not None else None,
        "occlusion": occlusion_quality(det_score, face_area_frac),
        "eyes_open": eyes_open_score(*eyes) if eyes else None,
        "frontalness": frontalness(pose),
        "pose_bucket": pose_bucket(pose),
    }
    # integrity = mean of the available 0..1 subject/composition signals
    parts = [_norm_sharp(out["face_sharpness"]), out["detail_retention"],
             out["occlusion"], out["centering"], out["headroom"]]
    parts = [p for p in parts if p is not None]
    out["integrity"] = round(float(np.mean(parts)), 4) if parts else None
    # round the rest for storage
    for k in ("whole_sharpness", "face_sharpness", "micro_detail",
              "detail_retention", "blockiness", "centering", "headroom",
              "occlusion", "eyes_open", "frontalness"):
        if isinstance(out[k], float):
            out[k] = round(out[k], 4)
    return out


# ---------------------------------------------------------------------------
# dataset-level aggregates
# ---------------------------------------------------------------------------

def scale_consistency(face_area_fracs) -> dict:
    """Body-proportion proxy: subject scale across frames. Big variance is fine
    for diversity, but extreme outliers (a face 10x bigger/smaller) flag bad
    framing. Returns mean/std and z-score threshold helper data."""
    a = np.asarray([x for x in face_area_fracs if x is not None], dtype=float)
    if a.size < 3:
        return {"n": int(a.size), "mean": None, "std": None}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std())}


def skin_texture_consistency(detail_scores) -> dict:
    """Distribution of detail-retention. A low-mean / wide spread means some
    frames are over-smoothed relative to others (inconsistent skin texture)."""
    a = np.asarray([x for x in detail_scores if x is not None], dtype=float)
    if a.size < 3:
        return {"n": int(a.size), "mean": None, "std": None}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "low_detail_frac": float((a < 0.25).mean())}


def pose_diversity(buckets) -> dict:
    """Entropy-based pose-angle diversity (0=all same angle, 1=balanced across
    frontal/three_quarter/profile). Encourages varied body angles."""
    c = Counter(b for b in buckets if b)
    total = sum(c.values())
    if not total:
        return {"distribution": {}, "diversity": 0.0}
    dist = {k: v / total for k, v in c.items()}
    H = -sum(p * math.log(p) for p in dist.values() if p > 0)
    return {"distribution": dict(c),
            "diversity": round(float(H / math.log(3)), 4)}


# ---------------------------------------------------------------------------
# Curation pass (DB) - computes + stores per-frame metrics, returns aggregates
# ---------------------------------------------------------------------------

def _parse_bbox(raw):
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return list(raw)
    s = str(raw).strip()
    import json
    try:
        return json.loads(s)
    except Exception:
        try:
            import ast
            return list(ast.literal_eval(s))
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            return None


def _default_load(path):
    # Route through the canonical orientation path (orient.exif_normalized_rgb)
    # so curation reads images upright exactly like the rest of the pipeline.
    try:
        from ..orient import exif_normalized_rgb
        rgb = exif_normalized_rgb(path)
        return np.asarray(rgb) if rgb is not None else None
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return None


def _eyes_from_106(landmark_2d_106):
    """Best-effort eyelid point extraction from insightface 106 landmarks.
    Guarded - returns None if the array/indices don't line up."""
    if landmark_2d_106 is None:
        return None
    try:
        lm = np.asarray(landmark_2d_106, dtype=np.float32)
        if lm.ndim != 2 or lm.shape[0] < 97:
            return None
        return lm[list(_EYE_106_LEFT)], lm[list(_EYE_106_RIGHT)]
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return None


def make_face_detector(prj=None):
    """Lazy InsightFace detector for the --with-face pass: returns per-image
    {bbox_px, eyes, pose, det_score, face_area_frac} or None. Returns
    (detector, reason); detector is None when InsightFace is unavailable."""
    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:
        return None, f"InsightFace unavailable ({exc}); install 'pixel-curator[ai]'"
    try:
        from ..runtime import onnx_providers
        providers = onnx_providers()
    except Exception:
        providers = None
    state = {"app": None}

    def detect(rgb):
        try:
            if state["app"] is None:
                app = (FaceAnalysis(name="buffalo_l", providers=providers)
                       if providers else FaceAnalysis(name="buffalo_l"))
                app.prepare(ctx_id=0, det_size=(640, 640))
                state["app"] = app
            arr = np.asarray(rgb)
            faces = state["app"].get(arr[:, :, ::-1])
            if not faces:
                return None
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            h, w = arr.shape[:2]
            bw, bh = float(f.bbox[2] - f.bbox[0]), float(f.bbox[3] - f.bbox[1])
            pose = getattr(f, "pose", None)
            return {"bbox_px": [float(f.bbox[0]), float(f.bbox[1]),
                                float(f.bbox[2]), float(f.bbox[3])],
                    "eyes": _eyes_from_106(getattr(f, "landmark_2d_106", None)),
                    "pose": tuple(float(x) for x in pose) if pose is not None else None,
                    "det_score": float(getattr(f, "det_score", 1.0)),
                    "face_area_frac": (bw * bh) / float(w * h)}
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            return None

    return detect, None



def _score_one_process(job):
    """Process-pool worker: score one frame without face detection.

    Runs in a fresh process; only uses picklable inputs and the default
    EXIF-aware loader. Returns ``(fid, m, det_conf, face_area, bbox_px,
    w, h)`` with ``m=None`` for unreadable frames."""
    fid, path, bbox_raw, det_conf, face_area = job
    rgb = _default_load(path)
    if rgb is None or np.asarray(rgb).size == 0:
        return (fid, None, det_conf, face_area, None, 0, 0)
    rgb = np.asarray(rgb)
    h, w = rgb.shape[:2]
    bn = _parse_bbox(bbox_raw)
    bbox_px = [bn[0] * w, bn[1] * h, bn[2] * w, bn[3] * h] if (
        bn and len(bn) == 4) else None
    m = analyze_frame(rgb, bbox_px=bbox_px, det_score=det_conf,
                      face_area_frac=face_area)
    return (fid, m, det_conf, face_area, bbox_px, w, h)


def analyze_quality_pass(conn, *, load_image=None, face_detector=None,
                         advanced=None, statuses=("selected", "packaged"),
                         limit: int = 0, store: bool = True,
                         resume: bool = False, progress: bool = False,
                         commit_every: int = 400, workers: int = 1) -> dict:
    """Score every curated frame's subject-integrity / composition / artifacts.

    By default (no face_detector) it reuses the stored detection bbox - fast,
    landmark-free. With a face_detector (the --with-face pass), it re-detects to
    also populate eyes-open, frontalness and pose bucket. Scores SELECTED /
    PACKAGED frames only (rejected frames are skipped). Upserts frame_quality
    keyed by frame_id with metric_version + calculated_at.

    resume=True skips frames already scored at the current signal version (so a
    Ctrl-C'd run picks up where it left off). progress=True prints a throttled
    live status line to stderr. Work is committed every ``commit_every`` frames
    and on interrupt, so resume always has durable progress to build on.

    workers>1 parallelizes scoring across processes (completion order is
    then nondeterministic); it activates only in the plain CPU path -
    face-detector, advanced-signal and custom-loader passes stay
    sequential."""
    import sys
    import time

    from ..util import now_iso
    custom_loader = load_image is not None
    load_image = load_image or _default_load
    qs = ",".join("?" for _ in statuses)
    rows = conn.execute(
        f"SELECT f.frame_id, f.path, d.bbox, d.det_conf, d.face_area "
        f"FROM frames f LEFT JOIN detections d ON d.frame_id = f.frame_id "
        f"WHERE f.status IN ({qs})", statuses).fetchall()
    if limit:
        rows = rows[:limit]

    now = now_iso()
    face_pass = 1 if face_detector is not None else 0
    adv_on = advanced is not None and getattr(advanced, "enabled", lambda: False)()
    summary = {"scanned": 0, "scored": 0, "unreadable": 0, "face_pass": face_pass,
               "advanced": 1 if adv_on else 0, "advanced_signals": 0,
               "skipped_resume": 0, "interrupted": 0}

    # --- resume: drop frames already done at the current version ---
    if resume and store:
        try:
            if adv_on:
                from .advanced_quality import ADVANCED_SIGNAL_VERSION as _AV
                done_ids = {r[0] for r in conn.execute(
                    "SELECT frame_id FROM frame_advanced_quality "
                    "WHERE signal_version = ?", (_AV,))}
            else:
                done_ids = {r[0] for r in conn.execute(
                    "SELECT frame_id FROM frame_quality WHERE metric_version = ?",
                    (METRIC_VERSION,))}
            if done_ids:
                before = len(rows)
                rows = [r for r in rows if r[0] not in done_ids]
                summary["skipped_resume"] = before - len(rows)
        except Exception:
            pass  # resume is best-effort; fall back to full pass

    total = len(rows)
    t0 = time.time()
    last_print = 0.0

    def _emit(force=False):
        nonlocal last_print
        if not progress:
            return
        tnow = time.time()
        if not force and (tnow - last_print) < 1.0:
            return
        last_print = tnow
        done = summary["scored"]
        el = tnow - t0
        rate = done / el if el > 0 else 0.0
        from ..progress import fmt_duration, fmt_rate
        pct = (100.0 * done / total) if total else 0.0
        eta = (fmt_duration((total - done) / rate)
               if rate > 1e-6 else "--")
        sys.stderr.write(
            f"\r\033[2Kscoring {done}/{total} ({pct:.1f}%) | "
            f"{fmt_rate(rate)} | ETA {eta} | adv "
            f"{summary['advanced_signals']} | skipped "
            f"{summary['skipped_resume']}")
        sys.stderr.flush()

    adv_warnings: list = []
    face_areas, details, buckets, integrities = [], [], [], []
    since_commit = 0

    parallel = (workers > 1 and face_detector is None
                and not custom_loader and not adv_on)
    jobs = [(fid, path, bbox_raw, det_conf, face_area)
            for fid, path, bbox_raw, det_conf, face_area in rows]

    def _sequential(job):
        fid, path, bbox_raw, det_conf, face_area = job
        rgb = load_image(path)
        if rgb is None or np.asarray(rgb).size == 0:
            return (fid, None, det_conf, face_area, None, 0, 0, None)
        rgb = np.asarray(rgb)
        h, w = rgb.shape[:2]
        eyes = pose = None
        if face_detector is not None:
            fd = face_detector(rgb)
            if fd:
                bbox_px = fd["bbox_px"]
                det_conf = fd["det_score"]
                face_area = fd["face_area_frac"]
                eyes, pose = fd["eyes"], fd["pose"]
            else:
                bbox_px = None
        else:
            bn = _parse_bbox(bbox_raw)
            bbox_px = [bn[0] * w, bn[1] * h, bn[2] * w, bn[3] * h] if (
                bn and len(bn) == 4) else None
        m = analyze_frame(rgb, bbox_px=bbox_px, det_score=det_conf,
                          face_area_frac=face_area, eyes=eyes, pose=pose)
        return (fid, m, det_conf, face_area, bbox_px, w, h, rgb)

    def _results(pool):
        if pool is not None:
            from concurrent.futures import as_completed
            futs = [pool.submit(_score_one_process, job) for job in jobs]
            for fut in as_completed(futs):
                fid, m, det_conf, face_area, bbox_px, w, h = fut.result()
                yield (fid, m, det_conf, face_area, bbox_px, w, h, None)
        else:
            for job in jobs:
                yield _sequential(job)

    try:
        pool = None
        if parallel:
            from concurrent.futures import ProcessPoolExecutor
            pool = ProcessPoolExecutor(max_workers=workers)
        try:
            for (fid, m, det_conf, face_area, bbox_px,
                 w, h, rgb) in _results(pool):
                summary["scanned"] += 1
                if m is None:
                    summary["unreadable"] += 1
                    continue
                if store:
                    conn.execute(
                        "INSERT OR REPLACE INTO frame_quality(frame_id, face_sharpness, "
                        "micro_detail, detail_retention, blockiness, centering, headroom, "
                        "occlusion, frontalness, eyes_open, pose_bucket, integrity, "
                        "metric_version, face_pass, calculated_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (fid, m["face_sharpness"], m["micro_detail"], m["detail_retention"],
                         m["blockiness"], m["centering"], m["headroom"], m["occlusion"],
                         m["frontalness"], m["eyes_open"], m["pose_bucket"],
                         m["integrity"], METRIC_VERSION, face_pass, now))
                # optional advanced quality signals (sequential path only)
                if adv_on:
                    try:
                        from .advanced_quality import (
                            ADVANCED_SIGNAL_VERSION,
                            compute_advanced_signals,
                        )
                        img_bgr = rgb[:, :, ::-1] if rgb.ndim == 3 else rgb
                        sig, warns = compute_advanced_signals(
                            img_bgr, face_bbox=bbox_px, img_w=w, img_h=h,
                            base_metrics=m, det_conf=det_conf, cfg=advanced)
                        adv_warnings.extend(warns)
                        if sig:
                            summary["advanced_signals"] += 1
                            if store:
                                import json as _json
                                conn.execute(
                                    "INSERT OR REPLACE INTO frame_advanced_quality("
                                    "frame_id, signals_json, signal_version, "
                                    "calculated_at) VALUES (?,?,?,?)",
                                    (fid, _json.dumps(sig), ADVANCED_SIGNAL_VERSION, now))
                    except Exception as exc:
                        adv_warnings.append(f"advanced_quality_failed: {exc}")
                summary["scored"] += 1
                if face_area is not None:
                    face_areas.append(face_area)
                if m["detail_retention"] is not None:
                    details.append(m["detail_retention"])
                if m["pose_bucket"]:
                    buckets.append(m["pose_bucket"])
                if m["integrity"] is not None:
                    integrities.append(m["integrity"])
                # durable checkpoint so a Ctrl-C keeps everything done so far
                since_commit += 1
                if store and since_commit >= commit_every:
                    conn.commit()
                    since_commit = 0
                _emit()
        finally:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
    except KeyboardInterrupt:
        summary["interrupted"] = 1
        if store:
            conn.commit()
        _emit(force=True)
        if progress:
            sys.stderr.write(
                f"\n^C — committed {summary['scored']} scored this run. "
                f"Re-run with --resume to continue.\n")
            sys.stderr.flush()
        return {"summary": summary, "warnings": adv_warnings[:50],
                "aggregates": _quality_aggregates(
                    integrities, face_areas, details, buckets)}
    if store:
        conn.commit()
    _emit(force=True)
    if progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    return {"summary": summary, "warnings": adv_warnings[:50],
            "aggregates": _quality_aggregates(
                integrities, face_areas, details, buckets)}


def _quality_aggregates(integrities, face_areas, details, buckets) -> dict:
    return {
        "mean_integrity": round(float(np.mean(integrities)), 4) if integrities else None,
        "scale_consistency": scale_consistency(face_areas),
        "skin_texture_consistency": skin_texture_consistency(details),
        "pose_diversity": pose_diversity(buckets),
    }


def render_quality(res: dict) -> str:
    s = res["summary"]
    a = res["aggregates"]
    lines = ["", "Deep quality signals", "=" * 56,
             f"scanned : {s['scanned']}   scored : {s['scored']}   "
             f"unreadable : {s['unreadable']}"]
    if s.get("skipped_resume"):
        lines.append(f"resumed : skipped {s['skipped_resume']} already-scored "
                     f"frame(s) at current version")
    if s.get("interrupted"):
        lines.append("INTERRUPTED (Ctrl-C) — progress saved; re-run with "
                     "--resume to finish")
    lines.append(f"mean subject-integrity : {a['mean_integrity']}")
    sk = a["skin_texture_consistency"]
    if sk.get("mean") is not None:
        lines.append(f"detail retention       : mean {sk['mean']:.3f} "
                     f"std {sk['std']:.3f}  (over-smoothed frac "
                     f"{sk.get('low_detail_frac', 0):.0%})")
    sc = a["scale_consistency"]
    if sc.get("mean") is not None:
        lines.append(f"subject scale          : mean {sc['mean']:.4f} "
                     f"std {sc['std']:.4f}  (body-proportion proxy)")
    pd = a["pose_diversity"]
    lines.append(f"pose diversity         : {pd['diversity']}  "
                 f"{pd['distribution'] or '(no pose data — run with face pass)'}")
    if s.get("advanced"):
        lines.append(f"advanced signals       : {s.get('advanced_signals', 0)} frames "
                     "(figure/region/coverage/texture, stored in frame_advanced_quality)")
    for w in (res.get("warnings") or [])[:5]:
        lines.append(f"  ! {w}")
    return "\n".join(lines)
