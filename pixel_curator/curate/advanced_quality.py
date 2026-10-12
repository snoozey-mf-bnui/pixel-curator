"""Optional advanced quality signals - additive integration layer.

Wires the modular scorers (figure_quality, region_quality, natural_texture,
coverage_classifier) into the curation pipeline as *optional enhancement
signals*. Everything here is:

  * opt-in        - every flag defaults to False; disabled => no-op, no output
  * non-destructive - it only enriches scoring/diagnostics, never filters/deletes
  * fail-safe     - each signal is wrapped so one failure (or a missing optional
                    dependency like cv2) records a warning and continues; it can
                    never break the curation run
  * dependency-light - the scorer modules are imported lazily, only when enabled

`pose_diversity` is dataset-level and is integrated separately in diverse.py.
`anatomy_detail` duplicates `figure_quality`, so it isn't wired a second time.
"""
from __future__ import annotations

from dataclasses import dataclass

ADVANCED_SIGNAL_VERSION = 3  # v3: integer-cast background_quality bbox (cv2.rectangle
#                                  float-coord fix). v2 was integer-cast region/
#                                  aesthetic/lighting/expression crops. A future
#                                  `quality --resume` treats older rows as stale and
#                                  recomputes them, backfilling background_quality.


@dataclass
class AdvancedQualityConfig:
    """Per-signal opt-in. All default off => existing behavior is unchanged."""
    # batch 1 (figure / region / coverage / texture)
    figure_quality: bool = False
    region_quality: bool = False
    coverage: bool = False
    natural_texture: bool = False
    # batch 2 (dynamic pose / aesthetic / skin / symmetry / expression / lighting)
    dynamic_pose: bool = False
    aesthetic: bool = False
    skin_rendering: bool = False
    body_symmetry: bool = False
    expression_quality: bool = False
    lighting: bool = False
    # batch 3 (proportion / color mood / background / motion blur / face occlusion)
    body_proportion: bool = False
    color_mood: bool = False
    background_quality: bool = False
    motion_blur: bool = False
    face_occlusion: bool = False
    # batch 4 (clothing / full body / eye openness / color cast / dup / exposure)
    clothing_type: bool = False
    full_body: bool = False
    eye_openness: bool = False
    color_cast: bool = False
    duplicate_detection: bool = False
    exposure_outlier: bool = False
    # priority 6 (region-based readiness: face / torso / full-body aggregation)
    region_readiness: bool = False
    # priority 7 (coverage / framing intensity classification)
    coverage_intensity: bool = False
    # priority 8 (per-frame body-pose signature; clustering is dataset-level)
    pose_duplicate: bool = False

    def enabled(self) -> bool:
        return any((self.figure_quality, self.region_quality, self.coverage,
                    self.natural_texture, self.dynamic_pose, self.aesthetic,
                    self.skin_rendering, self.body_symmetry,
                    self.expression_quality, self.lighting,
                    self.body_proportion, self.color_mood,
                    self.background_quality, self.motion_blur,
                    self.face_occlusion, self.clothing_type, self.full_body,
                    self.eye_openness, self.color_cast,
                    self.duplicate_detection, self.exposure_outlier,
                    self.region_readiness, self.coverage_intensity,
                    self.pose_duplicate))

    @classmethod
    def from_flags(cls, *, advanced=False, advanced_aesthetic=False,
                   advanced_composition=False,
                   figure_quality=False, region_quality=False, coverage=False,
                   natural_texture=False, dynamic_pose=False, aesthetic=False,
                   skin_rendering=False, body_symmetry=False,
                   expression_quality=False, lighting=False,
                   body_proportion=False, color_mood=False,
                   background_quality=False, motion_blur=False,
                   face_occlusion=False, advanced_curation=False,
                   clothing_type=False, full_body=False, eye_openness=False,
                   color_cast=False, duplicate_detection=False,
                   exposure_outlier=False, region_readiness=False,
                   coverage_intensity=False, pose_duplicate=False,
                   **aliases) -> "AdvancedQualityConfig":
        # defensive: also accept `with_<name>` aliases if a caller passes them
        def _pick(name, val):
            return bool(val or aliases.get(f"with_{name}", False))

        c = cls(figure_quality, region_quality, coverage, natural_texture)
        if advanced:                       # batch-1 group (unchanged behavior)
            c.figure_quality = c.region_quality = True
            c.coverage = c.natural_texture = True
        c.dynamic_pose = _pick("dynamic_pose", dynamic_pose)
        c.aesthetic = _pick("aesthetic", aesthetic)
        c.skin_rendering = _pick("skin_rendering", skin_rendering)
        c.body_symmetry = _pick("body_symmetry", body_symmetry)
        c.expression_quality = _pick("expression_quality", expression_quality)
        c.lighting = _pick("lighting", lighting)
        if advanced_aesthetic:             # image-based aesthetic group
            c.aesthetic = c.skin_rendering = True
            c.expression_quality = c.lighting = True
        # batch-3
        c.body_proportion = _pick("body_proportion", body_proportion)
        c.color_mood = _pick("color_mood", color_mood)
        c.background_quality = _pick("background_quality", background_quality)
        c.motion_blur = _pick("motion_blur", motion_blur)
        c.face_occlusion = _pick("face_occlusion", face_occlusion)
        c.region_readiness = _pick("region_readiness", region_readiness)
        c.coverage_intensity = _pick("coverage_intensity", coverage_intensity)
        c.pose_duplicate = _pick("pose_duplicate", pose_duplicate)
        if advanced_composition:           # image-based composition group
            c.color_mood = c.background_quality = True
            c.motion_blur = c.face_occlusion = True
        # batch-4
        c.clothing_type = _pick("clothing_type", clothing_type)
        c.full_body = _pick("full_body", full_body)
        c.eye_openness = _pick("eye_openness", eye_openness)
        c.color_cast = _pick("color_cast", color_cast)
        c.duplicate_detection = _pick("duplicate_detection", duplicate_detection)
        c.exposure_outlier = _pick("exposure_outlier", exposure_outlier)
        if advanced_curation:              # image/metadata curation group
            c.clothing_type = c.full_body = True
            c.color_cast = c.exposure_outlier = True
        return c


def _crop(image, bbox):
    if image is None or bbox is None:
        return image
    h, w = image.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = max(0, min(x0, w - 1)); x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1)); y1 = max(y0 + 1, min(y1, h))
    return image[y0:y1, x0:x1]


def compute_advanced_signals(image_bgr, *, face_bbox=None, body_bbox=None,
                             img_w=None, img_h=None, base_metrics=None,
                             det_conf=None, keypoints=None, landmarks=None,
                             cfg: AdvancedQualityConfig = None) -> tuple:
    """Compute the enabled optional signals for one frame.

    Returns (signals_dict, warnings_list). `signals_dict` is manifest-safe
    (plain dicts/floats/strings). `image_bgr` is a numpy BGR/grayscale array;
    `base_metrics` is the P2 per-frame metric dict (reused so figure_quality
    gets the scalars it expects). Never raises."""
    signals: dict = {}
    warnings: list = []
    base_metrics = base_metrics or {}
    if cfg is None or not cfg.enabled():
        return signals, warnings

    region_face_q = None
    if cfg.region_quality:
        try:
            from . import region_quality as rq
            r = rq.compute_from_bboxes(image_bgr, face_bbox=face_bbox,
                                       body_bbox=body_bbox)
            signals["region_quality"] = r.to_dict()
            signals["regional_quality_score"] = rq.get_combined_regional_score(r)
            region_face_q = r.face_quality
        except Exception as exc:
            warnings.append(f"region_quality_failed: {exc}")

    if cfg.natural_texture:
        try:
            from . import natural_texture as nt
            t = nt.compute_natural_texture(image_bgr)
            signals["natural_texture"] = t.to_dict()
            signals["texture_preservation_score"] = nt.get_natural_detail_score(t)
        except Exception as exc:
            warnings.append(f"natural_texture_failed: {exc}")

    if cfg.coverage:
        try:
            from . import coverage_classifier as cc
            c = cc.classify_from_bboxes(face_bbox=face_bbox, body_bbox=body_bbox,
                                        img_width=int(img_w or 0),
                                        img_height=int(img_h or 0))
            signals["coverage"] = c.to_dict()
            signals["coverage_label"] = cc.get_framing_category(c)
        except Exception as exc:
            warnings.append(f"coverage_failed: {exc}")

    if cfg.figure_quality:
        try:
            from . import figure_quality as fq
            # face_sharpness for figure_quality must be 0..1: prefer the region
            # face quality; else normalize the raw P2 face_sharpness.
            fs_raw = base_metrics.get("face_sharpness")
            if region_face_q is not None:
                face_for_fig = region_face_q
            elif fs_raw is not None:
                face_for_fig = fq.normalize_face_sharpness(fs_raw)
            else:
                face_for_fig = None
            f = fq.compute_figure_quality(
                face_sharpness=face_for_fig,
                body_proportion=base_metrics.get("body_proportion"),
                occlusion=base_metrics.get("occlusion", det_conf),
                centering=base_metrics.get("centering"),
                headroom=base_metrics.get("headroom"))
            signals["figure_quality"] = f.to_dict()
            signals["figure_integrity_score"] = fq.get_integrity_score(f)
        except Exception as exc:
            warnings.append(f"figure_quality_failed: {exc}")

    # ----- batch 2 (image-based: skin / expression / aesthetic / lighting) -----
    if cfg.skin_rendering:
        try:
            from . import skin_rendering as sr
            patch = _crop(image_bgr, face_bbox) if face_bbox is not None else image_bgr
            s = sr.compute_skin_rendering_from_image(patch)
            signals["skin_rendering"] = s.to_dict()
            signals["skin_rendering_score"] = sr.get_skin_rendering_score(s)
        except Exception as exc:
            warnings.append(f"skin_rendering_failed: {exc}")

    if cfg.expression_quality:
        try:
            from . import expression_quality as eq
            s = eq.compute_expression_from_signals(
                image=image_bgr,
                face_bbox=tuple(face_bbox) if face_bbox is not None else None,
                landmarks=landmarks)
            signals["expression_quality"] = s.to_dict()
            signals["expression_score"] = s.overall_expression_score
        except Exception as exc:
            warnings.append(f"expression_quality_failed: {exc}")

    # aesthetic is also the real source of the lighting signal - compute once
    if cfg.aesthetic or cfg.lighting:
        try:
            from . import aesthetic_scorer as ae
            subj = (tuple(body_bbox) if body_bbox is not None
                    else (tuple(face_bbox) if face_bbox is not None else None))
            aes = ae.compute_aesthetic_from_image(image_bgr, subject_bbox=subj)
            if cfg.aesthetic:
                signals["aesthetic"] = aes.to_dict()
                signals["aesthetic_score"] = ae.get_aesthetic_score(aes)
            if cfg.lighting:
                # the standalone lighting_analysis module is a misnamed copy of
                # skin_rendering; real lighting analysis lives in aesthetic.
                signals["lighting"] = {"lighting_quality": aes.lighting_quality}
                signals["lighting_score"] = aes.lighting_quality
        except Exception as exc:
            if cfg.aesthetic:
                warnings.append(f"aesthetic_failed: {exc}")
            if cfg.lighting:
                warnings.append(f"lighting_failed: {exc}")

    # ----- keypoint-based: dynamic pose / body symmetry (skip cleanly if none) -----
    if cfg.dynamic_pose:
        if not keypoints:
            warnings.append("dynamic_pose_skipped: no body keypoints available")
        else:
            try:
                from . import dynamic_pose_scorer as dp
                s = dp.compute_dynamic_from_keypoints(keypoints)
                signals["dynamic_pose"] = s.to_dict()
                signals["dynamic_pose_score"] = dp.get_dynamic_pose_score(s)
            except Exception as exc:
                warnings.append(f"dynamic_pose_failed: {exc}")

    if cfg.body_symmetry:
        if not keypoints:
            warnings.append("body_symmetry_skipped: no body keypoints available")
        else:
            try:
                from . import body_symmetry as bs
                s = bs.compute_symmetry_from_keypoints(keypoints)
                signals["body_symmetry"] = s.to_dict()
                signals["body_symmetry_score"] = bs.get_body_symmetry_score(s)
            except Exception as exc:
                warnings.append(f"body_symmetry_failed: {exc}")

    # ----- batch 3 (proportion / color / background / blur / occlusion) -----
    # body_proportion is geometry-based: needs body keypoints to be meaningful.
    if cfg.body_proportion:
        if not keypoints:
            warnings.append("body_proportion_skipped: no body keypoints/geometry available")
        else:
            try:
                from . import body_proportion as bp
                fb = tuple(face_bbox) if face_bbox is not None else None
                s = bp.evaluate_from_geometry(face_bbox=fb, keypoints=keypoints)
                signals["body_proportion"] = s.to_dict()
                signals["body_proportion_score"] = bp.get_proportion_score(s)
            except Exception as exc:
                warnings.append(f"body_proportion_failed: {exc}")

    if cfg.color_mood:
        try:
            from . import color_mood as cm
            s = cm.evaluate_color_mood_from_image(image_bgr)
            signals["color_mood"] = s.to_dict()
            signals["color_mood_score"] = cm.get_color_mood_score(s)
        except Exception as exc:
            warnings.append(f"color_mood_failed: {exc}")

    if cfg.background_quality:
        try:
            from . import background_quality as bg
            subj = (tuple(body_bbox) if body_bbox is not None
                    else (tuple(face_bbox) if face_bbox is not None else None))
            s = bg.evaluate_background_from_image(image_bgr, subject_bbox=subj)
            signals["background_quality"] = s.to_dict()
            signals["background_quality_score"] = bg.get_background_score(s)
        except Exception as exc:
            warnings.append(f"background_quality_failed: {exc}")

    if cfg.motion_blur:
        try:
            from . import motion_blur_detector as mb
            s = mb.detect_blur_from_image(image_bgr)
            signals["motion_blur"] = s.to_dict()
            signals["motion_blur_score"] = mb.get_blur_level(s)
        except Exception as exc:
            warnings.append(f"motion_blur_failed: {exc}")

    if cfg.face_occlusion:
        try:
            from . import face_occlusion as fo
            fb = tuple(face_bbox) if face_bbox is not None else None
            s = fo.evaluate_occlusion_from_image(image_bgr, face_bbox=fb)
            signals["face_occlusion"] = s.to_dict()
            signals["face_occlusion_score"] = fo.get_occlusion_level(s)
        except Exception as exc:
            warnings.append(f"face_occlusion_failed: {exc}")

    # ----- batch 4 (clothing / full body / eye / cast / duplicate / exposure) ---
    if cfg.clothing_type:
        try:
            from . import clothing_type_classifier as ct
            bb = tuple(body_bbox) if body_bbox is not None else None
            s = ct.classify_from_image(image_bgr, body_bbox=bb)
            signals["clothing_type"] = s.to_dict()
            signals["clothing_type_label"] = ct.get_clothing_type(s)
            if s.coverage_ratio is not None:
                signals["clothing_type_score"] = s.coverage_ratio
        except Exception as exc:
            warnings.append(f"clothing_type_failed: {exc}")

    if cfg.full_body:
        if face_bbox is None and body_bbox is None and not keypoints:
            warnings.append("full_body_skipped: no bbox/keypoint geometry available")
        else:
            try:
                from . import full_body_detector as fbd
                s = fbd.detect_from_geometry(
                    img_width=int(img_w or 0), img_height=int(img_h or 0),
                    face_bbox=tuple(face_bbox) if face_bbox is not None else None,
                    body_bbox=tuple(body_bbox) if body_bbox is not None else None,
                    keypoints=keypoints)
                signals["full_body"] = s.to_dict()
                signals["full_body_score"] = fbd.get_full_body_visibility(s)
            except Exception as exc:
                warnings.append(f"full_body_failed: {exc}")

    if cfg.eye_openness:
        # needs 6-point eye contour landmarks (left_eye / right_eye)
        le = landmarks.get("left_eye") if isinstance(landmarks, dict) else None
        re = landmarks.get("right_eye") if isinstance(landmarks, dict) else None
        if le is None and re is None:
            warnings.append("eye_openness_skipped: missing eye landmarks")
        else:
            try:
                from . import eye_openness as eo
                s = eo.evaluate_from_landmarks(left_eye=le, right_eye=re)
                signals["eye_openness"] = s.to_dict()
                signals["eye_openness_score"] = eo.get_eye_openness_score(s)
            except Exception as exc:
                warnings.append(f"eye_openness_failed: {exc}")

    if cfg.color_cast:
        try:
            from . import color_cast_detector as ccd
            s = ccd.detect_color_cast_from_image(image_bgr)
            signals["color_cast"] = s.to_dict()
            signals["color_cast_score"] = ccd.get_cast_strength(s)
        except Exception as exc:
            warnings.append(f"color_cast_failed: {exc}")

    if cfg.duplicate_detection:
        # single-frame, non-destructive: record dhash awareness; score only if a
        # reference hash is supplied (base_metrics['duplicate_reference_hash']).
        try:
            from . import duplicate_detector as dd
            dhash = dd.compute_dhash(image_bgr)
            if dhash is None:
                warnings.append("duplicate_detection_skipped: no hash (cv2 unavailable)")
            else:
                ref = base_metrics.get("duplicate_reference_hash")
                if ref is not None:
                    dist = dd.calculate_hamming_distance(int(dhash), int(ref))
                    s = dd.evaluate_duplicate(hamming_distance=dist)
                    signals["duplicate_detection"] = s.to_dict()
                    signals["duplicate_detection_score"] = dd.get_similarity_score(s)
                else:
                    # awareness diagnostic only (downstream dedupe can use it)
                    signals["duplicate_detection"] = {"dhash": str(dhash)}
        except Exception as exc:
            warnings.append(f"duplicate_detection_failed: {exc}")

    if cfg.exposure_outlier:
        try:
            from . import exposure_outlier as ex
            s = ex.evaluate_exposure_from_image(image_bgr)
            signals["exposure_outlier"] = s.to_dict()
            signals["exposure_outlier_score"] = ex.get_exposure_level(s)
        except Exception as exc:
            warnings.append(f"exposure_outlier_failed: {exc}")

    # ----- priority 6: region readiness (runs last; reuses prior signals) -----
    if cfg.region_readiness:
        try:
            from . import region_readiness as rr
            bb = {"face": tuple(face_bbox) if face_bbox is not None else None,
                  "body": tuple(body_bbox) if body_bbox is not None else None}
            result = rr.compute_region_readiness(
                image=image_bgr, metadata=base_metrics,
                quality_signals=signals, bboxes=bb)
            signals["region_readiness"] = result.to_dict()
            if result.face_region_score is not None:
                signals["face_region_score"] = result.face_region_score
            if result.torso_region_score is not None:
                signals["torso_region_score"] = result.torso_region_score
            if result.full_body_region_score is not None:
                signals["full_body_region_score"] = result.full_body_region_score
            if result.region_balance_score is not None:
                signals["region_balance_score"] = result.region_balance_score
            signals["best_training_use"] = result.best_training_use
            for w in result.warnings:
                warnings.append(f"region_readiness: {w}")
        except Exception as exc:
            warnings.append(f"region_readiness_failed: {exc}")

    # ----- priority 7: coverage / framing intensity (runs last) -----
    if cfg.coverage_intensity:
        try:
            from . import coverage_intensity as ci
            bb = {"face": tuple(face_bbox) if face_bbox is not None else None,
                  "body": tuple(body_bbox) if body_bbox is not None else None}
            result = ci.classify_coverage_intensity(
                image=image_bgr, metadata=base_metrics,
                quality_signals=signals, bboxes=bb)
            signals["coverage_intensity"] = result.to_dict()
            signals["coverage_label"] = result.coverage_label
            signals["framing_label"] = result.framing_label
            if result.coverage_score is not None:
                signals["coverage_score"] = result.coverage_score
            signals["training_use_hint"] = result.training_use_hint
            signals["coverage_review_recommended"] = result.review_recommended
            for w in result.warnings:
                warnings.append(f"coverage_intensity: {w}")
        except Exception as exc:
            warnings.append(f"coverage_intensity_failed: {exc}")

    # ----- priority 8: per-frame pose signature (clustering is dataset-level) -
    if cfg.pose_duplicate:
        try:
            from . import pose_duplicate_detector as pdd
            sample_view = dict(base_metrics)
            sample_view.update(signals)
            if keypoints:
                sample_view.setdefault("keypoints", keypoints)
            sig = pdd.build_pose_signature(sample_view)
            signals["pose_signature"] = sig.to_dict()
            signals["pose_signature_id"] = sig.signature_id
            signals["pose_signature_confidence"] = sig.confidence
            for w in sig.warnings:
                warnings.append(f"pose_signature: {w}")
        except Exception as exc:
            warnings.append(f"pose_signature_failed: {exc}")

    return signals, warnings
