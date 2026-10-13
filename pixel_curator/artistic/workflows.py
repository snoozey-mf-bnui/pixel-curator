"""Artistic workflow builders - payloads for existing generation systems.

Additive bridges from composed artistic prompts to the studio's proven
payload paths:

* txt2img studio composition  -> ForgeClient payload (+ Identity
  Integration Layer augmentation: FaceID / ControlNet readiness)
* selective region editing    -> Wardrobe edit payload (region presets,
  mask feathering, identity-preserving inpainting)

Everything degrades gracefully: missing optional models only produce
professional recommendations, never exceptions.  Heavy imports stay
lazy; runtime/device behavior goes through pixel_curator.runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import Project

logger = logging.getLogger(__name__)
from .prompt_composer import IDENTITY_ROLES, ComposedArtisticPrompt

# Region preset by workflow focus (Wardrobe / Dressing Studio ids)
WORKFLOW_REGIONS = {
    "garment_variation": "full_body_wardrobe",
    "sheer_garment_rendering": "upper_body_torso",
    "skin_texture_highlight_refinement": "upper_body_torso",
    "form_proportion_study": "full_body_wardrobe",
    "editorial_pose_variation": "full_body_wardrobe",
}

# ---------------------------------------------------------------------------
# Phase 4: figure-study region policy
#
# Friendly region names -> wardrobe region preset + conservative edit
# parameters + targeted negative chains. Identity-critical regions get
# lower denoise; hands get the stability chain and a ControlNet
# suggestion; backgrounds may move more freely.
# ---------------------------------------------------------------------------
FIGURE_STUDY_REGION_POLICY: dict[str, dict] = {
    "full": {
        "region_id": "full_body_wardrobe", "denoise": 0.50,
        "negative_chains": ["anatomical_coherence", "hand_stability"],
        "guidance": "Full-figure pass - keep denoise moderate and verify "
                    "identity after each batch step.",
    },
    "torso": {
        "region_id": "upper_body_torso", "denoise": 0.45,
        "negative_chains": ["garment_boundary_control",
                            "skin_texture_control"],
        "guidance": "Torso region - garment boundary and skin texture "
                    "chains active; the face stays outside the mask.",
    },
    "lower": {
        "region_id": "lower_body_bottomwear", "denoise": 0.45,
        "negative_chains": ["garment_boundary_control",
                            "anatomical_coherence"],
        "guidance": "Lower-body region - identity regions untouched; "
                    "anatomical coherence chain active.",
    },
    "hands": {
        "region_id": "arms_hands", "denoise": 0.35,
        "negative_chains": ["hand_stability", "anatomical_coherence"],
        "guidance": "Hand refinement - low denoise with the hand "
                    "stability chain; a depth ControlNet pass further "
                    "improves articulation.",
    },
    "background": {
        "region_id": "background_environment", "denoise": 0.60,
        "negative_chains": [],
        "guidance": "Background replacement - subject fully preserved; "
                    "higher denoise is safe outside the figure.",
    },
}


@dataclass
class ArtisticWorkflowResult:
    payload: dict
    workflow: str
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    identity_score_estimate: float = 0.8


def _lora_pairs(composed: ComposedArtisticPrompt) -> list[tuple[str, float]]:
    return [(s["name"], s["weight"]) for s in composed.lora_stack
            if s["weight"] > 0]


def resolve_identity_lora(prj: Project) -> tuple[str, list[str]]:
    """Best identity anchor from the LoRA Explorer library; falls back to
    the placeholder name with a professional warning."""
    warnings: list[str] = []
    try:
        from ..lora_explorer import scan_loras
        from ..starter_stacks import _pick_identity
        cards = scan_loras(prj)
        pick = _pick_identity(cards)
        if pick:
            return pick.lora_id, warnings
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    warnings.append(
        "No identity anchor LoRA resolved from the library - the stack "
        "uses the placeholder name 'identity_anchor'. Train or register "
        "an identity LoRA for character consistency.")
    return "identity_anchor", warnings


def build_artistic_txt2img_payload(
        prj: Project, composed: ComposedArtisticPrompt,
        seed: int = 42, steps: int = 0, cfg: float = 0.0,
        width: int = 1024, height: int = 1024,
        identity_reference: str = "",
        pose_guidance: bool = False) -> ArtisticWorkflowResult:
    """Studio-composition txt2img payload with Identity Integration Layer
    augmentation when reference assets are available."""
    from ..base_models import detect_profile
    from ..eval.forge_api import ForgeClient
    prof = detect_profile(prj.base_model)

    payload = ForgeClient.build_txt2img_payload(
        composed.prompt, negative=composed.negative_prompt,
        steps=int(steps or prof["steps"]), cfg=float(cfg or prof["cfg"]),
        width=width, height=height, seed=seed,
        sampler=prof["sampler"], loras=_lora_pairs(composed))

    res = ArtisticWorkflowResult(
        payload=payload, workflow="artistic_txt2img",
        warnings=list(composed.warnings),
        reason_codes=list(composed.reason_codes),
        identity_score_estimate=composed.identity_score_estimate)

    # Identity Integration Layer: FaceID + pose guidance when available
    try:
        from ..identity_integration import IdentityIntegrationConfig, augment_payload
        id_w = next((s["weight"] for s in composed.lora_stack
                     if s["role"] in IDENTITY_ROLES), None)
        cfg_id = IdentityIntegrationConfig(
            faceid_enabled=bool(identity_reference),
            pose_consistency=pose_guidance,
            identity_lora_weight=id_w,
            reference_image_path=identity_reference or None)
        result = augment_payload(prj, payload, cfg_id)
        res.payload = result.payload
        res.warnings += list(result.warnings)
        res.recommendations += list(result.recommendations)
        if identity_reference:
            res.reason_codes.append("identity_reference_active")
        else:
            res.reason_codes.append("identity_reference_recommended")
            res.recommendations.append(
                "Provide an identity reference image to enable FaceID "
                "guidance for stronger character consistency.")
        if pose_guidance:
            res.reason_codes.append("pose_guidance_recommended")
    except Exception:
        res.recommendations.append(
            "Identity Integration Layer unavailable - payload built "
            "without FaceID/ControlNet augmentation.")

    res.recommendations += runtime_guidance(prj)
    return res


def build_selective_region_payload(
        prj: Project, composed: ComposedArtisticPrompt,
        image_path: str, workflow: str = "garment_variation",
        region_id: str = "", direction_prompt: str = "",
        denoise: float = 0.0, seed: int = 42,
        identity_reference: str = "",
        pose_guidance: bool = True) -> ArtisticWorkflowResult:
    """Selective region editing payload (garment variation, skin texture
    highlight refinement, editorial pose variation) through the Wardrobe
    Variation engine: region presets, mask feathering and the Identity
    Integration Layer all reused as-is."""
    from ..wardrobe import WardrobeEditRequest, build_wardrobe_edit_payload
    region = region_id or WORKFLOW_REGIONS.get(workflow, "full_body_wardrobe")
    direction = direction_prompt or ", ".join(
        composed.prompt.split(", ")[3:8])      # subject fragments
    req = WardrobeEditRequest(
        image_path=image_path, region_id=region,
        edit_mode="style_variation",
        garment_direction_prompt=direction,
        selected_loras=_lora_pairs(composed),
        denoise=denoise, seed=seed,
        preserve_pose=pose_guidance,
        reference_image_path=identity_reference or "",
        negative_prompt=composed.negative_prompt)
    payload = build_wardrobe_edit_payload(prj, req)
    res = ArtisticWorkflowResult(
        payload=payload, workflow=workflow,
        warnings=list(composed.warnings),
        reason_codes=list(composed.reason_codes)
        + ["garment_region_mask_recommended"],
        identity_score_estimate=composed.identity_score_estimate)
    res.recommendations.append(
        "Selective region editing keeps identity regions untouched; "
        "review the generated mask before batch runs.")
    res.recommendations += runtime_guidance(prj)
    return res


def build_figure_study_region_payload(
        prj: Project, composed: ComposedArtisticPrompt,
        image_path: str, region: str = "full",
        denoise: float = 0.0, seed: int = 42,
        identity_reference: str = "") -> ArtisticWorkflowResult:
    """Region-aware figure-study img2img payload (Phase 4).

    Applies the FIGURE_STUDY_REGION_POLICY: conservative per-region
    denoise, the region's targeted negative chains layered into the
    composed negative, pose preservation always on, and FaceID /
    ControlNet readiness recommendations."""
    if region not in FIGURE_STUDY_REGION_POLICY:
        raise KeyError(
            f"unknown figure-study region '{region}'. "
            f"Options: {', '.join(sorted(FIGURE_STUDY_REGION_POLICY))}")
    policy = FIGURE_STUDY_REGION_POLICY[region]

    # layer the region's negative chains into the composed negative
    negative = composed.negative_prompt
    if policy["negative_chains"]:
        from .prompt_composer import _dedupe, compose_negative_chains
        negative = ", ".join(_dedupe(
            compose_negative_chains(policy["negative_chains"])
            + negative.split(", ")))

    from ..wardrobe import WardrobeEditRequest, build_wardrobe_edit_payload
    direction = ", ".join(composed.prompt.split(", ")[3:8])
    req = WardrobeEditRequest(
        image_path=image_path, region_id=policy["region_id"],
        edit_mode="style_variation",
        garment_direction_prompt=direction,
        selected_loras=_lora_pairs(composed),
        denoise=denoise or policy["denoise"], seed=seed,
        preserve_pose=True,
        reference_image_path=identity_reference or "",
        negative_prompt=negative)
    payload = build_wardrobe_edit_payload(prj, req)

    res = ArtisticWorkflowResult(
        payload=payload, workflow=f"figure_study_region_{region}",
        warnings=list(composed.warnings),
        reason_codes=list(composed.reason_codes)
        + [f"figure_study_region:{region}",
           "region_negative_chains_applied"],
        identity_score_estimate=composed.identity_score_estimate)
    res.recommendations.append(policy["guidance"])
    res.recommendations.append(
        "Mask feathering from the region preset is applied - widen the "
        "feather if hard edit boundaries appear.")
    if not identity_reference and region in ("full", "torso"):
        res.recommendations.append(
            "Add --identity-reference <image> for FaceID guidance on "
            "identity-adjacent regions.")
    if region == "hands":
        res.recommendations.append(
            "Suggested: depth ControlNet at 0.4-0.6 for hand "
            "articulation stability.")
    res.recommendations += runtime_guidance(prj)
    return res


# ---------------------------------------------------------------------------
# Runtime / memory guards (MPS + CUDA safe)
# ---------------------------------------------------------------------------

def runtime_guidance(prj: Project) -> list[str]:
    """Professional device-aware recommendations; never raises."""
    recs: list[str] = []
    try:
        from ..runtime import get_device_name
        dev = get_device_name((prj.runtime or {}).get("device", "auto"))
    except Exception:
        return recs
    if dev == "mps":
        recs.append("MPS: use a low preview batch size (1-2) and keep "
                    "other model-heavy apps closed during generation.")
        recs.append("MPS: clear the accelerator cache between heavy "
                    "selective-edit batches (handled automatically).")
    elif dev == "cuda":
        recs.append("CUDA: bfloat16/TF32 are applied from the project "
                    "runtime config where supported.")
        recs.append("Colab: copy active models to local /content storage "
                    "for faster generation; use Drive only for "
                    "persistence.")
    else:
        recs.append("CPU runtime detected - artistic generation will be "
                    "slow; payload building is unaffected.")
    return recs


def clear_cache_between_operations(prj: Project) -> None:
    """Best-effort accelerator cache clear between heavy operations."""
    try:
        from ..runtime import clear_accelerator_cache, get_device_name
        clear_accelerator_cache(
            get_device_name((prj.runtime or {}).get("device", "auto")))
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
