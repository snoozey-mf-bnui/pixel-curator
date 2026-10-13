"""Figure-study guidance planner - ControlNet / IP-Adapter / FaceID.

Plans professional, studio-safe guidance payloads for advanced figure
study generation: FaceID identity lock, IP-Adapter reference guidance,
ControlNet OpenPose (pose and composition guidance), ControlNet Depth
(form and proportion stability), ControlNet Canny/Lineart (edge and
garment boundary stability) and region-aware mask guidance.

Planning only: no model files are loaded and no backend is required.
Readiness comes from the Identity Integration Layer's detector; a
missing backend produces a clean warning and a disabled (but fully
described) guidance block, so dry-run payloads always build.

Identity preservation is the governing rule: the identity block is
always first, always strongest, and structural guidance is reduced -
never identity - when the combined guidance load gets heavy.
"""

from __future__ import annotations

from ..config import Project

# Conservative default strengths (identity always strongest)
GUIDANCE_STRENGTHS = {
    "identity": 0.85,
    "pose": 0.55,
    "structure": 0.45,
    "edges": 0.30,
}

# CLI/controlnet method -> guidance channel
CONTROLNET_CHANNELS = {
    "openpose": "pose",
    "depth": "structure",
    "canny": "edges",
    "lineart": "edges",
    "softedge": "edges",
}

# Region -> recommended guidance channels (mirrors the conservative
# region-aware ControlNet policy: never everything at once)
REGION_GUIDANCE = {
    "full": ["pose", "structure"],
    "torso": ["pose", "edges"],
    "lower": ["pose", "structure"],
    "hands": ["structure", "edges"],
    "background": ["structure"],
}

# Combined non-identity strength above which structural guidance yields
# to the identity anchor
_GUIDANCE_LOAD_CAP = 1.10


def _readiness(prj: Project | None) -> dict:
    """Identity Integration Layer readiness; never raises."""
    if prj is None:
        return {}
    try:
        from ..identity_integration import detect_identity_tools
        return detect_identity_tools(prj)
    except Exception:
        return {}


def _backend_found(readiness: dict, key: str) -> bool | None:
    """True/False when detection ran; None when readiness is unknown."""
    entry = readiness.get(key)
    if not entry:
        return None
    return entry.get("status") == "found"


def plan_figure_study_guidance(
        prj: Project | None = None,
        region: str = "",
        controlnet: list[str] | None = None,
        ip_adapter: bool = False,
        faceid: bool = False,
        reference_image: str = "",
        pose_image: str = "",
        mask_image: str = "",
        identity_strength: float = 0.85,
        recipe_guidance: list[str] | None = None) -> dict:
    """Plan a studio-safe guidance payload for a figure study pass.

    Returns {"guidance": {identity, pose, structure, edges, region_mask},
    "warnings": [...], "reason_codes": [...], "recommendations": [...]}.
    Every requested channel is described even when its backend is
    unavailable (enabled=False + warning), so plans stay dry-run safe.
    """
    controlnet = [c.strip().lower() for c in (controlnet or []) if c.strip()]
    for c in controlnet:
        if c not in CONTROLNET_CHANNELS:
            raise KeyError(
                f"unknown ControlNet guidance method '{c}'. "
                f"Options: {', '.join(sorted(CONTROLNET_CHANNELS))}")
    if region and region not in REGION_GUIDANCE:
        raise KeyError(
            f"unknown figure-study region '{region}'. "
            f"Options: {', '.join(sorted(REGION_GUIDANCE))}")

    readiness = _readiness(prj)
    warnings: list[str] = []
    codes: list[str] = ["identity_anchor_priority"]
    recs: list[str] = []

    # channels requested explicitly + recommended by recipe/region
    requested = {CONTROLNET_CHANNELS[c] for c in controlnet}
    recommended = set(recipe_guidance or [])
    if region:
        recommended.update(REGION_GUIDANCE[region])
        codes.append("region_mask_recommended")
    if pose_image:
        requested.add("pose")

    # ----- identity (always present, always first) -----
    identity_method = "identity_lora_only"
    identity_enabled = True
    identity_warnings: list[str] = []
    if faceid or ip_adapter or reference_image:
        identity_method = "faceid" if faceid else "ip_adapter"
        found = _backend_found(readiness, "ip_adapter_faceid")
        if found is False:
            identity_enabled = False
            identity_warnings.append(
                "FaceID/IP-Adapter backend not detected - identity "
                "preservation relies on the identity anchor LoRA for "
                "this plan.")
            warnings.append("Identity guidance backend unavailable - "
                            "plan continues with the identity anchor "
                            "LoRA only.")
            codes.append("guidance_backend_unavailable")
        else:
            codes.append("faceid_recommended" if faceid
                         else "ip_adapter_reference_recommended")
        if not reference_image:
            recs.append("Provide --reference-image for FaceID / "
                        "IP-Adapter identity guidance.")
    elif reference_image == "" and ("pose" in requested
                                    or "structure" in requested):
        recs.append("A FaceID reference image further strengthens "
                    "identity preservation under structural guidance.")
        codes.append("faceid_recommended")
    guidance: dict = {"identity": {
        "method": identity_method,
        "enabled": identity_enabled,
        "strength": round(min(max(identity_strength, 0.0), 1.0), 2),
        "reference_image": reference_image or None,
        "reason": "Identity preservation governs every figure study "
                  "pass; the identity anchor is never reduced.",
        "warnings": identity_warnings,
    }}

    # ----- pose / structure / edges -----
    channel_meta = {
        "pose": ("controlnet_openpose", "controlnet_openpose",
                 "Preserve editorial pose structure and composition.",
                 "openpose_guidance_recommended"),
        "structure": ("controlnet_depth", "controlnet_depth",
                      "Preserve form and proportion consistency.",
                      "depth_guidance_recommended"),
        "edges": ("controlnet_canny_or_lineart", "controlnet_canny",
                  "Stabilize garment boundaries and silhouette edges.",
                  "garment_boundary_guidance_recommended"),
    }
    for channel, (method, detect_key, reason, code) in channel_meta.items():
        wanted = channel in requested
        suggested = channel in recommended
        if not (wanted or suggested):
            continue
        found = _backend_found(readiness, detect_key)
        if channel == "edges" and found is False:
            # lineart/softedge variants count as the edge backend
            found = _backend_found(readiness, "controlnet_softedge")
        block = {
            "method": method,
            "enabled": bool(wanted and found is not False),
            "strength": GUIDANCE_STRENGTHS[channel],
            "reason": reason,
            "warnings": [],
        }
        if channel == "pose" and pose_image:
            block["pose_image"] = pose_image
        if wanted and found is False:
            block["warnings"].append(
                f"{method} backend unavailable - channel planned but "
                "disabled; install the model to enable it.")
            warnings.append(f"{method} backend unavailable.")
            if "guidance_backend_unavailable" not in codes:
                codes.append("guidance_backend_unavailable")
        elif suggested and not wanted:
            block["enabled"] = False
            block["warnings"].append(
                "Recommended for this recipe/region - enable with "
                f"--controlnet {next(k for k, v in CONTROLNET_CHANNELS.items() if v == channel)}.")
        if code not in codes:
            codes.append(code)
        guidance[channel] = block

    # ----- region mask -----
    if region or mask_image:
        guidance["region_mask"] = {
            "method": "selective_region_editing",
            "enabled": True,
            "region": region or "custom",
            "mask_image": mask_image or None,
            "reason": "Region-aware refinement keeps identity regions "
                      "untouched.",
            "warnings": [],
        }

    # ----- identity priority: trim structural load, never identity -----
    load = sum(b["strength"] for k, b in guidance.items()
               if k in ("pose", "structure", "edges") and b["enabled"])
    if load > _GUIDANCE_LOAD_CAP:
        for k in ("edges", "structure"):       # lowest priority first
            if load <= _GUIDANCE_LOAD_CAP:
                break
            b = guidance.get(k)
            if b and b["enabled"]:
                cut = round(min(b["strength"] - 0.20,
                                load - _GUIDANCE_LOAD_CAP), 2)
                if cut > 0:
                    b["strength"] = round(b["strength"] - cut, 2)
                    b["warnings"].append(
                        "Strength reduced to keep the combined guidance "
                        "load below the identity-preservation cap.")
                    load = round(load - cut, 2)
        codes.append("structure_guidance_reduced_for_identity")

    return {"guidance": guidance, "warnings": warnings,
            "reason_codes": codes, "recommendations": recs}


def render_guidance(plan: dict) -> str:
    lines = ["GUIDANCE PLAN"]
    for name, b in plan["guidance"].items():
        state = "on " if b.get("enabled") else "off"
        strength = (f"{b['strength']:.2f}" if "strength" in b else "  - ")
        lines.append(f"  {name:<12} [{state}] {strength}  "
                     f"{b.get('method', '')}")
        for w in b.get("warnings", []):
            lines.append(f"      note: {w}")
    for w in plan["warnings"]:
        lines.append(f"  WARNING: {w}")
    for r in plan["recommendations"]:
        lines.append(f"  RECOMMEND: {r}")
    lines.append(f"  reason codes: {', '.join(plan['reason_codes'])}")
    return "\n".join(lines)
