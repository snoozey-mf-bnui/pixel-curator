"""Identity Integration Layer - centralized face/body consistency.

One place for identity preservation across Playground img2img, inpainting,
Wardrobe Variation, and batch workflows: identity LoRA handling, IP-Adapter
FaceID reference guidance, InsightFace (inswapper_128.onnx) post-process
face lock, and region-aware ControlNet policy - with readiness detection,
graceful degradation, scoring, and manifest fragments.

No heavy imports at module load; InsightFace/ONNX are lazy and optional.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import Project

logger = logging.getLogger(__name__)

FACEID_PRESETS = {"off": 0.0, "balanced": 0.55, "strong": 0.75,
                  "maximum": 0.95}


@dataclass
class IdentityIntegrationConfig:
    enabled: bool = True
    strong_face_lock: bool = False     # FaceID + inswapper post-process
    faceid_enabled: bool = True
    faceid_strength: float = 0.75
    postprocess_face_lock: bool = False
    pose_consistency: bool = False
    body_structure_lock: bool = False
    silhouette_guidance: bool = False
    background_consistency: bool = False
    identity_lora_weight: float | None = None
    reference_image_path: str | None = None
    region_preset: str | None = None
    edit_mode: str | None = None

    @classmethod
    def from_payload(cls, data: dict) -> IdentityIntegrationConfig:
        known = cls.__dataclass_fields__
        cfg = cls(**{k: v for k, v in (data or {}).items() if k in known})
        preset = (data or {}).get("faceid_preset")
        if preset in FACEID_PRESETS:
            cfg.faceid_strength = FACEID_PRESETS[preset]
            cfg.faceid_enabled = preset != "off"
        return cfg


@dataclass
class IdentityIntegrationResult:
    payload: dict
    identity_score: float = 0.8
    active_tools: list = field(default_factory=list)
    degraded_features: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    readiness: dict = field(default_factory=dict)
    manifest_fragment: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Readiness detection (reuses the wardrobe detector; never raises)
# ---------------------------------------------------------------------------

def detect_identity_tools(prj: Project) -> dict:
    from .wardrobe import _detect
    root = Path(prj.forge_root).expanduser() if prj.forge_root else None
    out = {}
    s, p, hits = _detect(root, "models/ipadapter",
                         ("faceid", "ip-adapter", "ip_adapter"))
    out["ip_adapter_faceid"] = {"status": s, "path": p, "matches": hits}
    s, p, hits = _detect(root, "models/insightface", ("inswapper",))
    out["inswapper_postprocess"] = {"status": s, "path": p, "matches": hits}
    for cn in ("openpose", "depth", "canny", "softedge"):
        s, p, hits = _detect(root, "models/ControlNet", (cn,))
        out[f"controlnet_{cn}"] = {"status": s, "path": p, "matches": hits}
    try:
        import importlib.util
        out["insightface_runtime"] = {
            "status": "found" if importlib.util.find_spec("insightface")
            and importlib.util.find_spec("onnxruntime") else "missing"}
    except Exception:
        out["insightface_runtime"] = {"status": "missing"}
    return out


# ---------------------------------------------------------------------------
# Region-aware ControlNet policy (conservative: never everything at once)
# ---------------------------------------------------------------------------

def controlnet_policy(cfg: IdentityIntegrationConfig) -> list[str]:
    """Which guidance units apply, by region/edit mode + user toggles."""
    region = cfg.region_preset or ""
    units: list[str] = []
    if region == "full_body_wardrobe":
        if cfg.pose_consistency:
            units.append("openpose")
        if cfg.body_structure_lock or True:    # depth default for full body
            units.append("depth")
        if cfg.silhouette_guidance:
            units.append("softedge")
    elif region == "upper_body_torso":
        if cfg.pose_consistency:
            units.append("openpose")
        if cfg.silhouette_guidance:
            units.append("softedge")
        if cfg.body_structure_lock:
            units.append("depth")
    elif region == "lower_body_bottomwear":
        if cfg.pose_consistency:
            units.append("openpose")
        if cfg.body_structure_lock:
            units.append("depth")
    elif region == "arms_hands":
        if cfg.body_structure_lock:
            units.append("depth")
        if cfg.silhouette_guidance:
            units.append("canny")
    elif region == "background_environment":
        if cfg.background_consistency:
            units.append("depth")
        # FaceID/inswapper stay character-only: no identity units here
    else:                                   # plain img2img / inpainting
        if cfg.pose_consistency:
            units.append("openpose")
        if cfg.body_structure_lock:
            units.append("depth")
        if cfg.silhouette_guidance:
            units.append("softedge")
    return units


# ---------------------------------------------------------------------------
# Payload augmentation - the single integration point
# ---------------------------------------------------------------------------

def augment_payload(prj: Project, payload: dict,
                    cfg: IdentityIntegrationConfig
                    ) -> IdentityIntegrationResult:
    """Adds FaceID guidance + policy ControlNet units to a generation
    payload; reports active tools, degradations, warnings, score, and a
    manifest fragment.  Never blocks generation on optional tools."""
    res = IdentityIntegrationResult(payload=payload)
    if not cfg.enabled:
        res.manifest_fragment = {"identity_integration": {"enabled": False}}
        return res
    tools = detect_identity_tools(prj)
    res.readiness = tools
    active: list[str] = []
    if cfg.identity_lora_weight:
        active.append("identity_lora")
    is_background = cfg.region_preset == "background_environment"

    # --- FaceID reference guidance ---
    want_faceid = (cfg.faceid_enabled
                   and (cfg.strong_face_lock or cfg.faceid_strength > 0)
                   and not is_background)
    if want_faceid:
        if tools["ip_adapter_faceid"]["status"] == "found":
            ref = cfg.reference_image_path
            if ref and Path(ref).expanduser().exists():
                import base64
                ref_b64 = base64.b64encode(
                    Path(ref).expanduser().read_bytes()).decode()
            elif payload.get("init_images"):
                ref_b64 = payload["init_images"][0]   # input image as ref
            else:
                ref_b64 = ""
            if ref_b64:
                fid_hits = tools["ip_adapter_faceid"].get("matches") or []
                unit = {"module": "ip-adapter_face_id_plus",
                        "model": (Path(fid_hits[0]).stem if fid_hits
                                  else "auto:ip-adapter-faceid-plusv2"),
                        "weight": round(float(cfg.faceid_strength), 2),
                        "image": ref_b64,
                        "guidance_start": 0.0, "guidance_end": 1.0}
                payload.setdefault("alwayson_scripts", {}).setdefault(
                    "controlnet", {}).setdefault("args", []).append(unit)
                active.append("ip_adapter_faceid")
            else:
                res.degraded_features.append("ip_adapter_faceid")
                res.warnings.append(
                    "FaceID guidance skipped - no reference identity "
                    "image available.")
        else:
            res.degraded_features.append("ip_adapter_faceid")
            res.warnings.append(
                "IP-Adapter FaceID not configured - continuing with "
                "identity LoRA and structural guidance.")

    # --- region-aware ControlNet policy ---
    existing = {u.get("module", "").split("_")[0] for u in
                payload.get("alwayson_scripts", {})
                .get("controlnet", {}).get("args", [])}
    for cn in controlnet_policy(cfg):
        if cn in existing:
            continue
        if tools[f"controlnet_{cn}"]["status"] == "found":
            from .wardrobe import resolve_controlnet_model
            unit = {"module": "openpose_full" if cn == "openpose" else cn,
                    "model": resolve_controlnet_model(prj, cn),
                    "weight": 0.8,
                    "guidance_start": 0.0, "guidance_end": 0.9}
            payload.setdefault("alwayson_scripts", {}).setdefault(
                "controlnet", {}).setdefault("args", []).append(unit)
            active.append(f"controlnet_{cn}")
        else:
            res.degraded_features.append(f"controlnet_{cn}")
            res.warnings.append(
                f"ControlNet {cn} not configured - unit skipped.")

    # --- post-process face lock intent ---
    if ((cfg.strong_face_lock or cfg.postprocess_face_lock)
            and not is_background):
        ok = (tools["inswapper_postprocess"]["status"] == "found"
              and tools["insightface_runtime"]["status"] == "found")
        if ok:
            active.append("inswapper_postprocess")
        else:
            res.degraded_features.append("inswapper_postprocess")
            res.warnings.append(
                "Strong Face Lock post-process unavailable "
                "(inswapper_128.onnx or InsightFace runtime missing) - "
                "FaceID/LoRA identity support continues.")

    # --- scoring + recommendations ---
    score = 0.7
    score += 0.1 if "identity_lora" in active else 0.0
    score += 0.1 if "ip_adapter_faceid" in active else 0.0
    score += 0.05 if "inswapper_postprocess" in active else 0.0
    score += 0.05 if any(t.startswith("controlnet_") for t in active) else 0
    denoise = float(payload.get("denoising_strength", 0.5))
    if denoise > 0.7 and not is_background:
        score -= 0.1
        res.recommendations.append(
            "High denoise may alter facial continuity - reduce denoise or "
            "raise FaceID strength.")
    if not active:
        res.recommendations.append(
            "No identity tools active - add the identity LoRA or enable "
            "FaceID guidance before identity-critical edits.")
    res.identity_score = round(max(0.0, min(1.0, score)), 2)
    res.active_tools = active
    res.manifest_fragment = {"identity_integration": {
        "enabled": True, "strong_face_lock": cfg.strong_face_lock,
        "reference_image": cfg.reference_image_path or "",
        "active_tools": active, "faceid_strength": cfg.faceid_strength,
        "identity_lora_weight": cfg.identity_lora_weight,
        "pose_consistency": cfg.pose_consistency,
        "body_structure_lock": cfg.body_structure_lock,
        "identity_preservation_score": res.identity_score,
        "warnings": res.warnings,
        "degraded_features": res.degraded_features}}
    return res


# ---------------------------------------------------------------------------
# Post-process face lock hook (lazy; production interface, mockable)
# ---------------------------------------------------------------------------

_SWAPPER = {"app": None, "swapper": None}


def apply_face_identity_postprocess(image_path, reference_image_path,
                                    prj: Project | None = None) -> bool:
    """Restore reference face identity on a generated image in place.
    Lazy InsightFace/ONNX import; returns False (never raises) when the
    runtime or model is unavailable."""
    try:
        import cv2  # noqa: F401  (opencv ships with the [ai] extra)
        import insightface
        from insightface.app import FaceAnalysis
        root = Path(prj.forge_root).expanduser() if prj and prj.forge_root \
            else None
        model = None
        if root:
            cands = list((root / "models" / "insightface").glob(
                "inswapper*.onnx"))
            model = str(cands[0]) if cands else None
        if not model:
            return False
        if _SWAPPER["swapper"] is None:
            _SWAPPER["app"] = FaceAnalysis(name="buffalo_l", providers=_rt_providers())
            _SWAPPER["app"].prepare(ctx_id=0, det_size=(640, 640))
            _SWAPPER["swapper"] = insightface.model_zoo.get_model(model)
        img = cv2.imread(str(image_path))
        ref = cv2.imread(str(reference_image_path))
        faces = _SWAPPER["app"].get(img)
        ref_faces = _SWAPPER["app"].get(ref)
        if not faces or not ref_faces:
            return False
        out = _SWAPPER["swapper"].get(img, faces[0], ref_faces[0],
                                      paste_back=True)
        cv2.imwrite(str(image_path), out)
        return True
    except Exception:
        return False


def face_similarity(generated_path, reference_path,
                    prj: Project | None = None):
    """Measured face-consistency score (ArcFace cosine, 0..1) between a
    generated image and the reference identity.  Lazy InsightFace; returns
    None when unavailable - closes the QA loop when present."""
    try:
        import cv2
        import numpy as np
        from insightface.app import FaceAnalysis
        if _SWAPPER["app"] is None:
            _SWAPPER["app"] = FaceAnalysis(name="buffalo_l", providers=_rt_providers())
            _SWAPPER["app"].prepare(ctx_id=0, det_size=(640, 640))
        g = _SWAPPER["app"].get(cv2.imread(str(generated_path)))
        r = _SWAPPER["app"].get(cv2.imread(str(reference_path)))
        if not g or not r:
            return None
        a, b = g[0].normed_embedding, r[0].normed_embedding
        return round(float(max(0.0, np.dot(a, b))), 3)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return None


def register_hooks(prj: Project, reference_image_path: str) -> None:
    """Wire the face-lock post-process into the wardrobe OPTIONAL_HOOKS.

    After registration the wardrobe / region pipeline runs the inswapper
    face restore on each generated image: the reference (source) face is
    applied onto the generated (target) image for maximum facial
    continuity."""
    from . import wardrobe

    def _hook(fp):
        if not apply_face_identity_postprocess(
                fp, reference_image_path, prj):
            raise RuntimeError("face identity post-process unavailable")
    wardrobe.OPTIONAL_HOOKS["optional_identity_postprocess"] = _hook


def _rt_providers():
    from .runtime import onnx_providers
    return onnx_providers()


def identity_settings_payload() -> dict:
    """UI-ready control spec (presets + toggles)."""
    return {"faceid_presets": FACEID_PRESETS,
            "toggles": ["strong_face_lock", "pose_consistency",
                        "body_structure_lock", "silhouette_guidance",
                        "background_consistency"]}


# ---------------------------------------------------------------------------
# Artistic Figure Study Layer: progressive concept layering assessment
# (additive - non-artistic flows above are unchanged)
# ---------------------------------------------------------------------------

# Layer order for progressive artistic studies; later layers carry more
# identity pressure when overdriven.
ARTISTIC_LAYER_ORDER = (
    "identity_anchor", "character",
    "form_proportion", "garment_rendering", "sheer_garment_rendering",
    "pose", "lighting", "skin_texture_highlight", "composition", "style",
)

_LAYER_PRESSURE = {
    "form_proportion": 0.10, "garment_rendering": 0.12,
    "sheer_garment_rendering": 0.16, "pose": 0.10, "lighting": 0.06,
    "skin_texture_highlight": 0.14, "composition": 0.06, "style": 0.12,
}

ARTISTIC_SAFE_CONCEPT_TOTAL = 1.20
ARTISTIC_SKIN_HIGHLIGHT_MAX = 0.45


def assess_concept_layering(layers: list[dict],
                            prj: Project | None = None) -> dict:
    """Evaluate a progressive artistic concept stack
    (identity anchor -> fashion/form study -> garment rendering ->
    skin texture treatment -> editorial posing).

    `layers`: [{"name", "role", "weight"}].  Returns reason codes,
    professional warnings, a conservative adjusted stack and an identity
    pressure estimate.  Pure and deterministic - safe for tests/UI."""
    warnings: list[str] = []
    codes: list[str] = []
    adjusted = [dict(l) for l in layers]

    id_layers = [l for l in adjusted
                 if l.get("role") in ("identity_anchor", "character")]
    concepts = [l for l in adjusted if l not in id_layers]

    if id_layers and max(l["weight"] for l in id_layers) >= 0.60:
        codes.append("identity_anchor_preserved")
    else:
        codes.append("identity_anchor_missing" if not id_layers
                     else "identity_anchor_below_minimum")
        warnings.append("Strengthen the identity anchor (0.75-0.90) "
                        "before layering concept studies.")

    # role-specific caps come first - they are guidance regardless of the
    # stack total (the rebalance below works on the capped weights)
    for l in concepts:
        if l.get("role") == "skin_texture_highlight" and \
                l["weight"] > ARTISTIC_SKIN_HIGHLIGHT_MAX:
            l["weight"] = ARTISTIC_SKIN_HIGHLIGHT_MAX
            codes.append("skin_highlight_strength_reduced")
            warnings.append(
                "Skin texture highlight strength reduced to keep facial "
                "texture consistent with the identity anchor.")

    # pressure model: weighted sum of concept layers
    pressure = sum(_LAYER_PRESSURE.get(l.get("role", ""), 0.10)
                   * l["weight"] / 0.40 for l in concepts)
    total = round(sum(l["weight"] for l in concepts), 2)

    if total <= ARTISTIC_SAFE_CONCEPT_TOTAL and len(concepts) <= 3:
        codes.append("concept_stack_within_safe_range")
    else:
        codes.append("concept_stack_high_identity_risk")
        warnings.append(
            f"{len(concepts)} concept layers at total weight {total:.2f} "
            "may overpower the identity anchor - conservative trims "
            "applied to the lowest-priority layers.")
        order = {r: i for i, r in enumerate(ARTISTIC_LAYER_ORDER)}
        excess = total - ARTISTIC_SAFE_CONCEPT_TOTAL
        for l in sorted(concepts,
                        key=lambda x: -order.get(x.get("role", ""), 99)):
            if excess <= 0:
                break
            cut = min(round(l["weight"] * 0.3, 2), excess)
            l["weight"] = round(l["weight"] - cut, 2)
            excess = round(excess - cut, 4)

    # role-specific guidance
    for l in concepts:
        role = l.get("role", "")
        if role in ("pose", "form_proportion") and l["weight"] >= 0.35:
            if "pose_guidance_recommended" not in codes:
                codes.append("pose_guidance_recommended")
                warnings.append(
                    "Enable pose guidance (ControlNet openpose/depth) for "
                    "body-structure consistency at this pose strength.")
        if role in ("garment_rendering", "sheer_garment_rendering") and \
                l["weight"] >= 0.40:
            if "garment_region_mask_recommended" not in codes:
                codes.append("garment_region_mask_recommended")
                warnings.append(
                    "Use selective region editing with a garment mask so "
                    "garment changes never touch identity regions.")

    # reference availability (graceful when prj absent)
    if prj is not None:
        try:
            tools = detect_identity_tools(prj)
            if tools.get("ip_adapter_faceid", {}).get("status") != "found":
                codes.append("identity_reference_recommended")
        except Exception:
            codes.append("identity_reference_recommended")
    score = round(max(0.05, 1.0 - 0.5 * pressure
                      - (0.0 if id_layers else 0.35)), 3)
    return {"adjusted_layers": adjusted, "reason_codes": codes,
            "warnings": warnings, "identity_pressure": round(pressure, 3),
            "identity_score_estimate": score,
            "total_concept_weight": total}
