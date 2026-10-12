"""Wardrobe Variation & Selective Region Editing.

Identity-preserving wardrobe, garment, and region edits on the user's own
character images: region presets with studio-tuned settings, a model/tool
requirement registry with local detection (Forge/reForge folder layout),
an identity-preservation policy reusing stack intelligence, and an
inpainting/img2img payload builder for the existing Forge adapter.

Core works with zero heavy dependencies: payload construction, readiness
checks, and manifest tracking never require ML components.  Optional
backends (segmentation, IP-Adapter, face similarity, InsightFace post-
process) plug in through `OPTIONAL_HOOKS` without becoming requirements.
"""
from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .config import Project
from .util import now_iso

# Optional capability hooks (auto-masking, identity post-process, scoring).
# Register callables from optional extras; absence degrades gracefully.
OPTIONAL_HOOKS: dict[str, Callable] = {}
# reserved keys: optional_region_segmenter, optional_identity_postprocess,
#                optional_quality_evaluator, optional_pose_extractor


# ---------------------------------------------------------------------------
# Region presets
# ---------------------------------------------------------------------------

@dataclass
class WardrobeRegionPreset:
    region_id: str
    label: str
    description: str
    default_mask_blur: int
    default_mask_expansion: int
    recommended_denoise: tuple        # (low, high)
    identity_priority: str            # high | medium
    background_policy: str            # preserve | editable
    recommended_controlnets: list
    inpaint_mode: str                 # masked_only | whole_image
    notes: str


REGION_PRESETS: dict[str, WardrobeRegionPreset] = {p.region_id: p for p in [
    WardrobeRegionPreset(
        "upper_body_torso", "Upper Body / Torso Area",
        "Shirts, jackets, tops, layered garments, upper-body styling.",
        8, 24, (0.45, 0.65), "high", "preserve",
        ["softedge", "depth"], "masked_only",
        "Keep the face outside the mask; garment boundaries benefit from "
        "SoftEdge guidance."),
    WardrobeRegionPreset(
        "lower_body_bottomwear", "Lower Body / Bottomwear Area",
        "Pants, skirts, lower-body styling, footwear-adjacent composition.",
        8, 24, (0.5, 0.7), "medium", "preserve",
        ["depth", "softedge"], "masked_only",
        "Depth guidance preserves leg structure and volume."),
    WardrobeRegionPreset(
        "full_body_wardrobe", "Full Body Wardrobe Replacement",
        "Complete outfit changes while preserving identity and pose.",
        12, 32, (0.5, 0.75), "high", "preserve",
        ["openpose", "depth"], "masked_only",
        "Full-body wardrobe replacement benefits from pose guidance to "
        "preserve proportions; keep the face masked out or denoise low."),
    WardrobeRegionPreset(
        "arms_hands", "Arms & Hands Region",
        "Sleeve changes, gloves, hand-region styling, accessory continuity.",
        6, 16, (0.4, 0.6), "medium", "preserve",
        ["depth", "canny"], "masked_only",
        "Hands are anatomy-sensitive: keep denoise conservative."),
    WardrobeRegionPreset(
        "background_environment", "Background + Environment",
        "Scene changes while preserving character identity.",
        16, 40, (0.6, 0.85), "high", "editable",
        ["depth"], "masked_only",
        "Invert the subject mask; the character stays under composition "
        "lock while the environment is restyled."),
]}


def list_region_presets() -> list[dict]:
    return [asdict(p) for p in REGION_PRESETS.values()]


def get_region_preset(region_id: str) -> WardrobeRegionPreset:
    if region_id not in REGION_PRESETS:
        raise KeyError(f"Unknown region preset '{region_id}'. Available: "
                       f"{', '.join(REGION_PRESETS)}")
    return REGION_PRESETS[region_id]


# ---------------------------------------------------------------------------
# Model / tool requirement registry
# ---------------------------------------------------------------------------

@dataclass
class ModelRequirement:
    category: str
    required: bool
    guidance: str
    expected_paths: list = field(default_factory=list)
    detected: str = "not configured"   # found | missing | optional | not configured


def _detect(root: Optional[Path], rel_dir: str, keywords: tuple) -> tuple:
    """(status, expected_path, matches) against the Forge folder layout."""
    if not root:
        return "not configured", rel_dir, []
    d = Path(root) / rel_dir
    if not d.exists():
        return "missing", str(d), []
    hits = [p.name for p in d.glob("*") if p.suffix.lower() in
            (".safetensors", ".pth", ".pt", ".onnx", ".bin")
            and any(k in p.name.lower() for k in keywords)]
    return ("found" if hits else "missing"), str(d), hits


def resolve_controlnet_model(prj: Project, cn: str) -> str:
    """Resolve a guidance module to the actual installed model filename
    (falls back to the auto: placeholder when none is detected)."""
    root = Path(prj.forge_root).expanduser() if prj.forge_root else None
    _status, _path, hits = _detect(root, "models/ControlNet", (cn,))
    return Path(hits[0]).stem if hits else f"auto:{cn}"


def generate_region_mask(image_path, region_id: str,
                         out_path=None):
    """Automatic region mask (geometric baseline): professional region
    fractions of the frame, feathered by the preset's mask blur.  Returns
    the mask path or None (never raises; Pillow optional)."""
    try:
        from PIL import Image, ImageDraw, ImageFilter
        img = Image.open(str(image_path))
        w, h = img.size
        preset = get_region_preset(region_id)
        # region fractions (x0, y0, x1, y1) of the frame
        boxes = {
            "upper_body_torso": [(0.12, 0.20, 0.88, 0.62)],
            "lower_body_bottomwear": [(0.15, 0.52, 0.85, 0.98)],
            "full_body_wardrobe": [(0.08, 0.16, 0.92, 0.99)],
            "arms_hands": [(0.0, 0.25, 0.22, 0.85),
                           (0.78, 0.25, 1.0, 0.85)],
            "background_environment": "inverted",
        }[region_id]
        mask = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(mask)
        if boxes == "inverted":
            d.rectangle([0, 0, w, h], fill=255)
            d.rectangle([int(0.18 * w), int(0.04 * h),
                         int(0.82 * w), int(0.99 * h)], fill=0)
        else:
            for x0, y0, x1, y1 in boxes:
                d.rectangle([int(x0 * w), int(y0 * h),
                             int(x1 * w), int(y1 * h)], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(
            preset.default_mask_blur))
        out = Path(out_path) if out_path else Path(
            str(image_path)).with_suffix(".region_mask.png")
        mask.save(out)
        return out
    except Exception:
        return None


OPTIONAL_HOOKS.setdefault("optional_region_segmenter", generate_region_mask)


def get_region_model_requirements(prj: Project,
                                  region_id: str) -> list[dict]:
    """Requirement list for a region with local detection + placement
    guidance.  Never hard-fails on absent optional components."""
    preset = get_region_preset(region_id)
    root = Path(prj.forge_root).expanduser() if prj.forge_root else None
    out: list[ModelRequirement] = []

    status, path, hits = _detect(root, "models/Stable-diffusion",
                                 ("inpaint",))
    out.append(ModelRequirement(
        "inpainting_checkpoint", False,
        ("Required for high-quality inpainting. If no dedicated SDXL/Pony "
         "inpainting checkpoint is installed, masked img2img on the current "
         "base checkpoint is used instead."),
        [path], "found" if hits else ("optional" if status != "not configured"
                                      else status)))
    for cn in preset.recommended_controlnets:
        status, path, hits = _detect(root, "models/ControlNet", (cn,))
        out.append(ModelRequirement(
            f"controlnet_{cn}", False,
            ("Recommended for pose consistency." if cn == "openpose" else
             "Recommended for structure and volume consistency."
             if cn == "depth" else
             "Recommended for garment boundary and silhouette control."),
            [path], status if status != "missing" else "missing"))
    status, path, hits = _detect(root, "models/ipadapter",
                                 ("ip-adapter", "ip_adapter", "faceid"))
    out.append(ModelRequirement(
        "identity_guidance", False,
        "Recommended for identity preservation (IP-Adapter / FaceID).",
        [path], status))
    status, path, hits = _detect(root, "models/insightface",
                                 ("inswapper",))
    out.append(ModelRequirement(
        "identity_postprocess", False,
        ("Optional enhancement: InsightFace post-process "
         "(inswapper_128.onnx) for maximum facial continuity."),
        [path], "optional" if status != "found" else "found"))
    return [asdict(r) for r in out]


# ---------------------------------------------------------------------------
# Edit request + identity policy + payload builder
# ---------------------------------------------------------------------------

@dataclass
class WardrobeEditRequest:
    image_path: str
    region_id: str = "upper_body_torso"
    edit_mode: str = "garment_replacement"
    # garment_replacement | garment_layering | style_variation |
    # full_wardrobe_variation | background_environment_variation
    garment_direction_prompt: str = ""
    negative_prompt: str = ""
    mask_path: str = ""                # manual/uploaded mask (optional)
    selected_loras: list = field(default_factory=list)   # [[id, weight]]
    preserve_background: bool = True
    preserve_pose: bool = True
    identity_strength: float = 0.75
    denoise: float = 0.0               # 0 -> region recommendation
    seed: int = 42
    steps: int = 0                     # 0 -> base-model profile
    cfg: float = 0.0
    # Identity Integration Layer
    strong_face_lock: bool = False
    faceid_preset: str = "balanced"    # off|balanced|strong|maximum
    reference_image_path: str = ""
    body_structure_lock: bool = False
    silhouette_guidance: bool = False


EDIT_MODES = ("garment_replacement", "garment_layering", "style_variation",
              "full_wardrobe_variation", "background_environment_variation")


def analyze_edit_readiness(prj: Project, req: WardrobeEditRequest,
                           cards: Optional[list] = None) -> dict:
    """Identity policy + consistency notes + readiness, before generation."""
    preset = get_region_preset(req.region_id)
    lo, hi = preset.recommended_denoise
    denoise = req.denoise or round((lo + hi) / 2, 2)
    notes: list[str] = []
    suggestions: list[str] = []

    # identity scoring through stack intelligence when cards are known
    score, risk = 0.8, "medium"
    if cards:
        from .stack_intelligence import resolve_stack
        by_id = {c.lora_id: c for c in cards}
        sel = [by_id[n] for n, _ in req.selected_loras if n in by_id]
        weights = {n: w for n, w in req.selected_loras}
        st = resolve_stack(sel, weights, prj.base_model)
        score = st.identity_preservation_score
        risk = {"stable": "low", "watch": "medium",
                "elevated": "medium", "high": "high"}[st.risk_level]
        if st.total_concept_strength > 1.2:
            suggestions.append(
                "Concept strength may overpower the identity anchor. "
                "Consider reducing secondary LoRA weights.")
        if st.identity_anchor is None:
            suggestions.append(
                "No identity anchor in the stack - add the character "
                "identity LoRA before editing identity-sensitive regions.")
            risk = "high"

    if denoise > 0.7 and preset.identity_priority == "high":
        suggestions.append(
            "High denoise may alter facial continuity. Consider reducing "
            "denoise or enabling identity guidance.")
        score = round(max(0.0, score - 0.1), 2)
    if req.region_id == "full_body_wardrobe" and not req.preserve_pose:
        suggestions.append(
            "Full-body wardrobe replacement benefits from pose guidance "
            "to preserve proportions.")
    if req.preserve_background and preset.background_policy == "editable":
        notes.append("Background consistency is enabled; environment "
                     "changes will be limited.")
    if not req.mask_path:
        seg = OPTIONAL_HOOKS.get("optional_region_segmenter")
        notes.append("Automatic region segmentation "
                     + ("is available." if seg else
                        "is not configured - the edit runs as masked "
                        "img2img over the full region framing; provide a "
                        "mask image for precise garment boundaries."))
    from .identity_integration import (IdentityIntegrationConfig,
                                       augment_payload)
    id_res = augment_payload(prj, {"denoising_strength": denoise},
                             IdentityIntegrationConfig(
                                 strong_face_lock=req.strong_face_lock,
                                 pose_consistency=req.preserve_pose,
                                 region_preset=req.region_id,
                                 reference_image_path=req.image_path))
    notes += [f"Identity tools active: "
              f"{', '.join(id_res.active_tools) or 'none'}"]
    suggestions += id_res.warnings + id_res.recommendations
    return {"region": preset.label, "denoise": denoise,
            "identity_tools": id_res.active_tools,
            "degraded_features": id_res.degraded_features,
            "identity_preservation_score": round(score, 2),
            "identity_risk_level": risk,
            "consistency_notes": notes,
            "auto_adjustment_suggestions": suggestions,
            "model_requirements":
                get_region_model_requirements(prj, req.region_id)}


def build_wardrobe_edit_payload(prj: Project,
                                req: WardrobeEditRequest) -> dict:
    """Identity-preserving inpainting/img2img payload for the Forge
    adapter (sdapi/v1/img2img).  Never invents unsupported calls: mask and
    ControlNet units follow the documented A1111/reForge schema."""
    from .base_models import detect_profile
    from .eval.forge_api import ForgeClient
    if req.edit_mode not in EDIT_MODES:
        raise ValueError(f"Unknown edit mode '{req.edit_mode}'")
    preset = get_region_preset(req.region_id)
    prof = detect_profile(prj.base_model)
    lo, hi = preset.recommended_denoise
    denoise = req.denoise or round((lo + hi) / 2, 2)
    trig = f"{prj.trigger_token} {prj.class_word}".strip()

    prompt = (f"{prof['quality_prefix']}{trig}, consistent character "
              f"identity, {req.garment_direction_prompt.strip(', ')}, "
              f"high detail, natural anatomy, coherent proportions")
    negative = prof["negative"] + (", identity drift, inconsistent facial "
                                   "structure, distorted proportions")
    if req.negative_prompt:
        negative += f", {req.negative_prompt.strip(', ')}"

    img = Path(req.image_path).expanduser()
    payload: dict = {
        "init_images": [base64.b64encode(img.read_bytes()).decode()
                        if img.exists() else ""],
        "prompt": ForgeClient.lora_prompt(
            prompt, [(n, w) for n, w in req.selected_loras]),
        "negative_prompt": negative,
        "denoising_strength": float(denoise),
        "steps": int(req.steps or prof["steps"]),
        "cfg_scale": float(req.cfg or prof["cfg"]),
        "seed": int(req.seed),
        "sampler_name": prof["sampler"],
        "inpainting_fill": 1,                       # original content
        "inpaint_full_res": preset.inpaint_mode == "masked_only",
        "inpaint_full_res_padding": preset.default_mask_expansion,
        "mask_blur": preset.default_mask_blur,
        "override_settings": {
            "CLIP_stop_at_last_layers": prof["clip_skip"]},
        "override_settings_restore_afterwards": True,
    }
    mask = Path(req.mask_path).expanduser() if req.mask_path else None
    if mask and mask.exists():
        payload["mask"] = base64.b64encode(mask.read_bytes()).decode()

    # ControlNet units (alwayson_scripts) - pose/structure consistency
    # Identity Integration Layer: FaceID + region-aware guidance units
    from .identity_integration import (IdentityIntegrationConfig,
                                       augment_payload)
    id_weight = next((w for n, w in req.selected_loras
                      if "character" in n.lower()
                      or "identity" in n.lower()), None)
    cfg_id = IdentityIntegrationConfig(
        strong_face_lock=req.strong_face_lock,
        faceid_enabled=req.faceid_preset != "off",
        faceid_strength={"off": 0.0, "balanced": 0.55, "strong": 0.75,
                         "maximum": 0.95}.get(req.faceid_preset, 0.75),
        postprocess_face_lock=req.strong_face_lock,
        pose_consistency=req.preserve_pose,
        body_structure_lock=req.body_structure_lock,
        silhouette_guidance=req.silhouette_guidance,
        background_consistency=req.preserve_background,
        identity_lora_weight=id_weight,
        reference_image_path=req.reference_image_path or req.image_path,
        region_preset=req.region_id, edit_mode=req.edit_mode)
    result = augment_payload(prj, payload, cfg_id)
    payload["_identity_integration"] = result.manifest_fragment[
        "identity_integration"]
    return payload


# ---------------------------------------------------------------------------
# Manifest tracking (v10: wardrobe_edits)
# ---------------------------------------------------------------------------

def record_wardrobe_edit(conn, prj: Project, req: WardrobeEditRequest,
                         readiness: dict, output_path: str = "") -> int:
    cur = conn.execute(
        "INSERT INTO wardrobe_edits (image_path, mask_path, region_id, "
        "edit_mode, prompt, negative, loras, denoise, seed, "
        "preserve_background, preserve_pose, identity_score, risk_level, "
        "readiness, output_path, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (req.image_path, req.mask_path, req.region_id, req.edit_mode,
         req.garment_direction_prompt, req.negative_prompt,
         json.dumps(req.selected_loras),
         readiness.get("denoise"), req.seed,
         int(req.preserve_background), int(req.preserve_pose),
         readiness.get("identity_preservation_score"),
         readiness.get("identity_risk_level"),
         json.dumps({r["category"]: r["detected"]
                     for r in readiness.get("model_requirements", [])}),
         output_path, now_iso()))
    conn.commit()
    return cur.lastrowid


def generate_wardrobe_edit(prj: Project, conn, req: WardrobeEditRequest,
                           forge_url: str = "http://127.0.0.1:7860"):
    """Single edit through the Forge adapter; manifest-tracked.  Yields
    progress lines (hub JobQueue compatible)."""
    from .eval.forge_api import ForgeClient
    readiness = analyze_edit_readiness(prj, req)
    edit_id = record_wardrobe_edit(conn, prj, req, readiness)
    yield (f"Wardrobe edit #{edit_id}: {readiness['region']} | "
           f"{req.edit_mode} | denoise {readiness['denoise']}")
    for s in readiness["auto_adjustment_suggestions"]:
        yield f"  ! {s}"
    if not req.mask_path:
        seg = OPTIONAL_HOOKS.get("optional_region_segmenter")
        if seg:
            out_dir0 = prj.output_path / "wardrobe_edits"
            out_dir0.mkdir(parents=True, exist_ok=True)
            mp = seg(req.image_path, req.region_id,
                     out_dir0 / f"mask_{edit_id:05d}.png")
            if mp:
                req.mask_path = str(mp)
                yield f"  automatic region mask -> {Path(mp).name}"
    if req.strong_face_lock:
        from .identity_integration import register_hooks
        register_hooks(prj, req.reference_image_path or req.image_path)
        yield "  Strong Face Lock armed (post-process identity refinement)"
    client = ForgeClient(forge_url)
    if not client.alive():
        yield ("Forge API not reachable - payload + manifest saved; start "
               "Forge (Engines strip) and re-run.")
        return
    payload = build_wardrobe_edit_payload(prj, req)
    data = client._post("/sdapi/v1/img2img", payload)
    out_dir = prj.output_path / "wardrobe_edits"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"edit_{edit_id:05d}_seed{req.seed}.png"
    fp.write_bytes(base64.b64decode(data["images"][0]))
    post = OPTIONAL_HOOKS.get("optional_identity_postprocess")
    if post:
        try:
            post(fp)
            yield "  identity post-process applied"
        except Exception as exc:                       # noqa: BLE001
            yield f"  identity post-process skipped: {exc}"
    from .identity_integration import face_similarity
    sim = face_similarity(fp, req.reference_image_path or req.image_path, prj)
    if sim is not None:
        conn.execute("UPDATE wardrobe_edits SET identity_score=? "
                     "WHERE edit_id=?", (sim, edit_id))
        yield f"  measured face consistency: {sim}"
    conn.execute("UPDATE wardrobe_edits SET output_path=? WHERE edit_id=?",
                 (str(fp), edit_id))
    conn.commit()
    yield f"  -> {fp.name}"
