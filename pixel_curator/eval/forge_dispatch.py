"""Standalone Forge generation dispatch (CLI backend).

Relocated verbatim from the retired web UI (``pixel_curator.ui.server``) so the
capability survives the move to strictly-CLI operation: LoRA stack
validation, ControlNet/IP-Adapter unit assembly, region-aware inpainting,
optional inswapper face lock, Forge extras post-process, and best-effort
eval + learning-loop recording.

The entry point is :func:`dispatch_forge_generate` — a pure function of the
payload (plus an output base / project) returning a result dict.  Everything
heavy is imported lazily so the module is importable in minimal envs and
unit-testable with an injected ``ForgeClient``.
"""
from __future__ import annotations

import base64 as _b64
from pathlib import Path
from typing import Optional

__all__ = ["dispatch_forge_generate", "validated_loras", "build_control_units"]


def validated_loras(loras_text: str) -> tuple:
    """Run a LoRA stack field through the preflight validator so only
    resolvable, de-prefixed <lora:...> entries reach Forge.  Returns
    ((stem, weight) pairs, warnings).  Falls back to the legacy parser if
    the validator is unavailable for any reason."""
    try:
        from ..lora_validator import normalize_stack_text, validate_lora_stack
        cleaned, warnings = validate_lora_stack(
            normalize_stack_text(str(loras_text)))
        return [tuple(p) for p in cleaned["valid_pairs"]], warnings
    except Exception:                              # never block generation
        from .matrix import parse_lora_specs
        return (parse_lora_specs(str(loras_text).replace("\n", ",")), [])


def build_control_units(payload: dict, client) -> list:
    """Assemble ControlNet (pose/depth/canny/softedge) + IP-Adapter /
    FaceID units from image paths in the payload.  Missing/unreadable
    images are skipped.  Returns [] when nothing is enabled.

    Module + model names are verified against what THIS reForge install
    actually exposes — sending an unknown preprocessor crashes reForge's
    ControlNet, so an unverifiable unit is skipped, never sent."""
    from .forge_api import ForgeClient

    def _img(path: str) -> str:
        try:
            p = Path(str(path)).expanduser()
            if p.is_file():
                return _b64.b64encode(p.read_bytes()).decode()
        except Exception:
            pass
        return ""

    try:
        avail_modules = client.controlnet_modules() if client else []
        avail_models = client.controlnet_models() if client else []
    except Exception:
        avail_modules, avail_models = [], []

    def _resolve_module(desired: str) -> str:
        if not avail_modules:
            return ""                              # can't verify -> skip unit
        if desired in avail_modules:
            return desired
        low = desired.lower()
        for m in avail_modules:                    # case-insensitive / partial
            ml = m.lower()
            if ml == low or low in ml or ml in low:
                return m
        key = low.split("_")[0]                     # e.g. 'openpose'
        for m in avail_modules:
            if key and key in m.lower():
                return m
        return ""

    def _resolve_model(stem: str) -> str:
        if not avail_models:
            return ""
        for m in avail_models:                     # 'stem [hash]' or 'stem'
            base = m.split(" [")[0]
            if base == stem or m == stem or m.startswith(stem):
                return m
        low = stem.lower()
        for m in avail_models:
            if low in m.lower():
                return m
        return ""

    def _verified(unit: dict) -> Optional[dict]:
        mod = _resolve_module(unit.get("module", ""))
        mdl = _resolve_model(unit.get("model", ""))
        if not mod or not mdl:
            return None                            # unknown -> skip, no crash
        unit["module"], unit["model"] = mod, mdl
        return unit

    raw_units = []
    # ControlNet structural units
    for ctype in ("openpose", "depth", "canny", "softedge"):
        img = _img(payload.get(f"cn_{ctype}_image", ""))
        if img:
            raw_units.append(ForgeClient.controlnet_unit(
                img, control_type=ctype,
                weight=float(payload.get(f"cn_{ctype}_weight", 0.8)),
                guidance_end=float(payload.get(f"cn_{ctype}_end", 0.8))))
    # IP-Adapter / FaceID identity guidance (reference image)
    ip_img = _img(payload.get("ip_image", "")
                  or payload.get("reference_image", ""))
    if ip_img and payload.get("ip_enabled"):
        raw_units.append(ForgeClient.ipadapter_unit(
            ip_img, ip_type=str(payload.get("ip_type", "faceid")),
            weight=float(payload.get("ip_weight", 0.75)),
            guidance_end=float(payload.get("ip_end", 0.9))))
    return [u for u in (_verified(u) for u in raw_units) if u]


def dispatch_forge_generate(payload: dict, out_base=None, project=None) -> dict:
    """Forge txt2img / img2img / region-inpaint with eval recording.

    Returns ``{"ok": True, "path", "seed", "face_locked", "post_process",
    "lora_warnings", "image_b64", "mode"}`` on success, or
    ``{"ok": False, "error", "code"}`` when Forge is unreachable.
    Best-effort stages (face lock, post-process, recording) never fail the
    generation.
    """
    from .. import manifest
    from .forge_api import ForgeClient
    from .matrix import parse_lora_specs          # noqa: F401  (fallback path)
    from ..util import now_iso as _now

    out_base = Path(str(payload.get("output")
                        or out_base
                        or getattr(project, "output_base", ""))).expanduser()
    client = ForgeClient(str(payload.get("url", "http://127.0.0.1:7860")))
    if not client.alive():
        return {"ok": False, "code": 502,
                "error": "Forge API not reachable - start reForge with --api."}

    loras, lora_warnings = validated_loras(payload.get("loras", ""))
    prompt = str(payload.get("prompt", ""))
    seed = int(float(payload.get("seed", 42)))
    cn_units = build_control_units(payload, client)
    region = str(payload.get("region", "")).strip().lower()
    init_path = str(payload.get("init", "")).strip()
    mode = "txt2img"
    if region and region not in ("", "none") and init_path:
        # Region-aware inpainting: regenerate only the masked region of
        # the init image (garment/torso/lower/full/face presets).
        from .forge_api import region_mask_png
        init_png = Path(init_path).expanduser().read_bytes()
        mask_path = str(payload.get("mask", "")).strip()
        if mask_path and Path(mask_path).expanduser().is_file():
            mask_png = Path(mask_path).expanduser().read_bytes()
        else:
            mask_png = region_mask_png(init_png, region)
        preset = client.REGION_PRESETS.get(region, {})
        png = client.inpaint(
            init_png, mask_png, prompt=prompt,
            negative=str(payload.get("neg", "")),
            denoise=float(payload.get("strength", preset.get("denoise", 0.45))),
            mask_blur=int(preset.get("mask_blur", 16)),
            steps=int(float(payload.get("steps", 28))),
            cfg=float(payload.get("cfg", 6.0)),
            seed=seed, loras=loras, controlnet_units=cn_units,
            sampler=str(payload.get("sampler") or "DPM++ 2M Karras"),
            clip_skip=int(float(payload.get("clip_skip", 0))),
            checkpoint=str(payload.get("checkpoint", "")))
        mode = f"inpaint:{region}"
    else:
        png = client.txt2img(
            prompt=prompt, negative=str(payload.get("neg", "")),
            steps=int(float(payload.get("steps", 28))),
            cfg=float(payload.get("cfg", 6.0)),
            width=int(float(payload.get("w", 1024))),
            height=int(float(payload.get("h", 1024))),
            seed=seed, loras=loras,
            sampler=str(payload.get("sampler") or "DPM++ 2M Karras"),
            clip_skip=int(float(payload.get("clip_skip", 0))),
            checkpoint=str(payload.get("checkpoint", "")),
            controlnet_units=cn_units)
    out_dir = out_base / "evals" / "_single"
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"gen_{_now().replace(':', '')}_s{seed}.png"
    fpath.write_bytes(png)

    # Optional inswapper face lock: apply the reference (source) face onto
    # the generated (target) image.  Best-effort — skipped cleanly when
    # InsightFace / inswapper_128.onnx is unavailable.
    face_swap = bool(payload.get("face_swap"))
    reference = str(payload.get("reference_image", "")).strip()
    face_locked = False
    if face_swap and reference:
        try:
            from ..identity_integration import apply_face_identity_postprocess
            face_locked = apply_face_identity_postprocess(
                fpath, reference, project)
            if face_locked:
                png = fpath.read_bytes()              # return the swapped image
        except Exception:
            face_locked = False

    # Optional post-process via Forge's built-in extras pipeline: RealESRGAN
    # x4 upscale and/or GFPGAN face restore.  Best-effort — never loses the
    # render.
    do_upscale = bool(payload.get("upscale"))
    do_restore = bool(payload.get("restore_faces"))
    post_applied = []
    if do_upscale or do_restore:
        try:
            png2 = client.postprocess(
                png, upscale=do_upscale, restore_faces=do_restore,
                scale=float(payload.get("upscale_scale", 4.0)),
                gfpgan_weight=float(payload.get("gfpgan_weight", 0.3)))
            if png2 and png2 != png:
                png = png2
                fpath.write_bytes(png)                # persist the enhanced image
                if do_upscale:
                    post_applied.append(
                        f"upscale x{float(payload.get('upscale_scale', 4.0)):g}")
                if do_restore:
                    post_applied.append(
                        f"gfpgan {float(payload.get('gfpgan_weight', 0.3)):g}")
        except Exception:
            pass                              # post-process is best-effort

    try:
        conn = manifest.connect(out_base)
        conn.execute(
            "INSERT INTO evals(lora, label, category, prompt, seed, "
            "backend, image_path, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (";".join(f"{l}:{w}" for l, w in loras) or "(none)",
             "_single", "manual", prompt, seed, "forge", str(fpath), _now()))
        conn.commit()
        # self-improvement loop: record this generation with its full
        # settings (+ identity score if a reference face exists)
        try:
            from ..studio.learning import (auto_score_identity,
                                           record_generation)
            ids = (auto_score_identity(str(fpath), reference, project)
                   if (face_swap and reference) else None)
            record_generation(
                conn, str(fpath), prompt=prompt,
                negative_prompt=str(payload.get("neg", "")),
                lora_stack=[{"name": l, "weight": w} for l, w in loras],
                seed=seed,
                recipe=str(payload.get("recipe", "")),
                identity_score=ids,
                settings={"checkpoint": str(payload.get("checkpoint", "")),
                          "clip_skip": payload.get("clip_skip"),
                          "source": "single_txt2img"})
        except Exception:
            pass                    # learning capture is best-effort
        conn.close()
    except Exception:
        pass                    # recording is best-effort

    return {"ok": True, "path": str(fpath), "seed": seed, "mode": mode,
            "face_locked": face_locked, "post_process": post_applied,
            "lora_warnings": lora_warnings,
            "image_b64": _b64.b64encode(png).decode()}
