"""Feature 10 - production-quality payload enhancer.

Takes an existing Forge/reForge payload and returns a NEW enhanced payload
(the input is never mutated) tuned for catalog-grade output: production
sampler/scheduler defaults, clamped dynamic CFG, optional High-Res Fix,
optional ADetailer / Face Detailer planning, catalog prompt boosters, an
identity consistency guard, and a multi-stage post-processing chain plan.

Every enhancement is conservative and identity-preserving, degrades to a
clean warning when a backend capability is missing, and attaches a structured
``quality_metadata`` block. Pure planning - no generation, no network.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional

from . import quality_boosters as qb

log = logging.getLogger("redresser.production_quality")

# -- quality presets ---------------------------------------------------------
QUALITY_PRESETS: dict[str, dict] = {
    "catalog_pro": {
        "description": "Professional catalog-grade output with balanced "
                       "detail, identity consistency, and garment clarity.",
        "sampler": "DPM++ 2M SDE Karras", "fallback_sampler": "Euler a",
        "steps": 28, "cfg_scale": 6.0, "cfg_min": 3.0, "cfg_max": 7.0,
        "enable_high_res_fix": True, "hr_scale": 1.5,
        "hr_denoising_strength": 0.35, "hr_second_pass_steps": 12,
        "hr_upscaler": "Latent",
        "enable_adetailer": True, "enable_identity_guard": True,
        "ip_adapter_boost": 0.05, "faceid_boost": 0.05,
        "post_process": {"realesrgan": True, "gfpgan": True,
                         "gfpgan_weight": 0.3, "final_sharpen": False,
                         "final_denoise": False},
    },
    "figure_study": {
        "description": "Form and proportion consistency with pose stability "
                       "and clean garment boundaries.",
        "sampler": "DPM++ 2M SDE Karras", "fallback_sampler": "Euler a",
        "steps": 30, "cfg_scale": 6.0, "cfg_min": 3.0, "cfg_max": 7.0,
        "enable_high_res_fix": True, "hr_scale": 1.5,
        "hr_denoising_strength": 0.32, "hr_second_pass_steps": 12,
        "hr_upscaler": "Latent",
        "enable_adetailer": True, "enable_identity_guard": True,
        "ip_adapter_boost": 0.05, "faceid_boost": 0.05,
        "post_process": {"realesrgan": True, "gfpgan": True,
                         "gfpgan_weight": 0.3, "final_sharpen": False,
                         "final_denoise": False},
    },
    "sheer_fabric_detail": {
        "description": "Refined fabric interaction with edge/texture clarity "
                       "and garment boundary stability.",
        "sampler": "DPM++ 2M SDE Karras", "fallback_sampler": "Euler a",
        "steps": 30, "cfg_scale": 6.5, "cfg_min": 3.5, "cfg_max": 7.0,
        "enable_high_res_fix": True, "hr_scale": 1.5,
        "hr_denoising_strength": 0.30, "hr_second_pass_steps": 14,
        "hr_upscaler": "Latent",
        "enable_adetailer": False, "enable_identity_guard": True,
        "ip_adapter_boost": 0.05, "faceid_boost": 0.05,
        "post_process": {"realesrgan": True, "gfpgan": True,
                         "gfpgan_weight": 0.3, "final_sharpen": True,
                         "final_denoise": False},
    },
    "wardrobe_sequence": {
        "description": "Frame-to-frame consistency with stable identity and "
                       "lighting for motion-guided wardrobe variation.",
        "sampler": "Euler a", "fallback_sampler": "DPM++ 2M Karras",
        "steps": 22, "cfg_scale": 6.0, "cfg_min": 3.0, "cfg_max": 6.5,
        "enable_high_res_fix": False, "hr_scale": 1.25,
        "hr_denoising_strength": 0.28, "hr_second_pass_steps": 10,
        "hr_upscaler": "Latent",
        "enable_adetailer": False, "enable_identity_guard": True,
        "ip_adapter_boost": 0.05, "faceid_boost": 0.05,
        "post_process": {"realesrgan": True, "gfpgan": True,
                         "gfpgan_weight": 0.3, "final_sharpen": False,
                         "final_denoise": False},
    },
}
DEFAULT_PRESET = "catalog_pro"

STEP_MIN, STEP_MAX = 12, 40
CFG_HARD_MIN, CFG_HARD_MAX = 1.0, 12.0
IP_WEIGHT_CAP = 0.95


def list_quality_presets() -> list[str]:
    return list(QUALITY_PRESETS.keys())


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class ProductionQualityEnhancer:
    """Enhance a Forge payload to catalog-grade production settings."""

    def __init__(self, settings=None, backend_caps: Optional[dict] = None):
        self.settings = settings
        # backend_caps: {"hires": bool, "adetailer": bool}. Unknown -> None.
        self.caps = backend_caps or {}

    # -- public --------------------------------------------------------------
    def enhance_generation_payload(self, payload: dict,
                                   quality_preset: str = DEFAULT_PRESET
                                   ) -> dict:
        warnings: list[str] = []
        preset = QUALITY_PRESETS.get(quality_preset)
        if preset is None:
            warnings.append(f"Unknown quality preset '{quality_preset}'; "
                            f"using '{DEFAULT_PRESET}'.")
            quality_preset = DEFAULT_PRESET
            preset = QUALITY_PRESETS[DEFAULT_PRESET]

        new = copy.deepcopy(payload or {})

        sampler = self._apply_sampler_cfg(new, preset, warnings)
        hr = self._apply_high_res_fix(new, preset, warnings)
        adetailer = self._apply_adetailer(new, preset, warnings)
        self._apply_prompt_boosters(new, quality_preset)
        guard = self._apply_identity_guard(new, preset, warnings)
        chain = self._post_processing_chain(preset)

        quality_metadata = {
            "quality_preset": quality_preset,
            "high_res_fix_enabled": hr,
            "adetailer_requested": bool(preset.get("enable_adetailer")),
            "adetailer_applied": adetailer,
            "sampler": sampler,
            "cfg_policy": f"dynamic_{preset['cfg_min']:g}_to_"
                          f"{preset['cfg_max']:g}",
            "steps": new.get("steps"),
            "cfg_scale": new.get("cfg_scale"),
            "identity_guard": guard,
            "post_processing_chain": chain,
            "warnings": warnings,
        }
        new["_redresser_quality"] = quality_metadata
        return {"payload": new, "quality_metadata": quality_metadata}

    # -- stages --------------------------------------------------------------
    def _apply_sampler_cfg(self, p: dict, preset: dict,
                           warnings: list[str]) -> str:
        # production mode applies the preset sampler (records nothing
        # destructive; the original payload is untouched - this is a copy).
        sampler = preset["sampler"]
        p["sampler_name"] = sampler
        steps = int(_clamp(int(preset.get("steps", 28)), STEP_MIN, STEP_MAX))
        p["steps"] = steps
        cfg = float(p.get("cfg_scale", preset.get("cfg_scale", 6.0)))
        cfg = _clamp(cfg, preset["cfg_min"], preset["cfg_max"])
        cfg = _clamp(cfg, CFG_HARD_MIN, CFG_HARD_MAX)
        p["cfg_scale"] = round(cfg, 2)
        return sampler

    def _apply_high_res_fix(self, p: dict, preset: dict,
                            warnings: list[str]) -> bool:
        if not preset.get("enable_high_res_fix"):
            return False
        if self.caps.get("hires") is False:
            warnings.append("Forge backend does not advertise High-Res Fix "
                            "support; settings planned but may be ignored.")
        p["enable_hr"] = True
        p["hr_scale"] = float(preset.get("hr_scale", 1.5))
        p["hr_upscaler"] = preset.get("hr_upscaler", "Latent")
        p["denoising_strength"] = float(
            preset.get("hr_denoising_strength", 0.35))
        p["hr_second_pass_steps"] = int(
            preset.get("hr_second_pass_steps", 12))
        return True

    def _apply_adetailer(self, p: dict, preset: dict,
                         warnings: list[str]) -> bool:
        if not preset.get("enable_adetailer"):
            return False
        if self.caps.get("adetailer") is not True:
            warnings.append("ADetailer support not detected; face-detail "
                            "refinement skipped.")
            return False
        # Add an ADetailer unit WITHOUT disturbing existing alwayson scripts.
        aos = p.setdefault("alwayson_scripts", {})
        aos["ADetailer"] = {"args": [True, False, {
            "ad_model": "face_yolov8n.pt",
            "ad_prompt": "", "ad_negative_prompt": "",
            "ad_denoising_strength": 0.3, "ad_confidence": 0.3,
            "ad_mask_blur": 4, "ad_inpaint_only_masked": True,
        }]}
        return True

    def _apply_prompt_boosters(self, p: dict, preset: str) -> None:
        merged_pos, _ = qb.merge_prompt(p.get("prompt", ""),
                                        qb.positive_fragments(preset))
        p["prompt"] = merged_pos
        merged_neg, _ = qb.merge_prompt(p.get("negative_prompt", ""),
                                        qb.negative_fragments(preset))
        p["negative_prompt"] = merged_neg

    def _apply_identity_guard(self, p: dict, preset: dict,
                              warnings: list[str]) -> bool:
        if not preset.get("enable_identity_guard"):
            return False
        # ensure identity-protection negatives
        merged_neg, _ = qb.merge_prompt(p.get("negative_prompt", ""),
                                        qb.identity_negative_fragments())
        p["negative_prompt"] = merged_neg
        # conservatively boost IP-Adapter / FaceID units if present
        boosted = False
        units = (((p.get("alwayson_scripts") or {}).get("controlnet") or {})
                 .get("args") or [])
        for unit in units:
            if not isinstance(unit, dict):
                continue
            ident = " ".join(str(unit.get(k, "")) for k in
                             ("module", "model")).lower()
            if "ip-adapter" in ident or "ipadapter" in ident \
                    or "faceid" in ident or "insightface" in ident:
                boost = float(preset.get("faceid_boost", 0.05)
                              if "faceid" in ident
                              else preset.get("ip_adapter_boost", 0.05))
                unit["weight"] = round(min(
                    float(unit.get("weight", 0.75)) + boost,
                    IP_WEIGHT_CAP), 3)
                boosted = True
        if not boosted:
            warnings.append("Identity guidance not detected; production mode "
                            "will rely on the validated LoRA stack and "
                            "conservative denoise settings.")
        return True

    @staticmethod
    def _post_processing_chain(preset: dict) -> list[str]:
        pp = preset.get("post_process", {})
        chain: list[str] = []
        if pp.get("realesrgan"):
            chain.append("realesrgan_x4")
        if pp.get("gfpgan"):
            chain.append(f"gfpgan_{pp.get('gfpgan_weight', 0.3):g}")
        if pp.get("final_sharpen"):
            chain.append("final_sharpen")
        if pp.get("final_denoise"):
            chain.append("final_denoise")
        return chain


def enhance_generation_payload(payload: dict,
                               quality_preset: str = DEFAULT_PRESET,
                               settings=None,
                               backend_caps: Optional[dict] = None) -> dict:
    """Module-level wrapper. Returns {'payload', 'quality_metadata'}."""
    return ProductionQualityEnhancer(settings, backend_caps) \
        .enhance_generation_payload(payload, quality_preset)


# -- Feature 10I: lightweight quality scoring (planning-first) ---------------
QUALITY_METRICS = ("identity_consistency", "garment_boundary_quality",
                   "fabric_detail_quality", "prompt_consistency",
                   "lighting_consistency", "post_processing_success",
                   "sequence_stability")


def evaluate_output_quality(output_path: Path | str,
                            preset: str = DEFAULT_PRESET,
                            threshold: float = 0.65) -> dict:
    """Plan or compute a consistency score for an output. Without a metric
    backend this returns a 'planned' status (never crashes, never requires
    GPU, never deletes anything)."""
    warnings: list[str] = []
    try:
        exists = bool(output_path) and Path(output_path).expanduser().is_file()
    except Exception:
        exists = False
    if not exists:
        warnings.append("Output image not found; quality scoring skipped.")
    warnings.append("Quality metric backend unavailable; scoring skipped.")
    return {
        "status": "planned",
        "score": None,
        "threshold": float(threshold),
        "preset": preset,
        "metrics_planned": list(QUALITY_METRICS),
        "below_threshold": False,    # save all by default; never auto-delete
        "warnings": warnings,
    }
