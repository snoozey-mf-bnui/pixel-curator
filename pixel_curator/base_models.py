"""Base-model profiles - one source of truth for generation + merge tuning.

Every place that emits generation parameters (playground presets, eval
matrix, stack planner block weights) resolves the project's base model to a
profile here.  Detection is filename-keyed so swapping the checkpoint in
pixel-curator.toml retunes the whole studio.  Pure stdlib.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Profiles.  Block weights are te/down/mid/up multipliers used both by the
# Merging Lab (--blocks) and the Stack Planner's runtime-stack suggestions.
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict] = {
    # ---- PRIMARY: CyberRealistic Pony v18.0 CoreShift -------------------
    # Photoreal Pony-XL derivative.  Strengths: skin/texture rendering,
    # cinematic light, clean anatomy.  Tuning notes:
    #  * DPM++ SDE Karras @ 32 steps, CFG 5, clip skip 2 (author-recommended)
    #  * responds strongly to score tags; photoreal negatives matter
    #  * style/detail LoRAs must NOT overpower the base's texture engine,
    #    so flavor stacks get conservative down/mid blocks.
    "cyberrealistic_pony": {
        "label": "CyberRealistic Pony v18.0 CoreShift",
        "arch": "PONY",
        "match": ("cyberrealistic",),          # + pony in name or arch
        "sampler": "DPM++ SDE Karras",
        "sampler_alt": "DPM++ 2M Karras",
        "steps": 32,
        "cfg": 5.0,
        "clip_skip": 2,
        "width": 1024, "height": 1024,
        "quality_prefix": "score_9, score_8_up, score_7_up, ",
        "realism_tags": "photorealistic, realistic skin texture, detailed anatomy, ",
        "negative": ("score_6, score_5, score_4, lowres, bad anatomy, "
                     "bad hands, deformed, blurry, watermark, worst quality, "
                     "cartoon, anime, 3d render, painting, sketch"),
        "sample_styles": [
            "studio portrait, soft key light, 85mm, shallow depth of field, "
            "cinematic lighting",
            "full body, casual outfit, golden hour, outdoor, film grain, "
            "natural skin texture",
            "close-up, detailed skin, freckles, natural window light, "
            "realistic rendering",
        ],
        # merge/runtime block weights per LoRA role
        "blocks": {
            "primary":  {"te": 1.0, "down": 1.0, "mid": 1.0, "up": 1.0},
            "identity": {"te": 1.0, "down": 1.0, "mid": 1.0, "up": 1.0},
            "flavor":   {"te": 0.0, "down": 0.2, "mid": 0.4, "up": 0.9},
            "wardrobe": {"te": 0.0, "down": 1.0, "mid": 1.0, "up": 0.8},
            "refiner":  {"te": 0.0, "down": 0.1, "mid": 0.3, "up": 0.9},
        },
        "weights": {"primary": 0.85, "identity": 0.8, "flavor": 0.35,
                    "wardrobe": 0.6, "refiner": 0.45},
    },

    # ---- Pony Diffusion V6 XL (previous default, kept as fallback) ------
    "pony_v6": {
        "label": "Pony Diffusion V6 XL",
        "arch": "PONY",
        "match": ("pony",),
        "sampler": "DPM++ 2M Karras",
        "sampler_alt": "Euler a",
        "steps": 30,
        "cfg": 7.0,
        "clip_skip": 2,
        "width": 1024, "height": 1024,
        "quality_prefix": "score_9, score_8_up, ",
        "realism_tags": "",
        "negative": ("lowres, bad anatomy, bad hands, deformed, blurry, "
                     "watermark, worst quality"),
        "sample_styles": [
            "studio portrait, soft lighting, photorealistic",
            "full body, casual outfit, outdoor park, golden hour",
            "close-up, detailed skin, natural light",
        ],
        "blocks": {
            "primary":  {"te": 1.0, "down": 1.0, "mid": 1.0, "up": 1.0},
            "identity": {"te": 1.0, "down": 1.0, "mid": 1.0, "up": 1.0},
            "flavor":   {"te": 0.0, "down": 0.3, "mid": 0.5, "up": 1.0},
            "wardrobe": {"te": 0.0, "down": 1.0, "mid": 1.0, "up": 1.0},
            "refiner":  {"te": 0.0, "down": 0.2, "mid": 0.4, "up": 1.0},
        },
        "weights": {"primary": 0.85, "identity": 0.8, "flavor": 0.4,
                    "wardrobe": 0.6, "refiner": 0.5},
    },

    # ---- Generic SDXL fallback ------------------------------------------
    "generic_sdxl": {
        "label": "SDXL (generic)",
        "arch": "SDXL",
        "match": (),
        "sampler": "DPM++ 2M Karras",
        "sampler_alt": "Euler a",
        "steps": 30,
        "cfg": 6.0,
        "clip_skip": 1,
        "width": 1024, "height": 1024,
        "quality_prefix": "",
        "realism_tags": "",
        "negative": ("lowres, bad anatomy, bad hands, deformed, blurry, "
                     "watermark, worst quality"),
        "sample_styles": [
            "studio portrait, soft lighting",
            "full body, outdoor, golden hour",
            "close-up, natural light",
        ],
        "blocks": {
            "primary":  {"te": 1.0, "down": 1.0, "mid": 1.0, "up": 1.0},
            "identity": {"te": 1.0, "down": 1.0, "mid": 1.0, "up": 1.0},
            "flavor":   {"te": 0.0, "down": 0.3, "mid": 0.5, "up": 1.0},
            "wardrobe": {"te": 0.0, "down": 1.0, "mid": 1.0, "up": 1.0},
            "refiner":  {"te": 0.0, "down": 0.2, "mid": 0.4, "up": 1.0},
        },
        "weights": {"primary": 0.85, "identity": 0.8, "flavor": 0.4,
                    "wardrobe": 0.6, "refiner": 0.5},
    },
}

DEFAULT_PROFILE = "cyberrealistic_pony"   # studio standard as of v3.1


def detect_profile(base_model: Optional[str]) -> dict:
    """Resolve a checkpoint path/name to a profile (never raises)."""
    name = Path(base_model).stem.lower() if base_model else ""
    if "cyberrealistic" in name:
        return PROFILES["cyberrealistic_pony"]
    if "pony" in name:
        return PROFILES["pony_v6"]
    if name:
        return PROFILES["generic_sdxl"]
    return PROFILES[DEFAULT_PROFILE]


def checkpoint_label(base_model: Optional[str]) -> Optional[str]:
    """Derive the Playground checkpoint-dropdown label ('stem  [ARCH]')."""
    if not base_model:
        return None
    stem = Path(base_model).stem
    n = stem.lower()
    arch = ("PONY" if "pony" in n or "cyberrealistic" in n
            else "SDXL" if ("xl" in n or "sdxl" in n) else "SD15")
    return f"{stem}  [{arch}]"


def build_prompt(profile: dict, trigger: str, style: str = "") -> str:
    """Quality prefix + trigger + optional style tail, profile-tuned."""
    parts = [profile["quality_prefix"] + trigger.strip()]
    if profile.get("realism_tags") and style:
        parts.append(profile["realism_tags"].rstrip(", "))
    if style:
        parts.append(style)
    return ", ".join(p.strip(", ") for p in parts if p.strip(", ")) + ", "


def sample_prompts(profile: dict, trigger: str) -> list[str]:
    return [build_prompt(profile, trigger, s).rstrip(", ")
            for s in profile["sample_styles"]]


def blocks_string(blocks: dict) -> str:
    """{'te':0,'down':0.2,...} -> 'te=0,down=0.2,mid=0.4,up=0.9' (merge CLI)."""
    return ",".join(f"{k}={blocks[k]:g}" for k in ("te", "down", "mid", "up"))


def preset_payload(profile: dict, trigger: str,
                   base_model: Optional[str] = None,
                   loras: Optional[list] = None) -> dict:
    """Complete, ready-to-load Playground preset body (callers add metadata)."""
    return {
        "checkpoint": checkpoint_label(base_model),
        "base_model_path": base_model,
        "prompt": build_prompt(profile, trigger),
        "negative": profile["negative"],
        "sampler": profile["sampler"],
        "steps": profile["steps"],
        "cfg": profile["cfg"],
        "clip_skip": profile["clip_skip"],
        "width": profile["width"], "height": profile["height"],
        "loras": [[n, round(float(w), 2)] for n, w in (loras or [])],
    }
