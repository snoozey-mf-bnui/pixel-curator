"""Feature 11H - CyberRealisticPony Personal Base profile registry.

Defines exactly ONE named base profile - "CyberRealisticPony Personal Base" -
plus the target-specific training presets used by the phased training
orchestrator. The profile is additive: it does not replace or mutate the
existing ``pixel_curator.base_models.PROFILES`` (which keep their internal keys),
it simply layers the personal-base recipe metadata on top of the primary
checkpoint.

No other named realism base profile may be added here. ``assert_single_base``
enforces that invariant for the tests and the UI dropdown.

Pure stdlib, local-first, non-destructive.
"""
from __future__ import annotations

#: The one and only named base profile for Feature 11.
BASE_PROFILE_NAME = "CyberRealisticPony Personal Base"

#: Primary training/inference checkpoint for the personal base.
BASE_CHECKPOINT = "cyberrealisticPony_v180Coreshift.safetensors"

#: Short model hash as reported by Forge/reForge (disambiguation only).
BASE_CHECKPOINT_HASH = "9d0e340f5f"

#: Canonical local path to the checkpoint on the user's Forge/reForge install.
#: Used only as a placeholder hint in generated configs/runbooks; it is never
#: required to exist at plan time (training runs elsewhere, e.g. Colab).
BASE_CHECKPOINT_PATH = (
    "~/FORGE_REPO/stable-diffusion-webui-reForge/models/Stable-diffusion/"
    "cyberrealisticPony_v180Coreshift.safetensors")


# ---------------------------------------------------------------------------
# Named base profile.
# ---------------------------------------------------------------------------
_BASE_PROFILE: dict = {
    "name": BASE_PROFILE_NAME,
    "base_checkpoint": BASE_CHECKPOINT,
    "base_checkpoint_path": BASE_CHECKPOINT_PATH,
    "base_checkpoint_hash": BASE_CHECKPOINT_HASH,
    "arch": "PONY",
    "intended_use": (
        "Private self-modeling personal catalog production: virtual clothing "
        "try-on, wardrobe variation, garment transformation, and fashion "
        "catalog styling anchored to the user's personal identity LoRA."),
    "recommended_lora_stack": {
        "core_identity":  {"weight_range": [0.70, 0.90], "role": "identity"},
        "outfit":         {"weight_range": [0.50, 0.75], "role": "wardrobe"},
        "style":          {"weight_range": [0.25, 0.45], "role": "flavor"},
        "pose":           {"weight_range": [0.30, 0.55], "role": "pose"},
        "garment_detail": {"weight_range": [0.30, 0.50], "role": "refiner"},
    },
    "prompt_boosters": (
        "personal_identity_anchor, professional catalog reference, refined "
        "garment rendering, controlled studio lighting, polished editorial "
        "composition"),
    "negative_prompt_boosters": (
        "lowres, bad anatomy, deformed, blurry, watermark, worst quality, "
        "plastic skin, distorted garment, fabric artifacts, mismatched "
        "lighting"),
    "quality_preset": "catalog_pro",
    "controlnet_defaults": {
        "module": "openpose_full",
        "model": "auto",
        "weight": 0.55,
        "guidance_start": 0.0,
        "guidance_end": 0.7,
    },
    "ip_adapter_defaults": {
        "model": "ip-adapter-faceid",
        "weight": 0.6,
        "identity_consistency": True,
    },
    "post_processing_defaults": {
        "face_restoration": "GFPGAN",
        "gfpgan_weight": 0.3,
        "upscaler": "RealESRGAN",
        "upscale_factor": 2,
    },
    "sampler": "DPM++ SDE Karras",
    "steps": 32,
    "cfg": 5.0,
    "clip_skip": 2,
    "width": 1024, "height": 1024,
    "training_source_notes": (
        "Built from the user's own consenting personal media archive only. "
        "Datasets are classified locally into core_identity / outfit / style "
        "/ pose / garment_detail splits, versioned, and used to train a core "
        "identity LoRA followed by specialized refinement LoRAs, then merged "
        "via MergeForge into this personal base. No third-party or "
        "non-consenting subjects."),
}


# ---------------------------------------------------------------------------
# Target-specific training presets.
# ---------------------------------------------------------------------------
_TRAINING_PRESETS: dict[str, dict] = {
    "Core Identity Anchor Training": {
        "target": "core_identity",
        "output_name": "redresser_core_identity",
        "network_dim": 32, "network_alpha": 16,
        "weight_range": [0.70, 0.90],
        "intended_use": "Primary personal identity anchor LoRA.",
    },
    "Outfit Refinement Training": {
        "target": "outfit",
        "output_name": "redresser_outfit_refinement",
        "network_dim": 24, "network_alpha": 12,
        "weight_range": [0.50, 0.75],
        "intended_use": "Wardrobe variation and garment rendering refinement.",
    },
    "Style Refinement Training": {
        "target": "style",
        "output_name": "redresser_style_refinement",
        "network_dim": 16, "network_alpha": 8,
        "weight_range": [0.25, 0.45],
        "intended_use": "Catalog photography style and editorial composition.",
    },
    "Pose Refinement Training": {
        "target": "pose",
        "output_name": "redresser_pose_refinement",
        "network_dim": 16, "network_alpha": 8,
        "weight_range": [0.30, 0.55],
        "intended_use": "Fashion model pose and balanced body proportions.",
    },
    "Garment Detail Refinement Training": {
        "target": "garment_detail",
        "output_name": "redresser_garment_detail_refinement",
        "network_dim": 24, "network_alpha": 12,
        "weight_range": [0.30, 0.50],
        "intended_use": "Textile structure, fabric interaction, garment edges.",
    },
    "CyberRealisticPony Catalog Production": {
        "target": "catalog_reference",
        "output_name": "redresser_catalog_production",
        "network_dim": 0, "network_alpha": 0,   # inference preset, not training
        "weight_range": [0.0, 0.0],
        "intended_use": (
            "Production generation preset using the personal base profile "
            "for finished catalog renders (no training)."),
    },
}


def get_base_profile() -> dict:
    """Return a copy of the single named base profile."""
    import copy
    return copy.deepcopy(_BASE_PROFILE)


def list_base_profiles() -> list[str]:
    """Return the list of named base profiles. Exactly one by design."""
    return [BASE_PROFILE_NAME]


def list_training_presets() -> list[dict]:
    """Return target-specific training/production presets (UI-ready rows)."""
    rows = []
    for name, body in _TRAINING_PRESETS.items():
        row = {"name": name}
        row.update(body)
        rows.append(row)
    return rows


def get_training_preset(name: str) -> dict | None:
    body = _TRAINING_PRESETS.get(name)
    if body is None:
        return None
    row = {"name": name}
    row.update(body)
    return row


def assert_single_base() -> None:
    """Guard: exactly one named base profile, named as specified.

    Raises AssertionError if a second named realism base profile is ever
    introduced, or if the name/checkpoint drift from the Feature 11 spec.
    """
    names = list_base_profiles()
    assert len(names) == 1, f"expected exactly one base profile, got {names}"
    assert names[0] == BASE_PROFILE_NAME, f"unexpected base profile: {names}"
    assert _BASE_PROFILE["base_checkpoint"] == BASE_CHECKPOINT


def register_base_profile(output_dir=None, dry_run: bool = True) -> dict:
    """Produce a registration manifest for the personal base profile.

    Non-destructive: when ``output_dir`` is provided and ``dry_run`` is False
    the manifest is written to ``personalized_base_manifest.json`` inside it,
    otherwise the manifest is only returned (planning mode).
    """
    from pathlib import Path
    assert_single_base()
    manifest = {
        "profile_name": BASE_PROFILE_NAME,
        "base_checkpoint": BASE_CHECKPOINT,
        "registered": not dry_run,
        "dry_run": bool(dry_run),
        "profile": get_base_profile(),
        "training_presets": list_training_presets(),
        "warnings": [],
    }
    if output_dir is not None and not dry_run:
        import json
        out = Path(output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        path = out / "personalized_base_manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(path)
    return manifest
