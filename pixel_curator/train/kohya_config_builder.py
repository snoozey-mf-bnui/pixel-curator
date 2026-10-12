"""Feature 11C/11D/11E/11F - Kohya config, captions and versioned exports.

This module turns classified dataset splits into concrete, non-destructive
training assets:

* professional captions per LoRA target (11D)
* versioned, copy-plan dataset exports per target (11C)
* a conservative core-identity Kohya ``train_config.toml`` (11E)
* specialized LoRA config plans for outfit / style / pose / garment_detail
  with recommended network dimension, weight range, prompt boosters and an
  evaluation plan (11F)
* the exact ``accelerate launch`` training command string

Everything defaults to dry-run / planning: no source media is copied, moved,
or modified, and existing dataset versions are never overwritten.

Pure stdlib, local-first.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .base_profile import BASE_CHECKPOINT
from .dataset_classifier import TRAINABLE_TARGETS

log = logging.getLogger("redresser.kohya_config_builder")


# ---------------------------------------------------------------------------
# 11D - Professional captions per training target. SFW, catalog terminology.
# ---------------------------------------------------------------------------
CAPTION_TEMPLATES: dict[str, str] = {
    "core_identity": (
        "personal_identity_anchor, professional catalog reference, clear "
        "facial structure, consistent identity, studio-ready framing"),
    "outfit": (
        "personal_identity_anchor, wardrobe variation, fashion catalog "
        "styling, refined garment rendering, clean fabric boundaries"),
    "style": (
        "personal_identity_anchor, catalog photography style, controlled "
        "studio lighting, polished editorial composition"),
    "pose": (
        "personal_identity_anchor, fashion model pose, balanced body "
        "proportions, catalog-ready composition"),
    "garment_detail": (
        "personal_identity_anchor, garment detail study, refined textile "
        "structure, fabric interaction, clean garment edges"),
    "catalog_reference": (
        "personal_identity_anchor, fashion catalog reference, polished "
        "studio presentation, consistent identity"),
}


def caption_for_target(target: str, sample: dict | None = None) -> dict:
    """Return the caption for a target, honouring a manual override.

    A sample may carry ``caption_override`` (string) which is used verbatim;
    otherwise the professional template for the target is returned. Result
    records the caption ``source`` and ``target`` for the manifest.
    """
    sample = sample or {}
    override = sample.get("caption_override")
    if isinstance(override, str) and override.strip():
        return {"caption": override.strip(), "source": "manual_override",
                "target": target}
    base = CAPTION_TEMPLATES.get(target, CAPTION_TEMPLATES["catalog_reference"])
    return {"caption": base, "source": "template", "target": target}


# ---------------------------------------------------------------------------
# 11F - Specialized LoRA target recommendations.
# ---------------------------------------------------------------------------
SPECIALIZED_TARGETS: dict[str, dict] = {
    "outfit": {
        "output_name": "redresser_outfit_refinement",
        "network_dim": 24, "network_alpha": 12,
        "weight_range": [0.50, 0.75],
        "prompt_booster": CAPTION_TEMPLATES["outfit"],
        "evaluation_plan": "catalog consistency on held-out wardrobe samples",
    },
    "style": {
        "output_name": "redresser_style_refinement",
        "network_dim": 16, "network_alpha": 8,
        "weight_range": [0.25, 0.45],
        "prompt_booster": CAPTION_TEMPLATES["style"],
        "evaluation_plan": "editorial style adherence on catalog references",
    },
    "pose": {
        "output_name": "redresser_pose_refinement",
        "network_dim": 16, "network_alpha": 8,
        "weight_range": [0.30, 0.55],
        "prompt_booster": CAPTION_TEMPLATES["pose"],
        "evaluation_plan": "pose fidelity and body proportion balance",
    },
    "garment_detail": {
        "output_name": "redresser_garment_detail_refinement",
        "network_dim": 24, "network_alpha": 12,
        "weight_range": [0.30, 0.50],
        "prompt_booster": CAPTION_TEMPLATES["garment_detail"],
        "evaluation_plan": "textile structure and garment edge sharpness",
    },
}


# ---------------------------------------------------------------------------
# 11C - Versioned, non-destructive dataset export plan.
# ---------------------------------------------------------------------------
def _next_version(target: str, output_root: Path,
                  requested: str | None) -> str:
    if requested:
        return requested if requested.lower().startswith("v") else f"v{requested}"
    n = 1
    while (output_root / f"v{n:03d}_{target}").exists():
        n += 1
    return f"v{n:03d}"


def export_lora_target_dataset(samples: list[dict], target: str,
                               output_root, version: str | None = None,
                               dry_run: bool = True) -> dict:
    """Plan (or, when ``dry_run`` is False, perform) a versioned export.

    The export folder is ``<output_root>/<version>_<target>/`` containing
    ``images/``, ``captions/`` and ``dataset_manifest.json``. Source paths are
    preserved in the manifest; captions are written only inside the export
    folder. Missing source files are skipped with a warning. An existing
    dataset version is never overwritten.
    """
    output_root = Path(output_root).expanduser()
    warnings: list[str] = []
    version = _next_version(target, output_root, version)
    export_dir = output_root / f"{version}_{target}"

    plan_items: list[dict] = []
    skipped = 0
    for row in samples or []:
        src = row.get("sample_path") or row.get("path") or ""
        cap = caption_for_target(target, row)
        exists = bool(src) and Path(src).expanduser().is_file()
        if not exists:
            skipped += 1
            warnings.append(f"Source file missing, skipped: {src or '<empty>'}")
        stem = Path(src).stem or f"sample_{len(plan_items):04d}"
        plan_items.append({
            "source_path": str(src),
            "source_exists": exists,
            "image_target": f"images/{stem}{Path(src).suffix or '.png'}",
            "caption_target": f"captions/{stem}.txt",
            "caption": cap["caption"],
            "caption_source": cap["source"],
            "confidence": row.get("confidence"),
            "reason_codes": row.get("reason_codes", []),
        })

    manifest = {
        "target": target,
        "version": version,
        "export_dir": str(export_dir),
        "dry_run": bool(dry_run),
        "sample_count": len(plan_items),
        "exportable_count": len(plan_items) - skipped,
        "skipped_missing": skipped,
        "caption_extension": ".txt",
        "items": plan_items,
        "warnings": warnings,
    }

    if export_dir.exists() and not dry_run:
        warnings.append(
            f"Dataset version already exists; not overwriting: {export_dir}")
        manifest["status"] = "exists_skipped"
        return manifest

    if dry_run:
        manifest["status"] = "planned"
        return manifest

    # Real export: copy is non-destructive (source untouched).
    import shutil
    (export_dir / "images").mkdir(parents=True, exist_ok=True)
    (export_dir / "captions").mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in plan_items:
        if item["source_exists"]:
            try:
                shutil.copy2(item["source_path"],
                             export_dir / item["image_target"])
                copied += 1
            except Exception as exc:                # pragma: no cover
                warnings.append(f"copy failed for {item['source_path']}: {exc}")
        (export_dir / item["caption_target"]).write_text(
            item["caption"], encoding="utf-8")
    manifest["copied_images"] = copied
    manifest["status"] = "exported"
    (export_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(export_dir / "dataset_manifest.json")
    return manifest


# ---------------------------------------------------------------------------
# 11E - Kohya train_config.toml generation + command string.
# ---------------------------------------------------------------------------
def _conservative_config(dataset_dir: Path, output_dir: Path,
                         base_checkpoint: str, output_name: str,
                         network_dim: int, network_alpha: int,
                         max_train_epochs: int) -> dict:
    return {
        "pretrained_model_name_or_path": base_checkpoint,
        "train_data_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "output_name": output_name,
        "network_module": "networks.lora",
        "resolution": "1024,1024",
        "train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "max_train_epochs": max_train_epochs,
        "save_every_n_epochs": 2,
        "learning_rate": 1e-4,
        "unet_lr": 1e-4,
        "text_encoder_lr": 5e-5,
        "lr_scheduler": "cosine",
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "mixed_precision": "fp16",
        "save_precision": "fp16",
        "cache_latents": True,
        "caption_extension": ".txt",
        "shuffle_caption": True,
        "keep_tokens": 1,
    }


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        # keep scientific notation readable for learning rates
        return repr(v)
    return '"' + str(v).replace('"', '\\"') + '"'


def render_toml(config: dict, section: str = "") -> str:
    """Render a flat config dict to TOML text (stdlib only, deterministic)."""
    lines: list[str] = []
    if section:
        lines.append(f"[{section}]")
    for key, val in config.items():
        lines.append(f"{key} = {_toml_value(val)}")
    return "\n".join(lines) + "\n"


def build_training_command(config_path: Path | str) -> str:
    """Return the exact accelerate launch command for a config file."""
    return (f"accelerate launch train_network.py "
            f"--config_file {config_path}")


def _write_config(config: dict, config_dir: Path, command_dir: Path,
                  toml_name: str, sh_name: str,
                  dry_run: bool) -> dict:
    config_dir = Path(config_dir).expanduser()
    command_dir = Path(command_dir).expanduser()
    config_path = config_dir / toml_name
    command_path = command_dir / sh_name
    toml_text = render_toml(config)
    command = build_training_command(config_path)
    warnings: list[str] = []
    if not dry_run:
        config_dir.mkdir(parents=True, exist_ok=True)
        command_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(toml_text, encoding="utf-8")
        command_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                                + command + "\n", encoding="utf-8")
    else:
        warnings.append("Dry-run: config TOML and command script planned only.")
    return {
        "config_path": str(config_path),
        "command_path": str(command_path),
        "command": command,
        "toml_text": toml_text,
        "config": config,
        "warnings": warnings,
    }


def build_core_identity_kohya_config(
        dataset_dir, output_dir,
        base_checkpoint: str = BASE_CHECKPOINT,
        output_name: str = "redresser_core_identity",
        config_dir=None, command_dir=None, dry_run: bool = True) -> dict:
    """Build the conservative core-identity Kohya config (Feature 11E)."""
    dataset_dir = Path(dataset_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    config_dir = Path(config_dir).expanduser() if config_dir else output_dir.parent / "configs"
    command_dir = Path(command_dir).expanduser() if command_dir else output_dir.parent / "commands"
    cfg = _conservative_config(dataset_dir, output_dir, base_checkpoint,
                               output_name, network_dim=32, network_alpha=16,
                               max_train_epochs=10)
    return _write_config(cfg, config_dir, command_dir,
                         "core_identity_train_config.toml",
                         "train_core_identity.sh", dry_run)


def build_specialized_kohya_config(
        target: str, dataset_dir, output_dir,
        base_checkpoint: str = BASE_CHECKPOINT,
        config_dir=None, command_dir=None, dry_run: bool = True) -> dict:
    """Build a specialized LoRA config plan for outfit/style/pose/garment."""
    spec = SPECIALIZED_TARGETS.get(target)
    if spec is None:
        return {"warnings": [f"No specialized recipe for target '{target}'."],
                "config_path": "", "command_path": "", "command": "",
                "target": target}
    dataset_dir = Path(dataset_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    config_dir = Path(config_dir).expanduser() if config_dir else output_dir.parent / "configs"
    command_dir = Path(command_dir).expanduser() if command_dir else output_dir.parent / "commands"
    cfg = _conservative_config(
        dataset_dir, output_dir, base_checkpoint, spec["output_name"],
        network_dim=spec["network_dim"], network_alpha=spec["network_alpha"],
        max_train_epochs=8)
    result = _write_config(
        cfg, config_dir, command_dir,
        f"{target}_train_config.toml", f"train_{target}_lora.sh", dry_run)
    result.update({
        "target": target,
        "output_name": spec["output_name"],
        "recommended_network_dim": spec["network_dim"],
        "recommended_weight_range": spec["weight_range"],
        "prompt_booster": spec["prompt_booster"],
        "evaluation_plan": spec["evaluation_plan"],
    })
    return result


def build_all_specialized_configs(targets, dataset_root, output_root,
                                  base_checkpoint: str = BASE_CHECKPOINT,
                                  config_dir=None, command_dir=None,
                                  dry_run: bool = True) -> dict:
    """Build specialized config plans for each requested trainable target."""
    out: dict[str, dict] = {}
    for target in targets or []:
        if target not in SPECIALIZED_TARGETS:
            continue
        ds = Path(dataset_root).expanduser() / target
        od = Path(output_root).expanduser() / SPECIALIZED_TARGETS[target][
            "output_name"]
        out[target] = build_specialized_kohya_config(
            target, ds, od, base_checkpoint, config_dir, command_dir, dry_run)
    return out


__all__ = [
    "CAPTION_TEMPLATES", "SPECIALIZED_TARGETS", "TRAINABLE_TARGETS",
    "caption_for_target", "export_lora_target_dataset", "render_toml",
    "build_training_command", "build_core_identity_kohya_config",
    "build_specialized_kohya_config", "build_all_specialized_configs",
]
