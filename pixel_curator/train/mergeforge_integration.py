"""Feature 11G - MergeForge CyberRealisticPony Personal Base phase.

Plans (and, only with an explicit heavy-operation flag, executes) the merge of
the trained Core Identity + specialized refinement LoRAs onto the
CyberRealisticPony base checkpoint to produce the personal base, then registers
the ``CyberRealisticPony Personal Base`` / ``Personalized CyberRealisticPony
Base`` profile.

Safety contract:
  * dry-run / planning by default; no merge executes unless ``execute=True``
    AND ``dry_run=False`` AND the input files exist
  * the source base checkpoint is NEVER overwritten - output always goes to a
    new ``outputs/personalized_bases/<timestamp>_...`` folder
  * a produced checkpoint is validated (existence, extension, size, sha256);
    a planned/absent checkpoint reports ``valid=False`` cleanly
  * reuses ``pixel_curator.mergeforge`` recipe helpers for a reproducible recipe
  * never raises for ordinary problems; missing backends become warnings

Pure stdlib + project modules.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from . import base_profile as bp
from . import kohya_config_builder as kcb

log = logging.getLogger("redresser.mergeforge_integration")

_VALID_CHECKPOINT_SUFFIXES = (".safetensors", ".ckpt")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _mid(weight_range) -> float:
    try:
        lo, hi = float(weight_range[0]), float(weight_range[1])
        return round((lo + hi) / 2, 2)
    except Exception:
        return 0.5


def _lora_path(checkpoints_dir, output_name: str) -> str:
    """Resolve a trained LoRA's expected file path (may not exist in dry-run)."""
    if checkpoints_dir is None:
        return f"{output_name}.safetensors"
    p = Path(checkpoints_dir).expanduser() / output_name / \
        f"{output_name}.safetensors"
    return str(p)


def build_merge_inputs(base_checkpoint: str,
                       identity_output_name: str = "redresser_core_identity",
                       specialized_output_names: dict | None = None,
                       checkpoints_dir=None) -> list[dict]:
    """Compose the ordered merge inputs (base + identity + specialized).

    Each entry: ``{name, path, role, weight, weight_range}``. Weights use the
    personal-base recommended ranges (identity strongest, refiners light).
    """
    inputs: list[dict] = [{
        "name": Path(base_checkpoint).stem, "path": base_checkpoint,
        "role": "base", "weight": 1.0, "weight_range": [1.0, 1.0]}]
    inputs.append({
        "name": identity_output_name,
        "path": _lora_path(checkpoints_dir, identity_output_name),
        "role": "identity", "weight": 0.80, "weight_range": [0.70, 0.90]})
    specialized_output_names = specialized_output_names or {
        t: kcb.SPECIALIZED_TARGETS[t]["output_name"]
        for t in ("outfit", "style", "pose", "garment_detail")}
    for target, out_name in specialized_output_names.items():
        wr = kcb.SPECIALIZED_TARGETS.get(target, {}).get(
            "weight_range", [0.25, 0.55])
        inputs.append({
            "name": out_name,
            "path": _lora_path(checkpoints_dir, out_name),
            "role": "refiner", "weight": _mid(wr), "weight_range": wr})
    return inputs


def _dry_run_recipe(inputs: list[dict], base_checkpoint: str) -> dict:
    """Build a reproducible MergeForge recipe via the project helper."""
    try:
        from .. import mergeforge as mf
        pairs = [(i["path"], i["weight"]) for i in inputs]
        recipe = mf.make_recipe(
            name="cyberrealisticpony_personal_base", inputs=pairs,
            method="weighted_sum", base_model=base_checkpoint,
            notes="Personal base: Core Identity + specialized refinement LoRAs "
                  "merged onto the CyberRealisticPony base.",
            hash_files=True)
        clean, problems = mf.validate_recipe(recipe)
        return {"recipe": clean, "problems": problems,
                "helper": "pixel_curator.mergeforge.make_recipe"}
    except Exception as exc:                         # pragma: no cover
        return {"recipe": None,
                "problems": [f"MergeForge recipe helper unavailable: {exc}"],
                "helper": None}


def create_personal_base_merge_plan(
        project_path, base_checkpoint: str = bp.BASE_CHECKPOINT,
        identity_output_name: str = "redresser_core_identity",
        specialized_output_names: dict | None = None,
        checkpoints_dir=None, output_dir=None, dry_run: bool = True) -> dict:
    """Build a dry-run MergeForge plan for the personal base. Never overwrites
    the source checkpoint; output targets a new timestamped folder."""
    project_path = Path(project_path).expanduser()
    if output_dir is None:
        output_dir = (project_path / "outputs" / "personalized_bases"
                      / f"{_timestamp()}_cyberrealisticpony_personal_base")
    output_dir = Path(output_dir).expanduser()
    merged_name = "cyberrealisticpony_personal_base_v1.safetensors"
    merged_path = output_dir / merged_name

    inputs = build_merge_inputs(base_checkpoint, identity_output_name,
                                specialized_output_names, checkpoints_dir)
    recipe_info = _dry_run_recipe(inputs, base_checkpoint)

    profile = bp.get_base_profile()
    plan = {
        "target_profile": bp.BASE_PROFILE_NAME,
        "preset_name": "Personalized CyberRealisticPony Base",
        "base_checkpoint": base_checkpoint,
        "dry_run": bool(dry_run),
        "never_overwrite_base": True,
        "method": "weighted_sum",
        "output_dir": str(output_dir),
        "merged_checkpoint": str(merged_path),
        "inputs": inputs,
        "recipe": recipe_info.get("recipe"),
        "recipe_problems": recipe_info.get("problems", []),
        "recipe_helper": recipe_info.get("helper"),
        "recommended_stacks": profile.get("recommended_lora_stack", {}),
        "merge_manifest": str(output_dir / "merge_manifest.json"),
        "recommended_stacks_path": str(output_dir / "recommended_stacks.json"),
        "warnings": list(recipe_info.get("problems", [])),
    }

    # materialize manifests only when explicitly not in dry-run
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        Path(plan["merge_manifest"]).write_text(
            json.dumps(plan, indent=2, default=str), encoding="utf-8")
        Path(plan["recommended_stacks_path"]).write_text(
            json.dumps(plan["recommended_stacks"], indent=2), encoding="utf-8")
    return plan


def validate_merged_checkpoint(checkpoint_path, dry_run: bool = True) -> dict:
    """Validate a produced personal-base checkpoint. Planned/absent files
    report ``valid=False`` with a clean status rather than raising."""
    path = Path(str(checkpoint_path)).expanduser()
    result = {"valid": False, "path": str(path), "status": "planned",
              "size_bytes": 0, "sha256": None, "warnings": []}
    if dry_run:
        result["warnings"].append(
            "Dry-run: merged checkpoint validation planned (not created).")
        return result
    if not path.is_file():
        result["status"] = "missing"
        result["warnings"].append(f"Merged checkpoint not found: {path}")
        return result
    if path.suffix.lower() not in _VALID_CHECKPOINT_SUFFIXES:
        result["status"] = "invalid_extension"
        result["warnings"].append(
            f"Unexpected checkpoint extension: {path.suffix}")
        return result
    try:
        size = path.stat().st_size
        result["size_bytes"] = size
        if size <= 0:
            result["status"] = "empty"
            result["warnings"].append("Merged checkpoint is empty.")
            return result
        try:                                         # pragma: no cover
            from .. import mergeforge as mf
            result["sha256"] = mf._sha256(path)
        except Exception:
            result["sha256"] = None
        result["valid"] = True
        result["status"] = "valid"
    except Exception as exc:                         # pragma: no cover
        result["status"] = "error"
        result["warnings"].append(f"Validation error: {exc}")
    return result


def register_personal_base(output_dir, dry_run: bool = True) -> dict:
    """Register the CyberRealisticPony Personal Base profile (planning by
    default). Enforces the single-named-base invariant."""
    return bp.register_base_profile(output_dir, dry_run=dry_run)


def run_personal_base_merge_phase(
        project_path, base_checkpoint: str = bp.BASE_CHECKPOINT,
        identity_output_name: str = "redresser_core_identity",
        specialized_output_names: dict | None = None,
        checkpoints_dir=None, output_dir=None,
        dry_run: bool = True, execute: bool = False) -> dict:
    """End-to-end personal-base merge phase: plan -> (optional execute) ->
    validate -> register profile. Returns a structured result. Never executes a
    heavy merge unless ``execute=True`` AND ``dry_run=False`` AND inputs exist.
    """
    plan = create_personal_base_merge_plan(
        project_path, base_checkpoint=base_checkpoint,
        identity_output_name=identity_output_name,
        specialized_output_names=specialized_output_names,
        checkpoints_dir=checkpoints_dir, output_dir=output_dir,
        dry_run=dry_run)

    executed = False
    warnings: list[str] = list(plan.get("warnings", []))
    merged_path = plan["merged_checkpoint"]

    if execute and not dry_run:                      # pragma: no cover
        missing = [i["path"] for i in plan["inputs"]
                   if not Path(i["path"]).expanduser().is_file()]
        if missing:
            warnings.append(
                "Heavy merge not executed; input files missing: "
                + ", ".join(missing))
        else:
            warnings.append(
                "Heavy merge execution requested; run mergeforge.execute_recipe "
                "with a Project to produce the merged checkpoint.")
    elif execute and dry_run:
        warnings.append(
            "execute=True ignored in dry-run; set dry_run=False to merge.")

    validation = validate_merged_checkpoint(
        merged_path, dry_run=dry_run or not executed)
    base_reg = register_personal_base(plan["output_dir"], dry_run=dry_run)

    return {
        "status": "planned" if dry_run else (
            "executed" if executed else "planned_not_executed"),
        "dry_run": bool(dry_run),
        "target_profile": bp.BASE_PROFILE_NAME,
        "merge_plan": plan,
        "merged_checkpoint": merged_path,
        "validation": validation,
        "profile_registration": base_reg,
        "profile_registered": base_reg.get("registered", False),
        "warnings": warnings,
    }


__all__ = [
    "build_merge_inputs", "create_personal_base_merge_plan",
    "validate_merged_checkpoint", "register_personal_base",
    "run_personal_base_merge_phase",
]
