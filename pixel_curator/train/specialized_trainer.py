"""Feature 11F - Specialized LoRA training execution for refinement targets.

Drives reliable training runs for the specialized refinement LoRAs that layer
on top of the Core Identity anchor:

    outfit  (outfit_clothing)   -> redresser_outfit_refinement
    style   (style_aesthetic)   -> redresser_style_refinement
    pose    (pose_action)       -> redresser_pose_refinement
    garment_detail              -> redresser_garment_detail_refinement

Each run: verifies the exported target dataset, generates the Kohya config +
training script (conservative hyperparameters + prompt booster from
``kohya_config_builder.SPECIALIZED_TARGETS``), and - only when explicitly
requested and an accelerate backend is resolvable - launches training.

Safety contract (same as the Core Identity trainer):
  * dry-run / planning by default; nothing trains unless ``dry_run=False``
  * heavy training only runs with a resolvable ``accelerate`` backend, else a
    clean ``skipped_no_backend`` status; never crashes
  * never overwrites an existing LoRA output; never modifies source media
  * runs sequentially, never in parallel, after Core Identity

Pure stdlib + project modules; reuses the Core Identity trainer's command
resolution and evaluation helpers so behaviour stays consistent.
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path

from . import base_profile as bp
from . import kohya_config_builder as kcb
from . import core_identity_trainer as cit

log = logging.getLogger("redresser.specialized_trainer")

#: canonical sequential order for specialized refinement training
SPECIALIZED_ORDER: tuple[str, ...] = (
    "outfit", "style", "pose", "garment_detail")

#: DatasetSplitter labels -> specialized training-target labels
SPLIT_TO_TARGET: dict[str, str] = {
    "outfit_clothing": "outfit",
    "style_aesthetic": "style",
    "pose_action": "pose",
    "garment_detail": "garment_detail",
}


def _verify_dataset(dataset_dir: Path) -> tuple[int, list[str]]:
    """Count usable images in an exported target dataset. Never raises."""
    warnings: list[str] = []
    dataset_dir = Path(dataset_dir).expanduser()
    if not dataset_dir.is_dir():
        warnings.append(f"Dataset directory not present yet: {dataset_dir}")
        return 0, warnings
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    n = sum(1 for p in dataset_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in exts)
    if n == 0:
        warnings.append(f"No exported images found in {dataset_dir}.")
    return n, warnings


def run_specialized_training(
        target: str, dataset_dir, output_dir,
        base_checkpoint: str = bp.BASE_CHECKPOINT,
        lora_dir=None, forge_url: str = "http://127.0.0.1:7860",
        kohya_dir=None, train_network=None, accelerate_bin: str = "accelerate",
        dry_run: bool = True, generate_grid: bool = False,
        resume: bool = True) -> dict:
    """Plan or execute one specialized refinement LoRA training run.

    Mirrors ``core_identity_trainer.run_core_identity_training`` but uses the
    target-specific conservative recipe (network dim/alpha, weight range,
    prompt booster, evaluation plan) from ``SPECIALIZED_TARGETS``.
    """
    spec = kcb.SPECIALIZED_TARGETS.get(target)
    dataset_dir = Path(dataset_dir).expanduser()
    output_dir = Path(output_dir).expanduser()

    result: dict = {
        "status": "planned",
        "dry_run": bool(dry_run),
        "target": target,
        "base_checkpoint": base_checkpoint,
        "dataset_dir": str(dataset_dir),
        "output_name": "", "lora_output_path": "",
        "config_path": "", "command_path": "", "command": "",
        "recommended_network_dim": None,
        "recommended_weight_range": None,
        "prompt_booster": "", "evaluation_plan": "",
        "dataset_image_count": 0,
        "log_path": str(output_dir / "logs" / f"train_{target}_lora.log"),
        "test_grid": {},
        "warnings": [], "errors": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    if spec is None:
        result["status"] = "skipped_unknown_target"
        result["warnings"].append(
            f"No specialized recipe for target '{target}'.")
        return result

    output_name = spec["output_name"]
    lora_output_path = output_dir / f"{output_name}.safetensors"
    result["output_name"] = output_name
    result["lora_output_path"] = str(lora_output_path)

    # 1) dataset export verification
    n, dwarn = _verify_dataset(dataset_dir)
    result["dataset_image_count"] = n
    result["warnings"] += dwarn

    # 2) build the specialized Kohya config + training script
    recipe = kcb.build_specialized_kohya_config(
        target, dataset_dir, output_dir, base_checkpoint=base_checkpoint,
        config_dir=output_dir / "configs", command_dir=output_dir / "commands",
        dry_run=dry_run)
    result["config_path"] = recipe.get("config_path", "")
    result["command_path"] = recipe.get("command_path", "")
    result["command"] = recipe.get("command", "")
    result["recommended_network_dim"] = recipe.get("recommended_network_dim")
    result["recommended_weight_range"] = recipe.get("recommended_weight_range")
    result["prompt_booster"] = recipe.get("prompt_booster", "")
    result["evaluation_plan"] = recipe.get("evaluation_plan", "")
    result["warnings"] += recipe.get("warnings", [])

    # 3) dry-run: plan only
    if dry_run:
        result["status"] = "planned"
        return result

    # 4) resume guard: never overwrite an existing LoRA output
    if resume and lora_output_path.is_file():
        result["status"] = "exists_skipped"
        result["warnings"].append(
            f"LoRA output already exists; not overwriting: {lora_output_path}")
        return result

    # 5) real training (gated on a resolvable accelerate backend)
    cmd, cwd, note = cit._resolve_train_command(
        recipe, kohya_dir, train_network, accelerate_bin)
    if cmd is None:
        result["status"] = "skipped_no_backend"
        result["warnings"].append(note)
        return result

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{target}_lora.log"
    try:                                             # pragma: no cover
        with log_path.open("w", encoding="utf-8") as lf:
            lf.write(f"# {datetime.now().isoformat()}\n# {' '.join(cmd)}\n\n")
            proc = subprocess.run(cmd, cwd=str(cwd), stdout=lf,
                                  stderr=subprocess.STDOUT, check=False)
        result["return_code"] = proc.returncode
        result["status"] = "trained" if proc.returncode == 0 \
            else "completed_with_errors"
        if proc.returncode != 0:
            result["errors"].append(
                f"Training exited with code {proc.returncode}; see {log_path}")
    except Exception as exc:                         # pragma: no cover
        result["status"] = "completed_with_errors"
        result["errors"].append(f"Training launch failed: {exc}")
        return result

    if not lora_output_path.is_file():               # pragma: no cover
        result["warnings"].append(
            f"Expected LoRA output not found after training: "
            f"{lora_output_path}")
    if generate_grid:                                # pragma: no cover
        wr = spec["weight_range"]
        result["test_grid"] = cit.generate_test_grid(
            lora_output_path, weight=round(sum(wr) / 2, 2),
            forge_url=forge_url, output_dir=output_dir / "test_grid",
            dry_run=False)
    return result


def run_all_specialized_training(
        dataset_root, output_root,
        targets: tuple[str, ...] = SPECIALIZED_ORDER,
        base_checkpoint: str = bp.BASE_CHECKPOINT,
        dataset_dirs: dict | None = None,
        dry_run: bool = True, generate_grid: bool = False,
        resume: bool = True, **kw) -> dict:
    """Run specialized training for each target **sequentially**.

    ``dataset_dirs`` optionally maps target -> exported image dir; otherwise
    ``<dataset_root>/<target>/images`` is used. Returns a structured summary.
    """
    dataset_root = Path(dataset_root).expanduser()
    output_root = Path(output_root).expanduser()
    dataset_dirs = dataset_dirs or {}
    runs: dict[str, dict] = {}
    warnings: list[str] = []
    for target in targets:
        if target not in kcb.SPECIALIZED_TARGETS:
            warnings.append(f"Skipping unknown specialized target '{target}'.")
            continue
        ds = Path(dataset_dirs.get(target,
                                   dataset_root / target / "images"))
        out_dir = output_root / kcb.SPECIALIZED_TARGETS[target]["output_name"]
        runs[target] = run_specialized_training(
            target, ds, out_dir, base_checkpoint=base_checkpoint,
            dry_run=dry_run, generate_grid=generate_grid, resume=resume, **kw)
        warnings += [w for w in runs[target].get("warnings", [])
                     if w not in warnings]
    return {
        "status": "planned" if dry_run else "completed",
        "dry_run": bool(dry_run),
        "order": list(targets),
        "runs": runs,
        "training_commands": {t: r.get("command", "")
                              for t, r in runs.items()},
        "warnings": warnings,
    }


__all__ = [
    "SPECIALIZED_ORDER", "SPLIT_TO_TARGET", "run_specialized_training",
    "run_all_specialized_training",
]
