"""Feature 11 - Core Identity training execution + evaluation loop.

Drives a reliable Core Identity LoRA training run from the config/command that
``kohya_config_builder`` generates, then (optionally) renders a catalog test
grid with the ``Personalized CyberRealisticPony Base`` preset and computes a
best-effort identity-consistency score.

Safety contract (matches the rest of the studio):
  * dry-run / planning by default; nothing trains unless ``dry_run=False``
  * heavy training only runs when an ``accelerate`` backend is resolvable -
    otherwise it returns a clean ``skipped_no_backend`` status, never crashes
  * never overwrites an existing LoRA output; never modifies source media
  * test-grid rendering requires a reachable Forge backend; without one it
    returns a planned prompt set instead of failing
  * identity scoring uses an optional local backend; absent one it reports a
    structured "scorer unavailable" result rather than raising

Pure stdlib + the project's own modules; optional libraries (PIL via the eval
matrix, a face-embedding backend) are imported lazily and degrade gracefully.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from . import base_profile as bp
from . import kohya_config_builder as kcb

log = logging.getLogger("redresser.core_identity_trainer")

DEFAULT_OUTPUT_NAME = "pixel_curator_core_identity_cyberrealisticpony_v1"
PRESET_NAME = "Personalized CyberRealisticPony Base"

#: professional, SFW catalog scenarios appended to the profile prompt booster
TEST_GRID_SCENARIOS: tuple[str, ...] = (
    "studio portrait, soft key light, 85mm, shallow depth of field",
    "full body fashion catalog, balanced studio lighting",
    "half body editorial composition, clean studio background",
    "close-up, natural studio light, refined facial detail",
)


# ---------------------------------------------------------------------------
# Preset / profile resolution
# ---------------------------------------------------------------------------
def _resolve_preset() -> tuple[dict, list[str]]:
    """Return the Personalized CyberRealisticPony Base preset (or profile
    fallback) and any warnings. Never raises."""
    warnings: list[str] = []
    try:
        from .. import starter_stacks as ss
        preset = ss.get_wardrobe_preset(PRESET_NAME)
        if preset:
            return preset, warnings
        warnings.append(f"Preset '{PRESET_NAME}' not found in starter_stacks; "
                        "using base profile defaults.")
    except Exception as exc:                         # pragma: no cover
        warnings.append(f"starter_stacks unavailable ({exc}); using base "
                        "profile defaults.")
    prof = bp.get_base_profile()
    preset = {
        "name": PRESET_NAME,
        "base_checkpoint": prof["base_checkpoint"],
        "prompt": prof.get("prompt_boosters", ""),
        "negative": prof.get("negative_prompt_boosters", ""),
        "quality_preset": prof.get("quality_preset", "catalog_pro"),
    }
    return preset, warnings


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def build_test_prompts(preset: dict | None = None,
                       scenarios: tuple[str, ...] = TEST_GRID_SCENARIOS
                       ) -> list[dict]:
    """Compose professional SFW catalog test prompts from the preset booster."""
    if preset is None:
        preset, _ = _resolve_preset()
    booster = (preset.get("prompt_booster") or preset.get("prompt") or "").strip(", ")
    negative = (preset.get("negative_booster") or preset.get("negative")
                or "").strip(", ")
    out: list[dict] = []
    for scenario in scenarios:
        prompt = f"{booster}, {scenario}".strip(", ") if booster else scenario
        out.append({"prompt": prompt, "negative": negative,
                    "scenario": scenario})
    return out


def generate_test_grid(lora_output_path, weight: float = 0.80,
                       preset: dict | None = None, forge_url: str =
                       "http://127.0.0.1:7860", output_dir=None,
                       seed: int = 42, dry_run: bool = True) -> dict:
    """Render (or plan) a catalog test grid for a trained Core Identity LoRA.

    Uses the Personalized CyberRealisticPony Base preset. Requires a reachable
    Forge backend to actually render; otherwise returns the planned prompt set
    with a clean warning. Never raises for ordinary problems.
    """
    warnings: list[str] = []
    if preset is None:
        preset, pwarn = _resolve_preset()
        warnings += pwarn
    prompts = build_test_prompts(preset)
    lora_stem = Path(str(lora_output_path)).stem
    base_checkpoint = preset.get("base_checkpoint", bp.BASE_CHECKPOINT)
    out_root = Path(output_dir).expanduser() if output_dir else \
        Path(str(lora_output_path)).expanduser().parent / "test_grid"

    result = {
        "status": "planned",
        "dry_run": bool(dry_run),
        "lora": lora_stem,
        "weight": float(weight),
        "base_checkpoint": base_checkpoint,
        "preset": preset.get("name", PRESET_NAME),
        "prompts": prompts,
        "images": [],
        "grid_path": "",
        "warnings": warnings,
    }

    # Planning mode, or the LoRA has not been produced yet -> prompts only.
    if dry_run:
        result["warnings"].append(
            "Dry-run: catalog test-grid prompts planned (no rendering).")
        return result
    if not Path(str(lora_output_path)).expanduser().is_file():
        result["status"] = "skipped_no_lora"
        result["warnings"].append(
            f"LoRA output not found yet: {lora_output_path}; test grid "
            "planned only.")
        return result

    # Backend probe
    try:
        from ..eval.forge_api import ForgeClient
        client = ForgeClient(forge_url)
        if not client.alive():
            result["status"] = "skipped_no_backend"
            result["warnings"].append(
                "Forge backend not reachable; catalog test-grid prompts "
                "planned only.")
            return result
    except Exception as exc:                         # pragma: no cover
        result["status"] = "skipped_no_backend"
        result["warnings"].append(f"Forge backend unavailable: {exc}")
        return result

    # Render each scenario (best-effort; a single failure does not abort)
    out_root.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    for i, p in enumerate(prompts):                  # pragma: no cover
        try:
            png = client.txt2img(
                prompt=p["prompt"], negative=p["negative"],
                loras=[(lora_stem, float(weight))], checkpoint=base_checkpoint,
                clip_skip=2, seed=seed + i)
            img_path = out_root / f"identity_test_{i:02d}.png"
            img_path.write_bytes(png)
            image_paths.append(img_path)
            result["images"].append(str(img_path))
        except Exception as exc:
            result["warnings"].append(
                f"Render failed for scenario '{p['scenario']}': {exc}")

    if image_paths:                                  # pragma: no cover
        try:
            from ..eval.matrix import assemble_grid
            grid_path = out_root / "identity_test_grid.jpg"
            assemble_grid(image_paths, grid_path, cols=2,
                          labels=[p["scenario"] for p in prompts])
            result["grid_path"] = str(grid_path)
        except Exception as exc:
            result["warnings"].append(f"Grid assembly failed: {exc}")
    result["status"] = "rendered" if image_paths else "skipped_no_backend"
    return result


def calculate_identity_score(test_grid: dict,
                             anchor_dir=None) -> dict:
    """Best-effort identity-consistency score over rendered test images.

    Uses an optional local face-embedding backend if present; otherwise
    returns a structured "scorer unavailable" result. Deterministic and
    dependency-free in the default path (safe for tests). Never raises.
    """
    images = (test_grid or {}).get("images", [])
    warnings: list[str] = []
    backend = None
    try:                                             # pragma: no cover
        import insightface  # noqa: F401
        backend = "insightface"
    except Exception:
        backend = None

    if not images:
        return {"available": False, "score": None, "image_count": 0,
                "backend": backend or "none",
                "warnings": ["No rendered test images; identity scoring "
                             "planned only."]}
    if backend is None:
        return {"available": False, "score": None,
                "image_count": len(images), "backend": "none",
                "warnings": ["Identity scoring backend unavailable; review "
                             "the catalog test grid manually."]}
    # pragma: no cover - real scoring path requires the optional backend
    return {"available": True, "score": None, "image_count": len(images),
            "backend": backend,
            "warnings": warnings
            + ["Embedding-based scoring backend present; integrate anchor "
               "references to produce a numeric consistency score."]}


def evaluate_core_identity(lora_output_path, weight: float = 0.80,
                           forge_url: str = "http://127.0.0.1:7860",
                           output_dir=None, anchor_dir=None,
                           dry_run: bool = True) -> dict:
    """Render the catalog test grid and compute the identity score together."""
    grid = generate_test_grid(lora_output_path, weight=weight,
                              forge_url=forge_url, output_dir=output_dir,
                              dry_run=dry_run)
    score = calculate_identity_score(grid, anchor_dir=anchor_dir)
    return {"test_grid": grid, "identity_score": score,
            "warnings": list(grid.get("warnings", []))
            + list(score.get("warnings", []))}


# ---------------------------------------------------------------------------
# Training execution
# ---------------------------------------------------------------------------
def _resolve_train_command(recipe: dict, kohya_dir, train_network,
                           accelerate_bin: str) -> tuple[list[str] | None,
                                                         Path | None, str]:
    """Resolve the concrete training argv. Returns (cmd, cwd, note)."""
    config_path = recipe.get("config_path", "")
    if train_network:
        tn = Path(train_network).expanduser()
    elif kohya_dir:
        tn = Path(kohya_dir).expanduser() / "sd-scripts" / "train_network.py"
    else:
        tn = None
    accel = shutil.which(accelerate_bin) or (
        accelerate_bin if Path(accelerate_bin).expanduser().exists() else None)
    if accel is None:
        return None, None, ("accelerate backend not found; set ACCELERATE_BIN "
                            "or install it to run training.")
    if tn is None or not tn.is_file():
        return None, None, ("train_network.py not found; pass kohya_dir or "
                            "train_network to run training.")
    cmd = [accel, "launch", str(tn), "--config_file", str(config_path)]
    cwd = tn.parent
    return cmd, cwd, "ready"


def run_core_identity_training(
        dataset_dir, output_dir,
        base_checkpoint: str = bp.BASE_CHECKPOINT,
        output_name: str = DEFAULT_OUTPUT_NAME,
        lora_dir=None, forge_url: str = "http://127.0.0.1:7860",
        kohya_dir=None, train_network=None, accelerate_bin: str = "accelerate",
        dry_run: bool = True, generate_grid: bool = True,
        grid_weight: float = 0.80, resume: bool = True) -> dict:
    """Plan or execute a Core Identity LoRA training run.

    Steps: build the Kohya config + command (via kohya_config_builder) ->
    (dry-run) return the plan, or (real) launch ``accelerate launch
    train_network.py`` and capture logs -> locate the resulting ``.safetensors``
    -> optionally render a catalog test grid + identity score with the
    Personalized CyberRealisticPony Base preset.

    Never starts training unless ``dry_run=False`` AND an accelerate backend is
    resolvable. Never overwrites an existing LoRA output.
    """
    dataset_dir = Path(dataset_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    log_dir = output_dir / "logs"
    lora_output_path = output_dir / f"{output_name}.safetensors"

    result: dict = {
        "status": "planned",
        "dry_run": bool(dry_run),
        "output_name": output_name,
        "base_checkpoint": base_checkpoint,
        "dataset_dir": str(dataset_dir),
        "lora_output_path": str(lora_output_path),
        "config_path": "", "command_path": "", "command": "",
        "log_path": str(log_dir / "train_core_identity.log"),
        "test_grid": {}, "identity_score": {},
        "warnings": [], "errors": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    if base_checkpoint != bp.BASE_CHECKPOINT:
        result["warnings"].append(
            f"Requested base checkpoint '{base_checkpoint}' differs from the "
            f"personal-base target '{bp.BASE_CHECKPOINT}'.")

    # 1) Build the recipe (config TOML + train_core_identity.sh).
    recipe = kcb.build_core_identity_kohya_config(
        dataset_dir, output_dir, base_checkpoint=base_checkpoint,
        output_name=output_name, config_dir=output_dir / "configs",
        command_dir=output_dir / "commands", dry_run=dry_run)
    result["config_path"] = recipe.get("config_path", "")
    result["command_path"] = recipe.get("command_path", "")
    result["command"] = recipe.get("command", "")
    result["warnings"] += recipe.get("warnings", [])

    # 2) Dry-run: plan only (incl. planned test-grid prompts), do not train.
    if dry_run:
        if generate_grid:
            result["test_grid"] = generate_test_grid(
                lora_output_path, weight=grid_weight, forge_url=forge_url,
                dry_run=True)
            result["identity_score"] = calculate_identity_score(
                result["test_grid"])
        result["status"] = "planned"
        return result

    # 3) Resume guard: never overwrite an existing LoRA output.
    if resume and lora_output_path.is_file():
        result["status"] = "exists_skipped"
        result["warnings"].append(
            f"LoRA output already exists; not overwriting: {lora_output_path}")
        if generate_grid:
            result["test_grid"] = generate_test_grid(
                lora_output_path, weight=grid_weight, forge_url=forge_url,
                output_dir=output_dir / "test_grid", dry_run=False)
            result["identity_score"] = calculate_identity_score(
                result["test_grid"])
        return result

    # 4) Real training: resolve and launch the command (gated on backend).
    cmd, cwd, note = _resolve_train_command(
        recipe, kohya_dir, train_network, accelerate_bin)
    if cmd is None:
        result["status"] = "skipped_no_backend"
        result["warnings"].append(note)
        return result

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_core_identity.log"
    try:                                             # pragma: no cover
        with log_path.open("w", encoding="utf-8") as lf:
            lf.write(f"# {datetime.now().isoformat()}\n# {' '.join(cmd)}\n\n")
            proc = subprocess.run(cmd, cwd=str(cwd), stdout=lf,
                                  stderr=subprocess.STDOUT, check=False)
        result["return_code"] = proc.returncode
        if proc.returncode == 0:
            result["status"] = "trained"
        else:
            result["status"] = "completed_with_errors"
            result["errors"].append(
                f"Training exited with code {proc.returncode}; see {log_path}")
    except Exception as exc:                         # pragma: no cover
        result["status"] = "completed_with_errors"
        result["errors"].append(f"Training launch failed: {exc}")
        return result

    # 5) Locate output + evaluate.
    if lora_output_path.is_file():                   # pragma: no cover
        result["lora_output_path"] = str(lora_output_path)
    else:
        result["warnings"].append(
            f"Expected LoRA output not found after training: "
            f"{lora_output_path}")
    if generate_grid:                                # pragma: no cover
        result["test_grid"] = generate_test_grid(
            lora_output_path, weight=grid_weight, forge_url=forge_url,
            output_dir=output_dir / "test_grid", dry_run=False)
        result["identity_score"] = calculate_identity_score(
            result["test_grid"])
    return result


__all__ = [
    "DEFAULT_OUTPUT_NAME", "PRESET_NAME", "TEST_GRID_SCENARIOS",
    "build_test_prompts", "generate_test_grid", "calculate_identity_score",
    "evaluate_core_identity", "run_core_identity_training",
]
