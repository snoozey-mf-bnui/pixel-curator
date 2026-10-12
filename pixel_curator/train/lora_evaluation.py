"""Feature 11 - robust LoRA evaluation + iteration loop.

Enhanced, professional SFW evaluation tooling for a trained personal LoRA:

  * automated test-grid generation across seeds x catalog poses x garments
  * identity-consistency scoring (optional local backend; structured fallback)
  * before/after visual comparison grids (base checkpoint vs LoRA, same seed)
  * a single ``evaluate_lora()`` entry point that produces a saved report and
    integrates with the Test Generation card via the UI route

Safety contract:
  * dry-run / planning by default; rendering requires a reachable Forge backend
    and is otherwise reported as a planned matrix (never crashes)
  * non-destructive: writes only into the evaluation output folder
  * uses only professional fashion/catalog terminology in prompts and labels

Reuses ``core_identity_trainer`` (prompt/score helpers), ``eval.matrix``
(grid assembly), and ``qa_render`` (side-by-side composite). Pure stdlib +
project modules; PIL/Forge are imported lazily and degrade gracefully.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from . import base_profile as bp
from . import core_identity_trainer as cit

log = logging.getLogger("redresser.lora_evaluation")

#: professional SFW catalog pose variations
POSE_VARIATIONS: tuple[str, ...] = (
    "standing studio pose, balanced proportions",
    "seated editorial pose, relaxed posture",
    "three-quarter turn, catalog stance",
    "walking pose, dynamic catalog motion",
)

#: professional SFW garment / wardrobe variations
GARMENT_VARIATIONS: tuple[str, ...] = (
    "tailored blazer and trousers",
    "structured summer dress",
    "layered knitwear ensemble",
    "athleisure activewear set",
)

#: default evaluation seeds (deterministic, reproducible)
DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 777)


def _resolve_preset():
    return cit._resolve_preset()


def build_evaluation_matrix(preset: dict | None = None,
                            poses: tuple[str, ...] = POSE_VARIATIONS,
                            garments: tuple[str, ...] = GARMENT_VARIATIONS,
                            seeds: tuple[int, ...] = DEFAULT_SEEDS,
                            max_cells: int = 12) -> list[dict]:
    """Compose the seed x pose x garment evaluation matrix.

    Pairs poses with garments index-wise (so the matrix stays catalog-focused
    rather than combinatorially huge), then repeats across seeds. Capped at
    ``max_cells`` for a readable grid.
    """
    if preset is None:
        preset, _ = _resolve_preset()
    booster = (preset.get("prompt_booster") or preset.get("prompt")
               or "").strip(", ")
    negative = (preset.get("negative_booster") or preset.get("negative")
                or "").strip(", ")
    pairs = []
    n = max(len(poses), len(garments))
    for i in range(n):
        pose = poses[i % len(poses)] if poses else ""
        garment = garments[i % len(garments)] if garments else ""
        pairs.append((pose, garment))
    cells: list[dict] = []
    for seed in seeds:
        for pose, garment in pairs:
            scenario = ", ".join(p for p in (garment, pose) if p)
            prompt = f"{booster}, {scenario}".strip(", ") if booster \
                else scenario
            cells.append({
                "prompt": prompt, "negative": negative, "seed": int(seed),
                "pose": pose, "garment": garment,
                "label": f"s{seed} | {garment[:18]}"})
            if len(cells) >= max_cells:
                return cells
    return cells


def _probe_backend(forge_url: str):
    try:
        from ..eval.forge_api import ForgeClient
        client = ForgeClient(forge_url)
        return client if client.alive() else None
    except Exception:                                # pragma: no cover
        return None


def generate_evaluation_grid(
        lora_path, weight: float = 0.80, preset: dict | None = None,
        poses: tuple[str, ...] = POSE_VARIATIONS,
        garments: tuple[str, ...] = GARMENT_VARIATIONS,
        seeds: tuple[int, ...] = DEFAULT_SEEDS,
        forge_url: str = "http://127.0.0.1:7860", output_dir=None,
        base_checkpoint: str | None = None, dry_run: bool = True) -> dict:
    """Render (or plan) a multi-seed / pose / garment evaluation grid."""
    warnings: list[str] = []
    if preset is None:
        preset, pwarn = _resolve_preset()
        warnings += pwarn
    base_checkpoint = base_checkpoint or preset.get(
        "base_checkpoint", bp.BASE_CHECKPOINT)
    lora_stem = Path(str(lora_path)).stem
    matrix = build_evaluation_matrix(preset, poses, garments, seeds)
    out_root = Path(output_dir).expanduser() if output_dir else \
        Path(str(lora_path)).expanduser().parent / "evaluation"

    result = {
        "status": "planned", "dry_run": bool(dry_run), "lora": lora_stem,
        "weight": float(weight), "base_checkpoint": base_checkpoint,
        "preset": preset.get("name", cit.PRESET_NAME),
        "cells": matrix, "images": [], "grid_path": "", "warnings": warnings,
    }
    if dry_run:
        result["warnings"].append(
            "Dry-run: evaluation matrix planned (no rendering).")
        return result
    if not Path(str(lora_path)).expanduser().is_file():
        result["status"] = "skipped_no_lora"
        result["warnings"].append(f"LoRA output not found: {lora_path}")
        return result
    client = _probe_backend(forge_url)
    if client is None:
        result["status"] = "skipped_no_backend"
        result["warnings"].append(
            "Forge backend not reachable; evaluation matrix planned only.")
        return result

    out_root.mkdir(parents=True, exist_ok=True)     # pragma: no cover
    paths = []
    for i, cell in enumerate(matrix):                # pragma: no cover
        try:
            png = client.txt2img(
                prompt=cell["prompt"], negative=cell["negative"],
                loras=[(lora_stem, float(weight))], checkpoint=base_checkpoint,
                clip_skip=2, seed=cell["seed"])
            p = out_root / f"eval_{i:02d}.png"
            p.write_bytes(png)
            paths.append(p)
            result["images"].append(str(p))
        except Exception as exc:
            result["warnings"].append(f"Render failed (cell {i}): {exc}")
    if paths:                                        # pragma: no cover
        try:
            from ..eval.matrix import assemble_grid
            grid = out_root / "evaluation_grid.jpg"
            assemble_grid(paths, grid, cols=len(seeds) or 3,
                          labels=[c["label"] for c in matrix])
            result["grid_path"] = str(grid)
        except Exception as exc:
            result["warnings"].append(f"Grid assembly failed: {exc}")
    result["status"] = "rendered" if paths else "skipped_no_backend"
    return result


def calculate_identity_consistency(grid: dict, anchor_dir=None) -> dict:
    """Score identity consistency across the evaluation images.

    Wraps the Core Identity scorer (optional local backend) and adds a
    per-seed grouping summary. Deterministic and dependency-free in the
    default path. Never raises.
    """
    base = cit.calculate_identity_score(grid or {}, anchor_dir=anchor_dir)
    cells = (grid or {}).get("cells", [])
    seeds = sorted({c.get("seed") for c in cells if "seed" in c})
    base["seeds_evaluated"] = seeds
    base["cell_count"] = len(cells)
    base.setdefault("warnings", [])
    if base.get("available") is False and grid.get("images"):
        base["warnings"].append(
            "Review the evaluation grid manually for identity consistency "
            "across seeds, poses, and garments.")
    return base


def build_before_after_grid(before_images: list, after_images: list,
                            out_path, labels: list | None = None,
                            dry_run: bool = True) -> dict:
    """Assemble a before/after (base vs LoRA) comparison composite."""
    out_path = Path(out_path).expanduser()
    result = {"status": "planned", "dry_run": bool(dry_run),
              "grid_path": str(out_path), "pairs": 0, "warnings": []}
    pairs = min(len(before_images or []), len(after_images or []))
    result["pairs"] = pairs
    if dry_run or pairs == 0:
        if pairs == 0:
            result["warnings"].append(
                "No before/after image pairs available; comparison planned "
                "only.")
        return result
    try:                                             # pragma: no cover
        from ..eval.matrix import assemble_grid
        interleaved = []
        lbls = []
        for i in range(pairs):
            interleaved += [Path(before_images[i]), Path(after_images[i])]
            base_lbl = (labels[i] if labels and i < len(labels) else f"pair {i}")
            lbls += [f"base: {base_lbl}", f"LoRA: {base_lbl}"]
        assemble_grid(interleaved, out_path, cols=2, labels=lbls)
        result["status"] = "rendered"
    except Exception as exc:                         # pragma: no cover
        result["warnings"].append(f"Comparison grid assembly failed: {exc}")
    return result


def evaluate_lora(lora_path, weight: float = 0.80,
                  preset: dict | None = None,
                  poses: tuple[str, ...] = POSE_VARIATIONS,
                  garments: tuple[str, ...] = GARMENT_VARIATIONS,
                  seeds: tuple[int, ...] = DEFAULT_SEEDS,
                  forge_url: str = "http://127.0.0.1:7860",
                  output_dir=None, base_checkpoint: str | None = None,
                  compare_base: bool = True, anchor_dir=None,
                  dry_run: bool = True) -> dict:
    """Top-level LoRA evaluation: test grid + identity score + before/after.

    Produces a structured report (and writes ``evaluation_report.json`` when
    not dry-run). This is the ``evaluate_lora`` command. Never raises for
    ordinary problems; missing backends degrade to a planned report.
    """
    if preset is None:
        preset, _ = _resolve_preset()
    out_root = Path(output_dir).expanduser() if output_dir else \
        Path(str(lora_path)).expanduser().parent / "evaluation"

    grid = generate_evaluation_grid(
        lora_path, weight=weight, preset=preset, poses=poses,
        garments=garments, seeds=seeds, forge_url=forge_url,
        output_dir=out_root / "lora", base_checkpoint=base_checkpoint,
        dry_run=dry_run)

    # before/after: a base-only render (no LoRA) at the same seeds/scenarios
    before_after = {"status": "planned", "pairs": 0, "warnings": []}
    base_grid = {"status": "planned", "images": []}
    if compare_base:
        base_grid = generate_evaluation_grid(
            "__base__", weight=0.0, preset=preset, poses=poses,
            garments=garments, seeds=seeds, forge_url=forge_url,
            output_dir=out_root / "base", base_checkpoint=base_checkpoint,
            dry_run=dry_run)
        before_after = build_before_after_grid(
            base_grid.get("images", []), grid.get("images", []),
            out_root / "before_after_grid.jpg",
            labels=[c["label"] for c in grid.get("cells", [])],
            dry_run=dry_run or not grid.get("images"))

    identity = calculate_identity_consistency(grid, anchor_dir=anchor_dir)

    report = {
        "status": "planned" if dry_run else grid.get("status", "planned"),
        "dry_run": bool(dry_run),
        "lora": Path(str(lora_path)).stem,
        "weight": float(weight),
        "preset": preset.get("name", cit.PRESET_NAME),
        "base_checkpoint": base_checkpoint or preset.get(
            "base_checkpoint", bp.BASE_CHECKPOINT),
        "seeds": list(seeds),
        "poses": list(poses),
        "garments": list(garments),
        "test_grid": grid,
        "before_after": before_after,
        "identity_consistency": identity,
        "warnings": list(grid.get("warnings", []))
        + list(before_after.get("warnings", []))
        + list(identity.get("warnings", [])),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if not dry_run:                                  # pragma: no cover
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "evaluation_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        report["report_path"] = str(out_root / "evaluation_report.json")
    return report


def evaluate_lora_command(lora_path: str, weight: float = 0.80,
                          forge_url: str = "http://127.0.0.1:7860",
                          output_dir=None, dry_run: bool = True) -> dict:
    """CLI/UI-friendly wrapper for the evaluate_lora command."""
    return evaluate_lora(lora_path, weight=weight, forge_url=forge_url,
                         output_dir=output_dir, dry_run=dry_run)


__all__ = [
    "POSE_VARIATIONS", "GARMENT_VARIATIONS", "DEFAULT_SEEDS",
    "build_evaluation_matrix", "generate_evaluation_grid",
    "calculate_identity_consistency", "build_before_after_grid",
    "evaluate_lora", "evaluate_lora_command",
]
