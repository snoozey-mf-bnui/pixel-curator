"""Priority 11 - final training recipe builder.

Converts curated/exported/packaged datasets into reproducible, deterministic
training recipes (readiness score + target-specific hyperparameters + Kohya
config + Colab runbook). It only reads the JSON artifacts the earlier
priorities already wrote (dataset_card.json, dataset_manifest.json,
package_manifest.json, anchor_report.json) and writes plan files.

Contract: additive, source-safe (never modifies datasets/packages), default
DRY-RUN (returns the recipe; ``write_training_recipe`` is the only writer),
deterministic, stdlib-only, never launches training / downloads models, never
raises for ordinary input.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Canonical base profile (the project's single allowed named realism base).
try:
    from .base_profile import (BASE_PROFILE_NAME, BASE_CHECKPOINT,
                               BASE_CHECKPOINT_PATH)
except Exception:                                        # pragma: no cover
    BASE_PROFILE_NAME = "CyberRealisticPony Personal Base"
    BASE_CHECKPOINT = "cyberrealisticPony_v180Coreshift.safetensors"
    BASE_CHECKPOINT_PATH = (
        "~/FORGE_REPO/stable-diffusion-webui-reForge/models/Stable-diffusion/"
        "cyberrealisticPony_v180Coreshift.safetensors")

READINESS_LABELS = ("ready", "review_recommended", "not_ready")

# Conservative per-target hyperparameter defaults (dim, alpha, mode).
_TARGET_DEFAULTS = {
    "core_identity":     {"dim": 32, "alpha": 16, "mode": "lora", "keep_tokens": 1},
    "outfit_clothing":   {"dim": 32, "alpha": 16, "mode": "lora", "keep_tokens": 1},
    "pose_action":       {"dim": 16, "alpha": 8,  "mode": "lora", "keep_tokens": 1},
    "style_aesthetic":   {"dim": 16, "alpha": 8,  "mode": "lora", "keep_tokens": 1},
    "garment_detail":    {"dim": 16, "alpha": 8,  "mode": "lora", "keep_tokens": 1},
    "catalog_reference": {"dim": 32, "alpha": 16, "mode": "lora", "keep_tokens": 1},
}


@dataclass
class TrainingRecipeConfig:
    target: str = "core_identity"
    mode: str = "lora"                  # lora | dreambooth | hybrid
    base_model_profile: str = "cyberrealistic_pony"
    resolution: int = 1024
    network_dim: int = 32
    network_alpha: int = 16
    repeats: int = 1
    epochs: int = 10
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    unet_lr: float | None = None
    text_encoder_lr: float | None = None
    optimizer: str = "adamw8bit"
    scheduler: str = "cosine"
    mixed_precision: str = "fp16"
    save_every_n_epochs: int = 1
    seed: int = 42
    dry_run: bool = True

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "target", "mode", "base_model_profile", "resolution",
            "network_dim", "network_alpha", "repeats", "epochs",
            "train_batch_size", "gradient_accumulation_steps", "learning_rate",
            "unet_lr", "text_encoder_lr", "optimizer", "scheduler",
            "mixed_precision", "save_every_n_epochs", "seed", "dry_run")}


@dataclass
class DatasetTrainingReadiness:
    dataset_dir: str
    package_dir: str | None = None
    target: str = "unknown"
    sample_count: int = 0
    readiness_score: float = 0.0
    readiness_label: str = "not_ready"
    recommended_mode: str = "lora"
    recommended_rank: int = 32
    recommended_alpha: int = 16
    recommended_repeats: int = 1
    warnings: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset_dir": self.dataset_dir, "package_dir": self.package_dir,
            "target": self.target, "sample_count": self.sample_count,
            "readiness_score": round(float(self.readiness_score), 4),
            "readiness_label": self.readiness_label,
            "recommended_mode": self.recommended_mode,
            "recommended_rank": self.recommended_rank,
            "recommended_alpha": self.recommended_alpha,
            "recommended_repeats": self.recommended_repeats,
            "warnings": list(self.warnings), "metrics": self.metrics,
        }


@dataclass
class TrainingRecipe:
    name: str
    target: str
    mode: str
    base_model_profile: str
    dataset_dir: str
    package_dir: str | None
    output_name: str
    config: TrainingRecipeConfig
    readiness: DatasetTrainingReadiness
    kohya_config: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "target": self.target, "mode": self.mode,
            "base_model_profile": self.base_model_profile,
            "dataset_dir": self.dataset_dir, "package_dir": self.package_dir,
            "output_name": self.output_name, "config": self.config.to_dict(),
            "readiness": self.readiness.to_dict(),
            "kohya_config": self.kohya_config, "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_json(p: Path) -> dict:
    try:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _infer_target(dataset_dir: Path, card: dict, manifest: dict) -> str:
    for src in (card.get("summary", {}), manifest):
        t = src.get("target")
        if t:
            return str(t)
    name = dataset_dir.name.lower()
    for t in _TARGET_DEFAULTS:
        if t in name:
            return t
    return "core_identity"


def _recommend_repeats(n: int) -> int:
    if n <= 0:
        return 1
    return max(1, min(40, round(150 / n)))


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------
def evaluate_dataset_training_readiness(dataset_dir, package_dir=None
                                        ) -> DatasetTrainingReadiness:
    """Score a curated/packaged dataset for training readiness from its JSON
    artifacts. Deterministic; never raises."""
    dataset_dir = Path(dataset_dir)
    card = _load_json(dataset_dir / "dataset_card.json")
    manifest = _load_json(dataset_dir / "dataset_manifest.json")
    pkg = _load_json(Path(package_dir) / "package_manifest.json") \
        if package_dir else {}
    target = _infer_target(dataset_dir, card, manifest)

    samples = manifest.get("samples") or []
    n = manifest.get("sample_count") or len(samples) or \
        pkg.get("sample_count") or 0
    res = DatasetTrainingReadiness(
        dataset_dir=str(dataset_dir),
        package_dir=str(package_dir) if package_dir else None, target=target,
        sample_count=int(n))

    cap = card.get("caption_summary", {})
    qual = card.get("quality_summary", {})
    region = card.get("region_quality_summary", {})
    cov = card.get("coverage_framing_summary", {})
    posed = card.get("pose_duplicate_summary", {})
    sess = (card.get("metrics", {}) or {}).get("session_stats", {}) \
        or card.get("session_stats", {})
    validation = card.get("validation", {})

    score = 1.0
    w = res.warnings
    metrics: dict = {}

    if n < 15:
        score -= 0.25
        w.append("training_sample_count_low")
    metrics["sample_count"] = n

    rich = cap.get("richness_score")
    if rich is not None:
        metrics["caption_richness"] = rich
        if rich < 0.40:
            score -= 0.10
            w.append("training_caption_richness_low")
    if cap.get("duplicate_ratio") is not None:
        metrics["caption_duplicate_ratio"] = cap["duplicate_ratio"]
        if cap["duplicate_ratio"] > 0.25:
            score -= 0.08
            w.append("training_duplicate_ratio_high")
    if pkg.get("caption_stats"):
        metrics["package_caption_stats"] = pkg["caption_stats"]

    near = posed.get("near_duplicate_ratio")
    if near is not None:
        metrics["pose_near_duplicate_ratio"] = near
        if near > 0.25:
            score -= 0.10
            w.append("training_pose_duplicate_ratio_high")
    if sess.get("top_session_ratio") is not None:
        metrics["top_session_ratio"] = sess["top_session_ratio"]
        if sess["top_session_ratio"] > 0.35:
            score -= 0.08
            w.append("training_session_dominance_high")

    integ = qual.get("subject_integrity_avg")
    if integ is not None:
        metrics["subject_integrity_avg"] = integ
        if integ < 0.55:
            score -= 0.10
            w.append("training_subject_integrity_low")
    face = region.get("avg_face_region_score")
    if face is not None:
        metrics["avg_face_region_score"] = face
        if face < 0.55:
            score -= 0.08
            w.append("training_region_quality_low")
    if cov.get("dominant_framing_ratio") is not None:
        metrics["dominant_framing_ratio"] = cov["dominant_framing_ratio"]

    # anchor quality matters most for identity targets
    if target == "core_identity" and package_dir:
        anchor = _load_json(Path(package_dir) / "anchor_report.json")
        if anchor:
            acc = anchor.get("accepted", 0)
            metrics["anchor_accepted"] = acc
            if acc < 20:
                score -= 0.10
                w.append("training_anchor_quality_low")

    vw = validation.get("warnings") if isinstance(validation, dict) else None
    if vw:
        blocking = [x for x in vw if (x.get("severity") if isinstance(x, dict)
                    else "warning") == "warning"]
        if blocking:
            metrics["validation_warning_count"] = len(blocking)
            score -= min(0.10, 0.02 * len(blocking))
            w.append("training_validation_warnings_present")

    score = max(0.0, min(1.0, round(score, 4)))
    res.readiness_score = score
    res.readiness_label = ("ready" if score >= 0.70 else
                           "review_recommended" if score >= 0.45 else
                           "not_ready")
    res.metrics = metrics

    d = _TARGET_DEFAULTS.get(target, _TARGET_DEFAULTS["core_identity"])
    res.recommended_mode = d["mode"]
    res.recommended_rank = d["dim"]
    res.recommended_alpha = d["alpha"]
    res.recommended_repeats = _recommend_repeats(n)
    # identity sets with strong, plentiful refs can justify dreambooth
    if target == "core_identity" and n >= 40 and score >= 0.7:
        res.recommended_mode = "dreambooth"
    return res


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------
def _kohya_config_dict(recipe_cfg: TrainingRecipeConfig, output_name: str,
                       checkpoint: str) -> dict:
    """Kohya sd-scripts config. LoRA mode -> ``sdxl_train_network.py`` with a
    ``[network_arguments]`` section; DreamBooth/hybrid (full fine-tune) ->
    ``sdxl_train.py`` with NO network section and a much lower learning rate."""
    unet = recipe_cfg.unet_lr or recipe_cfg.learning_rate
    te = recipe_cfg.text_encoder_lr or (recipe_cfg.learning_rate / 2.0)
    train_args = {
        "output_name": output_name,
        "resolution": recipe_cfg.resolution,
        "train_batch_size": recipe_cfg.train_batch_size,
        "max_train_epochs": recipe_cfg.epochs,
        "learning_rate": recipe_cfg.learning_rate,
        "unet_lr": unet, "text_encoder_lr": te,
        "lr_scheduler": recipe_cfg.scheduler,
        "optimizer_type": recipe_cfg.optimizer,
        "mixed_precision": recipe_cfg.mixed_precision,
        "save_every_n_epochs": recipe_cfg.save_every_n_epochs,
        "seed": recipe_cfg.seed,
        "gradient_accumulation_steps": recipe_cfg.gradient_accumulation_steps,
    }
    cfg = {
        "model_arguments": {"pretrained_model_name_or_path": checkpoint},
        "dataset_arguments": {"dataset_config": "dataset_config.toml"},
        "training_arguments": train_args,
    }
    if recipe_cfg.mode in ("dreambooth", "hybrid"):
        # full fine-tune: sd-scripts sdxl_train.py, no LoRA network section.
        cfg["_trainer_script"] = "sdxl_train.py"
        train_args["gradient_checkpointing"] = True
        train_args["cache_latents"] = True
        train_args["enable_bucket"] = True
        # prior preservation keeps the broad "person" class from collapsing
        # onto the subject during a full fine-tune.
        cfg["dreambooth_arguments"] = {
            "prior_loss_weight": 1.0,
            "note": "pair with a regularization/class set "
                    "(export-dreambooth --reg)",
        }
    else:
        cfg["_trainer_script"] = "sdxl_train_network.py"
        cfg["network_arguments"] = {
            "network_module": "networks.lora",
            "network_dim": recipe_cfg.network_dim,
            "network_alpha": recipe_cfg.network_alpha,
        }
    return cfg


def build_training_recipe(dataset_dir, package_dir=None,
                          config: TrainingRecipeConfig | None = None
                          ) -> TrainingRecipe:
    """Build a deterministic training recipe from a curated/packaged dataset.
    Readiness drives recommended hyperparameters unless ``config`` overrides
    them. Never launches training. Never raises for ordinary input."""
    dataset_dir = Path(dataset_dir)
    readiness = evaluate_dataset_training_readiness(dataset_dir, package_dir)
    target = readiness.target
    cfg = config or TrainingRecipeConfig(target=target)
    # apply recommendations when the caller left defaults in place
    if config is None:
        cfg.target = target
        cfg.mode = readiness.recommended_mode
        cfg.network_dim = readiness.recommended_rank
        cfg.network_alpha = readiness.recommended_alpha
        cfg.repeats = readiness.recommended_repeats
        cfg.base_model_profile = BASE_PROFILE_NAME
    # full fine-tune needs a far lower LR than LoRA; only override when the
    # caller left the LoRA default in place so explicit configs are respected.
    if cfg.mode in ("dreambooth", "hybrid") and cfg.learning_rate == 1e-4:
        cfg.learning_rate = 1e-6
        if cfg.epochs == 10:
            cfg.epochs = 6              # full fine-tune converges in fewer epochs
    output_name = f"redresser_{target}"
    # local Forge/reForge checkpoint path as the placeholder hint (Colab users
    # repoint this to their Drive copy of the same checkpoint).
    kohya = _kohya_config_dict(cfg, output_name, BASE_CHECKPOINT_PATH)
    warnings = list(readiness.warnings)
    return TrainingRecipe(
        name=f"{target}_recipe", target=target, mode=cfg.mode,
        base_model_profile=cfg.base_model_profile, dataset_dir=str(dataset_dir),
        package_dir=str(package_dir) if package_dir else None,
        output_name=output_name, config=cfg, readiness=readiness,
        kohya_config=kohya, warnings=warnings)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def _render_kohya_toml(kohya: dict) -> str:
    def _v(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int,)) and not isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return repr(v)
        return '"' + str(v).replace('"', '\\"') + '"'
    lines = []
    for section, body in kohya.items():
        # meta keys (e.g. _trainer_script) render as a comment, not a table
        if section.startswith("_"):
            lines.append(f"# {section.lstrip('_')} = {body}")
            continue
        lines.append(f"[{section}]")
        for k, val in body.items():
            if k == "note":
                lines.append(f"# note: {val}")
            else:
                lines.append(f"{k} = {_v(val)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_plan_md(recipe: TrainingRecipe) -> str:
    r, c = recipe.readiness, recipe.config
    L = [f"# Training Plan: {recipe.name}", "",
         "## Summary",
         f"- Target: {recipe.target}",
         f"- Mode: {recipe.mode}",
         f"- Dataset: {recipe.dataset_dir}",
         f"- Package: {recipe.package_dir or '(none)'}",
         f"- Base model: {recipe.base_model_profile}",
         f"- Output name: {recipe.output_name}",
         f"- Readiness score: {r.readiness_score}",
         f"- Readiness label: {r.readiness_label}", "",
         "## Dataset Readiness",
         f"- Sample count: {r.sample_count}",
         f"- Metrics: {json.dumps(r.metrics)}",
         f"- Warnings: {', '.join(r.warnings) or 'none'}", "",
         "## Recommended Hyperparameters",
         "| Parameter | Value |", "|---|---:|",
         f"| mode | {c.mode} |",
         f"| network_dim | {c.network_dim} |",
         f"| network_alpha | {c.network_alpha} |",
         f"| repeats | {c.repeats} |",
         f"| epochs | {c.epochs} |",
         f"| resolution | {c.resolution} |",
         f"| learning_rate | {c.learning_rate} |",
         f"| optimizer | {c.optimizer} |",
         f"| scheduler | {c.scheduler} |",
         f"| mixed_precision | {c.mixed_precision} |",
         f"| seed | {c.seed} |", "",
         "## Kohya / Colab Files",
         f"- dataset_config.toml: package `dataset_config.toml`",
         f"- kohya_config: kohya_config_{recipe.target}.toml",
         f"- package: {recipe.package_dir or '(build with: package kohya)'}",
         "",
         "## Next Commands",
         "```bash",
         "# 1. Package the curated export (Priority 10)",
         f"python -m pixel_curator package kohya --dataset {recipe.dataset_dir} \\",
         f"    --output <PACKAGE_DIR> --mode {recipe.mode} --apply",
         "# 2. Upload the package zip to Colab and point the notebook's",
         f"#    dataset_config at dataset_config.toml (see colab_runbook.md)",
         "```", ""]
    return "\n".join(L)


def _render_colab_runbook(recipe: TrainingRecipe) -> str:
    c = recipe.config
    return "\n".join([
        f"# Colab Runbook: {recipe.name}", "",
        "Placeholders: `/content/datasets/<package_name>` (extracted package), "
        "`/content/drive/MyDrive/LoRA_Output` (checkpoints).", "",
        "## SDXL DreamBooth Colab",
        "1. Upload the package zip and extract to "
        "`/content/datasets/<package_name>`.",
        "2. Set `DATASET_ROOT` to that folder (it holds `dataset_config.toml` "
        "+ `instance/`).",
        f"3. Set `DATASET_CONFIG = \"dataset_config.toml\"`, "
        f"`RESOLUTION = {c.resolution}`, `MAX_TRAIN_EPOCHS = {c.epochs}`, "
        f"`LEARNING_RATE = {c.learning_rate}`.",
        "4. Run the train cell (runs from inside the dataset folder).",
        "5. Checkpoints save to `/content/drive/MyDrive/LoRA_Output`.", "",
        "## SDXL LoRA Colab",
        "1. Upload a zip whose subfolders each have `dataset_config.toml` + "
        "`instance/` (use `--mode lora`).",
        f"2. The notebook auto-discovers concepts; per-concept dim/alpha "
        f"recommended {c.network_dim}/{c.network_alpha}.",
        f"3. epochs {c.epochs}, batch {c.train_batch_size}, "
        f"optimizer {c.optimizer}, scheduler {c.scheduler}.",
        "4. Train all concepts in one session; LoRAs save per-epoch to Drive.",
        "",
    ])


def write_training_recipe(recipe: TrainingRecipe, output_dir) -> dict:
    """Write the plan artifacts. Returns ``{name: Path}``. Source datasets and
    packages are never touched. Never raises for ordinary input."""
    output_dir = Path(output_dir)
    paths: dict = {}
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        plan_json = output_dir / "training_plan.json"
        plan_json.write_text(json.dumps(recipe.to_dict(), indent=2),
                             encoding="utf-8")
        paths["training_plan_json"] = plan_json
        plan_md = output_dir / "training_plan.md"
        plan_md.write_text(_render_plan_md(recipe), encoding="utf-8")
        paths["training_plan_md"] = plan_md
        toml = output_dir / f"kohya_config_{recipe.target}.toml"
        toml.write_text(_render_kohya_toml(recipe.kohya_config),
                        encoding="utf-8")
        paths["kohya_config_toml"] = toml
        cfg_json = output_dir / f"training_config_{recipe.target}.json"
        cfg_json.write_text(json.dumps(recipe.kohya_config, indent=2),
                            encoding="utf-8")
        paths["training_config_json"] = cfg_json
        runbook = output_dir / "colab_runbook.md"
        runbook.write_text(_render_colab_runbook(recipe), encoding="utf-8")
        paths["colab_runbook_md"] = runbook
        # single-recipe stack manifest (extended by stack_planner)
        stack = output_dir / "stack_manifest.json"
        stack.write_text(json.dumps({
            "schema_version": "training_stack_v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "recipes": [recipe.to_dict()],
        }, indent=2), encoding="utf-8")
        paths["stack_manifest_json"] = stack
    except Exception as exc:                             # never crash
        recipe.warnings.append(f"write_failed: {exc}")
    return paths


__all__ = [
    "READINESS_LABELS", "TrainingRecipeConfig", "DatasetTrainingReadiness",
    "TrainingRecipe", "evaluate_dataset_training_readiness",
    "build_training_recipe", "write_training_recipe",
]
