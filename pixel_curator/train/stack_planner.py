"""Priority 11 - base-model profile + LoRA stack planner.

Aggregates one or more training recipes (Priority 11A) into a reproducible
multi-LoRA stack plan: conservative per-target stack weights, a recommended
generation order, and a Forge/reForge profile JSON + Colab runbook. It never
launches Forge, downloads models, or trains anything - it writes plan files
only.

The only named realism base profile is the project's canonical
"CyberRealisticPony Personal Base"; other requested profile names are accepted
as aliases that resolve to the same checkpoint (no new realism base is added).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .base_profile import (BASE_PROFILE_NAME, BASE_CHECKPOINT,
                               BASE_CHECKPOINT_PATH)
except Exception:                                        # pragma: no cover
    BASE_PROFILE_NAME = "CyberRealisticPony Personal Base"
    BASE_CHECKPOINT = "cyberrealisticPony_v180Coreshift.safetensors"
    BASE_CHECKPOINT_PATH = (
        "~/FORGE_REPO/stable-diffusion-webui-reForge/models/Stable-diffusion/"
        "cyberrealisticPony_v180Coreshift.safetensors")

# Conservative per-target stack weight bands (midpoint used as recommendation).
_WEIGHT_BANDS = {
    "core_identity":     (0.60, 0.85),
    "outfit_clothing":   (0.45, 0.75),
    "pose_action":       (0.35, 0.65),
    "style_aesthetic":   (0.25, 0.55),
    "garment_detail":    (0.25, 0.50),
    "catalog_reference": (0.40, 0.70),
}
# Generation order: identity first, then garment/outfit, pose, style, detail.
_ORDER_RANK = {"core_identity": 0, "catalog_reference": 1, "outfit_clothing": 2,
               "pose_action": 3, "style_aesthetic": 4, "garment_detail": 5}

# Accepted profile aliases -> the single canonical base (no new realism base).
_PROFILE_ALIASES = {
    "cyberrealistic_pony": BASE_PROFILE_NAME,
    "cyberrealisticpony personal base": BASE_PROFILE_NAME,
    "personalized cyberrealisticpony base": BASE_PROFILE_NAME,
    "personalized cyberrealisticpony v1": BASE_PROFILE_NAME,
    BASE_PROFILE_NAME.lower(): BASE_PROFILE_NAME,
}


@dataclass
class BaseModelProfile:
    name: str
    model_family: str
    checkpoint_hint: str
    resolution: int = 1024
    prompt_style: str = "pony_sdxl"
    recommended_sampler: str = "Euler a"
    recommended_clip_skip: int | None = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "model_family": self.model_family,
            "checkpoint_hint": self.checkpoint_hint,
            "resolution": self.resolution, "prompt_style": self.prompt_style,
            "recommended_sampler": self.recommended_sampler,
            "recommended_clip_skip": self.recommended_clip_skip,
            "notes": list(self.notes),
        }


@dataclass
class LoRAStackItem:
    name: str
    target: str
    recommended_weight: float
    training_recipe_path: str | None = None
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "target": self.target,
            "recommended_weight": round(float(self.recommended_weight), 3),
            "training_recipe_path": self.training_recipe_path,
            "warnings": list(self.warnings),
        }


@dataclass
class TrainingStackPlan:
    name: str
    base_model_profile: BaseModelProfile
    recipes: list = field(default_factory=list)
    stack_items: list = field(default_factory=list)
    recommended_generation_order: list = field(default_factory=list)
    forge_profile: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "base_model_profile": self.base_model_profile.to_dict(),
            "recipes": [r.to_dict() if hasattr(r, "to_dict") else r
                        for r in self.recipes],
            "stack_items": [s.to_dict() for s in self.stack_items],
            "recommended_generation_order": list(
                self.recommended_generation_order),
            "forge_profile": self.forge_profile,
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def resolve_base_profile(name: str) -> BaseModelProfile:
    """Resolve a requested profile name to the canonical base profile. Unknown
    names are accepted as aliases (note recorded) rather than adding a new
    realism base."""
    canonical = _PROFILE_ALIASES.get(str(name).strip().lower(), BASE_PROFILE_NAME)
    notes = ["Single canonical realism base for this project."]
    if str(name).strip() and str(name).strip() != canonical and \
            str(name).strip().lower() not in (BASE_PROFILE_NAME.lower(),
                                              "cyberrealistic_pony"):
        notes.append(f"Requested profile '{name}' resolved to '{canonical}'.")
    return BaseModelProfile(
        name=canonical, model_family="pony_sdxl",
        checkpoint_hint=BASE_CHECKPOINT_PATH, resolution=1024,
        prompt_style="pony_sdxl", recommended_sampler="Euler a", notes=notes)


def _load_recipe(recipe_dir: Path) -> dict:
    p = Path(recipe_dir) / "training_plan.json"
    try:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _weight_for(target: str) -> float:
    lo, hi = _WEIGHT_BANDS.get(target, (0.30, 0.60))
    return round((lo + hi) / 2.0, 3)


def build_training_stack_plan(recipe_dirs, output_dir,
                              base_model_profile: str = "cyberrealistic_pony"
                              ) -> TrainingStackPlan:
    """Aggregate recipes into a stack plan and (always) write
    ``stack_manifest.json`` + ``forge_stack_profile.json`` +
    ``colab_runbook.md`` into ``output_dir``. Never launches Forge / training.
    Never raises for ordinary input."""
    output_dir = Path(output_dir)
    profile = resolve_base_profile(base_model_profile)
    recipes, items, warnings = [], [], []
    for rd in recipe_dirs or []:
        rec = _load_recipe(Path(rd))
        if not rec:
            warnings.append(f"no_training_plan_in: {rd}")
            continue
        recipes.append(rec)
        target = rec.get("target", "unknown")
        item = LoRAStackItem(
            name=rec.get("output_name") or f"redresser_{target}",
            target=target, recommended_weight=_weight_for(target),
            training_recipe_path=str(Path(rd) / "training_plan.json"),
            warnings=list(rec.get("warnings", [])))
        items.append(item)

    items.sort(key=lambda it: (_ORDER_RANK.get(it.target, 99), it.name))
    order = [it.name for it in items]
    forge_profile = {
        "schema_version": "forge_stack_profile_v1",
        "base_model_profile": profile.to_dict(),
        "checkpoint_hint": profile.checkpoint_hint,
        "recommended_sampler": profile.recommended_sampler,
        "resolution": profile.resolution,
        "lora_stack": [{"name": it.name, "target": it.target,
                        "weight": it.recommended_weight} for it in items],
        "generation_order": order,
        "notes": ["Weights are conservative midpoints; tune per render.",
                  "Forge/reForge: load LoRAs in generation_order; identity "
                  "highest weight, detail lowest."],
    }
    plan = TrainingStackPlan(
        name=output_dir.name or "training_stack",
        base_model_profile=profile, recipes=recipes, stack_items=items,
        recommended_generation_order=order, forge_profile=forge_profile,
        warnings=warnings)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stack_manifest.json").write_text(
            json.dumps({
                "schema_version": "training_stack_v1",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "name": plan.name,
                "base_model_profile": profile.to_dict(),
                "stack_items": [it.to_dict() for it in items],
                "recommended_generation_order": order,
                "recipes": recipes, "warnings": warnings,
            }, indent=2), encoding="utf-8")
        (output_dir / "forge_stack_profile.json").write_text(
            json.dumps(forge_profile, indent=2), encoding="utf-8")
        (output_dir / "colab_runbook.md").write_text(
            _render_stack_runbook(plan), encoding="utf-8")
    except Exception as exc:                             # never crash
        plan.warnings.append(f"write_failed: {exc}")
    return plan


def _render_stack_runbook(plan: TrainingStackPlan) -> str:
    L = [f"# Training Stack Runbook: {plan.name}", "",
         f"Base model: **{plan.base_model_profile.name}** "
         f"(`{plan.base_model_profile.checkpoint_hint}`)", "",
         "## LoRA Stack (recommended weights)",
         "| Order | LoRA | Target | Weight |", "|---:|---|---|---:|"]
    for i, it in enumerate(plan.stack_items, start=1):
        L.append(f"| {i} | {it.name} | {it.target} | {it.recommended_weight} |")
    L += ["", "## Generation Order",
          " -> ".join(plan.recommended_generation_order) or "(none)", "",
          "## Forge / reForge",
          "Load `forge_stack_profile.json`; apply LoRAs in the order above "
          "(identity highest weight, detail lowest). No Forge launch is "
          "performed by this tool.", "",
          "## Train each concept",
          "Package each dataset (`package kohya --mode lora --apply`), then "
          "follow each recipe's `colab_runbook.md` in the SDXL LoRA Colab.",
          ""]
    return "\n".join(L)


__all__ = [
    "BaseModelProfile", "LoRAStackItem", "TrainingStackPlan",
    "resolve_base_profile", "build_training_stack_plan",
]
