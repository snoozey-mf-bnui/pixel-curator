"""Figure study batch planner - controlled variation, fully deterministic.

Plans a batch of identity-preserving artistic figure study generations
from one recipe: every item varies along named, professional variation
axes (pose energy, lighting treatment, camera framing, garment styling,
expression) while the identity anchor and negative control chains stay
constant. Variation selection is mixed-radix over the axis pools, so a
batch sweeps combinations in a stable, reproducible order; seeds advance
by a prime stride from the base seed.

No GPU, no network: this module composes prompts and writes a batch
manifest - payload building and rendering hand off to the existing
artistic workflow builders.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .prompt_composer import ArtisticPromptRecipe, compose_artistic_prompt, get_recipe

# ---------------------------------------------------------------------------
# Controlled variation axes (professional studio vocabulary)
# ---------------------------------------------------------------------------
VARIATION_AXES: dict[str, list[str]] = {
    "pose_energy": [
        "balanced standing pose", "seated contrapposto pose",
        "reclining editorial pose", "dynamic turning pose",
        "profile study pose",
    ],
    "lighting_treatment": [
        "chiaroscuro studio lighting", "soft north-light treatment",
        "backlit rim lighting", "dramatic single-source lighting",
        "even high-key lighting",
    ],
    "camera_framing": [
        "full body framing", "three-quarter framing", "waist-up framing",
        "close detail framing",
    ],
    "garment_styling": [
        "minimal styling", "draped fabric styling", "editorial layering",
        "structured garment styling",
    ],
    "expression": [
        "serene expression", "confident expression",
        "contemplative expression",
    ],
}

DEFAULT_VARIATION_AXES = ["pose_energy", "lighting_treatment",
                          "camera_framing"]

SEED_STRIDE = 1009          # prime: non-overlapping deterministic seeds
MAX_BATCH = 64


@dataclass
class FigureStudyItem:
    index: int
    seed: int
    variation: dict[str, str]
    prompt: str
    negative_prompt: str
    lora_stack: list[dict]
    identity_score_estimate: float
    warnings: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


def list_variation_axes() -> list[dict]:
    return [{"name": name, "options": list(opts),
             "default": name in DEFAULT_VARIATION_AXES}
            for name, opts in VARIATION_AXES.items()]


def _variation_for(index: int, axes: list[str]) -> dict[str, str]:
    """Mixed-radix decomposition: item N maps to a unique axis combination
    until the product of pool sizes is exhausted, then wraps."""
    out: dict[str, str] = {}
    n = index
    for axis in axes:
        pool = VARIATION_AXES[axis]
        out[axis] = pool[n % len(pool)]
        n //= len(pool)
    return out


def plan_figure_study_batch(
        recipe: ArtisticPromptRecipe | str,
        count: int = 4,
        base_seed: int = 42,
        variation_axes: list[str] | None = None,
        trigger_phrase: str = "",
        identity_lora_name: str = "",
        identity_strength: float | None = None,
        concept_strength: float | None = None,
        strength: float = 1.0,
        negative_chains: list[str] | None = None,
        extra_negative_terms: list[str] | None = None,
        negative_intensity: str = "standard") -> dict:
    """Plan a controlled-variation figure study batch.

    Returns a JSON-serializable plan: per-item composed prompt, negative
    chain, ordered LoRA stack, seed, variation descriptors and identity
    preservation estimate, plus batch-level summary statistics.
    """
    if isinstance(recipe, str):
        recipe = get_recipe(recipe)
    count = max(1, min(int(count), MAX_BATCH))
    axes = list(variation_axes or DEFAULT_VARIATION_AXES)
    for axis in axes:
        if axis not in VARIATION_AXES:
            raise KeyError(
                f"unknown variation axis '{axis}'. "
                f"Options: {', '.join(sorted(VARIATION_AXES))}")

    items: list[FigureStudyItem] = []
    batch_warnings: list[str] = []
    for i in range(count):
        variation = _variation_for(i, axes)
        varied = replace(
            recipe,
            positive_style=list(recipe.positive_style)
            + list(variation.values()))
        composed = compose_artistic_prompt(
            varied, trigger_phrase=trigger_phrase,
            identity_strength=identity_strength,
            concept_strength=concept_strength,
            strength=strength, identity_lora_name=identity_lora_name,
            negative_chains=negative_chains,
            extra_negative_terms=extra_negative_terms,
            negative_intensity=negative_intensity)
        items.append(FigureStudyItem(
            index=i, seed=base_seed + i * SEED_STRIDE,
            variation=variation, prompt=composed.prompt,
            negative_prompt=composed.negative_prompt,
            lora_stack=composed.lora_stack,
            identity_score_estimate=composed.identity_score_estimate,
            warnings=composed.warnings,
            reason_codes=composed.reason_codes))

    estimates = [it.identity_score_estimate for it in items]
    min_est = min(estimates)
    if min_est < 0.7:
        batch_warnings.append(
            "Identity preservation estimate drops below 0.70 for some "
            "items - consider raising identity strength or lowering "
            "concept strength before rendering the full batch.")

    combos = 1
    for axis in axes:
        combos *= len(VARIATION_AXES[axis])
    if count > combos:
        batch_warnings.append(
            f"Batch size {count} exceeds the {combos} unique combinations "
            f"of the selected axes - variation descriptors repeat (seeds "
            f"still differ).")

    return {
        "batch_name": f"{recipe.name}_seed{base_seed}_x{count}",
        "recipe": recipe.name,
        "count": count,
        "base_seed": base_seed,
        "seed_stride": SEED_STRIDE,
        "variation_axes": axes,
        "negative_intensity": negative_intensity,
        "items": [it.to_json() for it in items],
        "summary": {
            "identity_estimate_min": round(min_est, 3),
            "identity_estimate_mean": round(sum(estimates) / len(estimates),
                                            3),
            "unique_variation_combinations": min(count, combos),
        },
        "warnings": batch_warnings,
    }


def write_batch_manifest(plan: dict, output_base: Path) -> Path:
    """Persist the plan under <output_base>/figure_studies/<batch>/ as a
    reviewable JSON manifest + human-readable summary. Non-destructive:
    an existing batch directory gets a numeric suffix."""
    root = Path(output_base) / "figure_studies"
    target = root / plan["batch_name"]
    n = 2
    while target.exists():
        target = root / f"{plan['batch_name']}_{n}"
        n += 1
    target.mkdir(parents=True)

    (target / "batch_manifest.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8")

    lines = [f"# Figure Study Batch - {plan['batch_name']}", "",
             f"Recipe: {plan['recipe']}  |  items: {plan['count']}  |  "
             f"base seed: {plan['base_seed']} (stride {plan['seed_stride']})",
             f"Variation axes: {', '.join(plan['variation_axes'])}",
             f"Identity estimate: min "
             f"{plan['summary']['identity_estimate_min']:.2f} / mean "
             f"{plan['summary']['identity_estimate_mean']:.2f}", ""]
    for w in plan["warnings"]:
        lines.append(f"> WARNING: {w}")
    lines.append("")
    lines.append("| # | seed | " + " | ".join(plan["variation_axes"])
                 + " | identity est |")
    lines.append("|---" * (len(plan["variation_axes"]) + 3) + "|")
    for it in plan["items"]:
        cells = [str(it["index"]), str(it["seed"])]
        cells += [it["variation"][a] for a in plan["variation_axes"]]
        cells.append(f"{it['identity_score_estimate']:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    (target / "batch_summary.md").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Phase 5: multi-pass production batches
# ---------------------------------------------------------------------------

PRODUCTION_PASSES: dict[str, dict] = {
    "composition": {
        "purpose": "Initial composition and identity-preserving figure "
                   "study layout.",
        "kind": "generation",
    },
    "pose_variation": {
        "purpose": "Pose and composition variation around the strongest "
                   "compositions.",
        "kind": "generation",
        "lead_axis": "pose_energy",
    },
    "garment_refine": {
        "purpose": "Garment rendering refinement pass - boundary "
                   "stability and fabric detail.",
        "kind": "refinement", "denoise": 0.40, "region": "torso",
        "negative_chains": ["garment_boundary_control"],
    },
    "skin_refine": {
        "purpose": "Skin texture highlight refinement pass - specular "
                   "highlight control at low denoise.",
        "kind": "refinement", "denoise": 0.35, "region": "torso",
        "negative_chains": ["skin_texture_control", "specular_control"],
    },
    "identity": {
        "purpose": "Identity consistency pass - strict identity "
                   "protection at minimal denoise.",
        "kind": "refinement", "denoise": 0.30, "region": "full",
        "negative_chains": ["identity_protection",
                            "anatomical_coherence"],
    },
    "review": {
        "purpose": "Review pass - mark selected variants, refinement "
                   "needs and rejections in the review manifest.",
        "kind": "review",
    },
}

# Friendly aliases for --passes
PASS_ALIASES = {
    "refine": ["garment_refine", "skin_refine"],
    "pose": ["pose_variation"],
    "skin": ["skin_refine"],
    "garment": ["garment_refine"],
}

# variation strength (0..1) -> how many variation axes engage
_VARIATION_AXIS_LADDER = [
    (0.25, ["pose_energy", "lighting_treatment"]),
    (0.50, ["pose_energy", "lighting_treatment", "camera_framing"]),
    (0.75, ["pose_energy", "lighting_treatment", "camera_framing",
            "garment_styling"]),
    (1.01, ["pose_energy", "lighting_treatment", "camera_framing",
            "garment_styling", "expression"]),
]


def resolve_passes(spec: list[str]) -> list[str]:
    """Expand aliases, validate, de-duplicate (order preserved)."""
    out: list[str] = []
    for name in spec:
        expanded = PASS_ALIASES.get(name, [name])
        for p in expanded:
            if p not in PRODUCTION_PASSES:
                raise KeyError(
                    f"unknown production pass '{name}'. Options: "
                    f"{', '.join(sorted(PRODUCTION_PASSES))} "
                    f"(aliases: {', '.join(sorted(PASS_ALIASES))})")
            if p not in out:
                out.append(p)
    return out or ["composition"]


def axes_for_variation_strength(strength: float) -> list[str]:
    strength = max(0.0, min(1.0, float(strength)))
    for ceiling, axes in _VARIATION_AXIS_LADDER:
        if strength < ceiling:
            return list(axes)
    return list(_VARIATION_AXIS_LADDER[-1][1])


def plan_production_batch(
        recipe: ArtisticPromptRecipe | str,
        count: int = 4,
        base_seed: int = 42,
        passes: list[str] | None = None,
        variation_strength: float | None = None,
        variation_axes: list[str] | None = None,
        guidance: dict | None = None,
        refine_strength: float = 0.0,
        max_variants: int = 0,
        trigger_phrase: str = "",
        identity_lora_name: str = "",
        identity_strength: float | None = None,
        concept_strength: float | None = None,
        strength: float = 1.0,
        negative_chains: list[str] | None = None,
        negative_intensity: str = "standard") -> dict:
    """Plan a multi-pass figure study production batch.

    Generation passes plan controlled-variation items; refinement passes
    reference the previous generation pass at conservative denoise with
    targeted negative chains; the review pass initializes the review
    workflow. Fully deterministic and dry-run safe: nothing is rendered.
    """
    from datetime import datetime, timezone

    if isinstance(recipe, str):
        recipe = get_recipe(recipe)
    pass_names = resolve_passes(list(passes or ["composition"]))
    if variation_strength is not None and not variation_axes:
        variation_axes = axes_for_variation_strength(variation_strength)
    if max_variants:
        count = max(1, min(count, int(max_variants)))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_name = f"{stamp}_{recipe.name}"

    planned: list[dict] = []
    last_generation: dict | None = None
    item_serial = 0
    batch_warnings: list[str] = []

    for idx, name in enumerate(pass_names):
        spec = PRODUCTION_PASSES[name]
        entry: dict = {
            "pass_id": f"pass_{idx + 1:02d}_{name}",
            "pass_type": name,
            "purpose": spec["purpose"],
            "items": [],
            "guidance": guidance or {},
            "prompt_axis": {"recipe": recipe.name,
                            "negative_intensity": negative_intensity},
            "variation_axis": {},
            "warnings": [],
        }
        if spec["kind"] == "generation":
            axes = list(variation_axes or DEFAULT_VARIATION_AXES)
            lead = spec.get("lead_axis")
            if lead:
                axes = [lead] + [a for a in axes if a != lead]
            sub = plan_figure_study_batch(
                recipe, count=count,
                base_seed=base_seed + idx * SEED_STRIDE * MAX_BATCH,
                variation_axes=axes, trigger_phrase=trigger_phrase,
                identity_lora_name=identity_lora_name,
                identity_strength=identity_strength,
                concept_strength=concept_strength, strength=strength,
                negative_chains=negative_chains,
                negative_intensity=negative_intensity)
            for it in sub["items"]:
                it["item_id"] = f"item_{item_serial:04d}"
                item_serial += 1
            entry["items"] = sub["items"]
            entry["variation_axis"] = {
                "axes": axes, "strength": variation_strength}
            entry["warnings"] = sub["warnings"]
            batch_warnings += sub["warnings"]
            last_generation = entry
        elif spec["kind"] == "refinement":
            if last_generation is None:
                entry["warnings"].append(
                    "Refinement pass planned without a preceding "
                    "generation pass - it will refine externally "
                    "provided samples.")
            sources = (last_generation or {}).get("items", [])
            for src in sources:
                entry["items"].append({
                    "item_id": f"item_{item_serial:04d}",
                    "refines": src["item_id"],
                    "seed": src["seed"],
                    "denoise": refine_strength or spec["denoise"],
                    "region": spec.get("region", "full"),
                    "negative_chains": spec.get("negative_chains", []),
                    "prompt": src["prompt"],
                    "negative_prompt": src["negative_prompt"],
                    "lora_stack": src["lora_stack"],
                    "identity_score_estimate":
                        src["identity_score_estimate"],
                })
                item_serial += 1
            entry["variation_axis"] = {"axes": [], "strength": 0.0}
        else:                                   # review
            entry["review_template"] = {
                "statuses": ["selected", "needs_refinement", "rejected",
                             "strong_candidate", "final_candidate"],
                "issue_tags": ["identity_consistency_issue",
                               "pose_issue", "garment_boundary_issue",
                               "skin_texture_highlight_issue",
                               "lighting_issue",
                               "anatomical_coherence_issue"],
            }
        planned.append(entry)

    gen_items = [it for p in planned
                 if PRODUCTION_PASSES[p["pass_type"]]["kind"] == "generation"
                 for it in p["items"]]
    estimates = [it["identity_score_estimate"] for it in gen_items] or [1.0]

    return {
        "batch_name": batch_name,
        "recipe": recipe.name,
        "count": count,
        "base_seed": base_seed,
        "seed_stride": SEED_STRIDE,
        "variation_strength": variation_strength,
        "passes": planned,
        "pass_names": pass_names,
        "guidance": guidance or {},
        "negative_intensity": negative_intensity,
        "summary": {
            "total_items": item_serial,
            "generation_items": len(gen_items),
            "identity_estimate_min": round(min(estimates), 3),
            "identity_estimate_mean":
                round(sum(estimates) / len(estimates), 3),
        },
        "warnings": batch_warnings,
        "review_status": "unreviewed",
        "selected_variant": None,
    }


def write_production_batch(plan: dict, output_base: Path) -> Path:
    """Persist a production batch under figure_studies/<batch>/ with the
    full manifest set + working directories. Non-destructive."""
    root = Path(output_base) / "figure_studies"
    target = root / plan["batch_name"]
    n = 2
    while target.exists():
        target = root / f"{plan['batch_name']}_{n}"
        n += 1
    target.mkdir(parents=True)
    for sub in ("samples", "refined", "selected"):
        (target / sub).mkdir()

    (target / "batch_manifest.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8")
    prompts = [{"item_id": it["item_id"], "seed": it["seed"],
                "prompt": it["prompt"],
                "negative_prompt": it["negative_prompt"]}
               for p in plan["passes"] for it in p["items"]]
    (target / "prompts.json").write_text(
        json.dumps(prompts, indent=2), encoding="utf-8")
    (target / "guidance_payloads.json").write_text(
        json.dumps({"batch": plan["batch_name"],
                    "guidance": plan["guidance"],
                    "passes": [{"pass_id": p["pass_id"],
                                "guidance": p["guidance"]}
                               for p in plan["passes"]]}, indent=2),
        encoding="utf-8")
    review = {
        "batch": plan["batch_name"],
        "review_status": "unreviewed",
        "selected_variant": None,
        "items": [{"item_id": it["item_id"],
                   "review_status": "unreviewed",
                   "review_tags": [], "review_notes": "",
                   "next_action": ""}
                  for p in plan["passes"] for it in p["items"]],
    }
    (target / "review_manifest.json").write_text(
        json.dumps(review, indent=2), encoding="utf-8")
    return target
