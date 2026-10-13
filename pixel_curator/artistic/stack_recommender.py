"""Phase 3: Smart Stack Recommendation Engine.

`recommend_best_stack` turns dataset signals (the same collector the
preset recommender uses), Phase 1 quality/duplicate metrics, Phase 2
cluster diversity and the user's actual LoRA library into a
conservative, identity-first concept stack recommendation with
confidence, warnings, professional reason codes and simpler
alternatives.

Identity preservation is the governing rule: the identity anchor is
always strongest, concept richness scales with dataset strength, and
every stack is validated through the Identity Integration Layer's
progressive-layering assessment before it is returned.
"""

from __future__ import annotations

import logging

from ..config import Project
from ..identity_integration import assess_concept_layering
from .prompt_composer import ROLE_WEIGHT_RANGES

logger = logging.getLogger(__name__)

# Concept candidates: (library-name hint, role, dataset signal key,
# minimum signal value, professional reason)
_CONCEPT_CANDIDATES = [
    ("lingerie_form_study", "garment_rendering", "lingerie_fashion_rate",
     0.15, "Supports garment structure and form study."),
    ("fashion_editorial", "garment_rendering", "fashion_study_rate",
     0.25, "Editorial fashion study coverage supports garment styling."),
    ("figure_study", "form_proportion", "figure_study_rate",
     0.30, "Dataset form/proportion coverage supports figure study work."),
    ("sheer_clothing", "sheer_garment_rendering", "lingerie_fashion_rate",
     0.30, "Sheer garment rendering layered over strong garment coverage."),
    ("wet_glossy_skin", "skin_texture_highlight", "quality_score",
     0.75, "Adds controlled skin texture highlight treatment without "
           "overpowering identity."),
    ("boudoir_pose", "pose", "figure_study_rate",
     0.25, "Adds editorial pose direction while preserving character "
           "consistency."),
    ("studio_lighting", "lighting", "quality_score",
     0.55, "Lighting treatment modifier - low identity risk."),
]


def _mid(role: str) -> float:
    lo, hi = ROLE_WEIGHT_RANGES.get(role, (0.20, 0.45))
    return round((lo + hi) / 2, 2)


def _library(prj: Project | None) -> tuple[str | None, dict[str, str]]:
    """(identity anchor lora_id or None, lowercase-name -> lora_id)."""
    try:
        from ..lora_explorer import scan_loras
        from ..starter_stacks import _pick_identity
        cards = scan_loras(prj)
        ident = _pick_identity(cards)
        return (ident.lora_id if ident else None,
                {c.lora_id.lower(): c.lora_id for c in cards})
    except Exception:
        return None, {}


def _duplicate_density(conn, dataset_version: int) -> float:
    try:
        row = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN f.dup_role='candidate' "
            "THEN 1 ELSE 0 END) FROM frames f JOIN dataset_frames df "
            "ON df.frame_id=f.frame_id AND df.version=?",
            (dataset_version,)).fetchone()
        total, cands = row[0] or 0, row[1] or 0
        return round(cands / total, 3) if total else 0.0
    except Exception:
        return 0.0


def recommend_best_stack(conn, prj: Project | None,
                         dataset_version: int,
                         identity_strength: float | None = None,
                         concept_strength: float | None = None) -> dict:
    """Recommend an identity-preserving concept stack for a dataset.

    Conservative by design: richer stacks only when dataset quality and
    identity signals are strong; concept weights are down-weighted when
    identity confidence is weak; the final stack always passes through
    the progressive-layering assessment (identity never auto-reduced).
    """
    from ..train.preset_recommender import collect_signals
    s = collect_signals(conn, dataset_version)     # raises if version absent

    warnings: list[str] = []
    codes: list[str] = []

    ident = float(s.get("identity_strength") or 0.0)
    quality = float(s.get("quality_score") or 0.0)
    frames = int(s.get("selected_frames") or 0)
    clusters = int(s.get("cluster_count") or 0)
    dup_density = _duplicate_density(conn, dataset_version)

    # ----- richness tier -----
    strong = ident >= 0.55 and quality >= 0.75 and frames >= 300 \
        and dup_density <= 0.15
    weak = ident < 0.40 or frames < 100 or quality < 0.55
    max_concepts = 3 if strong else (1 if weak else 2)
    if strong:
        codes.append("dataset_signals_strong")
    if weak:
        codes.append("dataset_signals_weak")
        warnings.append(
            "Dataset signals are limited (identity confidence "
            f"{ident:.2f}, {frames} frames, quality {quality:.2f}) - "
            "recommending a simpler stack until the dataset matures.")
    if dup_density > 0.15:
        codes.append("duplicate_density_elevated")
        warnings.append(
            f"Duplicate candidate density {dup_density:.0%} - clean up "
            "duplicate groups before stacking aggressively.")
    if clusters and clusters < 8:
        codes.append("cluster_diversity_low")

    # ----- identity anchor -----
    anchor_name, library = _library(prj)
    id_w = identity_strength if identity_strength is not None else 0.85
    id_lo, id_hi = ROLE_WEIGHT_RANGES["identity_anchor"]
    id_w = round(max(id_lo, min(id_hi, id_w)), 2)
    stack = [{
        "name": anchor_name or "identity_anchor",
        "role": "identity_anchor",
        "weight": id_w,
        "reason": "Primary identity anchor for character consistency.",
    }]
    if anchor_name:
        codes.append("identity_anchor_preserved")
    else:
        codes.append("identity_anchor_unresolved")
        warnings.append(
            "No identity anchor LoRA found in the library - placeholder "
            "name used; train or register an identity LoRA first.")

    # ----- concept selection by dataset signals -----
    scale = float(concept_strength) if concept_strength else 1.0
    if ident and ident < 0.55:
        scale *= 0.85
        codes.append("concept_weights_down_weighted_for_identity")
    picked_roles: set[str] = set()
    signal_map = {"lingerie_fashion_rate": "dataset_supports_garment_study",
                  "fashion_study_rate": "dataset_supports_fashion_study",
                  "figure_study_rate": "dataset_supports_form_study",
                  "quality_score": "dataset_quality_supports_refinement"}
    for hint, role, key, floor, reason in _CONCEPT_CANDIDATES:
        if len(stack) - 1 >= max_concepts:
            break
        if role in picked_roles:
            continue
        val = float(s.get(key) or 0.0)
        if val < floor:
            continue
        real = next((rid for low, rid in library.items()
                     if hint in low), None)
        lo, hi = ROLE_WEIGHT_RANGES.get(role, (0.20, 0.45))
        w = round(max(lo * 0.5, min(hi, _mid(role) * scale)), 2)
        stack.append({"name": real or hint, "role": role, "weight": w,
                      "reason": reason
                      + ("" if real else " (library hint - LoRA not yet "
                                         "in the library)")})
        picked_roles.add(role)
        code = signal_map.get(key)
        if code and code not in codes:
            codes.append(code)
        if role == "pose":
            codes.append("editorial_pose_supported")

    # ----- layering assessment (identity never reduced) -----
    assessed = assess_concept_layering(
        [{"name": x["name"], "role": x["role"], "weight": x["weight"]}
         for x in stack], prj)
    adj = {l["name"]: l["weight"] for l in assessed["adjusted_layers"]}
    for x in stack:
        new_w = adj.get(x["name"], x["weight"])
        if abs(new_w - x["weight"]) > 1e-9:
            x["reason"] += (f" Weight adjusted to {new_w:.2f} by the "
                            "layering assessment.")
            x["weight"] = new_w
    for c in assessed["reason_codes"]:
        if c not in codes:
            codes.append(c)
    warnings += assessed["warnings"]

    concept_total = round(sum(x["weight"] for x in stack
                              if x["role"] != "identity_anchor"), 2)

    # ----- confidence -----
    confidence = 0.45
    confidence += 0.15 if ident >= 0.55 else 0.0
    confidence += 0.10 if quality >= 0.75 else 0.0
    confidence += 0.10 if frames >= 300 else 0.0
    confidence += 0.10 if anchor_name else 0.0
    confidence -= 0.05 * len([w for w in warnings])
    confidence = round(max(0.10, min(0.95, confidence)), 2)

    # ----- alternatives (always simpler) -----
    alternatives = []
    if len(stack) > 1:
        alternatives.append({
            "label": "Identity anchor only (baseline consistency check)",
            "stack": [stack[0]],
        })
    if len(stack) > 2:
        alternatives.append({
            "label": "Identity + single strongest concept",
            "stack": [stack[0], stack[1]],
        })

    return {
        "dataset_version": dataset_version,
        "recommended_stack": stack,
        "confidence": confidence,
        "identity_preservation_estimate":
            assessed["identity_score_estimate"],
        "concept_strength_total": concept_total,
        "signals": {"identity_strength": ident, "quality_score": quality,
                    "selected_frames": frames, "cluster_count": clusters,
                    "duplicate_density": dup_density},
        "warnings": warnings,
        "reason_codes": codes,
        "alternatives": alternatives,
    }


def render_recommendation(rec: dict) -> str:
    lines = [
        f"SMART STACK RECOMMENDATION | dataset v{rec['dataset_version']:03d}",
        f"  confidence {rec['confidence']:.2f} | identity preservation "
        f"estimate {rec['identity_preservation_estimate']:.2f} | "
        f"concept strength total {rec['concept_strength_total']:.2f}",
        "  stack (identity first):",
    ]
    for x in rec["recommended_stack"]:
        lines.append(f"    {x['name']:<30} {x['role']:<24} "
                     f"{x['weight']:.2f}  {x['reason']}")
    for a in rec["alternatives"]:
        names = " + ".join(f"{i['name']}@{i['weight']:.2f}"
                           for i in a["stack"])
        lines.append(f"  alternative: {a['label']}: {names}")
    for w in rec["warnings"]:
        lines.append(f"  WARNING: {w}")
    lines.append(f"  reason codes: {', '.join(rec['reason_codes'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 4: best epoch + stack in one recommendation
# ---------------------------------------------------------------------------

def recommend_best_epoch_and_stack(conn, prj: Project | None,
                                   run_name: str = "") -> dict:
    """Combine the best-epoch sweep verdict with the smart stack
    recommendation: which checkpoint to anchor identity on, and which
    concept layers to compose around it.

    Best-epoch evidence, in order of preference: the recorded sweep
    scoreboard (`sweep:<run>`), then the Phase 3 evaluation report, then
    the newest checkpoint on disk (clearly labeled as unscored).
    """
    from .. import manifest as mf

    row = conn.execute(
        "SELECT name, dataset_version, status FROM runs "
        + ("WHERE name=? " if run_name else "")
        + "ORDER BY run_id DESC LIMIT 1",
        ((run_name,) if run_name else ())).fetchone()
    if not row:
        raise KeyError("no training run found"
                       + (f" named '{run_name}'" if run_name else ""))
    run_name, dataset_version, status = row[0], int(row[1] or 0), row[2]

    warnings: list[str] = []
    codes: list[str] = []
    if status == "running":
        warnings.append(
            "This run is still training - the recommendation reflects the "
            "checkpoints written so far and will improve once the "
            "best-epoch sweep completes.")
        codes.append("run_still_training")

    # ----- best epoch evidence -----
    best: dict = {"epoch": None, "label": None, "likeness": None,
                  "flexibility": None, "source": "none"}
    sweep = mf.meta_get(conn, f"sweep:{run_name}") or {}
    if sweep.get("best"):
        b = sweep["best"]
        best = {"epoch": b.get("epoch"), "label": b.get("label"),
                "likeness": b.get("likeness"),
                "flexibility": b.get("flexibility"), "source": "sweep"}
        codes.append("best_epoch_from_sweep")
        if b.get("overfit_warning"):
            warnings.append(
                "Every scored epoch exceeded the overfit gap - prefer an "
                "earlier checkpoint and verify flexibility prompts.")
            codes.append("overfit_gap_exceeded")
    else:
        report = mf.meta_get(conn, f"evaluation:{run_name}") or {}
        cand = ((report.get("identity_consistency") or {})
                .get("best_checkpoint"))
        if cand:
            best = {"epoch": cand.get("epoch"), "label": cand.get("label"),
                    "likeness": cand.get("score"), "flexibility": None,
                    "source": "evaluation"}
            codes.append("best_epoch_from_evaluation")
    if best["source"] == "none":
        try:
            from ..monitor import checkpoints_for
            cps = checkpoints_for(prj, run_name) if prj else []
        except Exception:
            cps = []
        if cps:
            best = {"epoch": None, "label": cps[-1].stem,
                    "likeness": None, "flexibility": None,
                    "source": "latest_checkpoint_unscored"}
            codes.append("best_epoch_unscored_latest_checkpoint")
            warnings.append(
                "No sweep or evaluation scores recorded yet - using the "
                "newest checkpoint. Run the best-epoch sweep for a "
                "scored recommendation.")
        else:
            codes.append("no_checkpoints_found")
            warnings.append(
                "No checkpoints found for this run yet - the stack "
                "recommendation stands, but there is no epoch to anchor "
                "it on.")

    # ----- stack recommendation anchored on the chosen checkpoint -----
    try:
        stack_rec = recommend_best_stack(conn, prj, dataset_version)
    except Exception as exc:
        stack_rec = {"recommended_stack": [], "confidence": 0.1,
                     "identity_preservation_estimate": None,
                     "concept_strength_total": 0.0, "signals": {},
                     "warnings": [f"Stack signals unavailable: {exc}"],
                     "reason_codes": ["dataset_signals_unavailable"],
                     "alternatives": [], "dataset_version": dataset_version}
    if best["label"]:
        if not stack_rec["recommended_stack"]:
            # signals unavailable: identity-only baseline stack
            stack_rec["recommended_stack"] = [{
                "name": best["label"], "role": "identity_anchor",
                "weight": 0.85,
                "reason": "Identity-only baseline (dataset signals "
                          "unavailable).",
            }]
            codes.append("identity_only_baseline_stack")
        anchor = stack_rec["recommended_stack"][0]
        if anchor["role"] == "identity_anchor":
            anchor["name"] = best["label"]
            anchor["reason"] = (
                f"Identity anchored on the recommended checkpoint "
                f"({best['source']}).")
            codes.append("identity_anchored_on_best_epoch")

    confidence = stack_rec["confidence"]
    if best["source"] == "sweep":
        confidence = min(0.95, round(confidence + 0.10, 2))
    elif best["source"] in ("latest_checkpoint_unscored", "none"):
        confidence = max(0.10, round(confidence - 0.10, 2))

    return {
        "run": run_name,
        "run_status": status,
        "dataset_version": dataset_version,
        "best_epoch": best,
        "recommended_stack": stack_rec["recommended_stack"],
        "confidence": confidence,
        "identity_preservation_estimate":
            stack_rec["identity_preservation_estimate"],
        "concept_strength_total": stack_rec["concept_strength_total"],
        "signals": stack_rec["signals"],
        "warnings": warnings + stack_rec["warnings"],
        "reason_codes": codes + stack_rec["reason_codes"],
        "alternatives": stack_rec["alternatives"],
    }


def render_epoch_stack(rec: dict) -> str:
    b = rec["best_epoch"]
    epoch = (f"epoch {b['epoch']}" if b.get("epoch") is not None
             else "(epoch unscored)")
    lines = [
        f"BEST EPOCH + STACK | run '{rec['run']}' "
        f"(dataset v{rec['dataset_version']:03d}, {rec['run_status']})",
        f"  best checkpoint: {b.get('label') or '(none)'} {epoch} "
        f"[{b['source']}]"
        + (f" likeness {b['likeness']}" if b.get("likeness") is not None
           else ""),
        f"  confidence {rec['confidence']:.2f} | concept strength total "
        f"{rec['concept_strength_total']:.2f}",
        "  stack (identity first):",
    ]
    for x in rec["recommended_stack"]:
        lines.append(f"    {x['name']:<34} {x['role']:<24} "
                     f"{x['weight']:.2f}")
    for w in rec["warnings"]:
        lines.append(f"  WARNING: {w}")
    lines.append(f"  reason codes: {', '.join(rec['reason_codes'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 4: recipe-driven figure-study layering recommendation
# ---------------------------------------------------------------------------

def recommend_figure_study_stack(
        recipe_name: str, prj: Project | None = None,
        conn=None, dataset_version: int | None = None,
        identity_strength: float | None = None,
        concept_strength: float | None = None,
        negative_intensity: str = "standard") -> dict:
    """Layering recommendation for one figure-study recipe: ordered
    stack with per-layer reasoning, prompt/negative patterns, overstack
    protection verdict and simpler alternatives. Dataset signals (when a
    manifest is supplied) tighten or relax concept strength."""
    from .prompt_composer import compose_artistic_prompt, get_recipe

    recipe = get_recipe(recipe_name)
    warnings: list[str] = []
    codes: list[str] = []

    c_strength = concept_strength
    if conn is not None and dataset_version:
        try:
            from ..train.preset_recommender import collect_signals
            s = collect_signals(conn, dataset_version)
            ident = float(s.get("identity_strength") or 0.0)
            if c_strength is None and ident and ident < 0.55:
                c_strength = round(recipe.concept_strength * 0.85, 2)
                warnings.append(
                    f"Dataset identity confidence {ident:.2f} - concept "
                    f"strength relaxed to {c_strength:.2f} to protect "
                    "identity.")
                codes.append("concept_weights_down_weighted_for_identity")
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass

    id_name = ""
    if prj is not None:
        anchor, _ = _library(prj)
        if anchor:
            id_name = anchor
            codes.append("identity_anchor_resolved_from_library")
    composed = compose_artistic_prompt(
        recipe, trigger_phrase="", identity_strength=identity_strength,
        concept_strength=c_strength, identity_lora_name=id_name,
        negative_intensity=negative_intensity)

    stack = [{"name": s["name"], "role": s["role"], "weight": s["weight"],
              "reason": s["notes"] or f"{s['role']} layer from the "
                                      f"'{recipe.name}' recipe."}
             for s in composed.lora_stack]
    overstack_protected = "concept_stack_normalized" in composed.reason_codes
    if overstack_protected:
        codes.append("overstack_protection_applied")

    alternatives = []
    if len(stack) > 1:
        alternatives.append({
            "label": "Identity anchor only (baseline consistency check)",
            "stack": [stack[0]]})
    if len(stack) > 2:
        alternatives.append({
            "label": "Identity + primary concept layer",
            "stack": [stack[0], stack[1]]})

    return {
        "recipe": recipe.name,
        "description": recipe.description,
        "recommended_stack": stack,
        "prompt_pattern": composed.prompt,
        "negative_pattern": composed.negative_prompt,
        "negative_chains": list(recipe.negative_chains),
        "negative_intensity": negative_intensity,
        "identity_preservation_estimate":
            composed.identity_score_estimate,
        "overstack_protected": overstack_protected,
        "warnings": warnings + composed.warnings,
        "reason_codes": codes + composed.reason_codes,
        "alternatives": alternatives,
    }


def render_figure_study_recommendation(rec: dict) -> str:
    lines = [
        f"FIGURE STUDY LAYERING | recipe '{rec['recipe']}'",
        f"  {rec['description']}",
        f"  identity preservation estimate "
        f"{rec['identity_preservation_estimate']:.2f} | overstack "
        f"protection {'applied' if rec['overstack_protected'] else 'not needed'}",
        "  stack (identity first):",
    ]
    for x in rec["recommended_stack"]:
        lines.append(f"    {x['name']:<30} {x['role']:<24} "
                     f"{x['weight']:.2f}")
    lines.append(f"  prompt pattern: {rec['prompt_pattern']}")
    lines.append(f"  negative pattern: {rec['negative_pattern']}")
    for a in rec["alternatives"]:
        names = " + ".join(f"{i['name']}@{i['weight']:.2f}"
                           for i in a["stack"])
        lines.append(f"  alternative: {a['label']}: {names}")
    for w in rec["warnings"]:
        lines.append(f"  WARNING: {w}")
    lines.append(f"  reason codes: {', '.join(rec['reason_codes'])}")
    return "\n".join(lines)
