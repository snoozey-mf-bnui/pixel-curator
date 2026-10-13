"""Deterministic readiness scoring for datasets, runs and batches.

Readiness is a 0..1 score with a professional level label, an
actionable recommendation, blockers, warnings and reason codes. Scores
are lightweight and metric-driven (no model files, no GPU) and every
reader tolerates missing metadata - an absent signal lowers confidence
rather than raising an exception.
"""

from __future__ import annotations

from pathlib import Path

# Score -> professional level label
READINESS_LEVELS = (
    (0.80, "strong"),
    (0.60, "adequate"),
    (0.40, "developing"),
    (0.0, "insufficient"),
)


def readiness_level(score: float) -> str:
    for threshold, label in READINESS_LEVELS:
        if score >= threshold:
            return label
    return "insufficient"


def _clamp(v: float) -> float:
    return round(max(0.0, min(1.0, v)), 3)


# ---------------------------------------------------------------------------
# Dataset readiness
# ---------------------------------------------------------------------------

def dataset_readiness(conn, dataset_version: int) -> dict:
    """Score whether a dataset is ready for identity-preserving training.

    Combines selected-frame volume, duplicate density, quality
    distribution (sharpness / brightness / aesthetic), identity
    confidence, framing diversity and study-tag coverage. Read-only.
    """
    from ..insights import quality_dashboard
    blockers: list[str] = []
    warnings: list[str] = []
    codes: list[str] = []

    try:
        dash = quality_dashboard(conn, dataset=dataset_version)
    except Exception as exc:
        return {
            "dataset_version": dataset_version,
            "readiness_score": 0.0,
            "readiness_level": "insufficient",
            "recommendation": "Dataset metrics unavailable - build or "
                              "curate the dataset first.",
            "blockers": [f"quality metrics unavailable: {exc}"],
            "warnings": [], "reason_codes": ["dataset_metrics_unavailable"],
        }

    # signals from preset recommender (identity / framing / clusters)
    signals: dict = {}
    try:
        from ..train.preset_recommender import collect_signals
        signals = collect_signals(conn, dataset_version)
    except Exception:
        warnings.append("Dataset-version signals unavailable - scoring "
                        "from global quality metrics only.")
        codes.append("dataset_signals_unavailable")

    selected = dash["counts"]["selected"]
    needs_review = dash["counts"]["needs_review"]
    score = 0.0

    # --- volume (35%) ---
    if selected >= 300:
        score += 0.35
        codes.append("frame_volume_strong")
    elif selected >= 120:
        score += 0.25
        codes.append("frame_volume_adequate")
    elif selected >= 40:
        score += 0.12
        codes.append("frame_volume_low")
        warnings.append(f"Only {selected} curated frames - 120+ is "
                        "recommended for a robust identity LoRA.")
    else:
        codes.append("frame_volume_insufficient")
        blockers.append(f"Only {selected} curated frames - curate more "
                        "before training.")

    # --- duplicate density (15%) ---
    analyzed = max(1, dash["images_analyzed"])
    dup_density = round(dash["duplicates"]["review_candidates"] / analyzed,
                        3)
    if dup_density <= 0.10:
        score += 0.15
        codes.append("duplicate_density_low")
    elif dup_density <= 0.25:
        score += 0.08
        codes.append("duplicate_density_moderate")
    else:
        codes.append("duplicate_density_elevated")
        warnings.append(f"Duplicate candidate density {dup_density:.0%} - "
                        "collapse duplicates before training.")

    # --- identity confidence (25%) ---
    id_med = dash["quality"]["identity_confidence"]["median"]
    id_strength = signals.get("identity_strength")
    id_val = id_strength if id_strength is not None else id_med
    if id_val is None:
        warnings.append("No identity confidence scores - run smart "
                        "curation with an identity anchor for a reliable "
                        "readiness score.")
        codes.append("identity_confidence_unscored")
    elif id_val >= 0.60:
        score += 0.25
        codes.append("identity_confidence_strong")
    elif id_val >= 0.45:
        score += 0.15
        codes.append("identity_confidence_moderate")
    else:
        score += 0.05
        codes.append("identity_confidence_low")
        warnings.append(f"Identity confidence {id_val:.2f} is low - "
                        "review framing and anchor quality.")

    # --- quality distribution (15%) ---
    warn_counts = dash["quality"]["warning_counts"]
    quality_warns = (warn_counts.get("low_sharpness", 0)
                     + warn_counts.get("low_light", 0)
                     + warn_counts.get("overexposed", 0))
    if quality_warns <= analyzed * 0.1:
        score += 0.15
        codes.append("quality_distribution_balanced")
    elif quality_warns <= analyzed * 0.3:
        score += 0.08
        codes.append("quality_distribution_acceptable")
    else:
        codes.append("quality_distribution_weak")
        warnings.append("Many frames carry sharpness/exposure warnings - "
                        "tighten curation thresholds.")

    # --- framing / study diversity (10%) ---
    framing = dash["quality"]["framing_distribution"]
    distinct_framing = len([k for k in framing if k != "unscored"])
    if distinct_framing >= 3:
        score += 0.10
        codes.append("framing_diversity_sufficient")
    elif distinct_framing >= 2:
        score += 0.05
        codes.append("framing_diversity_limited")
    else:
        codes.append("framing_diversity_low")
        warnings.append("Limited framing diversity - add varied "
                        "compositions for a more flexible LoRA.")

    if needs_review > selected and selected:
        warnings.append(f"{needs_review} frames still await review - "
                        "complete curation for a confident score.")
        codes.append("review_backlog_present")

    score = _clamp(score)
    if blockers:
        # a hard blocker (e.g. too few frames) means not ready, regardless
        # of how the other signals score
        score = min(score, 0.39)
    level = readiness_level(score)
    if blockers:
        rec = ("Not ready for training - resolve blockers, then "
               "re-score.")
    elif level in ("strong", "adequate"):
        rec = ("Ready for identity-preserving concept training - proceed "
               "to preset selection and a training run.")
    else:
        rec = ("Developing - expand curated frames and resolve warnings "
               "before training.")

    return {
        "dataset_version": dataset_version,
        "readiness_score": score,
        "readiness_level": level,
        "recommendation": rec,
        "selected_frames": selected,
        "duplicate_density": dup_density,
        "identity_confidence": id_val,
        "blockers": blockers,
        "warnings": warnings,
        "reason_codes": codes,
    }


# ---------------------------------------------------------------------------
# Run readiness
# ---------------------------------------------------------------------------

def _checkpoints(prj, run_name: str) -> list[str]:
    try:
        from ..monitor import checkpoints_for
        return [p.name for p in checkpoints_for(prj, run_name)]
    except Exception:
        return []


def run_readiness(conn, prj, run_name: str) -> dict:
    """Score whether a training run is ready for evaluation, model card
    and stacking. Read-only; tolerates missing sweep/eval/card meta."""
    from .. import manifest as mf
    from ..train.kohya import list_runs

    rows = list_runs(conn)
    row = next((r for r in rows if r["name"] == run_name), None)
    if row is None:
        raise KeyError(f"no training run named '{run_name}'")

    blockers: list[str] = []
    warnings: list[str] = []
    codes: list[str] = []
    score = 0.0
    status = row["status"]

    # --- training status (30%) ---
    if status == "completed":
        score += 0.30
        codes.append("training_complete")
    elif status == "running":
        codes.append("training_active")
        warnings.append("Training run is still active - evaluation and "
                        "stacking should wait until it completes.")
    elif status in ("failed", "error"):
        blockers.append(f"Run status '{status}' - resolve and retrain.")
        codes.append("training_failed")
    else:
        codes.append(f"training_status_{status}")

    # --- checkpoints (20%) ---
    cps = _checkpoints(prj, run_name)
    if cps:
        score += 0.20
        codes.append("checkpoints_present")
    else:
        warnings.append("No checkpoints located yet on disk.")
        codes.append("checkpoints_missing")

    # --- best-epoch sweep (15%) ---
    sweep = mf.meta_get(conn, f"sweep:{run_name}") or {}
    if sweep.get("best"):
        score += 0.15
        codes.append("best_epoch_selected")
        if sweep["best"].get("overfit_warning"):
            warnings.append("Best-epoch sweep flagged overfit risk on "
                            "every scored epoch.")
            codes.append("overfit_risk_flagged")
    else:
        codes.append("best_epoch_sweep_pending")

    # --- evaluation (20%) ---
    report = mf.meta_get(conn, f"evaluation:{run_name}") or {}
    if report:
        score += 0.20
        codes.append("evaluation_complete")
    else:
        codes.append("evaluation_pending")

    # --- model card (15%) ---
    card = mf.meta_get(conn, f"model_card:{run_name}") or {}
    if card:
        score += 0.15
        codes.append("model_card_complete")
    else:
        codes.append("model_card_pending")

    score = _clamp(score)
    level = readiness_level(score)
    if status == "running":
        rec = "Training active - monitor progress; sweep and evaluate " \
              "once it completes."
    elif not report:
        rec = "Run the post-training evaluation matrix next."
    elif not card:
        rec = "Generate the professional model card next."
    else:
        rec = "Run complete and documented - recommend a stack and plan " \
              "a figure-study batch."

    return {
        "run": run_name,
        "status": status,
        "dataset_version": row["dataset_version"],
        "readiness_score": score,
        "readiness_level": level,
        "recommendation": rec,
        "checkpoints": len(cps),
        "has_sweep": bool(sweep.get("best")),
        "has_evaluation": bool(report),
        "has_model_card": bool(card),
        "blockers": blockers,
        "warnings": warnings,
        "reason_codes": codes,
    }


# ---------------------------------------------------------------------------
# Batch readiness
# ---------------------------------------------------------------------------

def batch_readiness(output_base: Path, batch_id: str) -> dict:
    """Score whether a figure-study batch is ready for review/export.
    File-backed; never raises on missing assets."""
    from ..artistic.review import (
        find_batch_dir,
        load_batch_manifest,
        review_summary,
        select_best_figure_study_variant,
    )
    bd = find_batch_dir(output_base, batch_id)
    plan = load_batch_manifest(bd)
    summary = review_summary(bd)

    blockers: list[str] = []
    warnings: list[str] = []
    codes: list[str] = []
    score = 0.0

    # --- manifest completeness (25%) ---
    have = [f for f in ("batch_manifest.json", "prompts.json",
                        "guidance_payloads.json", "review_manifest.json")
            if (bd / f).exists()]
    score += 0.25 * (len(have) / 4)
    if len(have) == 4:
        codes.append("manifest_complete")
    else:
        warnings.append("Batch manifest set is incomplete.")
        codes.append("manifest_incomplete")

    # --- guidance completeness (15%) ---
    guidance = (plan.get("guidance") or {}).get("guidance") or {}
    if guidance:
        score += 0.15
        codes.append("guidance_payload_present")
    else:
        codes.append("guidance_payload_absent")

    # --- rendered samples (20%) ---
    sample_count = 0
    for sub in ("selected", "refined", "samples"):
        d = bd / sub
        if d.is_dir():
            sample_count += len([p for p in d.iterdir() if p.is_file()])
    if sample_count:
        score += 0.20
        codes.append("samples_rendered")
    else:
        warnings.append("No rendered samples yet - this is a planning-"
                        "only batch.")
        codes.append("samples_pending")

    # --- review status (25%) ---
    reviewed = sum(n for s, n in summary["status_counts"].items()
                   if s != "unreviewed")
    total = max(1, summary["total_items"])
    if summary.get("selected_variant"):
        score += 0.25
        codes.append("best_variant_selected")
    elif reviewed:
        score += 0.12
        codes.append("review_in_progress")
    else:
        codes.append("review_pending")
        warnings.append("No items reviewed yet - run figure-review.")

    # --- warnings / issues (15%) ---
    issues = sum(summary.get("issue_counts", {}).values())
    if issues == 0 and not plan.get("warnings"):
        score += 0.15
        codes.append("no_unresolved_warnings")
    elif issues <= total * 0.3:
        score += 0.07
        codes.append("minor_issues_flagged")
    else:
        codes.append("issues_flagged")
        warnings.append(f"{issues} review issues flagged - resolve with "
                        "refinement passes.")

    best = select_best_figure_study_variant(plan)

    score = _clamp(score)
    level = readiness_level(score)
    if summary.get("selected_variant"):
        rec = "Best variant selected - export a production package / " \
              "studio handoff."
    elif not sample_count:
        rec = "Render the planned passes, then review and select the " \
              "strongest variant."
    else:
        rec = "Run figure-review to mark and select the strongest " \
              "variant."

    return {
        "batch_id": bd.name,
        "recipe": plan.get("recipe"),
        "readiness_score": score,
        "readiness_level": level,
        "recommendation": rec,
        "total_items": summary["total_items"],
        "rendered_samples": sample_count,
        "selected_variant": summary.get("selected_variant"),
        "suggested_best_variant": best.get("selected"),
        "blockers": blockers,
        "warnings": warnings,
        "reason_codes": codes,
    }
