"""Production state machine + automated next-step recommendations.

`build_studio_state` reads the whole project (datasets, runs, sweep /
evaluation / model-card meta, figure-study batches, packages, runtime
and asset registry) through read-only manifest access and reports the
current production stage, completed/missing stages, blockers, warnings
and confidence. `recommend_next_actions` turns that state into ordered,
copy-paste-ready commands, each flagged for whether it is safe to run
while training is active.

Read-only and degradation-safe: missing modules or metadata lower
confidence and surface professional warnings rather than raising.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Ordered production lifecycle
LIFECYCLE_STAGES = [
    "dataset_curated",
    "dataset_ready",
    "training_active",
    "training_complete",
    "best_epoch_selected",
    "evaluation_complete",
    "model_card_complete",
    "stack_recommended",
    "generation_batch_planned",
    "generation_batch_reviewed",
    "production_package_exported",
]


def _runtime_summary(prj) -> dict:
    try:
        from ..runtime import get_device_name
        device = get_device_name(
            (getattr(prj, "runtime", {}) or {}).get("device", "auto"))
    except Exception:
        device = "cpu"
    guidance = False
    try:
        from ..identity_integration import detect_identity_tools
        guidance = any(v.get("status") == "found"
                       for v in detect_identity_tools(prj).values())
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    note = {"mps": "Conservative batch sizes (4-6); one refinement pass "
                   "at a time.",
            "cuda": "Larger production batches fine; keep active models "
                    "on local storage.",
            "cpu": "Planning and dashboards run on CPU; generation goes "
                   "through the Forge backend."}.get(device, "")
    return {"device": device, "guidance_backends_available": guidance,
            "note": note}


def build_studio_state(prj, output_base: Path | None = None,
                       dataset_version: int | None = None,
                       run_name: str = "", batch_id: str = "") -> dict:
    """Assemble the full studio production state. Read-only."""
    from .. import manifest as mf
    base = Path(output_base or prj.output_path).expanduser()

    completed: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []
    references: dict = {}
    locked = False

    # ---- manifest (read-only) ----
    conn = None
    try:
        conn = mf.connect(base, readonly=True)
    except mf.ManifestLocked:
        locked = True
        warnings.append("Manifest is temporarily locked by an active "
                        "process - showing best-effort state.")
    except Exception as exc:
        warnings.append(f"Manifest unavailable: {exc}")

    datasets: list[dict] = []
    runs: list[dict] = []
    if conn is not None:
        try:
            from ..builder import list_datasets
            datasets = list_datasets(conn)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass
        try:
            from ..train.kohya import list_runs
            runs = list_runs(conn)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass

    # ---- dataset stage ----
    if datasets:
        completed.append("dataset_curated")
        references["datasets"] = [d["version"] for d in datasets]
        ds_v = dataset_version or datasets[-1]["version"]
        references["dataset_version"] = ds_v
        if conn is not None:
            try:
                from .readiness import dataset_readiness
                dr = dataset_readiness(conn, ds_v)
                references["dataset_readiness"] = dr
                if dr["readiness_level"] in ("strong", "adequate"):
                    completed.append("dataset_ready")
                elif dr["blockers"]:
                    blockers += dr["blockers"]
            except Exception:
                logger.debug("suppressed exception", exc_info=True)
                pass
    else:
        warnings.append("No built datasets yet - curate and build a "
                        "dataset version first.")

    # ---- run stages ----
    active = [r for r in runs if r["status"] == "running"]
    done = [r for r in runs if r["status"] == "completed"]
    training_active = bool(active)
    if active:
        completed.append("training_active")
        references["active_run"] = active[-1]["name"]
    target_run = run_name or (done[-1]["name"] if done
                              else (runs[-1]["name"] if runs else ""))
    references["run"] = target_run
    run_rd = None
    if target_run and conn is not None:
        try:
            from .readiness import run_readiness
            run_rd = run_readiness(conn, prj, target_run)
            references["run_readiness"] = run_rd
            if run_rd["status"] == "completed":
                completed.append("training_complete")
            if run_rd["has_sweep"]:
                completed.append("best_epoch_selected")
            if run_rd["has_evaluation"]:
                completed.append("evaluation_complete")
            if run_rd["has_model_card"]:
                completed.append("model_card_complete")
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass

    # ---- stack recommendation (meta marker is optional) ----
    if conn is not None and target_run:
        if (mf.meta_get(conn, f"evaluation:{target_run}")
                or mf.meta_get(conn, f"sweep:{target_run}")):
            # a stack can be recommended as soon as a run is documented
            completed.append("stack_recommended")

    # ---- figure-study batches ----
    batches: list[dict] = []
    try:
        from ..artistic.review import list_batches
        batches = list_batches(base)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    if batches:
        completed.append("generation_batch_planned")
        references["batches"] = [b["batch_id"] for b in batches]
        target_batch = batch_id or batches[0]["batch_id"]
        references["batch"] = target_batch
        if any(b.get("selected_variant") for b in batches):
            completed.append("generation_batch_reviewed")

    # ---- packages / handoffs ----
    pkg_roots = [base / "figure_study_packages", base / "studio_handoffs"]
    if any(r.is_dir() and any(r.iterdir()) for r in pkg_roots):
        completed.append("production_package_exported")

    if conn is not None:
        conn.close()

    # current stage = last completed in lifecycle order
    ordered_done = [s for s in LIFECYCLE_STAGES if s in completed]
    current = ordered_done[-1] if ordered_done else "uninitialized"
    missing = [s for s in LIFECYCLE_STAGES if s not in completed]

    # confidence: how far through the lifecycle, minus warnings/blockers
    confidence = round(min(1.0, len(set(ordered_done))
                           / len(LIFECYCLE_STAGES)
                           + (0.0 if (warnings or blockers) else 0.05)), 3)

    state = {
        "project": getattr(prj, "name", "") or "studio",
        "output_base": str(base),
        "manifest_locked": locked,
        "training_active": training_active,
        "runtime": _runtime_summary(prj),
        "current_stage": current,
        "completed_stages": ordered_done,
        "missing_stages": missing,
        "datasets": datasets,
        "runs": [{"name": r["name"], "status": r["status"],
                  "dataset_version": r["dataset_version"]} for r in runs],
        "batches": batches,
        "references": references,
        "blockers": blockers,
        "warnings": warnings,
        "confidence": confidence,
    }
    state["next_actions"] = recommend_next_actions(state)
    return state


# ---------------------------------------------------------------------------
# Next-step recommendations
# ---------------------------------------------------------------------------

def _action(action_id, title, reason, command, priority="medium",
            safe=True, confidence=0.8, blockers=None) -> dict:
    return {"action_id": action_id, "title": title, "reason": reason,
            "command": command, "priority": priority,
            "safe_during_training": safe, "confidence": confidence,
            "blockers": blockers or []}


def recommend_next_actions(state: dict) -> list[dict]:
    """Ordered, copy-paste-ready next steps from a studio state dict."""
    done = set(state.get("completed_stages", []))
    ref = state.get("references", {})
    run = ref.get("run", "<run_name>")
    ds = ref.get("dataset_version", 1)
    batch = ref.get("batch", "<batch_id>")
    pfx = "pixel-curator -p pixel-curator.toml"
    out: list[dict] = []

    if state.get("manifest_locked"):
        out.append(_action(
            "wait_for_lock",
            "Manifest temporarily locked",
            "An active process is writing to the manifest; read-only "
            "dashboards still work.",
            f"{pfx} studio status", priority="low", safe=True,
            confidence=0.6))

    if "dataset_curated" not in done:
        out.append(_action(
            "build_dataset", "Curate and build a dataset",
            "No built dataset version was found.",
            f"{pfx} insights --dataset 1", priority="high", safe=True))
        return out

    if "dataset_ready" not in done and "training_complete" not in done:
        out.append(_action(
            "improve_dataset", "Raise dataset readiness",
            "Dataset is built but not yet at training-ready readiness.",
            f"{pfx} studio readiness --dataset {ds}",
            priority="high", safe=True))

    if state.get("training_active"):
        out.append(_action(
            "monitor_training", "Monitor active training",
            "A training run is active; evaluation and stacking should "
            "wait until it completes.",
            f"{pfx} studio status", priority="medium", safe=True,
            confidence=0.9))

    if "training_complete" in done:
        if "best_epoch_selected" not in done:
            out.append(_action(
                "run_sweep", "Run the best-epoch sweep",
                "Training completed but no best epoch has been selected.",
                f"{pfx} sweep --run {run}", priority="high", safe=False,
                confidence=0.9))
        if "evaluation_complete" not in done:
            out.append(_action(
                "run_evaluation", "Run post-training evaluation",
                "A completed run has no evaluation report.",
                f"{pfx} eval --run {run}", priority="high", safe=False,
                confidence=0.9))
        elif "model_card_complete" not in done:
            out.append(_action(
                "generate_model_card", "Generate the model card",
                "Evaluation exists but the model card is missing.",
                f"{pfx} card --run {run}", priority="medium", safe=False))
        if ("model_card_complete" in done
                and "stack_recommended" not in done):
            out.append(_action(
                "recommend_stack", "Recommend an identity-first stack",
                "The run is documented; recommend a concept stack.",
                f"{pfx} recommend-stack --run {run}",
                priority="medium", safe=True))

    if ("training_complete" in done
            and "generation_batch_planned" not in done):
        out.append(_action(
            "plan_batch", "Plan a figure-study production batch",
            "No figure-study batch has been planned yet.",
            f"{pfx} generate-figure-study --recipe advanced_figure_study "
            f"--batch 6 --variation 0.35 --dry-run",
            priority="medium", safe=True))

    if ("generation_batch_planned" in done
            and "generation_batch_reviewed" not in done):
        out.append(_action(
            "review_batch", "Review the batch and select a variant",
            "A batch exists but no best variant has been selected.",
            f"{pfx} figure-review --batch {batch} --summary",
            priority="medium", safe=True))

    if ("generation_batch_reviewed" in done
            and "production_package_exported" not in done):
        out.append(_action(
            "export_handoff", "Export the production handoff",
            "A reviewed batch is ready for production handoff.",
            f"{pfx} studio export --batch {batch}",
            priority="medium", safe=True))

    if not out:
        out.append(_action(
            "complete", "Production lifecycle complete",
            "All lifecycle stages are satisfied - ship the selected "
            "variant or start a new study.",
            f"{pfx} studio status", priority="low", safe=True,
            confidence=0.9))
    return out


def render_studio_state(state: dict) -> str:
    rt = state["runtime"]
    lines = [
        f"GROKKIE Studio Control Center  [{state['project']}]",
        "",
        "Runtime:",
        f"  Device: {rt['device']} | guidance backends: "
        f"{'available' if rt['guidance_backends_available'] else 'unavailable'}",
        f"  {rt['note']}" if rt["note"] else "",
        "",
        f"Manifest: {'LOCKED (active process)' if state['manifest_locked'] else 'ready'}"
        f" | training {'ACTIVE' if state['training_active'] else 'idle'}",
        "",
        f"Lifecycle stage: {state['current_stage']}  "
        f"(confidence {state['confidence']:.2f})",
        f"  completed: {', '.join(state['completed_stages']) or '(none)'}",
    ]
    if state["datasets"]:
        lines.append("  datasets: "
                     + ", ".join(f"v{d['version']:03d}"
                                 for d in state["datasets"]))
    if state["runs"]:
        lines.append("  runs:")
        for r in state["runs"][-5:]:
            lines.append(f"    {r['name']:<40} {r['status']}")
    if state["batches"]:
        lines.append(f"  figure-study batches: {len(state['batches'])} "
                     f"(latest: {state['batches'][0]['batch_id']})")
    for b in state["blockers"]:
        lines.append(f"  BLOCKER: {b}")
    for w in state["warnings"]:
        lines.append(f"  WARNING: {w}")
    lines.append("")
    lines.append("Recommended Next Actions:")
    for i, a in enumerate(state["next_actions"], 1):
        tag = "" if a["safe_during_training"] else "  [not during training]"
        lines.append(f"  {i}. {a['title']} [{a['priority']}]{tag}")
        lines.append(f"     {a['reason']}")
        lines.append(f"     $ {a['command']}")
    return "\n".join(x for x in lines if x is not None)
