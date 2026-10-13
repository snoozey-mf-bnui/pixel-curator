"""Production handoff exporter.

Bundles a complete, portable production artifact set for a run and/or a
figure-study batch: model card, evaluation summary, recommended stack,
selected prompts + payloads, dataset readiness, asset registry,
reproducibility notes and a professional README. Dry-run safe and
missing-asset tolerant - absent files become metadata + warnings, never
failures. Non-destructive: everything lands in a fresh timestamped
directory; sources are copied, never moved.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _meta(conn, key: str) -> dict:
    try:
        from .. import manifest as mf
        return mf.meta_get(conn, key) or {}
    except Exception:
        return {}


def _reproducibility(prj, state: dict, run: str, batch: str) -> dict:
    from .. import __version__ as ver
    return {
        "studio_version": ver,
        "trigger_token": getattr(prj, "trigger_token", ""),
        "class_word": getattr(prj, "class_word", ""),
        "base_model": getattr(prj, "base_model", ""),
        "run": run or None,
        "batch": batch or None,
        "device": state.get("runtime", {}).get("device"),
        "note": "Regenerate with the same base model, identity anchor "
                "and seeds. LoRAs are invoked in the prompt the webui "
                "way: <lora:name:weight>.",
    }


def _readme(name: str, run: str, batch: str, included: list[str],
            missing: list[str], state: dict, best: dict | None) -> str:
    lines = [
        f"# Studio Production Handoff - {name}", "",
        f"Project: {state.get('project')}  |  lifecycle stage: "
        f"{state.get('current_stage')}",
    ]
    if run:
        lines.append(f"Run: {run}")
    if batch:
        lines.append(f"Figure-study batch: {batch}")
        if best and best.get("selected"):
            lines.append(f"Selected variant: {best['selected']}")
    lines += ["", "## Included artifacts", ""]
    lines += [f"- {f}" for f in included] or ["- (metadata only)"]
    if missing:
        lines += ["", "## Pending / missing assets", ""]
        lines += [f"- {m}" for m in missing]
    lines += ["", "## Next steps", ""]
    na = state.get("next_actions", [])
    if na:
        for a in na[:4]:
            lines.append(f"- {a['title']}: `{a['command']}`")
    else:
        lines.append("- Handoff complete - ship the selected variant.")
    lines += ["", "Personal, local production handoff - identity "
              "preservation governs every pass.", ""]
    return "\n".join(lines)


def export_studio_handoff(prj, output_base: Path | None = None,
                          run_name: str = "", batch_id: str = "",
                          dataset_version: int | None = None,
                          include_samples: bool = False,
                          dry_run: bool = False) -> dict:
    """Export a production handoff bundle.

    Returns a manifest dict (and, unless dry_run, writes the package
    directory under output_base/studio_handoffs/). Missing artifacts are
    recorded as warnings; the export never raises on absent files.
    """
    from .. import manifest as mf
    from .registry import build_asset_registry
    from .state import build_studio_state

    base = Path(output_base or prj.output_path).expanduser()
    state = build_studio_state(prj, base, dataset_version=dataset_version,
                               run_name=run_name, batch_id=batch_id)
    ref = state["references"]
    run = run_name or ref.get("run", "")
    batch = batch_id or ref.get("batch", "")
    ds_v = dataset_version or ref.get("dataset_version")

    included: list[str] = []
    missing: list[str] = []
    artifacts: dict = {}

    conn = None
    try:
        conn = mf.connect(base, readonly=True)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass

    # ---- run-scoped artifacts ----
    best_variant = None
    if conn is not None and run:
        evaluation = _meta(conn, f"evaluation:{run}")
        if evaluation:
            artifacts["evaluation_summary"] = evaluation
            included.append("evaluation_summary.json")
        else:
            missing.append(f"evaluation report for run '{run}' "
                           "(run: eval --run)")
        card = _meta(conn, f"model_card:{run}")
        if card.get("path"):
            artifacts["model_card_path"] = card["path"]
            included.append("model_card.md")
        else:
            missing.append(f"model card for run '{run}' (run: card --run)")
        try:
            from ..artistic.stack_recommender import recommend_best_epoch_and_stack
            artifacts["recommended_stack"] = \
                recommend_best_epoch_and_stack(conn, prj, run)
            included.append("recommended_stack.json")
        except Exception as exc:
            missing.append(f"recommended stack ({exc})")

    # ---- dataset readiness ----
    if conn is not None and ds_v:
        try:
            from .readiness import dataset_readiness
            artifacts["dataset_readiness"] = dataset_readiness(conn, ds_v)
            included.append("dataset_readiness.json")
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass

    # ---- batch-scoped artifacts ----
    batch_dir = None
    if batch:
        try:
            from ..artistic.review import export_review, find_batch_dir, load_batch_manifest
            batch_dir = find_batch_dir(base, batch)
            plan = load_batch_manifest(batch_dir)
            review = export_review(batch_dir)
            best_variant = review.get("best_variant")
            artifacts["batch_manifest"] = plan
            artifacts["review_manifest"] = review["review"]
            artifacts["best_variant"] = best_variant
            included += ["batch_manifest.json", "review_manifest.json"]
            items = plan.get("items") or [it for p in plan.get("passes", [])
                                          for it in p.get("items", [])]
            artifacts["selected_prompts"] = [
                {"item_id": it.get("item_id"), "seed": it.get("seed"),
                 "prompt": it.get("prompt"),
                 "negative_prompt": it.get("negative_prompt"),
                 "lora_stack": it.get("lora_stack")}
                for it in items]
            artifacts["generation_payloads"] = (plan.get("guidance") or {})
            included += ["selected_prompts.json", "generation_payloads.json"]
        except Exception as exc:
            missing.append(f"figure-study batch '{batch}' ({exc})")

    # ---- asset registry ----
    try:
        artifacts["asset_registry"] = build_asset_registry(
            prj, conn=conn, output_base=base)
        included.append("asset_registry.json")
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    if conn is not None:
        conn.close()

    artifacts["reproducibility"] = _reproducibility(prj, state, run, batch)
    included.append("reproducibility.json")

    name_part = run or batch or (f"dataset_v{ds_v:03d}" if ds_v
                                 else "studio")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pkg_name = f"{stamp}_{name_part}"
    readme = _readme(pkg_name, run, batch, included, missing, state,
                     best_variant)

    result = {
        "handoff_name": pkg_name,
        "run": run or None,
        "batch": batch or None,
        "dataset_version": ds_v,
        "included": included,
        "missing": missing,
        "best_variant": best_variant,
        "dry_run": dry_run,
    }

    if dry_run:
        result["package_dir"] = None
        result["plan"] = "Dry run - no files written. Re-run without "\
                          "--dry-run to export."
        return result

    root = base / "studio_handoffs"
    pkg = root / pkg_name
    n = 2
    while pkg.exists():
        pkg = root / f"{pkg_name}_{n}"
        n += 1
    pkg.mkdir(parents=True)

    # JSON artifacts
    json_map = {
        "evaluation_summary.json": "evaluation_summary",
        "recommended_stack.json": "recommended_stack",
        "dataset_readiness.json": "dataset_readiness",
        "batch_manifest.json": "batch_manifest",
        "review_manifest.json": "review_manifest",
        "selected_prompts.json": "selected_prompts",
        "generation_payloads.json": "generation_payloads",
        "asset_registry.json": "asset_registry",
        "reproducibility.json": "reproducibility",
    }
    for fname, key in json_map.items():
        if key in artifacts:
            (pkg / fname).write_text(
                json.dumps(artifacts[key], indent=2), encoding="utf-8")

    # model card markdown (copied if present)
    mc = artifacts.get("model_card_path")
    if mc and Path(mc).exists():
        try:
            shutil.copy2(mc, pkg / "model_card.md")
        except Exception:
            missing.append("model card file could not be copied")

    # optional sample images from the batch's selected/refined/samples
    copied = 0
    if include_samples and batch_dir is not None:
        for sub in ("selected", "refined", "samples"):
            src = batch_dir / sub
            if src.is_dir():
                files = [p for p in sorted(src.iterdir()) if p.is_file()]
                if files:
                    dst = pkg / "samples" / sub
                    dst.mkdir(parents=True, exist_ok=True)
                    for p in files:
                        shutil.copy2(p, dst / p.name)
                        copied += 1
    if include_samples and copied == 0:
        missing.append("no rendered sample assets to include")

    (pkg / "README.md").write_text(readme, encoding="utf-8")
    result["package_dir"] = str(pkg)
    result["copied_samples"] = copied
    return result
