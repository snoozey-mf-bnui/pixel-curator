"""Production output packaging for figure study batches.

Gathers everything a finished (or planned) figure study batch produced
into one portable, reviewable package: batch manifest, selected
prompts, guidance payloads, review manifest, best-variant verdict,
recommended stack and a README summary with next-pass recommendations.

Dry-run safe and missing-asset tolerant: when sample images do not
exist yet (planning-only batches), the package still assembles all
metadata and clearly reports which assets are pending. Non-destructive:
packages land in their own timestamped directory; sources are copied,
never moved.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from .review import export_review, find_batch_dir, load_batch_manifest


def _selected_prompts(plan: dict, best_id: str | None,
                      review: dict) -> list[dict]:
    """Prompts for selected/strong items; falls back to the best
    variant, then to every generation item."""
    chosen = {e["item_id"] for e in review.get("items", [])
              if e.get("review_status") in ("selected", "final_candidate",
                                            "strong_candidate")}
    if best_id:
        chosen.add(best_id)
    items = plan.get("items") or [it for p in plan.get("passes", [])
                                  for it in p.get("items", [])]
    out = [{"item_id": it.get("item_id"), "seed": it.get("seed"),
            "prompt": it.get("prompt"),
            "negative_prompt": it.get("negative_prompt"),
            "lora_stack": it.get("lora_stack")}
           for it in items
           if not chosen or it.get("item_id") in chosen]
    return out


def _collect_samples(batch_dir: Path, package_dir: Path,
                     missing: list[str]) -> int:
    """Copy existing sample assets (samples/refined/selected) into the
    package; record pending directories in `missing`."""
    copied = 0
    for sub in ("selected", "refined", "samples"):
        src = batch_dir / sub
        if not src.is_dir():
            continue
        files = [p for p in sorted(src.iterdir()) if p.is_file()]
        if not files:
            missing.append(f"{sub}/ contains no rendered assets yet")
            continue
        dst = package_dir / "samples" / sub
        dst.mkdir(parents=True, exist_ok=True)
        for p in files:
            shutil.copy2(p, dst / p.name)
            copied += 1
    return copied


def _recommended_stack(plan: dict, conn=None, prj=None) -> dict:
    """Best available stack: run-aware when a manifest connection is
    supplied, otherwise the recipe-driven layering recommendation."""
    try:
        if conn is not None:
            from .stack_recommender import recommend_best_epoch_and_stack
            return recommend_best_epoch_and_stack(conn, prj, "")
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    try:
        from .stack_recommender import recommend_figure_study_stack
        return recommend_figure_study_stack(plan["recipe"], prj=prj)
    except Exception as exc:
        return {"error": f"Stack recommendation unavailable: {exc}"}


def _readme(plan: dict, summary: dict, best: dict,
            missing: list[str], copied: int) -> str:
    next_steps: list[str] = []
    if missing:
        next_steps.append("Render the planned passes (the prompts and "
                          "guidance payloads in this package are "
                          "generation-ready).")
    if not summary.get("selected_variant"):
        next_steps.append("Run a review pass: figure-review --batch "
                          f"{summary['batch_id']} --summary, then "
                          "--select the strongest variant.")
    issue_counts = summary.get("issue_counts") or {}
    if issue_counts:
        worst = max(issue_counts, key=issue_counts.get)
        next_steps.append(f"Most flagged issue: {worst} - plan a "
                          "targeted refinement pass for it.")
    if not next_steps:
        next_steps.append("Batch reviewed and selected - proceed to the "
                          "final refinement pass or ship the selected "
                          "variant.")

    lines = [
        f"# Figure Study Production Package - {summary['batch_id']}", "",
        f"Recipe: {plan.get('recipe')}  |  passes: "
        f"{', '.join(plan.get('pass_names', [])) or '(single pass)'}",
        f"Planned items: {summary.get('total_items')}  |  identity "
        f"estimate (min): {summary.get('identity_estimate_min')}",
        f"Review status: {', '.join(f'{k}: {v}' for k, v in sorted(summary.get('status_counts', {}).items())) or 'unreviewed'}",
        f"Best variant: {best.get('selected') or '(not yet determined)'}"
        + (f" (score {best.get('score')})" if best.get("selected") else ""),
        "",
        "## Contents", "",
        "batch_manifest.json - full multi-pass production plan",
        "selected_prompts.json - prompts for selected/strong variants",
        "guidance_payloads.json - ControlNet / IP-Adapter / FaceID plans",
        "review_manifest.json - review marks, tags and notes",
        "recommended_stack.json - identity-first stack recommendation",
        f"samples/ - {copied} rendered asset(s) copied" if copied
        else "samples/ - no rendered assets yet (planning-only batch)",
        "",
        "## Pending assets", "",
    ]
    lines += [f"- {m}" for m in missing] or ["- none"]
    lines += ["", "## Recommended next steps", ""]
    lines += [f"{i + 1}. {s}" for i, s in enumerate(next_steps)]
    lines += ["", "Personal, local production package - "
              "identity preservation governs every pass.", ""]
    return "\n".join(lines)


def package_figure_study(output_base: Path, batch_id: str,
                         conn=None, prj=None) -> dict:
    """Assemble a production package for a batch.

    Returns {"package_dir": ..., "copied_assets": n, "missing": [...],
    "best_variant": {...}} - and writes the package directory."""
    batch_dir = find_batch_dir(output_base, batch_id)
    plan = load_batch_manifest(batch_dir)
    exported = export_review(batch_dir)
    summary = exported["summary"]
    review = exported["review"]
    best = exported["best_variant"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path(output_base) / "figure_study_packages"
    package_dir = root / f"{stamp}_{batch_dir.name}"
    n = 2
    while package_dir.exists():
        package_dir = root / f"{stamp}_{batch_dir.name}_{n}"
        n += 1
    package_dir.mkdir(parents=True)

    missing: list[str] = []
    copied = _collect_samples(batch_dir, package_dir, missing)

    (package_dir / "batch_manifest.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8")
    (package_dir / "review_manifest.json").write_text(
        json.dumps(review, indent=2), encoding="utf-8")
    (package_dir / "selected_prompts.json").write_text(
        json.dumps(_selected_prompts(plan, best.get("selected"), review),
                   indent=2), encoding="utf-8")
    gp = batch_dir / "guidance_payloads.json"
    (package_dir / "guidance_payloads.json").write_text(
        gp.read_text(encoding="utf-8") if gp.exists()
        else json.dumps({"guidance": plan.get("guidance", {})}, indent=2),
        encoding="utf-8")
    stack = _recommended_stack(plan, conn=conn, prj=prj)
    (package_dir / "recommended_stack.json").write_text(
        json.dumps(stack, indent=2), encoding="utf-8")
    (package_dir / "README.md").write_text(
        _readme(plan, summary, best, missing, copied), encoding="utf-8")

    return {"package_dir": str(package_dir), "batch_id": batch_dir.name,
            "copied_assets": copied, "missing": missing,
            "best_variant": best,
            "selected_variant": summary.get("selected_variant")}
