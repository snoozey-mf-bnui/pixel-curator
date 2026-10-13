"""Closed-loop figure study batch review - marking, summary, selection.

Lightweight, file-backed review metadata for figure study production
batches: each batch directory carries a review_manifest.json that the
CLI, API and (optionally) the Review UI can update. Best-variant
selection combines identity preservation estimates, guidance
completeness, review status/tags and warning counts - usable before any
image exists, and stronger once review marks land.

Non-destructive: review updates rewrite only review_manifest.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REVIEW_STATUSES = ("unreviewed", "selected", "needs_refinement",
                   "rejected", "strong_candidate", "final_candidate")

ISSUE_TAGS = ("identity_consistency_issue", "pose_issue",
              "garment_boundary_issue", "skin_texture_highlight_issue",
              "lighting_issue", "anatomical_coherence_issue")

POSITIVE_TAGS = ("strong_identity_consistency", "balanced_lighting",
                 "clean_garment_boundary", "refined_skin_texture",
                 "strong_composition", "editorial_pose_quality")

_STATUS_SCORE = {"final_candidate": 0.30, "selected": 0.25,
                 "strong_candidate": 0.15, "unreviewed": 0.0,
                 "needs_refinement": -0.15, "rejected": -1.0}


def find_batch_dir(output_base: Path, batch_id: str) -> Path:
    """Resolve a batch id to its directory under figure_studies/.
    Accepts the exact directory name or a unique substring."""
    root = Path(output_base) / "figure_studies"
    exact = root / batch_id
    if exact.is_dir():
        return exact
    if not root.is_dir():
        raise FileNotFoundError(
            "No figure study batches found yet - plan one with "
            "generate-figure-study first.")
    matches = [d for d in sorted(root.iterdir())
               if d.is_dir() and batch_id in d.name]
    if not matches:
        raise FileNotFoundError(
            f"No figure study batch matching '{batch_id}'. "
            f"Available: {', '.join(d.name for d in sorted(root.iterdir()) if d.is_dir()) or '(none)'}")
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Batch id '{batch_id}' is ambiguous: "
            f"{', '.join(d.name for d in matches)}")
    return matches[0]


def list_batches(output_base: Path) -> list[dict]:
    """Catalog of figure study batches (newest first)."""
    root = Path(output_base) / "figure_studies"
    out: list[dict] = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        bm = d / "batch_manifest.json"
        if not (d.is_dir() and bm.exists()):
            continue
        try:
            plan = json.loads(bm.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            continue
        rv = _load_review(d, create=False)
        items = (plan.get("summary", {}).get("total_items")
                 or plan.get("count") or len(plan.get("items", [])))
        out.append({
            "batch_id": d.name,
            "recipe": plan.get("recipe"),
            "items": items,
            "passes": plan.get("pass_names", []),
            "review_status": (rv or {}).get("review_status",
                                            "unreviewed"),
            "selected_variant": (rv or {}).get("selected_variant"),
        })
    return out


def _load_review(batch_dir: Path, create: bool = True) -> dict | None:
    rp = Path(batch_dir) / "review_manifest.json"
    if rp.exists():
        return json.loads(rp.read_text(encoding="utf-8"))
    if not create:
        return None
    # legacy Phase 4 batches: synthesize review entries from the manifest
    plan = load_batch_manifest(batch_dir)
    items = plan.get("items") or [it for p in plan.get("passes", [])
                                  for it in p.get("items", [])]
    review = {"batch": Path(batch_dir).name,
              "review_status": "unreviewed", "selected_variant": None,
              "items": [{"item_id": it.get("item_id",
                                           f"item_{it.get('index', i):04d}"),
                         "review_status": "unreviewed",
                         "review_tags": [], "review_notes": "",
                         "next_action": ""}
                        for i, it in enumerate(items)]}
    _save_review(batch_dir, review)
    return review


def _save_review(batch_dir: Path, review: dict) -> None:
    (Path(batch_dir) / "review_manifest.json").write_text(
        json.dumps(review, indent=2), encoding="utf-8")


def load_batch_manifest(batch_dir: Path) -> dict:
    return json.loads((Path(batch_dir) / "batch_manifest.json")
                      .read_text(encoding="utf-8"))


def mark_item(batch_dir: Path, item_id: str,
              status: str | None = None,
              tags: list[str] | None = None,
              notes: str = "",
              next_action: str = "") -> dict:
    """Update one item's review entry; returns the updated entry."""
    if status is not None and status not in REVIEW_STATUSES:
        raise KeyError(f"unknown review status '{status}'. "
                       f"Options: {', '.join(REVIEW_STATUSES)}")
    for t in tags or []:
        if t not in ISSUE_TAGS + POSITIVE_TAGS:
            raise KeyError(
                f"unknown review tag '{t}'. Options: "
                f"{', '.join(ISSUE_TAGS + POSITIVE_TAGS)}")
    review = _load_review(batch_dir)
    entry = next((e for e in review["items"]
                  if e["item_id"] == item_id), None)
    if entry is None:
        known = ", ".join(e["item_id"] for e in review["items"][:8])
        raise KeyError(f"unknown item '{item_id}' in this batch "
                       f"(first items: {known} ...)")
    if status:
        entry["review_status"] = status
        if status == "selected":
            review["selected_variant"] = item_id
            review["review_status"] = "reviewed"
    if tags:
        entry["review_tags"] = sorted(set(entry["review_tags"]) | set(tags))
    if notes:
        entry["review_notes"] = notes
    if next_action:
        entry["next_action"] = next_action
    _save_review(batch_dir, review)
    return entry


def review_summary(batch_dir: Path) -> dict:
    """Batch review roll-up for CLI/API display."""
    plan = load_batch_manifest(batch_dir)
    review = _load_review(batch_dir)
    counts: dict[str, int] = {}
    issues: dict[str, int] = {}
    for e in review["items"]:
        counts[e["review_status"]] = counts.get(e["review_status"], 0) + 1
        for t in e["review_tags"]:
            if t in ISSUE_TAGS:
                issues[t] = issues.get(t, 0) + 1
    return {
        "batch_id": Path(batch_dir).name,
        "recipe": plan.get("recipe"),
        "passes": plan.get("pass_names", []),
        "total_items": len(review["items"]),
        "status_counts": counts,
        "issue_counts": issues,
        "selected_variant": review.get("selected_variant"),
        "identity_estimate_min":
            plan.get("summary", {}).get("identity_estimate_min"),
        "warnings": plan.get("warnings", []),
    }


def _guidance_completeness(plan: dict) -> float:
    g = (plan.get("guidance") or {}).get("guidance") or {}
    if not g:
        return 0.0
    enabled = sum(1 for b in g.values() if b.get("enabled"))
    return round(enabled / max(len(g), 1), 3)


def select_best_figure_study_variant(
        batch_manifest: dict,
        review_manifest: dict | None = None) -> dict:
    """Score every planned item and return the strongest candidate.

    Scoring (deterministic, metric-driven): identity preservation
    estimate dominates; review status and positive tags lift, issue tags
    and per-item warnings reduce; guidance completeness adds a small
    bonus; rejected items never win.
    """
    review_by_id = {}
    if review_manifest:
        review_by_id = {e["item_id"]: e
                        for e in review_manifest.get("items", [])}
    items = batch_manifest.get("items") or \
        [it for p in batch_manifest.get("passes", [])
         for it in p.get("items", [])]
    g_bonus = 0.05 * _guidance_completeness(batch_manifest)

    scored: list[dict] = []
    for i, it in enumerate(items):
        item_id = it.get("item_id", f"item_{it.get('index', i):04d}")
        rv = review_by_id.get(item_id, {})
        status = rv.get("review_status", "unreviewed")
        score = float(it.get("identity_score_estimate", 0.5))
        score += _STATUS_SCORE.get(status, 0.0)
        score += 0.05 * len([t for t in rv.get("review_tags", [])
                             if t in POSITIVE_TAGS])
        score -= 0.10 * len([t for t in rv.get("review_tags", [])
                             if t in ISSUE_TAGS])
        score -= 0.02 * len(it.get("warnings", []))
        score += g_bonus
        scored.append({"item_id": item_id, "score": round(score, 4),
                       "review_status": status,
                       "seed": it.get("seed"),
                       "identity_score_estimate":
                           it.get("identity_score_estimate"),
                       "rejected": status == "rejected"})

    if not scored:
        return {"selected": None, "reason": "No planned items found.",
                "candidates": []}
    pool = [s for s in scored if not s["rejected"]] or scored
    best = max(pool, key=lambda s: (s["score"], -ord(s["item_id"][-1])))
    reasons = ["identity_preservation_priority"]
    if best["review_status"] in ("selected", "final_candidate"):
        reasons.append("user_review_confirmed")
    if g_bonus > 0:
        reasons.append("guidance_completeness_bonus")
    return {"selected": best["item_id"], "score": best["score"],
            "reason_codes": reasons,
            "candidates": sorted(scored, key=lambda s: -s["score"])[:8]}


def export_review(batch_dir: Path) -> dict:
    """Full review export payload (CLI --export / API)."""
    plan = load_batch_manifest(batch_dir)
    review = _load_review(batch_dir)
    best = select_best_figure_study_variant(plan, review)
    return {"batch_id": Path(batch_dir).name,
            "summary": review_summary(batch_dir),
            "review": review, "best_variant": best}


def render_review_summary(s: dict) -> str:
    lines = [f"FIGURE STUDY REVIEW | batch '{s['batch_id']}' "
             f"(recipe {s['recipe']})",
             f"  items: {s['total_items']} | passes: "
             f"{', '.join(s['passes']) or '(single pass)'}"]
    for status, n in sorted(s["status_counts"].items()):
        lines.append(f"    {status:<20} {n}")
    if s["issue_counts"]:
        lines.append("  issues flagged:")
        for tag, n in sorted(s["issue_counts"].items()):
            lines.append(f"    {tag:<32} {n}")
    if s["selected_variant"]:
        lines.append(f"  selected variant: {s['selected_variant']}")
    for w in s["warnings"]:
        lines.append(f"  WARNING: {w}")
    return "\n".join(lines)
