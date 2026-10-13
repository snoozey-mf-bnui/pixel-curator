"""Specialized Visual Focus Classifier - shared service.

Batch-classifies a dataset directory, persists results (JSONL + summary JSON +
Markdown card), summarizes distributions, and provides loading/filtering helpers
consumed by the CLI, Gradio GUI, legacy UI, and the Dataset Pipeline Explorer.
Read-only on source media; depends only on the core classifier + stdlib.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..curate import visual_focus_cv as vfc

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heic", ".heif")
RESULTS_NAME = "visual_focus_results.jsonl"
SUMMARY_NAME = "visual_focus_summary.json"
CARD_NAME = "visual_focus_card.md"


def _iter_images(dataset_dir: Path, limit: int | None):
    n = 0
    for p in sorted(dataset_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p
            n += 1
            if limit and n >= limit:
                return


def classify_dataset_visual_focus(dataset_dir, output_dir=None,
                                  config: vfc.VisualFocusConfig | None = None,
                                  limit: int | None = None) -> dict:
    """Classify every image under ``dataset_dir`` (recursive, capped by
    ``limit``). Writes results JSONL + summary + card to ``output_dir`` (default:
    ``dataset_dir``). Resumable. Returns the summary dict. Never raises for
    ordinary input; reports warnings."""
    cfg = config or vfc.VisualFocusConfig()
    src = Path(dataset_dir).expanduser()
    out = Path(output_dir).expanduser() if output_dir else src
    warnings: list = []
    if not src.exists():
        return {"error": f"dataset dir not found: {src}", "warnings": [],
                "total": 0}
    try:
        out.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return {"error": f"cannot create output dir: {exc}", "total": 0}
    results_path = out / RESULTS_NAME
    images = list(_iter_images(src, limit))
    if not images:
        warnings.append("no images found under dataset dir")
    vfc.classify_visual_focus_batch(images, output_jsonl=results_path,
                                    config=cfg, resume=True)
    summary = summarize_visual_focus_results(results_path)
    summary["warnings"] = warnings + summary.get("warnings", [])
    summary["dataset_dir"] = str(src)
    summary["results_path"] = str(results_path)
    try:
        (out / SUMMARY_NAME).write_text(json.dumps(summary, indent=2),
                                        encoding="utf-8")
        (out / CARD_NAME).write_text(render_visual_focus_card(summary),
                                     encoding="utf-8")
        summary["summary_path"] = str(out / SUMMARY_NAME)
        summary["card_path"] = str(out / CARD_NAME)
    except Exception as exc:
        warnings.append(f"could not write summary/card: {exc}")
    return summary


def load_visual_focus_results(results_path) -> list:
    p = Path(results_path)
    if p.is_dir():
        p = p / RESULTS_NAME
    out: list = []
    if not p.exists():
        return out
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    return out


def _dist(results, getter) -> dict:
    d: dict = {}
    for r in results:
        try:
            k = getter(r)
        except Exception:
            k = "unknown"
        d[k] = d.get(k, 0) + 1
    return dict(sorted(d.items(), key=lambda kv: -kv[1]))


def summarize_visual_focus_results(results_path) -> dict:
    results = (results_path if isinstance(results_path, list)
               else load_visual_focus_results(results_path))
    total = len(results)
    scores = [r.get("training_suitability", {}).get("score", 0) for r in results]
    target_counts: dict = {}
    for r in results:
        for t in r.get("training_suitability", {}).get("recommended_targets", []):
            target_counts[t] = target_counts.get(t, 0) + 1
    flag_counts: dict = {}
    for r in results:
        for f in r.get("flags", []):
            flag_counts[f] = flag_counts.get(f, 0) + 1
    return {
        "classifier_version": vfc.CLASSIFIER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "mean_suitability": round(sum(scores) / total, 2) if total else 0.0,
        "primary_focus_distribution": _dist(
            results, lambda r: r["primary_focus"]["label"]),
        "detail_tier_distribution": _dist(
            results, lambda r: r["detail_tier"]["label"]),
        "pose_distribution": _dist(
            results, lambda r: r["pose_expression"]["label"]),
        "suitability_tier_distribution": _dist(
            results, lambda r: r["training_suitability"]["tier"]),
        "recommended_target_distribution": dict(
            sorted(target_counts.items(), key=lambda kv: -kv[1])),
        "flag_distribution": dict(sorted(flag_counts.items(),
                                         key=lambda kv: -kv[1])),
        "warnings": [],
    }


def filter_visual_focus_results(results, focus=None, detail_tier=None, pose=None,
                                target=None, min_score=None) -> list:
    out = []
    for r in results:
        if focus and r.get("primary_focus", {}).get("label") != focus:
            continue
        if detail_tier and r.get("detail_tier", {}).get("label") != detail_tier:
            continue
        if pose and r.get("pose_expression", {}).get("label") != pose:
            continue
        ts = r.get("training_suitability", {})
        if target and target not in ts.get("recommended_targets", []):
            continue
        if min_score is not None and ts.get("score", 0) < float(min_score):
            continue
        out.append(r)
    return out


def render_visual_focus_card(summary: dict) -> str:
    lines = ["# Visual Focus Classification Card", "",
             f"Classifier: `{summary.get('classifier_version')}`",
             f"Images analyzed: **{summary.get('total', 0)}**",
             f"Mean training suitability: **{summary.get('mean_suitability', 0)}**",
             ""]

    def section(title, dist):
        lines.append(f"## {title}")
        if not dist:
            lines.append("_no data_\n")
            return
        for k, v in dist.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    section("Primary focus", summary.get("primary_focus_distribution", {}))
    section("Detail tier", summary.get("detail_tier_distribution", {}))
    section("Pose expression", summary.get("pose_distribution", {}))
    section("Suitability tiers", summary.get("suitability_tier_distribution", {}))
    section("Recommended targets", summary.get("recommended_target_distribution", {}))
    section("Flags", summary.get("flag_distribution", {}))
    for w in summary.get("warnings", []):
        lines.append(f"> ⚠ {w}")
    return "\n".join(lines)


def results_by_filename(results_path) -> dict:
    """Map basename -> compact fields for Dataset Explorer merge."""
    out: dict = {}
    for r in load_visual_focus_results(results_path):
        name = Path(r.get("image_path", "")).name
        if not name:
            continue
        ts = r.get("training_suitability", {})
        out[name] = {
            "primary_focus": r.get("primary_focus", {}).get("label"),
            "detail_tier": r.get("detail_tier", {}).get("label"),
            "pose_expression": r.get("pose_expression", {}).get("label"),
            "visual_focus_score": ts.get("score"),
            "suitability_tier": ts.get("tier"),
            "recommended_targets": ts.get("recommended_targets", []),
            "visual_focus_flags": r.get("flags", []),
        }
    return out


__all__ = ["classify_dataset_visual_focus", "summarize_visual_focus_results",
           "load_visual_focus_results", "filter_visual_focus_results",
           "render_visual_focus_card", "results_by_filename", "RESULTS_NAME",
           "SUMMARY_NAME", "CARD_NAME"]
