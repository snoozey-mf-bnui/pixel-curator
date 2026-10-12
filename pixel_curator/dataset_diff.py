"""Phase 2: dataset version visual diff (additive).

Builds on the membership diff the builder already provides
(added / removed / caption / split changes keyed by stable
(frame_id, concept) identifiers) and layers on:

* quality cohort deltas - median sharpness / brightness / identity
  confidence and warning counts compared across the two versions'
  member frames (current metrics, cohort vs cohort)
* duplicate candidate deltas (Phase 1 dup_* fields)
* framing distribution shifts
* cluster shifts - when both versions have a cached viz projection,
  common frames whose cluster assignment moved

Read-only over the manifest; never modifies datasets.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .builder import diff_datasets
from .insights import _median, frame_quality
from .viz.payload import viz_cache_dir


def _cohort(conn, version: int) -> list[dict]:
    rows = conn.execute(
        "SELECT DISTINCT f.frame_id, f.status, f.sharpness, f.brightness, "
        "d.framing, d.identity_sim, d.face_area, f.dup_role "
        "FROM frames f JOIN dataset_frames df ON df.frame_id = f.frame_id "
        "AND df.version = ? "
        "LEFT JOIN detections d ON d.frame_id = f.frame_id",
        (version,)).fetchall()
    keys = ("frame_id", "status", "sharpness", "brightness", "framing",
            "identity_sim", "face_area", "dup_role")
    return [dict(zip(keys, r)) for r in rows]


def _cohort_stats(rows: list[dict]) -> dict:
    warn: Counter = Counter()
    for r in rows:
        warn.update(frame_quality(r)["warnings"])
    framing = Counter(r["framing"] or "unscored" for r in rows)
    total = max(1, len(rows))
    return {
        "count": len(rows),
        "median_sharpness": round(_median(
            [r["sharpness"] for r in rows]), 1),
        "median_brightness": round(_median(
            [r["brightness"] for r in rows]), 1),
        "median_identity_confidence": round(_median(
            [r["identity_sim"] for r in rows
             if r["identity_sim"] is not None]), 3) if any(
            r["identity_sim"] is not None for r in rows) else None,
        "warning_counts": dict(warn),
        "duplicate_candidates": sum(
            1 for r in rows if r["dup_role"] == "candidate"),
        "framing_distribution": {
            k: round(v / total, 3) for k, v in framing.most_common()},
    }


def _cluster_assignments(output_base: Path, version: int) -> dict[str, str]:
    """frame_id -> cluster_id from any cached projection for a version."""
    d = viz_cache_dir(output_base, version)
    if not d.exists():
        return {}
    for f in sorted(d.glob("projection_*.json")):
        try:
            payload = json.loads(f.read_text())
            return {i["item_id"]: i["cluster_id"]
                    for i in payload.get("items", [])}
        except Exception:
            continue
    return {}


def dataset_version_diff(conn, output_base: Path, dataset: int,
                         vs: int) -> dict:
    """Structured diff of dataset `dataset` against baseline `vs`."""
    base = diff_datasets(conn, vs, dataset)     # b = newer version
    warnings: list[str] = []

    new_stats = _cohort_stats(_cohort(conn, dataset))
    old_stats = _cohort_stats(_cohort(conn, vs))

    def delta(key):
        a, b = old_stats.get(key), new_stats.get(key)
        return (round(b - a, 3) if isinstance(a, (int, float))
                and isinstance(b, (int, float)) else None)

    warn_delta = {
        k: new_stats["warning_counts"].get(k, 0)
        - old_stats["warning_counts"].get(k, 0)
        for k in set(new_stats["warning_counts"])
        | set(old_stats["warning_counts"])}
    framing_delta = {
        k: round(new_stats["framing_distribution"].get(k, 0)
                 - old_stats["framing_distribution"].get(k, 0), 3)
        for k in set(new_stats["framing_distribution"])
        | set(old_stats["framing_distribution"])}

    # cluster shifts (needs cached projections on both sides)
    cl_new = _cluster_assignments(output_base, dataset)
    cl_old = _cluster_assignments(output_base, vs)
    cluster_shifted: list[dict] = []
    if cl_new and cl_old:
        common = set(cl_new) & set(cl_old)
        cluster_shifted = [
            {"frame_id": fid, "from": cl_old[fid], "to": cl_new[fid]}
            for fid in sorted(common) if cl_old[fid] != cl_new[fid]]
    else:
        warnings.append(
            "Cluster diff unavailable: projection data not found for both "
            f"versions. Run `pixel-curator viz --dataset {dataset}` and "
            f"`pixel-curator viz --dataset {vs}` first.")

    quality_regressions = sum(1 for k, v in warn_delta.items() if v > 0)
    decision_changed = len(base["caption_changed"]) + len(
        base["split_changed"])

    return {
        "dataset": dataset, "vs": vs,
        "summary": {
            "added": len(base["added"]),
            "removed": len(base["removed"]),
            "unchanged": max(0, old_stats["count"] - len(base["removed"])
                             - decision_changed),
            "decision_changed": decision_changed,
            "quality_changed": sum(abs(v) for v in warn_delta.values()),
            "cluster_shifted": len(cluster_shifted),
        },
        "added_items": base["added"],
        "removed_items": base["removed"],
        "changed_items": base["caption_changed"] + base["split_changed"],
        "quality_delta": {
            "median_sharpness": delta("median_sharpness"),
            "median_brightness": delta("median_brightness"),
            "median_identity_confidence": delta(
                "median_identity_confidence"),
            "warning_counts": warn_delta,
            "duplicate_candidates": delta("duplicate_candidates"),
            "framing_distribution": framing_delta,
        },
        "cluster_delta": {
            "shifted_items": cluster_shifted[:200],
            "shifted_count": len(cluster_shifted),
        },
        "cohorts": {"new": new_stats, "baseline": old_stats},
        "concept_mix": {"baseline": base["concept_mix_a"],
                        "new": base["concept_mix_b"]},
        "quality_regressions": quality_regressions,
        "warnings": warnings,
    }


def render_dataset_diff(d: dict) -> str:
    s = d["summary"]
    q = d["quality_delta"]

    def signed(v, suffix=""):
        return "n/a" if v is None else f"{v:+g}{suffix}"

    lines = [
        f"Dataset Diff: {d['dataset']} vs {d['vs']}",
        "",
        "Summary:",
        f"- Added frames: {s['added']}",
        f"- Removed frames: {s['removed']}",
        f"- Unchanged frames: {s['unchanged']}",
        f"- Decision changes (caption/split): {s['decision_changed']}",
        f"- Quality warning changes: {s['quality_changed']}",
        f"- Cluster shifts: {s['cluster_shifted']}",
        "",
        "Quality Changes (new cohort vs baseline):",
        f"- Median sharpness: {signed(q['median_sharpness'])}",
        f"- Median identity confidence: "
        f"{signed(q['median_identity_confidence'])}",
        f"- Duplicate candidates: {signed(q['duplicate_candidates'])}",
    ]
    for k, v in sorted(q["warning_counts"].items()):
        if v:
            lines.append(f"- {k} warnings: {v:+d}")
    moved = [f"{k} {v:+.0%}" for k, v in
             sorted(q["framing_distribution"].items()) if abs(v) >= 0.01]
    if moved:
        lines.append("- Framing shift: " + ", ".join(moved))
    lines += [
        "",
        "Recommended Review:",
        f"- {s['decision_changed']} decision changes",
        f"- {s['cluster_shifted']} cluster-shifted frames",
        f"- {d['quality_regressions']} quality regressions",
    ]
    for w in d["warnings"]:
        lines.append(f"NOTE: {w}")
    return "\n".join(lines)
