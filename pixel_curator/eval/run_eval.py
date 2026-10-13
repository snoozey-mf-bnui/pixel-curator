"""Phase 3: post-training evaluation matrix for a finished LoRA run.

Produces a structured evaluation report from everything the studio
already knows about a run - run metadata, dataset signals, Phase 1
quality/duplicate metrics, stored eval-matrix scores (likeness /
flexibility, ArcFace-based when the [ai] extras are installed) - plus a
flexibility test grid (payload cells; sample generation stays optional
through the existing matrix pipeline).

Everything works on CPU with zero optional dependencies: missing
ArcFace/InsightFace, missing generated samples and missing projections
all degrade to clear professional notes.  The report is written to
  runs/<run_name>/evaluation/eval_report.json
  runs/<run_name>/evaluation/eval_summary.md
and stored in the manifest meta (`evaluation:<run_name>`).
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import manifest
from ..config import Project
from ..insights import aesthetic_proxy, frame_quality
from ..util import now_iso

# Flexibility grid dimensions (professional probe set)
GRID_DIMENSIONS = [
    ("neutral_identity_reference",
     "{t}, neutral portrait, clean background, natural lighting"),
    ("portrait_consistency",
     "{t}, studio portrait, soft lighting, looking at viewer"),
    ("half_body_composition",
     "{t}, upper body, three-quarter view, studio composition"),
    ("full_body_form_study",
     "{t}, full body framing, balanced standing pose, form and "
     "proportion clarity"),
    ("garment_rendering_study",
     "{t}, editorial fashion study, garment structure visible, "
     "textile detail"),
    ("studio_lighting_variation",
     "{t}, dramatic rim lighting, sculptural shadow, fine art "
     "photography"),
    ("editorial_pose_variation",
     "{t}, expressive editorial pose, dynamic composition"),
    ("closeup_facial_consistency",
     "{t}, close-up face, natural expression, detailed skin"),
    ("style_neutral_test",
     "{t}, candid photo, outdoor natural light"),
    ("concept_stack_test",
     "{t}, fine art figure study, chiaroscuro lighting, detailed "
     "anatomy, realistic rendering"),
]

_RISK_LABELS = ((0.66, "high"), (0.33, "moderate"), (0.0, "low"))


def _run_row(conn, run_name: str) -> dict | None:
    row = conn.execute(
        "SELECT run_id, name, dataset_version, preset, status, "
        "current_step, total_steps, current_epoch, total_epochs, "
        "last_loss, command, started_at, finished_at "
        "FROM runs WHERE name=? ORDER BY run_id DESC LIMIT 1",
        (run_name,)).fetchone()
    if not row:
        return None
    keys = ("run_id", "name", "dataset_version", "preset", "status",
            "step", "total_steps", "epoch", "total_epochs", "last_loss",
            "command", "started_at", "finished_at")
    return dict(zip(keys, row))


def _eval_scores(conn, run_name: str) -> dict[str, dict]:
    """Aggregated eval-matrix scores per category for this run's labels."""
    out: dict[str, dict] = {}
    for cat, n, avg in conn.execute(
            "SELECT category, COUNT(*), AVG(likeness) FROM evals "
            "WHERE label LIKE ? AND likeness IS NOT NULL "
            "GROUP BY category", (f"{run_name}%",)):
        out[cat] = {"avg": round(avg, 3), "n": n}
    return out


def _arcface_available() -> tuple[bool, str]:
    try:
        import importlib.util
        ok = (importlib.util.find_spec("insightface") is not None
              and importlib.util.find_spec("onnxruntime") is not None)
        return ok, ("insightface + onnxruntime present" if ok
                    else "insightface/onnxruntime not installed")
    except Exception as exc:
        return False, str(exc)


def _strength_label(score: float | None) -> str:
    if score is None:
        return "untested"
    if score >= 0.75:
        return "strong"
    if score >= 0.5:
        return "moderate"
    return "weak"


def _dataset_rows(conn, version: int) -> list[dict]:
    rows = conn.execute(
        "SELECT f.sharpness, f.brightness, d.framing, d.identity_sim, "
        "d.face_area, f.dup_role, f.cluster_id "
        "FROM frames f JOIN dataset_frames df ON df.frame_id=f.frame_id "
        "AND df.version=? "
        "LEFT JOIN detections d ON d.frame_id=f.frame_id", (version,)
    ).fetchall()
    keys = ("sharpness", "brightness", "framing", "identity_sim",
            "face_area", "dup_role", "cluster_id")
    return [dict(zip(keys, r)) for r in rows]


def build_flexibility_grid(prj: Project, run_name: str,
                           weight: float = 0.85,
                           seeds: tuple = (42, 1042)) -> list[dict]:
    """Payload cells for the flexibility test grid (generation optional -
    these are the exact prompts/seeds the matrix pipeline would render)."""
    trig = f"{prj.trigger_token} {prj.class_word}".strip() or "subject"
    cells = []
    for dim, template in GRID_DIMENSIONS:
        for seed in seeds:
            cells.append({
                "dimension": dim,
                "prompt": ("score_9, score_8_up, score_7_up, "
                           + template.format(t=trig)),
                "lora": run_name, "weight": weight, "seed": seed,
                "concept_stack": [{"name": run_name, "weight": weight}],
                "sample_path": None, "score": None, "warnings": [],
            })
    return cells


def _overfit(signals: dict, dup_density: float, cluster_count: int,
             steps_per_image: float | None,
             likeness: float | None,
             flexibility: float | None) -> dict:
    score = 0.0
    codes: list[str] = []
    recs: list[str] = []
    frames = int(signals.get("selected_frames") or 0)

    if dup_density > 0.25:
        score += 0.30
        codes.append("duplicate_density_high")
        recs.append("Reduce near-duplicate frames (duplicate review "
                    "candidates) before retraining.")
    elif dup_density > 0.10:
        score += 0.15
        codes.append("duplicate_density_moderate")
    if cluster_count and cluster_count < 8:
        score += 0.20
        codes.append("cluster_diversity_low")
        recs.append("Add more pose and lighting diversity before "
                    "retraining.")
    if frames and frames < 150:
        score += 0.20
        codes.append("dataset_small")
        recs.append("Grow the curated dataset toward 250-400 frames for "
                    "a more flexible identity anchor.")
    if steps_per_image is not None and steps_per_image > 40:
        score += 0.20
        codes.append("training_steps_high_for_dataset")
        recs.append("Reduce training epochs or repeats relative to the "
                    "dataset size for the next run.")
    if likeness is not None and flexibility is not None \
            and likeness - flexibility > 0.15:
        score += 0.25
        codes.append("likeness_flexibility_gap_high")
        recs.append("The likeness-flexibility gap exceeds the overfit "
                    "guard - prefer an earlier epoch checkpoint "
                    "(best-epoch sweep).")
    score = round(min(1.0, score), 2)
    risk = next(label for floor, label in _RISK_LABELS if score >= floor)
    if not recs:
        recs.append("No overfit indicators detected - proceed with the "
                    "standard best-epoch sweep validation.")
    return {"risk": risk, "score": score, "reason_codes": codes,
            "recommendations": recs}


def evaluate_run(conn, prj: Project, output_base: Path, run_name: str,
                 write: bool = True) -> dict:
    """Full post-training evaluation report for a run (CPU-only; uses
    stored eval-matrix scores when present, never requires them)."""
    run = _run_row(conn, run_name)
    if run is None:
        raise KeyError(f"run '{run_name}' not found - see `pixel-curator "
                       "runs` for available run names")
    warnings: list[str] = []

    # dataset signals (graceful when the version is missing)
    signals: dict = {}
    rows: list[dict] = []
    try:
        from ..train.preset_recommender import collect_signals
        signals = collect_signals(conn, run["dataset_version"])
        rows = _dataset_rows(conn, run["dataset_version"])
    except Exception as exc:
        warnings.append(f"dataset signals unavailable ({exc})")

    dup_density = (round(sum(1 for r in rows
                             if r.get("dup_role") == "candidate")
                         / len(rows), 3) if rows else 0.0)
    clusters = len({r["cluster_id"] for r in rows
                    if r.get("cluster_id") is not None})
    frames_n = int(signals.get("selected_frames") or len(rows) or 0)
    steps_per_image = (round(run["total_steps"] / frames_n, 1)
                       if run["total_steps"] and frames_n else None)

    # ----- identity consistency -----
    scores = _eval_scores(conn, run_name)
    likeness = scores.get("likeness", {}).get("avg")
    arc_ok, arc_reason = _arcface_available()
    if likeness is not None:
        identity = {"available": True, "method": "anchor_likeness_arcface",
                    "score": likeness, "warnings": [],
                    "status": _strength_label(likeness)}
    elif arc_ok:
        identity = {"available": False, "method": None, "score": None,
                    "status": "untested",
                    "warnings": ["No eval-matrix likeness rows for this "
                                 "run yet - run `pixel-curator sweep --run "
                                 f"{run_name}` or `matrix` with Forge to "
                                 "score identity consistency."]}
    else:
        identity = {"available": False, "method": None, "score": None,
                    "status": "untested",
                    "warnings": ["InsightFace/ArcFace unavailable; "
                                 "identity scoring skipped "
                                 f"({arc_reason})."]}

    # ----- flexibility -----
    flexibility_score = scores.get("flexibility", {}).get("avg")
    grid = build_flexibility_grid(prj, run_name)
    per_dim = {}
    for cat in ("likeness", "flexibility", "pose", "outfit", "style",
                "fine_art"):
        if cat in scores:
            per_dim[cat] = {"avg": scores[cat]["avg"],
                            "n": scores[cat]["n"],
                            "status": _strength_label(scores[cat]["avg"])}
    flexibility = {
        "score": flexibility_score,
        "status": _strength_label(flexibility_score),
        "tested_prompts": sum(s["n"] for s in scores.values()) or 0,
        "grid_cells": len(grid),
        "per_category": per_dim,
    }
    if not scores:
        warnings.append("No generated eval samples scored yet - the "
                        "flexibility grid below is a ready-to-render "
                        "payload (use --generate-samples with Forge "
                        "running, or the best-epoch sweep).")

    # ----- overfit -----
    overfit = _overfit(signals, dup_density, clusters, steps_per_image,
                       likeness, flexibility_score)

    # ----- artistic quality -----
    if rows:
        aes = [aesthetic_proxy(r) for r in rows]
        warn_ratio = (sum(1 for r in rows if frame_quality(r)["warnings"])
                      / len(rows))
        art_score = round(max(0.05, sum(aes) / len(aes)
                              - 0.2 * warn_ratio), 3)
        art_conf = 0.75 if scores else 0.5
    else:
        art_score, art_conf = None, 0.0
    artistic_quality = {"score": art_score, "confidence": art_conf,
                        "basis": "dataset aesthetic profile"
                        + (" + scored eval samples" if scores else
                           " (heuristic - no scored samples yet)")}

    # ----- recommended stack -----
    recommended_stack = None
    try:
        from ..artistic.stack_recommender import recommend_best_stack
        recommended_stack = recommend_best_stack(
            conn, prj, run["dataset_version"])
    except Exception as exc:
        warnings.append(f"stack recommendation unavailable ({exc})")

    # ----- next actions -----
    actions = []
    if identity["score"] is not None and identity["score"] >= 0.75:
        actions.append("Keep this LoRA as the identity anchor.")
    elif identity["score"] is None:
        actions.append("Score identity consistency via the best-epoch "
                       "sweep before promoting this run.")
    actions.append("Use concept weights conservatively (0.30-0.50) when "
                   "stacking over this anchor.")
    actions += overfit["recommendations"][:2]

    report = {
        "run": run_name,
        "run_id": run["run_id"],
        "dataset_version": run["dataset_version"],
        "preset": run["preset"],
        "status": run["status"],
        "training": {"steps": run["step"],
                     "total_steps": run["total_steps"],
                     "epochs": run["total_epochs"],
                     "final_loss": run["last_loss"],
                     "steps_per_image": steps_per_image},
        "identity_consistency": identity,
        "flexibility": flexibility,
        "flexibility_grid": grid,
        "overfit": overfit,
        "artistic_quality": artistic_quality,
        "dataset_signals": {
            "selected_frames": frames_n,
            "duplicate_density": dup_density,
            "cluster_count": clusters,
            "identity_strength": signals.get("identity_strength"),
            "quality_score": signals.get("quality_score"),
        },
        "recommended_stack": recommended_stack,
        "next_actions": actions,
        "warnings": warnings,
        "evaluated_at": now_iso(),
    }

    if write:
        eval_dir = Path(output_base) / "runs" / run_name / "evaluation"
        try:
            eval_dir.mkdir(parents=True, exist_ok=True)
            (eval_dir / "eval_report.json").write_text(
                json.dumps(report, indent=2))
            (eval_dir / "eval_summary.md").write_text(eval_summary_md(report))
            report["report_path"] = str(eval_dir / "eval_report.json")
        except Exception as exc:
            report["warnings"].append(f"report write failed ({exc})")
        manifest.meta_set(conn, f"evaluation:{run_name}", {
            k: report[k] for k in
            ("identity_consistency", "flexibility", "overfit",
             "artistic_quality", "next_actions", "evaluated_at")})
    return report


def eval_summary_md(r: dict) -> str:
    idc = r["identity_consistency"]
    fx = r["flexibility"]
    of = r["overfit"]
    aq = r["artistic_quality"]
    lines = [
        f"# Evaluation Summary: {r['run']}",
        "",
        f"- Dataset version: v{r['dataset_version']:03d} | preset "
        f"`{r['preset']}` | status {r['status']}",
        f"- Training: {r['training']['steps']}/"
        f"{r['training']['total_steps']} steps, "
        f"{r['training']['epochs']} epochs, final loss "
        f"{r['training']['final_loss']}",
        "",
        "## Identity Consistency",
        f"- Score: {idc['score'] if idc['score'] is not None else 'n/a'} "
        f"({idc['status']})",
        f"- Method: {idc['method'] or 'n/a'}",
        "",
        "## Flexibility",
        f"- Score: {fx['score'] if fx['score'] is not None else 'n/a'} "
        f"({fx['status']}) | scored prompts: {fx['tested_prompts']} | "
        f"grid cells ready: {fx['grid_cells']}",
    ]
    for cat, s in fx["per_category"].items():
        lines.append(f"- {cat}: {s['avg']} ({s['status']}, n={s['n']})")
    lines += [
        "",
        "## Overfit Risk",
        f"- Risk: **{of['risk']}** (score {of['score']})",
    ]
    for c in of["reason_codes"]:
        lines.append(f"- {c}")
    lines += [
        "",
        "## Artistic Quality",
        f"- Score: {aq['score'] if aq['score'] is not None else 'n/a'} "
        f"(confidence {aq['confidence']})",
        f"- Basis: {aq['basis']}",
        "",
        "## Recommended Next Actions",
    ]
    for a in r["next_actions"]:
        lines.append(f"- {a}")
    for w in r["warnings"]:
        lines.append(f"\n> NOTE: {w}")
    return "\n".join(lines)


def render_eval(r: dict) -> str:
    idc = r["identity_consistency"]
    fx = r["flexibility"]
    of = r["overfit"]
    lines = [
        f"Evaluation: {r['run']}",
        "",
        "Identity Consistency:",
        f"- Score: {idc['score'] if idc['score'] is not None else 'n/a'}",
        f"- Method: {idc['method'] or 'n/a'}",
        f"- Status: {idc['status']}",
        "",
        "Flexibility:",
        f"- Overall: {fx['score'] if fx['score'] is not None else 'n/a'} "
        f"({fx['status']}) | grid cells ready: {fx['grid_cells']}",
    ]
    for cat, s in fx["per_category"].items():
        lines.append(f"- {cat}: {s['avg']} ({s['status']})")
    lines += ["", "Overfit Risk:", f"- {of['risk'].capitalize()} "
              f"(score {of['score']})"]
    for rec in of["recommendations"]:
        lines.append(f"- {rec}")
    aq = r["artistic_quality"]
    lines += ["", "Artistic Quality:",
              f"- Score: {aq['score'] if aq['score'] is not None else 'n/a'}"
              f" (confidence {aq['confidence']})"]
    lines += ["", "Recommendation:"]
    for a in r["next_actions"]:
        lines.append(f"- {a}")
    if r.get("recommended_stack"):
        rs = r["recommended_stack"]["recommended_stack"]
        lines.append("- Recommended stack: " + " + ".join(
            f"{x['name']}@{x['weight']:.2f}" for x in rs))
    for w in r["warnings"]:
        lines.append(f"NOTE: {w}")
    if r.get("report_path"):
        lines.append(f"\nReport: {r['report_path']}")
    return "\n".join(lines)
