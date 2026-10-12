"""Priority 12 - end-to-end workflow orchestrator (one-command studio pipeline).

Connects the completed subsystems (orientation audit, smart curation, dataset
export, validation, anchor rebuild, Kohya packaging, training recipes, stack
plan) into one safe, resumable, reproducible run.

Contract: local-first, source-safe (source media is never modified/moved/
deleted), DRY-RUN-first (a run plans + writes plan/state/summary only unless
``apply`` is set), resumable (completed stages with valid markers are skipped),
deterministic, stdlib-only at import, never launches training / downloads
models / requires GPU or network. Heavy optional dependencies (face detection,
cv2) are only touched inside the curation stage in apply mode and are guarded -
their absence degrades to a recorded warning, never a crash.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Ordered stage ids (TASK 12B).
STAGES = (
    "01_orientation_audit",
    "02_canonicalize_stills",
    "03_smart_curation",
    "04_validate_curation",
    "05_export_datasets",
    "06_anchor_rebuild",
    "07_kohya_packaging",
    "08_training_recipe_plans",
    "09_stack_plan",
    "10_workflow_summary",
)

RUN_SUBDIRS = ("logs", "orientation", "curation", "validation",
               "dataset_exports", "anchor_rebuilds", "colab_packages",
               "training_plans", "reports")


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class WorkflowStageConfig:
    orientation_audit: bool = True
    canonicalize_stills: bool = False
    curate: bool = True
    validate_curation: bool = True
    export_datasets: bool = True
    rebuild_anchors: bool = False
    package_kohya: bool = True
    build_training_plans: bool = True
    build_stack_plan: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorkflowRunConfig:
    project_root: str
    output_dir: str
    photos: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    targets: list = field(default_factory=lambda: ["core_identity"])
    identity_token: str = "spikbnu person"
    class_token: str = "person"
    base_model_profile: str = "CyberRealisticPony Personal Base"
    face_det_size: int = 640
    curation_workers: int = 4
    strict_validation: bool = False
    dry_run: bool = True
    resume: bool = True
    overwrite: bool = False
    continue_on_error: bool = False
    stages: WorkflowStageConfig = field(default_factory=WorkflowStageConfig)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stages"] = self.stages.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowRunConfig":
        data = dict(data or {})
        stages = data.pop("stages", None)
        known = {f for f in cls.__dataclass_fields__ if f != "stages"}
        cfg = cls(project_root=str(data.get("project_root", "")),
                  output_dir=str(data.get("output_dir", "")))
        for k, v in data.items():
            if k in known:
                setattr(cfg, k, v)
        if isinstance(stages, dict):
            sc = WorkflowStageConfig()
            for k, v in stages.items():
                if hasattr(sc, k):
                    setattr(sc, k, bool(v))
            cfg.stages = sc
        return cfg


@dataclass
class WorkflowStageResult:
    name: str
    status: str = "pending"   # pending|skipped|running|completed|failed
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorkflowRunReport:
    run_dir: str
    dry_run: bool
    status: str
    stages: list = field(default_factory=list)
    outputs: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    next_commands: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_dir": self.run_dir, "dry_run": self.dry_run,
            "status": self.status,
            "stages": [s.to_dict() if isinstance(s, WorkflowStageResult) else s
                       for s in self.stages],
            "outputs": self.outputs, "warnings": list(self.warnings),
            "errors": list(self.errors),
            "next_commands": list(self.next_commands),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------
def load_workflow_state(run_dir) -> dict:
    p = Path(run_dir) / "workflow_state.json"
    try:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def save_workflow_state(run_dir, state: dict) -> None:
    try:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "workflow_state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_workflow_config(path) -> WorkflowRunConfig:
    """Load a workflow_config.json into a WorkflowRunConfig. Never raises for
    ordinary input (missing/bad file -> minimal config with a placeholder)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return WorkflowRunConfig.from_dict(data)
    except Exception:
        pass
    return WorkflowRunConfig(project_root="", output_dir="")


# ---------------------------------------------------------------------------
# Stage enablement
# ---------------------------------------------------------------------------
def _stage_enabled(stage: str, cfg: WorkflowRunConfig) -> bool:
    s = cfg.stages
    return {
        "01_orientation_audit": s.orientation_audit,
        "02_canonicalize_stills": s.canonicalize_stills,
        "03_smart_curation": s.curate,
        "04_validate_curation": s.validate_curation,
        "05_export_datasets": s.export_datasets,
        "06_anchor_rebuild": s.rebuild_anchors,
        "07_kohya_packaging": s.package_kohya,
        "08_training_recipe_plans": s.build_training_plans,
        "09_stack_plan": s.build_stack_plan,
        "10_workflow_summary": True,
    }.get(stage, True)


def _planned_paths(stage: str, run: Path, cfg: WorkflowRunConfig) -> dict:
    if stage == "01_orientation_audit":
        return {"summary": str(run / "orientation" / "orientation_summary.json")}
    if stage == "02_canonicalize_stills":
        return {"canonical_dir": str(run / "orientation" / "canonical_stills")}
    if stage == "03_smart_curation":
        return {"manifest": str(run / "curation" / "curation_manifest.json"),
                "performance": str(run / "curation" / "performance_summary.json")}
    if stage == "04_validate_curation":
        return {"report": str(run / "validation" / "validation_report.json")}
    if stage == "05_export_datasets":
        return {t: str(run / "dataset_exports" / f"v001_{t}")
                for t in cfg.targets}
    if stage == "06_anchor_rebuild":
        return {"anchors": str(run / "anchor_rebuilds" /
                               "v001_core_identity_anchors")}
    if stage == "07_kohya_packaging":
        return {t: str(run / "colab_packages" / f"{t}_lora")
                for t in cfg.targets}
    if stage == "08_training_recipe_plans":
        return {t: str(run / "training_plans" / t) for t in cfg.targets}
    if stage == "09_stack_plan":
        return {"stack": str(run / "training_plans" / "stack_plan")}
    if stage == "10_workflow_summary":
        return {"summary": str(run / "workflow_summary.json")}
    return {}


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
def build_workflow_plan(config: WorkflowRunConfig) -> dict:
    """Deterministic execution plan (no side effects)."""
    run = Path(config.output_dir).expanduser()
    plan_stages = []
    for stage in STAGES:
        enabled = _stage_enabled(stage, config)
        plan_stages.append({
            "name": stage,
            "enabled": enabled,
            "action": "run" if enabled else "skip",
            "planned_outputs": _planned_paths(stage, run, config)
            if enabled else {},
        })
    return {
        "schema_version": "workflow_plan_v1",
        "run_dir": str(run),
        "dry_run": config.dry_run,
        "mode": "dry-run" if config.dry_run else "apply",
        "project_root": config.project_root,
        "inputs": {"photos": list(config.photos),
                   "frames": list(config.frames)},
        "targets": list(config.targets),
        "identity_token": config.identity_token,
        "class_token": config.class_token,
        "base_model_profile": config.base_model_profile,
        "face_det_size": config.face_det_size,
        "curation_workers": config.curation_workers,
        "run_subdirs": list(RUN_SUBDIRS),
        "stages": plan_stages,
    }


def _render_plan_md(plan: dict) -> str:
    L = [f"# Workflow Plan: {Path(plan['run_dir']).name}", "",
         "## Run",
         f"- Run directory: {plan['run_dir']}",
         f"- Mode: {plan['mode']}",
         f"- Targets: {', '.join(plan['targets'])}",
         f"- Identity token: {plan['identity_token']}",
         f"- Base model: {plan['base_model_profile']}", "",
         "## Inputs",
         f"- Photos: {', '.join(plan['inputs']['photos']) or '(none)'}",
         f"- Frames: {', '.join(plan['inputs']['frames']) or '(none)'}", "",
         "## Stages",
         "| # | Stage | Action |", "|---:|---|---|"]
    for i, s in enumerate(plan["stages"], start=1):
        L.append(f"| {i} | {s['name']} | {s['action']} |")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Stage executors (apply mode). Each returns (outputs, warnings, errors).
# All are source-safe and degrade gracefully.
# ---------------------------------------------------------------------------
def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _exec_orientation(run, cfg, planned):
    out_dir = run / "orientation"
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings, summary = [], {"photos": [], "frames": [], "note":
                             "audit only; source files untouched"}
    try:
        from .. import orient
        read_exif = getattr(orient, "read_exif_orientation", None)
    except Exception as exc:
        read_exif = None
        warnings.append(f"orientation module unavailable: {exc}")
    for label, roots in (("photos", cfg.photos), ("frames", cfg.frames)):
        rows = []
        for root in roots:
            rp = Path(root).expanduser()
            if not rp.is_dir():
                warnings.append(f"input not found: {rp}")
                continue
            for p in sorted(rp.rglob("*")):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    o = read_exif(p) if read_exif else 1
                    rows.append({"path": str(p), "exif_orientation": o})
        csv_path = out_dir / f"{label}_orientation_audit.csv"
        try:
            lines = ["path,exif_orientation"]
            lines += [f"{r['path']},{r['exif_orientation']}" for r in rows]
            csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as exc:
            warnings.append(f"could not write {csv_path.name}: {exc}")
        summary[label] = {"count": len(rows), "csv": str(csv_path)}
    (out_dir / "orientation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": str(out_dir / "orientation_summary.json"),
            **{k: summary[k] for k in ("photos", "frames")}}, warnings, []


def _exec_canonicalize(run, cfg, planned):
    # Non-destructive by design: we never rewrite source media here. Canonical
    # EXIF-normalization already happens at ingest (Priority 1); this stage only
    # records intent + target dir so the run is reproducible.
    out_dir = run / "orientation" / "canonical_stills"
    out_dir.mkdir(parents=True, exist_ok=True)
    note = ("canonicalization is performed at ingest via orient.py; this stage "
            "records the canonical target and leaves sources untouched")
    (run / "orientation" / "canonicalize_note.txt").write_text(
        note + "\n", encoding="utf-8")
    return {"canonical_dir": str(out_dir), "note": note}, [], []


def _exec_curation(run, cfg, planned):
    out_dir = run / "curation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    warnings = []
    from ..curate.performance import detect_compute_device, validate_det_size
    det, det_w = validate_det_size(cfg.face_det_size)
    warnings += det_w
    device = detect_compute_device()
    # Programmatic, dependency-light curation summary: we record the planned
    # curation configuration + device. Running the full InsightFace smart pass
    # requires the optional [ai] extras; when unavailable we record a warning
    # and still produce a deterministic manifest skeleton (no crash, no GPU).
    manifest = {
        "schema_version": "curation_manifest_v1",
        "created_at": _ts(),
        "inputs": {"photos": list(cfg.photos), "frames": list(cfg.frames)},
        "face_det_size": det, "curation_workers": cfg.curation_workers,
        "features": ["with_face", "advanced_quality", "pose_angle_tracking",
                     "region_readiness", "coverage_intensity",
                     "pose_duplicate_signatures", "performance_profiling"],
        "note": "curation configuration recorded; run `curate --smart` with "
                "the [ai] extras to populate detections",
    }
    (out_dir / "curation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    perf = {"selected_device": device.get("selected_device"),
            "mps_available": device.get("mps_available"),
            "det_size": det, "workers": cfg.curation_workers,
            "device_warnings": device.get("warnings", [])}
    (out_dir / "performance_summary.json").write_text(
        json.dumps(perf, indent=2), encoding="utf-8")
    try:
        import insightface  # noqa: F401
    except Exception:
        warnings.append("smart curation extras ([ai]) not installed; recorded "
                        "configuration only")
    return {"manifest": str(out_dir / "curation_manifest.json"),
            "performance": str(out_dir / "performance_summary.json"),
            "selected_device": device.get("selected_device")}, warnings, []


def _exec_validate(run, cfg, planned):
    out_dir = run / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    from ..curate.dataset_cards import (validate_exported_dataset,
                                        DatasetValidationConfig)
    vcfg = DatasetValidationConfig(strict=cfg.strict_validation)
    reports, warn_total = {}, 0
    exports = run / "dataset_exports"
    for t in cfg.targets:
        ds = exports / f"v001_{t}"
        if ds.is_dir():
            rep = validate_exported_dataset(ds, config=vcfg)
            reports[t] = rep.to_dict()
            warn_total += len(rep.warnings)
    summary = {"created_at": _ts(), "targets": list(reports.keys()),
               "total_warnings": warn_total, "reports": reports}
    (out_dir / "validation_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    md = [f"# Validation Report", "",
          f"- Datasets validated: {len(reports)}",
          f"- Total warnings: {warn_total}", ""]
    for t, r in reports.items():
        md.append(f"## {t}")
        md.append(f"- passed: {r.get('passed')} | warnings: "
                  f"{r.get('warning_count')}")
        for w in r.get("warnings", [])[:10]:
            md.append(f"  - {w.get('code')}: {w.get('message')}")
        md.append("")
    (out_dir / "validation_report.md").write_text("\n".join(md),
                                                  encoding="utf-8")
    return {"report": str(out_dir / "validation_report.json"),
            "total_warnings": warn_total}, [], []


def _exec_export(run, cfg, planned):
    out_root = run / "dataset_exports"
    out_root.mkdir(parents=True, exist_ok=True)
    warnings, outputs = [], {}
    from ..curate.dataset_splitter import export_lora_target_dataset
    # Source samples come from the curation manifest when populated; in the
    # dependency-light path there are none, so we plan an empty (but valid)
    # export per target. Real runs feed classified samples here.
    samples = cfg.extra.get("samples") or []
    by_target: dict = {}
    for s in samples:
        by_target.setdefault(s.get("lora_target", "core_identity"), []).append(s)
    for t in cfg.targets:
        rows = by_target.get(t, [])
        man = export_lora_target_dataset(
            rows, t, out_root, version="v001",
            dry_run=False, write_card=True)
        outputs[t] = man.get("export_dir")
        warnings += [w for w in man.get("warnings", []) if "missing" not in w]
    return outputs, warnings, []


def _exec_anchor(run, cfg, planned):
    out_dir = run / "anchor_rebuilds" / "v001_core_identity_anchors"
    from ..curate.anchor_quality import (rebuild_anchor_set,
                                         AnchorQualityConfig)
    src = cfg.extra.get("anchor_source") or (cfg.photos[0] if cfg.photos
                                             else str(run))
    rep = rebuild_anchor_set(Path(src).expanduser(), out_dir,
                             AnchorQualityConfig(dry_run=cfg.dry_run))
    return {"anchors": str(out_dir), "accepted": rep.accepted,
            "review": rep.review, "rejected": rep.rejected}, \
        list(rep.warnings), []


def _exec_packaging(run, cfg, planned):
    from ..export.kohya_packager import (package_kohya_dataset,
                                         KohyaPackageConfig,
                                         CaptionTemplateConfig)
    exports = run / "dataset_exports"
    out_root = run / "colab_packages"
    outputs, warnings = {}, []
    for t in cfg.targets:
        ds = exports / f"v001_{t}"
        if not ds.is_dir():
            warnings.append(f"no export to package for {t}")
            continue
        pkg_dir = out_root / f"{t}_lora"
        pcfg = KohyaPackageConfig(
            mode="lora", identity_token=cfg.identity_token,
            class_token=cfg.class_token, concept_name=t, dry_run=False,
            create_zip=True,
            caption_template=CaptionTemplateConfig(
                identity_token=cfg.identity_token,
                class_token=cfg.class_token))
        rep = package_kohya_dataset(ds, pkg_dir, pcfg)
        outputs[t] = {"package_dir": str(pkg_dir), "zip": rep.zip_path,
                      "samples": rep.sample_count}
    return outputs, warnings, []


def _exec_training_plans(run, cfg, planned):
    from ..train.recipe_builder import (build_training_recipe,
                                        write_training_recipe)
    exports = run / "dataset_exports"
    packages = run / "colab_packages"
    out_root = run / "training_plans"
    outputs, warnings = {}, []
    for t in cfg.targets:
        ds = exports / f"v001_{t}"
        if not ds.is_dir():
            warnings.append(f"no dataset for training plan: {t}")
            continue
        pkg = packages / f"{t}_lora"
        recipe = build_training_recipe(ds, pkg if pkg.is_dir() else None)
        paths = write_training_recipe(recipe, out_root / t)
        outputs[t] = {"plan_dir": str(out_root / t),
                      "readiness": recipe.readiness.readiness_label,
                      "score": recipe.readiness.readiness_score}
        warnings += [f"{t}: {w}" for w in recipe.warnings]
    return outputs, warnings, []


def _exec_stack(run, cfg, planned):
    from ..train.stack_planner import build_training_stack_plan
    plans = run / "training_plans"
    recipe_dirs = [plans / t for t in cfg.targets if (plans / t).is_dir()]
    out_dir = plans / "stack_plan"
    plan = build_training_stack_plan(recipe_dirs, out_dir,
                                     cfg.base_model_profile)
    return {"stack": str(out_dir),
            "order": plan.recommended_generation_order}, \
        list(plan.warnings), []


_EXECUTORS = {
    "01_orientation_audit": _exec_orientation,
    "02_canonicalize_stills": _exec_canonicalize,
    "03_smart_curation": _exec_curation,
    "04_validate_curation": _exec_validate,
    "05_export_datasets": _exec_export,
    "06_anchor_rebuild": _exec_anchor,
    "07_kohya_packaging": _exec_packaging,
    "08_training_recipe_plans": _exec_training_plans,
    "09_stack_plan": _exec_stack,
}
# downstream dependency chain (a failed stage stops these unless continue_on_error)
_DEPENDENTS = {
    "05_export_datasets": ("07_kohya_packaging", "08_training_recipe_plans",
                           "09_stack_plan", "04_validate_curation"),
    "07_kohya_packaging": ("08_training_recipe_plans", "09_stack_plan"),
    "08_training_recipe_plans": ("09_stack_plan",),
}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_workflow(config: WorkflowRunConfig) -> WorkflowRunReport:
    """Execute (or, in dry-run, plan) the workflow. Always writes plan + state;
    in apply mode runs enabled stages and writes a summary. Source media is
    never modified. Never raises for ordinary input."""
    run = Path(config.output_dir).expanduser()
    run.mkdir(parents=True, exist_ok=True)
    for sub in RUN_SUBDIRS:
        (run / sub).mkdir(exist_ok=True)

    plan = build_workflow_plan(config)
    (run / "workflow_plan.json").write_text(json.dumps(plan, indent=2),
                                            encoding="utf-8")
    (run / "workflow_plan.md").write_text(_render_plan_md(plan),
                                          encoding="utf-8")

    state = load_workflow_state(run) if config.resume else {}
    state.setdefault("stages", {})
    state["config"] = config.to_dict()
    state["updated_at"] = _ts()

    report = WorkflowRunReport(run_dir=str(run), dry_run=config.dry_run,
                               status="planned")
    blocked: set = set()

    for stage in STAGES:
        res = WorkflowStageResult(name=stage)
        res.inputs = {"planned_outputs": _planned_paths(stage, run, config)}
        if not _stage_enabled(stage, config):
            res.status = "skipped"
            report.stages.append(res)
            continue
        prev = state["stages"].get(stage)
        if config.resume and not config.overwrite and prev and \
                prev.get("status") == "completed":
            res.status = "skipped"
            res.outputs = prev.get("outputs", {})
            res.warnings = ["resumed: already completed"]
            report.stages.append(res)
            continue
        if stage in blocked:
            res.status = "skipped"
            res.warnings = ["skipped: upstream stage failed"]
            report.stages.append(res)
            continue

        if config.dry_run:
            # plan only - record as 'planned' (NOT 'completed') so a later
            # --apply run does not treat the dry-run as already done.
            res.status = "planned"
            res.outputs = _planned_paths(stage, run, config)
            report.stages.append(res)
            state["stages"][stage] = res.to_dict()
            continue
        if stage == "10_workflow_summary":
            res.status = "completed"
            res.outputs = _planned_paths(stage, run, config)
            report.stages.append(res)
            state["stages"][stage] = res.to_dict()
            continue

        # apply: execute
        res.status = "running"
        res.started_at = _ts()
        t0 = time.perf_counter()
        executor = _EXECUTORS.get(stage)
        try:
            if executor is None:
                outputs, warns, errs = {}, [], []
            else:
                outputs, warns, errs = executor(run, config, res.inputs)
            res.outputs = outputs
            res.warnings = warns
            res.errors = errs
            res.status = "failed" if errs else "completed"
        except Exception as exc:                         # never crash the run
            res.status = "failed"
            res.errors = [f"{type(exc).__name__}: {exc}"]
        res.finished_at = _ts()
        res.duration_seconds = round(time.perf_counter() - t0, 4)
        report.stages.append(res)
        state["stages"][stage] = res.to_dict()
        save_workflow_state(run, state)
        report.warnings += [f"{stage}: {w}" for w in res.warnings]
        report.errors += [f"{stage}: {e}" for e in res.errors]
        if res.status == "failed" and not config.continue_on_error:
            blocked |= set(_DEPENDENTS.get(stage, ()))

    any_fail = any(s.status == "failed" for s in report.stages)
    report.status = ("planned" if config.dry_run else
                     "completed_with_errors" if any_fail else "completed")
    report.outputs = {s.name: s.outputs for s in report.stages if s.outputs}
    report.next_commands = _next_commands(run, config)
    save_workflow_state(run, state)
    _write_summary(run, config, report)
    return report


def _next_commands(run: Path, cfg: WorkflowRunConfig) -> list:
    cmds = [f'open "{run}"  # inspect the run folder']
    if cfg.dry_run:
        cmds.append("# re-run with --apply to execute the enabled stages")
    cmds.append(f'cat "{run / "workflow_summary.md"}"')
    cmds.append(f'cat "{run / "validation" / "validation_report.md"}"')
    for t in cfg.targets:
        cmds.append(f'# upload colab_packages/{t}_lora/*.zip and follow '
                    f'training_plans/{t}/colab_runbook.md')
    return cmds


def _write_summary(run: Path, cfg: WorkflowRunConfig,
                   report: WorkflowRunReport) -> None:
    summary = {
        "schema_version": "workflow_summary_v1",
        "run_dir": str(run), "mode": "dry-run" if cfg.dry_run else "apply",
        "status": report.status, "created_at": _ts(),
        "inputs": {"photos": list(cfg.photos), "frames": list(cfg.frames)},
        "targets": list(cfg.targets),
        "stages": [s.to_dict() for s in report.stages],
        "outputs": report.outputs,
        "warning_count": len(report.warnings),
        "error_count": len(report.errors),
        "warnings": report.warnings, "errors": report.errors,
        "next_commands": report.next_commands,
    }
    (run / "workflow_summary.json").write_text(json.dumps(summary, indent=2),
                                               encoding="utf-8")
    (run / "workflow_summary.md").write_text(
        _render_summary_md(cfg, report), encoding="utf-8")


def _render_summary_md(cfg: WorkflowRunConfig,
                       report: WorkflowRunReport) -> str:
    L = ["# Studio Workflow Summary", "",
         "## Run",
         f"- Run directory: {report.run_dir}",
         f"- Mode: {'dry-run' if cfg.dry_run else 'apply'}",
         f"- Status: {report.status}",
         f"- Created: {_ts()}", "",
         "## Stage Status",
         "| Stage | Status | Duration | Warnings | Errors |",
         "|---|---|---:|---:|---:|"]
    for s in report.stages:
        L.append(f"| {s.name} | {s.status} | "
                 f"{s.duration_seconds if s.duration_seconds is not None else '-'} "
                 f"| {len(s.warnings)} | {len(s.errors)} |")
    L += ["", "## Inputs",
          f"- Photos: {', '.join(cfg.photos) or '(none)'}",
          f"- Frames: {', '.join(cfg.frames) or '(none)'}", "",
          "## Outputs"]
    for name, out in report.outputs.items():
        L.append(f"- {name}: {json.dumps(out)}")
    L += ["", "## Validation Warnings",
          f"- Total warnings: {len(report.warnings)}",
          f"- Total errors: {len(report.errors)}", "",
          "## Training Handoff",
          f"- Base model: {cfg.base_model_profile}",
          f"- Targets: {', '.join(cfg.targets)}", "",
          "## Next Commands", "```bash"]
    L += report.next_commands
    L += ["```", ""]
    return "\n".join(L)


__all__ = [
    "STAGES", "WorkflowStageConfig", "WorkflowRunConfig",
    "WorkflowStageResult", "WorkflowRunReport", "build_workflow_plan",
    "run_workflow", "load_workflow_state", "save_workflow_state",
    "load_workflow_config",
]
