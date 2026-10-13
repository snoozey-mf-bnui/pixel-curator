"""Phase 7: declarative pipeline recipes - the orchestration layer that
ties prefiltering, curation, linting, intelligence, selection and export
into single-command workflows.

Sits NEXT TO the legacy fixed DAG (pipeline_dag.py, untouched): `pipeline
run full|mega` still runs the old engine; `pipeline run <recipe>` runs
this one. Recipes come from built-in templates or ``[pipelines.NAME]``
sections in the project TOML::

    [pipelines.nightly]
    description = "ingest -> curated, captioned, linted"
    stages = ["extract", "prefilter", "curate", "smart", "caption", "lint"]
    gates  = ["smart"]                  # pause AFTER these stages

    [pipelines.nightly.options.curate]
    min_sharpness = 40
    [pipelines.nightly.options.lint]
    fix = true

Engine guarantees:
  * every stage wraps an EXISTING generator/module - no duplicated logic
  * state checkpoints after EVERY stage (manifest meta, its own key) so an
    interrupt at any point resumes at the right stage
  * resume safety comes from the stages themselves (extract/curate/caption
    are incremental); re-running a stage never corrupts data
  * `pipeline plan <recipe>` shows the resolved plan and executes nothing
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path

from . import manifest
from .config import Project
from .util import now_iso, setup_logging

RECIPE_STATE_KEY = "pipeline_recipe_state"     # never collides with legacy
LEGACY_NAMES = ("full", "mega")                # dispatched to pipeline_dag


# ---------------------------------------------------------------------------
# Stage registry - each factory wraps an existing generator. Adding a stage
# is one new register() call; nothing else changes.
# ---------------------------------------------------------------------------

@dataclass
class StageSpec:
    name: str
    description: str
    factory: Callable  # (prj, base: Path, options: dict) -> Generator[str]
    options_doc: str = ""


STAGE_REGISTRY: dict[str, StageSpec] = {}


def register(name: str, description: str, options_doc: str = ""):
    def deco(fn):
        STAGE_REGISTRY[name] = StageSpec(name, description, fn, options_doc)
        return fn
    return deco


def _num_opts(options: dict, spec: dict) -> dict:
    """Coerce declared option keys to their types; ignore the rest."""
    out = {}
    for key, cast in spec.items():
        if key in options and options[key] is not None:
            try:
                out[key] = cast(options[key])
            except (TypeError, ValueError):
                pass
    return out


@register("extract", "ingest videos + photos (adaptive fps, EXIF-aware, "
          "zero-decode prefilter on photos)",
          "fps, no_photos, prefilter, limit_videos, dry_run")
def _stage_extract(prj: Project, base: Path, options: dict):
    from .config import ExtractConfig
    from .extract import pipeline_generator
    kw = _num_opts(options, {"fps": float, "limit_videos": int})
    ecfg = ExtractConfig(
        output_base=base,
        import_photos=not bool(options.get("no_photos", False)),
        prefilter=bool(options.get("prefilter", True)),
        dry_run=bool(options.get("dry_run", False)),
        **kw)
    yield from pipeline_generator(prj.video_dirs, prj.photos_dir, ecfg)


@register("photos", "import photos only (with the ingestion prefilter)",
          "prefilter, photo_mode")
def _stage_photos(prj: Project, base: Path, options: dict):
    from .config import ExtractConfig
    from .extract import pipeline_generator
    ecfg = ExtractConfig(
        output_base=base, import_photos=True,
        photo_import_mode=str(options.get("photo_mode", "hardlink")),
        prefilter=bool(options.get("prefilter", True)))
    yield from pipeline_generator([], prj.photos_dir, ecfg)


@register("prefilter", "fast junk rejection over 'new' frames (tiny files, "
          "low-res, heavy compression, truncated, caption bars)",
          "min_bytes, min_side, min_bpp, decode_checks, dry_run")
def _stage_prefilter(prj: Project, base: Path, options: dict):
    from .config import PrefilterConfig
    from .prefilter import prefilter_generator
    kw = _num_opts(options, {"min_bytes": int, "min_side": int,
                             "min_bpp": float, "extreme_blur": float})
    yield from prefilter_generator(PrefilterConfig(
        output_base=base,
        decode_checks=bool(options.get("decode_checks", True)),
        dry_run=bool(options.get("dry_run", False)), **kw))


@register("curate", "basic curation: score, blur/exposure filter, dedup "
          "(+ text-overlay check)",
          "min_sharpness, min_brightness, max_brightness, hamming, rescore")
def _stage_curate(prj: Project, base: Path, options: dict):
    from .config import CurateConfig
    from .curate import curate_generator
    kw = _num_opts(options, {"min_sharpness": float, "min_brightness": float,
                             "max_brightness": float, "workers": int})
    if "hamming" in options:
        kw["hamming_threshold"] = int(options["hamming"])
    yield from curate_generator(CurateConfig(
        output_base=base, rescore=bool(options.get("rescore", False)), **kw))


@register("smart", "face + identity filtering with orientation handling "
          "(needs [ai] extras)",
          "identity_threshold, min_face_area, rescan, det_size")
def _stage_smart(prj: Project, base: Path, options: dict):
    from .config import SmartCurateConfig
    from .curate import smart_curate_generator
    kw = _num_opts(options, {"identity_threshold": float,
                             "min_face_area": float, "det_size": int})
    yield from smart_curate_generator(SmartCurateConfig(
        output_base=base,
        anchor_dir=Path(prj.anchor_dir).expanduser() if prj.anchor_dir else None,
        rescan=bool(options.get("rescan", False)), **kw))


@register("cluster", "CLIP diversity clustering ([cluster] extras)",
          "k, reembed")
def _stage_cluster(prj: Project, base: Path, options: dict):
    from .config import ClusterConfig
    from .curate.diversity import cluster_generator
    kw = _num_opts(options, {"k": int, "batch_size": int})
    yield from cluster_generator(ClusterConfig(
        output_base=base, reembed=bool(options.get("reembed", False)), **kw))


@register("caption", "WD14 auto-captioning with the project trigger",
          "threshold, pony_prefix, prune, blacklist, max_tags, force")
def _stage_caption(prj: Project, base: Path, options: dict):
    from .caption import caption_generator
    from .config import CaptionConfig
    kw = _num_opts(options, {"threshold": float, "char_threshold": float,
                             "max_tags": int})
    yield from caption_generator(CaptionConfig(
        output_base=base, trigger=prj.trigger_token,
        class_word=prj.class_word,
        pony_prefix=bool(options.get("pony_prefix", False)),
        prune=str(options.get("prune", "")),
        blacklist=str(options.get("blacklist", "")),
        force=bool(options.get("force", False)), **kw))


@register("quality", "deep quality signals over curated frames "
          "(subject integrity / composition / artifacts)",
          "limit, resume")
def _stage_quality(prj: Project, base: Path, options: dict):
    from .curate.quality_signals import analyze_quality_pass, render_quality
    conn = manifest.connect(base)
    res = analyze_quality_pass(
        conn, limit=int(options.get("limit", 0) or 0),
        resume=bool(options.get("resume", True)))
    yield render_quality(res)


@register("lint", "active dataset lint (report; fix=true applies safe, "
          "journaled fixes)", "fix, only")
def _stage_lint(prj: Project, base: Path, options: dict):
    from . import lint as lint_mod
    conn = manifest.connect(base)
    issues = lint_mod.run_lint(conn)
    yield lint_mod.render_report(issues, fix_mode=bool(options.get("fix")))
    if options.get("fix"):
        only = options.get("only") or None
        r = lint_mod.apply_fixes(conn, issues, base,
                                 only=set(only) if only else None)
        yield (f"Applied {r['fixed']} fix(es)"
               + (f" | journal: {r['journal']}" if r.get("journal") else ""))


@register("intelligence", "holistic dataset analysis: usable estimate, "
          "strategy, training outlook", "sample")
def _stage_intelligence(prj: Project, base: Path, options: dict):
    from .intelligence import analyze_dataset, render_intelligence
    conn = manifest.connect(base, readonly=True)
    yield render_intelligence(analyze_dataset(
        conn, output_base=base, sample=int(options.get("sample", 0) or 0)))


@register("health", "dataset health score + reasons", "")
def _stage_health(prj: Project, base: Path, options: dict):
    from . import insights
    conn = manifest.connect(base, readonly=True)
    h = insights.dataset_health(conn)
    yield (f"Dataset health: {h['score']}/100 "
           f"({h.get('selected', 0)} curated frames)")
    for r in h["reasons"]:
        yield f"  - {r}"


@register("build", "versioned dataset build from a training recipe "
          "(prints parameter recommendations)", "recipe, note, parent")
def _stage_build(prj: Project, base: Path, options: dict):
    from .builder import BuildConfig, build_generator
    parent = options.get("parent")
    yield from build_generator(prj, BuildConfig(
        output_base=base,
        recipe=str(options.get("recipe", "character_v1")),
        note=str(options.get("note", f"pipeline run {now_iso()}")),
        parent=int(parent) if parent is not None else None))


@register("select", "configurable DreamBooth selection (profiles, quotas, "
          "explanations); apply=true exports + registers a version",
          "profile, target, output, apply, copy, explain, nsfw_mode")
def _stage_select(prj: Project, base: Path, options: dict):
    from .curate import dreambooth_nsfw_selector as sel
    # stage options act as the CLI layer over [dreambooth_selector]
    toml_section = dict(getattr(prj, "dreambooth_selector", {}) or {})
    stage_layer = {k: v for k, v in options.items()
                   if k not in ("output", "apply", "copy")}
    if options.get("copy"):
        stage_layer["link_mode"] = "copy"
    cfg, prov, warns = sel.resolve_selection_config(toml_section, stage_layer)
    for w in warns:
        yield f"config: {w}"
    yield sel.render_resolved_config(cfg, prov)
    conn = manifest.connect(base, readonly=True)
    try:
        cands = sel.load_candidates(conn)
        selected, exps, summary = sel.select_frames(cands, cfg)
    finally:
        conn.close()
    out = Path(options.get("output")
               or (base / "dreambooth_selection")).expanduser()
    sel.write_outputs(selected, exps, summary, cfg, out,
                      apply=bool(options.get("apply", False)),
                      manifest_base=base)
    yield (f"Selected {len(selected)} / target {cfg.target_count} "
           f"-> {out}")


@register("package", "kohya-style dataset packaging (repeats folders + "
          "captions)", "repeats, max_total, dataset_name")
def _stage_package(prj: Project, base: Path, options: dict):
    from .config import PackageConfig
    from .packager import package_generator
    kw = _num_opts(options, {"repeats": int, "max_per_video": int,
                             "max_total": int})
    yield from package_generator(PackageConfig(
        output_base=base, token=prj.trigger_token, class_word=prj.class_word,
        dataset_name=str(options.get("dataset_name", "DATASET")), **kw))


@register("train", "kohya LoRA training on the latest dataset build",
          "preset, dataset")
def _stage_train(prj: Project, base: Path, options: dict):
    from .train.kohya import TrainConfig, train_generator
    conn = manifest.connect(base)
    version = options.get("dataset")
    if version is None:
        row = conn.execute("SELECT MAX(version) FROM datasets").fetchone()
        if not row or not row[0]:
            yield "Train skipped: no dataset builds exist."
            return
        version = row[0]
    yield from train_generator(prj, TrainConfig(
        output_base=base, dataset_version=int(version),
        preset=str(options.get("preset", "character"))))


@register("matrix", "evaluation matrix against the newest completed run",
          "backend, forge_url")
def _stage_matrix(prj: Project, base: Path, options: dict):
    from .eval.matrix import MatrixConfig, matrix_generator
    conn = manifest.connect(base)
    row = conn.execute(
        "SELECT name FROM runs WHERE status='completed' "
        "ORDER BY run_id DESC LIMIT 1").fetchone()
    if not row:
        yield "Matrix skipped: no completed training run."
        return
    run_name = row[0]
    lora_dir = Path(prj.lora_output_dir
                    or (Path(prj.output_base) / "LORA_OUTPUT"))
    candidates = sorted(lora_dir.glob(f"{run_name}*.safetensors"))
    lora = str(candidates[-1]) if candidates else run_name
    yield from matrix_generator(prj, MatrixConfig(
        output_base=base, lora=lora,
        backend=str(options.get("backend", "forge")),
        forge_url=str(options.get("forge_url", "http://127.0.0.1:7860")),
        checkpoint=prj.base_model))


# ---------------------------------------------------------------------------
# Built-in pipeline templates (shadowable by [pipelines.NAME])
# ---------------------------------------------------------------------------

BUILTIN_PIPELINES: dict[str, dict] = {
    "curation": {
        "description": "raw media -> curated, captioned, linted dataset "
                       "with an intelligence report",
        "stages": ["extract", "prefilter", "curate", "smart", "cluster",
                   "caption", "lint", "intelligence"],
        "gates": [],
        "options": {"lint": {"fix": True}},
    },
    "selection": {
        "description": "health check -> configurable DreamBooth selection "
                       "(plan-only unless options.select.apply=true)",
        "stages": ["health", "select"],
        "gates": [],
        "options": {},
    },
    "complete": {
        "description": "curation pipeline + versioned dataset build "
                       "(training stays a deliberate, separate step)",
        "stages": ["extract", "prefilter", "curate", "smart", "cluster",
                   "caption", "lint", "build"],
        "gates": [],
        "options": {"lint": {"fix": True}},
    },
}


def available_pipelines(prj: Project) -> dict[str, dict]:
    """Built-ins merged with [pipelines.*]; user definitions shadow."""
    out = {k: dict(v) for k, v in BUILTIN_PIPELINES.items()}
    for name, spec in (getattr(prj, "pipelines", {}) or {}).items():
        if isinstance(spec, dict):
            out[name] = dict(spec)
            out[name]["_source"] = "toml"
    for name in out:
        out[name].setdefault("_source", "builtin")
    return out


# ---------------------------------------------------------------------------
# Recipe resolution
# ---------------------------------------------------------------------------

@dataclass
class RecipePipelineConfig:
    output_base: Path
    pipeline: str = "curation"
    only: list[str] = field(default_factory=list)     # run just these stages
    skip: list[str] = field(default_factory=list)
    from_stage: str | None = None
    until_stage: str | None = None
    gates: list[str] = field(default_factory=list)    # extra pause points
    overrides: list[str] = field(default_factory=list)  # "stage.key=value"
    dry_run: bool = False                              # plan only
    start_stage: str | None = None                  # internal (resume)


def _parse_override(spec: str) -> tuple[str, str, object]:
    """'stage.key=value' -> (stage, key, typed value)."""
    head, eq, raw = str(spec).partition("=")
    stage, _, key = head.partition(".")
    stage, key, raw = stage.strip(), key.strip(), raw.strip()
    if not stage or not key or not eq:
        raise ValueError(f"Bad --set {spec!r} - expected stage.key=value")
    low = raw.lower()
    value: object = raw
    if low in ("true", "false"):
        value = (low == "true")
    else:
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                pass
    return stage, key, value


def resolve_pipeline(prj: Project, cfg: RecipePipelineConfig
                     ) -> tuple[dict, list[str]]:
    """Resolve a recipe into an executable plan.

    Returns ``(plan, errors)``; plan = {name, source, description,
    stages: [(name, options, gate_after)], warnings}. Errors are fatal
    (unknown pipeline / unknown stage names).
    """
    errors: list[str] = []
    warnings: list[str] = []
    pipelines = available_pipelines(prj)
    spec = pipelines.get(cfg.pipeline)
    if spec is None:
        return {}, [f"Unknown pipeline '{cfg.pipeline}'. Available: "
                    + ", ".join(sorted(pipelines))
                    + " (plus legacy: full, mega)"]

    stages = [str(s) for s in (spec.get("stages") or [])]
    if not stages:
        return {}, [f"Pipeline '{cfg.pipeline}' defines no stages."]
    unknown = [s for s in stages if s not in STAGE_REGISTRY]
    if unknown:
        return {}, [f"Pipeline '{cfg.pipeline}' has unknown stage(s): "
                    f"{', '.join(unknown)}. Known: "
                    + ", ".join(sorted(STAGE_REGISTRY))]

    # stage windowing: --only beats --from/--until; --skip applies last
    if cfg.only:
        bad = [s for s in cfg.only if s not in stages]
        if bad:
            warnings.append(f"--stage not in this pipeline, ignored: "
                            f"{', '.join(bad)}")
        stages = [s for s in stages if s in cfg.only]
    else:
        if cfg.from_stage:
            if cfg.from_stage not in stages:
                return {}, [f"--from '{cfg.from_stage}' is not a stage of "
                            f"'{cfg.pipeline}' ({' -> '.join(stages)})"]
            stages = stages[stages.index(cfg.from_stage):]
        if cfg.until_stage:
            if cfg.until_stage not in stages:
                return {}, [f"--until '{cfg.until_stage}' is not a stage of "
                            f"'{cfg.pipeline}'"]
            stages = stages[:stages.index(cfg.until_stage) + 1]
    if cfg.skip:
        stages = [s for s in stages if s not in cfg.skip]
    if not stages:
        return {}, ["Stage selection left nothing to run."]

    # options: recipe options <- --set overrides
    options: dict[str, dict] = {
        s: dict((spec.get("options") or {}).get(s) or {}) for s in stages}
    for ov in cfg.overrides:
        try:
            stage, key, value = _parse_override(ov)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if stage not in options:
            warnings.append(f"--set for stage '{stage}' not in the run - "
                            "ignored")
            continue
        options[stage][key] = value

    gates = {str(g) for g in (spec.get("gates") or [])} | set(cfg.gates)
    plan_stages = [(s, options[s], s in gates) for s in stages]
    return {
        "name": cfg.pipeline,
        "source": spec.get("_source", "builtin"),
        "description": spec.get("description", ""),
        "stages": plan_stages,
        "warnings": warnings,
    }, errors


# ---------------------------------------------------------------------------
# State (checkpoint after EVERY stage)
# ---------------------------------------------------------------------------

def save_recipe_state(conn, cfg: RecipePipelineConfig, plan: dict,
                      completed: list[dict], next_stage: str | None,
                      error: str | None = None) -> None:
    manifest.meta_set(conn, RECIPE_STATE_KEY, {
        "pipeline": cfg.pipeline,
        "stages": [s for s, _, _ in plan["stages"]],
        "gates": [s for s, _, g in plan["stages"] if g],
        "overrides": cfg.overrides,
        "only": cfg.only, "skip": cfg.skip,
        "from_stage": cfg.from_stage, "until_stage": cfg.until_stage,
        "completed": completed,
        "next_stage": next_stage,
        "error": error,
        "saved_at": now_iso(),
    })


def load_recipe_state(conn) -> dict | None:
    return manifest.meta_get(conn, RECIPE_STATE_KEY)


def clear_recipe_state(conn) -> None:
    conn.execute("DELETE FROM meta WHERE key = ?", (RECIPE_STATE_KEY,))
    conn.commit()


# ---------------------------------------------------------------------------
# Plan rendering + execution engine
# ---------------------------------------------------------------------------

def _manifest_snapshot(base: Path) -> str:
    try:
        conn = manifest.connect(base, readonly=True)
        counts = dict(conn.execute(
            "SELECT status, COUNT(*) FROM frames GROUP BY status"))
        conn.close()
        if not counts:
            return "manifest: no frames yet"
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
        return ("manifest: " + ", ".join(f"{k}: {v}" for k, v in top)
                + f"  (total {sum(counts.values())})")
    except Exception:
        return "manifest: not created yet"


def render_plan(plan: dict, base: Path) -> str:
    lines = ["", f"PIPELINE PLAN '{plan['name']}' ({plan['source']})"]
    if plan.get("description"):
        lines.append(f"  {plan['description']}")
    lines.append("  " + _manifest_snapshot(base))
    lines.append("-" * 60)
    for i, (name, options, gate) in enumerate(plan["stages"], 1):
        spec = STAGE_REGISTRY[name]
        mark = "  |GATE| pause after" if gate else ""
        lines.append(f"  {i}. {name:<13} {spec.description}{mark}")
        if options:
            opts = ", ".join(f"{k}={v}" for k, v in sorted(options.items()))
            lines.append(f"     options: {opts}")
    for w in plan.get("warnings", []):
        lines.append(f"  ! {w}")
    lines.append("-" * 60)
    lines.append("  (plan only - nothing was executed; run with: "
                 f"pixel-curator pipeline run {plan['name']})")
    return "\n".join(lines)


def recipe_pipeline_generator(prj: Project, cfg: RecipePipelineConfig
                              ) -> Generator[str, None, None]:
    plan, errors = resolve_pipeline(prj, cfg)
    if errors:
        for e in errors:
            yield f"FATAL: {e}"
        return

    if cfg.dry_run:
        yield render_plan(plan, cfg.output_base)
        return

    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)
    names = [s for s, _, _ in plan["stages"]]

    start_idx = 0
    if cfg.start_stage:
        if cfg.start_stage not in names:
            yield (f"FATAL: resume stage '{cfg.start_stage}' is not in "
                   f"pipeline '{cfg.pipeline}' ({' -> '.join(names)})")
            return
        start_idx = names.index(cfg.start_stage)

    for w in plan.get("warnings", []):
        yield f"! {w}"
    route = " -> ".join(
        n + (" |GATE|" if g else "")
        for n, _, g in plan["stages"][start_idx:])
    yield (f"PIPELINE '{cfg.pipeline}' ({plan['source']})\n"
           f"Plan: {route}\n{_manifest_snapshot(cfg.output_base)}\n")

    completed: list[dict] = []
    if cfg.start_stage:
        prior = load_recipe_state(conn) or {}
        completed = list(prior.get("completed") or [])

    total = len(plan["stages"]) - start_idx
    for i in range(start_idx, len(plan["stages"])):
        name, options, gate = plan["stages"][i]
        step = i - start_idx + 1
        yield (f"\n========== STAGE {step}/{total}: {name} ==========")
        t0 = time.time()
        try:
            for update in STAGE_REGISTRY[name].factory(
                    prj, cfg.output_base, options):
                yield f"[{name}] {update}"
        except Exception as exc:
            save_recipe_state(conn, cfg, plan, completed,
                              next_stage=name, error=str(exc))
            import logging
            logging.exception("pipeline stage failed")
            yield (f"\nPIPELINE PAUSED: stage '{name}' failed: {exc}\n"
                   "Fix the issue, then: pixel-curator pipeline resume")
            return
        completed.append({"stage": name,
                          "seconds": round(time.time() - t0, 1),
                          "at": now_iso()})
        next_stage = names[i + 1] if i + 1 < len(names) else None
        # checkpoint after EVERY stage - an interrupt resumes correctly
        if next_stage:
            save_recipe_state(conn, cfg, plan, completed, next_stage)
        if gate and next_stage:
            yield (f"\nPIPELINE GATED after '{name}'. Review, then: "
                   "pixel-curator pipeline resume")
            return

    clear_recipe_state(conn)
    timing = "  ".join(f"{c['stage']} {c['seconds']}s" for c in completed)
    yield (f"\nPIPELINE '{cfg.pipeline}' COMPLETE "
           f"({len(completed)} stage(s)).\n{timing}")


def recipe_resume_generator(prj: Project, output_base: Path
                            ) -> Generator[str, None, None]:
    """Resume a recipe pipeline from its checkpoint. Returns silently
    (yields nothing) when no recipe state exists, so the CLI can fall back
    to the legacy resume path."""
    conn = manifest.connect(output_base)
    state = load_recipe_state(conn)
    if not state:
        return
    next_stage = state.get("next_stage")
    if not next_stage:
        yield "Recipe pipeline state is complete - nothing to resume."
        clear_recipe_state(conn)
        return
    cfg = RecipePipelineConfig(
        output_base=output_base,
        pipeline=state.get("pipeline", "curation"),
        only=state.get("only") or [],
        skip=state.get("skip") or [],
        from_stage=state.get("from_stage"),
        until_stage=state.get("until_stage"),
        gates=state.get("gates") or [],
        overrides=state.get("overrides") or [],
        start_stage=next_stage,
    )
    done = ", ".join(c["stage"] for c in state.get("completed") or []) \
        or "(none)"
    err = state.get("error")
    yield (f"Resuming pipeline '{cfg.pipeline}' at '{next_stage}' "
           f"(saved {state.get('saved_at')}; done: {done}"
           + (f"; last error: {err}" if err else "") + ")")
    yield from recipe_pipeline_generator(prj, cfg)


def render_pipeline_list(prj: Project) -> str:
    lines = ["", "Available pipelines", "=" * 60]
    for name, spec in sorted(available_pipelines(prj).items()):
        stages = " -> ".join(str(s) for s in (spec.get("stages") or []))
        lines.append(f"  {name:<14} [{spec.get('_source', 'builtin')}] "
                     f"{spec.get('description', '')}")
        lines.append(f"    {stages}")
    lines.append(f"  {'full':<14} [legacy] fixed DAG: extract -> curate -> "
                 "caption -> build -> train -> matrix")
    lines.append(f"  {'mega':<14} [legacy] one-click: raw media -> trained "
                 "best epoch -> playground preset")
    lines.append("")
    lines.append("Stages: " + ", ".join(sorted(STAGE_REGISTRY)))
    lines.append("Define your own under [pipelines.NAME] in the project "
                 "TOML (stages/gates/options).")
    return "\n".join(lines)


__all__ = [
    "STAGE_REGISTRY", "BUILTIN_PIPELINES", "LEGACY_NAMES",
    "RecipePipelineConfig", "StageSpec", "available_pipelines",
    "resolve_pipeline", "recipe_pipeline_generator",
    "recipe_resume_generator", "render_plan", "render_pipeline_list",
    "load_recipe_state", "save_recipe_state", "clear_recipe_state",
]
