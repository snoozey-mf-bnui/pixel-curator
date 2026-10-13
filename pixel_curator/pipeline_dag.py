"""Phase 7: full pipeline DAG with per-stage gates and resumability.

  pixel-curator pipeline run full --recipe character_v1 --preset character
  pixel-curator pipeline run full --gate curate --gate build   # pause points
  pixel-curator pipeline resume
  pixel-curator pipeline status

Stages: extract -> curate (basic [+smart] [+cluster]) -> caption -> build ->
train -> matrix. A *gate* pauses the chain AFTER the named stage (e.g. to
hand-review the grid before captioning); `pipeline resume` continues from
the saved position. State lives in the manifest meta table, so resume works
across restarts. Stages whose optional extras are missing skip themselves
gracefully (their generators already do).
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

from . import manifest

logger = logging.getLogger(__name__)
from .config import (
    CaptionConfig,
    ClusterConfig,
    CurateConfig,
    ExtractConfig,
    Project,
    SmartCurateConfig,
)
from .util import now_iso, setup_logging

STAGES = ["extract", "curate", "caption", "build", "train", "matrix"]
STATE_KEY = "pipeline_state"


@dataclass
class PipelineConfig:
    output_base: Path
    recipe: str = "character_v1"
    preset: str = "character"
    gates: list[str] = field(default_factory=list)   # pause AFTER these stages
    smart: bool = True
    cluster: bool = True
    matrix_backend: str = "forge"
    forge_url: str = "http://127.0.0.1:7860"
    dry_run_extract: bool = False
    start_stage: str | None = None                # internal (resume)


def _stage_generators(prj: Project, cfg: PipelineConfig):
    """Yields (stage_name, generator_factory) in DAG order."""
    base = cfg.output_base

    def g_extract():
        from .extract import pipeline_generator
        ecfg = ExtractConfig(output_base=base, dry_run=cfg.dry_run_extract)
        yield from pipeline_generator(prj.video_dirs, prj.photos_dir, ecfg)

    def g_curate():
        from .curate import curate_generator, smart_curate_generator
        yield from curate_generator(CurateConfig(output_base=base))
        if cfg.smart:
            yield from smart_curate_generator(SmartCurateConfig(output_base=base))
        if cfg.cluster:
            from .curate.diversity import cluster_generator
            yield from cluster_generator(ClusterConfig(output_base=base))

    def g_caption():
        from .caption import caption_generator
        yield from caption_generator(CaptionConfig(
            output_base=base, trigger=prj.trigger_token, class_word=prj.class_word,
        ))

    def g_build():
        from .builder import BuildConfig, build_generator
        yield from build_generator(prj, BuildConfig(
            output_base=base, recipe=cfg.recipe,
            note=f"pipeline run {now_iso()}",
        ))

    def g_train():
        from .train.kohya import TrainConfig, train_generator
        conn = manifest.connect(base)
        row = conn.execute("SELECT MAX(version) FROM datasets").fetchone()
        if not row or not row[0]:
            yield "Train skipped: no dataset builds exist."
            return
        yield from train_generator(prj, TrainConfig(
            output_base=base, dataset_version=int(row[0]), preset=cfg.preset,
        ))

    def g_matrix():
        from .eval.matrix import MatrixConfig, matrix_generator
        conn = manifest.connect(base)
        row = conn.execute(
            "SELECT name FROM runs WHERE status='completed' "
            "ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        if not row:
            yield "Matrix skipped: no completed training run."
            return
        run_name = row[0]
        lora_dir = Path(prj.lora_output_dir or (Path(prj.output_base) / "LORA_OUTPUT"))
        candidates = sorted(lora_dir.glob(f"{run_name}*.safetensors"))
        lora = str(candidates[-1]) if candidates else run_name
        yield from matrix_generator(prj, MatrixConfig(
            output_base=base, lora=lora, backend=cfg.matrix_backend,
            forge_url=cfg.forge_url, checkpoint=prj.base_model,
        ))

    return [
        ("extract", g_extract), ("curate", g_curate), ("caption", g_caption),
        ("build", g_build), ("train", g_train), ("matrix", g_matrix),
    ]


def save_state(conn, cfg: PipelineConfig, next_stage: str) -> None:
    manifest.meta_set(conn, STATE_KEY, {
        "next_stage": next_stage, "recipe": cfg.recipe, "preset": cfg.preset,
        "gates": cfg.gates, "smart": cfg.smart, "cluster": cfg.cluster,
        "matrix_backend": cfg.matrix_backend, "forge_url": cfg.forge_url,
        "saved_at": now_iso(),
    })


def load_state(conn) -> dict | None:
    return manifest.meta_get(conn, STATE_KEY)


def clear_state(conn) -> None:
    conn.execute("DELETE FROM meta WHERE key = ?", (STATE_KEY,))
    conn.commit()


def pipeline_generator(prj: Project, cfg: PipelineConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)
    stages = _stage_generators(prj, cfg)

    start_idx = 0
    if cfg.start_stage:
        names = [n for n, _ in stages]
        if cfg.start_stage not in names:
            yield f"FATAL: unknown stage '{cfg.start_stage}'"
            return
        start_idx = names.index(cfg.start_stage)

    plan = " -> ".join(
        n + (" |GATE|" if n in cfg.gates else "") for n, _ in stages[start_idx:]
    )
    yield f"PIPELINE [{cfg.recipe} / {cfg.preset}]\nPlan: {plan}\n"

    for i in range(start_idx, len(stages)):
        name, factory = stages[i]
        yield f"\n========== STAGE: {name} =========="
        try:
            for update in factory():
                yield f"[{name}] {update}"
        except Exception as exc:
            save_state(conn, cfg, name)
            logging.exception("pipeline stage failed")
            yield (f"\nPIPELINE PAUSED: stage '{name}' failed: {exc}\n"
                   "Fix the issue, then: pixel-curator pipeline resume")
            return
        if name in cfg.gates and i + 1 < len(stages):
            save_state(conn, cfg, stages[i + 1][0])
            yield (f"\nPIPELINE GATED after '{name}'. Review (UI: Review/Tags/"
                   f"Builds tabs), then: pixel-curator pipeline resume")
            return

    clear_state(conn)
    yield "\nPIPELINE COMPLETE - extract through eval matrix."


# ---------------------------------------------------------------------------
# Mega pipeline: raw media -> trained best epoch -> playground preset
# ---------------------------------------------------------------------------

@dataclass
class MegaConfig:
    output_base: Path
    lora_type: str = "auto"
    backend: str = "forge"
    forge_url: str = "http://127.0.0.1:7860"
    smart: bool = True
    cluster: bool = True
    preset_name: str = ""
    presets_path: str = ""           # default: <project parent>/outputs/playground_presets.json


def write_playground_preset(prj: Project, name: str,
                            loras: list[tuple[str, float]],
                            path: Path | None = None) -> Path:
    """Write/merge a Grokkie Playground preset so the stack is one click away."""
    import json
    target = Path(path) if path else (
        Path(__file__).resolve().parent / "outputs" / "playground_presets.json")
    presets: dict = {}
    try:
        presets = json.loads(target.read_text())
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    from .base_models import detect_profile, preset_payload, sample_prompts
    prof = detect_profile(prj.base_model)
    trig = f"{prj.trigger_token} {prj.class_word}".strip()
    body = preset_payload(prof, trig, base_model=prj.base_model, loras=loras)
    body["_studio"] = {"shipped_at": now_iso(), "profile": prof["label"],
                       "trigger": trig,
                       "sample_prompts": sample_prompts(prof, trig)}
    presets[name] = body
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(presets, indent=2))
    return target


def mega_generator(prj: Project, cfg: MegaConfig) -> Generator[str, None, None]:
    """One click: raw media -> curated -> captioned -> typed build -> train
    -> best-epoch sweep -> playground preset. Each stage streams progress
    and the chain stops with guidance on the first hard failure."""
    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)
    yield "FULL STUDIO PIPELINE\nraw media -> factory -> best epoch -> playground\n"

    # 1-3: ingest + curate + caption (reuse the DAG's stage closures)
    pcfg = PipelineConfig(output_base=cfg.output_base, smart=cfg.smart,
                          cluster=cfg.cluster)
    stages = dict((n, f) for n, f in _stage_generators(prj, pcfg))
    for stage in ("extract", "curate", "caption"):
        yield f"\n===== {stage} ====="
        try:
            for u in stages[stage]():
                yield f"[{stage}] {u}"
        except Exception as exc:
            yield f"\nSTOPPED at {stage}: {exc}"
            return

    # 4-5: wizard (type detect -> build -> train -> matrix -> card)
    yield "\n===== create (wizard) ====="
    from .wizard import WizardConfig, wizard_generator
    for u in wizard_generator(prj, WizardConfig(
            output_base=cfg.output_base, lora_type=cfg.lora_type,
            matrix_backend=cfg.backend, forge_url=cfg.forge_url)):
        yield f"[wizard] {u}"
    row = conn.execute(
        "SELECT name, status FROM runs ORDER BY run_id DESC LIMIT 1").fetchone()
    if not row or row[1] != "completed":
        yield "\nSTOPPED: training did not complete; fix and resume from the Train tab."
        return
    run_name = row[0]

    # 6: best-epoch sweep
    yield f"\n===== sweep ({run_name}) ====="
    from .sweep import SweepConfig, sweep_generator, sweep_summary
    for u in sweep_generator(prj, SweepConfig(
            output_base=cfg.output_base, run=run_name,
            backend=cfg.backend, forge_url=cfg.forge_url)):
        yield f"[sweep] {u}"

    # 7: ship preset (best epoch, else final checkpoint)
    s = sweep_summary(conn, run_name)
    best_label = (s or {}).get("best", {}) or {}
    stem = best_label.get("label") or run_name
    preset_name = cfg.preset_name or f"studio_{prj.trigger_token}"
    target = write_playground_preset(
        prj, preset_name, [(stem, 0.85)],
        Path(cfg.presets_path) if cfg.presets_path else None)
    yield (f"\nFULL PIPELINE COMPLETE.\n"
           f"Run: {run_name} | best epoch: {best_label.get('epoch', '(unscored - using final)')}\n"
           f"Playground preset '{preset_name}' -> {target}\n"
           "Open the Playground tab -> Presets -> Load.")


def resume_generator(prj: Project, output_base: Path) -> Generator[str, None, None]:
    conn = manifest.connect(output_base)
    state = load_state(conn)
    if not state:
        yield "Nothing to resume - no saved pipeline state."
        return
    cfg = PipelineConfig(
        output_base=output_base,
        recipe=state.get("recipe", "character_v1"),
        preset=state.get("preset", "character"),
        gates=state.get("gates", []),
        smart=state.get("smart", True),
        cluster=state.get("cluster", True),
        matrix_backend=state.get("matrix_backend", "forge"),
        forge_url=state.get("forge_url", "http://127.0.0.1:7860"),
        start_stage=state.get("next_stage"),
    )
    yield f"Resuming pipeline at stage '{cfg.start_stage}' (saved {state.get('saved_at')})"
    yield from pipeline_generator(prj, cfg)
