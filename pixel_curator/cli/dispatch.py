"""Command dispatch: builds the context and routes to handlers."""

from __future__ import annotations

import logging

from ..config import load_project
from .commands.captioning import *  # noqa: F401,F403
from .commands.curation import *  # noqa: F401,F403
from .commands.exporting import *  # noqa: F401,F403
from .commands.ingest import *  # noqa: F401,F403
from .commands.ops import *  # noqa: F401,F403
from .commands.studio import *  # noqa: F401,F403
from .commands.training import *  # noqa: F401,F403
from .common import Context
from .emitter import UpdateEmitter, set_emitter
from .parser import parse_args

logger = logging.getLogger(__name__)

HANDLERS = {
    "ab": cmd_ab,
    "analyze": cmd_analyze,
    "anchor": cmd_anchor,
    "anchors-rebuild": cmd_anchors_rebuild,
    "backup": cmd_backup,
    "build": cmd_build,
    "caption": cmd_caption,
    "card": cmd_card,
    "categorize": cmd_categorize,
    "classify": cmd_classify,
    "cluster": cmd_cluster,
    "combo": cmd_combo,
    "concept": cmd_concept,
    "curate": cmd_curate,
    "dataset-card": cmd_dataset_card,
    "datasets": cmd_datasets,
    "diff": cmd_diff,
    "diverse-select": cmd_diverse_select,
    "doctor": cmd_doctor,
    "duplicates": cmd_duplicates,
    "eval": cmd_eval,
    "evals": cmd_evals,
    "exp": cmd_exp,
    "explorer": cmd_explorer,
    "export": cmd_export,
    "export-dreambooth": cmd_export_dreambooth,
    "extract": cmd_extract,
    "face-fix": cmd_face_fix,
    "figure-review": cmd_figure_review,
    "filter": cmd_filter,
    "florence": cmd_florence,
    "framing-report": cmd_framing_report,
    "gc": cmd_gc,
    "generate-artistic": cmd_generate_artistic,
    "generate-figure-study": cmd_generate_figure_study,
    "golden": cmd_golden,
    "health": cmd_health,
    "history": cmd_history,
    "init": cmd_init,
    "insights": cmd_insights,
    "keyframes": cmd_keyframes,
    "leaderboard": cmd_leaderboard,
    "learn": cmd_learn,
    "likeness": cmd_likeness,
    "lookbook": cmd_lookbook,
    "matrix": cmd_matrix,
    "merge": cmd_merge,
    "mergeforge": cmd_mergeforge,
    "multi-recipe-packages": cmd_multi_recipe_packages,
    "orient": cmd_orient,
    "outfit": cmd_outfit,
    "overview": cmd_overview,
    "package": cmd_package,
    "package-figure-study": cmd_package_figure_study,
    "package-legacy": cmd_package_legacy,
    "photos": cmd_photos,
    "pick-epoch": cmd_pick_epoch,
    "pipeline": cmd_pipeline,
    "plan": cmd_plan,
    "prefilter": cmd_prefilter,
    "presets": cmd_presets,
    "previews": cmd_previews,
    "quality": cmd_quality,
    "recipes": cmd_recipes,
    "recommend-stack": cmd_recommend_stack,
    "registry": cmd_registry,
    "render-figure-study": cmd_render_figure_study,
    "rescore": cmd_rescore,
    "restore": cmd_restore,
    "review": cmd_review,
    "runs": cmd_runs,
    "selftest": cmd_selftest,
    "smart-extract": cmd_smart_extract,
    "space": cmd_space,
    "specialized-recipes": cmd_specialized_recipes,
    "stats": cmd_stats,
    "status": cmd_status,
    "studio": cmd_studio,
    "study": cmd_study,
    "suggest-preset": cmd_suggest_preset,
    "sweep": cmd_sweep,
    "testgen": cmd_testgen,
    "train": cmd_train,
    "train-legacy": cmd_train_legacy,
    "training-config": cmd_training_config,
    "tune-stack": cmd_tune_stack,
    "validate": cmd_validate,
    "visual-focus": cmd_visual_focus,
    "viz": cmd_viz,
    "vlm-caption": cmd_vlm_caption,
    "watch": cmd_watch,
    "wizard": cmd_wizard,
    "workflow": cmd_workflow,
}


def run(argv: list[str] | None = None) -> None:
    """Parse args, build the shared context, and run the command handler."""
    args = parse_args(argv)
    from ..logutil import configure_logging
    configure_logging(quiet=getattr(args, "quiet", False),
                      verbose=getattr(args, "verbose", False))
    set_emitter(UpdateEmitter(
        quiet=getattr(args, "quiet", False),
        color=(False if getattr(args, "no_color", False) else None)))
    prj = load_project(args.project)
    cmd = args.command or "ui"

    # Apply the project's [runtime] policy (torch device, tf32, dedicated
    # onnxruntime provider) before any command runs.
    if cmd != "init":
        try:
            from ..runtime import configure_from_project
            configure_from_project(prj)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass

    def out_base(flag):
        from pathlib import Path
        return Path(flag or prj.output_base).expanduser()

    handler = HANDLERS.get(cmd)
    if handler is not None:
        handler(Context(args=args, prj=prj, out_base=out_base))
