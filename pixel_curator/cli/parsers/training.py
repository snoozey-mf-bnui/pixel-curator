"""Parser registration: training commands."""

from __future__ import annotations

from ...config import DEFAULT_FPS_TIERS, DEFAULT_QUOTA, UI_PORT  # noqa: F401


def register(sub):
    p_lk = sub.add_parser(
        "likeness", help="Score generated eval images by ArcFace cosine to the "
                         "dataset identity; writes evals.likeness")
    p_lk.add_argument("--output", default=None)
    p_lk.add_argument("--lora", default=None, help="only score this LoRA's evals")
    p_lk.add_argument("--all", action="store_true",
                      help="re-score rows that already have a likeness")
    p_lk.add_argument("--min-identity-sim", type=float, default=0.0,
                      help="only build the reference centroid from frames with "
                           "at least this stored identity_sim")
    p_lk.add_argument("--json", action="store_true")

    p_lb = sub.add_parser(
        "leaderboard", help="Rank candidate LoRAs against each other on shared "
                            "eval prompts (uses evals.likeness)")
    p_lb.add_argument("--output", default=None)
    p_lb.add_argument("--label", action="append", default=[],
                      help="restrict to these candidate labels (repeatable)")
    p_lb.add_argument("--vs", nargs=2, metavar=("A", "B"), default=None,
                      help="head-to-head between two candidate labels")
    p_lb.add_argument("--json", action="store_true")

    p_pe = sub.add_parser("pick-epoch",
                          help="Score every saved epoch of a LoRA by likeness "
                               "and name the keeper (needs Forge + anchor)")
    p_pe.add_argument("--lora-base", required=True,
                      help="output_name stem, e.g. ohwx_character_v012")
    p_pe.add_argument("--lora-dir", default=None,
                      help="dir with the -NNNNNN.safetensors (default: lora_output_dir)")
    p_pe.add_argument("--weights", default="0.6,0.75,0.9")
    p_pe.add_argument("--checkpoint", default="")
    p_pe.add_argument("--forge-url", default="http://127.0.0.1:7860")
    p_pe.add_argument("--trigger", default="")
    p_pe.add_argument("--output", default=None)

    p_ts = sub.add_parser("tune-stack",
                          help="Sweep LoRA stack weights, rank by likeness "
                               "(needs Forge + anchor)")
    p_ts.add_argument("--identity", required=True, help="identity LoRA stem")
    p_ts.add_argument("--concepts", default="",
                      help="concept LoRAs to tune: stem:lo-hi,stem2:val")
    p_ts.add_argument("--identity-weights", default="0.8,0.9,1.0")
    p_ts.add_argument("--cap", type=int, default=40)
    p_ts.add_argument("--checkpoint", default="")
    p_ts.add_argument("--forge-url", default="http://127.0.0.1:7860")
    p_ts.add_argument("--trigger", default="")
    p_ts.add_argument("--output", default=None)

    sub.add_parser("recipes", help="List recipes defined in the project file")

    p_sr = sub.add_parser(
        "specialized-recipes", help="Build specialized training-recipe plans "
                                    "from visual-focus metadata (planning only)")
    p_sr.add_argument("--input", default=None,
                      help="visual_focus_classifications.jsonl")
    p_sr.add_argument("--dataset", default=None,
                      help="dir containing visual_focus/visual_focus_classifications.jsonl")
    p_sr.add_argument("--output", default=None)
    p_sr.add_argument("--project", dest="project_toml", default=None,
                      help="project.toml for recipe existence checks (the global "
                           "-p/--project still selects the studio project)")
    p_sr.add_argument("--architecture", choices=("sdxl", "unknown"),
                      default="sdxl")
    p_sr.add_argument("--min-confidence", type=float, default=0.55)
    p_sr.add_argument("--min-suitability", type=float, default=50.0)
    p_sr.add_argument("--max-recipes-per-image", type=int, default=4)
    p_sr.add_argument("--allow-mixed-architecture", action="store_true")
    p_sr.add_argument("--overwrite", action="store_true")
    p_sr.add_argument("--all", dest="process_all", action="store_true",
                      help="Process the full manifest (paginate past the 500 "
                           "UI page cap) instead of a single sample page.")
    p_sr.add_argument("--max-records", type=int, default=1_000_000,
                      help="Upper bound on records loaded with --all.")
    p_sr.add_argument("--json", action="store_true")

    # Multi-Recipe Training Package Generator (recipe plans -> package folders)

    from ...train.presets import PRESETS as _PRESETS
    # NOTE: renamed from "train" to "train-legacy" to avoid argparse conflict
    # with the newer recipe-builder `train` subparser registered below. The
    # recipe builder (train plan | stack-plan) is what the tests
    # and the UI dispatch to; this legacy launcher is kept reachable under
    # the new name `train-legacy`.
    p_tr = sub.add_parser("train-legacy", help="Train a LoRA from a dataset build (Phase 5, legacy launcher)")
    p_tr.add_argument("--dataset", type=int, required=True, help="dataset version")
    p_tr.add_argument("--preset", default="character",
                      choices=sorted(_PRESETS))
    p_tr.add_argument("--name", default="")
    p_tr.add_argument("--output", default=None)
    p_tr.add_argument("--dry-run", action="store_true",
                      help="print the sd-scripts command without running")
    p_tr.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                      help="override any sd-scripts arg, e.g. --set max_train_epochs=12")
    p_tr.add_argument("--min-free-gb", type=float, default=20.0,
                      help="refuse to launch if the boot volume has less "
                           "free space than this (swap/MPS-temp safety; "
                           "0 disables)")
    p_tr.add_argument("--register-experiment", action="store_true",
                      help="on success, record an experiment linking the "
                           "trained model to this dataset version "
                           "(Phase 8; see `pixel-curator exp list`)")

    sub.add_parser("presets", help="Show training preset parameters")

    p_sp = sub.add_parser("suggest-preset",
                          help="Recommend a training preset for a built "
                               "dataset (dataset analysis + Study "
                               "Intelligence + recipe context)")
    p_sp.add_argument("--dataset", type=int, required=True,
                      help="dataset version")
    p_sp.add_argument("--recipe", default=None,
                      help="recipe context override (defaults to the "
                           "recipe stored with the build)")
    p_sp.add_argument("--json", action="store_true",
                      help="emit the full recommendation as JSON")
    p_sp.add_argument("--no-params", dest="with_params",
                      action="store_false", default=True,
                      help="skip the dataset-aware parameter suggestions "
                           "(repeats/epochs/LR/steps/time estimate)")
    p_sp.add_argument("--output", default=None)

    p_tc = sub.add_parser(
        "training-config",
        help="Generate (or regenerate) the ready-to-train kohya config + "
             "launch script for a dataset export (Phase 10)")
    p_tc.add_argument("--dataset", type=int, default=None,
                      help="dataset version to analyze (recommended; "
                           "enables full dataset-aware sizing)")
    p_tc.add_argument("--dir", default=None,
                      help="export folder to write beside (default: the "
                           "version's registered directory)")
    p_tc.add_argument("--frames", type=int, default=None,
                      help="frame count for unversioned folders "
                           "(count-based fallback sizing)")
    p_tc.add_argument("--model-type", choices=["lora", "dreambooth"],
                      default="lora")
    p_tc.add_argument("--preset", default=None,
                      help="force a preset (character/style/outfit/pose/"
                           "detail); default: auto from dataset analysis")
    p_tc.add_argument("--base-model", default=None,
                      help="checkpoint path to bake into the config")
    p_tc.add_argument("--name", default=None,
                      help="output_name for the trained model")
    p_tc.add_argument("--set", action="append", default=[],
                      metavar="KEY=VALUE",
                      help="override any config key (repeatable), e.g. "
                           "--set max_train_epochs=6 --set unet_lr=8e-5")
    p_tc.add_argument("--dry-run", action="store_true",
                      help="print the config instead of writing files")
    p_tc.add_argument("--json", action="store_true")
    p_tc.add_argument("--output", default=None)

    p_mx = sub.add_parser("matrix", help="Likeness/flexibility/pose/outfit/style test grids")
    p_mx.add_argument("--lora", required=True, help="LoRA path (or model stem for forge)")
    p_mx.add_argument("--label", default="")
    p_mx.add_argument("--weight", type=float, default=0.85)
    p_mx.add_argument("--categories", default="",
                      help="comma list; default all (likeness,flexibility,pose,outfit,style)")
    p_mx.add_argument("--backend", choices=["forge", "diffusers"], default="forge")
    p_mx.add_argument("--forge-url", default="http://127.0.0.1:7860")
    p_mx.add_argument("--checkpoint", default="")
    p_mx.add_argument("--steps", type=int, default=0,
                      help="0 = base-model profile default")
    p_mx.add_argument("--cfg", type=float, default=0.0,
                      help="0 = base-model profile default")
    p_mx.add_argument("--sampler", default="",
                      help="default: base-model profile sampler")
    p_mx.add_argument("--clip-skip", type=int, default=0,
                      help="0 = base-model profile default")
    p_mx.add_argument("--no-pony-prefix", dest="pony", action="store_false", default=True)
    p_mx.add_argument("--output", default=None)

    sub.add_parser("evals", help="Eval summary (avg likeness per LoRA/category)")

    p_sw = sub.add_parser("sweep", help="Best-Epoch Sweep: eval every epoch, recommend best")
    p_sw.add_argument("--run", required=True, help="run name (pixel-curator runs)")
    p_sw.add_argument("--backend", default="forge", choices=["forge", "diffusers"])
    p_sw.add_argument("--forge-url", default="http://127.0.0.1:7860")
    p_sw.add_argument("--checkpoint", default="")
    p_sw.add_argument("--weight", type=float, default=0.85)
    p_sw.add_argument("--max-gap", type=float, default=0.15)
    p_sw.add_argument("--limit-epochs", type=int, default=0)
    p_sw.add_argument("--categories", default="likeness,flexibility",
                      help="comma list; add fine_art for figure-study "
                           "presets, e.g. likeness,flexibility,fine_art")
    p_sw.add_argument("--output", default=None)

    # ----- Phase 7 -----

    sub.add_parser("registry", help="List all known LoRAs (trained/merged/external)")

    p_cd = sub.add_parser("card", help="Export a model card (legacy: "
                          "positional LoRA name; Phase 3: --run)")
    p_cd.add_argument("name", nargs="?", default=None)
    p_cd.add_argument("--run", default="",
                      help="generate a professional model card for a "
                           "training run")
    p_cd.add_argument("--output", default="",
                      help="card output path (default: "
                           "runs/<run>/model_card.md)")
    p_cd.add_argument("--no-samples", dest="include_samples",
                      action="store_false", default=True)

    p_ev = sub.add_parser("eval", help="Post-training evaluation matrix: "
                          "identity consistency, flexibility grid, overfit "
                          "risk, artistic quality (Phase 3)")
    p_ev.add_argument("--run", required=True, help="run name "
                      "(pixel-curator runs)")
    p_ev.add_argument("--json", action="store_true")
    p_ev.add_argument("--dry-run", action="store_true",
                      help="report only - do not write files/meta")
    p_ev.add_argument("--generate-samples", action="store_true",
                      help="render + score the eval matrix via Forge "
                           "before reporting (requires Forge running)")
    p_ev.add_argument("--output", default=None)

    # ----- Phase 6: Studio Orchestration & Production Control Center -----

    p_mg = sub.add_parser("merge", help="Merge LoRAs (weighted + block weights)")
    p_mg.add_argument("--lora", action="append", required=True,
                      metavar="PATH:WEIGHT", help="repeatable; negative weights ok")
    p_mg.add_argument("--name", default="merged")
    p_mg.add_argument("--blocks", default="",
                      help="default block multipliers, e.g. 'te=0,down=1,mid=1,up=0.5'")
    p_mg.add_argument("--preview", action="store_true",
                      help="run a likeness matrix on the merged LoRA afterwards")
    p_mg.add_argument("--backend", default="forge", choices=["forge", "diffusers"])
    p_mg.add_argument("--output", default=None)

    p_mf = sub.add_parser("mergeforge", help="LoRA merge engine: scan / "
                                             "plan / merge / recipes")
    mf_sub = p_mf.add_subparsers(dest="mf_cmd", required=True)
    mf_sub.add_parser("scan", help="Classify + health-score the LoRA library")
    mf_p = mf_sub.add_parser("plan", help="Compatibility + weight plan for "
                                          "library lora_ids")
    mf_p.add_argument("--lora", action="append", required=True,
                      help="repeatable library lora_id")
    mf_m = mf_sub.add_parser("merge", help="Execute a merge (weighted_sum "
                                           "with SVD re-extract, or concat)")
    mf_m.add_argument("--lora", action="append", required=True,
                      metavar="PATH:WEIGHT", help="repeatable")
    mf_m.add_argument("--name", default="mergeforge_v1")
    mf_m.add_argument("--method", default="weighted_sum",
                      choices=["weighted_sum", "concat"])
    mf_m.add_argument("--rank", type=int, default=0,
                      help="target rank (0 = max input rank)")
    mf_m.add_argument("--output", default=None)
    mf_r = mf_sub.add_parser("recipes", help="List saved merge recipes")
    mf_r.add_argument("--output", default=None)
    mf_x = mf_sub.add_parser("rerun", help="Re-execute a saved recipe by id")
    mf_x.add_argument("recipe_id")
    mf_x.add_argument("--output", default=None)

    p_cc = sub.add_parser(
        "concept", help="Concept Control Layer: explorer scan / stack / batch")
    cc_sub = p_cc.add_subparsers(dest="concept_cmd", required=True)
    cc_s = cc_sub.add_parser("scan", help="Scan LoRA folders -> influence "
                                          "profiles (manifest + sidecars)")
    cc_s.add_argument("--write-sidecars", action="store_true",
                      help="write .concept.json sidecars for new LoRAs")
    cc_k = cc_sub.add_parser("stack", help="Resolve an explained LoRA stack")
    cc_k.add_argument("--lora", action="append", default=[],
                      metavar="ID[:WEIGHT]", help="repeatable")
    cc_b = cc_sub.add_parser("batch", help="Expand a controlled variation "
                                           "grid (manifest-tracked)")
    cc_b.add_argument("--spec", required=True,
                      help="JSON grid spec file (see docs)")
    cc_b.add_argument("--mode", default="low_risk",
                      choices=["low_risk", "balanced", "creative"],
                      help="smart variation mode (caps + value ceilings)")
    cc_b.add_argument("--run", action="store_true",
                      help="generate via Forge if reachable")

    # Study Intelligence Layer

    p_cb = sub.add_parser("combo", help="One-click character+style+outfit merge")
    p_cb.add_argument("--character", required=True)
    p_cb.add_argument("--style", default="")
    p_cb.add_argument("--outfit", default="")
    p_cb.add_argument("--name", default="combo")
    p_cb.add_argument("--char-weight", type=float, default=1.0)
    p_cb.add_argument("--style-weight", type=float, default=0.4)
    p_cb.add_argument("--outfit-weight", type=float, default=0.6)
    p_cb.add_argument("--output", default=None)

    # NOTE: renamed from "package" to "package-legacy" to avoid argparse
    # conflict with the newer Kohya/Colab `package` subparser registered
    # below (which has a `kohya` subcommand and is what the tests cover).

    p_train = sub.add_parser(
        "train", help="Training recipe / stack planning (writes plans only; "
                      "never launches training)")
    train_sub = p_train.add_subparsers(dest="train_cmd")
    p_tp = train_sub.add_parser("plan", help="Build a training recipe + plan")
    p_tp.add_argument("--dataset", required=True)
    p_tp.add_argument("--package", default=None)
    p_tp.add_argument("--output", required=True)
    p_tp.add_argument("--target", default=None)
    p_tp.add_argument("--mode", choices=["lora", "dreambooth", "hybrid"],
                      default=None)
    p_tp.add_argument("--base-model-profile", default="cyberrealistic_pony")
    p_tp.add_argument("--apply", action="store_true",
                      help="write plan files (default dry-run: report only)")
    p_tp.add_argument("--json", action="store_true")
    p_ts = train_sub.add_parser("stack-plan", help="Aggregate recipes into a "
                                                   "LoRA stack plan")
    p_ts.add_argument("--recipes", nargs="+", required=True)
    p_ts.add_argument("--output", required=True)
    p_ts.add_argument("--base-model-profile", default="cyberrealistic_pony")
    p_ts.add_argument("--apply", action="store_true")
    p_ts.add_argument("--json", action="store_true")

