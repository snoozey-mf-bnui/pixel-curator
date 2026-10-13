"""Parser registration: studio commands."""

from __future__ import annotations

from ...config import DEFAULT_FPS_TIERS, DEFAULT_QUOTA, UI_PORT  # noqa: F401


def register(sub):
    p_bd = sub.add_parser("build", help="Recipe-driven versioned dataset build (Phase 4)")
    p_bd.add_argument("--recipe", required=True)
    p_bd.add_argument("--output", default=None)
    p_bd.add_argument("--note", default="")
    p_bd.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    p_bd.add_argument("--parent", type=int, default=None,
                      help="dataset version this build was derived/refined "
                           "from (records lineage for `history`)")
    p_bd.add_argument("--no-training-config", dest="training_config",
                      action="store_false", default=True,
                      help="skip the auto-generated training_config.toml/"
                           "train.sh beside the dataset (Phase 10)")

    p_pln = sub.add_parser("plan",
                           help="Preview frames-per-LoRA-type (available vs "
                                "selected) without writing anything")
    p_pln.add_argument("--recipe", action="append", default=[],
                       help="limit to these recipes (repeatable); default all")
    p_pln.add_argument("--output", default=None)

    p_lb = sub.add_parser("lookbook",
                          help="Identity x outfits x poses -> catalog contact "
                               "sheet (needs Forge)")
    p_lb.add_argument("--identity", required=True, help="identity LoRA stem")
    p_lb.add_argument("--outfits", required=True, help="comma list of outfit descriptions")
    p_lb.add_argument("--poses", default="standing")
    p_lb.add_argument("--weight", type=float, default=0.85)
    p_lb.add_argument("--extra-loras", default="",
                      help="stack trained concept LoRAs: "
                           "stem:weight,stem2:weight2 (e.g. "
                           "ohwx_outfit_v014-000006:0.5,ohwx_pose_v015-000005:0.4)")
    p_lb.add_argument("--scene", default="")
    p_lb.add_argument("--checkpoint", default="")
    p_lb.add_argument("--forge-url", default="http://127.0.0.1:7860")
    p_lb.add_argument("--seed", type=int, default=1234)
    p_lb.add_argument("--trigger", default="")
    p_lb.add_argument("--output", default=None)

    p_ff = sub.add_parser("face-fix", help="Face Consistency Studio: restore / "
                          "identity-align your OWN generated images (local-only, "
                          "no model downloads)")
    p_ff.add_argument("action", nargs="?", choices=["check", "run"], default="run",
                      help="'check' reports backend availability; 'run' (default) "
                           "processes images")
    p_ff.add_argument("--target", default="",
                      help="a generated image OR a folder of generated images")
    p_ff.add_argument("--source-face", default="",
                      help="consenting high-quality reference face image")
    p_ff.add_argument("--output", default="",
                      help="output dir (default: <target_parent>/face_fixed)")
    p_ff.add_argument("--batch", action="store_true", help="force batch (folder) mode")
    p_ff.add_argument("--mode", choices=["auto", "single", "batch"], default="auto")
    p_ff.add_argument("--face-model", default="")
    p_ff.add_argument("--gfpgan-model", default="")
    p_ff.add_argument("--realesrgan-model", default="")
    p_ff.add_argument("--gfpgan-strength", type=float, default=0.30)
    p_ff.add_argument("--realesrgan-tile", type=int, default=0)
    p_ff.add_argument("--realesrgan-scale", type=int, default=2)
    p_ff.add_argument("--no-gfpgan", dest="ff_gfpgan", action="store_false", default=True)
    p_ff.add_argument("--no-realesrgan", dest="ff_realesrgan", action="store_false",
                      default=True)
    p_ff.add_argument("--no-identity-transfer", dest="ff_identity",
                      action="store_false", default=True)
    p_ff.add_argument("--overwrite", action="store_true")
    p_ff.add_argument("--dry-run", action="store_true")
    p_ff.add_argument("--json", action="store_true")

    p_ex = sub.add_parser(
        "exp", help="Training experiment tracking: link trained models "
                    "(.safetensors) to dataset versions, params and notes")
    ex_sub = p_ex.add_subparsers(dest="exp_action", required=True)

    ex_reg = ex_sub.add_parser("register", help="Register an experiment")
    ex_reg.add_argument("--name", default="",
                        help="label (default: model file stem)")
    ex_reg.add_argument("--model", default=None,
                        help="path to the trained .safetensors")
    ex_reg.add_argument("--dataset", type=int, default=None,
                        help="dataset version it was trained on")
    ex_reg.add_argument("--dataset-dir", default=None,
                        help="dataset folder when unversioned/external")
    ex_reg.add_argument("--type", dest="model_type", default="lora",
                        help="lora | dreambooth | checkpoint | other")
    ex_reg.add_argument("--base-model", default=None)
    ex_reg.add_argument("--param", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="training parameter (repeatable), e.g. "
                             "--param lr=1e-4 --param epochs=10")
    ex_reg.add_argument("--note", default="")
    ex_reg.add_argument("--outcome", default=None,
                        help="e.g. keeper / testing / discarded")
    ex_reg.add_argument("--rating", type=int, default=None,
                        help="1-5")
    ex_reg.add_argument("--from-file", default=None,
                        help="prefill from an experiment_template.json "
                             "(written into dataset exports) or a "
                             "*.experiment.json sidecar")
    ex_reg.add_argument("--full-hash", action="store_true",
                        help="full sha256 of the model file (slow on "
                             "multi-GB checkpoints; default is a quick "
                             "fingerprint)")
    ex_reg.add_argument("--no-hash", dest="hash_model",
                        action="store_false", default=True)

    ex_ls = ex_sub.add_parser("list", help="List experiments")
    ex_ls.add_argument("--dataset", type=int, default=None)
    ex_ls.add_argument("--model", default=None,
                       help="substring of name or model path")
    ex_ls.add_argument("--outcome", default=None)
    ex_ls.add_argument("--since", default=None, metavar="YYYY-MM-DD")
    ex_ls.add_argument("--json", action="store_true")

    ex_sh = ex_sub.add_parser("show", help="Full detail + dataset lineage")
    ex_sh.add_argument("exp", help="experiment id or name")
    ex_sh.add_argument("--json", action="store_true")

    ex_no = ex_sub.add_parser("note", help="Append a timestamped note")
    ex_no.add_argument("exp", help="experiment id or name")
    ex_no.add_argument("text")

    ex_st = ex_sub.add_parser("set", help="Update fields of an experiment")
    ex_st.add_argument("exp", help="experiment id or name")
    ex_st.add_argument("--model", default=None)
    ex_st.add_argument("--dataset", type=int, default=None)
    ex_st.add_argument("--name", default=None)
    ex_st.add_argument("--outcome", default=None)
    ex_st.add_argument("--rating", type=int, default=None)
    ex_st.add_argument("--base-model", default=None)
    ex_st.add_argument("--type", dest="model_type", default=None)
    ex_st.add_argument("--param", action="append", default=[],
                       metavar="KEY=VALUE")
    ex_st.add_argument("--full-hash", action="store_true")

    ex_fr = ex_sub.add_parser(
        "from-run", help="Promote an internal training run (see `runs`) "
                         "into an experiment record")
    ex_fr.add_argument("run_id", type=int)
    ex_fr.add_argument("--note", default="")

    p_ga = sub.add_parser(
        "generate-artistic",
        help="Artistic Figure Study Layer: compose an identity-preserving "
             "prompt recipe + LoRA stack (payload builder / Playground "
             "handoff)")
    p_ga.add_argument("--recipe", default="",
                      help="artistic recipe name (use --list to see all)")
    p_ga.add_argument("--list", action="store_true", dest="list_recipes",
                      help="list available artistic recipes and exit")
    p_ga.add_argument("--strength", type=float, default=1.0,
                      help="global concept-layer scale (identity excluded)")
    p_ga.add_argument("--identity-strength", type=float, default=None)
    p_ga.add_argument("--concept-strength", type=float, default=None)
    p_ga.add_argument("--seed", type=int, default=42)
    p_ga.add_argument("--batch", type=int, default=1,
                      help="plan a controlled-variation batch of N items "
                           "(writes a batch manifest; Phase 4)")
    p_ga.add_argument("--variation", default="",
                      help="comma-separated variation axes for --batch")
    p_ga.add_argument("--image", default="",
                      help="source image -> selective region editing "
                           "payload instead of txt2img")
    p_ga.add_argument("--workflow", default="garment_variation",
                      choices=["garment_variation", "sheer_garment_rendering",
                               "skin_texture_highlight_refinement",
                               "form_proportion_study",
                               "editorial_pose_variation"],
                      help="selective region workflow (with --image)")
    p_ga.add_argument("--identity-reference", default="",
                      help="reference image for FaceID identity guidance")
    p_ga.add_argument("--pose-guidance", action="store_true")
    p_ga.add_argument("--dry-run", action="store_true",
                      help="compose + report only (no payload build)")
    p_ga.add_argument("--json", action="store_true",
                      help="emit the full payload as JSON")
    p_ga.add_argument("--send-to-playground", action="store_true",
                      help="write the stack as a Playground preset")
    p_ga.add_argument("--output", default=None)

    p_tg = sub.add_parser("testgen", help="Single test generation (Phase 6)")
    p_tg.add_argument("--prompt", required=True)
    p_tg.add_argument("--negative", default="")
    p_tg.add_argument("--lora", action="append", default=[],
                      metavar="PATH:WEIGHT", help="repeatable; negative weights ok")
    p_tg.add_argument("--backend", choices=["forge", "diffusers"], default="forge")
    p_tg.add_argument("--forge-url", default="http://127.0.0.1:7860")
    p_tg.add_argument("--checkpoint", default="", help="diffusers backend base model")
    p_tg.add_argument("--init-image", default="", help="img2img / inpaint source")
    p_tg.add_argument("--region", default="",
                      help="region preset for inpainting (garment/torso/lower/"
                           "full/face); requires --init-image")
    p_tg.add_argument("--mask", default="",
                      help="custom mask PNG (white = edit area); default: "
                           "auto region box")
    p_tg.add_argument("--face-swap", action="store_true",
                      help="lock the generated face to --reference-image")
    p_tg.add_argument("--reference-image", default="",
                      help="identity reference for face swap / IP-Adapter")
    p_tg.add_argument("--upscale", action="store_true",
                      help="post-process via Forge extras (RealESRGAN x4)")
    p_tg.add_argument("--restore-faces", action="store_true",
                      help="post-process via Forge extras (GFPGAN)")
    p_tg.add_argument("--cn-openpose", default="", help="ControlNet pose ref image")
    p_tg.add_argument("--cn-depth", default="", help="ControlNet depth ref image")
    p_tg.add_argument("--cn-canny", default="", help="ControlNet canny ref image")
    p_tg.add_argument("--cn-softedge", default="", help="ControlNet softedge ref image")
    p_tg.add_argument("--sampler", default="")
    p_tg.add_argument("--clip-skip", type=int, default=0)
    p_tg.add_argument("--strength", type=float, default=0.6)
    p_tg.add_argument("--steps", type=int, default=28)
    p_tg.add_argument("--cfg", type=float, default=6.0)
    p_tg.add_argument("--seed", type=int, default=42)
    p_tg.add_argument("--width", type=int, default=1024)
    p_tg.add_argument("--height", type=int, default=1024)
    p_tg.add_argument("--out", default="testgen.png",
                      help="extra copy destination (generation is always "
                           "saved under <output>/evals/_single/)")
    p_tg.add_argument("--output", default=None)

    p_pl = sub.add_parser(
        "pipeline",
        help="Pipeline orchestrator: declarative recipes (curation/"
             "selection/complete or [pipelines.*] in the TOML) plus the "
             "legacy fixed DAG (full/mega)")
    p_pl.add_argument("action", choices=["run", "resume", "status",
                                         "plan", "list"])
    p_pl.add_argument("what", nargs="?", default="full",
                      help="pipeline name: a recipe (see `pipeline list`) "
                           "or legacy 'full'/'mega'")
    p_pl.add_argument("--recipe", default="character_v1",
                      help="(legacy full/mega) training dataset recipe")
    p_pl.add_argument("--preset", default="character",
                      help="(legacy full/mega) training preset")
    p_pl.add_argument("--gate", action="append", default=[],
                      help="pause after this stage (repeatable)")
    p_pl.add_argument("--no-smart", dest="smart", action="store_false", default=True)
    p_pl.add_argument("--no-cluster", dest="cluster", action="store_false", default=True)
    p_pl.add_argument("--backend", default="forge", choices=["forge", "diffusers"])
    p_pl.add_argument("--output", default=None)
    # Phase 7 recipe-engine controls (ignored by legacy full/mega)
    p_pl.add_argument("--stage", action="append", default=[], dest="only",
                      help="run only these stage(s) of the recipe "
                           "(repeatable)")
    p_pl.add_argument("--skip", action="append", default=[],
                      help="skip these stage(s) (repeatable)")
    p_pl.add_argument("--from", dest="from_stage", default=None,
                      help="start the recipe at this stage")
    p_pl.add_argument("--until", dest="until_stage", default=None,
                      help="stop the recipe after this stage")
    p_pl.add_argument("--set", action="append", default=[], dest="overrides",
                      metavar="STAGE.KEY=VALUE",
                      help="per-stage option override, e.g. "
                           "--set curate.min_sharpness=45 (repeatable)")
    p_pl.add_argument("--dry-run", action="store_true",
                      help="show the resolved plan without executing "
                           "(same as `pipeline plan <name>`)")

    p_studio = sub.add_parser(
        "studio",
        help="Studio Control Center: unified status, production plan, "
             "readiness scoring, asset registry and handoff export")
    studio_sub = p_studio.add_subparsers(dest="studio_cmd", required=False)
    sp_status = studio_sub.add_parser(
        "status", help="Studio Control Center summary")
    sp_status.add_argument("--json", action="store_true")
    sp_status.add_argument("--output", default=None)
    sp_plan = studio_sub.add_parser(
        "plan", help="Actionable production plan (alias of next)")
    sp_plan.add_argument("--json", action="store_true")
    sp_plan.add_argument("--output", default=None)
    sp_next = studio_sub.add_parser(
        "next", help="Recommended next actions with copy-paste commands")
    sp_next.add_argument("--json", action="store_true")
    sp_next.add_argument("--output", default=None)
    sp_ready = studio_sub.add_parser(
        "readiness", help="Readiness scoring for a dataset, run or batch")
    sp_ready.add_argument("--dataset", type=int, default=None)
    sp_ready.add_argument("--run", default="")
    sp_ready.add_argument("--batch", default="")
    sp_ready.add_argument("--json", action="store_true")
    sp_ready.add_argument("--output", default=None)
    sp_assets = studio_sub.add_parser(
        "assets", help="Unified studio asset registry")
    sp_assets.add_argument("--role", default="",
                           help="filter: identity_anchor | concept")
    sp_assets.add_argument("--missing-metadata", action="store_true")
    sp_assets.add_argument("--json", action="store_true")
    sp_assets.add_argument("--output", default=None)
    sp_export = studio_sub.add_parser(
        "export", help="Export a production handoff bundle")
    sp_export.add_argument("--run", default="")
    sp_export.add_argument("--batch", default="")
    sp_export.add_argument("--dataset", type=int, default=None)
    sp_export.add_argument("--include-samples", action="store_true")
    sp_export.add_argument("--dry-run", action="store_true")
    sp_export.add_argument("--json", action="store_true")
    sp_export.add_argument("--output", default=None)

    # ----- general export (manifest / selection) -----

    p_learn = sub.add_parser(
        "learn",
        help="Self-improvement loop: rate generations, learn a "
             "preference profile, recommend proven settings, harvest the "
             "best images for the next LoRA")
    learn_sub = p_learn.add_subparsers(dest="learn_cmd", required=False)
    lp_rate = learn_sub.add_parser("rate", help="Rate a generated image")
    lp_rate.add_argument("--image", default="")
    lp_rate.add_argument("--id", type=int, default=0, dest="feedback_id")
    lp_rate.add_argument("--rating", type=int, required=True,
                         help="-1 (down) or 1..5")
    lp_rate.add_argument("--tags", default="")
    lp_rate.add_argument("--output", default=None)
    lp_prof = learn_sub.add_parser("profile",
                                   help="Show the learned preference profile")
    lp_prof.add_argument("--json", action="store_true")
    lp_prof.add_argument("--rebuild", action="store_true",
                         help="recompute from feedback history")
    lp_prof.add_argument("--output", default=None)
    lp_rec = learn_sub.add_parser("recommend",
                                  help="Recommend proven settings")
    lp_rec.add_argument("--json", action="store_true")
    lp_rec.add_argument("--seed", type=int, default=0)
    lp_rec.add_argument("--output", default=None)
    lp_ing = learn_sub.add_parser(
        "ingest", help="Scan a folder of stored generations (e.g. Forge "
                       "txt2img-images) into the learning loop")
    lp_ing.add_argument("--dir", required=True,
                        help="folder of generated images to ingest")
    lp_ing.add_argument("--reference-image", default="",
                        help="source face for identity auto-scoring")
    lp_ing.add_argument("--recipe", default="")
    lp_ing.add_argument("--no-recursive", action="store_true")
    lp_ing.add_argument("--limit", type=int, default=0)
    lp_ing.add_argument("--json", action="store_true")
    lp_ing.add_argument("--output", default=None)
    lp_auto = learn_sub.add_parser(
        "auto", help="Unattended: ingest one or more folders + rebuild the "
                     "profile (cron/launchd-friendly, no interaction)")
    lp_auto.add_argument("--dir", action="append", default=[],
                         help="folder to ingest (repeatable)")
    lp_auto.add_argument("--reference-image", default="")
    lp_auto.add_argument("--json", action="store_true")
    lp_auto.add_argument("--output", default=None)
    lp_harv = learn_sub.add_parser(
        "harvest", help="Stage top-rated images as next-LoRA training data")
    lp_harv.add_argument("--min-rating", type=int, default=4)
    lp_harv.add_argument("--min-identity", type=float, default=0.6)
    lp_harv.add_argument("--apply", action="store_true",
                         help="copy the images (default: plan only)")
    lp_harv.add_argument("--json", action="store_true")
    lp_harv.add_argument("--output", default=None)

    p_rs = sub.add_parser("recommend-stack",
                          help="Smart Stack Recommendation Engine: "
                               "identity-preserving concept stack for a "
                               "dataset (--dataset), a trained run's best "
                               "epoch + stack (--run), or a figure-study "
                               "recipe layering plan (--recipe)")
    p_rs.add_argument("--dataset", type=int, default=None)
    p_rs.add_argument("--run", default="",
                      help="run name (or latest) -> best epoch + stack "
                           "recommendation (Phase 4)")
    p_rs.add_argument("--recipe", default="",
                      help="figure-study recipe -> recipe-driven layering "
                           "recommendation (Phase 4)")
    p_rs.add_argument("--identity-strength", type=float, default=None)
    p_rs.add_argument("--concept-strength", type=float, default=None)
    p_rs.add_argument("--strict", action="store_true",
                      help="strict negative intensity (with --recipe)")
    p_rs.add_argument("--json", action="store_true")
    p_rs.add_argument("--send-to-playground", action="store_true")
    p_rs.add_argument("--output", default=None)

    p_fs = sub.add_parser(
        "generate-figure-study",
        help="One-click artistic figure study: compose recipe + layered "
             "negative chains, plan a controlled-variation batch, write "
             "the batch manifest (Phase 4)")
    p_fs.add_argument("--recipe", default="artistic_figure_study",
                      help="artistic recipe (use generate-artistic --list)")
    p_fs.add_argument("--batch", type=int, default=1,
                      help="number of controlled-variation items (1-64)")
    p_fs.add_argument("--seed", type=int, default=42)
    p_fs.add_argument("--variation", default="",
                      help="comma-separated variation axes "
                           "(default: pose_energy,lighting_treatment,"
                           "camera_framing); use --list-axes to see all")
    p_fs.add_argument("--list-axes", action="store_true",
                      help="list variation axes + negative chains, exit")
    p_fs.add_argument("--list", action="store_true", dest="list_recipes",
                      help="list figure-study recipes and exit")
    # ----- Phase 5: multi-pass production + guidance planning -----
    p_fs.add_argument("--passes", default="",
                      help="production passes, comma-separated: "
                           "composition,pose_variation,garment_refine,"
                           "skin_refine,identity,review (aliases: "
                           "refine, pose, skin, garment)")
    p_fs.add_argument("--refine", action="store_true",
                      help="append refinement passes (garment + skin) "
                           "and a review pass to the plan")
    p_fs.add_argument("--refine-strength", type=float, default=0.0,
                      help="override refinement pass denoise")
    p_fs.add_argument("--select-best", action="store_true",
                      help="report the best-variant verdict for the "
                           "planned batch")
    p_fs.add_argument("--max-variants", type=int, default=0,
                      help="cap planned variants per generation pass")
    p_fs.add_argument("--controlnet", action="append", default=[],
                      choices=["openpose", "depth", "canny", "lineart",
                               "softedge"],
                      help="ControlNet guidance (repeatable): pose / "
                           "structure / edge stability")
    p_fs.add_argument("--ip-adapter", action="store_true",
                      help="IP-Adapter reference guidance")
    p_fs.add_argument("--faceid", action="store_true",
                      help="FaceID identity lock recommendation")
    p_fs.add_argument("--reference-image", default="",
                      help="reference image for FaceID / IP-Adapter "
                           "identity guidance")
    p_fs.add_argument("--pose-image", default="",
                      help="pose reference image for OpenPose guidance")
    p_fs.add_argument("--mask", default="",
                      help="custom mask image for region guidance")
    p_fs.add_argument("--strength", type=float, default=1.0)
    p_fs.add_argument("--identity-strength", type=float, default=None)
    p_fs.add_argument("--concept-strength", type=float, default=None)
    p_fs.add_argument("--negative-chains", default="",
                      help="extra negative control chains (comma-separated)")
    p_fs.add_argument("--strict", action="store_true",
                      help="strict negative intensity: emphasized "
                           "identity/anatomy/hand protection")
    p_fs.add_argument("--image", default="",
                      help="source image -> region-aware img2img payload")
    p_fs.add_argument("--region", default="full",
                      choices=["full", "torso", "lower", "hands",
                               "background"],
                      help="figure-study region policy (with --image)")
    p_fs.add_argument("--identity-reference", default="",
                      help="reference image for FaceID identity guidance")
    p_fs.add_argument("--denoise", type=float, default=0.0,
                      help="override region policy denoise (with --image)")
    p_fs.add_argument("--dry-run", action="store_true",
                      help="compose + report only (no files written)")
    p_fs.add_argument("--json", action="store_true")
    p_fs.add_argument("--send-to-playground", action="store_true")
    p_fs.add_argument("--output", default=None)

    p_fr = sub.add_parser(
        "figure-review",
        help="Closed-loop figure study batch review: summary, marking, "
             "best-variant selection, export (Phase 5)")
    p_fr.add_argument("--batch", default="",
                      help="batch id (directory name or unique substring)")
    p_fr.add_argument("--list", action="store_true", dest="list_batches",
                      help="list figure study batches and exit")
    p_fr.add_argument("--summary", action="store_true",
                      help="show the batch review summary")
    p_fr.add_argument("--select", default="", metavar="ITEM",
                      help="mark an item as the selected variant")
    p_fr.add_argument("--mark", nargs=2, default=None,
                      metavar=("ITEM", "STATUS"),
                      help="set an item's review status (selected, "
                           "needs_refinement, rejected, "
                           "strong_candidate, final_candidate)")
    p_fr.add_argument("--tag", action="append", default=[],
                      help="review tag to attach with --mark/--select "
                           "(repeatable)")
    p_fr.add_argument("--notes", default="",
                      help="review notes to attach with --mark/--select")
    p_fr.add_argument("--next-action", default="",
                      help="planned next action for the item")
    p_fr.add_argument("--best", action="store_true",
                      help="report the best-variant verdict")
    p_fr.add_argument("--export", action="store_true",
                      help="print the full review export payload (JSON)")
    p_fr.add_argument("--json", action="store_true")
    p_fr.add_argument("--output", default=None)

    p_pk = sub.add_parser(
        "package-figure-study",
        help="Production output packaging: gather manifests, prompts, "
             "guidance, review and assets for a batch (Phase 5)")
    p_pk.add_argument("--batch", required=True,
                      help="batch id (directory name or unique substring)")
    p_pk.add_argument("--json", action="store_true")
    p_pk.add_argument("--output", default=None)

    p_rfs = sub.add_parser(
        "render-figure-study",
        help="Render a planned figure-study batch through Forge into its "
             "samples/ folder (txt2img + img2img refinement passes)")
    p_rfs.add_argument("--batch", required=True,
                       help="batch id (directory name or unique substring)")
    p_rfs.add_argument("--forge-url", default="http://127.0.0.1:7860")
    p_rfs.add_argument("--checkpoint", default="",
                       help="pin checkpoint (default: project base model)")
    p_rfs.add_argument("--clip-skip", type=int, default=2)
    p_rfs.add_argument("--steps", type=int, default=28)
    p_rfs.add_argument("--cfg", type=float, default=6.0)
    p_rfs.add_argument("--width", type=int, default=1024)
    p_rfs.add_argument("--height", type=int, default=1024)
    p_rfs.add_argument("--limit", type=int, default=0,
                       help="render only the first N items")
    p_rfs.add_argument("--overwrite", action="store_true",
                       help="re-render items that already have a sample")
    p_rfs.add_argument("--face-lock", action="store_true",
                       help="apply inswapper source face to each render")
    p_rfs.add_argument("--reference-image", default="",
                       help="source face image for --face-lock")
    p_rfs.add_argument("--output", default=None)

    sub.add_parser("analyze", help="Dataset analysis + LoRA type detection")

    p_ab = sub.add_parser("ab", help="Same-seed A/B compare of two LoRAs")
    p_ab.add_argument("--a", required=True, metavar="NAME:W")
    p_ab.add_argument("--b", required=True, metavar="NAME:W")
    p_ab.add_argument("--seed", type=int, default=4242)
    p_ab.add_argument("--tail", default="", help="composition descriptors")

    p_gd = sub.add_parser("golden", help="Golden prompt set regression "
                                         "for a LoRA")
    p_gd.add_argument("--lora", required=True)
    p_gd.add_argument("--weight", type=float, default=0.75)
    p_gd.add_argument("--previous", default="",
                      help="prior LoRA name to diff against")

    p_st = sub.add_parser(
        "study", help="Study Intelligence Layer: classify / report / "
                      "recipes / stack / presets")
    st_sub = p_st.add_subparsers(dest="study_cmd", required=True)
    st_c = st_sub.add_parser("classify",
                             help="Classify frames into study categories")
    st_c.add_argument("--rescan", action="store_true",
                      help="re-classify frames that already have labels")
    st_sub.add_parser("report", help="Study classification statistics")
    st_r = st_sub.add_parser("recipes",
                             help="Register the 4 study dataset recipes")
    st_r.add_argument("--apply", action="store_true",
                      help="persist the recipes into the project file")
    st_k = st_sub.add_parser("stack",
                             help="Recommend a study production stack")
    st_k.add_argument("--mode", default="character_figure_study")
    st_p = st_sub.add_parser("presets",
                             help="Write the study Playground preset pack")
    st_p.add_argument("--path", default="",
                      help="presets file (default: outputs/playground_presets.json)")

    p_wz = sub.add_parser("wizard", help="Creator wizard: typed build->train->eval->card")
    p_wz.add_argument("--type", dest="lora_type", default="auto",
                      choices=["auto", "character", "style", "outfit", "pose",
                               "detail", "explicit"])
    p_wz.add_argument("--trigger", default="")
    p_wz.add_argument("--class-word", default="")
    p_wz.add_argument("--name", default="")
    p_wz.add_argument("--no-train", dest="train", action="store_false", default=True)
    p_wz.add_argument("--no-matrix", dest="matrix", action="store_false", default=True)
    p_wz.add_argument("--backend", default="forge", choices=["forge", "diffusers"])
    p_wz.add_argument("--output", default=None)

    p_wf = sub.add_parser(
        "workflow", help="End-to-end studio pipeline orchestrator (dry-run "
                         "first; source media never modified; no training)")
    wf_sub = p_wf.add_subparsers(dest="workflow_cmd")
    for _wc in ("run", "plan", "resume", "status"):
        _p = wf_sub.add_parser(_wc)
        _p.add_argument("--config", default=None,
                        help="workflow_config.json (fields overridden by flags)")
        _p.add_argument("--project-root", default=".")
        _p.add_argument("--photos", nargs="*", default=[])
        _p.add_argument("--frames", nargs="*", default=[])
        _p.add_argument("--output", default=None)
        _p.add_argument("--targets", nargs="*",
                        default=["core_identity"])
        _p.add_argument("--identity-token", default="spikbnu person")
        _p.add_argument("--class-token", default="person")
        _p.add_argument("--base-model-profile",
                        default="CyberRealisticPony Personal Base")
        _p.add_argument("--face-det-size", type=int, default=640)
        _p.add_argument("--curation-workers", type=int, default=4)
        _p.add_argument("--apply", action="store_true")
        _p.add_argument("--dry-run", dest="dry_run", action="store_true",
                        default=False)
        _p.add_argument("--no-resume", dest="resume", action="store_false",
                        default=True)
        _p.add_argument("--overwrite", action="store_true")
        _p.add_argument("--strict-validation", action="store_true")
        _p.add_argument("--skip-orientation", action="store_true")
        _p.add_argument("--canonicalize-stills", action="store_true")
        _p.add_argument("--skip-curation", action="store_true")
        _p.add_argument("--skip-validation", action="store_true")
        _p.add_argument("--skip-export", action="store_true")
        _p.add_argument("--with-anchor-rebuild", action="store_true")
        _p.add_argument("--skip-packaging", action="store_true")
        _p.add_argument("--skip-training-plans", action="store_true")
        _p.add_argument("--json", action="store_true")

