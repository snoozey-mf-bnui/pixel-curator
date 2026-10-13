"""Parser registration: exporting commands."""

from __future__ import annotations

from ...config import DEFAULT_FPS_TIERS, DEFAULT_QUOTA, UI_PORT  # noqa: F401


def register(sub):
    p_db = sub.add_parser(
        "export-dreambooth", help="[legacy] Export a Colab-ready DreamBooth "
                                  "dataset. The defined DreamBooth export "
                                  "path is now the selector: python -m "
                                  "pixel_curator.curate.dreambooth_nsfw_selector")
    p_db.add_argument("--count", type=int, default=1000,
                      help="instance images to select (default 1000); ignored "
                           "in lineage mode")
    p_db.add_argument("--dataset", type=int, default=None,
                      help="LINEAGE: reproduce the exact frames + captions of "
                           "this dataset version")
    p_db.add_argument("--from-run", default=None,
                      help="LINEAGE: resolve the dataset version from a training "
                           "run name (post-training only)")
    p_db.add_argument("--from-recipe", default=None,
                      help="LINEAGE: resolve the latest built dataset version for "
                           "a recipe name (e.g. character_a) - use this to export "
                           "a built-but-not-yet-trained dataset")
    p_db.add_argument("--repeats", type=int, default=1)
    p_db.add_argument("--reg", action="store_true",
                      help="also scaffold a regularization (class) set")
    p_db.add_argument("--reg-count", type=int, default=300)
    p_db.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"],
                      default="symlink")
    p_db.add_argument("--no-diversity", dest="diversity", action="store_false",
                      default=True, help="disable cluster round-robin selection")
    p_db.add_argument("--quality-aware", action="store_true",
                      help="rank by advanced-quality signals (region readiness, "
                           "subject integrity, aesthetic; demotes blur / "
                           "occlusion / pose near-duplicates). Needs the "
                           "'quality --with-...' pass to have run.")
    p_db.add_argument("--diverse", action="store_true",
                      help="use the full diversity curator (caps per session & "
                           "scene, balances framing) - for a FLEXIBLE base")
    p_db.add_argument("--max-source-frac", type=float, default=0.15)
    p_db.add_argument("--max-cluster-frac", type=float, default=0.20)
    p_db.add_argument("--out", default=None, help="export root (default: "
                      "<output_base>/dreambooth_exports/<token>)")
    p_db.add_argument("--apply", action="store_true",
                      help="actually export (default: dry-run plan only)")
    p_db.add_argument("--output", default=None)

    p_mp = sub.add_parser(
        "multi-recipe-packages", help="Generate DreamBooth + specialized LoRA "
                                     "training packages from recipe routing "
                                     "(dry-run by default; --apply to write)")
    p_mp.add_argument("--input", default=None,
                      help="specialized_recipe_assignments.jsonl")
    p_mp.add_argument("--specialized-plan", default=None,
                      help="specialized_recipe_plan.json")
    p_mp.add_argument("--dataset", default=None,
                      help="dir containing specialized_recipes/ outputs")
    p_mp.add_argument("--output", default=None)
    p_mp.add_argument("--project", dest="project_toml", default=None)
    p_mp.add_argument("--architecture", choices=("sdxl", "unknown"),
                      default="sdxl")
    p_mp.add_argument("--dreambooth-min-suitability", type=float, default=75.0)
    p_mp.add_argument("--lora-min-suitability", type=float, default=55.0)
    p_mp.add_argument("--min-confidence", type=float, default=0.55)
    p_mp.add_argument("--min-images-per-package", type=int, default=10)
    p_mp.add_argument("--max-images-per-package", type=int, default=None)
    p_mp.add_argument("--all", dest="process_all", action="store_true",
                      help="Process the full manifest (paginate past the 500 "
                           "UI page cap) when falling back to the manifest.")
    p_mp.add_argument("--max-records", type=int, default=1_000_000,
                      help="Upper bound on records loaded with --all.")
    p_mp.add_argument("--link-mode", choices=("hardlink", "copy", "symlink"),
                      default="hardlink")
    p_mp.add_argument("--no-dreambooth-main", dest="include_dreambooth",
                      action="store_false", default=True)
    p_mp.add_argument("--no-lora-packages", dest="include_lora",
                      action="store_false", default=True)
    p_mp.add_argument("--allow-review-records", action="store_true")
    p_mp.add_argument("--overwrite", action="store_true")
    p_mp.add_argument("--apply", action="store_true",
                      help="materialize packages (default: dry-run plan only)")
    p_mp.add_argument("--register-version", action="store_true",
                      help="register each materialized package as a dataset "
                           "version (Phase 2 registry; requires --apply)")
    p_mp.add_argument("--parent", type=int, default=None,
                      help="parent dataset version for lineage tracking")
    p_mp.add_argument("--version-note", default="",
                      help="note stored with the registered version(s)")
    p_mp.add_argument("--json", action="store_true")

    p_xp = sub.add_parser(
        "export", help="General-purpose export: flat manifest (CSV/JSONL/"
        "Parquet) and score-based selection (copy or symlink)")
    xp2_sub = p_xp.add_subparsers(dest="export_cmd")
    p_xp_m = xp2_sub.add_parser(
        "manifest", help="Flat per-frame manifest for any tool")
    p_xp_m.add_argument("--out", required=True, help="output file path")
    p_xp_m.add_argument("--format", choices=["csv", "jsonl", "parquet"],
                        default="csv")
    p_xp_m.add_argument("--json", action="store_true")
    p_xp_s = xp2_sub.add_parser(
        "select", help="Export top-N%% / min-score selection to a folder")
    p_xp_s.add_argument("--out", required=True, help="output folder")
    p_xp_s.add_argument("--top-pct", type=float, default=None,
                        help="keep the best N percent by score")
    p_xp_s.add_argument("--min-score", type=float, default=None,
                        help="keep frames with unified score >= this (0..1)")
    p_xp_s.add_argument("--mode", choices=["symlink", "copy"],
                        default="symlink",
                        help="symlink = zero-copy culling of large libraries")
    p_xp_s.add_argument("--balanced", type=int, default=0, metavar="N",
                        help="pick N frames spread evenly across CLIP "
                             "clusters (run 'cluster' first)")
    p_xp_s.add_argument("--no-captions", dest="captions",
                        action="store_false", default=True)
    p_xp_s.add_argument("--json", action="store_true")

    # ----- semantic text filter -----

    p_pv = sub.add_parser("previews", help="Forge-render default/low/"
                                           "medium/high LoRA previews")
    p_pv.add_argument("--lora", required=True, help="LoRA path or stem")

    p_pk = sub.add_parser("package-legacy", help="Build kohya-style training dataset (legacy/manual)")
    p_pk.add_argument("--output", default=None)
    p_pk.add_argument("--token", default=None)
    p_pk.add_argument("--class-word", default=None)
    p_pk.add_argument("--repeats", type=int, default=10)
    p_pk.add_argument("--max-per-video", type=int, default=40)
    p_pk.add_argument("--max-total", type=int, default=0)
    p_pk.add_argument("--caption", default="")
    p_pk.add_argument("--no-captions", dest="captions", action="store_false", default=True)
    p_pk.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    p_pk.add_argument("--quota", default=DEFAULT_QUOTA,
                      help="framing quotas, e.g. 'closeup=0.3,portrait=0.3,"
                           "upper_body=0.25,full_body=0.15' (needs --max-total)")
    p_pk.add_argument("--no-quota", dest="quota", action="store_const", const="")
    p_pk.add_argument("--static-captions-only", dest="use_caption_table",
                      action="store_false", default=True)

    p_pkg = sub.add_parser(
        "package", help="Build a training-ready Kohya/Colab dataset package "
                        "from a curated export (dry-run by default)")
    pkg_sub = p_pkg.add_subparsers(dest="package_cmd")
    p_pk_k = pkg_sub.add_parser("kohya", help="Kohya / Colab package")
    p_pk_k.add_argument("--dataset", required=True)
    p_pk_k.add_argument("--output", required=True)
    p_pk_k.add_argument("--mode", choices=["dreambooth", "lora"],
                        default="dreambooth")
    p_pk_k.add_argument("--identity-token", default="spikbnu person")
    p_pk_k.add_argument("--class-token", default="person")
    p_pk_k.add_argument("--concept-name", default="core_identity")
    p_pk_k.add_argument("--create-zip", dest="create_zip", action="store_true",
                        default=True)
    p_pk_k.add_argument("--no-zip", dest="create_zip", action="store_false")
    p_pk_k.add_argument("--overwrite-captions", action="store_true")
    p_pk_k.add_argument("--max-caption-words", type=int, default=75)
    p_pk_k.add_argument("--resolution", type=int, default=1024)
    p_pk_k.add_argument("--repeats", type=int, default=1)
    p_pk_k.add_argument("--apply", action="store_true",
                        help="write the package (default dry-run: report only)")
    p_pk_k.add_argument("--json", action="store_true")
    p_pk_k.add_argument("--no-version", dest="register_version",
                        action="store_false", default=True,
                        help="skip registering this package in the dataset "
                             "version registry")
    p_pk_k.add_argument("--parent", type=int, default=None,
                        help="dataset version this package derives from "
                             "(records lineage)")
    p_pk_k.add_argument("--version-note", default="",
                        help="note stored with the version registry entry")
    p_pk_k.add_argument("--no-training-config", dest="training_config",
                        action="store_false", default=True,
                        help="skip the auto-generated training_config.toml/"
                             "train.sh beside the package (Phase 10)")

    p_dc = sub.add_parser(
        "dataset-card", help="Validate an exported dataset and (re)write its "
                             "rich dataset card + structured manifest (additive)")
    p_dc.add_argument("dataset_dir",
                      help="exported dataset directory containing "
                           "dataset_manifest.json")
    p_dc.add_argument("--manifest", default=None,
                      help="explicit manifest path (defaults to "
                           "<dataset_dir>/dataset_manifest.json)")
    p_dc.add_argument("--validate-dataset", action="store_true",
                      help="run validation and print the report")
    p_dc.add_argument("--strict-validation", action="store_true",
                      help="treat warnings as failing (non-zero exit)")
    p_dc.add_argument("--no-dataset-card", dest="write_dataset_card",
                      action="store_false", default=True,
                      help="validate only; do not write card files")
    p_dc.add_argument("--max-rows", type=int, default=500,
                      help="max frame-table rows in the markdown card")
    p_dc.add_argument("--json", action="store_true")

