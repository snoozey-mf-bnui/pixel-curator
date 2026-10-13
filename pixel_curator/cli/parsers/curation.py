"""Parser registration: curation commands."""

from __future__ import annotations

from ...config import DEFAULT_FPS_TIERS, DEFAULT_QUOTA, UI_PORT  # noqa: F401


def register(sub):
    p_ds = sub.add_parser(
        "diverse-select", help="Plan a diversity-balanced selection (caps per "
                               "session & scene, balances framing) for a "
                               "flexible base - prevents one session baking in")
    p_ds.add_argument("--count", type=int, default=1500)
    p_ds.add_argument("--max-source-frac", type=float, default=0.15,
                      help="max fraction of the pick from any one source/session")
    p_ds.add_argument("--max-cluster-frac", type=float, default=0.20,
                      help="max fraction from any one visual cluster/scene")
    p_ds.add_argument("--min-identity", type=float, default=0.0)
    p_ds.add_argument("--min-sharpness", type=float, default=0.0)
    p_ds.add_argument("--json", action="store_true")
    p_ds.add_argument("--output", default=None)

    p_q = sub.add_parser(
        "quality", help="Deep quality signals: face-region sharpness, detail "
                        "retention (over-smoothing), composition, artifacts + "
                        "dataset-level consistency/pose diversity")
    p_q.add_argument("--limit", type=int, default=0)
    p_q.add_argument("--workers", type=int, default=1,
                     help="parallel scoring processes (plain CPU path only; "
                          "face/advanced passes stay sequential)")
    p_q.add_argument("--with-face", action="store_true",
                     help="re-detect faces to also populate eyes-open, "
                          "frontalness and pose bucket (needs [ai] extras)")
    p_q.add_argument("--with-advanced-quality", action="store_true",
                     help="compute ALL optional advanced signals "
                          "(figure/region/coverage/texture)")
    p_q.add_argument("--with-figure-quality", action="store_true")
    p_q.add_argument("--with-region-quality", action="store_true")
    p_q.add_argument("--with-coverage", action="store_true")
    p_q.add_argument("--with-natural-texture", action="store_true")
    # batch 2 (image-based work now; keypoint-based skip cleanly w/o keypoints)
    p_q.add_argument("--with-advanced-aesthetic", action="store_true",
                     help="aesthetic + skin_rendering + expression_quality + lighting")
    p_q.add_argument("--with-aesthetic", action="store_true")
    p_q.add_argument("--with-skin-rendering", action="store_true")
    p_q.add_argument("--with-expression-quality", action="store_true")
    p_q.add_argument("--with-lighting", action="store_true")
    p_q.add_argument("--with-dynamic-pose", action="store_true",
                     help="needs body keypoints; skips cleanly otherwise")
    p_q.add_argument("--with-body-symmetry", action="store_true",
                     help="needs body keypoints; skips cleanly otherwise")
    # batch 3 (composition: color mood / background / motion blur / occlusion;
    # body_proportion is geometry-based and skips cleanly without keypoints)
    p_q.add_argument("--with-advanced-composition", action="store_true",
                     help="color_mood + background_quality + motion_blur + face_occlusion")
    p_q.add_argument("--with-color-mood", action="store_true")
    p_q.add_argument("--with-background-quality", action="store_true")
    p_q.add_argument("--with-motion-blur", action="store_true")
    p_q.add_argument("--with-face-occlusion", action="store_true")
    p_q.add_argument("--with-body-proportion", action="store_true",
                     help="needs body keypoints; skips cleanly otherwise")
    # priority 6 (region-based readiness: aggregates existing region signals)
    p_q.add_argument("--with-region-readiness", action="store_true",
                     help="face/torso/full-body region readiness + balance + "
                          "best-training-use hint (reuses computed signals)")
    # priority 7 (coverage / framing intensity classification)
    p_q.add_argument("--with-coverage-intensity", "--with-framing-intensity",
                     dest="with_coverage_intensity", action="store_true",
                     help="classify coverage/framing intensity (full_body / "
                          "torso_focus / close_crop ...) from computed signals")
    # priority 8 (per-frame body-pose signature; clustering is dataset-level)
    p_q.add_argument("--with-pose-duplicate-detection", "--with-pose-signatures",
                     dest="with_pose_duplicate", action="store_true",
                     help="compute a per-frame body-pose signature for "
                          "downstream pose near-duplicate detection")
    # batch 4 (curation: clothing / full body / color cast / exposure;
    # eye_openness needs landmarks, duplicate needs a reference hash -> skip clean)
    p_q.add_argument("--with-advanced-curation", action="store_true",
                     help="clothing_type + full_body + color_cast + exposure_outlier")
    p_q.add_argument("--with-clothing-type", action="store_true")
    p_q.add_argument("--with-full-body", action="store_true")
    p_q.add_argument("--with-eye-openness", action="store_true",
                     help="needs eye landmarks; skips cleanly otherwise")
    p_q.add_argument("--with-color-cast", action="store_true")
    p_q.add_argument("--with-duplicate-detection", action="store_true",
                     help="records dhash awareness; non-destructive")
    p_q.add_argument("--with-exposure-outlier", action="store_true")
    p_q.add_argument("--no-store", dest="store", action="store_false",
                     default=True, help="report only, don't write frame_quality")
    p_q.add_argument("--resume", action="store_true",
                     help="skip frames already scored at the current signal "
                          "version (continue a Ctrl-C'd run where it left off)")
    p_q.add_argument("--no-progress", dest="progress", action="store_false",
                     default=True, help="disable the live progress line")
    p_q.add_argument("--json", action="store_true")
    p_q.add_argument("--output", default=None)

    p_cat = sub.add_parser(
        "categorize", help="Assign curated frames to non-character LoRA "
                           "categories (outfit / pose / detail) using framing + "
                           "advanced-quality signals; quality-ranked per category")
    p_cat.add_argument("--export", dest="export_dir", default=None,
                       help="write each category's ranked frames into "
                            "<dir>/<category>/ (+ manifest)")
    p_cat.add_argument("--count", type=int, default=0,
                       help="cap frames per category (0 = all)")
    p_cat.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"),
                       default="symlink",
                       help="how exported frames reference the source (default: "
                            "symlink; never moves the original)")
    p_cat.add_argument("--architecture", choices=("pony",), default=None,
                       help="also write per-image score-prefixed caption stubs")
    p_cat.add_argument("--apply", action="store_true",
                       help="actually write the export (default: dry-run plan)")
    p_cat.add_argument("--top", type=int, default=5,
                       help="examples shown per category in the report")
    p_cat.add_argument("--json", action="store_true")
    p_cat.add_argument("--output", default=None)

    p_of = sub.add_parser(
        "outfit", help="Outfit/accessory separation: tag frames by dominant "
                       "clothing colour + garments for clean outfit datasets")
    p_of.add_argument("--limit", type=int, default=0)
    p_of.add_argument("--no-store", dest="store", action="store_false", default=True)
    p_of.add_argument("--json", action="store_true")
    p_of.add_argument("--output", default=None)

    p_fr = sub.add_parser(
        "framing-report", help="Framing/coverage-level distribution "
                               "(closeup/portrait/upper/full body)")
    p_fr.add_argument("--output", default=None)

    p_an = sub.add_parser("anchor", help="Build identity anchor from reference images")
    p_an.add_argument("--anchor-dir", default=None,
                      help="Folder with 5-15 clear reference images of the subject")
    p_an.add_argument("--output", default=None)
    p_an.add_argument("--det-size", type=int, default=640)

    p_cu = sub.add_parser("curate", help="Score, filter and dedup frames")
    p_cu.add_argument("--list-scorers", action="store_true",
                      help="print the scorer registry with availability "
                           "and exit")
    p_cu.add_argument("--output", default=None)
    p_cu.add_argument("--hamming", type=int, default=4)
    p_cu.add_argument("--min-sharpness", type=float, default=35.0)
    p_cu.add_argument("--min-brightness", type=float, default=18.0)
    p_cu.add_argument("--max-brightness", type=float, default=242.0)
    p_cu.add_argument("--workers", type=int, default=8)
    p_cu.add_argument("--rescore", action="store_true")
    p_cu.add_argument("--no-text-check", dest="prefilter_text",
                      action="store_false", default=True,
                      help="skip the caption-bar/text-overlay detector that "
                           "piggybacks scoring (Phase 5 prefilter)")
    p_cu.add_argument("--no-auto-rotate", dest="auto_rotate",
                      action="store_false", default=True,
                      help="disable 90/180/270 orientation rescue")
    p_cu.add_argument("--orient-min-score", type=float, default=0.55,
                      help="confidence floor for orientation corrections "
                           "during the smart pass (same gate as `orient`)")
    p_cu.add_argument("--orient-min-axis-ratio", type=float, default=1.25,
                      help="eyes->mouth axis ambiguity gate for orientation "
                           "corrections (same gate as `orient`)")
    p_cu.add_argument("--smart", action="store_true",
                      help="Run face+identity filtering after the basic pass")
    p_cu.add_argument("--smart-only", action="store_true",
                      help="Skip basic pass; only run face+identity filtering")
    p_cu.add_argument("--identity-threshold", type=float, default=0.35)
    p_cu.add_argument("--min-face-area", type=float, default=0.015)
    p_cu.add_argument("--rescan", action="store_true",
                      help="Re-run smart scan over previously scanned frames")
    p_cu.add_argument("--metadata-dir", default=None,
                      help="Google Takeout sidecar JSON folder (subject face "
                           "tags rescue good frames at bad angles)")
    p_cu.add_argument("--subject", action="append", default=[], dest="subject_names",
                      help="Subject name as tagged by Google Photos "
                           "(repeatable); e.g. --subject Michelle_Huang")
    p_cu.add_argument("--sidecar-relax", type=float, default=0.10,
                      help="Lower the identity threshold by this much when the "
                           "subject's face tag confirms presence")
    p_cu.add_argument("--face-det-size", type=int, default=640,
                      help="SCRFD detector input size (320/480/512/640/768/"
                           "1024). Smaller = faster large-library passes; "
                           "640 default keeps current accuracy.")
    p_cu.add_argument("--profile-curation-performance", dest="profile_performance",
                      action="store_true",
                      help="collect per-stage timings + device/throughput "
                           "summary for the smart pass (default off)")

    p_cl = sub.add_parser("cluster", help="CLIP diversity clustering ([cluster] extras)")
    p_cl.add_argument("--output", default=None)
    p_cl.add_argument("--k", type=int, default=0, help="0 = auto")
    p_cl.add_argument("--batch-size", type=int, default=16)
    p_cl.add_argument("--reembed", action="store_true")

    p_rsc = sub.add_parser("rescore",
                           help="Fast: re-apply a rebuilt anchor to stored "
                                "embeddings (no face re-detection)")
    p_rsc.add_argument("--output", default=None)
    p_rsc.add_argument("--identity-threshold", type=float, default=0.35)
    p_rsc.add_argument("--metadata-dir", default=None)
    p_rsc.add_argument("--subject", action="append", default=[], dest="subject_names")
    p_rsc.add_argument("--sidecar-relax", type=float, default=0.10)

    sub.add_parser("datasets", help="List versioned dataset builds")

    # Phase X - Dataset Pipeline Explorer (read-focused data-lineage view)

    p_exp = sub.add_parser(
        "explorer", help="Dataset Pipeline Explorer: stages, image records, "
                         "stage comparison, image lineage (read-only)")
    exp_sub = p_exp.add_subparsers(dest="explorer_cmd")

    def _exp_common(p):
        p.add_argument("--output-base", default=None,
                       help="manifest output base (defaults to project output)")
        p.add_argument("--json", action="store_true")

    xp_sum = exp_sub.add_parser("summary", help="Pipeline stage summary")
    _exp_common(xp_sum)
    xp_sum.add_argument("--write-status", action="store_true",
                        help="also write legacy-UI status JSON snapshots")

    xp_stage = exp_sub.add_parser("stage", help="List image records for a stage")
    _exp_common(xp_stage)
    xp_stage.add_argument("--stage", default=None,
                          help="stage key (e.g. post_quality, post_smart_curation)")
    xp_stage.add_argument("--limit", type=int, default=100)
    xp_stage.add_argument("--offset", type=int, default=0)
    xp_stage.add_argument("--min-quality", type=float, default=None)
    xp_stage.add_argument("--max-quality", type=float, default=None)
    xp_stage.add_argument("--architecture", choices=("sdxl", "unknown"),
                          default=None)
    xp_stage.add_argument("--target", default=None,
                          help="target label (outfit/pose/detail/identity)")
    xp_stage.add_argument("--caption", default="", help="caption text search")
    xp_stage.add_argument("--source", default="", help="source folder/batch search")
    xp_stage.add_argument("--version", default=None, help="dataset version (vNNN)")
    xp_stage.add_argument("--cluster", default=None)
    xp_stage.add_argument("--framing", default=None)
    xp_stage.add_argument("--coverage", default=None)
    xp_stage.add_argument("--flag", action="append", default=[],
                          help="require flag (repeatable)")
    xp_stage.add_argument("--exclude-flag", action="append", default=[],
                          help="exclude flag (repeatable)")
    xp_stage.add_argument("--duplicate", choices=("duplicate", "unique"),
                          default=None)

    xp_cmp = exp_sub.add_parser("compare", help="Compare two pipeline stages")
    _exp_common(xp_cmp)
    xp_cmp.add_argument("--stage-a", required=True)
    xp_cmp.add_argument("--stage-b", required=True)

    xp_img = exp_sub.add_parser("image", help="Single-image detail + lineage")
    _exp_common(xp_img)
    xp_img.add_argument("--image", "--path", dest="image", required=True,
                        help="image path or frame id")

    xp_lin = exp_sub.add_parser("lineage", help="Image lineage only")
    _exp_common(xp_lin)
    xp_lin.add_argument("--image", "--path", dest="image", required=True)

    # Specialized Visual Focus Classifier

    p_vf = sub.add_parser(
        "visual-focus", help="Classify images by visual focus / detail / pose "
                            "for training-subset organization (read-only)")
    vf_sub = p_vf.add_subparsers(dest="vf_cmd")
    vf_cl = vf_sub.add_parser("classify", help="Classify a dataset directory")
    vf_cl.add_argument("--dataset", required=True)
    vf_cl.add_argument("--output", default=None)
    vf_cl.add_argument("--limit", type=int, default=None)
    vf_cl.add_argument("--use-vlm-refinement", action="store_true",
                       help="enable optional VLM refinement if configured")
    vf_cl.add_argument("--json", action="store_true")
    vf_im = vf_sub.add_parser("image", help="Classify a single image")
    vf_im.add_argument("--image", required=True)
    vf_im.add_argument("--use-vlm-refinement", action="store_true")
    vf_im.add_argument("--json", action="store_true")

    # Visual Focus Classification & Routing (metadata-only routing layer)

    p_cls = sub.add_parser(
        "classify", help="Visual focus classification & routing from "
                        "quality metadata (no model load)")
    cls_sub = p_cls.add_subparsers(dest="classify_cmd")
    cls_vf = cls_sub.add_parser(
        "visual-focus", help="Route a captions JSONL or dataset")
    cls_vf.add_argument("--input", default=None,
                        help="captions.jsonl (or any record JSONL)")
    cls_vf.add_argument("--dataset", default=None,
                        help="dir containing captions.jsonl")
    cls_vf.add_argument("--output", default=None)
    cls_vf.add_argument("--limit", type=int, default=None)
    cls_vf.add_argument("--overwrite", action="store_true")
    cls_vf.add_argument("--json", action="store_true")
    cls_im = cls_sub.add_parser(
        "visual-focus-image", help="Route a single image's metadata")
    cls_im.add_argument("--image", required=True)
    cls_im.add_argument("--captions-json",
                        dest="captions_json", default=None,
                        help="caption-record JSON for the image")
    cls_im.add_argument("--quality-json", default=None)
    cls_im.add_argument("--json", action="store_true")

    # Specialized Training Recipe Builder (visual-focus -> recipe plans)

    p_vz = sub.add_parser("viz", help="Embedding map: CLIP embeddings -> "
                          "2D projection + clusters (UI Viz tab / JSON)")
    p_vz.add_argument("--dataset", type=int, default=0,
                      help="dataset version (default: all selected frames)")
    p_vz.add_argument("--method", default="auto",
                      choices=["auto", "umap", "tsne", "pca"])
    p_vz.add_argument("--clusters", type=int, default=0,
                      help="cluster count (0 = automatic)")
    p_vz.add_argument("--recompute", action="store_true",
                      help="ignore the cached projection")
    p_vz.add_argument("--json", action="store_true")
    p_vz.add_argument("--export", default="",
                      help="write the viz payload JSON to a file")
    p_vz.add_argument("--output", default=None)


    p_in = sub.add_parser("insights", help="Quality Metrics Dashboard / "
                                           "dataset health / caption lint "
                                           "/ cluster naming")
    p_in.add_argument("--dataset", type=int, default=0,
                      help="scope the quality dashboard to a dataset "
                           "version (default: all curated frames)")
    p_in.add_argument("--json", action="store_true",
                      help="emit the dashboard as JSON (with per-image "
                           "metrics)")
    p_in.add_argument("--export", default="",
                      help="write the dashboard JSON to a file")
    in_sub = p_in.add_subparsers(dest="insights_cmd", required=False)
    in_sub.add_parser("health", help="Dataset health score + reasons")
    in_l = in_sub.add_parser(
        "lint", help="Active dataset lint: caption/detection contradictions, "
                     "orphan rows, missing files, count drift (report-only "
                     "by default; --fix applies safe, journaled fixes)")
    in_l.add_argument("--fix", action="store_true",
                      help="apply the safe auto-fixes (journaled to "
                           "lint_fixes/ for exact undo)")
    in_l.add_argument("--only", action="append", default=[],
                      help="restrict --fix to these issue codes (repeatable)")
    in_l.add_argument("--undo", default=None, metavar="JOURNAL",
                      help="reverse a previous --fix run from its journal")
    in_l.add_argument("--json", action="store_true")
    in_i = in_sub.add_parser(
        "intelligence",
        help="Holistic dataset analysis: usable-data estimate, curation "
             "strategy, training outlook (works on raw and curated data)")
    in_i.add_argument("--sample", type=int, default=0,
                      help="on a raw archive, score N random frames to "
                           "project reject rates (0 = DB-only)")
    in_i.add_argument("--json", action="store_true")
    in_sub.add_parser("clusters", help="Auto-name clusters from tags")
    in_c = in_sub.add_parser("collapse", help="Duplicate-pose collapse "
                                              "(near-identical frames)")
    in_c.add_argument("--apply", action="store_true")
    in_c.add_argument("--max-hamming", type=int, default=6)
    in_r = in_sub.add_parser("rotation", help="Detect/fix upside-down "
                                              "frames (landmark check)")
    in_r.add_argument("--apply", action="store_true")
    in_r.add_argument("--limit", type=int, default=0)

    p_dp = sub.add_parser("duplicates",
                          help="Duplicate & near-duplicate scan (phash + "
                               "optional CLIP embeddings) - flags and "
                               "groups for review, never deletes")
    p_dp.add_argument("--dataset", type=int, default=0,
                      help="scope to a dataset version (default: all "
                           "curated frames)")
    p_dp.add_argument("--method", default="phash",
                      choices=["phash", "clip", "both"])
    p_dp.add_argument("--max-hamming", type=int, default=6)
    p_dp.add_argument("--clip-threshold", type=float, default=0.97)
    p_dp.add_argument("--no-write", dest="write", action="store_false",
                      default=True, help="report only - do not store "
                                         "group fields in the manifest")
    p_dp.add_argument("--json", action="store_true")
    p_dp.add_argument("--output", default=None)

    p_ar = sub.add_parser(
        "anchors-rebuild", help="Score identity-anchor references and rebuild a "
                                "reviewed anchor set (dry-run by default; "
                                "source files are never modified)")
    p_ar.add_argument("--source", required=True,
                      help="folder of anchor reference images")
    p_ar.add_argument("--output", required=True,
                      help="anchor rebuild output folder")
    p_ar.add_argument("--apply", action="store_true",
                      help="copy images into accepted/review/rejected (default "
                           "dry-run writes report/manifest/card only)")
    p_ar.add_argument("--min-face-score", type=float, default=0.60)
    p_ar.add_argument("--min-subject-integrity", type=float, default=0.60)
    p_ar.add_argument("--min-face-region-score", type=float, default=0.60)
    p_ar.add_argument("--json", action="store_true")

