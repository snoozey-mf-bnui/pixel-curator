"""Argument parser construction for the pixel-curator CLI."""

from __future__ import annotations

import argparse

from ..config import (
    DEFAULT_QUOTA,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pixel-curator",
        description="pixel-curator - extraction, curation, dataset packaging",
    )
    parser.add_argument("--project", "-p", default=None,
                        help="Project file (.toml or .json). Default: built-in paths")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="suppress live progress lines (summaries, "
                             "warnings and errors still print)")
    parser.add_argument("--no-color", action="store_true",
                        help="disable ANSI colors (also honors NO_COLOR)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="enable debug logging (also honors "
                             "PIXEL_CURATOR_DEBUG=1)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="Check dependencies")
    sub.add_parser("stats", help="Show manifest statistics")

    p_mod = sub.add_parser(
        "models", help="List required AI models and download missing ones "
                       "(first-run setup)")
    p_mod.add_argument("models_cmd", nargs="?", default="status",
                       choices=("status", "download"),
                       help="status (default) or download")
    p_mod.add_argument("--only", default=None,
                       help="comma-separated model keys to download "
                            "(default: everything missing)")
    p_mod.add_argument("--json", action="store_true")

    p_ov = sub.add_parser(
        "overview", help="One-screen status: frame funnel, caption/quality "
                         "coverage, pending work + next commands, paused "
                         "pipelines (read-only)")
    p_ov.add_argument("--output", default=None)
    p_ov.add_argument("--json", action="store_true")

    p_val = sub.add_parser(
        "validate", help="Data doctor: read-only manifest/dataset integrity "
                         "checks (caption drift, orphans, dataset mismatch)")
    p_val.add_argument("--output", default=None)
    p_val.add_argument("--json", action="store_true",
                       help="Emit the structured report as JSON")

    p_st = sub.add_parser(
        "selftest", help="Run the standalone test suite (tests/run_all.py) "
                         "and report a single pass/fail board")
    p_st.add_argument("filters", nargs="*",
                      help="substring filters on test filenames")
    p_st.add_argument("--timeout", type=int, default=300)

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

    p_or = sub.add_parser(
        "orient", help="Accurate EXIF-safe orientation fix: confidence-gated, "
                       "verify-after, writes a JSON manifest for exact undo")
    p_or.add_argument("--dir", default=None,
                      help="operate on images in this folder (default: manifest "
                           "selected/packaged frames)")
    p_or.add_argument("--apply", action="store_true",
                      help="actually rotate (default: dry-run)")
    p_or.add_argument("--undo", default=None, metavar="MANIFEST.json",
                      help="reverse a previous run exactly from its manifest")
    p_or.add_argument("--min-det-score", type=float, default=0.55,
                      help="reject face detections below this confidence")
    p_or.add_argument("--min-axis-ratio", type=float, default=1.25,
                      help="eyes->mouth axis must be this much more vertical than "
                           "horizontal (guards tilted faces)")
    p_or.add_argument("--no-verify", dest="verify", action="store_false",
                      default=True, help="skip the re-check after rotating")
    p_or.add_argument("--limit", type=int, default=0)
    p_or.add_argument("--output", default=None)

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

    p_kf = sub.add_parser(
        "keyframes", help="Collapse long-video frame floods to a temporally + "
                          "visually diverse subset (marks redundant frames)")
    p_kf.add_argument("--max-per-video", type=int, default=80)
    p_kf.add_argument("--min-spacing", type=float, default=2.0,
                      help="min ordinal spacing between kept frames")
    p_kf.add_argument("--min-quality", type=float, default=0.0)
    p_kf.add_argument("--apply", action="store_true",
                      help="mark redundant frames (default: dry-run)")
    p_kf.add_argument("--output", default=None)

    p_q = sub.add_parser(
        "quality", help="Deep quality signals: face-region sharpness, detail "
                        "retention (over-smoothing), composition, artifacts + "
                        "dataset-level consistency/pose diversity")
    p_q.add_argument("--limit", type=int, default=0)
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

    p_init = sub.add_parser("init", help="Write a starter project file")
    p_init.add_argument("path", nargs="?", default="project.toml")

    p_ex = sub.add_parser("extract", help="Extract frames from videos")
    p_ex.add_argument("--video-dir", action="append", default=[])
    p_ex.add_argument("--photos-dir", default=None)
    p_ex.add_argument("--output", default=None)
    p_ex.add_argument("--fps", type=float, default=0.25,
                      help="Fallback rate when a clip's duration is unknown")
    p_ex.add_argument("--adaptive", action="store_true", default=True,
                      help="Duration-tiered sampling: live/short/medium/long/"
                           "extra-long clips each get their own fps (default on)")
    p_ex.add_argument("--no-adaptive", dest="adaptive", action="store_false",
                      help="Use a uniform --fps for every clip (legacy)")
    # Per-tier sample rates (frames/sec). --short-clip-fps kept as an alias for
    # the live tier so older commands keep working.
    p_ex.add_argument("--live-fps", "--short-clip-fps", dest="live_fps",
                      type=float, default=10.0,
                      help="Live Photo clips (<= --live-max-seconds)")
    p_ex.add_argument("--short-fps", type=float, default=4.0,
                      help="short clips up to --short-max-seconds")
    p_ex.add_argument("--medium-fps", type=float, default=2.0,
                      help="medium clips up to --medium-max-seconds")
    p_ex.add_argument("--long-fps", type=float, default=1.0,
                      help="long clips up to --long-max-seconds")
    p_ex.add_argument("--xlong-fps", type=float, default=1.0,
                      help="extra-long clips beyond --long-max-seconds "
                           "(e.g. a 45-min capture at ~1 fps)")
    # Tier boundaries (seconds).
    p_ex.add_argument("--live-max-seconds", type=float, default=6.0)
    p_ex.add_argument("--short-max-seconds", type=float, default=60.0)
    p_ex.add_argument("--medium-max-seconds", type=float, default=300.0)
    p_ex.add_argument("--long-max-seconds", type=float, default=1200.0)
    p_ex.add_argument("--max-frames-per-video", type=int, default=0,
                      help="Global per-clip frame ceiling override "
                           "(0 = use each tier's own cap)")
    p_ex.add_argument("--min-frames-per-video", type=int, default=4,
                      help="Per-clip frame floor for very short clips (0 = none)")
    p_ex.add_argument("--jpeg-quality", type=int, default=2)
    p_ex.add_argument("--segment-seconds", type=int, default=300)
    p_ex.add_argument("--max-side", type=int, default=0)
    p_ex.add_argument("--personfromvid", action="store_true")
    p_ex.add_argument("--no-photos", action="store_true")
    p_ex.add_argument("--photo-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    p_ex.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    p_ex.add_argument("--dry-run", action="store_true")
    p_ex.add_argument("--overwrite", action="store_true")
    p_ex.add_argument("--limit-videos", type=int, default=0)
    p_ex.add_argument("--no-prefilter", dest="prefilter",
                      action="store_false", default=True,
                      help="skip the zero-decode ingestion prefilter "
                           "(tiny/low-res/heavy-compression/truncated)")

    # Smart NSFW Extractor: same engine as `extract`, but reads the rich
    # [extraction] section from an NSFW TOML (pixel-curator_nsfw_v2.toml).
    # Use with: python -m pixel_curator -p pixel-curator_nsfw_v2.toml smart-extract
    p_se = sub.add_parser(
        "smart-extract",
        help="NSFW extraction driven by [extraction] section of the project "
             "TOML (same engine as `extract`; reads fps_strategy, hdr, "
             "resolution, deduplication, flags)")
    p_se.add_argument("--dry-run", action="store_true",
                      help="plan only; no frames written")
    p_se.add_argument("--overwrite", action="store_true",
                      help="re-extract from scratch (deletes prior output dirs)")
    p_se.add_argument("--limit-videos", type=int, default=0,
                      help="only process the first N discovered videos (0 = all)")

    p_ph = sub.add_parser("photos", help="Import photos only")
    p_ph.add_argument("--photos-dir", default=None)
    p_ph.add_argument("--output", default=None)
    p_ph.add_argument("--photo-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    p_ph.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    p_ph.add_argument("--dry-run", action="store_true")
    p_ph.add_argument("--no-prefilter", dest="prefilter",
                      action="store_false", default=True,
                      help="skip the zero-decode ingestion prefilter")

    # Phase 5: early pre-filtering over already-registered 'new' frames -
    # for archives ingested before the prefilter existed, or for re-running
    # with different thresholds. Fully reversible (--restore).
    p_pf = sub.add_parser(
        "prefilter",
        help="Fast junk rejection over 'new' frames (tiny files, low-res, "
             "heavy compression, truncated JPEGs, caption-bar screenshots)")
    p_pf.add_argument("--output", default=None)
    p_pf.add_argument("--min-bytes", type=int, default=12_000,
                      help="file-size floor (bytes)")
    p_pf.add_argument("--min-side", type=int, default=256,
                      help="shortest image side floor (px, header read only)")
    p_pf.add_argument("--min-bpp", type=float, default=0.05,
                      help="bytes-per-pixel floor for JPEG/WebP "
                           "(heavy social re-compression sits below)")
    p_pf.add_argument("--no-text-check", dest="text_check",
                      action="store_false", default=True,
                      help="skip the caption-bar/text-overlay detector")
    p_pf.add_argument("--no-decode", dest="decode_checks",
                      action="store_false", default=True,
                      help="zero-decode checks only (no text/blur pass)")
    p_pf.add_argument("--extreme-blur", type=float, default=5.0,
                      help="Laplacian-variance floor on the 96x96 gray "
                           "(0 = disable; far below curate's 35)")
    p_pf.add_argument("--workers", type=int, default=8)
    p_pf.add_argument("--dry-run", action="store_true",
                      help="report what would be rejected; write nothing")
    p_pf.add_argument("--restore", action="store_true",
                      help="flip prefilter-rejected frames back to 'new'")
    p_pf.add_argument("--reason", default=None,
                      help="with --restore: only this rejection reason "
                           "(e.g. text_overlay)")

    p_an = sub.add_parser("anchor", help="Build identity anchor from reference images")
    p_an.add_argument("--anchor-dir", default=None,
                      help="Folder with 5-15 clear reference images of the subject")
    p_an.add_argument("--output", default=None)
    p_an.add_argument("--det-size", type=int, default=640)

    p_cu = sub.add_parser("curate", help="Score, filter and dedup frames")
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

    p_ca = sub.add_parser("caption", help="WD14 auto-captioning ([caption] extras)")
    p_ca.add_argument("--output", default=None)
    p_ca.add_argument("--trigger", default=None)
    p_ca.add_argument("--class-word", default=None)
    p_ca.add_argument("--threshold", type=float, default=0.35)
    p_ca.add_argument("--char-threshold", type=float, default=0.85)
    p_ca.add_argument("--pony-prefix", action="store_true",
                      help="Prepend score_9, score_8_up, score_7_up")
    p_ca.add_argument("--blacklist", default="", help="comma-separated tags to drop")
    p_ca.add_argument("--remap", default="", help="old:new,old2:new2")
    p_ca.add_argument("--prune", default="",
                      help="permanent-trait tags to absorb into the trigger")
    p_ca.add_argument("--max-tags", type=int, default=30)
    p_ca.add_argument("--force", action="store_true")
    p_ca.add_argument("--repo-id", default="SmilingWolf/wd-swinv2-tagger-v3")
    p_ca.add_argument("--tagger", default=None,
                      choices=("wd14", "mps"),
                      help="Tagging backend: wd14 (ONNX, current default) | "
                           "mps (same WD14-v3 weights on the Apple Silicon "
                           "GPU via torch+timm). Unavailable backends fall "
                           "back to wd14. "
                           "Unset -> project.toml default_tagger -> wd14.")
    p_ca.add_argument("--reprune", action="store_true",
                      help="Rebuild caption_text from stored tags (no tagger, "
                           "no re-inference). Dry-run unless --apply is given.")
    p_ca.add_argument("--apply", action="store_true",
                      help="With --reprune: write the rebuilt captions to the DB.")

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

    p_hist = sub.add_parser(
        "history", help="Dataset version history: every registered version "
                        "(builds + folder exports) with lineage, change "
                        "summaries and curation snapshots")
    p_hist.add_argument("--dataset", type=int, default=None,
                        help="show the lineage chain of one version")
    p_hist.add_argument("--name", default=None,
                        help="filter by recipe / export name")
    p_hist.add_argument("--curation", action="store_true",
                        help="with --dataset: print the curation snapshot "
                             "captured when that version was created")
    p_hist.add_argument("--json", action="store_true")
    p_hist.add_argument("--output", default=None)

    p_pln = sub.add_parser("plan",
                           help="Preview frames-per-LoRA-type (available vs "
                                "selected) without writing anything")
    p_pln.add_argument("--recipe", action="append", default=[],
                       help="limit to these recipes (repeatable); default all")
    p_pln.add_argument("--output", default=None)

    p_hl = sub.add_parser("health",
                          help="Read-only dataset health report (HTML + JSON)")
    p_hl.add_argument("--output", default=None)
    p_hl.add_argument("--open", action="store_true", help="open the HTML report")

    p_rsc = sub.add_parser("rescore",
                           help="Fast: re-apply a rebuilt anchor to stored "
                                "embeddings (no face re-detection)")
    p_rsc.add_argument("--output", default=None)
    p_rsc.add_argument("--identity-threshold", type=float, default=0.35)
    p_rsc.add_argument("--metadata-dir", default=None)
    p_rsc.add_argument("--subject", action="append", default=[], dest="subject_names")
    p_rsc.add_argument("--sidecar-relax", type=float, default=0.10)

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

    sub.add_parser("recipes", help="List recipes defined in the project file")
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

    p_df = sub.add_parser("diff", help="Diff two dataset versions "
                          "(membership + quality/duplicate/cluster deltas)")
    p_df.add_argument("a", type=int, nargs="?", default=None)
    p_df.add_argument("b", type=int, nargs="?", default=None)
    p_df.add_argument("--dataset", type=int, default=0,
                      help="new dataset version (visual diff mode)")
    p_df.add_argument("--vs", type=int, default=0,
                      help="baseline dataset version to compare against")
    p_df.add_argument("--json", action="store_true")
    p_df.add_argument("--export", default="",
                      help="write the structured diff JSON to a file")
    p_df.add_argument("--output", default=None)
    p_df.add_argument("--full", action="store_true", help="list every changed frame")

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

    from ..train.presets import PRESETS as _PRESETS
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

    sub.add_parser("runs", help="List training runs")

    # Phase 8 - Training Experiment Tracking
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

    p_su = sub.add_parser("status", help="Live training status: step/ETA/"
                          "loss/disk/checkpoints (reads the manifest; safe "
                          "to run while training)")
    p_su.add_argument("--json", action="store_true")
    p_su.add_argument("--follow", type=float, default=0, metavar="SECONDS",
                      help="poll every N seconds; macOS-notifies on new "
                           "epoch checkpoints, low disk, stalls and completion")
    p_su.add_argument("--no-notify", dest="notify", action="store_false",
                      default=True)
    p_su.add_argument("--stall-seconds", type=float, default=900.0,
                      help="alert when a 'running' run records no new step for "
                           "this many seconds (default 900 = 15 min)")
    p_su.add_argument("--output", default=None)

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

    p_wa = sub.add_parser("watch", help="Watch source folders; auto-ingest new media")
    p_wa.add_argument("--interval", type=int, default=300)
    p_wa.add_argument("--auto-curate", action="store_true")
    p_wa.add_argument("--once", action="store_true")
    p_wa.add_argument("--output", default=None)

    p_gc = sub.add_parser("gc", help="Disk cleanup (dry-run by default)")
    p_gc.add_argument("--apply", action="store_true")
    p_gc.add_argument("--keep-builds", type=int, default=2)
    p_gc.add_argument("--output", default=None)

    sub.add_parser("space", help="Disk usage dashboard")

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
    p_ft = sub.add_parser(
        "filter", help="Semantic text filter: keep/drop frames matching a "
                       "natural-language query (CLIP; run 'cluster' first)")
    p_ft.add_argument("--text", required=True,
                      help="e.g. 'beach photos' or 'images with text'")
    p_ft.add_argument("--threshold", type=float, default=0.25,
                      help="cosine similarity cutoff (default 0.25)")
    p_ft.add_argument("--invert", action="store_true",
                      help="drop MATCHING frames instead of keeping them")
    p_ft.add_argument("--apply", action="store_true",
                      help="apply status changes (default: dry-run report)")
    p_ft.add_argument("--json", action="store_true")

    # ----- visual review round-trip -----
    p_rv = sub.add_parser(
        "review", help="Visual review: static HTML report + keep/reject "
                       "round-trip via review_sheet.csv")
    rv_sub = p_rv.add_subparsers(dest="review_cmd")
    p_rv_e = rv_sub.add_parser("export", help="Write index.html + sheet")
    p_rv_e.add_argument("--out", required=True, help="output folder")
    p_rv_e.add_argument("--top-pct", type=float, default=None)
    p_rv_e.add_argument("--min-score", type=float, default=None)
    p_rv_e.add_argument("--balanced", type=int, default=0, metavar="N")
    p_rv_e.add_argument("--max-cards", type=int, default=2000)
    p_rv_e.add_argument("--json", action="store_true")
    p_rv_i = rv_sub.add_parser("import", help="Apply keep/reject decisions")
    p_rv_i.add_argument("--sheet", required=True, help="review_sheet.csv")
    p_rv_i.add_argument("--apply", action="store_true",
                        help="commit status changes (default: dry-run)")
    p_rv_i.add_argument("--json", action="store_true")

    # ----- self-improvement loop -----
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

    sub.add_parser("analyze", help="Dataset analysis + LoRA type detection")

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

    p_pv = sub.add_parser("previews", help="Forge-render default/low/"
                                           "medium/high LoRA previews")
    p_pv.add_argument("--lora", required=True, help="LoRA path or stem")
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

    p_bk = sub.add_parser("backup", help="Archive manifest + presets + "
                                         "project file + LoRA sidecars")
    p_bk.add_argument("--out", default="")
    p_rs = sub.add_parser("restore", help="Extract a backup for manual "
                                          "placement (non-destructive)")
    p_rs.add_argument("archive")
    p_rs.add_argument("--dest", default="restored_backup")

    # Concept Control Layer
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

    return parser.parse_args(argv)


