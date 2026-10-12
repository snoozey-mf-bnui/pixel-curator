"""Command-line interface."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Optional

from . import manifest
from .config import (
    DB_NAME, DEFAULT_FPS_TIERS, DEFAULT_QUOTA, UI_PORT, CaptionConfig,
    ClusterConfig, CurateConfig, ExtractConfig, PackageConfig,
    PrefilterConfig, SmartCurateConfig, load_project, write_template,
)
from .curate import build_anchor, curate_generator, smart_curate_generator
from .curate.diversity import cluster_generator
from .caption import caption_generator
from .extract import pipeline_generator
from .packager import package_generator
from .progress import ProgressLine
from .util import check_dependencies


# ---------------------------------------------------------------------------
# Phase 9: update emitter - live progress on a TTY, quiet mode, light color
# ---------------------------------------------------------------------------

_ANSI = {"red": "\033[31m", "yellow": "\033[33m",
         "green": "\033[32m", "dim": "\033[2m", "reset": "\033[0m"}


class UpdateEmitter:
    """Prints generator updates.

    * :class:`ProgressLine` updates rewrite one live line on a TTY,
      print normally when piped, and vanish under ``--quiet``.
    * Warning / error / completion lines get minimal ANSI color on a
      TTY (suppressed by ``--no-color`` or the NO_COLOR env var).
    """

    def __init__(self, quiet: bool = False, color: Optional[bool] = None,
                 stream=None):
        import os
        self.out = stream or sys.stdout
        self.quiet = quiet
        tty = bool(getattr(self.out, "isatty", lambda: False)())
        self.tty = tty
        self.color = (tty and not os.environ.get("NO_COLOR")
                      if color is None else color)
        self._live = False       # a \r progress line is on screen

    def _colorize(self, text: str) -> str:
        if not self.color:
            return text
        probe = text.lstrip()
        if probe.startswith(("FATAL", "ERROR")):
            return _ANSI["red"] + text + _ANSI["reset"]
        if probe.startswith(("WARNING", "!")):
            return _ANSI["yellow"] + text + _ANSI["reset"]
        if ("COMPLETE" in probe[:60] or "COMPLETED" in probe[:60]
                or probe.startswith(("DONE", "OK "))
                or " complete:" in probe[:80]):
            return _ANSI["green"] + text + _ANSI["reset"]
        return text

    def emit(self, update) -> None:
        if isinstance(update, ProgressLine):
            if self.quiet:
                return
            if self.tty:
                print("\r\033[2K  " + str(update), end="",
                      file=self.out, flush=True)
                self._live = True
                return
            print(str(update), file=self.out)
            return
        if self._live:
            print(file=self.out)      # finish the live line first
            self._live = False
        print(self._colorize(str(update)), file=self.out)

    def close(self) -> None:
        if self._live:
            print(file=self.out)
            self._live = False


_EMITTER = UpdateEmitter()


def stream(gen) -> None:
    """Drain a stage generator through the configured emitter."""
    try:
        for update in gen:
            _EMITTER.emit(update)
    finally:
        _EMITTER.close()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
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
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="Check dependencies")
    sub.add_parser("stats", help="Show manifest statistics")

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

    p_vl = sub.add_parser(
        "vlm-caption", help="Dual captioning: add optional Florence-2/VLM natural-"
                            "language scene descriptions alongside WD14 tags")
    p_vl.add_argument("--model", default="microsoft/Florence-2-base")
    p_vl.add_argument("--compose", action="store_true",
                      help="merge the VLM description into the trained caption_text")
    p_vl.add_argument("--limit", type=int, default=0)
    p_vl.add_argument("--output", default=None)

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
                      choices=("wd14", "mps", "florence"),
                      help="Tagging backend: wd14 (ONNX, current default) | "
                           "mps (same WD14-v3 weights on the Apple Silicon "
                           "GPU via torch+timm) | florence (Florence-2 "
                           "prose). Unavailable backends fall back to wd14. "
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

    # Florence-2 natural-language captioning (optional external VLM backend)
    p_fl = sub.add_parser(
        "florence", help="Florence-2 natural-language captioning via the "
                        "external VLM env (optional; complements WD14 tags)")
    fl_sub = p_fl.add_subparsers(dest="fl_cmd")

    def _fl_runtime_args(p):
        p.add_argument("--model-variant", choices=("base-ft", "large-ft", "custom"),
                       default="base-ft")
        p.add_argument("--model-path", default="")
        p.add_argument("--vlm-python", default=None)
        p.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"),
                       default="auto")
        p.add_argument("--dtype", choices=("float32", "float16", "bfloat16", "auto"),
                       default="float32")
        p.add_argument("--local-files-only", dest="local_files_only",
                       action="store_true", default=True)
        p.add_argument("--allow-downloads", action="store_true", default=False)
        p.add_argument("--json", action="store_true")

    fl_chk = fl_sub.add_parser("check", help="Report Florence runtime readiness")
    _fl_runtime_args(fl_chk)

    fl_img = fl_sub.add_parser("image", help="Caption a single image")
    _fl_runtime_args(fl_img)
    fl_img.add_argument("--image", required=True)
    fl_img.add_argument("--task", default="<MORE_DETAILED_CAPTION>")
    fl_img.add_argument("--mode", default="caption",
                        choices=("caption", "dense_caption", "analysis", "routing"))
    fl_img.add_argument("--sidecars", action="store_true",
                        help="write <stem>.florence.json/.txt next to output")

    fl_bat = fl_sub.add_parser("batch", help="Caption a dataset directory")
    _fl_runtime_args(fl_bat)
    fl_bat.add_argument("--dataset", required=True)
    fl_bat.add_argument("--output", default=None)
    fl_bat.add_argument("--mode", default="caption",
                        choices=("caption", "dense_caption", "analysis", "routing"))
    fl_bat.add_argument("--limit", type=int, default=None)

    # Visual Focus Classification & Routing (metadata-only routing layer)
    p_cls = sub.add_parser(
        "classify", help="Visual focus classification & routing from Florence + "
                        "quality metadata (no model load)")
    cls_sub = p_cls.add_subparsers(dest="classify_cmd")
    cls_vf = cls_sub.add_parser(
        "visual-focus", help="Route a Florence-captions JSONL or dataset")
    cls_vf.add_argument("--input", default=None,
                        help="florence_captions.jsonl (or any record JSONL)")
    cls_vf.add_argument("--dataset", default=None,
                        help="dir containing florence_captions.jsonl")
    cls_vf.add_argument("--output", default=None)
    cls_vf.add_argument("--limit", type=int, default=None)
    cls_vf.add_argument("--overwrite", action="store_true")
    cls_vf.add_argument("--json", action="store_true")
    cls_im = cls_sub.add_parser(
        "visual-focus-image", help="Route a single image's metadata")
    cls_im.add_argument("--image", required=True)
    cls_im.add_argument("--florence-json", default=None)
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

    from .train.presets import PRESETS as _PRESETS
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


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point with professional handling for a busy manifest.

    While a training run is writing to the manifest, commands surface a
    clear, actionable message instead of a raw sqlite traceback."""
    import sqlite3 as _sqlite3
    try:
        _main(argv)
    except KeyboardInterrupt:
        # Phase 9: the heavy stages (extract/photos/curate/caption/
        # quality) commit in batches and skip finished work on rerun, so
        # an interrupt only ever costs the current batch.
        _EMITTER.close()
        print("\nInterrupted - progress through the last batch is saved "
              "in the manifest.\nRerun the same command to resume where "
              "you left off (pipelines: pixel-curator pipeline resume).")
        raise SystemExit(130)
    except manifest.ManifestLocked as exc:
        print(f"\n{exc}")
        raise SystemExit(2)
    except _sqlite3.OperationalError as exc:
        if "locked" not in str(exc).lower():
            raise
        print("\nThe manifest database is temporarily locked - an active "
              "process (typically a training run) is writing to it.\n"
              "Read-only commands (runs, status, insights) work during "
              "training; retry this command in a moment or after the "
              "training run finishes.")
        raise SystemExit(2)


def _main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    global _EMITTER
    _EMITTER = UpdateEmitter(
        quiet=getattr(args, "quiet", False),
        color=(False if getattr(args, "no_color", False) else None))
    prj = load_project(args.project)
    cmd = args.command or "ui"

    # Apply the project's [runtime] policy (torch device, tf32, dedicated
    # onnxruntime provider) before any command runs.
    if cmd != "init":
        try:
            from .runtime import configure_from_project
            configure_from_project(prj)
        except Exception:
            pass

    def out_base(flag: Optional[str]) -> Path:
        return Path(flag or prj.output_base).expanduser()

    if cmd == "init":
        path = Path(args.path)
        if path.exists():
            print(f"Refusing to overwrite existing {path}")
            return
        write_template(path)
        print(f"Wrote starter project file: {path}\nEdit it, then run: "
              f"pixel-curator -p {path} health")
    elif cmd == "doctor":
        print(json.dumps(check_dependencies(), indent=2))
        from .util import project_data_plane
        plane = project_data_plane(prj)
        print("\nProject data plane:")
        print(json.dumps(plane, indent=2))
        from .curate import smart_available
        ok, reason = smart_available()
        print(f"smart curation: {'available' if ok else reason}")
        from .runtime import configure_from_project, onnx_providers
        info = configure_from_project(prj)
        print("\nRuntime / accelerator:")
        print(json.dumps(info, indent=2))
        print(f"onnxruntime providers: {onnx_providers()}")
        from .curate.performance import detect_compute_device
        print("\nCuration compute device:")
        print(json.dumps(detect_compute_device(), indent=2))
        if not plane["ok"]:
            print("\nDOCTOR: project data plane is NOT ready (errors above).")
            return 1
    elif cmd == "stats":
        base = prj.output_path
        if not (base / DB_NAME).exists():
            print(f"No manifest at {base / DB_NAME}")
            return
        conn = manifest.connect(base)
        print(json.dumps(manifest.stats(conn), indent=2))
    elif cmd == "overview":
        from .overview import overview_report, render_overview
        rep = overview_report(prj, out_base(args.output))
        print(json.dumps(rep, indent=2) if args.json
              else render_overview(rep))
    elif cmd == "validate":
        from .validate import run_validation, render_report
        base = out_base(args.output)
        if not (base / DB_NAME).exists():
            print(f"No manifest at {base / DB_NAME}")
            return
        conn = manifest.connect(base, readonly=True)
        report = run_validation(conn)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(render_report(report))
        raise SystemExit(0 if report["healthy"] else 1)
    elif cmd == "workflow":
        from .workflow.orchestrator import (WorkflowRunConfig,
                                            WorkflowStageConfig,
                                            build_workflow_plan, run_workflow,
                                            load_workflow_config,
                                            load_workflow_state)
        wcmd = getattr(args, "workflow_cmd", None) or "run"
        cfg = (load_workflow_config(args.config) if args.config
               else WorkflowRunConfig(project_root=args.project_root,
                                      output_dir=args.output or "."))
        # flags override config-file values
        if args.output:
            cfg.output_dir = args.output
        if args.project_root and args.project_root != ".":
            cfg.project_root = args.project_root
        if args.photos:
            cfg.photos = args.photos
        if args.frames:
            cfg.frames = args.frames
        if args.targets and args.targets != ["core_identity"]:
            cfg.targets = args.targets
        cfg.identity_token = args.identity_token
        cfg.class_token = args.class_token
        cfg.base_model_profile = args.base_model_profile
        cfg.face_det_size = args.face_det_size
        cfg.curation_workers = args.curation_workers
        cfg.resume = args.resume
        cfg.overwrite = args.overwrite
        cfg.strict_validation = args.strict_validation
        # apply only when explicitly requested (and not for plan/status)
        cfg.dry_run = not (args.apply and wcmd in ("run", "resume"))
        s = WorkflowStageConfig(
            orientation_audit=not args.skip_orientation,
            canonicalize_stills=args.canonicalize_stills,
            curate=not args.skip_curation,
            validate_curation=not args.skip_validation,
            export_datasets=not args.skip_export,
            rebuild_anchors=args.with_anchor_rebuild,
            package_kohya=not args.skip_packaging,
            build_training_plans=not args.skip_training_plans)
        cfg.stages = s
        if wcmd == "status":
            st = load_workflow_state(Path(cfg.output_dir).expanduser())
            print(json.dumps(st, indent=2) if args.json else
                  "\n".join(f"{k}: {v.get('status')}"
                            for k, v in (st.get("stages") or {}).items())
                  or "no workflow state found")
            return
        if wcmd == "plan":
            plan = build_workflow_plan(cfg)
            print(json.dumps(plan, indent=2) if args.json else
                  "\n".join(f"{s['name']}: {s['action']}"
                            for s in plan["stages"]))
            return
        report = run_workflow(cfg)
        if args.json:
            print(report.to_json())
        else:
            print(f"Workflow {report.status} "
                  f"({'DRY-RUN' if cfg.dry_run else 'APPLY'}) -> {report.run_dir}")
            for st in report.stages:
                print(f"  {st.name}: {st.status}")
            if report.errors:
                print(f"errors: {len(report.errors)}")
            print(f"summary: {Path(report.run_dir) / 'workflow_summary.md'}")
        return
    elif cmd == "train":
        tcmd = getattr(args, "train_cmd", None)
        if tcmd == "plan":
            from .train.recipe_builder import (TrainingRecipeConfig,
                                               build_training_recipe,
                                               write_training_recipe)
            cfg = None
            if args.target or args.mode:
                cfg = TrainingRecipeConfig(
                    target=args.target or "core_identity",
                    mode=args.mode or "lora",
                    base_model_profile=args.base_model_profile)
            recipe = build_training_recipe(
                Path(args.dataset).expanduser(),
                Path(args.package).expanduser() if args.package else None, cfg)
            written = {}
            if args.apply:
                written = write_training_recipe(
                    recipe, Path(args.output).expanduser())
            if args.json:
                print(json.dumps(recipe.to_dict(), indent=2, default=str))
            else:
                r = recipe.readiness
                print(f"Training plan ({'APPLY' if args.apply else 'DRY-RUN'})"
                      f"\ntarget {recipe.target} | mode {recipe.mode} | "
                      f"readiness {r.readiness_score} ({r.readiness_label})\n"
                      f"recommend: dim {recipe.config.network_dim}, alpha "
                      f"{recipe.config.network_alpha}, repeats "
                      f"{recipe.config.repeats}, epochs {recipe.config.epochs}")
                for w in recipe.warnings:
                    print(f"  ! {w}")
                for k, v in written.items():
                    print(f"wrote {k}: {v}")
            return
        elif tcmd == "stack-plan":
            from .train.stack_planner import build_training_stack_plan
            plan = build_training_stack_plan(
                [Path(p).expanduser() for p in args.recipes],
                Path(args.output).expanduser(), args.base_model_profile)
            if args.json:
                print(json.dumps(plan.to_dict(), indent=2, default=str))
            else:
                print(f"Stack plan: {plan.name} | base "
                      f"{plan.base_model_profile.name}")
                for it in plan.stack_items:
                    print(f"  {it.target}: {it.name} @ {it.recommended_weight}")
                print(f"order: {' -> '.join(plan.recommended_generation_order)}")
                for w in plan.warnings:
                    print(f"  ! {w}")
            return
        else:
            print("usage: pixel-curator train plan|stack-plan ...")
            return
    elif cmd == "export":
        ecmd = getattr(args, "export_cmd", None)
        if ecmd not in ("manifest", "select"):
            print("usage: pixel-curator export {manifest|select} ...")
            return
        if not (prj.output_path / DB_NAME).exists():
            print(f"No manifest at {prj.output_path / DB_NAME}")
            return
        conn = manifest.connect(prj.output_path)
        from .export.general import export_manifest, export_selection
        if ecmd == "manifest":
            rep = export_manifest(conn, Path(args.out), args.format)
        else:
            rep = export_selection(conn, Path(args.out),
                                   top_pct=args.top_pct,
                                   min_score=args.min_score,
                                   mode=args.mode,
                                   with_captions=args.captions,
                                   balanced=args.balanced)
        print(json.dumps(rep, indent=2))
    elif cmd == "filter":
        if not (prj.output_path / DB_NAME).exists():
            print(f"No manifest at {prj.output_path / DB_NAME}")
            return
        conn = manifest.connect(prj.output_path)
        from .curate.semantic import apply_text_filter
        rep = apply_text_filter(conn, args.text, threshold=args.threshold,
                                invert=args.invert, apply=args.apply)
        if args.json:
            print(json.dumps(rep, indent=2))
        else:
            if not rep.get("ok"):
                print(f"Filter unavailable: {rep.get('warning', 'unknown')}")
                return
            verb = "Dropped" if rep.get("apply") else "Would drop"
            mode = "matching" if rep.get("invert") else "non-matching"
            print(f"Query: '{rep['query']}'  (threshold {rep['threshold']})")
            print(f"Matched: {rep['matched']}/{rep['embedded']} embedded "
                  f"frames")
            print(f"{verb} {rep['would_drop']} {mode} frames"
                  + ("" if rep.get("apply") else "  (dry-run; --apply to "
                                                  "commit)"))
            print("Similarity histogram: "
                  + json.dumps(rep["similarity_histogram"]))
    elif cmd == "review":
        rcmd = getattr(args, "review_cmd", None)
        if rcmd not in ("export", "import"):
            print("usage: pixel-curator review {export|import} ...")
            return
        if not (prj.output_path / DB_NAME).exists():
            print(f"No manifest at {prj.output_path / DB_NAME}")
            return
        conn = manifest.connect(prj.output_path)
        from .export.review import (build_review_report,
                                    import_review_decisions)
        if rcmd == "export":
            rep = build_review_report(conn, Path(args.out),
                                      top_pct=args.top_pct,
                                      min_score=args.min_score,
                                      balanced=args.balanced,
                                      max_cards=args.max_cards)
        else:
            rep = import_review_decisions(conn, Path(args.sheet),
                                          apply=args.apply)
        print(json.dumps(rep, indent=2))
    elif cmd == "package":
        if getattr(args, "package_cmd", None) != "kohya":
            print("usage: pixel-curator package kohya --dataset ... --output ...")
            return
        from .export.kohya_packager import (KohyaPackageConfig,
                                            CaptionTemplateConfig,
                                            package_kohya_dataset)
        _pkg_mode = args.mode
        pcfg = KohyaPackageConfig(
            mode=_pkg_mode, identity_token=args.identity_token,
            class_token=args.class_token, concept_name=args.concept_name,
            create_zip=args.create_zip,
            overwrite_captions=args.overwrite_captions,
            resolution=args.resolution, repeats=args.repeats,
            dry_run=not args.apply,
            caption_template=CaptionTemplateConfig(
                identity_token=args.identity_token,
                class_token=args.class_token,
                max_caption_words=args.max_caption_words))
        report = package_kohya_dataset(Path(args.dataset).expanduser(),
                                       Path(args.output).expanduser(), pcfg)
        record: dict = {}
        if args.apply and args.register_version:
            from .dataset_versioning import (KIND_KOHYA_PACKAGE,
                                             register_folder_export)
            from .export.kohya_packager import registry_files
            record = register_folder_export(
                prj.output_path, kind=KIND_KOHYA_PACKAGE,
                name=args.concept_name,
                export_dir=Path(args.output).expanduser(),
                files=registry_files(report),
                image_count=report.sample_count,
                params={"mode": _pkg_mode, "architecture": _arch,
                        "source_dataset": str(report.source_dataset),
                        "resolution": args.resolution,
                        "repeats": args.repeats},
                note=args.version_note, parent_version=args.parent)
            if record.get("version") is not None:
                print(f"Registered as dataset version "
                      f"v{record['version']:03d} (kohya_package)")
            for w in record.get("warnings", []):
                print(f"  ! {w}")
        if args.apply and args.training_config:
            # Phase 10: ready-to-train config beside the package
            from .train.export_config import generate_for_export
            conn = manifest.connect(prj.output_path)
            try:
                _, tc_msgs = generate_for_export(
                    conn, Path(args.output).expanduser(),
                    dataset_version=record.get("version"),
                    frame_count=report.sample_count,
                    model_type=("dreambooth" if _pkg_mode == "dreambooth"
                                else "lora"),
                    output_name=args.concept_name)
            finally:
                conn.close()
            for m in tc_msgs:
                print(m)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, default=str))
        else:
            print(f"Kohya package ({'DRY-RUN' if report.dry_run else 'APPLY'})"
                  f"\nsource: {report.source_dataset}\noutput: {report.output_dir}"
                  f"\nmode {report.mode} | samples {report.sample_count}")
            cs = report.summary.get("caption_stats", {})
            print(f"captions: avg {cs.get('avg_words')} words, "
                  f"{cs.get('empty_captions')} empty, "
                  f"{cs.get('duplicate_captions')} duplicates")
            if report.zip_path:
                print(f"zip: {report.zip_path}")
            for w in report.warnings[:10]:
                print(f"  ! {w}")
        return
    elif cmd == "anchors-rebuild":
        from .curate.anchor_quality import (AnchorQualityConfig,
                                            rebuild_anchor_set,
                                            check_anchor_report)
        acfg = AnchorQualityConfig(
            dry_run=not args.apply, min_face_score=args.min_face_score,
            min_subject_integrity=args.min_subject_integrity,
            min_face_region_score=args.min_face_region_score)
        report = rebuild_anchor_set(Path(args.source).expanduser(),
                                    Path(args.output).expanduser(), acfg)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, default=str))
        else:
            print(f"Anchor rebuild ({'DRY-RUN' if report.dry_run else 'APPLY'})"
                  f"\nsource: {report.source_dir}\noutput: {report.output_dir}\n"
                  f"total {report.total} | accept {report.accepted} | "
                  f"review {report.review} | reject {report.rejected}")
            for code, msg in check_anchor_report(report):
                print(f"  ! {code}: {msg}")
            print(f"wrote: {Path(report.output_dir) / 'anchor_card.md'}")
        return
    elif cmd == "dataset-card":
        from .curate.dataset_cards import (
            DatasetValidationConfig, DatasetCardConfig,
            validate_exported_dataset, generate_dataset_card)
        dataset_dir = Path(args.dataset_dir).expanduser()
        man_path = Path(args.manifest).expanduser() if args.manifest else None
        vcfg = DatasetValidationConfig(strict=args.strict_validation)
        report = validate_exported_dataset(
            dataset_dir, config=vcfg, manifest_path=man_path)
        result = {"validation": report.to_dict()}
        if args.write_dataset_card:
            ccfg = DatasetCardConfig(max_frame_rows_markdown=args.max_rows)
            card = generate_dataset_card(
                dataset_dir, manifest_path=man_path, config=ccfg,
                validation_report=report)
            result["card_paths"] = card.get("_paths", {})
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            r = report
            print(f"Dataset: {r.dataset_path}\nSamples: {r.sample_count} | "
                  f"passed: {r.passed} | warnings: {len(r.warnings)}")
            for w in r.warnings:
                wd = w.to_dict() if hasattr(w, "to_dict") else w
                print(f"  - {wd['code']} (count={wd['count']}, "
                      f"severity={wd['severity']}): {wd['message']}")
            if args.write_dataset_card:
                for k, v in result.get("card_paths", {}).items():
                    print(f"wrote {k}: {v}")
        raise SystemExit(0 if report.passed else 1)
    elif cmd == "selftest":
        import subprocess
        runner = Path(__file__).resolve().parent / "tests" / "run_all.py"
        if not runner.exists():
            print(f"Test runner not found at {runner}")
            raise SystemExit(1)
        rc = subprocess.call(
            [sys.executable, str(runner), "--timeout", str(args.timeout),
             *args.filters])
        raise SystemExit(rc)
    elif cmd == "likeness":
        from .eval.likeness import run_likeness, render_likeness
        base = out_base(args.output)
        if not (base / DB_NAME).exists():
            print(f"No manifest at {base / DB_NAME}")
            return
        conn = manifest.connect(base)
        rep = run_likeness(conn, prj=prj, lora=args.lora,
                           only_missing=not args.all,
                           min_identity_sim=args.min_identity_sim)
        print(json.dumps(rep, indent=2) if args.json else render_likeness(rep))
    elif cmd == "leaderboard":
        from .eval.leaderboard import (build_leaderboard, head_to_head,
                                       render_leaderboard)
        base = out_base(args.output)
        if not (base / DB_NAME).exists():
            print(f"No manifest at {base / DB_NAME}")
            return
        conn = manifest.connect(base, readonly=True)
        if args.vs:
            res = head_to_head(conn, args.vs[0], args.vs[1])
            print(json.dumps(res, indent=2) if args.json else
                  f"{res['a']} vs {res['b']}: {res['shared_cells']} shared "
                  f"cells | wins {res['wins_a']}-{res['wins_b']} ties "
                  f"{res['ties']} | delta {res['mean_delta_a_minus_b']} | "
                  f"winner: {res['winner']}")
        else:
            board = build_leaderboard(conn, labels=args.label or None)
            print(json.dumps(board, indent=2) if args.json
                  else render_leaderboard(board))
    elif cmd == "export-dreambooth":
        from .train.dreambooth_export import (DreamBoothExportConfig,
                                              plan_dreambooth_export, render_plan)
        base = out_base(args.output)
        if not (base / DB_NAME).exists():
            print(f"No manifest at {base / DB_NAME}")
            return
        conn = manifest.connect(base)
        bm = getattr(prj, "base_model", "") or ""
        cfg = DreamBoothExportConfig(
            output_base=base,
            out_dir=(Path(args.out).expanduser() if args.out else None),
            token=prj.trigger_token, class_word=prj.class_word,
            count=args.count, repeats=args.repeats, diversity=args.diversity,
            quality_aware=args.quality_aware,
            reg=args.reg, reg_count=args.reg_count, link_mode=args.link_mode,
            dry_run=not args.apply,
            dataset_version=args.dataset, from_run=args.from_run,
            from_recipe=args.from_recipe, diverse=args.diverse,
            max_source_frac=args.max_source_frac,
            max_cluster_frac=args.max_cluster_frac,
            base_model_name=(Path(bm).name if bm
                             else "cyberrealisticPony_v180Coreshift.safetensors"))
        plan = plan_dreambooth_export(conn, prj, cfg)
        print(render_plan(plan))
    elif cmd == "orient":
        from .orient import (OrientConfig, run_orientation, undo_from_manifest,
                             render_summary, make_insightface_detector, IMG_EXTS)
        if args.undo:
            res = undo_from_manifest(args.undo)
            print(f"Undo: reverted {res['reverted']} rotation(s) from {res['from']}")
            for e in res["errors"][:10]:
                print(f"  ! {e}")
            return
        base = out_base(args.output)
        if args.dir:
            root = Path(args.dir).expanduser()
            paths = [str(p) for p in sorted(root.rglob("*"))
                     if p.suffix.lower() in IMG_EXTS]
        else:
            if not (base / DB_NAME).exists():
                print(f"No manifest at {base / DB_NAME}")
                return
            conn = manifest.connect(base)
            paths = [r[0] for r in conn.execute(
                "SELECT path FROM frames WHERE status IN "
                "('selected','packaged')").fetchall()]
        if args.limit:
            paths = paths[:args.limit]
        if not paths:
            print("No images to process.")
            return
        detector, reason = make_insightface_detector(prj)
        if detector is None:
            print(f"Cannot detect orientation: {reason}")
            return
        cfg = OrientConfig(min_det_score=args.min_det_score,
                           min_axis_ratio=args.min_axis_ratio,
                           apply=args.apply, verify=args.verify)
        man = run_orientation(paths, detector, cfg,
                              manifest_dir=base / "orient_manifests")
        print(render_summary(man))
    elif cmd == "diverse-select":
        from .diverse import fetch_candidates, diverse_select, render_plan
        base = out_base(args.output)
        if not (base / DB_NAME).exists():
            print(f"No manifest at {base / DB_NAME}")
            return
        conn = manifest.connect(base, readonly=True)
        plan = diverse_select(
            fetch_candidates(conn), args.count,
            max_source_frac=args.max_source_frac,
            max_cluster_frac=args.max_cluster_frac,
            min_identity=args.min_identity, min_sharpness=args.min_sharpness)
        if args.json:
            out = {k: v for k, v in plan.items() if k != "selected"}
            out["selected_frame_ids"] = [f["frame_id"] for f in plan["selected"]]
            print(json.dumps(out, indent=2))
        else:
            print(render_plan(plan))
    elif cmd == "keyframes":
        from .curate.keyframes import prune_video_keyframes, render_keyframes
        base = out_base(args.output)
        if not (base / DB_NAME).exists():
            print(f"No manifest at {base / DB_NAME}")
            return
        conn = manifest.connect(base)
        res = prune_video_keyframes(
            conn, max_per_video=args.max_per_video, min_spacing=args.min_spacing,
            min_quality=args.min_quality, apply=args.apply)
        print(render_keyframes(res))
    elif cmd == "quality":
        from .curate.quality_signals import analyze_quality_pass, render_quality
        base = out_base(args.output)
        if not (base / DB_NAME).exists():
            print(f"No manifest at {base / DB_NAME}")
            return
        conn = manifest.connect(base)
        face_detector = None
        if args.with_face:
            from .curate.quality_signals import make_face_detector
            face_detector, reason = make_face_detector(prj)
            if face_detector is None:
                print(f"--with-face unavailable: {reason}\nRunning bbox-only pass.")
        from .curate.advanced_quality import AdvancedQualityConfig
        advanced = AdvancedQualityConfig.from_flags(
            advanced=args.with_advanced_quality,
            advanced_aesthetic=args.with_advanced_aesthetic,
            figure_quality=args.with_figure_quality,
            region_quality=args.with_region_quality,
            coverage=args.with_coverage,
            natural_texture=args.with_natural_texture,
            aesthetic=args.with_aesthetic,
            skin_rendering=args.with_skin_rendering,
            expression_quality=args.with_expression_quality,
            lighting=args.with_lighting,
            dynamic_pose=args.with_dynamic_pose,
            body_symmetry=args.with_body_symmetry,
            advanced_composition=args.with_advanced_composition,
            color_mood=args.with_color_mood,
            background_quality=args.with_background_quality,
            motion_blur=args.with_motion_blur,
            face_occlusion=args.with_face_occlusion,
            body_proportion=args.with_body_proportion,
            advanced_curation=args.with_advanced_curation,
            clothing_type=args.with_clothing_type,
            full_body=args.with_full_body,
            eye_openness=args.with_eye_openness,
            color_cast=args.with_color_cast,
            duplicate_detection=args.with_duplicate_detection,
            exposure_outlier=args.with_exposure_outlier,
            region_readiness=args.with_region_readiness,
            coverage_intensity=args.with_coverage_intensity,
            pose_duplicate=args.with_pose_duplicate)
        res = analyze_quality_pass(conn, limit=args.limit, store=args.store,
                                   face_detector=face_detector,
                                   advanced=advanced if advanced.enabled() else None,
                                   resume=args.resume,
                                   progress=getattr(args, "progress", True)
                                   and not args.json)
        print(json.dumps(res, indent=2) if args.json else render_quality(res))
    elif cmd == "categorize":
        from .curate.categorize import (categorize_pass, render_categorize,
                                        export_categories)
        base = out_base(args.output)
        if not (base / DB_NAME).exists():
            print(f"No manifest at {base / DB_NAME}")
            return
        conn = manifest.connect(base, readonly=True)
        report = categorize_pass(conn)
        export_plan = None
        if args.export_dir:
            export_plan = export_categories(
                report, args.export_dir, count=args.count,
                link_mode=args.link_mode, dry_run=not args.apply,
                architecture=args.architecture,
                token=prj.trigger_token, class_word=prj.class_word)
        if args.json:
            payload = {"counts": report["counts"], "total": report["total"],
                       "has_advanced": report["has_advanced"],
                       "warnings": report["warnings"]}
            if export_plan is not None:
                payload["export"] = export_plan
            print(json.dumps(payload, indent=2))
        else:
            print(render_categorize(report, top=args.top))
            if export_plan is not None:
                mode = "PLAN (dry-run)" if export_plan["dry_run"] else "WROTE"
                print(f"\nExport {mode} -> {export_plan['output_dir']}")
                for cat, info in export_plan["exported"].items():
                    print(f"  {cat:<10}{info['count']:>8}  {info['dir']}")
                for w in export_plan["warnings"]:
                    print(f"  ! {w}")
                if export_plan["dry_run"]:
                    print("  (re-run with --apply to write)")
    elif cmd == "vlm-caption":
        from .caption.vlm import (make_florence_captioner, run_vlm_pass, render_vlm)
        base = out_base(args.output)
        if not (base / DB_NAME).exists():
            print(f"No manifest at {base / DB_NAME}")
            return
        caption_fn, model_id = make_florence_captioner(prj, model_id=args.model)
        if caption_fn is None:
            print(f"VLM unavailable: {model_id}")
            return
        conn = manifest.connect(base)
        res = run_vlm_pass(conn, caption_fn, model_id=model_id, limit=args.limit,
                           compose=args.compose)
        print(render_vlm(res))
    elif cmd == "outfit":
        from .curate.outfit import run_outfit_pass, render_outfit
        base = out_base(args.output)
        if not (base / DB_NAME).exists():
            print(f"No manifest at {base / DB_NAME}")
            return
        conn = manifest.connect(base)
        res = run_outfit_pass(conn, limit=args.limit, store=args.store)
        print(json.dumps(res, indent=2, default=str) if args.json
              else render_outfit(res))
    elif cmd == "framing-report":
        from .curate.framing import framing_distribution, render_framing
        base = out_base(args.output)
        if not (base / DB_NAME).exists():
            print(f"No manifest at {base / DB_NAME}")
            return
        conn = manifest.connect(base, readonly=True)
        print(render_framing(framing_distribution(conn)))
    elif cmd == "extract":
        cfg = ExtractConfig(
            output_base=out_base(args.output),
            fps=args.fps, jpeg_quality=args.jpeg_quality,
            segment_seconds=args.segment_seconds,
            use_personfromvid=args.personfromvid,
            import_photos=not args.no_photos,
            photo_import_mode=args.photo_mode,
            resume=args.resume, dry_run=args.dry_run,
            limit_videos=args.limit_videos, overwrite=args.overwrite,
            max_side=args.max_side,
            adaptive=args.adaptive,
            fps_tiers=(
                # (max_duration_seconds, fps, per-tier frame cap)
                (args.live_max_seconds,   args.live_fps,   DEFAULT_FPS_TIERS[0][2]),
                (args.short_max_seconds,  args.short_fps,  DEFAULT_FPS_TIERS[1][2]),
                (args.medium_max_seconds, args.medium_fps, DEFAULT_FPS_TIERS[2][2]),
                (args.long_max_seconds,   args.long_fps,   DEFAULT_FPS_TIERS[3][2]),
                (float("inf"),            args.xlong_fps,  DEFAULT_FPS_TIERS[4][2]),
            ),
            max_frames_per_video=args.max_frames_per_video,
            min_frames_per_video=args.min_frames_per_video,
            prefilter=getattr(args, "prefilter", True),
        )
        vdirs = args.video_dir or prj.video_dirs
        stream(pipeline_generator(vdirs, args.photos_dir or prj.photos_dir, cfg))
    elif cmd == "smart-extract":
        # NSFW smart extraction: same engine as `extract`, but the config is
        # built from the [extraction] section of the project TOML (which -p
        # points at — typically pixel-curator_nsfw_v2.toml).
        from .smart_extract import run as run_smart_extract
        rc = run_smart_extract(
            args.project,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            limit_videos=args.limit_videos,
        )
        if rc:
            raise SystemExit(rc)
    elif cmd == "photos":
        cfg = ExtractConfig(
            output_base=out_base(args.output),
            import_photos=True, photo_import_mode=args.photo_mode,
            resume=args.resume, dry_run=args.dry_run,
            prefilter=getattr(args, "prefilter", True),
        )
        stream(pipeline_generator([], args.photos_dir or prj.photos_dir, cfg))
    elif cmd == "prefilter":
        from .prefilter import (prefilter_generator, prefilter_summary,
                                restore_prefiltered)
        base = out_base(args.output)
        if args.restore:
            conn = manifest.connect(base)
            n = restore_prefiltered(conn, reason=args.reason)
            scope = f" (reason {args.reason})" if args.reason else ""
            print(f"Restored {n} prefilter-rejected frame(s){scope} "
                  "back to 'new'.")
            left = prefilter_summary(conn)
            if left:
                print("Still rejected: " + ", ".join(
                    f"{k}: {v}" for k, v in sorted(left.items())))
        else:
            pcfg = PrefilterConfig(
                output_base=base, min_bytes=args.min_bytes,
                min_side=args.min_side, min_bpp=args.min_bpp,
                text_check=args.text_check,
                decode_checks=args.decode_checks,
                extreme_blur=args.extreme_blur,
                workers=args.workers, dry_run=args.dry_run,
            )
            stream(prefilter_generator(pcfg))
    elif cmd == "anchor":
        cfg = SmartCurateConfig(
            output_base=out_base(args.output),
            anchor_dir=Path(args.anchor_dir or prj.anchor_dir).expanduser()
            if (args.anchor_dir or prj.anchor_dir) else None,
            det_size=args.det_size,
        )
        for update in build_anchor(cfg):
            print(update)
    elif cmd == "curate":
        base = out_base(args.output)
        if not args.smart_only:
            cfg = CurateConfig(
                output_base=base,
                hamming_threshold=args.hamming,
                min_sharpness=args.min_sharpness,
                min_brightness=args.min_brightness,
                max_brightness=args.max_brightness,
                workers=args.workers, rescore=args.rescore,
                prefilter_text=getattr(args, "prefilter_text", True),
            )
            stream(curate_generator(cfg))
        if args.smart or args.smart_only:
            meta_dir = args.metadata_dir or prj.metadata_dir or None
            subjects = tuple(args.subject_names or prj.subject_names or ())
            scfg = SmartCurateConfig(auto_rotate=args.auto_rotate,
                output_base=base,
                identity_threshold=args.identity_threshold,
                min_face_area=args.min_face_area,
                orient_min_det_score=getattr(args, "orient_min_score", 0.55),
                orient_min_axis_ratio=getattr(args, "orient_min_axis_ratio",
                                              1.25),
                det_size=getattr(args, "face_det_size", 640),
                rescan=args.rescan,
                metadata_dir=Path(meta_dir).expanduser() if meta_dir else None,
                subject_names=subjects,
                sidecar_identity_relax=args.sidecar_relax,
                profile_performance=getattr(args, "profile_performance", False),
            )
            stream(smart_curate_generator(scfg))
        # Phase 5: cheap DB-only intelligence digest after any curate pass
        try:
            from .intelligence import analyze_dataset, intelligence_digest
            conn = manifest.connect(base, readonly=True)
            print("\n" + intelligence_digest(
                analyze_dataset(conn, output_base=base)))
        except Exception:
            pass
    elif cmd == "pipeline":
        # NOTE: aliased import - a bare `pipeline_generator` here would shadow
        # the module-level extract.pipeline_generator across ALL branches
        from .pipeline_dag import (PipelineConfig, load_state,
                                   pipeline_generator as dag_generator,
                                   resume_generator)
        from .pipeline_recipes import (
            RecipePipelineConfig, load_recipe_state,
            recipe_pipeline_generator, recipe_resume_generator,
            render_pipeline_list, render_plan, resolve_pipeline)
        base = out_base(getattr(args, "output", None))
        legacy = args.what in ("full", "mega")
        if args.action == "list":
            print(render_pipeline_list(prj))
        elif args.action == "status":
            conn = manifest.connect(base)
            rstate = load_recipe_state(conn)
            state = load_state(conn)
            if rstate:
                print(json.dumps(rstate, indent=2))
            elif state:
                print(json.dumps(state, indent=2))
            else:
                print("No pipeline in progress.")
        elif args.action == "resume":
            # Phase 7 recipe state takes priority; the generator yields
            # nothing when no recipe state exists, so we fall back cleanly.
            resumed = False
            for update in recipe_resume_generator(prj, base):
                resumed = True
                _EMITTER.emit(update)
            _EMITTER.close()
            if not resumed:
                stream(resume_generator(prj, base))
        elif args.action == "plan" or (args.dry_run and not legacy):
            rcfg = RecipePipelineConfig(
                output_base=base, pipeline=args.what,
                only=args.only, skip=args.skip,
                from_stage=args.from_stage, until_stage=args.until_stage,
                gates=args.gate, overrides=args.overrides, dry_run=True,
            )
            plan, errors = resolve_pipeline(prj, rcfg)
            if errors:
                for e in errors:
                    print(f"ERROR: {e}")
                print("\nAvailable pipelines:")
                print(render_pipeline_list(prj))
            else:
                print(render_plan(plan, base))
        elif args.what == "mega":
            from .pipeline_dag import MegaConfig, mega_generator
            stream(mega_generator(prj, MegaConfig(
                output_base=base, backend=args.backend)))
        elif args.what == "full":
            cfg = PipelineConfig(
                output_base=base, recipe=args.recipe, preset=args.preset,
                gates=args.gate, smart=args.smart, cluster=args.cluster,
                matrix_backend=args.backend,
            )
            stream(dag_generator(prj, cfg))
        else:
            # Phase 7 recipe engine (builtin or [pipelines.*] recipe)
            rcfg = RecipePipelineConfig(
                output_base=base, pipeline=args.what,
                only=args.only, skip=args.skip,
                from_stage=args.from_stage, until_stage=args.until_stage,
                gates=args.gate, overrides=args.overrides,
            )
            stream(recipe_pipeline_generator(prj, rcfg))
    elif cmd == "watch":
        from .maintenance import WatchConfig, watch_generator
        cfg = WatchConfig(output_base=out_base(args.output),
                          interval=args.interval,
                          auto_curate=args.auto_curate, once=args.once)
        try:
            for update in watch_generator(prj, cfg):
                print(update)
        except KeyboardInterrupt:
            print("\nwatch stopped.")
    elif cmd == "gc":
        from .maintenance import GcConfig, gc_generator
        for update in gc_generator(prj, GcConfig(
            output_base=out_base(args.output), apply=args.apply,
            keep_builds=args.keep_builds,
        )):
            print(update)
    elif cmd == "space":
        from .maintenance import space_report
        for row in space_report(prj, prj.output_path):
            print(f"  {row['area']:<16} {str(row['files']):>8} files  {row['human']:>10}")
    elif cmd == "previews":
        from pathlib import Path as _P
        from .lora_explorer import lora_dirs
        from .qa_render import render_lora_previews
        lp = _P(args.lora).expanduser()
        if not lp.exists():
            for d in lora_dirs(prj):
                cand = d / f"{args.lora}.safetensors"
                if cand.exists():
                    lp = cand
                    break
        for u in render_lora_previews(prj, lp):
            print(u)
    elif cmd == "ab":
        from .qa_render import ab_compare
        def _split(s):
            n, _, w = s.rpartition(":")
            return (n or s, float(w) if w and n else 0.75)
        for u in ab_compare(prj, _split(args.a), _split(args.b),
                            prompt_tail=args.tail, seed=args.seed):
            print(u)
    elif cmd == "golden":
        from .qa_render import golden_regression
        for u in golden_regression(prj, args.lora, args.weight,
                                   args.previous or None):
            print(u)
    elif cmd == "backup":
        from .maintenance import backup_project
        out = backup_project(prj, args.out or None)
        print(f"Backup written -> {out}")
    elif cmd == "restore":
        from .maintenance import restore_project
        dest = restore_project(args.archive, args.dest)
        print(f"Extracted to {dest} - review and place files manually.")
    elif cmd == "mergeforge":
        from . import mergeforge as mf
        if args.mf_cmd == "scan":
            rep = mf.analyze_library(prj)
            print(f"{rep['library_size']} LoRAs | roles: "
                  + " ".join(f"{k}={v}" for k, v
                             in sorted(rep["role_counts"].items()))
                  + (" | merge-ready" if rep["merge_ready"] else ""))
            for n in rep["notes"]:
                print(f"  note: {n}")
            for e in rep["entries"]:
                h = e["health"]
                print(f"  {e['lora_id']:<36} {e['role']:<16} "
                      f"health {h['grade']}({h['score']})  "
                      f"[{', '.join(e['reason_codes'])}]")
        elif args.mf_cmd == "plan":
            plan = mf.wizard_plan(prj, args.lora)
            if plan.get("error"):
                print(f"FATAL: {plan['error']}")
                raise SystemExit(1)
            c = plan["compatibility"]
            print(f"Compatibility: {c['overall_verdict']} "
                  f"({c['overall_score']}/100)")
            for p in c["pairs"]:
                for w in p["warnings"]:
                    print(f"  ! {w}")
            print("Recommended weights:")
            for k, w in plan["weights"]["weights"].items():
                print(f"  {k:<36} {w}")
            for n in plan["weights"]["notes"]:
                print(f"  note: {n}")
            print(f"Recipe draft id: {plan['recipe_draft']['recipe_id']}")
        elif args.mf_cmd == "merge":
            from .eval.matrix import parse_lora_specs
            loras = []
            for spec in args.lora:
                loras.extend(parse_lora_specs(spec))
            rec = mf.make_recipe(args.name, loras, method=args.method,
                                 base_model=prj.base_model)
            for update in mf.execute_recipe(prj, out_base(args.output),
                                            rec, output_name=args.name,
                                            target_rank=args.rank):
                print(update)
        elif args.mf_cmd == "recipes":
            for r in mf.load_recipes(out_base(args.output)):
                print(f"  {r['recipe_id']}  {r['name']:<28} {r['method']:<13}"
                      f" {r['created_at'][:16]}  -> {r.get('output') or '-'}")
        elif args.mf_cmd == "rerun":
            recs = {r["recipe_id"]: r
                    for r in mf.load_recipes(out_base(args.output))}
            if args.recipe_id not in recs:
                print(f"FATAL: unknown recipe id {args.recipe_id}")
                raise SystemExit(1)
            r = recs[args.recipe_id]
            for update in mf.execute_recipe(prj, out_base(args.output), r,
                                            output_name=r["name"] + "_rerun"):
                print(update)
    elif cmd == "insights":
        from . import insights
        # collapse/rotation write; dashboard/health/lint are read-only and must
        # keep working while training holds the manifest. `clusters` prefers a
        # writable handle to cache names, but degrades gracefully (best-effort
        # write in name_clusters) if the manifest is locked or read-only.
        _writes = args.insights_cmd in ("collapse", "rotation", "clusters") \
            or (args.insights_cmd == "lint"
                and (getattr(args, "fix", False)
                     or getattr(args, "undo", None)))
        conn = manifest.connect(prj.output_path, readonly=not _writes)
        if not args.insights_cmd:
            # Quality Metrics Dashboard (Advanced Curation Phase 1)
            dash = insights.quality_dashboard(
                conn, dataset=args.dataset,
                per_image=bool(args.json or args.export))
            if args.export:
                out = Path(args.export).expanduser()
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(dash, indent=2))
                print(f"Dashboard exported -> {out}")
            if args.json:
                print(json.dumps(dash, indent=2))
            if not (args.json or args.export):
                print(insights.render_dashboard(dash))
        elif args.insights_cmd == "health":
            h = insights.dataset_health(conn)
            print(f"Dataset health: {h['score']}/100  "
                  f"({h.get('selected', 0)} curated frames)")
            for r in h["reasons"]:
                print(f"  - {r}")
        elif args.insights_cmd == "lint":
            from . import lint as lint_mod
            if getattr(args, "undo", None):
                r = lint_mod.undo_fixes(conn, Path(args.undo).expanduser())
                print(f"Restored {r['restored']} change(s) from "
                      f"{r['from']}")
                for e in r["errors"]:
                    print(f"  ! {e}")
            else:
                issues = lint_mod.run_lint(conn)
                if args.json:
                    print(json.dumps([i.to_dict() for i in issues],
                                     indent=2))
                else:
                    print(lint_mod.render_report(
                        issues, fix_mode=getattr(args, "fix", False)))
                if getattr(args, "fix", False):
                    only = {c.strip() for c in (args.only or []) if c.strip()}
                    r = lint_mod.apply_fixes(conn, issues, prj.output_path,
                                             only=only or None)
                    print(f"\nApplied {r['fixed']} fix(es): "
                          + (", ".join(f"{k} x{v}"
                                       for k, v in r["counts"].items())
                             or "nothing to fix"))
                    if r["journal"]:
                        print(f"Journal: {r['journal']}")
                        print(f"Undo with: pixel-curator insights lint "
                              f"--undo {r['journal']}")
        elif args.insights_cmd == "intelligence":
            from .intelligence import analyze_dataset, render_intelligence
            report = analyze_dataset(conn, output_base=prj.output_path,
                                     sample=getattr(args, "sample", 0))
            if args.json:
                print(json.dumps(report, indent=2, default=str))
            else:
                print(render_intelligence(report))
        elif args.insights_cmd == "collapse":
            r = insights.collapse_near_duplicates(
                conn, max_hamming=args.max_hamming, apply=args.apply)
            verb = ("rejected" if r["applied"]
                    else "would be rejected (use --apply)")
            print(f"{r['groups']} near-duplicate groups | "
                  f"{r['would_reject']} frames {verb}")
        elif args.insights_cmd == "rotation":
            for u in insights.fix_rotations(conn, prj, apply=args.apply,
                                            limit=args.limit):
                print(u)
        elif args.insights_cmd == "clusters":
            names = insights.name_clusters(conn)
            for cid in sorted(names, key=int):
                print(f"  c{cid}: {names[cid]}")
    elif cmd == "duplicates":
        from .curate.duplicates import DuplicateScanConfig, scan_duplicates, \
            scan_generator
        conn = manifest.connect(out_base(args.output))
        dcfg = DuplicateScanConfig(
            dataset_version=args.dataset, method=args.method,
            max_hamming=args.max_hamming,
            clip_threshold=args.clip_threshold, write=args.write)
        if args.json:
            print(json.dumps(scan_duplicates(conn, dcfg), indent=2))
        else:
            for update in scan_generator(conn, dcfg):
                print(update)
    elif cmd == "concept":
        from . import lora_explorer as lx
        cards = lx.scan_loras(prj)
        if args.concept_cmd == "scan":
            try:
                conn = manifest.connect(prj.output_path)
                n = lx.sync_profiles_to_manifest(conn, cards)
            except Exception as exc:                             # noqa: BLE001
                n = len(cards)
                print(f"(manifest unavailable - listing only: {exc})")
            print(f"Indexed {n} LoRAs from "
                  f"{', '.join(str(d) for d in lx.lora_dirs(prj)) or '(none)'}")
            for c in lx.filter_cards(cards, sort="family"):
                print(f"  {c.lora_id:<36} {c.profile.family:<12} "
                      f"risk={c.profile.identity_risk:<6} "
                      f"w={c.profile.weight_default}")
            if args.write_sidecars:
                from pathlib import Path as _P
                for c in cards:
                    sp = lx.sidecar_path(_P(c.path))
                    if not sp.exists():
                        lx.save_sidecar(_P(c.path), c.profile)
                        print(f"  sidecar -> {sp.name}")
        elif args.concept_cmd == "stack":
            from .stack_intelligence import resolve_stack
            sel, weights = [], {}
            by_id = {c.lora_id: c for c in cards}
            for spec in args.lora:
                lid, _, w = spec.partition(":")
                if lid not in by_id:
                    print(f"  unknown LoRA id: {lid}")
                    continue
                sel.append(by_id[lid])
                if w:
                    weights[lid] = float(w)
            st = resolve_stack(sel, weights, prj.base_model)
            print(f"Base model: {st.base_model}")
            if st.identity_anchor:
                a = st.identity_anchor
                print(f"  ANCHOR {a.lora_id:<30} {a.weight:<5} {a.reason}")
            for i in st.concept_loras:
                print(f"  {i.family:<12} {i.lora_id:<24} {i.weight:<5} "
                      f"blocks {i.blocks_cli}")
            print(f"Concept strength {st.total_concept_strength} | "
                  f"identity preservation {st.identity_preservation_score}")
            for w in st.warnings:
                print(f"  [{w.severity}] {w.message}")
        elif args.concept_cmd == "batch":
            import json as _json
            from pathlib import Path as _P
            from .batch_variations import (VariationGrid, expand_grid,
                                           run_batch_generator, save_batch)
            spec = _json.loads(_P(args.spec).expanduser().read_text())
            sel_ids = set(spec.pop("loras", []))
            sel = [c for c in cards if not sel_ids or c.lora_id in sel_ids]
            known = VariationGrid.__dataclass_fields__
            spec.setdefault("mode", args.mode)
            grid = VariationGrid(**{k: v for k, v in spec.items()
                                    if k in known})
            jobs = expand_grid(prj, sel, grid)
            conn = manifest.connect(prj.output_path)
            bid = save_batch(conn, prj, grid, jobs)
            print(f"Batch {bid}: {len(jobs)} variation jobs saved.")
            for j in jobs[:5]:
                print(f"  {j.variation_id} seed={j.seed} "
                      f"loras={j.loras} sliders={j.slider_state}")
            if len(jobs) > 5:
                print(f"  ... +{len(jobs) - 5} more")
            if args.run:
                for u in run_batch_generator(prj, conn, jobs):
                    print(u)
    elif cmd == "study":
        from . import study as study_mod
        if args.study_cmd == "classify":
            from .study import StudyConfig, classify_generator
            for update in classify_generator(
                    prj, StudyConfig(output_base=prj.output_path,
                                     rescan=args.rescan)):
                print(update)
        elif args.study_cmd == "report":
            conn = manifest.connect(prj.output_path)
            rep = study_mod.study_report(conn)
            print("Study Intelligence Layer report")
            for cat, n in sorted(rep["categories"].items()):
                print(f"  {cat:<32} {n}")
            print(f"  export_eligible {rep['export_eligible']} | "
                  f"needs_review {rep['needs_review']}")
            for k, v in rep.items():
                if k.endswith("_mean"):
                    print(f"  {k:<32} {v}")
        elif args.study_cmd == "recipes":
            added = study_mod.register_study_recipes(prj)
            for n in study_mod.STUDY_RECIPES:
                mark = "added" if n in added else "exists"
                print(f"  {n:<32} [{mark}]")
            if args.apply and added:
                if not args.project:
                    print("No project file (-p) - cannot persist.")
                else:
                    from pathlib import Path as _P
                    from .config import save_project
                    save_project(prj, _P(args.project))
                    print(f"Saved to {args.project}")
            elif added:
                print("(dry run - use --apply to persist into the project file)")
        elif args.study_cmd == "stack":
            rec = study_mod.suggest_study_stack(args.mode, prj.base_model)
            print(f"Mode: {rec['mode']}   profile: {rec['profile']}")
            for s in rec["stack"]:
                print(f"  {s['type']:<18} {s['role']:<9} w={s['weight']:<5} "
                      f"blocks {s['blocks_cli']}")
            print("Merge order:", " -> ".join(rec["merge_order"]))
            for w in rec["warnings"]:
                print(f"  ! {w}")
        elif args.study_cmd == "presets":
            from pathlib import Path as _P
            target = study_mod.write_study_presets(
                prj, _P(args.path) if args.path else None)
            print(f"Study preset pack written -> {target}")
    elif cmd == "registry":
        from .registry import build_registry, best_likeness
        conn = manifest.connect(prj.output_path)
        entries = build_registry(prj, conn)
        if not entries:
            print("No LoRAs known yet. Train one or drop .safetensors in lora_output_dir.")
        for e in entries:
            lk = best_likeness(e)
            print(f"  {e['name']:<32} [{e['kind']:<8}] "
                  f"likeness={lk if lk is not None else '-':<8} "
                  f"files={len(e.get('files', []))}")
    elif cmd == "card":
        conn = manifest.connect(prj.output_path)
        if args.run:
            from .model_card import generate_model_card
            try:
                path, _md = generate_model_card(
                    conn, prj, prj.output_path, args.run,
                    output=args.output,
                    include_samples=args.include_samples)
                print(f"Model card: {path}")
            except KeyError as exc:
                print(f"FATAL: {exc}")
            return
        if not args.name:
            print("Usage: card <lora-name>  (legacy registry card)\n"
                  "   or: card --run <run_name>  (Phase 3 model card)")
            return
        from .registry import model_card
        path = model_card(prj, conn, args.name)
        print(f"Model card: {path}" if path else f"Unknown LoRA: {args.name}")
    elif cmd == "eval":
        from .eval.run_eval import evaluate_run, render_eval
        conn = manifest.connect(out_base(args.output))
        if args.generate_samples:
            from .sweep import SweepConfig, sweep_generator
            print("Rendering + scoring eval samples via the best-epoch "
                  "sweep (Forge required)...")
            for update in sweep_generator(prj, SweepConfig(
                    output_base=out_base(args.output), run=args.run)):
                print(update)
        try:
            report = evaluate_run(conn, prj, out_base(args.output),
                                  args.run, write=not args.dry_run)
        except KeyError as exc:
            print(f"FATAL: {exc}")
            return
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(render_eval(report))
    elif cmd == "studio":
        from . import studio as st_mod
        scmd = getattr(args, "studio_cmd", None) or "status"
        base = out_base(getattr(args, "output", None))
        as_json = getattr(args, "json", False)
        if scmd in ("status", "plan", "next"):
            from .studio.state import render_studio_state
            state = st_mod.build_studio_state(prj, base)
            if as_json:
                print(json.dumps(state, indent=2))
            elif scmd == "status":
                print(render_studio_state(state))
            else:
                print(f"PRODUCTION PLAN | stage: {state['current_stage']} "
                      f"(confidence {state['confidence']:.2f})")
                for i, a in enumerate(state["next_actions"], 1):
                    tag = ("" if a["safe_during_training"]
                           else "  [not during training]")
                    print(f"  {i}. {a['title']} [{a['priority']}]{tag}")
                    print(f"     {a['reason']}")
                    print(f"     $ {a['command']}")
        elif scmd == "readiness":
            from .studio.readiness import (batch_readiness,
                                           dataset_readiness, run_readiness)
            try:
                if args.run:
                    conn = manifest.connect(prj.output_path, readonly=True)
                    rec = run_readiness(conn, prj, args.run)
                    conn.close()
                elif args.batch:
                    rec = batch_readiness(base, args.batch)
                else:
                    conn = manifest.connect(prj.output_path, readonly=True)
                    rec = dataset_readiness(conn, args.dataset or 1)
                    conn.close()
            except (KeyError, FileNotFoundError) as exc:
                print(f"FATAL: {exc}")
                return
            if args.json:
                print(json.dumps(rec, indent=2))
            else:
                print(f"READINESS: {rec.get('readiness_level', '?').upper()}"
                      f"  ({rec['readiness_score']:.2f})")
                print(f"  {rec['recommendation']}")
                for b in rec.get("blockers", []):
                    print(f"  BLOCKER: {b}")
                for w in rec.get("warnings", []):
                    print(f"  WARNING: {w}")
                print(f"  reason codes: "
                      f"{', '.join(rec.get('reason_codes', []))}")
        elif scmd == "assets":
            from .studio.registry import render_registry
            conn = None
            try:
                conn = manifest.connect(prj.output_path, readonly=True)
            except Exception:
                pass
            reg = st_mod.build_asset_registry(
                prj, conn=conn, output_base=base, role=args.role,
                missing_metadata=args.missing_metadata)
            if conn is not None:
                conn.close()
            if args.json:
                print(json.dumps(reg, indent=2))
            else:
                print(render_registry(reg))
        elif scmd == "export":
            res = st_mod.export_studio_handoff(
                prj, base, run_name=args.run, batch_id=args.batch,
                dataset_version=args.dataset,
                include_samples=args.include_samples, dry_run=args.dry_run)
            if args.json:
                print(json.dumps(res, indent=2))
            else:
                if res.get("package_dir"):
                    print(f"STUDIO HANDOFF -> {res['package_dir']}")
                else:
                    print("STUDIO HANDOFF (dry run - nothing written)")
                print(f"  included: {', '.join(res['included'])}")
                for m in res["missing"]:
                    print(f"  PENDING: {m}")
    elif cmd == "learn":
        from . import studio as st_mod
        lcmd = getattr(args, "learn_cmd", None) or "profile"
        as_json = getattr(args, "json", False)
        if lcmd == "rate":
            conn = manifest.connect(prj.output_path)
            try:
                res = st_mod.rate_generation(
                    conn, image_path=args.image,
                    feedback_id=args.feedback_id, rating=args.rating,
                    tags=[t.strip() for t in args.tags.split(",")
                          if t.strip()])
                st_mod.aggregate_preferences(conn)   # refresh profile
            except KeyError as exc:
                print(f"FATAL: {exc}")
                conn.close()
                return
            conn.close()
            print(f"Rated {res['feedback_id']} -> {res['rating']} "
                  f"(tags: {', '.join(res['tags']) or 'none'})")
        elif lcmd == "recommend":
            conn = manifest.connect(prj.output_path, readonly=True)
            rec = st_mod.recommend_from_preferences(conn, seed=args.seed)
            conn.close()
            if as_json:
                print(json.dumps(rec, indent=2))
            elif rec["status"] == "cold_start":
                print(f"COLD START: {rec['reason']}")
            else:
                print(f"LEARNED RECOMMENDATION (mean reward "
                      f"{rec['mean_reward']:.2f}):")
                for l in rec["loras"]:
                    print(f"  {l['name']:<40} "
                          f"weight {l['preferred_weight_bucket']} "
                          f"(reward {l['mean_reward']:.2f}, n={l['n']})")
                if rec["recipe"]:
                    print(f"  best recipe: {rec['recipe']}")
                if rec.get("seed"):
                    print(f"  strong seed: {rec['seed']}")
        elif lcmd == "ingest":
            conn = manifest.connect(prj.output_path)
            res = st_mod.ingest_directory(
                conn, args.dir, reference_image=args.reference_image,
                prj=prj, recipe=args.recipe,
                recursive=not args.no_recursive, limit=args.limit)
            st_mod.aggregate_preferences(conn)       # refresh profile
            conn.close()
            if as_json:
                print(json.dumps(res, indent=2))
            elif res.get("error"):
                print(f"FATAL: {res['error']}")
            else:
                print(f"INGEST {res['directory']}")
                print(f"  scanned {res['scanned']} | recorded "
                      f"{res['recorded']} new | updated {res['updated']} | "
                      f"identity-scored {res['scored']}")
                print("  rate your favorites: pixel-curator learn rate "
                      "--image <path> --rating 5")
        elif lcmd == "auto":
            dirs = args.dir or []
            if not dirs:
                # sensible defaults: the app's single-gen folder + Forge's
                # gallery, if they exist
                cand = [Path(prj.output_path) / "evals" / "_single"]
                if getattr(prj, "forge_root", ""):
                    cand.append(Path(prj.forge_root).expanduser()
                                / "outputs" / "txt2img-images")
                dirs = [str(c) for c in cand if c.exists()]
            conn = manifest.connect(prj.output_path)
            summary = []
            for d in dirs:
                r = st_mod.ingest_directory(
                    conn, d, reference_image=args.reference_image, prj=prj)
                summary.append(r)
            prof = st_mod.aggregate_preferences(conn)
            conn.close()
            if as_json:
                print(json.dumps({"ingested": summary,
                                  "profile_samples": prof["samples"]},
                                 indent=2))
            else:
                for r in summary:
                    if r.get("error"):
                        print(f"  skip: {r['error']}")
                    else:
                        print(f"  {r['directory']}: +{r['recorded']} new, "
                              f"{r['updated']} updated")
                print(f"Profile rebuilt: {prof['samples']} generations "
                      f"known.")
        elif lcmd == "harvest":
            conn = manifest.connect(prj.output_path, readonly=not args.apply)
            plan = st_mod.stage_harvest(
                conn, out_base(args.output), min_rating=args.min_rating,
                min_identity=args.min_identity, apply=args.apply)
            conn.close()
            if as_json:
                print(json.dumps(plan, indent=2))
            else:
                print(f"HARVEST: {plan['candidate_count']} candidate "
                      f"image(s) (rating>={args.min_rating}, "
                      f"identity>={args.min_identity})")
                if plan["applied"]:
                    print(f"  staged {plan.get('copied', 0)} -> "
                          f"{plan.get('staged_dir')}")
                else:
                    print("  plan only - add --apply to stage the images")
                print(f"  {plan['next_step']}")
        else:                                   # profile
            conn = manifest.connect(
                prj.output_path,
                readonly=not getattr(args, "rebuild", False))
            prof = (st_mod.aggregate_preferences(conn)
                    if getattr(args, "rebuild", False)
                    else st_mod.load_preferences(conn)
                    or st_mod.aggregate_preferences(conn, store=False))
            conn.close()
            if as_json:
                print(json.dumps(prof, indent=2))
            else:
                print(st_mod.render_profile(prof))
    elif cmd == "recommend-stack":
        from .artistic.stack_recommender import (
            recommend_best_epoch_and_stack, recommend_best_stack,
            recommend_figure_study_stack, render_epoch_stack,
            render_figure_study_recommendation, render_recommendation)
        intensity = "strict" if args.strict else "standard"
        preset_label = ""
        try:
            if args.recipe:
                conn = (manifest.connect(out_base(args.output),
                                         readonly=True)
                        if args.dataset else None)
                rec = recommend_figure_study_stack(
                    args.recipe, prj=prj, conn=conn,
                    dataset_version=args.dataset,
                    identity_strength=args.identity_strength,
                    concept_strength=args.concept_strength,
                    negative_intensity=intensity)
                rendered = render_figure_study_recommendation(rec)
                preset_label = f"figure_study:{args.recipe}"
            elif args.run or args.dataset is None:
                conn = manifest.connect(out_base(args.output),
                                        readonly=True)
                rec = recommend_best_epoch_and_stack(conn, prj, args.run)
                rendered = render_epoch_stack(rec)
                preset_label = f"best_epoch:{rec['run']}"
            else:
                conn = manifest.connect(out_base(args.output))
                rec = recommend_best_stack(
                    conn, prj, args.dataset,
                    identity_strength=args.identity_strength,
                    concept_strength=args.concept_strength)
                rendered = render_recommendation(rec)
                preset_label = f"smart_stack:v{args.dataset:03d}"
        except (KeyError, ValueError) as exc:
            print(f"FATAL: {exc}")
            return
        if args.json:
            print(json.dumps(rec, indent=2))
        else:
            print(rendered)
        if args.send_to_playground:
            from .pipeline_dag import write_playground_preset
            loras = [(x["name"], x["weight"])
                     for x in rec["recommended_stack"]]
            target = write_playground_preset(prj, preset_label, loras)
            print(f"Playground preset written -> {target}")
    elif cmd == "sweep":
        from .sweep import SweepConfig, sweep_generator
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]
        cfg = SweepConfig(
            output_base=out_base(args.output), run=args.run,
            backend=args.backend, forge_url=args.forge_url,
            checkpoint=args.checkpoint, weight=args.weight,
            max_gap=args.max_gap, limit_epochs=args.limit_epochs,
            categories=cats or ["likeness", "flexibility"],
        )
        for update in sweep_generator(prj, cfg):
            print(update)
    elif cmd == "analyze":
        from .wizard import analyze, detect_type
        conn = manifest.connect(prj.output_path)
        a = analyze(conn)
        print(json.dumps(a, indent=2))
        print("\nType detection:")
        for r in detect_type(a):
            print(f"  {r['type']:<10} {r['score']:.2f}  ({r['reason']})")
    elif cmd == "wizard":
        from .wizard import WizardConfig, wizard_generator
        cfg = WizardConfig(
            output_base=out_base(args.output), lora_type=args.lora_type,
            trigger=args.trigger, class_word=args.class_word, name=args.name,
            train=args.train, matrix=args.matrix, matrix_backend=args.backend,
        )
        for update in wizard_generator(prj, cfg):
            print(update)
    elif cmd == "merge":
        from .eval.matrix import parse_lora_specs
        from .merge import MergeConfig, merge_generator, parse_block_string
        loras = []
        for spec in args.lora:
            loras.extend(parse_lora_specs(spec))
        cfg = MergeConfig(
            output_base=out_base(args.output), loras=loras,
            output_name=args.name,
            default_blocks=parse_block_string(args.blocks),
        )
        for update in merge_generator(prj, cfg):
            print(update)
        if args.preview:
            from .merge import safe_slug as _slug
            from .eval.matrix import MatrixConfig, matrix_generator
            lora_dir = Path(prj.lora_output_dir
                            or (prj.output_path / "LORA_OUTPUT")).expanduser()
            merged = lora_dir / f"{_slug(args.name)}.safetensors"
            if merged.exists():
                for update in matrix_generator(prj, MatrixConfig(
                    output_base=out_base(args.output), lora=str(merged),
                    label=f"{_slug(args.name)}_preview",
                    categories=["likeness"], backend=args.backend,
                    checkpoint=prj.base_model,
                )):
                    print(update)
    elif cmd == "combo":
        from .merge import combo_generator
        for update in combo_generator(
            prj, out_base(args.output), character=args.character,
            style=args.style, outfit=args.outfit, name=args.name,
            char_w=args.char_weight, style_w=args.style_weight,
            outfit_w=args.outfit_weight,
        ):
            print(update)
    elif cmd == "testgen":
        if args.backend == "forge":
            from .eval.forge_dispatch import dispatch_forge_generate
            payload = {
                "url": args.forge_url, "prompt": args.prompt,
                "neg": args.negative,
                "loras": ",".join(args.lora),
                "steps": args.steps, "cfg": args.cfg, "seed": args.seed,
                "w": args.width, "h": args.height,
                "strength": args.strength,
                "checkpoint": args.checkpoint,
                "sampler": args.sampler, "clip_skip": args.clip_skip,
                "init": args.init_image, "region": args.region,
                "mask": args.mask,
                "face_swap": args.face_swap,
                "reference_image": args.reference_image,
                "upscale": args.upscale,
                "restore_faces": args.restore_faces,
            }
            for ctype in ("openpose", "depth", "canny", "softedge"):
                p = getattr(args, f"cn_{ctype}")
                if p:
                    payload[f"cn_{ctype}_image"] = p
            result = dispatch_forge_generate(payload, out_base=out_base(),
                                             project=prj)
            if not result.get("ok"):
                print(result.get("error", "generation failed"))
                return 1
            if args.out:
                Path(args.out).write_bytes(
                    base64.b64decode(result["image_b64"]))
            print(f"Saved: {result['path']}")
            if args.out and Path(args.out).resolve() != Path(result["path"]).resolve():
                print(f"Copy:  {args.out}")
            if result.get("face_locked"):
                print("Face locked to reference.")
            if result.get("post_process"):
                print("Post-process: " + ", ".join(result["post_process"]))
            for w in result.get("lora_warnings", []):
                print(f"warning: {w}")
            return
        from .eval.matrix import parse_lora_specs
        loras = []
        for spec in args.lora:
            loras.extend(parse_lora_specs(spec))
        from .eval.pipeline import diffusers_available, get_pipeline
        ok, reason = diffusers_available()
        if not ok:
            print(reason)
            return
        ckpt = args.checkpoint or prj.base_model
        tp = get_pipeline(ckpt, "", loras)
        if args.init_image:
            from PIL import Image
            init = Image.open(Path(args.init_image).expanduser()).convert("RGB")
            img = tp.img2img(init, args.prompt, args.negative,
                             args.strength, args.steps, args.cfg, args.seed)
        else:
            img = tp.txt2img(args.prompt, args.negative, args.steps,
                             args.cfg, args.width, args.height, args.seed)
        img.save(args.out)
        print(f"Saved: {args.out}")
    elif cmd == "matrix":
        from .eval.matrix import CATEGORIES, MatrixConfig, matrix_generator
        cats = ([c.strip() for c in args.categories.split(",") if c.strip()]
                or list(CATEGORIES))
        cfg = MatrixConfig(
            output_base=out_base(args.output), lora=args.lora, label=args.label,
            categories=cats, backend=args.backend, forge_url=args.forge_url,
            checkpoint=args.checkpoint, lora_weight=args.weight,
            steps=args.steps, cfg=args.cfg, sampler=args.sampler,
            clip_skip=args.clip_skip, pony_prefix=args.pony,
        )
        for update in matrix_generator(prj, cfg):
            print(update)
    elif cmd == "evals":
        from .eval.matrix import eval_summary
        conn = manifest.connect(prj.output_path)
        rows = eval_summary(conn)
        if not rows:
            print("No evals yet. Run: pixel-curator matrix --lora <path>")
        cur = None
        for r in rows:
            if r["label"] != cur:
                cur = r["label"]
                print(f"  {cur}")
            print(f"    {r['category']:<12} n={r['n']:<4} avg_likeness={r['avg_likeness']}")
    elif cmd == "presets":
        from .train import PRESETS
        for name, p in PRESETS.items():
            print(f"  {name:<10} dim {p['network_dim']}/{p['network_alpha']:<3} "
                  f"lr {p['unet_lr']}/{p['te_lr']} batch {p['batch_size']} "
                  f"epochs {p['epochs']}  - {p['notes']}")
    elif cmd == "suggest-preset":
        from .train.preset_recommender import (format_recommendation,
                                               suggest_preset)
        conn = manifest.connect(out_base(args.output))
        try:
            rec = suggest_preset(conn, args.dataset,
                                 recipe_name=args.recipe)
            params = None
            if getattr(args, "with_params", True) \
                    and rec.get("recommended_preset"):
                from .train.param_advisor import (recommend_training_params,
                                                  render_params)
                try:
                    params = recommend_training_params(
                        conn, args.dataset,
                        preset_name=rec["recommended_preset"],
                        recipe_name=args.recipe)
                except Exception as exc:
                    params = None
                    print(f"(parameter advisor unavailable: {exc})")
        except ValueError as exc:
            print(f"FATAL: {exc}")
            raise SystemExit(1)
        finally:
            conn.close()
        if args.json:
            out = dict(rec)
            if params:
                out["parameter_recommendation"] = params
            print(json.dumps(out, indent=2))
        else:
            print(format_recommendation(rec))
            if params:
                print(render_params(params))
            print(f"\nTrain with it:\n  pixel-curator -p "
                  f"{args.project or 'project.toml'} train "
                  f"--dataset {args.dataset} "
                  f"--preset {rec['recommended_preset']} --dry-run")
    elif cmd == "training-config":
        from .train.export_config import generate_training_bundle

        def _coerce(v: str):
            for cast in (int, float):
                try:
                    return cast(v)
                except ValueError:
                    pass
            if v.lower() in ("true", "false"):
                return v.lower() == "true"
            return v

        overrides = {}
        for kv in args.set:
            if "=" not in kv:
                print(f"FATAL: --set expects KEY=VALUE, got {kv!r}")
                raise SystemExit(1)
            k, v = kv.split("=", 1)
            overrides[k.strip()] = _coerce(v.strip())
        conn = manifest.connect(out_base(args.output))
        try:
            target_dir = args.dir
            if target_dir is None and args.dataset is not None:
                row = conn.execute(
                    "SELECT dir FROM datasets WHERE version = ?",
                    (args.dataset,)).fetchone()
                target_dir = row[0] if row and row[0] else None
            if target_dir is None:
                print("FATAL: no target folder - pass --dir (or --dataset "
                      "with a registered directory)")
                raise SystemExit(1)
            result = generate_training_bundle(
                conn, target_dir, dataset_version=args.dataset,
                frame_count=args.frames, model_type=args.model_type,
                preset=args.preset, base_model=args.base_model,
                output_name=args.name, overrides=overrides or None,
                write=not args.dry_run)
        except ValueError as exc:
            print(f"FATAL: {exc}")
            raise SystemExit(1)
        finally:
            conn.close()
        if args.json:
            print(json.dumps(result["bundle"], indent=2))
        elif args.dry_run:
            print(result["toml_text"])
            print(f"# launch: {result['command']}")
        else:
            print("Training config bundle written:")
            for k, p in result["paths"].items():
                print(f"  {k:5} {p}")
            print(f"launch: {result['command']}")
            for h in result["bundle"]["experiment_hints"]:
                print(f"  experience: {h['text']}")
            for w in result["warnings"]:
                print(f"  ! {w}")
    elif cmd == "train-legacy":
        from .train.kohya import TrainConfig, train_generator
        overrides = {}
        for kv in args.set:
            if "=" in kv:
                k, v = kv.split("=", 1)
                overrides[k.strip()] = v.strip()
        cfg = TrainConfig(
            output_base=out_base(args.output),
            dataset_version=args.dataset, preset=args.preset,
            name=args.name, overrides=overrides, dry_run=args.dry_run,
            min_free_gb=args.min_free_gb,
            register_experiment=args.register_experiment,
        )
        for update in train_generator(prj, cfg):
            print(update)
    elif cmd == "status":
        from .monitor import (FollowConfig, follow_generator, render_status,
                              status_json, status_snapshot)
        base = out_base(args.output)
        stall = getattr(args, "stall_seconds", 900.0)
        if args.json:
            print(json.dumps(status_snapshot(prj, base, stall_seconds=stall),
                             indent=2, default=str))
        elif args.follow > 0:
            fcfg = FollowConfig(output_base=base, interval=args.follow,
                                notify=args.notify, stall_seconds=stall)
            try:
                for update in follow_generator(prj, fcfg):
                    print(f"[{__import__('datetime').datetime.now():%H:%M:%S}] "
                          f"{update}\n")
            except KeyboardInterrupt:
                print("\nstatus --follow stopped (training continues).")
        else:
            print(render_status(status_snapshot(prj, base, stall_seconds=stall)))
    elif cmd == "generate-artistic":
        from .artistic import (compose_artistic_prompt, get_recipe,
                               list_recipes)
        if args.list_recipes or not args.recipe:
            print("Artistic Figure Study recipes:")
            for r in list_recipes():
                layers = ", ".join(f"{l['name']}({l['role']} "
                                   f"{l['weight']:.2f})"
                                   for l in r["layers"])
                print(f"  {r['name']:<26} {r['description']}\n"
                      f"  {'':<26} layers: {layers}")
            return
        try:
            recipe = get_recipe(args.recipe)
        except KeyError as exc:
            print(f"FATAL: {exc}")
            return
        from .artistic.workflows import (build_artistic_txt2img_payload,
                                         build_selective_region_payload,
                                         resolve_identity_lora)
        id_name, id_warnings = resolve_identity_lora(prj)
        trig = f"{prj.trigger_token} {prj.class_word}".strip()
        if args.batch > 1 and not args.image:
            # controlled-variation batch (Phase 4) - shared planner
            from .artistic import (plan_figure_study_batch,
                                   write_batch_manifest)
            axes = [a.strip() for a in args.variation.split(",")
                    if a.strip()]
            try:
                plan = plan_figure_study_batch(
                    recipe, count=args.batch, base_seed=args.seed,
                    variation_axes=axes or None, trigger_phrase=trig,
                    identity_lora_name=id_name,
                    identity_strength=args.identity_strength,
                    concept_strength=args.concept_strength,
                    strength=args.strength)
            except KeyError as exc:
                print(f"FATAL: {exc}")
                return
            print(f"ARTISTIC BATCH '{plan['batch_name']}' - "
                  f"{plan['count']} items")
            for it in plan["items"]:
                print(f"    #{it['index']} seed {it['seed']:<8} "
                      f"{', '.join(it['variation'].values())}")
            for w in id_warnings + plan["warnings"]:
                print(f"  WARNING: {w}")
            if args.json:
                print(json.dumps(plan, indent=2))
            if not args.dry_run:
                target = write_batch_manifest(plan, out_base(args.output))
                print(f"  Batch manifest written -> {target}")
            return
        composed = compose_artistic_prompt(
            recipe, trigger_phrase=trig,
            identity_strength=args.identity_strength,
            concept_strength=args.concept_strength,
            strength=args.strength, identity_lora_name=id_name)
        composed.warnings = id_warnings + composed.warnings

        print(f"ARTISTIC RECIPE '{composed.recipe_name}'")
        print(f"  identity preservation estimate: "
              f"{composed.identity_score_estimate:.2f}")
        print(f"  prompt: {composed.prompt}")
        print(f"  negative: {composed.negative_prompt}")
        print("  stack (identity first):")
        for s in composed.lora_stack:
            print(f"    {s['name']:<28} {s['role']:<24} {s['weight']:.2f}")
        for w in composed.warnings:
            print(f"  WARNING: {w}")
        print(f"  reason codes: {', '.join(composed.reason_codes)}")

        if args.dry_run:
            return
        if args.image:
            res = build_selective_region_payload(
                prj, composed, image_path=args.image,
                workflow=args.workflow, seed=args.seed,
                identity_reference=args.identity_reference,
                pose_guidance=True)   # region edits preserve pose
        else:
            res = build_artistic_txt2img_payload(
                prj, composed, seed=args.seed,
                identity_reference=args.identity_reference,
                pose_guidance=args.pose_guidance)
        for r in res.recommendations:
            print(f"  RECOMMEND: {r}")
        if args.json:
            safe = {k: v for k, v in res.payload.items()
                    if k not in ("init_images", "mask")}
            print(json.dumps({"workflow": res.workflow, "payload": safe,
                              "reason_codes": res.reason_codes}, indent=2))
        if args.send_to_playground:
            from .pipeline_dag import write_playground_preset
            loras = [(s["name"], s["weight"]) for s in composed.lora_stack
                     if s["weight"] > 0]
            target = write_playground_preset(
                prj, f"artistic:{composed.recipe_name}", loras)
            print(f"  Playground preset written -> {target}")
        if not (args.json or args.send_to_playground):
            print("  (payload built - add --json to inspect, "
                  "--send-to-playground to hand off)")
    elif cmd == "generate-figure-study":
        from .artistic import (compose_artistic_prompt, get_recipe,
                               list_negative_chains, list_recipes,
                               list_variation_axes,
                               plan_figure_study_batch,
                               write_batch_manifest)
        if args.list_axes:
            print("Variation axes (controlled variation):")
            for a in list_variation_axes():
                star = "*" if a["default"] else " "
                print(f"  {star} {a['name']:<20} "
                      f"{', '.join(a['options'])}")
            print("\nNegative control chains:")
            for c in list_negative_chains():
                crit = " (critical)" if c["critical"] else ""
                print(f"    {c['name']:<26}{crit} "
                      f"{', '.join(c['terms'][:4])}...")
            return
        if args.list_recipes:
            print("Figure study recipes:")
            for r in list_recipes():
                print(f"  {r['name']:<34} {r['description']}")
            return
        try:
            recipe = get_recipe(args.recipe)
        except KeyError as exc:
            print(f"FATAL: {exc}")
            return
        from .artistic.workflows import (build_figure_study_region_payload,
                                         resolve_identity_lora)
        id_name, id_warnings = resolve_identity_lora(prj)
        trig = f"{prj.trigger_token} {prj.class_word}".strip()
        intensity = "strict" if args.strict else "standard"
        chains = [c.strip() for c in args.negative_chains.split(",")
                  if c.strip()]
        # --variation: float -> variation strength; names -> axes list
        axes: list[str] = []
        variation_strength = None
        if args.variation:
            try:
                variation_strength = float(args.variation)
            except ValueError:
                axes = [a.strip() for a in args.variation.split(",")
                        if a.strip()]

        # ----- guidance plan (planning-only; never loads models) -----
        guidance_requested = bool(args.controlnet or args.ip_adapter
                                  or args.faceid or args.pose_image
                                  or args.mask or args.reference_image)
        guidance_plan: dict = {}
        if guidance_requested or recipe.guidance_recommendations:
            from .artistic.guidance import (plan_figure_study_guidance,
                                            render_guidance)
            try:
                guidance_plan = plan_figure_study_guidance(
                    prj, region=(args.region if (args.image or
                                                 guidance_requested)
                                 else recipe.region_policy),
                    controlnet=args.controlnet,
                    ip_adapter=args.ip_adapter, faceid=args.faceid,
                    reference_image=(args.reference_image
                                     or args.identity_reference),
                    pose_image=args.pose_image, mask_image=args.mask,
                    identity_strength=args.identity_strength or 0.85,
                    recipe_guidance=recipe.guidance_recommendations)
            except KeyError as exc:
                print(f"FATAL: {exc}")
                return

        production = bool(args.passes or args.refine or args.select_best
                          or args.max_variants
                          or (guidance_requested and not args.image))
        if production and not args.image:
            from .artistic import (plan_production_batch, resolve_passes,
                                   write_production_batch)
            from .artistic.guidance import render_guidance
            from .artistic.review import select_best_figure_study_variant
            pass_spec = [p.strip() for p in args.passes.split(",")
                         if p.strip()]
            if args.refine:
                pass_spec = (pass_spec or ["composition"]) + ["refine"]
            if not pass_spec:
                pass_spec = list(recipe.batch_defaults.get(
                    "passes", ["composition", "review"]))
            if "review" not in pass_spec:
                pass_spec.append("review")
            count = args.batch if args.batch > 1 else \
                int(recipe.batch_defaults.get("count", args.batch))
            try:
                plan = plan_production_batch(
                    recipe, count=count, base_seed=args.seed,
                    passes=pass_spec,
                    variation_strength=variation_strength,
                    variation_axes=(axes
                                    or recipe.variation_axes_default
                                    or None),
                    guidance=guidance_plan,
                    refine_strength=args.refine_strength,
                    max_variants=args.max_variants,
                    trigger_phrase=trig, identity_lora_name=id_name,
                    identity_strength=args.identity_strength,
                    concept_strength=args.concept_strength,
                    strength=args.strength,
                    negative_chains=chains or None,
                    negative_intensity=intensity)
            except KeyError as exc:
                print(f"FATAL: {exc}")
                return
            print(f"FIGURE STUDY PRODUCTION BATCH '{plan['batch_name']}'")
            print(f"  recipe: {plan['recipe']} | items: "
                  f"{plan['summary']['total_items']} | identity estimate "
                  f"min {plan['summary']['identity_estimate_min']:.2f}")
            for p in plan["passes"]:
                print(f"  {p['pass_id']:<28} {len(p['items']):>3} items  "
                      f"{p['purpose']}")
            if guidance_plan:
                print("  " + render_guidance(guidance_plan)
                      .replace("\n", "\n  "))
            for w in id_warnings + plan["warnings"]:
                print(f"  WARNING: {w}")
            for n in recipe.runtime_notes:
                print(f"  NOTE: {n}")
            if args.select_best:
                best = select_best_figure_study_variant(plan)
                print(f"  BEST VARIANT (planned metrics): "
                      f"{best['selected']} score {best['score']}")
            if args.json:
                print(json.dumps(plan, indent=2))
            if not args.dry_run:
                target = write_production_batch(plan,
                                                out_base(args.output))
                print(f"  Production batch written -> {target}")
                print(f"  Review it: pixel-curator figure-review --batch "
                      f"{target.name} --summary")
            return

        if args.image:
            # region-aware img2img figure study (single item)
            composed = compose_artistic_prompt(
                recipe, trigger_phrase=trig,
                identity_strength=args.identity_strength,
                concept_strength=args.concept_strength,
                strength=args.strength, identity_lora_name=id_name,
                negative_chains=chains or None,
                negative_intensity=intensity)
            composed.warnings = id_warnings + composed.warnings
            try:
                res = build_figure_study_region_payload(
                    prj, composed, image_path=args.image,
                    region=args.region, denoise=args.denoise,
                    seed=args.seed,
                    identity_reference=args.identity_reference)
            except KeyError as exc:
                print(f"FATAL: {exc}")
                return
            print(f"FIGURE STUDY (region: {args.region}) | recipe "
                  f"'{composed.recipe_name}'")
            print(f"  identity preservation estimate: "
                  f"{composed.identity_score_estimate:.2f}")
            for s in composed.lora_stack:
                print(f"    {s['name']:<28} {s['role']:<24} "
                      f"{s['weight']:.2f}")
            for w in res.warnings:
                print(f"  WARNING: {w}")
            for r in res.recommendations:
                print(f"  RECOMMEND: {r}")
            if args.json:
                safe = {k: v for k, v in res.payload.items()
                        if k not in ("init_images", "mask")}
                print(json.dumps({"workflow": res.workflow,
                                  "payload": safe,
                                  "reason_codes": res.reason_codes},
                                 indent=2))
        else:
            if variation_strength is not None and not axes:
                from .artistic import axes_for_variation_strength
                axes = axes_for_variation_strength(variation_strength)
            try:
                plan = plan_figure_study_batch(
                    recipe, count=args.batch, base_seed=args.seed,
                    variation_axes=axes or None, trigger_phrase=trig,
                    identity_lora_name=id_name,
                    identity_strength=args.identity_strength,
                    concept_strength=args.concept_strength,
                    strength=args.strength,
                    negative_chains=chains or None,
                    negative_intensity=intensity)
            except KeyError as exc:
                print(f"FATAL: {exc}")
                return
            if guidance_plan:
                from .artistic.guidance import render_guidance
                print("  " + render_guidance(guidance_plan)
                      .replace("\n", "\n  "))
            print(f"FIGURE STUDY BATCH '{plan['batch_name']}' - "
                  f"{plan['count']} items, axes: "
                  f"{', '.join(plan['variation_axes'])}")
            print(f"  identity estimate: min "
                  f"{plan['summary']['identity_estimate_min']:.2f} / mean "
                  f"{plan['summary']['identity_estimate_mean']:.2f}")
            for it in plan["items"]:
                var = ", ".join(it["variation"].values())
                print(f"    #{it['index']} seed {it['seed']:<8} {var}")
            for w in id_warnings + plan["warnings"]:
                print(f"  WARNING: {w}")
            if args.json:
                print(json.dumps(plan, indent=2))
            if not args.dry_run:
                target = write_batch_manifest(plan, out_base(args.output))
                print(f"  Batch manifest written -> {target}")
            if args.send_to_playground and plan["items"]:
                from .pipeline_dag import write_playground_preset
                loras = [(s["name"], s["weight"])
                         for s in plan["items"][0]["lora_stack"]
                         if s["weight"] > 0]
                target = write_playground_preset(
                    prj, f"figure_study:{recipe.name}", loras)
                print(f"  Playground preset written -> {target}")
    elif cmd == "figure-review":
        from .artistic.review import (export_review, find_batch_dir,
                                      list_batches, load_batch_manifest,
                                      mark_item, render_review_summary,
                                      review_summary,
                                      select_best_figure_study_variant)
        base = out_base(args.output)
        if args.list_batches or not args.batch:
            batches = list_batches(base)
            if not batches:
                print("No figure study batches yet. Plan one with: "
                      "pixel-curator generate-figure-study --batch 6")
                return
            print("Figure study batches (newest first):")
            for b in batches:
                sel = (f" selected: {b['selected_variant']}"
                       if b["selected_variant"] else "")
                print(f"  {b['batch_id']:<44} {b['recipe']:<30} "
                      f"{b['items']:>3} items  {b['review_status']}{sel}")
            return
        try:
            bd = find_batch_dir(base, args.batch)
        except FileNotFoundError as exc:
            print(f"FATAL: {exc}")
            return
        try:
            if args.select:
                entry = mark_item(bd, args.select, status="selected",
                                  tags=args.tag, notes=args.notes,
                                  next_action=args.next_action)
                print(f"Selected variant: {entry['item_id']}")
            if args.mark:
                item, status = args.mark
                entry = mark_item(bd, item, status=status, tags=args.tag,
                                  notes=args.notes,
                                  next_action=args.next_action)
                print(f"Marked {entry['item_id']} -> "
                      f"{entry['review_status']}")
            elif args.tag and not args.select:
                print("NOTE: --tag applies with --mark or --select.")
        except KeyError as exc:
            print(f"FATAL: {exc}")
            return
        if args.best:
            import json as _json
            review = _json.loads(
                (bd / "review_manifest.json").read_text(encoding="utf-8"))
            best = select_best_figure_study_variant(
                load_batch_manifest(bd), review)
            print(f"BEST VARIANT: {best['selected']} "
                  f"(score {best.get('score')}; "
                  f"{', '.join(best.get('reason_codes', []))})")
            for c in best["candidates"][:5]:
                print(f"  {c['item_id']:<12} score {c['score']:<8} "
                      f"{c['review_status']}")
        if args.export:
            print(json.dumps(export_review(bd), indent=2))
        elif args.summary or not (args.select or args.mark or args.best):
            s = review_summary(bd)
            if args.json:
                print(json.dumps(s, indent=2))
            else:
                print(render_review_summary(s))
    elif cmd == "package-figure-study":
        from .artistic.packaging import package_figure_study
        base = out_base(args.output)
        conn = None
        try:
            conn = manifest.connect(base, readonly=True)
        except Exception:
            pass
        try:
            res = package_figure_study(base, args.batch, conn=conn,
                                       prj=prj)
        except FileNotFoundError as exc:
            print(f"FATAL: {exc}")
            return
        finally:
            if conn is not None:
                conn.close()
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"FIGURE STUDY PACKAGE -> {res['package_dir']}")
            print(f"  batch: {res['batch_id']} | assets copied: "
                  f"{res['copied_assets']}")
            if res["best_variant"].get("selected"):
                print(f"  best variant: "
                      f"{res['best_variant']['selected']}")
            for m in res["missing"]:
                print(f"  PENDING: {m}")
            print("  README.md summarizes contents and next steps.")
    elif cmd == "render-figure-study":
        from .artistic.batch_render import render_batch
        from .artistic.review import find_batch_dir
        base = out_base(args.output)
        try:
            bd = find_batch_dir(base, args.batch)
        except FileNotFoundError as exc:
            print(f"FATAL: {exc}")
            return
        gen = render_batch(
            bd, prj, forge_url=args.forge_url, checkpoint=args.checkpoint,
            clip_skip=args.clip_skip, steps=args.steps, cfg=args.cfg,
            width=args.width, height=args.height, limit=args.limit,
            overwrite=args.overwrite, face_lock=args.face_lock,
            reference_image=args.reference_image)
        try:
            while True:
                print(next(gen))
        except StopIteration:
            pass
    elif cmd == "runs":
        from .train.kohya import list_runs
        conn = manifest.connect(prj.output_path, readonly=True)
        rows = list_runs(conn)
        if not rows:
            print("No training runs yet. Run: pixel-curator train-legacy --dataset N --preset character")
        for r in rows:
            print(f"  #{r['run_id']:<3} {r['name']:<28} v{r['dataset_version']:03d} "
                  f"{r['preset']:<9} {r['status']:<9} "
                  f"step {r['step']}/{r['total_steps']} loss {r['last_loss']}")
    elif cmd == "exp":
        from . import experiments as xp

        def _kv(pairs):
            out = {}
            for spec in pairs or []:
                k, sep, v = str(spec).partition("=")
                if not sep or not k.strip():
                    print(f"WARNING: bad --param {spec!r} "
                          "(expected KEY=VALUE) - ignored")
                    continue
                v = v.strip()
                low = v.lower()
                if low in ("true", "false"):
                    out[k.strip()] = (low == "true")
                    continue
                try:
                    out[k.strip()] = int(v)
                except ValueError:
                    try:
                        out[k.strip()] = float(v)
                    except ValueError:
                        out[k.strip()] = v
            return out

        conn = manifest.connect(prj.output_path)
        act = args.exp_action
        if act == "register":
            kw = {}
            if args.from_file:
                try:
                    kw = xp.load_experiment_file(args.from_file)
                except ValueError as exc:
                    print(f"ERROR: {exc}")
                    return
            if args.model:
                kw["model_path"] = args.model
            if args.dataset is not None:
                kw["dataset_version"] = args.dataset
            if args.dataset_dir:
                kw["dataset_dir"] = args.dataset_dir
            if args.base_model:
                kw["base_model"] = args.base_model
            if args.model_type != "lora" or "model_type" not in kw:
                kw["model_type"] = args.model_type
            params = dict(kw.pop("params", {}) or {})
            params.update(_kv(args.param))
            name = args.name or kw.pop("name", "") or (
                Path(kw["model_path"]).stem if kw.get("model_path") else "")
            if not name:
                print("ERROR: give the experiment a label with --name "
                      "(or --model / --from-file to derive one).")
                return
            kw.pop("name", None)
            rec = xp.register_experiment(
                conn, name=name, params=params, note=args.note,
                outcome=args.outcome, rating=args.rating,
                source="file" if args.from_file else "manual",
                hash_model=args.hash_model, full_hash=args.full_hash,
                **kw)
            for w in rec.get("warnings", []):
                print(f"WARNING: {w}")
            print(f"Registered experiment #{rec['exp_id']} '{rec['name']}'"
                  + (f"  dataset v{rec['dataset_version']:03d}"
                     if rec.get("dataset_version") is not None else ""))
            if rec.get("sidecar"):
                print(f"sidecar: {rec['sidecar']}")
        elif act == "list":
            recs = xp.list_experiments(
                conn, dataset_version=args.dataset, model=args.model,
                outcome=args.outcome, since=args.since)
            if args.json:
                print(json.dumps(recs, indent=2, ensure_ascii=False))
            else:
                hints = xp.unregistered_runs(conn, args.dataset) \
                    if args.dataset is not None else None
                print(xp.render_experiments(recs, hints))
        elif act == "show":
            rec = xp.get_experiment(conn, args.exp)
            if not rec:
                print(f"No experiment matches {args.exp!r}. "
                      "See: pixel-curator exp list")
                return
            if args.json:
                print(json.dumps(rec, indent=2, ensure_ascii=False))
            else:
                print(xp.render_experiment_detail(conn, rec))
        elif act == "note":
            rec = xp.get_experiment(conn, args.exp)
            if not rec:
                print(f"No experiment matches {args.exp!r}.")
                return
            xp.append_note(conn, int(rec["exp_id"]), args.text)
            print(f"Note added to experiment #{rec['exp_id']} "
                  f"'{rec['name']}'.")
        elif act == "set":
            rec = xp.get_experiment(conn, args.exp)
            if not rec:
                print(f"No experiment matches {args.exp!r}.")
                return
            fields = {}
            if args.model:
                fields["model_path"] = args.model
            if args.dataset is not None:
                fields["dataset_version"] = args.dataset
            if args.name:
                fields["name"] = args.name
            if args.outcome is not None:
                fields["outcome"] = args.outcome
            if args.rating is not None:
                fields["rating"] = args.rating
            if args.base_model:
                fields["base_model"] = args.base_model
            if args.model_type:
                fields["model_type"] = args.model_type
            fresh = xp.update_experiment(
                conn, int(rec["exp_id"]), params=_kv(args.param) or None,
                full_hash=args.full_hash, **fields)
            print(f"Updated experiment #{fresh['exp_id']}.")
            print(xp.render_experiment_detail(conn, fresh))
        elif act == "from-run":
            lora_dir = Path(prj.lora_output_dir
                            or (prj.output_path / "LORA_OUTPUT"))
            rec, warns = xp.experiment_from_run(
                conn, args.run_id, lora_dir=lora_dir, note=args.note)
            for w in warns:
                print(f"NOTE: {w}")
            if rec:
                print(f"Experiment #{rec['exp_id']} '{rec['name']}' "
                      "(from run)")
                print(xp.render_experiment_detail(conn, rec))
    elif cmd == "recipes":
        if not prj.recipes:
            print("No recipes defined. Add [recipes.NAME] sections to the project file\n"
                  "or regenerate a template with: pixel-curator init")
        for name, r in sorted(prj.recipes.items()):
            kind = r.get("type", "custom")
            extra = (f"concepts={r['concepts']}" if r.get("concepts")
                     else f"repeats={r.get('repeats', 10)} max_total={r.get('max_total', 0)}")
            print(f"  {name:<18} [{kind}] {extra}")
    elif cmd == "health":
        from .health import health_generator
        for update in health_generator(prj, out_base(args.output), args.open):
            print(update)
    elif cmd == "rescore":
        from .rescore import rescore_identity_generator
        meta_dir = args.metadata_dir or prj.metadata_dir or None
        for update in rescore_identity_generator(
                prj, out_base(args.output),
                identity_threshold=args.identity_threshold,
                metadata_dir=Path(meta_dir).expanduser() if meta_dir else None,
                subject_names=tuple(args.subject_names or prj.subject_names or ()),
                sidecar_relax=args.sidecar_relax):
            print(update)
    elif cmd == "pick-epoch":
        from .eval.epoch_sweep import EpochSweepConfig, epoch_sweep_generator, parse_weights
        lora_dir = args.lora_dir or prj.lora_output_dir
        if not lora_dir:
            print("FATAL: no --lora-dir and project lora_output_dir is unset.")
            return
        cfg = EpochSweepConfig(
            output_base=out_base(args.output), lora_base=args.lora_base,
            lora_dir=Path(lora_dir).expanduser(),
            weights=parse_weights(args.weights), checkpoint=args.checkpoint,
            forge_url=args.forge_url,
            trigger=args.trigger or f"{prj.trigger_token} {prj.class_word}".strip(),
        )
        for update in epoch_sweep_generator(prj, cfg):
            print(update)
    elif cmd == "tune-stack":
        from .eval.stack_tune import (StackTuneConfig, stack_tune_generator,
                                      parse_concept_ranges)
        from .eval.epoch_sweep import parse_weights
        cfg = StackTuneConfig(
            output_base=out_base(args.output), identity_lora=args.identity,
            concepts=parse_concept_ranges(args.concepts),
            identity_weights=parse_weights(args.identity_weights),
            cap=args.cap, checkpoint=args.checkpoint, forge_url=args.forge_url,
            trigger=args.trigger or f"{prj.trigger_token} {prj.class_word}".strip(),
        )
        for update in stack_tune_generator(prj, cfg):
            print(update)
    elif cmd == "lookbook":
        from .artistic.lookbook import (LookbookConfig, lookbook_generator,
                                        parse_list, parse_lora_specs)
        cfg = LookbookConfig(
            output_base=out_base(args.output), identity_lora=args.identity,
            outfits=parse_list(args.outfits), poses=parse_list(args.poses) or ["standing"],
            identity_weight=args.weight, scene=args.scene,
            extra_loras=parse_lora_specs(args.extra_loras),
            checkpoint=args.checkpoint, forge_url=args.forge_url, seed=args.seed,
            trigger=args.trigger or f"{prj.trigger_token} {prj.class_word}".strip(),
        )
        for update in lookbook_generator(prj, cfg):
            print(update)
    elif cmd == "plan":
        from .builder import plan_generator
        for update in plan_generator(prj, out_base(args.output),
                                     args.recipe or None):
            print(update)
    elif cmd == "build":
        from .builder import BuildConfig, build_generator
        cfg = BuildConfig(
            output_base=out_base(args.output), recipe=args.recipe,
            note=args.note, link_mode=args.link_mode,
            parent=args.parent, training_config=args.training_config,
        )
        for update in build_generator(prj, cfg):
            print(update)
    elif cmd == "history":
        from . import dataset_versioning as dsv
        conn = manifest.connect(out_base(args.output))
        if args.dataset is not None:
            chain = dsv.lineage_chain(conn, args.dataset)
            if args.json:
                payload = {"lineage": chain}
                if args.curation:
                    payload["curation_snapshot"] = dsv.get_curation_snapshot(
                        conn, args.dataset)
                print(json.dumps(payload, indent=2))
            else:
                print(dsv.render_lineage(chain))
                if args.curation:
                    snap = dsv.get_curation_snapshot(conn, args.dataset)
                    print("\nCuration snapshot at creation:")
                    print(json.dumps(snap, indent=2) if snap
                          else "  (none recorded - pre-versioning build)")
        else:
            entries = dsv.version_history(conn, name=args.name)
            print(json.dumps(entries, indent=2) if args.json
                  else dsv.render_history(entries))
    elif cmd == "multi-recipe-packages":
        from .services import multi_recipe_packager as mp
        recs = []
        if args.input and Path(args.input).expanduser().exists():
            for line in Path(args.input).expanduser().read_text(
                    encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
        elif args.specialized_plan and Path(args.specialized_plan).expanduser().exists():
            data = json.loads(Path(args.specialized_plan).expanduser()
                              .read_text(encoding="utf-8"))
            recs = data.get("image_assignments", [])
        elif args.dataset:
            base = Path(args.dataset).expanduser()
            for cand in (base / "specialized_recipes" /
                         "specialized_recipe_assignments.jsonl",
                         base / "specialized_recipe_assignments.jsonl"):
                if cand.exists():
                    recs = [json.loads(l) for l in
                            cand.read_text(encoding="utf-8").splitlines() if l.strip()]
                    break
        if not recs and (prj.output_path / DB_NAME).exists():
            from .services import dataset_explorer as _dx
            if getattr(args, "process_all", False):
                erecs = _dx.load_all_image_records(
                    prj.output_path, _dx.ExplorerFilter(limit=_dx.MAX_PAGE_SIZE),
                    hard_cap=args.max_records)
                _scope = f"full manifest ({len(erecs)} records)"
            else:
                erecs = _dx.load_image_records(
                    prj.output_path, _dx.ExplorerFilter(limit=_dx.MAX_PAGE_SIZE))
                _scope = "capped page; pass --all for the full manifest"
            recs = [{**(r.metadata.get("recipe_plan") or {}),
                     "image_path": r.path, "caption_wd14": r.caption,
                     "suitability_score": (r.metadata.get("routing") or {})
                     .get("suitability_score"),
                     "architecture_target": r.architecture_target}
                    for r in erecs]
            if recs:
                print(f"loaded {len(recs)} records from manifest "
                      f"(recipe_plan merge if present; {_scope})")
        if not recs:
            print("no input records (use --input <assignments.jsonl>, "
                  "--specialized-plan <plan.json>, --dataset <dir>, or a project "
                  "manifest)")
            return
        out = Path(args.output).expanduser() if args.output else (
            (Path(args.dataset).expanduser() if args.dataset
             else (Path(args.input).expanduser().parent if args.input
                   else prj.output_path)) / "training_packages")
        cfg = mp.MultiRecipePackagerConfig(
            link_mode=args.link_mode, dry_run=not args.apply,
            overwrite=args.overwrite,
            include_dreambooth_main=args.include_dreambooth,
            include_lora_packages=args.include_lora,
            max_images_per_package=args.max_images_per_package,
            min_images_per_package=args.min_images_per_package,
            dreambooth_min_suitability=args.dreambooth_min_suitability,
            lora_min_suitability=args.lora_min_suitability,
            specialized_min_confidence=args.min_confidence,
            allow_review_records=args.allow_review_records,
            architecture_default=args.architecture,
            identity_token=getattr(prj, "trigger_token", None)
            and f"{prj.trigger_token} {getattr(prj, 'class_word', 'person')}"
            or "spikbnui person",
            class_token=getattr(prj, "class_word", "person"))
        plan = mp.build_multi_recipe_package_plan(recs, out, cfg)
        written = mp.write_multi_recipe_package_plan(plan, out,
                                                     overwrite=args.overwrite)
        mat = mp.materialize_multi_recipe_packages(plan, cfg)
        try:
            mp.write_ui_status_summary(plan, prj.output_path)
        except Exception:
            pass
        version_records: list = []
        if args.apply and args.register_version:
            version_records = mp.register_materialized_packages(
                plan, mat, prj.output_path,
                parent_version=args.parent, note=args.version_note)
        if args.json:
            print(json.dumps({"plan_id": plan.plan_id, "output_dir": str(out),
                              "dry_run": cfg.dry_run, "summary": plan.summary,
                              "materialized": mat.get("materialized", {}),
                              "registered_versions": [
                                  {k: r.get(k) for k in
                                   ("package_name", "version", "kind",
                                    "parent_version", "warnings")}
                                  for r in version_records]},
                             indent=2))
        else:
            print(mp.render_package_card(plan))
            print(f"\n{'APPLIED' if args.apply else 'DRY-RUN'} -> {out}")
            if not args.apply:
                print("  (re-run with --apply to materialize package folders)")
            for r in version_records:
                if r.get("version") is not None:
                    print(f"Registered {r['package_name']} as dataset "
                          f"version v{r['version']:03d} "
                          f"(multi_recipe_package)")
                for w in r.get("warnings", []):
                    print(f"  ! {w}")
            if args.apply and args.register_version and not version_records:
                print("  ! no packages were materialized; nothing to "
                      "register")
    elif cmd == "face-fix":
        from .services import face_consistency_studio as fcs
        cfg = fcs.FaceConsistencyConfig(
            output_dir=args.output, mode=args.mode,
            face_model_path=args.face_model,
            gfpgan_model_path=args.gfpgan_model,
            realesrgan_model_path=args.realesrgan_model,
            gfpgan_strength=args.gfpgan_strength,
            realesrgan_tile=args.realesrgan_tile,
            realesrgan_scale=args.realesrgan_scale,
            apply_gfpgan=args.ff_gfpgan, apply_realesrgan=args.ff_realesrgan,
            apply_identity_transfer=args.ff_identity,
            overwrite=args.overwrite, dry_run=args.dry_run)
        if args.action == "check":
            backends = fcs.detect_face_consistency_backends(config=cfg)
            print(json.dumps(backends, indent=2) if args.json
                  else "Face Consistency backends:\n  " + "\n  ".join(
                      f"{k}: {'available' if v.get('available') else 'unavailable'}"
                      for k, v in backends.items()
                      if isinstance(v, dict) and "available" in v)
                  + ("\n" + "\n".join("  ! " + n for n in backends["notes"])
                     if backends["notes"] else ""))
            return
        if not args.target or not args.source_face:
            print("face-fix run needs --target and --source-face "
                  "(or use 'face-fix check')")
            return
        is_batch = args.batch or args.mode == "batch" or (
            args.mode == "auto" and Path(args.target).expanduser().is_dir())
        if is_batch:
            res = fcs.process_face_consistency_batch(args.target, args.source_face, cfg)
            try:
                fcs.write_ui_status_summary(res, prj.output_path)
            except Exception:
                pass
            print(json.dumps(res.to_dict(), indent=2) if args.json else (
                f"{'DRY-RUN' if cfg.dry_run else 'DONE'} -> {res.output_dir}\n"
                f"  total {res.total_images} | processed {res.processed_images} | "
                f"failed {res.failed_images} | skipped {res.skipped_images}\n"
                f"  backends available: "
                f"{res.backends.get('any_backend_available', False)}"
                + ("\n  manifest: " + res.manifest_path if res.manifest_path else "")))
        else:
            r = fcs.process_face_consistency_single(args.target, args.source_face, cfg)
            print(json.dumps(r.to_dict(), indent=2) if args.json else (
                f"{'ok' if r.ok else 'failed'} -> {r.output_image}\n"
                f"  steps: {', '.join(r.steps_applied)}"
                + ("\n  warnings: " + "; ".join(r.warnings) if r.warnings else "")
                + ("\n  errors: " + "; ".join(r.errors) if r.errors else "")))
    elif cmd == "specialized-recipes":
        from .services import specialized_recipe_builder as rb
        from .services import visual_focus_classifier as _vc
        recs = []
        if args.input:
            ip = Path(args.input).expanduser()
            if ip.exists():
                recs = _vc.load_visual_focus_classifications(ip)
        elif args.dataset:
            base = Path(args.dataset).expanduser()
            for cand in (base / "visual_focus" / _vc.RESULTS_NAME,
                         base / _vc.RESULTS_NAME):
                if cand.exists():
                    recs = _vc.load_visual_focus_classifications(cand)
                    break
            if not recs:
                print("no visual_focus_classifications.jsonl found - falling "
                      "back to manifest WD14 tags + quality signals")
        # Florence-independent fallback: read WD14 tags + quality from the
        # manifest via the dataset explorer (works without visual_focus/Florence)
        if not recs and (prj.output_path / DB_NAME).exists():
            from .services import dataset_explorer as _dx
            if getattr(args, "process_all", False):
                erecs = _dx.load_all_image_records(
                    prj.output_path, _dx.ExplorerFilter(limit=_dx.MAX_PAGE_SIZE),
                    hard_cap=args.max_records)
                _scope = f"full manifest ({len(erecs)} records)"
            else:
                erecs = _dx.load_image_records(
                    prj.output_path, _dx.ExplorerFilter(limit=_dx.MAX_PAGE_SIZE))
                _scope = "capped page; pass --all for the full manifest"
            recs = [{"image_path": r.path, "caption_wd14": r.caption,
                     "quality_signals": r.quality_signals,
                     "architecture_target": r.architecture_target}
                    for r in erecs]
            if recs:
                print(f"loaded {len(recs)} records from manifest "
                      f"(WD14 + quality; {_scope})")
        if not recs:
            print("no input records (use --input <jsonl>, --dataset <dir>, or "
                  "run inside a project with a manifest)")
            return
        out = Path(args.output).expanduser() if args.output else (
            (Path(args.dataset).expanduser() if args.dataset
             else Path(args.input).expanduser().parent) / "specialized_recipes")
        cfg = rb.SpecializedRecipeBuilderConfig(
            min_recipe_confidence=args.min_confidence,
            min_image_suitability=args.min_suitability,
            max_recipes_per_image=args.max_recipes_per_image,
            architecture_default=args.architecture,
            allow_mixed_architecture=args.allow_mixed_architecture,
            overwrite=args.overwrite)
        plan = rb.build_specialized_recipe_plan(recs, out, cfg)
        if getattr(args, "project_toml", None):
            plan.project_toml_suggestions = \
                rb.generate_project_toml_recipe_suggestions(
                    plan, Path(args.project_toml).expanduser())
        written = rb.write_specialized_recipe_plan(plan, out, overwrite=args.overwrite)
        try:
            from .services import status_writer as sw
            sw.write_specialized_recipe_status(str(out))
        except Exception:
            pass
        if args.json:
            print(json.dumps({"plan_id": plan.plan_id, "output_dir": str(out),
                              "written": {k: str(v) for k, v in written.items()},
                              "summary": plan.summary}, indent=2))
        else:
            print(rb.render_recipe_card(plan))
            print(f"\nplan written to: {out}")
    elif cmd == "classify":
        from .services import visual_focus_classifier as vc
        ccmd = getattr(args, "classify_cmd", None)
        if ccmd is None:
            print("usage: pixel-curator classify "
                  "{visual-focus,visual-focus-image} ...")
            return
        if ccmd == "visual-focus":
            recs = []
            if args.input:
                ip = Path(args.input).expanduser()
                if ip.exists():
                    for line in ip.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            try:
                                recs.append(json.loads(line))
                            except Exception:
                                pass
            elif args.dataset:
                from .services import florence_captioning as flc
                fp = Path(args.dataset).expanduser() / flc.RESULTS_NAME
                if fp.exists():
                    recs = flc.load_florence_results(fp)
                else:
                    print(f"no {flc.RESULTS_NAME} under {args.dataset} - run "
                          "'florence batch' first")
            if not recs:
                print("no input records found (use --input <jsonl> or --dataset "
                      "<dir with florence_captions.jsonl>)")
                return
            if args.limit:
                recs = recs[:args.limit]
            out = Path(args.output).expanduser() if args.output else \
                (Path(args.dataset).expanduser() if args.dataset
                 else Path(args.input).expanduser().parent) / "visual_focus"
            out.mkdir(parents=True, exist_ok=True)
            res = vc.classify_visual_focus_batch(
                recs, out / vc.RESULTS_NAME, overwrite=args.overwrite)
            try:
                (out / vc.SUMMARY_NAME).write_text(
                    json.dumps(res["summary"], indent=2), encoding="utf-8")
                (out / vc.CARD_NAME).write_text(
                    vc.render_routing_card(res["summary"]), encoding="utf-8")
            except Exception:
                pass
            print(json.dumps(res, indent=2) if args.json
                  else vc.render_routing_card(res["summary"])
                  + f"\nresults: {res['output_jsonl']}")
        elif ccmd == "visual-focus-image":
            florence = quality = None
            if args.florence_json and Path(args.florence_json).expanduser().exists():
                florence = json.loads(Path(args.florence_json).expanduser()
                                      .read_text(encoding="utf-8"))
            if args.quality_json and Path(args.quality_json).expanduser().exists():
                quality = json.loads(Path(args.quality_json).expanduser()
                                     .read_text(encoding="utf-8"))
            r = vc.classify_visual_focus(
                str(Path(args.image).expanduser()), florence_result=florence,
                quality_signals=quality)
            if args.json:
                print(json.dumps(r.to_dict(), indent=2))
            else:
                print(f"image: {r.image_path}")
                print(f"  focus  : {r.primary_focus.label} "
                      f"({r.primary_focus.confidence:.2f})")
                print(f"  recipe : {r.suggested_recipe_type}  | suitability "
                      f"{r.suitability_score:.1f} ({r.suitability_tier})")
                print(f"  targets: {', '.join(t.target for t in r.recommended_targets if not t.blocked) or '(none)'}")
                print(f"  tags   : {', '.join(r.routing_tags) or '(none)'}")
                for m in r.review_flags + r.warnings:
                    print(f"  ! {m}")
    elif cmd == "florence":
        from .services import florence_captioning as flc
        fcmd = getattr(args, "fl_cmd", None)
        if fcmd is None:
            print("usage: pixel-curator florence {check,image,batch} ...")
            return
        cfg = flc.FlorenceRuntimeConfig(
            python_executable=(args.vlm_python or
                               flc.FlorenceRuntimeConfig().python_executable),
            model_path=args.model_path or "", model_variant=args.model_variant,
            device=args.device, dtype=args.dtype,
            local_files_only=args.local_files_only,
            allow_downloads=args.allow_downloads)
        if fcmd == "check":
            rt = flc.detect_florence_runtime(cfg)
            if args.json:
                print(json.dumps(rt, indent=2))
            else:
                print("Florence-2 runtime check")
                print(f"  available    : {rt['available']}")
                print(f"  VLM python   : {rt['python_executable'] or '(not found)'}")
                print(f"  model variant: {rt['model_variant']}")
                print(f"  model path   : {rt['model_path']} "
                      f"({'ok' if rt['model_path_exists'] else 'missing'})")
                print(f"  models found : {rt['models_detected']}")
                for w in rt["warnings"]:
                    print(f"  ! {w}")
        elif fcmd == "image":
            req = flc.FlorenceCaptionRequest(
                image_path=str(Path(args.image).expanduser()),
                task_prompt=args.task, mode=args.mode)
            res = flc.run_florence_caption(req, cfg)
            if getattr(args, "sidecars", False) and res.ok:
                flc.write_florence_sidecars(
                    res, Path(args.image).expanduser().parent)
            if args.json:
                print(json.dumps(res.to_dict(), indent=2))
            else:
                print(f"image: {res.image_path}")
                print(f"  ok      : {res.ok}")
                if res.caption:
                    print(f"  caption : {res.caption}")
                va = res.visual_attributes
                print(f"  focus   : {va.get('primary_focus')} | pose "
                      f"{va.get('pose_type')} | detail {va.get('detail_level')}")
                if res.routing_tags:
                    print(f"  routing : {', '.join(res.routing_tags)}")
                for m in res.errors + res.warnings:
                    print(f"  ! {m}")
        elif fcmd == "batch":
            src = Path(args.dataset).expanduser()
            out = Path(args.output).expanduser() if args.output else src
            out.mkdir(parents=True, exist_ok=True)
            imgs = [p for p in sorted(src.rglob("*"))
                    if p.is_file() and p.suffix.lower() in flc.IMAGE_EXTS]
            summ = flc.run_florence_batch(
                imgs, out / flc.RESULTS_NAME, config=cfg, mode=args.mode,
                limit=args.limit)
            try:
                (out / flc.SUMMARY_NAME).write_text(
                    json.dumps(summ["summary"], indent=2), encoding="utf-8")
                (out / flc.CARD_NAME).write_text(
                    flc.render_florence_card(summ["summary"]), encoding="utf-8")
            except Exception:
                pass
            print(json.dumps(summ, indent=2) if args.json
                  else flc.render_florence_card(summ["summary"])
                  + f"\nresults: {summ['output_jsonl']}")
    elif cmd == "visual-focus":
        from .curate import visual_focus_classifier as vfc
        from .services import visual_focus_service as vfs
        vcmd = getattr(args, "vf_cmd", None)
        cfg = vfc.VisualFocusConfig(
            use_vlm_refinement=getattr(args, "use_vlm_refinement", False))
        if getattr(args, "use_vlm_refinement", False):
            print("note: VLM refinement requested; baseline CV classifier is "
                  "used unless a local VLM is configured.")
        if vcmd == "image":
            res = vfc.classify_visual_focus(Path(args.image).expanduser(),
                                            config=cfg)
            if args.json:
                print(json.dumps(res.to_dict(), indent=2))
            else:
                print(f"image: {res.image_path}")
                print(f"  focus   : {res.primary_focus.label} "
                      f"({res.primary_focus.confidence:.2f})")
                print(f"  detail  : {res.detail_tier.label}")
                print(f"  pose    : {res.pose_expression.label}")
                print(f"  suitability: {res.suitability_score:.1f} "
                      f"({res.suitability_tier})")
                print(f"  targets : {', '.join(res.recommended_targets)}")
                if res.flags:
                    print(f"  flags   : {', '.join(res.flags)}")
                if res.caption_hint:
                    print(f"  hint    : {res.caption_hint}")
        elif vcmd == "classify":
            summ = vfs.classify_dataset_visual_focus(
                Path(args.dataset).expanduser(),
                output_dir=(Path(args.output).expanduser() if args.output else None),
                config=cfg, limit=args.limit)
            if args.json:
                print(json.dumps(summ, indent=2))
            else:
                print(vfs.render_visual_focus_card(summ))
                if summ.get("results_path"):
                    print(f"\nresults: {summ['results_path']}")
        else:
            print("usage: pixel-curator visual-focus {classify,image} ...")
    elif cmd == "explorer":
        from .services import dataset_explorer as dx
        from .services import pipeline_lineage as pl
        xbase = Path(args.output_base).expanduser() if getattr(
            args, "output_base", None) else prj.output_path
        xcmd = getattr(args, "explorer_cmd", None)
        if xcmd is None:
            print("usage: pixel-curator explorer "
                  "{summary,stage,compare,image,lineage} ...")
            return
        if xcmd == "summary":
            summ = dx.get_stage_summary(xbase)
            if getattr(args, "write_status", False):
                from .services import status_writer as sw
                sw.write_explorer_status(str(xbase))
            print(json.dumps(summ, indent=2) if args.json
                  else dx.render_summary_markdown(summ))
        elif xcmd == "stage":
            flt = dx.ExplorerFilter(
                stage_key=args.stage, limit=args.limit, offset=args.offset,
                min_quality=args.min_quality, max_quality=args.max_quality,
                architecture_target=args.architecture, target_label=args.target,
                caption_query=args.caption, source_query=args.source,
                version=args.version, cluster_id=args.cluster,
                framing=args.framing, coverage=args.coverage,
                required_flags=list(args.flag or []),
                excluded_flags=list(args.exclude_flag or []),
                duplicate_status=args.duplicate)
            recs = dx.load_image_records(xbase, flt)
            if args.json:
                print(json.dumps({"stage": args.stage, "count": len(recs),
                                  "offset": args.offset, "limit": flt.limit,
                                  "records": [r.to_dict() for r in recs]}, indent=2))
            else:
                print(f"stage={args.stage or 'all'}  showing {len(recs)} "
                      f"(offset {args.offset}, limit {flt.limit})")
                for r in recs:
                    q = f"{r.overall_quality_score:.2f}" if r.overall_quality_score \
                        is not None else "  - "
                    print(f"  [{q}] {r.architecture_target:<7} "
                          f"{r.target_label or '-':<8} {r.filename}"
                          + (f"  flags={','.join(r.flags)}" if r.flags else ""))
        elif xcmd == "compare":
            cmp = dx.compare_stages(xbase, args.stage_a, args.stage_b)
            if args.json:
                print(json.dumps(cmp.to_dict(), indent=2))
            else:
                print(f"{cmp.stage_a.label} ({cmp.stage_a.count}) -> "
                      f"{cmp.stage_b.label} ({cmp.stage_b.count})")
                print(f"  kept={cmp.kept_count}  removed={cmp.removed_count}  "
                      f"added={cmp.added_count}  retention="
                      f"{cmp.retention_rate}")
                if cmp.removal_reasons:
                    print("  removal reasons:", json.dumps(cmp.removal_reasons))
                for w in cmp.warnings:
                    print(f"  ! {w}")
        elif xcmd == "image":
            recs = dx.load_image_records(
                xbase, dx.ExplorerFilter(limit=1))
            lin = pl.get_image_lineage(xbase, args.image)
            # fetch the single record by resolving via lineage path/id
            detail = {"lineage": lin.to_dict()}
            if args.json:
                print(json.dumps(detail, indent=2))
            else:
                print(f"image: {lin.path or args.image}")
                print(f"  present : {', '.join(lin.stages_present) or '(none)'}")
                print(f"  missing : {', '.join(lin.stages_missing) or '(none)'}")
                print(f"  versions: {', '.join(lin.version_membership) or '(none)'}")
                print(f"  flags   : {', '.join(lin.flags) or '(none)'}")
                if lin.likely_removal_reason:
                    print(f"  removal : {lin.likely_removal_reason}")
                for w in lin.warnings:
                    print(f"  ! {w}")
        elif xcmd == "lineage":
            lin = pl.get_image_lineage(xbase, args.image)
            print(json.dumps(lin.to_dict(), indent=2) if args.json else
                  "\n".join([f"image: {lin.path or args.image}",
                             f"  present : {', '.join(lin.stages_present)}",
                             f"  missing : {', '.join(lin.stages_missing)}",
                             f"  versions: {', '.join(lin.version_membership)}",
                             f"  removal : {lin.likely_removal_reason or '-'}"]
                            + [f"  ! {w}" for w in lin.warnings]))
    elif cmd == "datasets":
        from .builder import list_datasets
        conn = manifest.connect(prj.output_path)
        rows = list_datasets(conn)
        if not rows:
            print("No dataset builds yet. Run: pixel-curator build --recipe <name>")
        for d in rows:
            parent = (f" parent=v{d['parent']:03d}"
                      if d.get("parent") is not None else "")
            kind = (f" [{d['kind']}]" if d.get("kind")
                    and d["kind"] != "build" else "")
            print(f"  v{d['version']:03d}  {d['recipe']:<16} train={d['train']:<5} "
                  f"val={d['val']:<4} hash={d['hash']}  {d['built_at']}  "
                  f"{d['note']}{kind}{parent}")
    elif cmd == "diff":
        conn = manifest.connect(out_base(args.output))
        if args.dataset:
            # Phase 2 visual diff mode: --dataset N --vs M
            from .dataset_diff import dataset_version_diff, \
                render_dataset_diff
            try:
                d = dataset_version_diff(conn, out_base(args.output),
                                         args.dataset, args.vs)
            except KeyError as exc:
                print(f"Error: {exc}")
                return
            if args.export:
                target = Path(args.export).expanduser()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(d, indent=2))
                print(f"Diff exported -> {target}")
            if args.json:
                print(json.dumps(d, indent=2))
            if not (args.json or args.export):
                print(render_dataset_diff(d))
            return
        if args.a is None or args.b is None:
            print("Usage: diff A B  (legacy membership diff)\n"
                  "   or: diff --dataset N --vs M  (visual diff)")
            return
        from .builder import diff_datasets
        try:
            d = diff_datasets(conn, args.a, args.b)
        except KeyError as exc:
            print(f"Error: {exc}")
            return
        print(d["summary"])
        print(f"concept mix: {d['concept_mix_a']} -> {d['concept_mix_b']}")
        if args.full:
            for label in ("added", "removed", "caption_changed", "split_changed"):
                for fid, concept in d[label]:
                    print(f"  {label:<16} {fid} ({concept})")
    elif cmd == "viz":
        from .viz import build_viz_payload, render_viz_summary
        conn = manifest.connect(out_base(args.output))
        payload = build_viz_payload(
            conn, out_base(args.output), dataset=args.dataset,
            method=args.method, recompute=args.recompute, k=args.clusters)
        if args.export:
            target = Path(args.export).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2))
            print(f"Viz payload exported -> {target}")
        if args.json:
            print(json.dumps(payload, indent=2))
        if not (args.json or args.export):
            print(render_viz_summary(payload))
    elif cmd == "cluster":
        cfg = ClusterConfig(
            output_base=out_base(args.output), k=args.k,
            batch_size=args.batch_size, reembed=args.reembed,
        )
        stream(cluster_generator(cfg))
    elif cmd == "caption":
        cfg = CaptionConfig(
            output_base=out_base(args.output),
            trigger=args.trigger or prj.trigger_token,
            class_word=args.class_word or prj.class_word,
            threshold=args.threshold, char_threshold=args.char_threshold,
            pony_prefix=args.pony_prefix, blacklist=args.blacklist,
            remap=args.remap, prune=args.prune, max_tags=args.max_tags,
            force=args.force,
            repo_id={"swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
                     "vit": "SmilingWolf/wd-vit-tagger-v3",
                     "vit-large": "SmilingWolf/wd-vit-large-tagger-v3",
                     "convnext": "SmilingWolf/wd-convnext-tagger-v3",
                     }.get(args.repo_id, args.repo_id),
            tagger=(getattr(args, "tagger", None)
                    or getattr(prj, "default_tagger", "wd14") or "wd14"),
        )
        if getattr(args, "reprune", False):
            from .caption.wd14 import reprune_generator
            for update in reprune_generator(cfg, apply=args.apply):
                print(update)
        else:
            stream(caption_generator(cfg))
    elif cmd == "package-legacy":
        cfg = PackageConfig(
            output_base=out_base(args.output),
            token=args.token or prj.trigger_token,
            class_word=args.class_word or prj.class_word,
            repeats=args.repeats, max_per_video=args.max_per_video,
            max_total=args.max_total, caption_text=args.caption,
            write_captions=args.captions, link_mode=args.link_mode,
            quota=args.quota, use_caption_table=args.use_caption_table,
        )
        for update in package_generator(cfg):
            print(update)
