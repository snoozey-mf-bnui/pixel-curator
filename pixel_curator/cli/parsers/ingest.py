"""Parser registration: ingest commands."""

from __future__ import annotations

from ...config import DEFAULT_FPS_TIERS, DEFAULT_QUOTA, UI_PORT  # noqa: F401


def register(sub):
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

