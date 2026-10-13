"""Parser registration: captioning commands."""

from __future__ import annotations

from ...config import DEFAULT_FPS_TIERS, DEFAULT_QUOTA, UI_PORT  # noqa: F401


def register(sub):
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

