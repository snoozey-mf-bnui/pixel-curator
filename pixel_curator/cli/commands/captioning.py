"""CLI commands: captioning."""

from __future__ import annotations

import logging

from ..common import *  # noqa: F401,F403

logger = logging.getLogger(__name__)


def cmd_filter(ctx: Context):
    """Handler for the `filter` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    if not (prj.output_path / DB_NAME).exists():
        print(f"No manifest at {prj.output_path / DB_NAME}")
        return
    conn = manifest.connect(prj.output_path)
    from ...curate.semantic import apply_text_filter
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


def cmd_review(ctx: Context):
    """Handler for the `review` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    rcmd = getattr(args, "review_cmd", None)
    if rcmd not in ("export", "import"):
        print("usage: pixel-curator review {export|import} ...")
        return
    if not (prj.output_path / DB_NAME).exists():
        print(f"No manifest at {prj.output_path / DB_NAME}")
        return
    conn = manifest.connect(prj.output_path)
    from ...export.review import build_review_report, import_review_decisions
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


def cmd_caption(ctx: Context):
    """Handler for the `caption` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
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
        from ...caption.wd14 import reprune_generator
        for update in reprune_generator(cfg, apply=args.apply):
            print(update)
    else:
        stream(caption_generator(cfg))

