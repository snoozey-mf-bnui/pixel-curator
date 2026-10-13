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
    from ..curate.semantic import apply_text_filter
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
    from ..export.review import build_review_report, import_review_decisions
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


def cmd_vlm_caption(ctx: Context):
    """Handler for the `vlm-caption` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..caption.vlm import make_florence_captioner, render_vlm, run_vlm_pass
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


def cmd_florence(ctx: Context):
    """Handler for the `florence` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..services import florence_captioning as flc
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
            logger.debug("suppressed exception", exc_info=True)
            pass
        print(json.dumps(summ, indent=2) if args.json
              else flc.render_florence_card(summ["summary"])
              + f"\nresults: {summ['output_jsonl']}")


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
        from ..caption.wd14 import reprune_generator
        for update in reprune_generator(cfg, apply=args.apply):
            print(update)
    else:
        stream(caption_generator(cfg))

