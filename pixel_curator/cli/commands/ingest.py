"""CLI commands: ingest."""

from __future__ import annotations

from ..common import *  # noqa: F401,F403


def cmd_orient(ctx: Context):
    """Handler for the `orient` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...orient import (
        IMG_EXTS,
        OrientConfig,
        make_insightface_detector,
        render_summary,
        run_orientation,
        undo_from_manifest,
    )
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


def cmd_keyframes(ctx: Context):
    """Handler for the `keyframes` command."""
    args, out_base = ctx.args, ctx.out_base
    from ...curate.keyframes import prune_video_keyframes, render_keyframes
    base = out_base(args.output)
    if not (base / DB_NAME).exists():
        print(f"No manifest at {base / DB_NAME}")
        return
    conn = manifest.connect(base)
    res = prune_video_keyframes(
        conn, max_per_video=args.max_per_video, min_spacing=args.min_spacing,
        min_quality=args.min_quality, apply=args.apply)
    print(render_keyframes(res))


def cmd_extract(ctx: Context):
    """Handler for the `extract` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
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


def cmd_smart_extract(ctx: Context):
    """Handler for the `smart-extract` command."""
    args = ctx.args
    # NSFW smart extraction: same engine as `extract`, but the config is
    # built from the [extraction] section of the project TOML (which -p
    # points at — typically pixel-curator_nsfw_v2.toml).
    from ...smart_extract import run as run_smart_extract
    rc = run_smart_extract(
        args.project,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        limit_videos=args.limit_videos,
    )
    if rc:
        raise SystemExit(rc)


def cmd_photos(ctx: Context):
    """Handler for the `photos` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    cfg = ExtractConfig(
        output_base=out_base(args.output),
        import_photos=True, photo_import_mode=args.photo_mode,
        resume=args.resume, dry_run=args.dry_run,
        prefilter=getattr(args, "prefilter", True),
    )
    stream(pipeline_generator([], args.photos_dir or prj.photos_dir, cfg))


def cmd_prefilter(ctx: Context):
    """Handler for the `prefilter` command."""
    args, out_base = ctx.args, ctx.out_base
    from ...prefilter import prefilter_generator, prefilter_summary, restore_prefiltered
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

