"""CLI commands: curation."""

from __future__ import annotations

import logging

from ..common import *  # noqa: F401,F403

logger = logging.getLogger(__name__)


def cmd_anchors_rebuild(ctx: Context):
    """Handler for the `anchors-rebuild` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...curate.anchor_quality import (
        AnchorQualityConfig,
        check_anchor_report,
        rebuild_anchor_set,
    )
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


def cmd_diverse_select(ctx: Context):
    """Handler for the `diverse-select` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...diverse import diverse_select, fetch_candidates, render_plan
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


def cmd_quality(ctx: Context):
    """Handler for the `quality` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...curate.quality_signals import analyze_quality_pass, render_quality
    base = out_base(args.output)
    if not (base / DB_NAME).exists():
        print(f"No manifest at {base / DB_NAME}")
        return
    conn = manifest.connect(base)
    face_detector = None
    if args.with_face:
        from ...curate.quality_signals import make_face_detector
        face_detector, reason = make_face_detector(prj)
        if face_detector is None:
            print(f"--with-face unavailable: {reason}\nRunning bbox-only pass.")
    from ...curate.advanced_quality import AdvancedQualityConfig
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


def cmd_categorize(ctx: Context):
    """Handler for the `categorize` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...curate.categorize import categorize_pass, export_categories, render_categorize
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


def cmd_outfit(ctx: Context):
    """Handler for the `outfit` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...curate.outfit import render_outfit, run_outfit_pass
    base = out_base(args.output)
    if not (base / DB_NAME).exists():
        print(f"No manifest at {base / DB_NAME}")
        return
    conn = manifest.connect(base)
    res = run_outfit_pass(conn, limit=args.limit, store=args.store)
    print(json.dumps(res, indent=2, default=str) if args.json
          else render_outfit(res))


def cmd_framing_report(ctx: Context):
    """Handler for the `framing-report` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...curate.framing import framing_distribution, render_framing
    base = out_base(args.output)
    if not (base / DB_NAME).exists():
        print(f"No manifest at {base / DB_NAME}")
        return
    conn = manifest.connect(base, readonly=True)
    print(render_framing(framing_distribution(conn)))


def cmd_anchor(ctx: Context):
    """Handler for the `anchor` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    cfg = SmartCurateConfig(
        output_base=out_base(args.output),
        anchor_dir=Path(args.anchor_dir or prj.anchor_dir).expanduser()
        if (args.anchor_dir or prj.anchor_dir) else None,
        det_size=args.det_size,
    )
    for update in build_anchor(cfg):
        print(update)


def cmd_curate(ctx: Context):
    """Handler for the `curate` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
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
        from ...intelligence import analyze_dataset, intelligence_digest
        conn = manifest.connect(base, readonly=True)
        print("\n" + intelligence_digest(
            analyze_dataset(conn, output_base=base)))
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass


def cmd_insights(ctx: Context):
    """Handler for the `insights` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
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
        from ...intelligence import analyze_dataset, render_intelligence
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


def cmd_duplicates(ctx: Context):
    """Handler for the `duplicates` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...curate.duplicates import DuplicateScanConfig, scan_duplicates, scan_generator
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


def cmd_rescore(ctx: Context):
    """Handler for the `rescore` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...rescore import rescore_identity_generator
    meta_dir = args.metadata_dir or prj.metadata_dir or None
    for update in rescore_identity_generator(
            prj, out_base(args.output),
            identity_threshold=args.identity_threshold,
            metadata_dir=Path(meta_dir).expanduser() if meta_dir else None,
            subject_names=tuple(args.subject_names or prj.subject_names or ()),
            sidecar_relax=args.sidecar_relax):
        print(update)


def cmd_classify(ctx: Context):
    """Handler for the `classify` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...services import visual_focus_classifier as vc
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
                            logger.debug("suppressed exception", exc_info=True)
                            pass
        elif args.dataset:
            fp = Path(args.dataset).expanduser() / "captions.jsonl"
            if fp.exists():
                for line in fp.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            recs.append(json.loads(line))
                        except Exception:
                            logger.debug("suppressed exception", exc_info=True)
                            pass
            else:
                print(f"no captions.jsonl under {args.dataset}")
        if not recs:
            print("no input records found (use --input <jsonl> or --dataset "
                  "<dir with captions.jsonl>)")
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
            logger.debug("suppressed exception", exc_info=True)
            pass
        print(json.dumps(res, indent=2) if args.json
              else vc.render_routing_card(res["summary"])
              + f"\nresults: {res['output_jsonl']}")
    elif ccmd == "visual-focus-image":
        captions = quality = None
        if args.captions_json and Path(args.captions_json).expanduser().exists():
            captions = json.loads(Path(args.captions_json).expanduser()
                                  .read_text(encoding="utf-8"))
        if args.quality_json and Path(args.quality_json).expanduser().exists():
            quality = json.loads(Path(args.quality_json).expanduser()
                                 .read_text(encoding="utf-8"))
        r = vc.classify_visual_focus(
            str(Path(args.image).expanduser()), caption_result=captions,
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


def cmd_visual_focus(ctx: Context):
    """Handler for the `visual-focus` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...curate import visual_focus_classifier as vfc
    from ...services import visual_focus_service as vfs
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


def cmd_explorer(ctx: Context):
    """Handler for the `explorer` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...services import dataset_explorer as dx
    from ...services import pipeline_lineage as pl
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
            from ...services import status_writer as sw
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


def cmd_datasets(ctx: Context):
    """Handler for the `datasets` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...builder import list_datasets
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


def cmd_viz(ctx: Context):
    """Handler for the `viz` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...viz import build_viz_payload, render_viz_summary
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


def cmd_cluster(ctx: Context):
    """Handler for the `cluster` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    cfg = ClusterConfig(
        output_base=out_base(args.output), k=args.k,
        batch_size=args.batch_size, reembed=args.reembed,
    )
    stream(cluster_generator(cfg))

