"""CLI commands: exporting."""

from __future__ import annotations

import logging

from ..common import *  # noqa: F401,F403

logger = logging.getLogger(__name__)


def cmd_export(ctx: Context):
    """Handler for the `export` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    ecmd = getattr(args, "export_cmd", None)
    if ecmd not in ("manifest", "select"):
        print("usage: pixel-curator export {manifest|select} ...")
        return
    if not (prj.output_path / DB_NAME).exists():
        print(f"No manifest at {prj.output_path / DB_NAME}")
        return
    conn = manifest.connect(prj.output_path)
    from ..export.general import export_manifest, export_selection
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


def cmd_package(ctx: Context):
    """Handler for the `package` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    if getattr(args, "package_cmd", None) != "kohya":
        print("usage: pixel-curator package kohya --dataset ... --output ...")
        return
    from ..export.kohya_packager import (
        CaptionTemplateConfig,
        KohyaPackageConfig,
        package_kohya_dataset,
    )
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
        from ..dataset_versioning import KIND_KOHYA_PACKAGE, register_folder_export
        from ..export.kohya_packager import registry_files
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
        from ..train.export_config import generate_for_export
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


def cmd_dataset_card(ctx: Context):
    """Handler for the `dataset-card` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..curate.dataset_cards import (
        DatasetCardConfig,
        DatasetValidationConfig,
        generate_dataset_card,
        validate_exported_dataset,
    )
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


def cmd_export_dreambooth(ctx: Context):
    """Handler for the `export-dreambooth` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..train.dreambooth_export import (
        DreamBoothExportConfig,
        plan_dreambooth_export,
        render_plan,
    )
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


def cmd_previews(ctx: Context):
    """Handler for the `previews` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from pathlib import Path as _P

    from ..lora_explorer import lora_dirs
    from ..qa_render import render_lora_previews
    lp = _P(args.lora).expanduser()
    if not lp.exists():
        for d in lora_dirs(prj):
            cand = d / f"{args.lora}.safetensors"
            if cand.exists():
                lp = cand
                break
    for u in render_lora_previews(prj, lp):
        print(u)


def cmd_multi_recipe_packages(ctx: Context):
    """Handler for the `multi-recipe-packages` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..services import multi_recipe_packager as mp
    recs = []
    if args.input and Path(args.input).expanduser().exists():
        for line in Path(args.input).expanduser().read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                try:
                    recs.append(json.loads(line))
                except Exception:
                    logger.debug("suppressed exception", exc_info=True)
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
        from ..services import dataset_explorer as _dx
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
        logger.debug("suppressed exception", exc_info=True)
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


def cmd_package_legacy(ctx: Context):
    """Handler for the `package-legacy` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
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

