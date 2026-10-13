"""CLI commands: ops."""

from __future__ import annotations

from ..common import *  # noqa: F401,F403


def cmd_init(ctx: Context):
    """Handler for the `init` command."""
    args = ctx.args
    path = Path(args.path)
    if path.exists():
        print(f"Refusing to overwrite existing {path}")
        return
    write_template(path)
    print(f"Wrote starter project file: {path}\nEdit it, then run: "
          f"pixel-curator -p {path} health")


def cmd_models(ctx: Context):
    """Handler for the `models` command: status + first-run downloads."""
    args = ctx.args
    from ...model_assets import download_missing, status
    if args.models_cmd == "download":
        keys = ([k.strip() for k in args.only.split(",") if k.strip()]
                if args.only else None)
        results = download_missing(keys)
        failed = [k for k, ok in results.items() if not ok]
        if failed:
            print(f"failed: {', '.join(failed)}")
            raise SystemExit(1)
        return
    rows = status()
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    for r in rows:
        mark = "OK " if r["installed"] else "-- "
        opt = " (optional)" if r["optional"] else ""
        print(f"{mark}{r['key']:<24} ~{r['size_mb']:>4} MB{opt}  "
              f"{r['purpose']}")
        if r["installed"] and r["path"]:
            print(f"    {r['path']}")
    missing = [r["key"] for r in rows if not r["installed"]]
    if missing:
        print(f"\n{len(missing)} not downloaded - they auto-download on "
              f"first use, or fetch now: pixel-curator models download")


def cmd_doctor(ctx: Context):
    """Handler for the `doctor` command."""
    prj = ctx.prj
    print(json.dumps(check_dependencies(), indent=2))
    from ...util import project_data_plane
    plane = project_data_plane(prj)
    print("\nProject data plane:")
    print(json.dumps(plane, indent=2))
    from ...curate import smart_available
    ok, reason = smart_available()
    print(f"smart curation: {'available' if ok else reason}")
    from ...runtime import configure_from_project, onnx_providers
    info = configure_from_project(prj)
    print("\nRuntime / accelerator:")
    print(json.dumps(info, indent=2))
    print(f"onnxruntime providers: {onnx_providers()}")
    from ...curate.performance import detect_compute_device
    print("\nCuration compute device:")
    print(json.dumps(detect_compute_device(), indent=2))
    if not plane["ok"]:
        print("\nDOCTOR: project data plane is NOT ready (errors above).")
        return 1


def cmd_stats(ctx: Context):
    """Handler for the `stats` command."""
    prj = ctx.prj
    base = prj.output_path
    if not (base / DB_NAME).exists():
        print(f"No manifest at {base / DB_NAME}")
        return
    conn = manifest.connect(base)
    print(json.dumps(manifest.stats(conn), indent=2))


def cmd_overview(ctx: Context):
    """Handler for the `overview` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...overview import overview_report, render_overview
    rep = overview_report(prj, out_base(args.output))
    print(json.dumps(rep, indent=2) if args.json
          else render_overview(rep))


def cmd_validate(ctx: Context):
    """Handler for the `validate` command."""
    args, out_base = ctx.args, ctx.out_base
    from ...validate import render_report, run_validation
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


def cmd_selftest(ctx: Context):
    """Handler for the `selftest` command."""
    args = ctx.args
    import subprocess
    runner = Path(__file__).resolve().parents[2] / "tests" / "run_all.py"
    if not runner.exists():
        print(f"Test runner not found at {runner}")
        raise SystemExit(1)
    rc = subprocess.call(
        [sys.executable, str(runner), "--timeout", str(args.timeout),
         *args.filters])
    raise SystemExit(rc)


def cmd_watch(ctx: Context):
    """Handler for the `watch` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...maintenance import WatchConfig, watch_generator
    cfg = WatchConfig(output_base=out_base(args.output),
                      interval=args.interval,
                      auto_curate=args.auto_curate, once=args.once)
    try:
        for update in watch_generator(prj, cfg):
            print(update)
    except KeyboardInterrupt:
        print("\nwatch stopped.")


def cmd_gc(ctx: Context):
    """Handler for the `gc` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...maintenance import GcConfig, gc_generator
    for update in gc_generator(prj, GcConfig(
        output_base=out_base(args.output), apply=args.apply,
        keep_builds=args.keep_builds,
    )):
        print(update)


def cmd_space(ctx: Context):
    """Handler for the `space` command."""
    prj = ctx.prj
    from ...maintenance import space_report
    for row in space_report(prj, prj.output_path):
        print(f"  {row['area']:<16} {str(row['files']):>8} files  {row['human']:>10}")


def cmd_backup(ctx: Context):
    """Handler for the `backup` command."""
    args, prj = ctx.args, ctx.prj
    from ...maintenance import backup_project
    out = backup_project(prj, args.out or None)
    print(f"Backup written -> {out}")


def cmd_restore(ctx: Context):
    """Handler for the `restore` command."""
    args = ctx.args
    from ...maintenance import restore_project
    dest = restore_project(args.archive, args.dest)
    print(f"Extracted to {dest} - review and place files manually.")


def cmd_status(ctx: Context):
    """Handler for the `status` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...monitor import (
        FollowConfig,
        follow_generator,
        render_status,
        status_snapshot,
    )
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


def cmd_runs(ctx: Context):
    """Handler for the `runs` command."""
    prj = ctx.prj
    from ...train.kohya import list_runs
    conn = manifest.connect(prj.output_path, readonly=True)
    rows = list_runs(conn)
    if not rows:
        print("No training runs yet. Run: pixel-curator train-legacy --dataset N --preset character")
    for r in rows:
        print(f"  #{r['run_id']:<3} {r['name']:<28} v{r['dataset_version']:03d} "
              f"{r['preset']:<9} {r['status']:<9} "
              f"step {r['step']}/{r['total_steps']} loss {r['last_loss']}")


def cmd_health(ctx: Context):
    """Handler for the `health` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...health import health_generator
    for update in health_generator(prj, out_base(args.output), args.open):
        print(update)


def cmd_history(ctx: Context):
    """Handler for the `history` command."""
    args, out_base = ctx.args, ctx.out_base
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


def cmd_diff(ctx: Context):
    """Handler for the `diff` command."""
    args, out_base = ctx.args, ctx.out_base
    conn = manifest.connect(out_base(args.output))
    if args.dataset:
        # Phase 2 visual diff mode: --dataset N --vs M
        from ...dataset_diff import dataset_version_diff, render_dataset_diff
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
    from ...builder import diff_datasets
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

