"""CLI commands: training."""

from __future__ import annotations

import logging

from ..common import *  # noqa: F401,F403

logger = logging.getLogger(__name__)


def cmd_train(ctx: Context):
    """Handler for the `train` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    tcmd = getattr(args, "train_cmd", None)
    if tcmd == "plan":
        from ..train.recipe_builder import (
            TrainingRecipeConfig,
            build_training_recipe,
            write_training_recipe,
        )
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
        from ..train.stack_planner import build_training_stack_plan
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


def cmd_likeness(ctx: Context):
    """Handler for the `likeness` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..eval.likeness import render_likeness, run_likeness
    base = out_base(args.output)
    if not (base / DB_NAME).exists():
        print(f"No manifest at {base / DB_NAME}")
        return
    conn = manifest.connect(base)
    rep = run_likeness(conn, prj=prj, lora=args.lora,
                       only_missing=not args.all,
                       min_identity_sim=args.min_identity_sim)
    print(json.dumps(rep, indent=2) if args.json else render_likeness(rep))


def cmd_leaderboard(ctx: Context):
    """Handler for the `leaderboard` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..eval.leaderboard import build_leaderboard, head_to_head, render_leaderboard
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


def cmd_mergeforge(ctx: Context):
    """Handler for the `mergeforge` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
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
        from ..eval.matrix import parse_lora_specs
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


def cmd_concept(ctx: Context):
    """Handler for the `concept` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
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
        from ..stack_intelligence import resolve_stack
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

        from ..batch_variations import VariationGrid, expand_grid, run_batch_generator, save_batch
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


def cmd_registry(ctx: Context):
    """Handler for the `registry` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..registry import best_likeness, build_registry
    conn = manifest.connect(prj.output_path)
    entries = build_registry(prj, conn)
    if not entries:
        print("No LoRAs known yet. Train one or drop .safetensors in lora_output_dir.")
    for e in entries:
        lk = best_likeness(e)
        print(f"  {e['name']:<32} [{e['kind']:<8}] "
              f"likeness={lk if lk is not None else '-':<8} "
              f"files={len(e.get('files', []))}")


def cmd_card(ctx: Context):
    """Handler for the `card` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    conn = manifest.connect(prj.output_path)
    if args.run:
        from ..model_card import generate_model_card
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
    from ..registry import model_card
    path = model_card(prj, conn, args.name)
    print(f"Model card: {path}" if path else f"Unknown LoRA: {args.name}")


def cmd_eval(ctx: Context):
    """Handler for the `eval` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..eval.run_eval import evaluate_run, render_eval
    conn = manifest.connect(out_base(args.output))
    if args.generate_samples:
        from ..sweep import SweepConfig, sweep_generator
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


def cmd_sweep(ctx: Context):
    """Handler for the `sweep` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..sweep import SweepConfig, sweep_generator
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


def cmd_merge(ctx: Context):
    """Handler for the `merge` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..eval.matrix import parse_lora_specs
    from ..merge import MergeConfig, merge_generator, parse_block_string
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
        from ..eval.matrix import MatrixConfig, matrix_generator
        from ..merge import safe_slug as _slug
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


def cmd_combo(ctx: Context):
    """Handler for the `combo` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..merge import combo_generator
    for update in combo_generator(
        prj, out_base(args.output), character=args.character,
        style=args.style, outfit=args.outfit, name=args.name,
        char_w=args.char_weight, style_w=args.style_weight,
        outfit_w=args.outfit_weight,
    ):
        print(update)


def cmd_matrix(ctx: Context):
    """Handler for the `matrix` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..eval.matrix import CATEGORIES, MatrixConfig, matrix_generator
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


def cmd_evals(ctx: Context):
    """Handler for the `evals` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..eval.matrix import eval_summary
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


def cmd_presets(ctx: Context):
    """Handler for the `presets` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..train import PRESETS
    for name, p in PRESETS.items():
        print(f"  {name:<10} dim {p['network_dim']}/{p['network_alpha']:<3} "
              f"lr {p['unet_lr']}/{p['te_lr']} batch {p['batch_size']} "
              f"epochs {p['epochs']}  - {p['notes']}")


def cmd_suggest_preset(ctx: Context):
    """Handler for the `suggest-preset` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..train.preset_recommender import format_recommendation, suggest_preset
    conn = manifest.connect(out_base(args.output))
    try:
        rec = suggest_preset(conn, args.dataset,
                             recipe_name=args.recipe)
        params = None
        if getattr(args, "with_params", True) \
                and rec.get("recommended_preset"):
            from ..train.param_advisor import recommend_training_params, render_params
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


def cmd_training_config(ctx: Context):
    """Handler for the `training-config` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..train.export_config import generate_training_bundle

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


def cmd_train_legacy(ctx: Context):
    """Handler for the `train-legacy` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..train.kohya import TrainConfig, train_generator
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


def cmd_recipes(ctx: Context):
    """Handler for the `recipes` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    if not prj.recipes:
        print("No recipes defined. Add [recipes.NAME] sections to the project file\n"
              "or regenerate a template with: pixel-curator init")
    for name, r in sorted(prj.recipes.items()):
        kind = r.get("type", "custom")
        extra = (f"concepts={r['concepts']}" if r.get("concepts")
                 else f"repeats={r.get('repeats', 10)} max_total={r.get('max_total', 0)}")
        print(f"  {name:<18} [{kind}] {extra}")


def cmd_pick_epoch(ctx: Context):
    """Handler for the `pick-epoch` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..eval.epoch_sweep import EpochSweepConfig, epoch_sweep_generator, parse_weights
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


def cmd_tune_stack(ctx: Context):
    """Handler for the `tune-stack` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..eval.epoch_sweep import parse_weights
    from ..eval.stack_tune import StackTuneConfig, parse_concept_ranges, stack_tune_generator
    cfg = StackTuneConfig(
        output_base=out_base(args.output), identity_lora=args.identity,
        concepts=parse_concept_ranges(args.concepts),
        identity_weights=parse_weights(args.identity_weights),
        cap=args.cap, checkpoint=args.checkpoint, forge_url=args.forge_url,
        trigger=args.trigger or f"{prj.trigger_token} {prj.class_word}".strip(),
    )
    for update in stack_tune_generator(prj, cfg):
        print(update)


def cmd_specialized_recipes(ctx: Context):
    """Handler for the `specialized-recipes` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ..services import specialized_recipe_builder as rb
    from ..services import visual_focus_classifier as _vc
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
        from ..services import status_writer as sw
        sw.write_specialized_recipe_status(str(out))
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    if args.json:
        print(json.dumps({"plan_id": plan.plan_id, "output_dir": str(out),
                          "written": {k: str(v) for k, v in written.items()},
                          "summary": plan.summary}, indent=2))
    else:
        print(rb.render_recipe_card(plan))
        print(f"\nplan written to: {out}")

