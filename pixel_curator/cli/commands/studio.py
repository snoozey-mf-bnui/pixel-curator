"""CLI commands: studio."""

from __future__ import annotations

import logging

from ..common import *  # noqa: F401,F403

logger = logging.getLogger(__name__)


def cmd_workflow(ctx: Context):
    """Handler for the `workflow` command."""
    args = ctx.args
    from ...workflow.orchestrator import (
        WorkflowRunConfig,
        WorkflowStageConfig,
        build_workflow_plan,
        load_workflow_config,
        load_workflow_state,
        run_workflow,
    )
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


def cmd_pipeline(ctx: Context):
    """Handler for the `pipeline` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    # NOTE: aliased import - a bare `pipeline_generator` here would shadow
    # the module-level extract.pipeline_generator across ALL branches
    from ...pipeline_dag import PipelineConfig, load_state, resume_generator
    from ...pipeline_dag import pipeline_generator as dag_generator
    from ...pipeline_recipes import (
        RecipePipelineConfig,
        load_recipe_state,
        recipe_pipeline_generator,
        recipe_resume_generator,
        render_pipeline_list,
        render_plan,
        resolve_pipeline,
    )
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
            get_emitter().emit(update)
        get_emitter().close()
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
        from ...pipeline_dag import MegaConfig, mega_generator
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


def cmd_ab(ctx: Context):
    """Handler for the `ab` command."""
    args, prj = ctx.args, ctx.prj
    from ...qa_render import ab_compare
    def _split(s):
        n, _, w = s.rpartition(":")
        return (n or s, float(w) if w and n else 0.75)
    for u in ab_compare(prj, _split(args.a), _split(args.b),
                        prompt_tail=args.tail, seed=args.seed):
        print(u)


def cmd_golden(ctx: Context):
    """Handler for the `golden` command."""
    args, prj = ctx.args, ctx.prj
    from ...qa_render import golden_regression
    for u in golden_regression(prj, args.lora, args.weight,
                               args.previous or None):
        print(u)


def cmd_study(ctx: Context):
    """Handler for the `study` command."""
    args, prj = ctx.args, ctx.prj
    from . import study as study_mod
    if args.study_cmd == "classify":
        from ...study import StudyConfig, classify_generator
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

                from ...config import save_project
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


def cmd_studio(ctx: Context):
    """Handler for the `studio` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from . import studio as st_mod
    scmd = getattr(args, "studio_cmd", None) or "status"
    base = out_base(getattr(args, "output", None))
    as_json = getattr(args, "json", False)
    if scmd in ("status", "plan", "next"):
        from ...studio.state import render_studio_state
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
        from ...studio.readiness import batch_readiness, dataset_readiness, run_readiness
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
        from ...studio.asset_registry import render_registry
        conn = None
        try:
            conn = manifest.connect(prj.output_path, readonly=True)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
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


def cmd_learn(ctx: Context):
    """Handler for the `learn` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
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


def cmd_recommend_stack(ctx: Context):
    """Handler for the `recommend-stack` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...artistic.stack_recommender import (
        recommend_best_epoch_and_stack,
        recommend_best_stack,
        recommend_figure_study_stack,
        render_epoch_stack,
        render_figure_study_recommendation,
        render_recommendation,
    )
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
        from ...pipeline_dag import write_playground_preset
        loras = [(x["name"], x["weight"])
                 for x in rec["recommended_stack"]]
        target = write_playground_preset(prj, preset_label, loras)
        print(f"Playground preset written -> {target}")


def cmd_analyze(ctx: Context):
    """Handler for the `analyze` command."""
    prj = ctx.prj
    from ...wizard import analyze, detect_type
    conn = manifest.connect(prj.output_path)
    a = analyze(conn)
    print(json.dumps(a, indent=2))
    print("\nType detection:")
    for r in detect_type(a):
        print(f"  {r['type']:<10} {r['score']:.2f}  ({r['reason']})")


def cmd_wizard(ctx: Context):
    """Handler for the `wizard` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...wizard import WizardConfig, wizard_generator
    cfg = WizardConfig(
        output_base=out_base(args.output), lora_type=args.lora_type,
        trigger=args.trigger, class_word=args.class_word, name=args.name,
        train=args.train, matrix=args.matrix, matrix_backend=args.backend,
    )
    for update in wizard_generator(prj, cfg):
        print(update)


def cmd_testgen(ctx: Context):
    """Handler for the `testgen` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    if args.backend == "forge":
        from ...eval.forge_dispatch import dispatch_forge_generate
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
    from ...eval.matrix import parse_lora_specs
    loras = []
    for spec in args.lora:
        loras.extend(parse_lora_specs(spec))
    from ...eval.pipeline import diffusers_available, get_pipeline
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


def cmd_generate_artistic(ctx: Context):
    """Handler for the `generate-artistic` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...artistic import compose_artistic_prompt, get_recipe, list_recipes
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
    from ...artistic.workflows import (
        build_artistic_txt2img_payload,
        build_selective_region_payload,
        resolve_identity_lora,
    )
    id_name, id_warnings = resolve_identity_lora(prj)
    trig = f"{prj.trigger_token} {prj.class_word}".strip()
    if args.batch > 1 and not args.image:
        # controlled-variation batch (Phase 4) - shared planner
        from ...artistic import plan_figure_study_batch, write_batch_manifest
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
        from ...pipeline_dag import write_playground_preset
        loras = [(s["name"], s["weight"]) for s in composed.lora_stack
                 if s["weight"] > 0]
        target = write_playground_preset(
            prj, f"artistic:{composed.recipe_name}", loras)
        print(f"  Playground preset written -> {target}")
    if not (args.json or args.send_to_playground):
        print("  (payload built - add --json to inspect, "
              "--send-to-playground to hand off)")


def cmd_generate_figure_study(ctx: Context):
    """Handler for the `generate-figure-study` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...artistic import (
        compose_artistic_prompt,
        get_recipe,
        list_negative_chains,
        list_recipes,
        list_variation_axes,
        plan_figure_study_batch,
        write_batch_manifest,
    )
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
    from ...artistic.workflows import build_figure_study_region_payload, resolve_identity_lora
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
        from ...artistic.guidance import plan_figure_study_guidance, render_guidance
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
        from ...artistic import plan_production_batch, write_production_batch
        from ...artistic.batch_review import select_best_figure_study_variant
        from ...artistic.guidance import render_guidance
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
            from ...artistic import axes_for_variation_strength
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
            from ...artistic.guidance import render_guidance
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
            from ...pipeline_dag import write_playground_preset
            loras = [(s["name"], s["weight"])
                     for s in plan["items"][0]["lora_stack"]
                     if s["weight"] > 0]
            target = write_playground_preset(
                prj, f"figure_study:{recipe.name}", loras)
            print(f"  Playground preset written -> {target}")


def cmd_figure_review(ctx: Context):
    """Handler for the `figure-review` command."""
    args, out_base = ctx.args, ctx.out_base
    from ...artistic.batch_review import (
        export_review,
        find_batch_dir,
        list_batches,
        load_batch_manifest,
        mark_item,
        render_review_summary,
        review_summary,
        select_best_figure_study_variant,
    )
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


def cmd_package_figure_study(ctx: Context):
    """Handler for the `package-figure-study` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...artistic.packaging import package_figure_study
    base = out_base(args.output)
    conn = None
    try:
        conn = manifest.connect(base, readonly=True)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
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


def cmd_render_figure_study(ctx: Context):
    """Handler for the `render-figure-study` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...artistic.batch_render import render_batch
    from ...artistic.batch_review import find_batch_dir
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


def cmd_exp(ctx: Context):
    """Handler for the `exp` command."""
    args, prj = ctx.args, ctx.prj
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


def cmd_lookbook(ctx: Context):
    """Handler for the `lookbook` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...artistic.lookbook import (
        LookbookConfig,
        lookbook_generator,
        parse_list,
        parse_lora_specs,
    )
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


def cmd_plan(ctx: Context):
    """Handler for the `plan` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...builder import plan_generator
    for update in plan_generator(prj, out_base(args.output),
                                 args.recipe or None):
        print(update)


def cmd_build(ctx: Context):
    """Handler for the `build` command."""
    args, prj, out_base = ctx.args, ctx.prj, ctx.out_base
    from ...builder import BuildConfig, build_generator
    cfg = BuildConfig(
        output_base=out_base(args.output), recipe=args.recipe,
        note=args.note, link_mode=args.link_mode,
        parent=args.parent, training_config=args.training_config,
    )
    for update in build_generator(prj, cfg):
        print(update)


def cmd_face_fix(ctx: Context):
    """Handler for the `face-fix` command."""
    args, prj = ctx.args, ctx.prj
    from ...services import face_consistency_studio as fcs
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
            logger.debug("suppressed exception", exc_info=True)
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

