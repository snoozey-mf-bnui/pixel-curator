"""Feature 8A/8B - REDRESSER central pipeline orchestrator.

One cohesive entry point that chains the existing subsystems:

    validate -> wardrobe preset -> Forge payload -> (generate) ->
    region-aware inpaint -> post-process -> AnimateDiff sequence ->
    JSONL dataset logging -> structured result

Every optional subsystem (Forge backend, post-processing models, AnimateDiff)
degrades to a clean warning - the orchestrator never crashes on an
unavailable subsystem, and supports plan-only / dry-run when the Forge
backend is not reachable. Source files are never modified.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..eval.forge_api import ForgeClient
from ..lora_validator import normalize_stack_text, validate_lora_stack
from .generation_log import add_generation_to_log, build_sample_metadata, identity_anchor_hash
from .generation_mode import resolve_generation_mode

log = logging.getLogger("redresser.orchestrator")

MODES = ("static_catalog", "wardrobe_variation",
         "garment_transformation_sequence", "production_catalog",
         "phased_training")


class RedresserPipelineOrchestrator:
    """Coordinate a full personal catalog generation pipeline."""

    def __init__(self, forge_url: str = "http://127.0.0.1:7860",
                 output_base: Path | str = "outputs",
                 lora_dir: Path | str | None = None,
                 enable_post_processing: bool = False,
                 enable_dataset_log: bool = False,
                 dataset_name: str = "personal_catalog_dataset",
                 dry_run: bool = False) -> None:
        self.forge_url = forge_url
        self.output_base = Path(output_base).expanduser()
        self.lora_dir = Path(lora_dir).expanduser() if lora_dir else None
        self.enable_post_processing = bool(enable_post_processing)
        self.enable_dataset_log = bool(enable_dataset_log)
        self.dataset_name = dataset_name
        self.dry_run = bool(dry_run)
        self._client: ForgeClient | None = None

    # -- helpers -------------------------------------------------------------
    def _forge(self) -> ForgeClient | None:
        if self._client is None:
            self._client = ForgeClient(self.forge_url)
        return self._client

    def _backend_available(self) -> bool:
        if self.dry_run:
            return False
        try:
            return self._forge().alive()
        except Exception:
            return False

    @staticmethod
    def _load_preset(name: str) -> tuple[dict, list[str]]:
        from .. import starter_stacks as ss
        warns: list[str] = []
        preset = ss.get_wardrobe_preset(name) if hasattr(
            ss, "get_wardrobe_preset") else None
        if preset is None and hasattr(ss, "get_motion_preset"):
            preset = ss.get_motion_preset(name)
        if preset is None:
            warns.append(f"Wardrobe preset not found: '{name}'; using a "
                         "neutral catalog plan.")
            preset = {"name": name or "Neutral Catalog", "loras": [],
                      "prompt": "high-end catalog styling, controlled "
                      "wardrobe variation", "negative": "",
                      "recommended_regions": ["torso"]}
        return preset, warns

    # -- main ----------------------------------------------------------------
    #: default img2img denoising strength when an init image is provided and
    #: the caller does not specify one (catalog/portrait try-on sweet spot).
    DEFAULT_DENOISE = 0.60

    def run_full_pipeline(self, init_image_path: Path | str,
                          lora_stack: list, wardrobe_preset: str,
                          mode: str = "static_catalog",
                          quality_preset: str = "catalog_pro",
                          consistency_threshold: float = 0.65,
                          enable_quality_enhancement: bool = True,
                          denoising_strength: float = DEFAULT_DENOISE) -> dict:
        from ..util import now_iso
        result: dict = {
            "status": "success", "mode": mode,
            "init_image_path": str(Path(init_image_path).expanduser())
            if init_image_path else "",
            "wardrobe_preset": wardrobe_preset, "lora_stack": list(lora_stack)
            if not isinstance(lora_stack, str) else lora_stack,
            "validated_loras": {}, "payload": {},
            "original_outputs": [], "post_processed_outputs": [],
            "sequence_outputs": [], "sequence_id": "",
            "dataset_log": self.dataset_name,
            "controlnet_enabled": False, "ip_adapter_enabled": False,
            "post_processing_enabled": self.enable_post_processing,
            "identity_anchor_hash": identity_anchor_hash(init_image_path),
            "quality_preset": quality_preset,
            "consistency_threshold": consistency_threshold,
            "quality_metadata": {}, "quality_score": {},
            "generation_mode": "txt2img", "denoising_strength": None,
            "warnings": [], "errors": [],
        }
        if mode not in MODES:
            result["warnings"].append(
                f"Unknown mode '{mode}'; using 'static_catalog'.")
            mode = result["mode"] = "static_catalog"

        # validate denoising strength (img2img conditioning level)
        try:
            denoise = float(denoising_strength)
        except (TypeError, ValueError):
            denoise = self.DEFAULT_DENOISE
        if not 0.0 <= denoise <= 1.0:
            result["status"] = "error"
            result["errors"].append(
                f"denoising_strength must be in [0.0, 1.0], got {denoising_strength}")
            result["run_dir"] = ""
            return result

        # 1) source image -> central txt2img/img2img decision (single source of
        # truth). A non-empty but invalid path is a hard error, NOT a silent
        # txt2img fallback. phased_training uses init_image_path as a dataset
        # source, so it is exempt from image resolution.
        src = None
        if mode != "phased_training":
            resolution = resolve_generation_mode(init_image_path)
            result["generation_mode"] = resolution.mode
            if resolution.error:
                result["status"] = "error"
                result["errors"].append(resolution.error)
                log.warning("img2img init image rejected: %s", resolution.error)
                result["run_dir"] = ""
                return result
            src = resolution.init_image_path
            if resolution.is_img2img:
                result["denoising_strength"] = denoise
            else:
                result["warnings"].append(
                    "No init image provided; running txt2img / plan mode.")

        # 2) preset + 3) validated LoRA stack
        preset, pwarn = self._load_preset(wardrobe_preset)
        result["warnings"] += pwarn
        entries = (normalize_stack_text(lora_stack)
                   if isinstance(lora_stack, str) else
                   [str(x) for x in (lora_stack or [])])
        entries += list(preset.get("loras", []))
        cleaned, lwarn = validate_lora_stack(entries, lora_dir=self.lora_dir)
        result["validated_loras"] = cleaned
        result["warnings"] += lwarn
        if not cleaned["valid_pairs"]:
            result["warnings"].append(
                "No identity/garment LoRAs resolved; identity consistency "
                "may be reduced.")

        # output dir
        ts = now_iso().replace(":", "").replace("-", "")[:15]
        run_dir = self.output_base / "redresser_pipeline" / f"{ts}_{mode}"
        for sub in ("outputs", "post_processed", "sequences"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

        # 4/5/6) dispatch by mode
        backend = self._backend_available()
        if not backend:
            result["warnings"].append(
                "Forge backend not reachable; produced a validated payload "
                "plan (no image generated).")
        try:
            if mode == "garment_transformation_sequence":
                self._run_sequence(result, src, lora_stack, wardrobe_preset,
                                   run_dir, backend)
            elif mode == "wardrobe_variation":
                self._run_variation(result, src, cleaned, preset, run_dir,
                                    backend)
            elif mode == "production_catalog":
                self._run_production(result, src, cleaned, preset, run_dir,
                                     backend, quality_preset,
                                     consistency_threshold,
                                     enable_quality_enhancement)
            elif mode == "phased_training":
                self._run_phased(result, init_image_path, run_dir)
            else:
                self._run_static(result, src, cleaned, preset, run_dir,
                                 backend)
        except Exception as exc:                   # never crash the caller
            result["status"] = "completed_with_errors"
            result["errors"].append(f"Pipeline step error: {exc}")
            log.warning("Pipeline step error: %s", exc)

        # 8) post-processing already handled per-mode when applicable

        # 10) JSONL dataset logging (metadata always computed)
        if self.enable_dataset_log:
            report = add_generation_to_log(result, self.dataset_name,
                                           self.output_base)
            if not report.get("ok"):
                result["warnings"].append(
                    report.get("warning", "Dataset logging skipped."))
            result["dataset_report"] = report
        else:
            # still expose the metadata we WOULD log (dry-run friendly)
            outs = (result["post_processed_outputs"]
                    or result["original_outputs"] or result["sequence_outputs"]
                    or [""])
            result["dataset_metadata"] = [
                build_sample_metadata(result, output_path=str(o))
                for o in outs]

        # 11) structured result file
        self._write_result(run_dir, result)
        result["run_dir"] = str(run_dir)
        return result

    # -- phased training (Feature 11J integration) ---------------------------
    #: dashboard progress mapping by completed phase
    _PHASE_PROGRESS = {"classification": 25, "identity": 50,
                       "specialized_loras": 75, "merge": 100}
    _PHASE_ORDER = ("classification", "identity", "specialized_loras", "merge")

    def run_phased_training(
            self, dataset_source=None,
            base_checkpoint: str =
            "cyberrealisticPony_v180Coreshift.safetensors",
            target_profile: str = "Personalized CyberRealisticPony Base",
            phases: list | None = None, dataset_version: str = "v001",
            resume: bool = True, copy_mode: str = "copy") -> dict:
        """Public entry: plan the phased dataset + LoRA training pipeline.

        Chain: validate project path -> resolve dataset source -> run
        DatasetSplitter first (discover -> classify -> versioned export) ->
        hand the split result to the PhasedTrainingOrchestrator -> Core
        Identity / Specialized / MergeForge phase plans -> dashboard progress
        -> structured result. Always dry-run; never moves or modifies source
        media. Optional subsystems degrade to clean warnings.
        """
        from ..util import now_iso
        result: dict = {"status": "planned", "mode": "phased_training",
                        "dry_run": True, "warnings": [], "errors": []}
        ts = now_iso().replace(":", "").replace("-", "")[:15]
        run_dir = self.output_base / "redresser_pipeline" / f"{ts}_phased"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_phased(result, dataset_source, run_dir,
                         base_checkpoint=base_checkpoint,
                         target_profile=target_profile, phases=phases,
                         dataset_version=dataset_version, resume=resume,
                         copy_mode=copy_mode)
        result["run_dir"] = str(run_dir)
        self._write_result(run_dir, result)
        return result

    def _resolve_dataset_source(self, dataset_source, warnings: list):
        """Resolve a dataset source for the splitter. Never hard-fails.

        Accepts a raw folder, a list of paths/dicts, or an existing
        ``outputs/dataset_splits`` folder.
        Returns the resolved source (or None to plan without samples).
        """
        if dataset_source is None or (isinstance(dataset_source, str)
                                      and not dataset_source.strip()):
            warnings.append(
                "No dataset source provided; phased training requires a "
                "source folder or path list.")
            return None
        if isinstance(dataset_source, (list, tuple)):
            return list(dataset_source)
        src = str(dataset_source)
        p = Path(src).expanduser()
        if p.exists():
            return p
        warnings.append(
            f"Dataset source '{src}' not found on disk; planning without "
            "samples.")
        return None

    def _dashboard_block(self, plan: dict, splitter: dict) -> dict:
        phases = (plan or {}).get("phases", {}) or {}
        completed = [p for p in self._PHASE_ORDER
                     if phases.get(p, {}).get("status") in
                     ("completed", "skipped_completed")]
        current = completed[-1] if completed else self._PHASE_ORDER[0]
        progress = self._PHASE_PROGRESS.get(completed[-1], 0) if completed \
            else 0
        return {
            "current_phase": current,
            "progress_percent": progress,
            "phase_status": {p: phases.get(p, {}).get("status", "pending")
                             for p in self._PHASE_ORDER},
            "dataset_source": splitter.get("source", ""),
            "dry_run": True,
            "resume": splitter.get("resume", True),
            "discovered_count": splitter.get("discovered_count", 0),
            "classified_count": splitter.get("classified_count", 0),
            "target_counts": splitter.get("target_counts", {}),
            "core_identity_dataset_status":
                "ready" if splitter.get("target_counts", {}).get(
                    "core_identity") else "empty",
            "specialized_dataset_status": {
                t: ("ready" if splitter.get("target_counts", {}).get(t)
                    else "empty")
                for t in ("outfit_clothing", "style_aesthetic", "pose_action",
                          "garment_detail")},
            "manifest_path": (plan or {}).get("manifest_path", ""),
            "warnings": splitter.get("warnings", []),
        }

    def _run_phased(self, result, dataset_source, run_dir,
                    base_checkpoint:
                    str = "cyberrealisticPony_v180Coreshift.safetensors",
                    target_profile:
                    str = "Personalized CyberRealisticPony Base",
                    phases: list | None = None, dataset_version: str = "v001",
                    resume: bool = True, copy_mode: str = "copy") -> None:
        """Run DatasetSplitter first, then the PhasedTrainingOrchestrator.

        Never crashes: an unavailable subsystem becomes a structured warning.
        """
        result.setdefault("dry_run", True)
        project_path = (self.output_base.parent
                        if self.output_base.name == "outputs"
                        else self.output_base)

        # 1) DatasetSplitter is the first phase owner.
        splitter_result = {"status": "planned", "discovered_count": 0,
                           "classified_count": 0, "target_counts": {},
                           "exports": {}, "warnings": [], "resume": resume}
        resolved = self._resolve_dataset_source(dataset_source,
                                                result["warnings"])
        if resolved is not None:
            try:
                from ..curate.dataset_splitter import DatasetSplitter
                splitter = DatasetSplitter(
                    resolved, output_root=project_path / "outputs",
                    dry_run=True, resume=resume, copy_mode=copy_mode)
                splitter_result = splitter.run(version=dataset_version)
                splitter_result["resume"] = resume
            except Exception as exc:                 # pragma: no cover
                result["warnings"].append(f"DatasetSplitter unavailable: {exc}")
        result["dataset_splitter"] = {
            "status": splitter_result.get("status", "planned"),
            "discovered_count": splitter_result.get("discovered_count", 0),
            "classified_count": splitter_result.get("classified_count", 0),
            "target_counts": splitter_result.get("target_counts", {}),
            "exports": splitter_result.get("exports", {}),
            "warnings": splitter_result.get("warnings", []),
        }

        # 2) hand the split result to the PhasedTrainingOrchestrator.
        try:
            from ..train.phased_orchestrator import PhasedTrainingOrchestrator
        except Exception as exc:                     # pragma: no cover
            result["warnings"].append(
                f"Phased training orchestrator unavailable: {exc}")
            result["phased_training"] = {}
            result["dashboard"] = self._dashboard_block({}, splitter_result)
            return
        pto = PhasedTrainingOrchestrator(project_path, dry_run=True)
        plan = pto.run_phased_training(
            project_path=str(project_path), base_checkpoint=base_checkpoint,
            phases=phases, dataset_source=dataset_source,
            dataset_split_result=splitter_result, dry_run=True, resume=resume)

        result["phased_training"] = plan
        # canonical base profile name from the plan (back-compat); the chosen
        # wardrobe/preset profile is recorded separately.
        result["target_profile"] = plan.get("target_profile")
        result["requested_profile"] = target_profile
        result["manifest_path"] = plan.get("manifest_path")
        result["dataset_split_counts"] = plan.get("dataset_split_counts", {})
        result["training_commands"] = plan.get("training_commands", {})
        result["dashboard"] = self._dashboard_block(plan, splitter_result)
        result["warnings"] += [w for w in plan.get("warnings", [])
                               if w not in result["warnings"]]
        result["warnings"] += [w for w in splitter_result.get("warnings", [])
                               if w not in result["warnings"]]
        result["errors"] += plan.get("errors", [])

    # -- modes ---------------------------------------------------------------
    def _build_payload(self, cleaned: dict, preset: dict) -> dict:
        prompt = str(preset.get("prompt", "")).strip(", ")
        frag = cleaned["prompt_fragment"]
        if frag:
            prompt = f"{prompt} {frag}".strip()
        negative = str(preset.get("negative", "")).strip(", ")
        return ForgeClient.build_txt2img_payload(
            prompt=prompt, negative=negative, clip_skip=2)

    def _run_static(self, result, src, cleaned, preset, run_dir, backend):
        payload = self._build_payload(cleaned, preset)
        is_img2img = bool(src and src.is_file())
        result["generation_mode"] = "img2img" if is_img2img else "txt2img"
        if is_img2img:
            denoise = float(result.get("denoising_strength")
                            or self.DEFAULT_DENOISE)
            payload["denoising_strength"] = denoise
        result["payload"] = payload
        prompt = payload["prompt"].split(" <lora:")[0]
        log.info("static generation_mode=%s denoise=%s init=%s",
                 result["generation_mode"],
                 payload.get("denoising_strength"), str(src) if src else "")
        if backend:                                # pragma: no cover
            if is_img2img:
                png = self._forge().img2img(
                    init_png=src.read_bytes(), prompt=prompt,
                    negative=payload["negative_prompt"],
                    strength=float(payload["denoising_strength"]),
                    loras=cleaned["valid_pairs"], clip_skip=2)
            else:
                png = self._forge().txt2img(
                    prompt=prompt, negative=payload["negative_prompt"],
                    loras=cleaned["valid_pairs"], clip_skip=2)
            out = run_dir / "outputs" / "catalog_000.png"
            out.write_bytes(png)
            result["original_outputs"].append(str(out))
            self._maybe_post(result, run_dir)

    def _run_production(self, result, src, cleaned, preset, run_dir, backend,
                        quality_preset, threshold, enable_quality):
        from .production_quality import ProductionQualityEnhancer, evaluate_output_quality
        base = self._build_payload(cleaned, preset)
        if enable_quality:
            caps = {}
            if backend:                            # probe backend caps
                try:
                    ups = self._forge().upscalers()
                    caps["hires"] = bool(ups)
                    scripts = self._forge()._get("/sdapi/v1/scripts")
                    import json as _j
                    caps["adetailer"] = "adetailer" in _j.dumps(
                        scripts).lower()
                except Exception:
                    caps = {}
            enhanced = ProductionQualityEnhancer(
                backend_caps=caps).enhance_generation_payload(
                base, quality_preset)
            result["payload"] = enhanced["payload"]
            result["quality_metadata"] = enhanced["quality_metadata"]
            result["warnings"] += enhanced["quality_metadata"]["warnings"]
            result["controlnet_enabled"] = bool(
                (base.get("alwayson_scripts") or {}).get("controlnet"))
        else:
            result["payload"] = base
        # generation (best-effort) reuses the static path settings
        is_img2img = bool(src and src.is_file())
        result["generation_mode"] = "img2img" if is_img2img else "txt2img"
        p = result["payload"]
        if is_img2img:
            denoise = float(result.get("denoising_strength")
                            or self.DEFAULT_DENOISE)
            p["denoising_strength"] = denoise
        log.info("production generation_mode=%s denoise=%s init=%s",
                 result["generation_mode"], p.get("denoising_strength"),
                 str(src) if src else "")
        if backend:                                # pragma: no cover
            prompt = p["prompt"].split(" <lora:")[0]
            if is_img2img:
                png = self._forge().img2img(
                    init_png=src.read_bytes(), prompt=prompt,
                    negative=p.get("negative_prompt", ""),
                    strength=float(p["denoising_strength"]),
                    steps=int(p.get("steps", 28)),
                    cfg=float(p.get("cfg_scale", 6)),
                    sampler=p.get("sampler_name", "DPM++ 2M Karras"),
                    loras=cleaned["valid_pairs"], clip_skip=2)
            else:
                png = self._forge().txt2img(
                    prompt=prompt, negative=p.get("negative_prompt", ""),
                    steps=int(p.get("steps", 28)),
                    cfg=float(p.get("cfg_scale", 6)),
                    sampler=p.get("sampler_name", "DPM++ 2M Karras"),
                    loras=cleaned["valid_pairs"], clip_skip=2)
            out = run_dir / "outputs" / "production_000.png"
            out.write_bytes(png)
            result["original_outputs"].append(str(out))
            self._maybe_post(result, run_dir)
        # quality scoring (planned when no metric backend)
        target = (result["post_processed_outputs"]
                  or result["original_outputs"] or [""])[0]
        result["quality_score"] = evaluate_output_quality(
            target, preset=quality_preset, threshold=threshold)
        if not backend:
            result["warnings"].append(
                "Production payload enhanced and post-processing chain "
                "planned; generation pending an available Forge backend.")

    def _run_variation(self, result, src, cleaned, preset, run_dir, backend):
        region = (preset.get("recommended_regions") or ["torso"])[0]
        result["controlnet_enabled"] = True
        result["ip_adapter_enabled"] = True
        payload = self._build_payload(cleaned, preset)
        payload["_region"] = region
        payload["_denoise"] = 0.45
        result["payload"] = payload
        if not src or not src.is_file():
            result["warnings"].append(
                "Region-aware wardrobe variation needs a source image; "
                "produced an inpaint payload plan only.")
            return
        if backend:                                # pragma: no cover
            from ..eval.forge_api import region_mask_png
            init = src.read_bytes()
            mask = region_mask_png(init, region)
            png = self._forge().inpaint(
                init, mask, prompt=payload["prompt"].split(" <lora:")[0],
                negative=payload["negative_prompt"], denoise=0.45,
                loras=cleaned["valid_pairs"], clip_skip=2)
            out = run_dir / "outputs" / "variation_000.png"
            out.write_bytes(png)
            result["original_outputs"].append(str(out))
            self._maybe_post(result, run_dir)

    def _run_sequence(self, result, src, lora_stack, preset_name, run_dir,
                      backend):
        from .animation import GarmentTransformationSequence
        seq = GarmentTransformationSequence(
            forge_api=self._forge() if backend else None,
            output_dir=run_dir / "sequences", lora_dir=self.lora_dir,
            enable_post_processing=self.enable_post_processing)
        frames = seq.generate_garment_transformation_sequence(
            init_image_path=str(src) if src else "",
            lora_stack=lora_stack, target_preset=preset_name,
            num_frames=8, fps=12)
        result["sequence_outputs"] = [str(p) for p in frames]
        # surface the sequence manifest id + warnings
        seq_dirs = sorted((run_dir / "sequences").glob("*"))
        if seq_dirs:
            mf = seq_dirs[-1] / "sequence_manifest.json"
            if mf.is_file():
                data = json.loads(mf.read_text())
                result["sequence_id"] = data.get("sequence_id", "")
                result["payload"] = {"sequence_manifest": str(mf)}
                result["warnings"] += [w for w in data.get("warnings", [])
                                       if w not in result["warnings"]]

    def _maybe_post(self, result, run_dir):        # pragma: no cover
        if not self.enable_post_processing:
            return
        try:
            from ..post_processing import PostProcessor
        except Exception:
            result["warnings"].append("Post-processing module unavailable.")
            return
        pp = PostProcessor()
        for src_path in list(result["original_outputs"]):
            res = pp.process(src_path, gfpgan_weight=0.3)
            result["warnings"] += res.warnings
            tgt = Path(res.post_processed_path)
            if tgt != Path(src_path) and tgt.is_file():
                dest = run_dir / "post_processed" / tgt.name
                dest.write_bytes(tgt.read_bytes())
                result["post_processed_outputs"].append(str(dest))

    @staticmethod
    def _write_result(run_dir: Path, result: dict) -> Path:
        path = run_dir / "pipeline_result.json"
        try:
            path.write_text(json.dumps(result, indent=2, default=str),
                            encoding="utf-8")
        except Exception:                          # pragma: no cover
            pass
        return path
