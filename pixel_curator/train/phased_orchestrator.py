"""Feature 11A/11F/11G/11K - Phased Training Orchestrator.

Chains the full personal-base pipeline as a dry-run-first plan:

    raw personal media archive
      -> local classification               (phase: classification)
      -> quality filtering + dataset splits
      -> versioned dataset exports
      -> core identity LoRA training plan     (phase: identity)
      -> specialized LoRA training plans       (phase: specialized_loras)
      -> evaluation plan
      -> MergeForge CyberRealisticPony Personal Base plan (phase: merge)
      -> updated base profile + reusable assets

Everything defaults to dry-run / planning. No source media is moved or
modified, no training or merge is executed, and no base checkpoint is
overwritten. A ``phased_training_manifest.json`` tracks per-phase status and
supports resume (completed phases are not re-run or overwritten unless forced).

Pure stdlib, local-first. Optional subsystems degrade to clean warnings.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from . import base_profile as bp
from . import dataset_classifier as dc
from . import kohya_config_builder as kcb

log = logging.getLogger("redresser.phased_orchestrator")

DEFAULT_PHASES: tuple[str, ...] = (
    "classification", "identity", "specialized_loras", "merge")

#: trainable specialized targets handled in the specialized_loras phase
SPECIALIZED_ORDER: tuple[str, ...] = (
    "outfit", "style", "pose", "garment_detail")

#: map DatasetSplitter canonical labels -> phased-training target labels
SPLIT_TO_TRAIN_TARGET: dict[str, str] = {
    "core_identity": "core_identity",
    "outfit_clothing": "outfit",
    "style_aesthetic": "style",
    "pose_action": "pose",
    "garment_detail": "garment_detail",
    "catalog_reference": "catalog_reference",
    "needs_review": "needs_review",
    "excluded": "excluded",
}


class PhasedTrainingOrchestrator:
    """Coordinate phased dataset curation and LoRA training planning."""

    def __init__(self, project_path, settings=None,
                 dry_run: bool = True) -> None:
        self.project_path = Path(project_path).expanduser()
        self.settings = settings
        self.dry_run = bool(dry_run)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _load_samples(dataset_source) -> tuple[list[dict], list[str]]:
        """Best-effort load of sample metadata dicts. Never raises.

        Accepts: a list of dicts (used directly), a path to a JSON file
        containing a list of dicts, or a folder (each image becomes a minimal
        sample row). Returns (samples, warnings).
        """
        warnings: list[str] = []
        if dataset_source is None:
            return [], ["No dataset source provided; planning with empty split."]
        if isinstance(dataset_source, (list, tuple)):
            return [dict(s) for s in dataset_source], warnings
        src = Path(str(dataset_source)).expanduser()
        if src.is_file() and src.suffix.lower() == ".json":
            try:
                data = json.loads(src.read_text(encoding="utf-8"))
                rows = data.get("samples", data) if isinstance(data, dict) else data
                return [dict(r) for r in rows], warnings
            except Exception as exc:
                return [], [f"Could not read dataset JSON: {exc}"]
        if src.is_dir():
            exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            rows = [{"sample_path": str(p)} for p in sorted(src.iterdir())
                    if p.suffix.lower() in exts]
            if not rows:
                warnings.append(
                    f"No images found in dataset folder: {src}")
            return rows, warnings
        warnings.append(f"Dataset source not found: {dataset_source}")
        return [], warnings

    @staticmethod
    def _read_manifest(path: Path) -> dict | None:
        try:
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:                           # pragma: no cover
            pass
        return None

    def _write_manifest(self, path: Path, manifest: dict) -> None:
        if self.dry_run:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest, indent=2, default=str),
                            encoding="utf-8")
        except Exception as exc:                    # pragma: no cover
            log.warning("manifest write failed: %s", exc)

    # -- main ----------------------------------------------------------------
    def run_phased_training(
            self, project_path: str | None = None,
            base_checkpoint: str = bp.BASE_CHECKPOINT,
            phases: list[str] | None = None,
            dataset_source=None,
            dataset_split_result: dict | None = None,
            dry_run: bool | None = None,
            resume: bool = True,
            force: bool = False) -> dict:
        """Plan (and, only with explicit non-dry-run, execute) phased training.

        Returns the structured Feature-11 result. Never raises for ordinary
        problems; optional-subsystem issues are reported as warnings.
        """
        if dry_run is not None:
            self.dry_run = bool(dry_run)
        phases = list(phases) if phases else list(DEFAULT_PHASES)
        project_path = str(project_path or self.project_path)

        ts = self._timestamp()
        run_dir = (self.project_path.parent if self.project_path.name
                   else self.project_path)
        # Use a stable outputs/ tree under the project path.
        base_out = Path(project_path).expanduser() / "outputs" / \
            "phased_training" / f"{ts}_cyberrealisticpony_personal_base"
        manifest_path = base_out / "phased_training_manifest.json"

        result: dict = {
            "status": "planned",
            "dry_run": self.dry_run,
            "base_checkpoint": base_checkpoint,
            "target_profile": bp.BASE_PROFILE_NAME,
            "phases": {},
            "project_path": project_path,
            "manifest_path": str(manifest_path),
            "run_dir": str(base_out),
            "dataset_exports": {},
            "training_configs": {},
            "training_commands": {},
            "mergeforge_plan": {},
            "base_profile": {},
            "dataset_split_counts": {},
            "warnings": [],
            "errors": [],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "run_id": ts,
            "selected_phases": phases,
        }

        if base_checkpoint != bp.BASE_CHECKPOINT:
            result["warnings"].append(
                f"Requested base checkpoint '{base_checkpoint}' differs from "
                f"the personal-base target '{bp.BASE_CHECKPOINT}'.")

        # output folder scaffold (only when actually running)
        subdirs = ["classification", "datasets", "configs", "commands",
                   "checkpoints", "evaluation", "mergeforge"]
        if not self.dry_run:
            for sub in subdirs:
                (base_out / sub).mkdir(parents=True, exist_ok=True)

        # resume: load prior manifest if present
        prior = self._read_manifest(manifest_path) if resume else None
        completed_prior = set()
        if prior:
            completed_prior = {p for p, st in
                               (prior.get("phases", {}) or {}).items()
                               if st.get("status") == "completed"}
            result["warnings"].append(
                f"Resuming from existing manifest; completed phases preserved: "
                f"{sorted(completed_prior) or 'none'}.")

        # When a DatasetSplitter result is supplied it is the first phase
        # owner: classification is taken from it rather than re-splitting.
        preclassified: list[dict] | None = None
        counts_override: dict | None = None
        if dataset_split_result is not None:
            preclassified = self._classified_from_split(dataset_split_result)
            counts_override = self._counts_from_split(dataset_split_result)
            self._record_split_handoff(result, dataset_split_result)
            samples: list[dict] = []
        else:
            # load samples once (used by classification phase)
            samples, swarn = self._load_samples(dataset_source)
            result["warnings"] += swarn

        classified: list[dict] = []

        for phase in phases:
            if phase in completed_prior and resume and not force:
                result["phases"][phase] = dict(
                    prior["phases"][phase],
                    status="skipped_completed")
                continue
            try:
                if phase == "classification":
                    classified = self._phase_classification(
                        result, samples, base_out,
                        preclassified=preclassified,
                        counts_override=counts_override)
                elif phase == "identity":
                    self._phase_identity(result, classified, base_out,
                                         base_checkpoint)
                elif phase == "identity_training":
                    self._phase_identity_training(result, classified,
                                                  base_out, base_checkpoint)
                elif phase == "specialized_loras":
                    self._phase_specialized(result, classified, base_out,
                                            base_checkpoint)
                elif phase == "merge":
                    self._phase_merge(result, base_out, base_checkpoint)
                else:
                    result["phases"][phase] = {
                        "status": "skipped",
                        "warnings": [f"Unknown phase '{phase}'."]}
                    result["warnings"].append(f"Unknown phase '{phase}'.")
            except Exception as exc:                 # never crash caller
                result["phases"][phase] = {"status": "error",
                                           "errors": [str(exc)]}
                result["errors"].append(f"Phase '{phase}' error: {exc}")
                log.warning("phase %s error: %s", phase, exc)

        result["updated_at"] = datetime.now().isoformat(timespec="seconds")
        if result["errors"]:
            result["status"] = "completed_with_errors"
        self._write_manifest(manifest_path, result)
        return result

    # -- split-result handoff (Feature 11 integration) -----------------------
    @staticmethod
    def _classified_from_split(dsr: dict) -> list[dict]:
        """Build a phased-target classified list from a DatasetSplitter result.

        Uses the splitter's per-target export samples (existing files only) so
        training plans reference real, exported assets. Never raises.
        """
        rows: list[dict] = []
        for split_target, manifest in (dsr.get("exports") or {}).items():
            tgt = SPLIT_TO_TRAIN_TARGET.get(split_target, split_target)
            for s in (manifest.get("samples") or []):
                rows.append({
                    "sample_path": s.get("source_path", ""),
                    "lora_target": tgt,
                    "source_type": s.get("source_type", "image"),
                    "confidence": s.get("confidence"),
                    "reason_codes": list(s.get("reason_codes") or [])
                    + ["dataset_splitter"],
                    "warnings": [],
                })
        return rows

    @staticmethod
    def _counts_from_split(dsr: dict) -> dict:
        """Map the splitter's classification counts into phased-target space."""
        counts = {t: 0 for t in dc.LORA_TARGETS}
        for split_target, n in (dsr.get("target_counts") or {}).items():
            tgt = SPLIT_TO_TRAIN_TARGET.get(split_target, split_target)
            counts[tgt] = counts.get(tgt, 0) + int(n or 0)
        return counts

    @staticmethod
    def _record_split_handoff(result: dict, dsr: dict) -> None:
        """Record DatasetSplitter -> phased manifest handoff fields."""
        exports = dsr.get("exports") or {}

        def _dir(t):
            return (exports.get(t) or {}).get("export_dir", "")
        result["dataset_split_handoff"] = {
            "dataset_split_manifest": dsr.get("splits_root", ""),
            "dataset_exports": {t: _dir(t) for t in exports},
            "target_counts": dsr.get("target_counts", {}),
            "core_identity_dataset": _dir("core_identity"),
            "specialized_datasets": {
                "outfit_clothing": _dir("outfit_clothing"),
                "style_aesthetic": _dir("style_aesthetic"),
                "pose_action": _dir("pose_action"),
                "garment_detail": _dir("garment_detail"),
            },
        }

    # -- phases --------------------------------------------------------------
    def _phase_classification(self, result, samples, base_out,
                              preclassified: list[dict] | None = None,
                              counts_override: dict | None = None
                              ) -> list[dict]:
        if preclassified is not None:
            classified = preclassified
            counts = counts_override if counts_override is not None \
                else dc.split_counts(classified)
            source = "dataset_splitter"
        else:
            classified = dc.classify_samples(samples)
            counts = dc.split_counts(classified)
            source = "metadata_classifier"
        result["dataset_split_counts"] = counts
        warns = []
        if any("Advanced classifier unavailable" in w
               for row in classified for w in row.get("warnings", [])):
            warns.append(
                "Advanced classifier unavailable; using metadata-based "
                "dataset splitting.")
        result["phases"]["classification"] = {
            "status": "completed",
            "source": source,
            "sample_count": sum(counts.values()) if preclassified is not None
            else len(classified),
            "split_counts": counts,
            "warnings": warns,
        }
        # write classification record when running for real
        if not self.dry_run:
            (base_out / "classification").mkdir(parents=True, exist_ok=True)
            (base_out / "classification" / "classification.json").write_text(
                json.dumps(classified, indent=2), encoding="utf-8")
        return classified

    def _phase_identity(self, result, classified, base_out, base_checkpoint):
        groups = dc.group_by_target(classified)
        core = groups.get("core_identity", [])
        datasets_root = base_out / "datasets"
        # versioned export plan
        export = kcb.export_lora_target_dataset(
            core, "core_identity", datasets_root, dry_run=self.dry_run)
        result["dataset_exports"]["core_identity"] = export
        # kohya config + command
        dataset_dir = Path(export["export_dir"]) / "images"
        out_dir = base_out / "checkpoints" / "redresser_core_identity"
        cfg = kcb.build_core_identity_kohya_config(
            dataset_dir, out_dir, base_checkpoint=base_checkpoint,
            config_dir=base_out / "configs",
            command_dir=base_out / "commands", dry_run=self.dry_run)
        result["training_configs"]["core_identity"] = cfg
        result["training_commands"]["core_identity"] = cfg["command"]
        warns = list(export.get("warnings", [])) + list(cfg.get("warnings", []))
        if not core:
            warns.append("No core_identity samples resolved; identity LoRA "
                         "plan created with an empty dataset.")
        result["phases"]["identity"] = {
            "status": "completed",
            "dataset_export": export["export_dir"],
            "config_path": cfg["config_path"],
            "command": cfg["command"],
            "sample_count": len(core),
            "warnings": warns,
        }

    def _phase_identity_training(self, result, classified, base_out,
                                 base_checkpoint):
        """Plan (dry-run) or execute the Core Identity training run via the
        core_identity_trainer, reusing the exported core_identity dataset."""
        # ensure the identity export + config exist (idempotent, dry-run safe)
        if "identity" not in result.get("phases", {}):
            self._phase_identity(result, classified, base_out, base_checkpoint)
        export = result.get("dataset_exports", {}).get("core_identity", {})
        dataset_dir = Path(export.get("export_dir", base_out / "datasets"
                                      / "v001_core_identity")) / "images"
        out_dir = base_out / "checkpoints" / "redresser_core_identity"
        try:
            from . import core_identity_trainer as cit
            train = cit.run_core_identity_training(
                dataset_dir, out_dir, base_checkpoint=base_checkpoint,
                output_name=cit.DEFAULT_OUTPUT_NAME,
                dry_run=self.dry_run, generate_grid=True)
        except Exception as exc:                     # never crash the caller
            result["phases"]["identity_training"] = {
                "status": "error", "errors": [str(exc)]}
            result["errors"].append(f"identity_training error: {exc}")
            return
        result.setdefault("training", {})["core_identity"] = train
        result["phases"]["identity_training"] = {
            "status": train.get("status", "planned"),
            "dry_run": train.get("dry_run", True),
            "lora_output_path": train.get("lora_output_path", ""),
            "config_path": train.get("config_path", ""),
            "command": train.get("command", ""),
            "test_grid_status": train.get("test_grid", {}).get("status", ""),
            "warnings": train.get("warnings", []),
        }

    def _phase_specialized(self, result, classified, base_out,
                           base_checkpoint):
        """Run specialized refinement LoRAs sequentially (after Core Identity).

        Each target: export + verify its dataset, then plan/execute training
        through the specialized trainer (dry-run by default; never parallel,
        never overwrites). Falls back to config-only planning if the trainer
        module is unavailable.
        """
        try:
            from . import specialized_trainer as st
        except Exception:                            # pragma: no cover
            st = None
        groups = dc.group_by_target(classified)
        datasets_root = base_out / "datasets"
        phase_info = {"status": "completed", "order": list(SPECIALIZED_ORDER),
                      "targets": {}, "warnings": []}
        result.setdefault("training", {})
        for target in SPECIALIZED_ORDER:             # sequential, ordered
            rows = groups.get(target, [])
            export = kcb.export_lora_target_dataset(
                rows, target, datasets_root, dry_run=self.dry_run)
            result["dataset_exports"][target] = export
            dataset_dir = Path(export["export_dir"]) / "images"
            out_dir = base_out / "checkpoints" / \
                kcb.SPECIALIZED_TARGETS[target]["output_name"]

            if st is not None:
                run = st.run_specialized_training(
                    target, dataset_dir, out_dir,
                    base_checkpoint=base_checkpoint, dry_run=self.dry_run,
                    generate_grid=False, resume=True)
                result["training"].setdefault("specialized", {})[target] = run
                result["training_configs"][target] = {
                    "config_path": run.get("config_path", ""),
                    "command": run.get("command", ""),
                    "config": {"output_name": run.get("output_name", "")},
                }
                result["training_commands"][target] = run.get("command", "")
                phase_info["targets"][target] = {
                    "sample_count": len(rows),
                    "training_status": run.get("status", "planned"),
                    "config_path": run.get("config_path", ""),
                    "command": run.get("command", ""),
                    "recommended_network_dim": run.get(
                        "recommended_network_dim"),
                    "recommended_weight_range": run.get(
                        "recommended_weight_range"),
                    "prompt_booster": run.get("prompt_booster", ""),
                    "evaluation_plan": run.get("evaluation_plan", ""),
                }
                phase_info["warnings"] += [
                    w for w in run.get("warnings", [])
                    if w not in phase_info["warnings"]]
            else:                                    # pragma: no cover
                cfg = kcb.build_specialized_kohya_config(
                    target, dataset_dir, out_dir,
                    base_checkpoint=base_checkpoint,
                    config_dir=base_out / "configs",
                    command_dir=base_out / "commands", dry_run=self.dry_run)
                result["training_configs"][target] = cfg
                result["training_commands"][target] = cfg.get("command", "")
                phase_info["targets"][target] = {
                    "sample_count": len(rows),
                    "config_path": cfg.get("config_path", ""),
                    "command": cfg.get("command", ""),
                    "evaluation_plan": cfg.get("evaluation_plan"),
                }
            phase_info["warnings"] += [
                w for w in export.get("warnings", [])
                if w not in phase_info["warnings"]]
        result["phases"]["specialized_loras"] = phase_info

    def _phase_merge(self, result, base_out, base_checkpoint):
        """Final phase: plan the MergeForge CyberRealisticPony Personal Base,
        validate any produced checkpoint, and register the profile. Delegates
        to mergeforge_integration; never overwrites the base checkpoint."""
        identity_ckpt = result.get("training_configs", {}).get(
            "core_identity", {}).get("config", {}).get(
                "output_name", "redresser_core_identity")
        specialized = {t: kcb.SPECIALIZED_TARGETS[t]["output_name"]
                       for t in SPECIALIZED_ORDER}
        checkpoints_dir = base_out / "checkpoints"
        try:
            from . import mergeforge_integration as mfi
            phase = mfi.run_personal_base_merge_phase(
                Path(result["project_path"]),
                base_checkpoint=base_checkpoint,
                identity_output_name=identity_ckpt,
                specialized_output_names=specialized,
                checkpoints_dir=checkpoints_dir,
                dry_run=self.dry_run, execute=False)
        except Exception as exc:                     # never crash the caller
            result["phases"]["merge"] = {"status": "error",
                                         "errors": [str(exc)]}
            result["errors"].append(f"merge phase error: {exc}")
            return

        merge_plan = phase["merge_plan"]
        base_reg = phase["profile_registration"]
        # backward-compatible result keys
        result["mergeforge_plan"] = merge_plan
        result["base_profile"] = base_reg
        result["phases"]["merge"] = {
            "status": "completed",
            "merge_plan": merge_plan["output_dir"],
            "merged_checkpoint": phase["merged_checkpoint"],
            "validation_status": phase["validation"]["status"],
            "profile_registered": base_reg.get("registered", False),
            "profile_name": bp.BASE_PROFILE_NAME,
            "warnings": phase.get("warnings", []),
        }


__all__ = ["PhasedTrainingOrchestrator", "DEFAULT_PHASES",
           "SPECIALIZED_ORDER"]
