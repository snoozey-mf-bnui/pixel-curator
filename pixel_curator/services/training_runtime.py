"""Shared service - architecture-aware training launch (SDXL/Pony only).

Torch is imported lazily inside device checks only. Plan-only is always safe;
execution is gated behind explicit confirmation, a supported architecture +
training type, a real entrypoint, and an available device.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import model_architecture as arch
from .training_jobs import TrainingJob, TrainingJobStore
from .log_streaming import ProcessLogManager, new_job_id

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICES = ("auto", "mps", "cuda", "cpu")


def detect_training_devices(prefer_mps: bool = True) -> dict:
    """Report training devices without requiring torch. Never raises."""
    info = {"torch_available": False, "cuda_available": False,
            "mps_available": False, "recommended": "cpu", "warnings": []}
    try:
        import torch
        info["torch_available"] = True
        try:
            info["cuda_available"] = bool(torch.cuda.is_available())
        except Exception:
            pass
        try:
            info["mps_available"] = bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available())
        except Exception:
            pass
    except Exception:
        info["warnings"].append("torch not installed - plan-only mode")
        return info
    if info["cuda_available"]:
        info["recommended"] = "cuda"
    elif prefer_mps and info["mps_available"]:
        info["recommended"] = "mps"
    else:
        info["recommended"] = "cpu"
        info["warnings"].append("no CUDA/MPS device - local training limited")
    return info


def resolve_device(requested: str, devices: dict | None = None) -> str:
    d = devices or detect_training_devices()
    r = (requested or "auto").lower()
    return d.get("recommended", "cpu") if r == "auto" else r


@dataclass
class TrainingLaunchRequest:
    training_type: str = "plan_only"
    model_architecture: str = "sdxl"
    device: str = "auto"
    dataset_path: str = ""
    package_path: str = ""
    output_path: str = ""
    base_model_path: str = ""
    base_model_profile: str = "CyberRealisticPony Personal Base"
    target: str = "core_identity"
    sd_scripts_dir: str = ""
    dry_run: bool = True
    execute: bool = False
    confirmed: bool = False
    params: dict = field(default_factory=dict)


@dataclass
class TrainingLaunchPlan:
    ok: bool
    status: str   # ready | dry_run | unavailable | blocked | error
    command: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    config_path: str | None = None
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    job: TrainingJob | None = None

    def to_dict(self) -> dict:
        return {"ok": self.ok, "status": self.status,
                "command": list(self.command), "env": dict(self.env),
                "config_path": self.config_path,
                "warnings": list(self.warnings), "errors": list(self.errors),
                "job": self.job.to_dict() if self.job else None}


_TRAIN_SCRIPT = {
    ("sdxl", "lora"): "sdxl_train_network.py",
    ("sdxl", "dreambooth"): "sdxl_train.py",
}


def _dataset_config(req: TrainingLaunchRequest) -> str:
    for base in (req.package_path, req.dataset_path):
        if base:
            p = Path(base).expanduser() / "dataset_config.toml"
            if p.is_file():
                return str(p)
    base = req.package_path or req.dataset_path
    return str(Path(base).expanduser() / "dataset_config.toml") if base \
        else "dataset_config.toml"


def validate_training_launch_request(request: TrainingLaunchRequest
                                     ) -> TrainingLaunchPlan:
    """Validate gates only (no command build). Plan-only/dry-run always ok."""
    tt = (request.training_type or "plan_only").lower()
    if request.dry_run or tt == "plan_only":
        return TrainingLaunchPlan(True, "dry_run")
    blockers = []
    if not request.execute:
        blockers.append("'Execute training' is not checked")
    if not request.confirmed:
        blockers.append("'Confirm launch' is not checked")
    has_data = (request.dataset_path and
                Path(request.dataset_path).expanduser().exists()) or \
               (request.package_path and
                Path(request.package_path).expanduser().exists())
    if not has_data:
        blockers.append("dataset/package path does not exist")
    if not request.output_path:
        blockers.append("output path is required")
    archk = arch.normalize_architecture(request.model_architecture)
    if archk != arch.SDXL:
        blockers.append(f"architecture '{archk}' is not trainable here - "
                        "this studio trains SDXL/Pony only")
    d = detect_training_devices()
    dev = resolve_device(request.device, d)
    if dev == "cuda" and not d.get("cuda_available"):
        blockers.append("CUDA requested but not available")
    if dev == "mps" and not d.get("mps_available"):
        blockers.append("MPS requested but not available")
    if dev == "cpu":
        blockers.append("CPU cannot execute SDXL training (use plan-only)")
    if blockers:
        return TrainingLaunchPlan(False, "blocked", errors=blockers)
    return TrainingLaunchPlan(True, "ready")


def build_training_launch_plan(request: TrainingLaunchRequest
                               ) -> TrainingLaunchPlan:
    """Build the command for a request. Plan-only -> safe ``train plan``.
    Execute -> the real kohya command when assemblable; otherwise
    ``unavailable`` with reasons. Never fakes training."""
    tt = (request.training_type or "plan_only").lower()
    archk = arch.normalize_architecture(request.model_architecture)
    dev = resolve_device(request.device)

    if tt == "plan_only" or request.dry_run:
        mode = "dreambooth" if "dreambooth" in tt else "lora"
        out = request.output_path or str(REPO_ROOT / "TRAINING_PLANS"
                                         / request.target)
        cmd = [sys.executable, "-m", "pixel_curator", "train", "plan",
               "--dataset", request.dataset_path or "<dataset>",
               "--output", out, "--target", request.target, "--mode", mode,
               "--base-model-profile", request.base_model_profile, "--apply"]
        if request.package_path:
            cmd += ["--package", request.package_path]
        return TrainingLaunchPlan(
            True, "dry_run", command=cmd,
            warnings=["plan/config only (no model is trained)"])

    # execute path
    gate = validate_training_launch_request(request)
    if not gate.ok:
        return TrainingLaunchPlan(False, "blocked", errors=gate.errors)

    script = _TRAIN_SCRIPT.get((archk, tt))
    sd = Path(request.sd_scripts_dir).expanduser() if request.sd_scripts_dir \
        else None
    errs = []
    if not script:
        errs.append(f"no trainer mapping for {archk}/{tt}")
    if not sd or not sd.is_dir():
        errs.append("sd_scripts_dir not set/found")
    elif script and not (sd / script).is_file():
        errs.append(f"{script} not found in sd_scripts_dir")
    if not request.base_model_path:
        errs.append("base_model_path not set")
    if errs:
        return TrainingLaunchPlan(False, "unavailable", errors=errs)

    p = dict(request.params or {})
    res = int(p.get("resolution", 1024))
    cmd = ["accelerate", "launch", "--num_cpu_threads_per_process=2",
           str(sd / script),
           f"--pretrained_model_name_or_path={request.base_model_path}",
           f"--dataset_config={_dataset_config(request)}",
           f"--output_dir={request.output_path}",
           f"--output_name=redresser_{request.target}",
           "--save_model_as=safetensors",
           f"--resolution={res},{res}",
           f"--train_batch_size={int(p.get('batch_size', 1))}",
           f"--max_train_epochs={int(p.get('epochs', 6))}",
           f"--learning_rate={p.get('learning_rate', 1e-4)}",
           f"--mixed_precision={p.get('mixed_precision', 'bf16')}",
           f"--seed={int(p.get('seed', 42))}",
           "--gradient_checkpointing", "--cache_latents"]
    if tt == "lora":
        cmd += ["--network_module=networks.lora",
                f"--network_dim={int(p.get('network_dim', 32))}",
                f"--network_alpha={int(p.get('network_alpha', 16))}"]
    env = {"PYTORCH_ENABLE_MPS_FALLBACK": "1"} if dev == "mps" else {}
    return TrainingLaunchPlan(True, "ready", command=cmd, env=env)


def start_training_job(request: TrainingLaunchRequest,
                       job_store: TrainingJobStore,
                       log_manager: ProcessLogManager) -> TrainingLaunchPlan:
    """Build, gate, and (if executing) launch a training job, recording it in
    the shared job store. Plan-only/dry-run record a ``planned`` job and run the
    safe plan command; execute launches the real command only when allowed."""
    plan = build_training_launch_plan(request)
    archk = arch.normalize_architecture(request.model_architecture)
    job_id = new_job_id("train")
    out = Path(request.output_path).expanduser() if request.output_path \
        else REPO_ROOT
    log_path = out / "gui_logs" / f"{job_id}.log"

    job = TrainingJob(
        job_id=job_id, name=f"{archk}/{request.training_type}",
        training_type=request.training_type, model_architecture=archk,
        device=resolve_device(request.device), dataset_path=request.dataset_path,
        package_path=request.package_path, output_path=request.output_path,
        base_model_path=request.base_model_path,
        base_model_profile=request.base_model_profile,
        command=plan.command, dry_run=request.dry_run,
        confirmed=request.confirmed, log_path=str(log_path),
        warnings=plan.warnings, errors=plan.errors)

    if not plan.ok:
        job.status = "unavailable" if plan.status == "unavailable" else "failed"
        job_store.save_job(job)
        plan.job = job
        return plan

    if plan.status == "dry_run":
        # run the safe plan command in the background, record as planned
        job.status = "planned"
        res = log_manager.start_process(
            plan.command, cwd=str(REPO_ROOT), log_path=log_path, job_id=job_id)
        job.metadata["manager_status"] = res.get("status")
        job_store.save_job(job)
        plan.job = job
        return plan

    # ready -> execute
    env = {**os.environ, **plan.env}
    cwd = str(Path(request.sd_scripts_dir).expanduser()
              if request.sd_scripts_dir else REPO_ROOT)
    res = log_manager.start_process(plan.command, cwd=cwd, log_path=log_path,
                                    env=env, job_id=job_id)
    job.status = "running"
    job.started_at = res.get("started_at")
    job_store.save_job(job)
    plan.job = job
    return plan


__all__ = ["DEVICES", "TrainingLaunchRequest", "TrainingLaunchPlan",
           "detect_training_devices", "resolve_device",
           "validate_training_launch_request", "build_training_launch_plan",
           "start_training_job"]
