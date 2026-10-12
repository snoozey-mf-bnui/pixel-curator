"""Shared service - LoRA/model merge planning with architecture safety gates.

Wraps the repo's existing ``merge`` command rather than re-implementing a risky
safetensors merge. ``build_merge_plan`` is always safe (dry-run by default) and
warns on architecture mismatch (this studio merges SDXL-family artifacts only);
``execute_merge`` refuses to run without explicit confirmation and a
destructive-overwrite guard.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import model_architecture as arch

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class MergeRequest:
    base_model_path: str
    lora_paths: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    output_path: str = ""
    architecture: str = "unknown"
    dry_run: bool = True
    confirmed: bool = False
    allow_architecture_mismatch: bool = False


@dataclass
class MergePlan:
    ok: bool
    status: str   # dry_run | ready | blocked | unavailable | error
    command: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    base_model_architecture: str = "unknown"
    lora_architectures: list = field(default_factory=list)
    compatible: bool = True
    blocked_reason: str = ""
    recommended_action: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "status": self.status,
                "command": list(self.command), "warnings": list(self.warnings),
                "errors": list(self.errors),
                "base_model_architecture": self.base_model_architecture,
                "lora_architectures": list(self.lora_architectures),
                "compatible": self.compatible,
                "blocked_reason": self.blocked_reason,
                "recommended_action": self.recommended_action}


def _arch_set(req: MergeRequest) -> set:
    archs = set()
    if req.base_model_path:
        archs.add(arch.detect_model_architecture(req.base_model_path))
    for lp in req.lora_paths:
        archs.add(arch.detect_model_architecture(lp))
    declared = arch.normalize_architecture(req.architecture)
    if declared != arch.UNKNOWN:
        archs.add(declared)
    return {a for a in archs if a != arch.UNKNOWN}


def build_merge_plan(request: MergeRequest) -> MergePlan:
    """Validate + build the merge command. Always safe to call. Warns on
    architecture mismatch; errors block execution unless overridden."""
    warnings, errors = [], []
    if not request.base_model_path:
        errors.append("base_model_path is required")
    if not request.lora_paths:
        errors.append("at least one LoRA/model path is required")
    if request.weights and len(request.weights) != len(request.lora_paths):
        warnings.append("weights count != lora count; defaulting missing to 1.0")

    base_arch = arch.detect_model_architecture(request.base_model_path) \
        if request.base_model_path else "unknown"
    lora_archs = [arch.detect_model_architecture(lp)
                  for lp in request.lora_paths]
    archs = _arch_set(request)
    compatible = True
    blocked_reason = ""
    recommended = ""
    foreign = archs - {arch.SDXL}
    if arch.SDXL in archs and foreign:
        compatible = False
        msg = ("architecture mismatch: mixing SDXL with "
               f"{', '.join(sorted(foreign))} artifacts is not compatible")
        if request.allow_architecture_mismatch:
            warnings.append(msg + " (override enabled - planning only; execute "
                            "remains blocked)")
            recommended = "keep different architectures in separate merges"
        else:
            errors.append(msg + " (enable advanced override to plan)")
            blocked_reason = msg
            recommended = "merge SDXL artifacts with SDXL artifacts only"
    if arch.UNKNOWN in {base_arch, *lora_archs}:
        warnings.append("one or more artifacts have unknown architecture - "
                        "verify compatibility before executing")

    def _fill(plan: MergePlan) -> MergePlan:
        plan.base_model_architecture = base_arch
        plan.lora_architectures = lora_archs
        plan.compatible = compatible
        plan.blocked_reason = blocked_reason
        plan.recommended_action = recommended
        return plan

    if errors:
        return _fill(MergePlan(False, "blocked", warnings=warnings,
                               errors=errors))

    weights = list(request.weights) + \
        [1.0] * (len(request.lora_paths) - len(request.weights))
    spec = ",".join(f"{Path(p).name}:{w}"
                    for p, w in zip(request.lora_paths, weights))
    cmd = [sys.executable, "-m", "pixel_curator", "merge",
           "--base", request.base_model_path,
           "--loras", spec,
           "--output", request.output_path or "<output>"]
    status = "dry_run" if request.dry_run else "ready"
    return _fill(MergePlan(True, status, command=cmd, warnings=warnings))


def execute_merge(request: MergeRequest, log_manager=None) -> MergePlan:
    """Execute a merge - only when ``confirmed`` and not dry-run, with a
    destructive-overwrite guard. Returns a blocked plan otherwise."""
    plan = build_merge_plan(request)
    if not plan.ok:
        return plan
    if request.dry_run:
        return MergePlan(True, "dry_run", command=plan.command,
                         warnings=plan.warnings +
                         ["dry-run: nothing executed"])
    if not request.confirmed:
        return MergePlan(False, "blocked", command=plan.command,
                         errors=["merge execution requires explicit "
                                 "confirmation"])
    if not request.output_path:
        return MergePlan(False, "blocked",
                         errors=["output_path required for execution"])
    if Path(request.output_path).expanduser().exists():
        return MergePlan(False, "blocked",
                         errors=["output already exists - refusing to "
                                 "overwrite without an explicit new path"])
    if log_manager is None:
        return MergePlan(True, "ready", command=plan.command,
                         warnings=plan.warnings +
                         ["no log manager provided; command not launched"])
    from .log_streaming import new_job_id
    jid = new_job_id("merge")
    log_path = Path(request.output_path).expanduser().parent / \
        f"gui_logs/{jid}.log"
    log_manager.start_process(plan.command, cwd=str(REPO_ROOT),
                              log_path=log_path, job_id=jid)
    return MergePlan(True, "ready", command=plan.command,
                     warnings=plan.warnings + [f"merge launched (job {jid})"])


__all__ = ["MergeRequest", "MergePlan", "build_merge_plan", "execute_merge"]
