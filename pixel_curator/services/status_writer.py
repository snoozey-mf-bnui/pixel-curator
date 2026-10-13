"""Shared service - Phase 3 status snapshots for legacy-UI parity.

Writes small JSON snapshots under ``<output_base>/ui_status/`` (or the repo
``projects/ui_status/`` fallback) so the legacy HTTP dashboard can display the
same Phase 3 state the Gradio GUI shows - one source of truth, two surfaces.
Also provides the dict payloads the legacy ``/api/phase3/*`` routes return.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import media_support, training_runtime
from . import model_architecture as arch
from .log_streaming import DEFAULT_MANAGER, discover_recent_logs
from .training_jobs import TrainingJobStore, default_jobs_path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def status_dir(output_base: str | None = None) -> Path:
    if output_base:
        try:
            return Path(output_base).expanduser() / "ui_status"
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass
    return REPO_ROOT / "projects" / "ui_status"


# ---------------------------------------------------------------------------
# Payload builders (also used directly by the legacy /api/phase3 routes)
# ---------------------------------------------------------------------------
def device_status() -> dict:
    return training_runtime.detect_training_devices()


def architectures_status() -> dict:
    return {"architectures": [a.to_dict() for a in arch.known_architectures()]}


def heic_status() -> dict:
    return media_support.heif_support_available()


def jobs_status(output_base: str | None = None) -> dict:
    store = TrainingJobStore(str(default_jobs_path(output_base)))
    return {"jobs": [j.to_dict() for j in store.list_jobs()],
            "managed": DEFAULT_MANAGER.list_jobs(),
            "active": DEFAULT_MANAGER.active_count()}


def logs_status(output_base: str | None = None) -> dict:
    root = Path(output_base).expanduser() / "gui_logs" if output_base \
        else REPO_ROOT / "projects" / "gui_logs"
    return {"recent_logs": discover_recent_logs(root)}


def explorer_summary_status(output_base: str | None = None) -> dict:
    from . import dataset_explorer as dx
    return dx.get_stage_summary(output_base or str(REPO_ROOT))


def explorer_stage_sample(output_base: str | None = None, stage: str | None = None,
                          limit: int = 100) -> dict:
    from . import dataset_explorer as dx
    flt = dx.ExplorerFilter(stage_key=stage, limit=limit)
    recs = dx.load_image_records(output_base or str(REPO_ROOT), flt)
    return {"stage": stage, "count": len(recs),
            "records": [r.to_dict() for r in recs]}


def explorer_compare_status(output_base: str | None = None,
                            stage_a: str = "raw_import",
                            stage_b: str = "post_smart_curation") -> dict:
    from . import dataset_explorer as dx
    return dx.compare_stages(output_base or str(REPO_ROOT), stage_a, stage_b).to_dict()


def visual_focus_status(output_base: str | None = None) -> dict:
    """Summary of the latest Visual Focus Classifier results, if present."""
    from . import visual_focus_service as vfs
    base = Path(output_base).expanduser() if output_base else REPO_ROOT
    for cand in (base / vfs.RESULTS_NAME, base / "visual_focus" / vfs.RESULTS_NAME):
        if cand.exists():
            return vfs.summarize_visual_focus_results(cand)
    return {"total": 0, "note": "no visual_focus_results.jsonl found"}


def write_visual_focus_status(output_base: str | None = None) -> dict:
    """Legacy-UI parity: write visual_focus_summary.json under ui_status/."""
    d = status_dir(output_base)
    written = {}
    try:
        d.mkdir(parents=True, exist_ok=True)
        payload = _safe(visual_focus_status, output_base)
        (d / "visual_focus_summary.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")
        written["visual_focus_summary.json"] = str(d / "visual_focus_summary.json")
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    return written


def multi_recipe_package_status(output_base: str | None = None) -> dict:
    """Summary of the latest multi-recipe package plan, if present."""
    base = Path(output_base).expanduser() if output_base else REPO_ROOT
    for cand in (base / "ui_status" / "multi_recipe_package_summary.json",
                 base / "training_packages" / "multi_recipe_package_summary.json"):
        if cand.exists():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except Exception:
                return {"error": "could not read summary"}
    return {"total_packages": 0, "note": "multi-recipe packages not generated yet"}


def face_consistency_status(output_base: str | None = None) -> dict:
    """Summary of the latest Face Consistency Studio run, if present."""
    base = Path(output_base).expanduser() if output_base else REPO_ROOT
    cand = base / "ui_status" / "face_consistency_summary.json"
    if cand.exists():
        try:
            return json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            return {"error": "could not read face_consistency summary"}
    try:
        from . import face_consistency_studio as fcs
        return {"service": fcs.SERVICE_NAME, "processed": 0,
                "note": "no face-fix run yet",
                "backends": fcs.detect_face_consistency_backends()}
    except Exception:
        return {"processed": 0, "note": "face consistency not run yet"}


def write_face_consistency_status(output_base: str | None = None) -> dict:
    """Legacy-UI parity: write face_consistency_summary.json under ui_status/."""
    d = status_dir(output_base)
    written = {}
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / "face_consistency_summary.json").write_text(
            json.dumps(_safe(face_consistency_status, output_base), indent=2,
                       default=str), encoding="utf-8")
        written["face_consistency_summary.json"] = str(
            d / "face_consistency_summary.json")
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    return written


def specialized_recipe_status(output_base: str | None = None) -> dict:
    """Summary of the latest specialized recipe plan, if present."""
    from . import specialized_recipe_builder as rb
    base = Path(output_base).expanduser() if output_base else REPO_ROOT
    for cand in (base / rb.OUTPUT_FILES["summary"],
                 base / "specialized_recipes" / rb.OUTPUT_FILES["summary"]):
        if cand.exists():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except Exception:
                return {"error": "could not read summary"}
    return {"total_images": 0, "note": "specialized recipe plan not generated yet"}


def write_specialized_recipe_status(output_base: str | None = None) -> dict:
    """Legacy-UI parity: write specialized_recipe_summary.json under ui_status/."""
    d = status_dir(output_base)
    written = {}
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / "specialized_recipe_summary.json").write_text(
            json.dumps(_safe(specialized_recipe_status, output_base), indent=2,
                       default=str), encoding="utf-8")
        written["specialized_recipe_summary.json"] = str(
            d / "specialized_recipe_summary.json")
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    return written


def routing_status(output_base: str | None = None) -> dict:
    """Summary of visual-focus routing classifications, if present."""
    from . import visual_focus_classifier as vc
    base = Path(output_base).expanduser() if output_base else REPO_ROOT
    for cand in (base / vc.RESULTS_NAME, base / "visual_focus" / vc.RESULTS_NAME):
        if cand.exists():
            return vc.summarize_visual_focus_classifications(cand)
    return {"total": 0, "note": "visual focus classification not yet run"}


def write_routing_status(output_base: str | None = None) -> dict:
    """Legacy-UI parity: write visual_focus_routing_summary.json under ui_status/."""
    d = status_dir(output_base)
    written = {}
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / "visual_focus_routing_summary.json").write_text(
            json.dumps(_safe(routing_status, output_base), indent=2, default=str),
            encoding="utf-8")
        written["visual_focus_routing_summary.json"] = str(
            d / "visual_focus_routing_summary.json")
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    return written




def write_explorer_status(output_base: str | None = None) -> dict:
    """Write the Dataset Pipeline Explorer snapshots the legacy UI can read
    (Option C parity). Returns {name: path}. Never raises."""
    d = status_dir(output_base)
    written = {}
    payloads = {
        "dataset_explorer_summary.json": _safe(explorer_summary_status, output_base),
        "dataset_explorer_stage_sample.json": _safe(
            explorer_stage_sample, output_base, "post_smart_curation"),
        "dataset_explorer_compare.json": _safe(
            explorer_compare_status, output_base),
    }
    try:
        d.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            try:
                (d / name).write_text(json.dumps(payload, indent=2, default=str),
                                      encoding="utf-8")
                written[name] = str(d / name)
            except Exception:
                logger.debug("suppressed exception", exc_info=True)
                continue
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    return written


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as exc:
        return {"error": str(exc)}


def write_all(output_base: str | None = None) -> dict:
    """Write every Phase 3 status snapshot. Returns {name: path}. Never raises."""
    d = status_dir(output_base)
    written = {}
    payloads = {
        "device_status.json": device_status(),
        "architectures.json": architectures_status(),
        "heic_status.json": heic_status(),
        "training_jobs.json": jobs_status(output_base),
        "live_logs.json": logs_status(output_base),
    }
    try:
        d.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            try:
                (d / name).write_text(json.dumps(payload, indent=2,
                                                 default=str), encoding="utf-8")
                written[name] = str(d / name)
            except Exception:
                logger.debug("suppressed exception", exc_info=True)
                continue
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    return written


# Map of legacy /api/phase3/<key> -> builder (output_base from query string).
def route_payload(key: str, output_base: str | None = None) -> dict | None:
    builders = {
        "devices": lambda: device_status(),
        "architectures": lambda: architectures_status(),
        "heic": lambda: heic_status(),
        "jobs": lambda: jobs_status(output_base),
        "logs": lambda: logs_status(output_base),
        "explorer_summary": lambda: explorer_summary_status(output_base),
        "explorer_stage": lambda: explorer_stage_sample(output_base),
        "explorer_compare": lambda: explorer_compare_status(output_base),
        "visual_focus": lambda: visual_focus_status(output_base),
        "routing": lambda: routing_status(output_base),
        "specialized_recipes": lambda: specialized_recipe_status(output_base),
        "multi_recipe_packages": lambda: multi_recipe_package_status(output_base),
        "face_consistency": lambda: face_consistency_status(output_base),
    }
    fn = builders.get(key)
    return fn() if fn else None


__all__ = ["status_dir", "write_all", "route_payload", "device_status",
           "architectures_status", "heic_status", "jobs_status",
           "logs_status", "write_explorer_status", "explorer_summary_status",
           "explorer_stage_sample", "explorer_compare_status",
           "visual_focus_status", "write_visual_focus_status",
           "routing_status", "write_routing_status",
           "specialized_recipe_status", "write_specialized_recipe_status",
           "multi_recipe_package_status", "face_consistency_status",
           "write_face_consistency_status"]
