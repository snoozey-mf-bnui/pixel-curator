"""Feature 8E/8G/8H - REDRESSER pipeline dashboard status.

Reports the availability of every subsystem in professional, actionable
language for the Pipeline dashboard. Pure status reporting: read-only, no
generation, no network beyond an optional local Forge ping. Never raises.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

from .redresser_settings import RedresserSettings, load_settings

logger = logging.getLogger(__name__)

log = logging.getLogger("redresser.status")

PATCH_MARKER = "REDRESSER safety patch"
KNOWN_BAD_MODEL = "sdxl-inpainting-1.0.safetensors"


def _exists(p: str | Path) -> bool:
    try:
        return Path(p).expanduser().exists()
    except Exception:
        return False


def _module_available(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def reforge_patch_status(networks_path: Path) -> str:
    """applied | not_applied | unknown (file missing/unreadable)."""
    try:
        if not networks_path.is_file():
            return "unknown"
        return ("applied" if PATCH_MARKER in
                networks_path.read_text(encoding="utf-8", errors="ignore")
                else "not_applied")
    except Exception:
        return "unknown"


def corrupt_model_check(stable_diffusion_dir: Path) -> dict:
    """Detect the known-problematic inpainting checkpoint and report it
    without ever deleting it."""
    info = {"present": False, "path": "", "message": ""}
    try:
        candidate = stable_diffusion_dir / KNOWN_BAD_MODEL
        if not candidate.is_file():
            return info
        info["present"] = True
        info["path"] = str(candidate)
        ok = False
        try:
            with open(candidate, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                head = fh.read(2)
            ok = 0 < n < 64 * 1024 * 1024 and head in (b'{"', b"{'")
        except Exception:
            ok = False
        if not ok:
            info["message"] = (
                f"Known model warning detected: {KNOWN_BAD_MODEL}. The file "
                "may be incomplete or incompatible. REDRESSER will avoid "
                "selecting it automatically. Replace or quarantine it "
                "manually if needed.")
    except Exception:                              # pragma: no cover
        pass
    return info


def subsystem_status(settings: RedresserSettings | None = None,
                     output_base: Path | str | None = None,
                     ping_forge: bool = True) -> dict:
    """Assemble the full dashboard status payload."""
    s = settings or load_settings(output_base)
    lora_dir = Path(s.lora_dir).expanduser()
    networks_path = s.networks_path()
    sd_dir = networks_path.parent.parent.parent / "Stable-diffusion" \
        if networks_path.name == "networks.py" else lora_dir.parent \
        / "Stable-diffusion"

    # Forge backend
    forge = {"available": False, "url": s.forge_url}
    if ping_forge:
        try:
            from ..eval.forge_api import ForgeClient
            forge["available"] = ForgeClient(s.forge_url).alive()
        except Exception:
            forge["available"] = False

    # LoRA validator + index
    lora_index = {"ready": False, "files": 0}
    try:
        from ..lora_validator import LoraStackValidator
        v = LoraStackValidator(lora_dir)
        v._ensure_index()
        lora_index = {"ready": True, "files": len(v._by_name)}
    except Exception:
        lora_index = {"ready": False, "files": 0}

    # presets
    preset_counts = {"wardrobe": 0, "motion": 0}
    try:
        from .. import starter_stacks as ss
        preset_counts = {"wardrobe": len(ss.list_wardrobe_presets()),
                         "motion": len(ss.list_motion_presets())}
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass

    # post-processing
    post = {
        "realesrgan_model": _exists(s.realesrgan_model_path),
        "gfpgan_model": _exists(s.gfpgan_model_path),
        "realesrgan_pkg": _module_available("realesrgan"),
        "gfpgan_pkg": _module_available("gfpgan"),
    }

    # ControlNet / IP-Adapter assets (model folders)
    models_root = lora_dir.parent
    controlnet = {"models_present":
                  _exists(models_root / "ControlNet")
                  and any((models_root / "ControlNet").glob("*.safetensors"))
                  if _exists(models_root / "ControlNet") else False}
    ipadapter = {"models_present":
                 _exists(models_root / "ipadapter")
                 and any((models_root / "ipadapter").glob("*"))
                 if _exists(models_root / "ipadapter") else False}

    # AnimateDiff (planning always; backend detection best-effort)
    animatediff = {"planning": True, "backend": False}
    if forge["available"]:
        try:
            from ..eval.forge_api import ForgeClient
            scripts = ForgeClient(s.forge_url)._get("/sdapi/v1/scripts")
            animatediff["backend"] = "animatediff" in json.dumps(
                scripts).lower()
        except Exception:
            animatediff["backend"] = False

    # Feature 10L - production quality subsystem
    try:
        from .production_quality import list_quality_presets
        quality = {"ready": True, "presets": list_quality_presets(),
                   "high_res_fix": True,
                   "adetailer": bool(animatediff.get("backend")) and False
                   or _module_available("adetailer"),
                   "identity_guard": True,
                   "scoring": "planned"}
    except Exception:
        quality = {"ready": False, "presets": [], "high_res_fix": False,
                   "adetailer": False, "identity_guard": False,
                   "scoring": "planned"}

    # Feature 11 - phased training subsystem (DatasetSplitter -> phased plan)
    try:
        from ..curate.dataset_splitter import DatasetSplitter  # noqa: F401
        splitter_ok = True
    except Exception:
        splitter_ok = False
    try:
        from ..train.kohya_config_builder import build_core_identity_kohya_config  # noqa: F401
        recipe_ok = True
    except Exception:
        recipe_ok = False
    phased_training = {
        "available": True,
        "dataset_splitter": splitter_ok,
        "core_identity_recipe": recipe_ok,
        "specialized_lora_planning": True,
        "mergeforge_planning": True,
    }

    status = {
        "forge": forge,
        "lora_validator": {"ready": True},
        "phased_training": phased_training,
        "lora_index": lora_index,
        "presets": preset_counts,
        "post_processing": post,
        "production_quality": quality,
        "controlnet": controlnet,
        "ip_adapter": ipadapter,
        "animatediff": animatediff,
        "dataset_log": {"name": s.dataset_log_name},
        "reforge_patch": reforge_patch_status(networks_path),
        "corrupt_model": corrupt_model_check(sd_dir),
        "paths": s.as_dict(),
        "labels": [],
    }
    status["labels"] = _labels(status)
    return status


def _labels(st: dict) -> list[str]:
    """Professional, human-readable one-line summaries for the dashboard."""
    out = []
    f = st["forge"]
    out.append(f"Forge backend: {'available at ' + f['url'] if f['available'] else 'not reachable at ' + f['url']}")
    out.append("LoRA validator: ready")
    pt = st.get("phased_training", {})
    if pt:
        out.append(
            "Phased Training: " + ("available" if pt.get("available")
                                   else "unavailable")
            + f" (DatasetSplitter {'ready' if pt.get('dataset_splitter') else 'unavailable'},"
            + f" Core Identity recipe {'ready' if pt.get('core_identity_recipe') else 'unavailable'})")
    out.append(f"LoRA index: {st['lora_index']['files']} file(s)"
               if st["lora_index"]["ready"] else "LoRA index: unavailable")
    out.append(f"Wardrobe presets: {st['presets']['wardrobe']} | "
               f"motion presets: {st['presets']['motion']}")
    p = st["post_processing"]
    re_ok = "available" if (p["realesrgan_model"] and p["realesrgan_pkg"]) \
        else ("model present" if p["realesrgan_model"] else "unavailable")
    gf_ok = "available" if (p["gfpgan_model"] and p["gfpgan_pkg"]) \
        else ("model present" if p["gfpgan_model"] else "unavailable")
    out.append(f"Post-processing: RealESRGAN {re_ok}, GFPGAN {gf_ok}")
    q = st.get("production_quality", {})
    out.append("Production Quality: " + ("ready" if q.get("ready")
                                         else "unavailable"))
    if q.get("presets"):
        out.append("Quality presets: " + ", ".join(q["presets"]))
    out.append("High-Res Fix: " + ("payload support available"
                                   if q.get("high_res_fix") else "unavailable"))
    out.append("ADetailer: " + ("available" if q.get("adetailer")
                                else "unavailable, optional"))
    out.append(f"Quality scoring: {q.get('scoring', 'planned')}")
    out.append("Identity guard: " + ("ready" if q.get("identity_guard")
                                     else "unavailable"))
    out.append(f"ControlNet: {'models present' if st['controlnet']['models_present'] else 'not found'}")
    out.append(f"IP-Adapter / FaceID: {'models present' if st['ip_adapter']['models_present'] else 'not found'}")
    ad = st["animatediff"]
    out.append("AnimateDiff: " + ("backend available" if ad["backend"]
                                  else "payload planning available"))
    out.append(f"Dataset log: {st['dataset_log']['name']} (JSONL)")
    out.append(f"reForge LoRA safety patch: {st['reforge_patch']}")
    if st["corrupt_model"]["present"] and st["corrupt_model"]["message"]:
        out.append("Model warning: " + st["corrupt_model"]["message"])
    return out
