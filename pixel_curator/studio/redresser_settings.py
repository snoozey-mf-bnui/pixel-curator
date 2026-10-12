"""Feature 8F - REDRESSER local settings (key paths).

A tiny, local-first JSON settings store. No remote settings, no network.
Safe to read/write; falls back to in-memory defaults if the file cannot be
written.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger("redresser.settings")

_SETTINGS_ROOT = Path(__file__).resolve().parent.parent


def _default_model_path(filename: str, subdir: str, legacy: str) -> str:
    """Prefer the repo-local ``models/<subdir>/<filename>`` copy; fall back
    to the legacy external-SSD path when the local file is absent."""
    local = _SETTINGS_ROOT / "models" / subdir / filename
    return str(local) if local.exists() else legacy


DEFAULT_FORGE_URL = "http://127.0.0.1:7860"
DEFAULT_LORA_DIR = (
    "~/FORGE_REPO/stable-diffusion-webui-reForge/models/Lora")
DEFAULT_IDENTITY_IMAGE = ""  # set your own identity anchor image path locally
DEFAULT_REALESRGAN = _default_model_path(
    "RealESRGAN_x4plus.pth", "realesrgan",
    "/Volumes/CURATOR_SSD/TAKEOUTS/RealESRGAN_x4plus.pth")
DEFAULT_GFPGAN = _default_model_path(
    "GFPGANv1.4.pth", "gfpgan",
    "/Volumes/CURATOR_SSD/TAKEOUTS/GFPGANv1.4.pth")
DEFAULT_REFORGE_REPO = "~/FORGE_REPO/stable-diffusion-webui-reForge"
DEFAULT_CONFIG_NAME = ".redresser_settings.json"


@dataclass
class RedresserSettings:
    forge_url: str = DEFAULT_FORGE_URL
    lora_dir: str = DEFAULT_LORA_DIR
    identity_image_path: str = DEFAULT_IDENTITY_IMAGE
    realesrgan_model_path: str = DEFAULT_REALESRGAN
    gfpgan_model_path: str = DEFAULT_GFPGAN
    output_dir: str = ""
    dataset_log_name: str = "personal_catalog_dataset"
    reforge_repo: str = DEFAULT_REFORGE_REPO
    networks_py_path: str = ""
    # Feature 10M - production-quality defaults (conservative, local-safe)
    enable_production_quality_default: bool = False
    default_quality_preset: str = "catalog_pro"
    default_consistency_threshold: float = 0.65
    enable_high_res_fix_default: bool = True
    enable_adetailer_default: bool = False
    default_sampler: str = "DPM++ 2M SDE Karras"
    default_steps: int = 28
    default_cfg_scale: float = 6.0
    default_hr_scale: float = 1.5
    default_hr_denoise: float = 0.35
    enable_final_sharpen_default: bool = False
    enable_final_denoise_default: bool = False

    def networks_path(self) -> Path:
        if self.networks_py_path:
            return Path(self.networks_py_path).expanduser()
        return (Path(self.reforge_repo).expanduser()
                / "extensions-builtin" / "Lora" / "networks.py")

    def as_dict(self) -> dict:
        return asdict(self)


def _config_path(output_base: Path | str | None) -> Path:
    base = Path(output_base).expanduser() if output_base else Path.home()
    return base / DEFAULT_CONFIG_NAME


def load_settings(output_base: Path | str | None = None) -> RedresserSettings:
    path = _config_path(output_base)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = RedresserSettings().as_dict()
        return RedresserSettings(**{k: v for k, v in data.items()
                                    if k in known})
    except FileNotFoundError:
        return RedresserSettings()
    except Exception as exc:                       # corrupt -> defaults
        log.warning("Could not read settings (%s); using defaults.", exc)
        return RedresserSettings()


def save_settings(settings: RedresserSettings,
                  output_base: Path | str | None = None) -> dict:
    path = _config_path(output_base)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings.as_dict(), indent=2),
                        encoding="utf-8")
        return {"ok": True, "path": str(path)}
    except Exception as exc:                       # pragma: no cover
        log.warning("Could not write settings: %s", exc)
        return {"ok": False, "warning": f"settings not saved: {exc}"}
