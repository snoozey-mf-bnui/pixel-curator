"""Shared service - base-model architecture awareness (SDXL / unknown).

Pure-python, import-safe. Both UI surfaces use this to label models and
adapt training options. This studio targets SDXL/Pony exclusively
(cyberrealisticPony base); anything else is reported as unknown and
kept plan-only so it is never trained against by accident.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SDXL = "sdxl"
UNKNOWN = "unknown"


@dataclass
class ModelArchitecture:
    key: str
    display_name: str
    supported_training_types: list = field(default_factory=list)
    supported_devices: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    status: str = "available"   # available | partial | planned | unavailable

    def to_dict(self) -> dict:
        return {"key": self.key, "display_name": self.display_name,
                "supported_training_types": list(self.supported_training_types),
                "supported_devices": list(self.supported_devices),
                "notes": list(self.notes), "status": self.status}


# alias -> canonical key (pony/illustrious/etc. are SDXL-family)
_ALIASES = {
    "sdxl": SDXL, "sd_xl": SDXL, "sdxl1.0": SDXL, "pony": SDXL,
    "pony_sdxl": SDXL, "ponysdxl": SDXL, "ponyxl": SDXL,
    "cyberrealisticpony": SDXL, "cyberrealistic_pony": SDXL,
    "illustrious": SDXL, "illustrious_sdxl": SDXL, "noobai": SDXL,
    "animagine": SDXL, "sd15": "sd15", "sd1.5": "sd15",
}


def normalize_architecture(value: str) -> str:
    """Map any label/profile name to a canonical key. Unknown -> ``unknown``."""
    if not value:
        return UNKNOWN
    v = str(value).strip().lower().replace(" ", "_")
    if v in (SDXL, UNKNOWN, "sd15"):
        return v
    if v in _ALIASES:
        return _ALIASES[v]
    # substring heuristics on the normalized token
    if "pony" in v or "sdxl" in v or "illustrious" in v or "cyberrealistic" in v:
        return SDXL
    if "sd15" in v or "sd1.5" in v or "sd1_5" in v:
        return "sd15"
    return UNKNOWN


def detect_model_architecture(path_or_name: str) -> str:
    """Best-effort architecture from a checkpoint path or model name. Reads the
    filename/path tokens only - never opens or downloads the model."""
    if not path_or_name:
        return UNKNOWN
    from pathlib import Path
    name = Path(str(path_or_name)).name.lower()
    return normalize_architecture(name if name else str(path_or_name))


def known_architectures() -> list:
    return [
        ModelArchitecture(
            key=SDXL, display_name="SDXL (incl. Pony / CyberRealisticPony)",
            supported_training_types=["plan_only", "lora", "dreambooth"],
            supported_devices=["auto", "cuda", "mps", "cpu"],
            notes=["Pony / CyberRealisticPony are SDXL-family.",
                   "Local LoRA via kohya sd-scripts; full DreamBooth best on "
                   "CUDA/A100."],
            status="available"),
        ModelArchitecture(
            key=UNKNOWN, display_name="Unknown / Auto-detect",
            supported_training_types=["plan_only"],
            supported_devices=["auto", "cpu"],
            notes=["Architecture could not be determined; plan-only."],
            status="partial"),
    ]


def architecture_capabilities(architecture: str) -> ModelArchitecture:
    key = normalize_architecture(architecture)
    for a in known_architectures():
        if a.key == key:
            return a
    # sd15 or other recognized-but-not-listed -> partial unknown-ish
    return ModelArchitecture(
        key=key if key in (SDXL, UNKNOWN) else UNKNOWN,
        display_name=key.upper(), supported_training_types=["plan_only"],
        supported_devices=["auto", "cpu"], status="partial",
        notes=[f"limited support for '{architecture}'"])


__all__ = ["SDXL", "UNKNOWN", "ModelArchitecture",
           "normalize_architecture", "detect_model_architecture",
           "known_architectures", "architecture_capabilities"]
