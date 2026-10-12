"""Central generation-mode resolver: the single source of truth for deciding
txt2img vs img2img from a user-supplied init image path.

Rules (strict, no silent fallbacks):
    None / "" / "   "             -> txt2img
    valid .png/.jpg/.jpeg/.webp   -> img2img
    non-empty but nonexistent     -> error  (NOT a txt2img fallback)
    directory path                -> error
    unsupported extension         -> error

The caller must check ``.error`` first: a non-None error means refuse to
generate and surface the message, never quietly run txt2img instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

GenerationMode = Literal["txt2img", "img2img"]

#: image extensions accepted as an img2img init image
SUPPORTED_INIT_EXTS = (".png", ".jpg", ".jpeg", ".webp")


@dataclass
class InitImageResolution:
    """Result of resolving an init image path into a generation mode."""
    mode: GenerationMode
    init_image_path: Optional[Path]
    error: Optional[str] = None

    @property
    def is_img2img(self) -> bool:
        return self.mode == "img2img" and self.error is None


def resolve_generation_mode(
        init_image_path: "str | Path | None") -> InitImageResolution:
    """Decide txt2img vs img2img from an init image path. Pure / no I/O beyond
    a stat() existence + suffix check. Never raises."""
    # None / empty / whitespace -> txt2img (intentional, not an error)
    if init_image_path is None:
        return InitImageResolution("txt2img", None)
    raw = init_image_path if isinstance(init_image_path, str) \
        else str(init_image_path)
    if not raw.strip():
        return InitImageResolution("txt2img", None)

    p = Path(raw).expanduser()
    if p.is_dir():
        return InitImageResolution(
            "txt2img", None,
            error=f"Init image path is a directory, not an image file: {p}")
    if not p.exists():
        return InitImageResolution(
            "txt2img", None,
            error=f"Init image path does not exist: {p}")
    if p.suffix.lower() not in SUPPORTED_INIT_EXTS:
        return InitImageResolution(
            "txt2img", None,
            error=(f"Unsupported init image type '{p.suffix or '(none)'}'; "
                   f"use one of {', '.join(SUPPORTED_INIT_EXTS)}: {p}"))
    return InitImageResolution("img2img", p)
