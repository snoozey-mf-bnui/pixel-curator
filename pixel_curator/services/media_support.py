"""Shared service - image/HEIC support detection + safe thumbnails.

Import-safe: ``pillow-heif`` and even Pillow are optional. HEIC/HEIF files are
*detected* by extension without decoding, so a gallery never crashes on an
iPhone library even when HEIF support isn't installed - those files are marked
unsupported and skipped rather than erroring.
"""
from __future__ import annotations

from pathlib import Path

# extensions we can always *list*; decoding may still need optional libs
BASE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
HEIF_EXTS = (".heic", ".heif")
IMAGE_EXTS = BASE_EXTS + HEIF_EXTS

_HEIF_REGISTERED = False


def heif_support_available() -> dict:
    """Report whether HEIC/HEIF decoding is available. Never imports heavily at
    module load; checks lazily here. Never raises."""
    info = {"pillow_available": False, "pillow_heif_available": False,
            "heif_registered": _HEIF_REGISTERED, "warnings": []}
    try:
        import PIL  # noqa: F401
        info["pillow_available"] = True
    except Exception:
        info["warnings"].append("Pillow not installed - thumbnails unavailable")
    try:
        import pillow_heif  # noqa: F401
        info["pillow_heif_available"] = True
    except Exception:
        info["warnings"].append(
            "pillow-heif not installed - .heic/.heif files will be listed but "
            "not decoded. Install with: pip install pillow-heif")
    return info


def register_heif_opener_if_available() -> dict:
    """Register the HEIF opener with Pillow if pillow-heif is present. Safe to
    call repeatedly. Returns a status dict; never raises."""
    global _HEIF_REGISTERED
    status = heif_support_available()
    if status["pillow_available"] and status["pillow_heif_available"] \
            and not _HEIF_REGISTERED:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
            _HEIF_REGISTERED = True
            status["heif_registered"] = True
        except Exception as exc:
            status["warnings"].append(f"could not register HEIF opener: {exc}")
    return status


def is_supported_image(path) -> bool:
    """True if the extension is a known image type (listing-level support)."""
    try:
        return Path(path).suffix.lower() in IMAGE_EXTS
    except Exception:
        return False


def is_decodable_image(path) -> bool:
    """True if this file can actually be decoded right now (HEIF only if the
    opener is registered)."""
    try:
        ext = Path(path).suffix.lower()
        if ext in BASE_EXTS:
            return True
        if ext in HEIF_EXTS:
            return heif_support_available()["pillow_heif_available"]
        return False
    except Exception:
        return False


def build_thumbnail_safe(image_path, output_path, max_side: int = 512) -> dict:
    """Write a downscaled thumbnail. Non-destructive (source never modified).
    Degrades gracefully: missing Pillow or undecodable HEIF -> a structured
    warning result, never an exception."""
    res = {"ok": False, "thumbnail": None, "warnings": []}
    src = Path(image_path)
    if not src.is_file():
        res["warnings"].append("source not found")
        return res
    if not is_decodable_image(src):
        res["warnings"].append(
            f"cannot decode {src.suffix} (optional support not installed)")
        return res
    try:
        register_heif_opener_if_available()
        from PIL import Image, ImageOps
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((max_side, max_side))
            im.save(out, format="JPEG", quality=88)
        res["ok"] = True
        res["thumbnail"] = str(out)
    except Exception as exc:
        res["warnings"].append(f"thumbnail failed: {exc}")
    return res


__all__ = ["IMAGE_EXTS", "BASE_EXTS", "HEIF_EXTS", "heif_support_available",
           "register_heif_opener_if_available", "is_supported_image",
           "is_decodable_image", "build_thumbnail_safe"]
