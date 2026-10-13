"""Feature 6A - optional, conservative post-processing.

A standalone, file-based refinement pass for generated catalog images:
RealESRGAN x4 for garment / fabric detail, and GFPGAN at a conservative
weight for face-consistency preservation. This complements the in-app
reForge extras path; use it for offline or batch refinement directly from
the ``.pth`` model files.

Design goals: optional, non-destructive (writes a ``_upscaled.png`` sibling,
never overwrites the original), and fully graceful - if a model file or an
optional dependency is missing, it logs a professional warning and returns
the original image path unchanged. It never raises for ordinary
missing-model / missing-dependency situations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("redresser.post_processing")

_PACKAGE_ROOT = Path(__file__).resolve().parent


def _default_model(filename: str, subdir: str, legacy: str) -> Path:
    """Prefer the repo-local ``models/<subdir>/<filename>`` copy; fall back
    to the legacy external-SSD path when the local file is absent."""
    local = _PACKAGE_ROOT / "models" / subdir / filename
    return local if local.exists() else Path(legacy)


DEFAULT_REALESRGAN_MODEL = _default_model(
    "RealESRGAN_x4plus.pth", "realesrgan",
    "/Volumes/CURATOR_SSD/TAKEOUTS/RealESRGAN_x4plus.pth")
DEFAULT_GFPGAN_MODEL = _default_model(
    "GFPGANv1.4.pth", "gfpgan",
    "/Volumes/CURATOR_SSD/TAKEOUTS/GFPGANv1.4.pth")

GFPGAN_WEIGHT_MIN = 0.0
GFPGAN_WEIGHT_MAX = 0.7
GFPGAN_WEIGHT_DEFAULT = 0.3
OUTPUT_SUFFIX = "_upscaled"


def clamp_gfpgan_weight(weight: float) -> tuple[float, str | None]:
    """Clamp the GFPGAN weight into the safe conservative range."""
    try:
        w = float(weight)
    except (TypeError, ValueError):
        return GFPGAN_WEIGHT_DEFAULT, (
            f"GFPGAN weight invalid; defaulted to {GFPGAN_WEIGHT_DEFAULT:.2f}.")
    if w < GFPGAN_WEIGHT_MIN or w > GFPGAN_WEIGHT_MAX:
        c = max(GFPGAN_WEIGHT_MIN, min(GFPGAN_WEIGHT_MAX, w))
        return c, f"GFPGAN weight clamped to safe range: {c:.2f}."
    return w, None


@dataclass
class PostProcessResult:
    original_path: str
    post_processed_path: str          # == original_path when nothing applied
    realesrgan_applied: bool = False
    gfpgan_applied: bool = False
    gfpgan_weight: float = GFPGAN_WEIGHT_DEFAULT
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "original_path": self.original_path,
            "post_processed_path": self.post_processed_path,
            "post_processing": {
                "realesrgan": self.realesrgan_applied,
                "gfpgan": self.gfpgan_applied,
                "gfpgan_weight": self.gfpgan_weight,
                "warnings": list(self.warnings),
            },
        }


class PostProcessor:
    """Lazy-initialized RealESRGAN + GFPGAN refiner. Models are constructed
    on first use and cached on the instance."""

    def __init__(self,
                 realesrgan_model_path: Path | str = DEFAULT_REALESRGAN_MODEL,
                 gfpgan_model_path: Path | str = DEFAULT_GFPGAN_MODEL,
                 output_suffix: str = OUTPUT_SUFFIX) -> None:
        self.realesrgan_model_path = Path(realesrgan_model_path).expanduser()
        self.gfpgan_model_path = Path(gfpgan_model_path).expanduser()
        self.output_suffix = output_suffix
        self._upsampler = None
        self._restorer = None
        self._device: str | None = None

    # -- device --------------------------------------------------------------
    def _select_device(self) -> str:
        if self._device:
            return self._device
        dev = "cpu"
        try:
            import torch
            if torch.backends.mps.is_available():
                dev = "mps"
            elif torch.cuda.is_available():
                dev = "cuda"
        except Exception:                          # torch missing/odd build
            dev = "cpu"
        self._device = dev
        log.info("Post-processing device: %s", dev)
        return dev

    # -- IO ------------------------------------------------------------------
    @staticmethod
    def _read_rgb(path: Path):
        import numpy as np
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))

    @staticmethod
    def _write_rgb(path: Path, rgb) -> None:
        import numpy as np
        from PIL import Image
        Image.fromarray(np.clip(rgb, 0, 255).astype("uint8")).save(
            path, "PNG")

    def _output_path(self, image_path: Path) -> Path:
        return image_path.with_name(
            f"{image_path.stem}{self.output_suffix}.png")

    # -- model construction (best-effort) ------------------------------------
    def _ensure_realesrgan(self, warnings: list[str]) -> bool:
        if self._upsampler is not None:
            return True
        if not self.realesrgan_model_path.is_file():
            try:  # first run: auto-download into models/realesrgan/
                from .model_assets import ensure_model
                self.realesrgan_model_path = ensure_model(
                    "realesrgan-x4plus", progress=log.info)
            except Exception as exc:
                warnings.append("RealESRGAN model not found and "
                                "auto-download failed; skipping "
                                "garment-detail upscale.")
                log.warning("%s (%s)", warnings[-1], exc)
                return False
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except Exception:
            warnings.append("Post-processing dependencies unavailable; "
                            "returning original image.")
            log.warning(warnings[-1])
            return False
        try:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
            self._upsampler = RealESRGANer(
                scale=4, model_path=str(self.realesrgan_model_path),
                model=model, tile=0, tile_pad=10, pre_pad=0, half=False,
                device=self._select_device())
            return True
        except Exception as exc:                   # pragma: no cover
            warnings.append("RealESRGAN initialization failed; skipping "
                            "garment-detail upscale.")
            log.warning("%s (%s)", warnings[-1], exc)
            return False

    def _ensure_gfpgan(self, warnings: list[str]) -> bool:
        if self._restorer is not None:
            return True
        if not self.gfpgan_model_path.is_file():
            try:  # first run: auto-download into models/gfpgan/
                from .model_assets import ensure_model
                self.gfpgan_model_path = ensure_model(
                    "gfpgan-v1.4", progress=log.info)
            except Exception as exc:
                warnings.append("GFPGAN model not found and auto-download "
                                "failed; skipping conservative face "
                                "refinement.")
                log.warning("%s (%s)", warnings[-1], exc)
                return False
        try:
            from gfpgan import GFPGANer
        except Exception:
            warnings.append("Post-processing dependencies unavailable; "
                            "returning original image.")
            log.warning(warnings[-1])
            return False
        try:
            self._restorer = GFPGANer(
                model_path=str(self.gfpgan_model_path), upscale=1,
                arch="clean", channel_multiplier=2, bg_upsampler=None,
                device=self._select_device())
            return True
        except Exception as exc:                   # pragma: no cover
            warnings.append("GFPGAN initialization failed; skipping "
                            "conservative face refinement.")
            log.warning("%s (%s)", warnings[-1], exc)
            return False

    # -- public --------------------------------------------------------------
    def process(self, image_path: Path | str, *, use_realesrgan: bool = True,
                use_gfpgan: bool = True,
                gfpgan_weight: float = GFPGAN_WEIGHT_DEFAULT
                ) -> PostProcessResult:
        """Refine one image. Returns a :class:`PostProcessResult`; on any
        skip/failure the post-processed path equals the original path."""
        src = Path(image_path).expanduser()
        weight, clamp_warn = clamp_gfpgan_weight(gfpgan_weight)
        res = PostProcessResult(original_path=str(src),
                                post_processed_path=str(src),
                                gfpgan_weight=weight)
        if clamp_warn:
            res.warnings.append(clamp_warn)
        if not src.is_file():
            res.warnings.append(f"Image not found; post-processing skipped: "
                                f"{src}")
            log.warning(res.warnings[-1])
            return res
        if not (use_realesrgan or use_gfpgan):
            res.warnings.append("No post-processing requested; returning "
                                "original image.")
            return res

        try:
            import numpy as np  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception:
            res.warnings.append("Post-processing dependencies unavailable; "
                                "returning original image.")
            log.warning(res.warnings[-1])
            return res

        try:
            rgb = self._read_rgb(src)
        except Exception as exc:                   # pragma: no cover
            res.warnings.append(f"Could not read image; post-processing "
                                f"skipped: {exc}")
            log.warning(res.warnings[-1])
            return res

        worked = False
        # 1) RealESRGAN garment/fabric detail upscale (BGR for the library)
        if use_realesrgan and self._ensure_realesrgan(res.warnings):
            try:
                bgr = rgb[:, :, ::-1]
                out, _ = self._upsampler.enhance(bgr, outscale=4)
                rgb = out[:, :, ::-1]
                res.realesrgan_applied = True
                worked = True
            except Exception as exc:               # pragma: no cover
                res.warnings.append("RealESRGAN refinement failed; using "
                                    "prior image.")
                log.warning("%s (%s)", res.warnings[-1], exc)
        # 2) GFPGAN conservative face refinement
        if use_gfpgan and self._ensure_gfpgan(res.warnings):
            try:
                bgr = rgb[:, :, ::-1]
                _, _, restored = self._restorer.enhance(
                    bgr, has_aligned=False, only_center_face=False,
                    paste_back=True, weight=weight)
                rgb = restored[:, :, ::-1]
                res.gfpgan_applied = True
                worked = True
            except Exception as exc:               # pragma: no cover
                res.warnings.append("GFPGAN refinement failed; using prior "
                                    "image.")
                log.warning("%s (%s)", res.warnings[-1], exc)

        if not worked:
            return res                              # original path preserved
        out_path = self._output_path(src)
        try:
            self._write_rgb(out_path, rgb)
            res.post_processed_path = str(out_path)
            log.info("input:  %s", src)
            log.info("output: %s", out_path)
        except Exception as exc:                   # pragma: no cover
            res.warnings.append(f"Could not write refined image; original "
                                f"preserved: {exc}")
            log.warning(res.warnings[-1])
        return res


def apply_upscalers(image_path: Path | str, use_realesrgan: bool = True,
                    use_gfpgan: bool = True,
                    gfpgan_weight: float = GFPGAN_WEIGHT_DEFAULT,
                    realesrgan_model_path: Path | str = DEFAULT_REALESRGAN_MODEL,
                    gfpgan_model_path: Path | str = DEFAULT_GFPGAN_MODEL
                    ) -> Path:
    """Convenience wrapper: refine an image and return the resulting path
    (the ``_upscaled.png`` sibling, or the original path if nothing was
    applied)."""
    pp = PostProcessor(realesrgan_model_path, gfpgan_model_path)
    res = pp.process(image_path, use_realesrgan=use_realesrgan,
                     use_gfpgan=use_gfpgan, gfpgan_weight=gfpgan_weight)
    return Path(res.post_processed_path)
