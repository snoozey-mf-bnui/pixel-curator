"""Phase 6 local generation backend: MPS-optimized diffusers test pipeline.

Engineering patterns adopted from the Dressing Studio analysis:
  - pipeline cache keyed by (checkpoint, vae, lora-stack) tuple; rebuilt only
    on config change; explicit unload() with gc + torch.mps.empty_cache()
  - 3-tier LoRA loading fallback: standard diffusers -> direct file ->
    UNet-only (kohya key filtering); stacked adapters with per-LoRA weights,
    NEGATIVE weights supported (set_adapters handles them)
  - MPS hygiene: fp16, attention slicing, VAE tiling, CPU-seeded generators
  - feathered masks (dilate + gaussian) for inpaint-based tests

Install: pip install 'pixel-curator[eval]'  (or run inside your Forge env -
diffusers/torch are already there).
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

_PIPE_CACHE: dict = {"key": None, "pipe": None}


def diffusers_available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        from PIL import Image  # noqa: F401
        return True, "ok"
    except Exception as exc:
        return False, f"missing [eval] extras: {exc}"


def _device():
    from ..runtime import get_device_name
    return get_device_name("auto")        # cuda -> mps -> cpu


def _mps_flush() -> None:
    """Free accelerator cache (cuda OR mps) between heavy jobs."""
    from ..runtime import clear_accelerator_cache
    gc.collect()
    clear_accelerator_cache()


class TestPipeline:
    """SDXL txt2img / img2img / inpaint wrapper for LoRA evaluation."""

    def __init__(self, checkpoint: str, vae: str = "", device: str = ""):
        import torch  # noqa: F401
        from diffusers import AutoPipelineForText2Image
        from ..runtime import (apply_cuda_settings, get_recommended_dtype)

        self.device = device or _device()
        self.dtype = get_recommended_dtype(self.device)
        if self.device == "cuda":
            apply_cuda_settings(True)
        ckpt = Path(checkpoint).expanduser()
        logging.info("Loading checkpoint %s on %s", ckpt.name, self.device)

        if ckpt.is_file():
            from diffusers import StableDiffusionXLPipeline
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                str(ckpt), torch_dtype=self.dtype
            )
        else:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                str(ckpt), torch_dtype=self.dtype
            )
        if vae:
            from diffusers import AutoencoderKL
            self.pipe.vae = AutoencoderKL.from_single_file(
                str(Path(vae).expanduser()), torch_dtype=self.dtype
            )
        self.pipe = self.pipe.to(self.device)
        # MPS hygiene (Dressing Studio pattern)
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            self.pipe.enable_vae_tiling()
        except Exception:
            pass
        self.adapters: list[tuple[str, float]] = []

    # ----- LoRA stack (3-tier fallback) -----

    def load_lora(self, lora_path: str, strength: float = 1.0) -> bool:
        lp = Path(lora_path).expanduser()
        name = lp.stem
        for ch in (".", " ", "-", "%20"):
            name = name.replace(ch, "_")
        if any(n == name for n, _ in self.adapters):
            name = f"{name}_{len(self.adapters)}"

        # 1) standard diffusers
        try:
            self.pipe.load_lora_weights(str(lp.parent), weight_name=lp.name,
                                        adapter_name=name)
            self._register(name, strength)
            return True
        except Exception as exc:
            logging.info("LoRA standard load failed (%s); trying direct", exc)
        # 2) direct file
        try:
            self.pipe.load_lora_weights(str(lp), adapter_name=name)
            self._register(name, strength)
            return True
        except Exception as exc:
            logging.info("LoRA direct load failed (%s); trying UNet-only", exc)
        # 3) UNet-only (kohya key filtering)
        try:
            from safetensors.torch import load_file
            state = load_file(str(lp))
            unet_state = {k: v for k, v in state.items() if "text_encoder" not in k}
            self.pipe.load_lora_weights(unet_state, adapter_name=name)
            self._register(name, strength)
            logging.info("LoRA '%s' loaded UNet-only", name)
            return True
        except Exception as exc:
            logging.error("LoRA load failed entirely for %s: %s", lp.name, exc)
            return False

    def _register(self, name: str, strength: float) -> None:
        self.adapters.append((name, float(strength)))
        names = [n for n, _ in self.adapters]
        weights = [w for _, w in self.adapters]   # negative weights pass through
        self.pipe.set_adapters(names, adapter_weights=weights)

    def unload_loras(self) -> None:
        if self.adapters:
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass
            self.adapters = []

    # ----- generation -----

    def txt2img(self, prompt: str, negative: str = "", steps: int = 28,
                cfg: float = 6.0, width: int = 1024, height: int = 1024,
                seed: int = 42):
        import torch
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        out = self.pipe(
            prompt=prompt, negative_prompt=negative,
            num_inference_steps=int(steps), guidance_scale=float(cfg),
            width=int(width), height=int(height), generator=gen,
        )
        return out.images[0]

    def img2img(self, init_image, prompt: str, negative: str = "",
                strength: float = 0.6, steps: int = 28, cfg: float = 6.0,
                seed: int = 42):
        import torch
        from diffusers import AutoPipelineForImage2Image
        pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        out = pipe(
            prompt=prompt, negative_prompt=negative, image=init_image,
            strength=float(strength), num_inference_steps=int(steps),
            guidance_scale=float(cfg), generator=gen,
        )
        return out.images[0]

    def inpaint(self, init_image, mask_image, prompt: str, negative: str = "",
                strength: float = 0.85, steps: int = 30, cfg: float = 6.5,
                seed: int = 42, feather: float = 8.0):
        import torch
        from diffusers import AutoPipelineForInpainting
        pipe = AutoPipelineForInpainting.from_pipe(self.pipe)
        mask = feather_mask(mask_image, sigma=feather)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        out = pipe(
            prompt=prompt, negative_prompt=negative, image=init_image,
            mask_image=mask, strength=float(strength),
            num_inference_steps=int(steps), guidance_scale=float(cfg),
            generator=gen,
        )
        return out.images[0]

    def close(self) -> None:
        self.unload_loras()
        self.pipe = None
        _mps_flush()


def feather_mask(mask_image, sigma: float = 8.0, dilate: int = 12):
    """Dilate then gaussian-feather a mask for smooth inpaint blending."""
    from PIL import ImageFilter
    m = mask_image.convert("L")
    if dilate > 0:
        m = m.filter(ImageFilter.MaxFilter(dilate | 1))  # odd kernel
    if sigma > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=sigma))
    return m


# -----------------------------
# Cache (Dressing Studio pattern)
# -----------------------------

def get_pipeline(checkpoint: str, vae: str,
                 loras: list[tuple[str, float]]) -> TestPipeline:
    key = (checkpoint, vae, tuple(loras))
    if _PIPE_CACHE["key"] == key and _PIPE_CACHE["pipe"] is not None:
        return _PIPE_CACHE["pipe"]
    unload_pipeline()
    tp = TestPipeline(checkpoint, vae)
    for path, weight in loras:
        tp.load_lora(path, weight)
    _PIPE_CACHE.update(key=key, pipe=tp)
    return tp


def unload_pipeline() -> str:
    if _PIPE_CACHE["pipe"] is not None:
        try:
            _PIPE_CACHE["pipe"].close()
        except Exception:
            pass
    _PIPE_CACHE.update(key=None, pipe=None)
    _mps_flush()
    return "Pipeline unloaded - memory freed."
