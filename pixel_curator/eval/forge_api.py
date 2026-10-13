"""Forge/A1111/reForge API client (stdlib only). Start reForge with --api.

LoRAs are injected the webui way: <lora:name:weight> in the prompt
(negative weights work for subtractive effects). Your reForge install at
forge_root serves this on http://127.0.0.1:7860 by default.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request


class ForgeClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7860", timeout: int = 600):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, route: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base}{route}",
            json.dumps(payload).encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def alive(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base}/sdapi/v1/options")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False

    def _get(self, route: str) -> list:
        try:
            req = urllib.request.Request(f"{self.base}{route}")
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except Exception:
            return []

    def checkpoints(self) -> list[str]:
        """Checkpoint titles Forge can switch to (sd_model_checkpoint)."""
        return [m.get("title") or m.get("model_name", "")
                for m in self._get("/sdapi/v1/sd-models")]

    def loras(self) -> list[str]:
        """LoRA names available to the <lora:name:weight> prompt syntax."""
        return [l.get("name", "") for l in self._get("/sdapi/v1/loras")]

    def upscalers(self) -> list[str]:
        """Names of upscalers Forge knows (for the extras API)."""
        return [u.get("name", "") for u in self._get("/sdapi/v1/upscalers")]

    def controlnet_modules(self) -> list[str]:
        """Preprocessor module names reForge's ControlNet extension knows."""
        d = self._get("/controlnet/module_list")
        return list(d.get("module_list", [])) if isinstance(d, dict) else []

    def controlnet_models(self) -> list[str]:
        """ControlNet/IP-Adapter model names reForge's extension can load."""
        d = self._get("/controlnet/model_list")
        return list(d.get("model_list", [])) if isinstance(d, dict) else []

    def resolve_realesrgan(self) -> str:
        """Pick the RealESRGAN x4plus upscaler name as Forge labels it.
        Falls back to the conventional 'R-ESRGAN 4x+' if probing fails."""
        names = self.upscalers()
        for want in ("R-ESRGAN 4x+", "RealESRGAN_x4plus", "RealESRGAN 4x+",
                     "R-ESRGAN 4x+ Anime6B"):
            if want in names:
                return want
        for n in names:                       # any RealESRGAN-ish 4x entry
            low = n.lower()
            if "esrgan" in low and ("4x" in low or "x4" in low):
                return n
        return "R-ESRGAN 4x+"

    def set_options(self, opts: dict) -> bool:
        """Set global Forge settings (e.g. ESRGAN tile size). Best-effort."""
        try:
            self._post("/sdapi/v1/options", opts)
            return True
        except Exception:
            return False

    def extra_single_image(self, png_bytes: bytes, *, upscale: bool = True,
                           scale: float = 4.0, upscaler: str = "",
                           gfpgan_visibility: float = 0.0,
                           tile: int | None = 0,
                           upscale_first: bool = True) -> bytes:
        """Post-process one image through Forge's built-in extras pipeline:
        RealESRGAN upscaling and/or GFPGAN face restoration. Mirrors the
        webui 'Extras' tab. Returns PNG bytes (falls back to the input
        image if the call fails)."""
        # 'zero tiling' = whole-image pass (no tiling). Global ESRGAN setting.
        if tile is not None:
            self.set_options({"ESRGAN_tile": int(tile),
                              "ESRGAN_tile_overlap": 0})
        if upscale and not upscaler:
            upscaler = self.resolve_realesrgan()
        payload = {
            "resize_mode": 0,                       # 0 = scale by multiplier
            "show_extras_results": True,
            "image": base64.b64encode(png_bytes).decode(),
            "gfpgan_visibility": float(gfpgan_visibility),
            "codeformer_visibility": 0.0,
            "codeformer_weight": 0.0,
            "upscaling_resize": float(scale) if upscale else 1.0,
            "upscaling_crop": False,
            "upscaler_1": upscaler if upscale else "None",
            "upscaler_2": "None",
            "extras_upscaler_2_visibility": 0.0,
            "upscale_first": bool(upscale_first),
        }
        try:
            data = self._post("/sdapi/v1/extra-single-image", payload)
            return base64.b64decode(data["image"])
        except Exception:
            return png_bytes                        # never lose the render

    def postprocess(self, png_bytes: bytes, *, upscale: bool = False,
                    restore_faces: bool = False, scale: float = 4.0,
                    gfpgan_weight: float = 0.3) -> bytes:
        """Convenience wrapper for the Test-tab checkboxes. Runs the extras
        pass only if at least one option is enabled."""
        if not (upscale or restore_faces):
            return png_bytes
        return self.extra_single_image(
            png_bytes, upscale=upscale, scale=scale,
            gfpgan_visibility=(gfpgan_weight if restore_faces else 0.0),
            tile=0, upscale_first=True)

    @staticmethod
    def lora_prompt(prompt: str, loras: list[tuple[str, float]]) -> str:
        """Append <lora:stem:weight> tags (webui convention; negatives ok)."""
        from pathlib import Path
        tags = "".join(
            f" <lora:{Path(p).stem}:{w:g}>" for p, w in loras
        )
        return prompt + tags

    # ControlNet model files installed under models/ControlNet (SDXL).
    # Maps a friendly control type -> (preprocessor module, model stem).
    CONTROLNET_MODELS = {
        "openpose": ("openpose_full", "controlnet-sdxl-openpose-xinsir"),
        "depth": ("depth_midas", "controlnet-sdxl-depth-xinsir"),
        "canny": ("canny", "controlnet-sdxl-canny-xinsir"),
        "softedge": ("softedge_pidinet",
                     "controlnet-sd-xl-1.0-softedge-dexined"),
    }
    # IP-Adapter / FaceID is exposed through the ControlNet unit interface in
    # reForge. type -> (preprocessor module, model stem under models/ipadapter)
    IPADAPTER_MODELS = {
        "faceid": ("InsightFace+CLIP (IPAdapter)",
                   "ip-adapter-faceid-plusv2_sdxl"),
        "plus_face": ("ip-adapter_clip_sdxl",
                      "ip-adapter-plus-face_sdxl_vit-h"),
        "reference": ("ip-adapter_clip_sdxl",
                      "ip-adapter-plus_sdxl_vit-h"),
    }

    @staticmethod
    def controlnet_unit(image_b64: str, control_type: str = "openpose",
                        weight: float = 0.8, guidance_start: float = 0.0,
                        guidance_end: float = 0.8,
                        control_mode: str = "Balanced",
                        model: str = "", module: str = "") -> dict:
        """Build one ControlNet unit for alwayson_scripts. control_type is a
        key of CONTROLNET_MODELS (openpose/depth/canny/softedge); model and
        module override the defaults when needed."""
        mod, mdl = ForgeClient.CONTROLNET_MODELS.get(
            control_type, ("none", ""))
        return {
            "enabled": True,
            "image": image_b64,
            "module": module or mod,
            "model": model or mdl,
            "weight": float(weight),
            "resize_mode": "Crop and Resize",
            "processor_res": 1024,
            "guidance_start": float(guidance_start),
            "guidance_end": float(guidance_end),
            "control_mode": control_mode,        # Balanced | "My prompt"|"CN"
            "pixel_perfect": True,
        }

    @staticmethod
    def ipadapter_unit(image_b64: str, ip_type: str = "faceid",
                       weight: float = 0.75, guidance_start: float = 0.0,
                       guidance_end: float = 0.9) -> dict:
        """Build an IP-Adapter / FaceID identity-guidance unit (ControlNet
        interface). ip_type: faceid | plus_face | reference."""
        mod, mdl = ForgeClient.IPADAPTER_MODELS.get(
            ip_type, ForgeClient.IPADAPTER_MODELS["faceid"])
        return {
            "enabled": True,
            "image": image_b64,
            "module": mod,
            "model": mdl,
            "weight": float(weight),
            "resize_mode": "Crop and Resize",
            "guidance_start": float(guidance_start),
            "guidance_end": float(guidance_end),
            "control_mode": "Balanced",
            "pixel_perfect": True,
        }

    @staticmethod
    def build_txt2img_payload(prompt: str, negative: str = "", steps: int = 28,
                              cfg: float = 6.0, width: int = 1024,
                              height: int = 1024, seed: int = 42,
                              sampler: str = "DPM++ 2M Karras",
                              loras: list[tuple[str, float]] | None = None,
                              hires: bool = False, clip_skip: int = 0,
                              checkpoint: str = "",
                              controlnet_units: list[dict] | None = None
                              ) -> dict:
        payload = {
            "prompt": ForgeClient.lora_prompt(prompt, loras or []),
            "negative_prompt": negative,
            "steps": int(steps), "cfg_scale": float(cfg),
            "width": int(width), "height": int(height),
            "seed": int(seed), "sampler_name": sampler,
        }
        # override_settings lets the API pin the exact checkpoint + clip
        # skip used in your working Forge session, so in-app output matches
        overrides: dict = {}
        if int(clip_skip) > 1:
            overrides["CLIP_stop_at_last_layers"] = int(clip_skip)
        if checkpoint:
            overrides["sd_model_checkpoint"] = checkpoint
        if overrides:
            payload["override_settings"] = overrides
            payload["override_settings_restore_afterwards"] = True
        if hires:
            payload.update(enable_hr=True, hr_scale=1.5,
                           hr_upscaler="Latent", denoising_strength=0.45)
        # ControlNet + IP-Adapter units (pose / depth / canny / identity)
        units = [u for u in (controlnet_units or []) if u]
        if units:
            payload["alwayson_scripts"] = {"controlnet": {"args": units}}
        return payload

    def txt2img(self, **kw) -> bytes:
        data = self._post("/sdapi/v1/txt2img", self.build_txt2img_payload(**kw))
        return base64.b64decode(data["images"][0])

    # Region presets for masked inpainting. box = (x0,y0,x1,y1) as
    # fractions of the image; denoise/mask_blur are conservative defaults.
    REGION_PRESETS = {
        "torso":      {"box": (0.18, 0.24, 0.82, 0.62),
                       "denoise": 0.45, "mask_blur": 16},
        "garment":    {"box": (0.14, 0.24, 0.86, 0.72),
                       "denoise": 0.45, "mask_blur": 16},
        "lower_body": {"box": (0.20, 0.55, 0.80, 0.97),
                       "denoise": 0.45, "mask_blur": 16},
        "full_body":  {"box": (0.05, 0.05, 0.95, 0.98),
                       "denoise": 0.50, "mask_blur": 20},
        "face":       {"box": (0.34, 0.08, 0.66, 0.40),
                       "denoise": 0.25, "mask_blur": 8},
    }

    def inpaint(self, init_png: bytes, mask_png: bytes, prompt: str,
                negative: str = "", denoise: float = 0.45,
                mask_blur: int = 16, steps: int = 28, cfg: float = 6.0,
                seed: int = 42, sampler: str = "DPM++ 2M Karras",
                loras: list[tuple[str, float]] | None = None,
                controlnet_units: list[dict] | None = None,
                clip_skip: int = 0, checkpoint: str = "",
                inpaint_full_res: bool = True) -> bytes:
        """Region-aware inpainting: regenerate only the masked area
        (white = edit) of init_png. mask_blur feathers the boundary;
        inpaint_full_res = 'Only masked' for sharp localized detail.
        Honors the LoRA stack + ControlNet units."""
        payload = {
            "init_images": [base64.b64encode(init_png).decode()],
            "mask": base64.b64encode(mask_png).decode(),
            "prompt": self.lora_prompt(prompt, loras or []),
            "negative_prompt": negative,
            "denoising_strength": float(denoise),
            "mask_blur": int(mask_blur),
            "inpainting_fill": 1,            # 1 = original (keep context)
            "inpaint_full_res": bool(inpaint_full_res),
            "inpaint_full_res_padding": 32,
            "inpainting_mask_invert": 0,
            "steps": int(steps), "cfg_scale": float(cfg),
            "seed": int(seed), "sampler_name": sampler,
        }
        overrides: dict = {}
        if int(clip_skip) > 1:
            overrides["CLIP_stop_at_last_layers"] = int(clip_skip)
        if checkpoint:
            overrides["sd_model_checkpoint"] = checkpoint
        if overrides:
            payload["override_settings"] = overrides
            payload["override_settings_restore_afterwards"] = True
        units = [u for u in (controlnet_units or []) if u]
        if units:
            payload["alwayson_scripts"] = {"controlnet": {"args": units}}
        data = self._post("/sdapi/v1/img2img", payload)
        return base64.b64decode(data["images"][0])

    def img2img(self, init_png: bytes, prompt: str, negative: str = "",
                strength: float = 0.6, steps: int = 28, cfg: float = 6.0,
                seed: int = 42, sampler: str = "DPM++ 2M Karras",
                loras: list[tuple[str, float]] | None = None,
                resize_mode: int = 0, width: int | None = None,
                height: int | None = None, clip_skip: int = 0,
                checkpoint: str = "") -> bytes:
        """Image-to-image: regenerate `init_png` under `strength` denoising,
        honoring the LoRA stack. clip_skip/checkpoint pin the exact Forge
        session (parity with txt2img/inpaint). width/height are omitted when
        None so output keeps the init image's resolution."""
        payload = {
            "init_images": [base64.b64encode(init_png).decode()],
            "prompt": self.lora_prompt(prompt, loras or []),
            "negative_prompt": negative,
            "denoising_strength": float(strength),
            "resize_mode": int(resize_mode),
            "steps": int(steps), "cfg_scale": float(cfg),
            "seed": int(seed), "sampler_name": sampler,
        }
        if width:
            payload["width"] = int(width)
        if height:
            payload["height"] = int(height)
        overrides: dict = {}
        if int(clip_skip) > 1:
            overrides["CLIP_stop_at_last_layers"] = int(clip_skip)
        if checkpoint:
            overrides["sd_model_checkpoint"] = checkpoint
        if overrides:
            payload["override_settings"] = overrides
            payload["override_settings_restore_afterwards"] = True
        data = self._post("/sdapi/v1/img2img", payload)
        return base64.b64decode(data["images"][0])


MOTION_TYPES = ("catalog_turntable", "garment_detail_reveal",
                "soft_pose_transition", "full_body_wardrobe_variation")


def build_animatediff_payload(prompt: str, negative_prompt: str = "",
                              init_image_path=None,
                              controlnet_units: list[dict] | None = None,
                              ip_adapter_units: list[dict] | None = None,
                              num_frames: int = 8, fps: int = 12,
                              motion_type: str = "catalog_turntable",
                              seed: int | None = None, steps: int = 26,
                              cfg: float = 6.0, width: int = 896,
                              height: int = 1152,
                              sampler: str = "DPM++ 2M Karras",
                              clip_skip: int = 2, checkpoint: str = "",
                              motion_module: str = "", **kwargs) -> dict:
    """Build an AnimateDiff txt2img/img2img-style payload for a wardrobe
    motion sequence. AnimateDiff and ControlNet are attached through
    ``alwayson_scripts``. When the AnimateDiff backend is not installed this
    still returns a complete payload PLAN (with warnings under ``_warnings``)
    rather than failing - the caller strips ``_warnings`` before POSTing."""
    warnings: list[str] = []
    nf = max(4, min(int(num_frames), 32))
    if motion_type not in MOTION_TYPES:
        warnings.append(f"Unknown motion type '{motion_type}'; using "
                        "'catalog_turntable'.")
        motion_type = "catalog_turntable"

    payload: dict = {
        "prompt": prompt, "negative_prompt": negative_prompt,
        "steps": int(steps), "cfg_scale": float(cfg),
        "width": int(width), "height": int(height),
        "seed": int(seed if seed is not None else 42),
        "sampler_name": sampler,
    }
    overrides: dict = {}
    if int(clip_skip) > 1:
        overrides["CLIP_stop_at_last_layers"] = int(clip_skip)
    if checkpoint:
        overrides["sd_model_checkpoint"] = checkpoint
    if overrides:
        payload["override_settings"] = overrides
        payload["override_settings_restore_afterwards"] = True

    alwayson: dict = {}
    units = [u for u in (controlnet_units or []) + (ip_adapter_units or [])
             if u]
    if units:
        alwayson["controlnet"] = {"args": units}
    # sd-webui-animatediff schema (conservative, identity-preserving)
    alwayson["AnimateDiff"] = {"args": [{
        "model": motion_module or "mm_sd15_v3.safetensors",
        "enable": True, "video_length": nf, "fps": int(fps),
        "loop_number": 0, "closed_loop": "R+P",
        "batch_size": min(nf, 16), "stride": 1, "overlap": -1,
        "format": ["PNG"], "interp": "Off", "motion_type": motion_type,
    }]}
    if not motion_module:
        warnings.append("AnimateDiff motion module is a conservative "
                        "default; install/select a motion module to "
                        "execute the sequence.")
    payload["alwayson_scripts"] = alwayson
    if init_image_path:
        payload["init_image_path"] = str(init_image_path)   # planning hint
    payload["_warnings"] = warnings
    return payload


def _gif_command(seq_pattern: str, output_path: str, fps: int = 12,
                 width: int = 0) -> list[str]:
    """Construct the ffmpeg command for a professional GIF (palette pass for
    clean colors). Exposed for testing without invoking ffmpeg."""
    scale = (f"scale={int(width)}:-1:flags=lanczos,"
             if width else "scale=trunc(iw):trunc(ih):flags=lanczos,")
    vf = (scale + "split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
          "[s1][p]paletteuse=dither=bayer")
    return ["ffmpeg", "-y", "-framerate", str(int(fps)), "-i", seq_pattern,
            "-vf", vf, "-loop", "0", output_path]


def export_sequence_gif(frame_paths, output_path, fps: int = 12):
    """Export an ordered list of frames to a professional GIF preview via
    ffmpeg. Returns (Path|None, warnings). Skips cleanly (warning, no crash)
    when ffmpeg is missing or no frames exist. Non-destructive."""
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path
    warnings: list[str] = []
    frames = [Path(p) for p in (frame_paths or []) if Path(p).is_file()]
    out = Path(output_path).expanduser()
    if not frames:
        warnings.append("No frames available; GIF export skipped.")
        return None, warnings
    if not shutil.which("ffmpeg"):
        warnings.append("ffmpeg not found; professional GIF preview "
                        "skipped.")
        return None, warnings
    # stage frames under sequential names so ffmpeg reads a stable order
    staging = Path(tempfile.mkdtemp(prefix="redresser_gif_"))
    for i, fp in enumerate(frames):
        try:
            (staging / f"{i:05d}{fp.suffix.lower()}").write_bytes(
                fp.read_bytes())
        except OSError:
            continue
    suffix = frames[0].suffix.lower() or ".png"
    pattern = str(staging / f"%05d{suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = _gif_command(pattern, str(out), fps)
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    except Exception as exc:                       # pragma: no cover
        warnings.append(f"GIF export did not complete: {exc}")
        return None, warnings
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if not out.is_file():
        warnings.append("GIF export produced no output; skipped.")
        return None, warnings
    return out, warnings


def region_mask_png(init_png: bytes, region: str) -> bytes:
    """Generate a white-on-black rectangular mask for a named region
    (key of ForgeClient.REGION_PRESETS), sized to the init image. White =
    the area to inpaint. Requires Pillow (present in the Forge venv)."""
    import io

    from PIL import Image, ImageDraw
    preset = ForgeClient.REGION_PRESETS.get(region)
    if not preset:
        raise ValueError(f"unknown region '{region}'; choose from "
                         f"{list(ForgeClient.REGION_PRESETS)}")
    im = Image.open(io.BytesIO(init_png))
    w, h = im.size
    x0, y0, x1, y1 = preset["box"]
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle(
        [int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)], fill=255)
    buf = io.BytesIO()
    mask.save(buf, "PNG")
    return buf.getvalue()
