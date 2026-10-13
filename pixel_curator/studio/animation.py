"""Feature 7 - garment transformation sequence (AnimateDiff, planning-first).

Plans and (where a motion backend is available) generates a short, strictly
professional catalog-style wardrobe variation sequence from a personal
reference image:

    base catalog outfit -> garment transition keyframes
    -> refined wardrobe variation -> final catalog styling frame

Design: planning always works, even with no AnimateDiff / ControlNet models,
no Forge server, no GPU and no ffmpeg. It validates the LoRA stack (Feature
5), reuses ControlNet / IP-Adapter unit builders, optional region-aware
inpainting plans, optional Feature 6 post-processing, writes a manifest, and
optionally exports a professional GIF preview. Missing optional backends
degrade to clear warnings - never a crash.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..eval.forge_api import ForgeClient, build_animatediff_payload, export_sequence_gif
from ..lora_validator import normalize_stack_text, validate_lora_stack

log = logging.getLogger("redresser.animation")

# Ordered catalog stages; a sequence of any length samples across these.
STAGE_LABELS = [
    "base_catalog_frame", "early_garment_transition",
    "subtle_fabric_interaction", "mid_sequence_wardrobe_variation",
    "refined_garment_layering", "catalog_styling_adjustment",
    "final_wardrobe_composition", "final_catalog_frame",
]
STAGE_PROMPTS = {
    "base_catalog_frame": "base catalog wardrobe, neutral editorial stance",
    "early_garment_transition": "early garment transition, refined fabric "
                                "interaction",
    "subtle_fabric_interaction": "subtle fabric interaction, soft studio "
                                 "shadows",
    "mid_sequence_wardrobe_variation": "wardrobe variation, layered garment "
                                       "rendering",
    "refined_garment_layering": "refined garment layering, clean garment "
                                "boundaries",
    "catalog_styling_adjustment": "catalog styling adjustment, premium "
                                  "lookbook composition",
    "final_wardrobe_composition": "final wardrobe composition, polished "
                                  "catalog styling",
    "final_catalog_frame": "final catalog frame, editorial studio lighting",
}
IDENTITY_NEGATIVE = ("identity drift, inconsistent facial structure, "
                     "garment boundary artifacts, unstable hands, "
                     "frame-to-frame lighting shift, distorted proportions")


def _stage_for(i: int, n: int) -> str:
    if n <= 1:
        return STAGE_LABELS[0]
    idx = round(i * (len(STAGE_LABELS) - 1) / (n - 1))
    return STAGE_LABELS[max(0, min(idx, len(STAGE_LABELS) - 1))]


def _denoise_for(stage: str) -> float:
    # conservative, identity-preserving; gentlest at the anchor + final frames
    return {
        "base_catalog_frame": 0.30, "early_garment_transition": 0.38,
        "subtle_fabric_interaction": 0.40,
        "mid_sequence_wardrobe_variation": 0.45,
        "refined_garment_layering": 0.42,
        "catalog_styling_adjustment": 0.38,
        "final_wardrobe_composition": 0.34, "final_catalog_frame": 0.32,
    }.get(stage, 0.40)


@dataclass
class FramePlan:
    frame_index: int
    stage: str
    prompt: str
    negative_prompt: str
    lora_prompt_fragment: str
    seed: int
    denoise: float
    controlnet: dict = field(default_factory=dict)
    ip_adapter: dict = field(default_factory=dict)
    region: str = ""
    post_process: dict = field(default_factory=dict)
    output_path: str = ""
    warnings: list[str] = field(default_factory=list)


class GarmentTransformationSequence:
    """Plan (and optionally generate) a wardrobe transformation sequence."""

    def __init__(self, forge_api: ForgeClient | None = None,
                 output_dir: Path | str | None = None,
                 lora_dir: Path | str | None = None,
                 enable_post_processing: bool = False) -> None:
        self.forge = forge_api
        self.output_dir = Path(output_dir or "outputs/garment_sequences"
                               ).expanduser()
        self.lora_dir = Path(lora_dir).expanduser() if lora_dir else None
        self.enable_post_processing = bool(enable_post_processing)

    # -- preset lookup -------------------------------------------------------
    @staticmethod
    def _load_preset(target_preset: str) -> tuple[dict, list[str]]:
        from .. import starter_stacks as ss
        warns: list[str] = []
        preset = (ss.get_motion_preset(target_preset)
                  if hasattr(ss, "get_motion_preset") else None)
        if preset is None and hasattr(ss, "get_wardrobe_preset"):
            preset = ss.get_wardrobe_preset(target_preset)
        if preset is None:
            warns.append(f"Preset not found: '{target_preset}'; using a "
                         "neutral catalog plan.")
            preset = {"name": target_preset or "Neutral Catalog Sequence",
                      "loras": [], "prompt": "high-end catalog styling, "
                      "controlled wardrobe variation",
                      "negative": "", "motion": {}}
        return preset, warns

    # -- planning ------------------------------------------------------------
    def plan(self, init_image_path: Path | str, lora_stack: list,
             target_preset: str, num_frames: int = 8, fps: int = 12) -> dict:
        from ..util import now_iso
        warnings: list[str] = []
        preset, pwarn = self._load_preset(target_preset)
        warnings += pwarn
        motion = preset.get("motion") or {}
        nf = max(4, min(int(num_frames or motion.get("frames", 8)), 16))
        fps = int(fps or motion.get("fps", 12))
        motion_type = str(motion.get("type", "catalog_turntable"))

        # merge UI stack + preset stack, then validate (Feature 5)
        entries = []
        if isinstance(lora_stack, str):
            entries += normalize_stack_text(lora_stack)
        elif lora_stack:
            entries += [str(x) for x in lora_stack]
        entries += list(preset.get("loras", []))
        cleaned, lwarn = validate_lora_stack(entries, lora_dir=self.lora_dir)
        warnings += lwarn
        lora_fragment = cleaned["prompt_fragment"]
        if not cleaned["valid_pairs"]:
            warnings.append("No identity/garment LoRAs resolved; sequence "
                            "may have reduced identity consistency.")

        base_prompt = str(preset.get("prompt", "")).strip(", ")
        base_negative = ", ".join(
            x for x in (str(preset.get("negative", "")).strip(", "),
                        IDENTITY_NEGATIVE) if x)
        pp_default = preset.get("post_process") or {
            "realesrgan": True, "gfpgan": True, "gfpgan_weight": 0.3}

        ts = now_iso().replace(":", "").replace("-", "")[:15]
        slug = "".join(c.lower() if c.isalnum() else "_"
                       for c in (preset.get("name") or "sequence"))[:48]
        seq_dir = self.output_dir / f"{ts}_{slug}"
        frames_dir = seq_dir / "frames"

        base_seed = 12001
        frames: list[FramePlan] = []
        for i in range(nf):
            stage = _stage_for(i, nf)
            add = STAGE_PROMPTS.get(stage, "")
            prompt = ", ".join(x for x in (base_prompt, add) if x)
            if lora_fragment:
                prompt = f"{prompt} {lora_fragment}".strip()
            region = ""
            if stage in ("mid_sequence_wardrobe_variation",
                         "refined_garment_layering"):
                region = (preset.get("recommended_regions") or
                          ["upper_garment"])[0]
            frames.append(FramePlan(
                frame_index=i, stage=stage, prompt=prompt,
                negative_prompt=base_negative,
                lora_prompt_fragment=lora_fragment, seed=base_seed + i,
                denoise=_denoise_for(stage),
                controlnet={"openpose": 0.78, "depth": 0.5}
                if motion.get("pose_guidance") else {},
                ip_adapter={"type": "faceid", "weight": 0.75},
                region=region, post_process=dict(pp_default),
                output_path=str(frames_dir / f"frame_{i:03d}.png")))

        manifest = {
            "sequence_id": f"{ts}_{slug}",
            "target_preset": preset.get("name", target_preset),
            "num_frames": nf, "fps": fps, "motion_type": motion_type,
            "init_image_path": str(Path(init_image_path).expanduser()),
            "output_dir": str(seq_dir),
            "lora_prompt_fragment": lora_fragment,
            "total_lora_weight": cleaned["total_weight"],
            "frames": [asdict(f) for f in frames],
            "gif_path": str(seq_dir / "preview.gif"),
            "post_processing_enabled": self.enable_post_processing,
            "warnings": warnings,
        }
        return manifest

    # -- execution (best-effort) --------------------------------------------
    def generate_garment_transformation_sequence(
            self, init_image_path: Path | str, lora_stack: list,
            target_preset: str, num_frames: int = 8, fps: int = 12
            ) -> list[Path]:
        """Plan the sequence, write the manifest, and generate frames when a
        motion backend is reachable. Always returns the list of frame paths
        that exist on disk (empty in plan-only mode)."""
        manifest = self.plan(init_image_path, lora_stack, target_preset,
                              num_frames, fps)
        seq_dir = Path(manifest["output_dir"])
        (seq_dir / "frames").mkdir(parents=True, exist_ok=True)
        (seq_dir / "post_processed").mkdir(parents=True, exist_ok=True)

        produced: list[Path] = []
        backend_ok = bool(self.forge) and self._animatediff_available()
        if not backend_ok:
            manifest["warnings"].append(
                "AnimateDiff backend not available; wrote sequence plan + "
                "manifest only (no frames generated).")
        else:                                          # pragma: no cover
            produced = self._execute(manifest)

        if self.enable_post_processing and produced:   # pragma: no cover
            produced = self._post_process(manifest, produced)

        if produced:                                   # pragma: no cover
            gif, gwarn = export_sequence_gif(
                produced, seq_dir / "preview.gif", fps=manifest["fps"])
            manifest["warnings"] += gwarn
            manifest["gif_path"] = str(gif) if gif else ""
        else:
            manifest["gif_path"] = ""

        self._write_manifest(seq_dir, manifest)
        return produced

    # -- helpers -------------------------------------------------------------
    def _animatediff_available(self) -> bool:
        try:
            scripts = self.forge._get("/sdapi/v1/scripts")  # type: ignore
            txt = json.dumps(scripts).lower()
            return "animatediff" in txt
        except Exception:
            return False

    def build_frame_payload(self, frame: dict, manifest: dict) -> dict:
        """Construct the AnimateDiff payload for one frame plan (plan mode)."""
        cn_units, ip_units = [], []
        if frame.get("controlnet"):
            for ctype, w in frame["controlnet"].items():
                cn_units.append(ForgeClient.controlnet_unit(
                    "", control_type=ctype, weight=float(w)))
        if frame.get("ip_adapter"):
            ip_units.append(ForgeClient.ipadapter_unit(
                "", ip_type=str(frame["ip_adapter"].get("type", "faceid")),
                weight=float(frame["ip_adapter"].get("weight", 0.75))))
        payload = build_animatediff_payload(
            prompt=frame["prompt"], negative_prompt=frame["negative_prompt"],
            init_image_path=manifest["init_image_path"],
            controlnet_units=cn_units, ip_adapter_units=ip_units,
            num_frames=manifest["num_frames"], fps=manifest["fps"],
            motion_type=manifest["motion_type"], seed=frame["seed"])
        return payload

    def _execute(self, manifest: dict) -> list[Path]:   # pragma: no cover
        produced: list[Path] = []
        for frame in manifest["frames"]:
            payload = self.build_frame_payload(frame, manifest)
            payload.pop("_warnings", None)
            try:
                data = self.forge._post("/sdapi/v1/txt2img", payload)
                import base64
                png = base64.b64decode(data["images"][0])
                out = Path(frame["output_path"])
                out.write_bytes(png)
                produced.append(out)
            except Exception as exc:
                frame["warnings"].append(f"Frame generation skipped: {exc}")
        return produced

    def _post_process(self, manifest: dict,                # pragma: no cover
                      frames: list[Path]) -> list[Path]:
        try:
            from ..post_processing import PostProcessor
        except Exception:
            manifest["warnings"].append("Post-processing module unavailable; "
                                        "frames left unrefined.")
            return frames
        pp = PostProcessor()
        out_dir = Path(manifest["output_dir"]) / "post_processed"
        refined: list[Path] = []
        for fp in frames:
            res = pp.process(fp, gfpgan_weight=0.3)
            manifest["warnings"] += res.warnings
            target = Path(res.post_processed_path)
            if target != fp and target.is_file():
                dest = out_dir / target.name
                dest.write_bytes(target.read_bytes())
                refined.append(dest)
            else:
                refined.append(fp)
        return refined

    @staticmethod
    def _write_manifest(seq_dir: Path, manifest: dict) -> Path:
        seq_dir.mkdir(parents=True, exist_ok=True)
        path = seq_dir / "sequence_manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path


def generate_garment_transformation_sequence(
        init_image_path: Path | str, lora_stack: list, target_preset: str,
        num_frames: int = 8, fps: int = 12,
        forge_api: ForgeClient | None = None,
        output_dir: Path | str | None = None,
        lora_dir: Path | str | None = None,
        enable_post_processing: bool = False) -> list[Path]:
    """Module-level wrapper around :class:`GarmentTransformationSequence`."""
    seq = GarmentTransformationSequence(
        forge_api=forge_api, output_dir=output_dir, lora_dir=lora_dir,
        enable_post_processing=enable_post_processing)
    return seq.generate_garment_transformation_sequence(
        init_image_path, lora_stack, target_preset, num_frames, fps)
