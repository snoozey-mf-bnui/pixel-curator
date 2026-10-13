"""Lookbook / virtual try-on - identity x outfits x poses -> a catalog sheet.

Given your trained identity LoRA, a list of outfit descriptions and a list of
poses, this generates one catalog shot per combination through Forge and
assembles them into a contact sheet. Reuses the existing ForgeClient (LoRA
injection, txt2img) and grid assembler - it's the orchestration layer on top of
primitives that already exist (inpaint / ControlNet pose / IP-Adapter are in
forge_api for region-aware garment swaps later).

SFW fashion/catalog use only. Outputs to ``samples/`` - non-destructive.

CLI:
  pixel-curator lookbook --identity ohwx_character_v012-000008 \
    --outfits "red cocktail dress, denim jacket and jeans, winter coat" \
    --poses "standing, sitting" --weight 0.85
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Project
from ..util import safe_slug

PONY_PREFIX = "score_9, score_8_up, score_7_up, "
NEG_DEFAULT = ("lowres, blurry, bad anatomy, bad hands, deformed, watermark, "
               "text, extra limbs")


def parse_list(spec: str) -> list[str]:
    return [s.strip() for s in str(spec or "").split(",") if s.strip()]


def parse_lora_specs(spec: str, default_weight: float = 0.5) -> list[tuple[str, float]]:
    """Parse 'stem:weight,stem2:weight2' into [(stem, weight), ...].

    A bare stem (no ':weight') gets ``default_weight``.
    """
    out: list[tuple[str, float]] = []
    for tok in str(spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            name, w = tok.rsplit(":", 1)
            try:
                out.append((name.strip(), float(w)))
            except ValueError:
                out.append((tok, default_weight))
        else:
            out.append((tok, default_weight))
    return out


def compose_lookbook_prompts(trigger: str, outfits: list[str], poses: list[str],
                             pony_prefix: bool = True, scene: str = "") -> list[tuple[str, str]]:
    """Return (label, prompt) pairs for every outfit x pose combination."""
    prefix = PONY_PREFIX if pony_prefix else ""
    scene_part = f", {scene}" if scene else ""
    out: list[tuple[str, str]] = []
    for o in outfits:
        for p in poses:
            label = f"{o} / {p}"
            prompt = (f"{prefix}{trigger}, wearing {o}, {p}, full body, "
                      f"fashion catalog photography, studio lighting, "
                      f"sharp focus{scene_part}")
            out.append((label, prompt))
    return out


@dataclass
class LookbookConfig:
    output_base: Path
    identity_lora: str                  # stem of the identity LoRA
    outfits: list[str]
    poses: list[str] = field(default_factory=lambda: ["standing"])
    identity_weight: float = 0.85
    trigger: str = "ohwx person"
    scene: str = ""
    negative: str = NEG_DEFAULT
    checkpoint: str = ""
    forge_url: str = "http://127.0.0.1:7860"
    seed: int = 1234
    pony_prefix: bool = True
    extra_loras: list[tuple[str, float]] = field(default_factory=list)
    generate_fn: Callable | None = None   # test hook: (prompt, seed) -> png bytes


def lookbook_generator(prj: Project, cfg: LookbookConfig) -> Generator[str, None, None]:
    from ..eval.matrix import assemble_grid

    pairs = compose_lookbook_prompts(cfg.trigger, cfg.outfits, cfg.poses,
                                     cfg.pony_prefix, cfg.scene)
    if not pairs:
        yield "Nothing to generate - provide --outfits (and optionally --poses)."
        return
    yield (f"Lookbook: {len(cfg.outfits)} outfit(s) x {len(cfg.poses)} pose(s) "
           f"= {len(pairs)} shots | identity {cfg.identity_lora}@{cfg.identity_weight:g}")

    gen = cfg.generate_fn
    client = None
    if gen is None:
        from ..eval.forge_api import ForgeClient
        client = ForgeClient(cfg.forge_url)
        if not client.alive():
            yield (f"Forge not reachable at {cfg.forge_url}. Start reForge with "
                   "--api (or pass a generate_fn).")
            return
        if cfg.checkpoint:
            client.set_options({"sd_model_checkpoint": cfg.checkpoint})

    loras = [(cfg.identity_lora, cfg.identity_weight), *cfg.extra_loras]
    out_dir = (cfg.output_base / "samples" /
               f"lookbook_{safe_slug(cfg.identity_lora)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs: list[Path] = []
    labels: list[str] = []
    for i, (label, prompt) in enumerate(pairs):
        try:
            if gen is not None:
                png = gen(prompt, cfg.seed + i)
            else:
                png = client.txt2img(prompt=prompt, negative=cfg.negative,
                                     seed=cfg.seed + i, loras=loras)
            path = out_dir / f"{i:03d}_{safe_slug(label)}.png"
            path.write_bytes(png)
            imgs.append(path)
            labels.append(label)
            yield f"  [{i + 1}/{len(pairs)}] {label}"
        except Exception as exc:                            # pragma: no cover
            yield f"  [{i + 1}/{len(pairs)}] FAILED {label}: {exc}"

    if imgs:
        grid = out_dir / "lookbook.jpg"
        try:
            assemble_grid(imgs, grid, cols=max(1, len(cfg.poses)), labels=labels)
            yield f"\nLookbook sheet: {grid}"
        except Exception as exc:                            # pragma: no cover
            yield f"(grid assembly skipped: {exc})"
    yield f"Shots in: {out_dir}"
