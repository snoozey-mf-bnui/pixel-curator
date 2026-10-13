"""QA rendering - Forge-rendered LoRA previews, same-seed A/B compare,
and golden prompt set regression.  All three run through the existing
Forge adapter with CoreShift profile settings; offline they explain and
exit cleanly."""
from __future__ import annotations

import json
import logging
from collections.abc import Generator
from pathlib import Path

from .config import Project

logger = logging.getLogger(__name__)

PREVIEW_WEIGHTS = {"low": 0.4, "medium": 0.7, "high": 1.0}
GOLDEN_SEED = 4242


def _client(prj: Project, url: str = "http://127.0.0.1:7860"):
    from .eval.forge_api import ForgeClient
    c = ForgeClient(url)
    return c if c.alive() else None


def _render(client, prj: Project, prompt: str,
            loras: list, seed: int) -> bytes:
    from .base_models import detect_profile
    prof = detect_profile(prj.base_model)
    return client.txt2img(
        prompt=prompt, negative=prof["negative"], steps=prof["steps"],
        cfg=prof["cfg"], width=prof["width"], height=prof["height"],
        seed=seed, sampler=prof["sampler"], loras=loras,
        clip_skip=prof["clip_skip"])


def render_lora_previews(prj: Project, lora_path,
                         forge_url: str = "http://127.0.0.1:7860"
                         ) -> Generator[str, None, None]:
    """Default/low/medium/high strength previews beside the LoRA file -
    the Explorer picks them up on the next rescan."""
    from .base_models import build_prompt, detect_profile
    lp = Path(lora_path).expanduser()
    client = _client(prj, forge_url)
    if client is None:
        yield "Forge API not reachable - start it from the Engines strip."
        return
    prof = detect_profile(prj.base_model)
    trig = f"{prj.trigger_token} {prj.class_word}".strip()
    prompt = build_prompt(prof, trig).rstrip(", ") + \
        ", studio portrait, soft lighting, clean background"
    for level, w in [("default", 0.75), *PREVIEW_WEIGHTS.items()]:
        suffix = ".preview.png" if level == "default" else f".{level}.png"
        out = lp.with_name(lp.stem + suffix)
        png = _render(client, prj, prompt, [(lp.stem, w)], GOLDEN_SEED)
        out.write_bytes(png)
        yield f"  {level} (weight {w}) -> {out.name}"
    yield "Previews rendered - Rescan folders in the Explorer to see them."


def ab_compare(prj: Project, lora_a: tuple, lora_b: tuple,
               prompt_tail: str = "", seed: int = GOLDEN_SEED,
               forge_url: str = "http://127.0.0.1:7860"
               ) -> Generator[str, None, None]:
    """Same-seed side-by-side of two LoRAs/epochs; composite saved to
    evals/ab/."""
    from .base_models import build_prompt, detect_profile
    client = _client(prj, forge_url)
    if client is None:
        yield "Forge API not reachable - start it from the Engines strip."
        return
    prof = detect_profile(prj.base_model)
    trig = f"{prj.trigger_token} {prj.class_word}".strip()
    prompt = build_prompt(prof, trig).rstrip(", ")
    if prompt_tail:
        prompt += f", {prompt_tail.strip(', ')}"
    imgs = []
    for name, w in (lora_a, lora_b):
        yield f"  rendering {name}:{w} @ seed {seed}..."
        imgs.append(_render(client, prj, prompt, [(name, w)], seed))
    out_dir = prj.output_path / "evals" / "ab"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"ab_{lora_a[0]}_vs_{lora_b[0]}_s{seed}.png"
    side_by_side(imgs[0], imgs[1], fp)
    yield f"A/B composite -> {fp}"


def side_by_side(png_a: bytes, png_b: bytes, out_path) -> Path:
    import io

    from PIL import Image
    a = Image.open(io.BytesIO(png_a))
    b = Image.open(io.BytesIO(png_b))
    h = max(a.height, b.height)
    canvas = Image.new("RGB", (a.width + b.width, h), "black")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width, 0))
    canvas.save(out_path)
    return Path(out_path)


def golden_prompts(prj: Project, path=None) -> list[str]:
    """The reference prompt set; created from the base-model profile on
    first use, then user-editable at outputs/golden_prompts.json."""
    from .base_models import detect_profile, sample_prompts
    repo = Path(__file__).resolve().parent
    fp = Path(path) if path else repo / "outputs" / "golden_prompts.json"
    if fp.exists():
        try:
            return list(json.loads(fp.read_text()))
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass
    prof = detect_profile(prj.base_model)
    trig = f"{prj.trigger_token} {prj.class_word}".strip()
    base = sample_prompts(prof, trig)
    extra_tails = ["neutral portrait, clean background",
                   "full body framing, balanced standing pose",
                   "upper body framing, expressive pose, editorial lighting",
                   "close-up, natural expression, soft window light",
                   "full body, outdoor, golden hour, film grain",
                   "seated pose, studio composition, high detail",
                   "profile view, rim lighting, cinematic mood"]
    prompts = base + [f"{prof['quality_prefix']}{trig}, {t}"
                      for t in extra_tails]
    prompts = prompts[:10]
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(prompts, indent=2))
    return prompts


def golden_regression(prj: Project, lora: str, weight: float = 0.75,
                      previous: str | None = None,
                      forge_url: str = "http://127.0.0.1:7860"
                      ) -> Generator[str, None, None]:
    """Render the 10 golden prompts at a fixed seed, score measured face
    similarity vs the anchor reference, and diff against a previous LoRA's
    golden run when available."""
    client = _client(prj, forge_url)
    if client is None:
        yield "Forge API not reachable - start it from the Engines strip."
        return
    prompts = golden_prompts(prj)
    out_dir = prj.output_path / "evals" / "golden" / lora
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = None
    if prj.anchor_dir:
        refs = sorted(Path(prj.anchor_dir).expanduser().glob("*.[jp][pn]g"))
        ref = refs[0] if refs else None
    sims = {}
    for i, prompt in enumerate(prompts):
        fp = out_dir / f"g{i:02d}.png"
        if not fp.exists():
            fp.write_bytes(_render(client, prj, prompt, [(lora, weight)],
                                   GOLDEN_SEED + i))
        sim = None
        if ref is not None:
            from .identity_integration import face_similarity
            sim = face_similarity(fp, ref, prj)
        sims[f"g{i:02d}"] = sim
        yield f"  g{i:02d}: face similarity {sim if sim is not None else '-'}"
    (out_dir / "scores.json").write_text(json.dumps(sims, indent=2))
    valid = [s for s in sims.values() if s is not None]
    mean = round(sum(valid) / len(valid), 3) if valid else None
    yield f"Golden run complete: mean face similarity {mean}"
    if previous:
        prev_fp = (prj.output_path / "evals" / "golden" / previous
                   / "scores.json")
        if prev_fp.exists():
            prev = json.loads(prev_fp.read_text())
            yield f"Regression vs {previous}:"
            for k in sorted(sims):
                a, b = prev.get(k), sims[k]
                if a is not None and b is not None:
                    mark = "+" if b >= a else "-"
                    yield f"  {k}: {a} -> {b} ({mark}{abs(round(b-a,3))})"
        else:
            yield f"No previous golden scores for '{previous}'."
