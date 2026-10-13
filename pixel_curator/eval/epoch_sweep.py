"""Epoch picker - score every saved epoch of a LoRA and name the keeper.

A training run saves one checkpoint per epoch. This sweeps each epoch across a
few LoRA weights, generates identity probe images through Forge, scores each
with ArcFace likeness vs the anchor, and ranks them so you don't have to eyeball
ten files. Reuses the existing `likeness_scorer`, `ForgeClient` and grid
assembly - nothing new about generation or scoring.

CLI:  pixel-curator pick-epoch --lora-base ohwx_character_v012 --weights 0.6,0.75,0.9
"""

from __future__ import annotations

import re
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path

from .. import manifest
from ..config import Project

# Identity probes - face-forward so the scorer always has a face to read.
DEFAULT_PROMPTS = (
    "{trigger}, portrait, looking at camera, soft studio lighting, sharp focus",
    "{trigger}, upper body, neutral background, natural light",
)
DEFAULT_SEEDS = (12345, 6789)


@dataclass
class EpochSweepConfig:
    output_base: Path
    lora_base: str                      # e.g. ohwx_character_v012
    lora_dir: Path                      # where the -NNNNNN.safetensors live
    weights: tuple[float, ...] = (0.6, 0.75, 0.9)
    checkpoint: str = ""
    forge_url: str = "http://127.0.0.1:7860"
    trigger: str = "ohwx person"
    prompts: tuple[str, ...] = DEFAULT_PROMPTS
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    pony_prefix: bool = True
    generate_fn: Callable | None = None   # test hook: (prompt, seed, lora_stem, weight)->png bytes
    score_fn: Callable | None = None       # test hook: (png_path)->float|None


_EPOCH_RE = re.compile(r"-0*(\d+)\.safetensors$")


def discover_epochs(lora_dir, lora_base: str) -> list[tuple[int, Path]]:
    """Find ``{base}-NNNNNN.safetensors`` checkpoints, sorted by epoch number."""
    d = Path(lora_dir).expanduser()
    if not d.exists():
        return []
    out: list[tuple[int, Path]] = []
    for p in d.glob(f"{lora_base}-*.safetensors"):
        m = _EPOCH_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    # a final "{base}.safetensors" (no epoch suffix) counts as the last epoch
    final = d / f"{lora_base}.safetensors"
    if final.exists():
        nxt = (max((e for e, _ in out), default=0) + 1)
        out.append((nxt, final))
    return sorted(out, key=lambda t: t[0])


def rank_results(rows: list[dict]) -> list[dict]:
    """Sort scored (epoch, weight) rows by mean likeness, best first.

    Each row: {epoch, weight, mean_likeness, n_scored}. Rows with no scored
    faces (mean_likeness None) sort to the bottom.
    """
    def key(r):
        ml = r.get("mean_likeness")
        return (ml is not None, ml if ml is not None else -1.0)
    return sorted(rows, key=key, reverse=True)


def parse_weights(spec: str) -> tuple[float, ...]:
    out = []
    for tok in str(spec or "").split(","):
        tok = tok.strip()
        if tok:
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return tuple(out) or (0.75,)


def _mean(vals: list[float]) -> float | None:
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def epoch_sweep_generator(prj: Project, cfg: EpochSweepConfig
                          ) -> Generator[str, None, None]:
    """Score each epoch x weight; yield progress, then the ranking + best pick."""
    from .matrix import assemble_grid, compose_prompt, likeness_scorer

    conn = manifest.connect(cfg.output_base)
    scorer = cfg.score_fn or (likeness_scorer(conn) if cfg.generate_fn is None else None)

    epochs = discover_epochs(cfg.lora_dir, cfg.lora_base)
    if not epochs:
        yield (f"No checkpoints found for '{cfg.lora_base}' in {cfg.lora_dir}. "
               "Has training saved any epochs yet?")
        return
    yield (f"Found {len(epochs)} epoch checkpoint(s); weights {list(cfg.weights)}; "
           f"{len(cfg.prompts)} prompt(s) x {len(cfg.seeds)} seed(s).")

    # generation function: test hook, or real Forge client
    gen = cfg.generate_fn
    client = None
    if gen is None:
        from .forge_api import ForgeClient
        client = ForgeClient(cfg.forge_url)
        if not client.alive():
            yield (f"Forge not reachable at {cfg.forge_url}. Start reForge with "
                   "--api, or pass a generate_fn.")
            return
        if scorer is None:
            yield "No identity anchor / [ai] extras - cannot score likeness. Run `anchor` first."
            return
        if cfg.checkpoint:
            client.set_options({"sd_model_checkpoint": cfg.checkpoint})

    prompts = [compose_prompt(p.replace("{trigger}", cfg.trigger), prj, cfg.pony_prefix)
               if "{trigger}" in p else
               compose_prompt(p, prj, cfg.pony_prefix) for p in cfg.prompts]

    out_dir = cfg.output_base / "eval" / f"epoch_sweep_{cfg.lora_base}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    best_image_per_epoch: dict[int, Path] = {}

    for epoch, ckpt in epochs:
        stem = ckpt.stem
        for w in cfg.weights:
            scores: list[float] = []
            for pi, prompt in enumerate(prompts):
                for seed in cfg.seeds:
                    try:
                        if gen is not None:
                            png = gen(prompt, seed, stem, w)
                        else:
                            png = client.txt2img(
                                prompt=prompt, seed=seed, loras=[(stem, w)])
                        img_path = out_dir / f"e{epoch:03d}_w{w:g}_p{pi}_s{seed}.png"
                        img_path.write_bytes(png)
                        s = scorer(img_path) if scorer else None
                        if s is not None:
                            scores.append(s)
                            best_image_per_epoch.setdefault(epoch, img_path)
                    except Exception as exc:                # pragma: no cover
                        yield f"  gen/score failed (e{epoch} w{w}): {exc}"
            rows.append({"epoch": epoch, "weight": w,
                         "mean_likeness": _mean(scores), "n_scored": len(scores)})
            yield (f"  epoch {epoch:>3} @ w{w:g}: likeness "
                   f"{rows[-1]['mean_likeness']} (n={len(scores)})")

    ranked = rank_results(rows)
    yield "\n=== EPOCH RANKING (best likeness first) ==="
    yield f"{'epoch':>6} {'weight':>7} {'likeness':>9} {'n':>4}"
    for r in ranked:
        yield (f"{r['epoch']:>6} {r['weight']:>7g} "
               f"{str(r['mean_likeness']):>9} {r['n_scored']:>4}")

    best = next((r for r in ranked if r["mean_likeness"] is not None), None)
    if best:
        stem = f"{cfg.lora_base}-{best['epoch']:06d}"
        yield (f"\nBEST: epoch {best['epoch']} @ weight {best['weight']:g} "
               f"(likeness {best['mean_likeness']}) -> {stem}.safetensors")

    # contact sheet of the best image per epoch
    try:
        imgs = [best_image_per_epoch[e] for e, _ in epochs if e in best_image_per_epoch]
        if imgs:
            labels = [f"e{e}" for e, _ in epochs if e in best_image_per_epoch]
            grid = out_dir / "epoch_contact_sheet.jpg"
            assemble_grid(imgs, grid, cols=min(5, len(imgs)), labels=labels)
            yield f"Contact sheet: {grid}"
    except Exception as exc:                                # pragma: no cover
        yield f"(contact sheet skipped: {exc})"
