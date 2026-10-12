"""Phase 6: test-matrix generator with anchor-based likeness scoring.

One command tests a trained LoRA across 5 probe categories at fixed seeds:
  likeness    - studio portraits at varied angles (does it look like the subject?)
  flexibility - scenes/outfits NOT in the dataset (is it overfit?)
  pose        - dynamic poses and framings
  outfit      - outfit-change probes (does identity survive clothing swaps?)
  style       - rendering-style shifts (does identity survive style transfer?)

Every image is scored for likeness against the Phase 2 identity anchor
(InsightFace cosine sim), recorded in the `evals` table, and assembled into
a labeled grid per category. Comparing average likeness across epoch
checkpoints finds your best epoch with numbers instead of vibes.

Backends: 'forge' (reForge --api) or 'diffusers' (local TestPipeline).
Tests inject a fake generate_fn the same way kohya tests inject trainer_cmd.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Optional

from .. import manifest
from ..config import Project
from ..util import now_iso, safe_slug, setup_logging

SEEDS = [42, 1042, 2042]

# Professional probe prompts; {t} = trigger phrase ("token class_word").
CATEGORIES: dict[str, list[str]] = {
    "likeness": [
        "{t}, studio portrait, soft lighting, looking at viewer, photorealistic",
        "{t}, close-up face, natural light, neutral expression, detailed skin",
        "{t}, profile view portrait, rim lighting, photorealistic",
        "{t}, upper body, three-quarter view, smiling, sharp focus",
    ],
    "flexibility": [
        "{t}, astronaut suit, standing on mars, cinematic lighting",
        "{t}, victorian era dress suit, oil painting style interior",
        "{t}, chef uniform, busy restaurant kitchen, candid",
        "{t}, winter coat and scarf, snowy street at night, bokeh",
    ],
    "pose": [
        "{t}, full body, dynamic action pose, jumping mid-air",
        "{t}, sitting cross-legged reading a book, side view",
        "{t}, walking toward viewer, full body, street photography",
        "{t}, from behind, looking over shoulder, dramatic lighting",
    ],
    "outfit": [
        "{t}, formal black suit, studio backdrop, full body",
        "{t}, casual jeans and t-shirt, outdoor park, full body",
        "{t}, athletic sportswear, gym setting, dynamic pose",
        "{t}, elegant evening wear, gala event, upper body",
    ],
    "style": [
        "{t}, anime style illustration, vibrant colors",
        "{t}, black and white film photography, grainy, 35mm",
        "{t}, watercolor painting, soft brushstrokes",
        "{t}, comic book style, bold lines, halftone shading",
    ],
    # Probes matched to the fine_art_figure_study training preset: does the
    # LoRA hold likeness under the lighting/framing it was trained for?
    "fine_art": [
        "{t}, fine art figure study, chiaroscuro lighting, detailed anatomy, "
        "realistic rendering",
        "{t}, classical figure study pose, soft window light, expressive "
        "character, fine art photography",
        "{t}, full body dynamic pose, sculptural studio lighting, detailed "
        "anatomy, gallery print quality",
        "{t}, reclining figure study, painterly rim light, realistic "
        "rendering, dramatic shadows",
    ],
}

DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, deformed, blurry, watermark, "
    "text, signature, jpeg artifacts, worst quality"
)


@dataclass
class MatrixConfig:
    output_base: Path
    lora: str                          # path (diffusers) or model stem (forge)
    label: str = ""                    # defaults to lora stem
    categories: list[str] = field(default_factory=lambda: list(CATEGORIES))
    seeds: list[int] = field(default_factory=lambda: list(SEEDS))
    backend: str = "forge"             # forge | diffusers
    forge_url: str = "http://127.0.0.1:7860"
    checkpoint: str = ""               # diffusers backend: base model path
    lora_weight: float = 0.85
    extra_loras: list[tuple[str, float]] = field(default_factory=list)
    steps: int = 0                     # 0 -> base-model profile default
    cfg: float = 0.0                   # 0 -> base-model profile default
    sampler: str = ""                  # "" -> base-model profile default
    clip_skip: int = 0                 # 0 -> base-model profile default
    width: int = 1024
    height: int = 1024
    negative: str = ""                 # "" -> base-model profile default
    pony_prefix: bool = True
    generate_fn: Optional[Callable] = None    # test hook: (prompt, seed) -> png bytes


def parse_lora_specs(spec: str) -> list[tuple[str, float]]:
    """'path1:0.85, path2:-0.4, path3' -> [(path, weight), ...].
    Splitting on the LAST colon keeps Windows-style or odd paths intact."""
    out: list[tuple[str, float]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            path, _, w = part.rpartition(":")
            try:
                out.append((path.strip(), float(w)))
                continue
            except ValueError:
                pass
        out.append((part, 1.0))
    return out


def trigger_phrase(prj: Project) -> str:
    return f"{prj.trigger_token} {prj.class_word}".strip()


def compose_prompt(template: str, prj: Project, pony_prefix: bool) -> str:
    p = template.format(t=trigger_phrase(prj))
    if pony_prefix:
        p = "score_9, score_8_up, score_7_up, " + p
    return p


# -----------------------------
# Likeness scoring (reuses smart-curation backend + anchor)
# -----------------------------

def likeness_scorer(conn) -> Optional[Callable]:
    """Returns score(png_path) -> float|None, or None when unavailable."""
    anchor_list = manifest.meta_get(conn, "identity_anchor")
    if not anchor_list:
        return None
    try:
        from ..curate.smart import _load_backend, _read_bgr
        import numpy as np
        app, np_, cv2 = _load_backend()
        anchor = np.asarray(anchor_list, dtype=np.float32)

        def score(path: Path) -> Optional[float]:
            img = _read_bgr(path, np_, cv2)
            if img is None:
                return None
            faces = app.get(img)
            if not faces:
                return None
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            return float(np.dot(anchor, face.normed_embedding))

        return score
    except Exception as exc:
        logging.info("Likeness scoring unavailable: %s", exc)
        return None


# -----------------------------
# Grid assembly
# -----------------------------

def grid_layout(n_images: int, cols: int) -> tuple[int, int]:
    rows = (n_images + cols - 1) // cols
    return rows, cols


def assemble_grid(image_paths: list[Path], out_path: Path, cols: int,
                  cell: int = 384, labels: Optional[list[str]] = None) -> None:
    from PIL import Image, ImageDraw
    rows, cols = grid_layout(len(image_paths), cols)
    grid = Image.new("RGB", (cols * cell, rows * cell), (16, 18, 24))
    for i, p in enumerate(image_paths):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((cell, cell))
                x = (i % cols) * cell + (cell - im.width) // 2
                y = (i // cols) * cell + (cell - im.height) // 2
                grid.paste(im, (x, y))
        except Exception:
            continue
        if labels and i < len(labels) and labels[i]:
            d = ImageDraw.Draw(grid)
            d.text(((i % cols) * cell + 6, (i // cols) * cell + 4),
                   labels[i], fill=(0, 212, 170))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, "JPEG", quality=88)


# -----------------------------
# Backends
# -----------------------------

def _apply_profile_defaults(cfg: MatrixConfig, prj: Project) -> None:
    """Fill unset generation fields from the base-model profile."""
    from ..base_models import detect_profile
    prof = detect_profile(prj.base_model)
    if not cfg.steps:
        cfg.steps = prof["steps"]
    if not cfg.cfg:
        cfg.cfg = prof["cfg"]
    if not cfg.sampler:
        cfg.sampler = prof["sampler"]
    if not cfg.clip_skip:
        cfg.clip_skip = prof["clip_skip"]
    if not cfg.negative:
        cfg.negative = prof["negative"] or DEFAULT_NEGATIVE


def _make_generate_fn(cfg: MatrixConfig, prj: Project) -> Callable:
    loras = [(cfg.lora, cfg.lora_weight)] + list(cfg.extra_loras)
    if cfg.backend == "forge":
        from .forge_api import ForgeClient
        client = ForgeClient(cfg.forge_url)
        if not client.alive():
            raise RuntimeError(
                f"Forge API not reachable at {cfg.forge_url}. "
                f"Start reForge with --api (root: {prj.forge_root})."
            )

        def gen(prompt: str, seed: int) -> bytes:
            return client.txt2img(
                prompt=prompt, negative=cfg.negative, steps=cfg.steps,
                cfg=cfg.cfg, width=cfg.width, height=cfg.height,
                seed=seed, loras=loras, sampler=cfg.sampler,
                clip_skip=cfg.clip_skip,
            )
        return gen

    if cfg.backend == "diffusers":
        from .pipeline import diffusers_available, get_pipeline
        ok, reason = diffusers_available()
        if not ok:
            raise RuntimeError(reason)
        ckpt = cfg.checkpoint or prj.base_model
        if not ckpt:
            raise RuntimeError("checkpoint/base_model not set")
        tp = get_pipeline(ckpt, "", loras)

        def gen(prompt: str, seed: int) -> bytes:
            import io
            img = tp.txt2img(prompt, cfg.negative, cfg.steps, cfg.cfg,
                             cfg.width, cfg.height, seed)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return buf.getvalue()
        return gen

    raise RuntimeError(f"unknown backend: {cfg.backend}")


# -----------------------------
# Matrix generator
# -----------------------------

def matrix_generator(prj: Project, cfg: MatrixConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)
    _apply_profile_defaults(cfg, prj)
    yield (f"generation: {cfg.sampler} @ {cfg.steps} steps, CFG {cfg.cfg}, "
           f"clip skip {cfg.clip_skip}")

    label = cfg.label or safe_slug(Path(cfg.lora).stem)
    out_dir = cfg.output_base / "evals" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        gen = cfg.generate_fn or _make_generate_fn(cfg, prj)
    except RuntimeError as exc:
        yield f"FATAL: {exc}"
        return

    scorer = likeness_scorer(conn)
    cats = [c for c in cfg.categories if c in CATEGORIES]
    total = sum(len(CATEGORIES[c]) for c in cats) * len(cfg.seeds)
    yield (f"Test matrix '{label}' | backend {cfg.backend} | {total} images "
           f"({len(cats)} categories x {len(cfg.seeds)} seeds)\n"
           f"Likeness scoring: {'ON (anchor found)' if scorer else 'off (no anchor / no [ai])'}\n")

    done = 0
    t0 = time.time()
    cat_scores: dict[str, list[float]] = {c: [] for c in cats}

    for cat in cats:
        paths: list[Path] = []
        labels: list[str] = []
        for pi, template in enumerate(CATEGORIES[cat]):
            prompt = compose_prompt(template, prj, cfg.pony_prefix)
            for seed in cfg.seeds:
                fname = f"{cat}_p{pi}_s{seed}.png"
                fpath = out_dir / fname
                try:
                    png = gen(prompt, seed)
                    fpath.write_bytes(png)
                except Exception as exc:
                    yield f"  ERR {fname}: {str(exc)[:200]}"
                    continue
                like = scorer(fpath) if scorer else None
                if like is not None:
                    cat_scores[cat].append(like)
                conn.execute(
                    "INSERT INTO evals(lora, label, category, prompt, seed, "
                    "backend, image_path, likeness, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (cfg.lora, label, cat, prompt, seed, cfg.backend,
                     str(fpath), like, now_iso()),
                )
                conn.commit()
                paths.append(fpath)
                labels.append(f"{like:.2f}" if like is not None else "")
                done += 1
                rate = done / max(0.001, time.time() - t0)
                yield (f"[{done}/{total}] {cat} seed {seed} "
                       f"{'likeness %.3f' % like if like is not None else ''} "
                       f"({rate*60:.1f} img/min)")
        if paths:
            grid_path = out_dir / f"GRID_{cat}.jpg"
            assemble_grid(paths, grid_path, cols=len(cfg.seeds), labels=labels)
            yield f"  grid -> {grid_path}"

    report = []
    for cat in cats:
        scores = cat_scores[cat]
        if scores:
            report.append(f"  {cat:<12} avg {sum(scores)/len(scores):.3f}  "
                          f"min {min(scores):.3f}  n={len(scores)}")
        else:
            report.append(f"  {cat:<12} (unscored)")
    verdict = ""
    lk, fx = cat_scores.get("likeness", []), cat_scores.get("flexibility", [])
    if lk and fx:
        gap = (sum(lk)/len(lk)) - (sum(fx)/len(fx))
        verdict = ("\nOverfit check: likeness-flexibility gap "
                   f"{gap:.3f} ({'OK' if gap < 0.15 else 'HIGH - consider fewer epochs/repeats'})")
    yield (
        f"\nMATRIX DONE: {done}/{total} images -> {out_dir}\n"
        f"Likeness by category:\n" + "\n".join(report) + verdict +
        "\n\nCompare epochs: run with --lora <epoch file> and diff averages in `pixel-curator evals`."
    )


def eval_summary(conn) -> list[dict]:
    return [
        {"label": r[0], "category": r[1], "n": r[2],
         "avg_likeness": round(r[3], 4) if r[3] is not None else None}
        for r in conn.execute(
            "SELECT label, category, COUNT(*), AVG(likeness) FROM evals "
            "GROUP BY label, category ORDER BY label, category"
        )
    ]
