"""Stack weight auto-tuner - find the LoRA mix that maximises likeness.

When you stack identity + concept LoRAs, the per-LoRA weights matter a lot.
This sweeps a bounded grid of weight combinations, generates probe images
through Forge, scores each with ArcFace likeness vs the anchor, and ranks them -
so you don't hand-tune weights by eye. Reuses `likeness_scorer` and
`ForgeClient`; nothing new about generation/scoring.

CLI:  pixel-curator tune-stack --identity ohwx_character_v012-000008 \
        --concepts "ohwx_outfit_v014-000006:0.4-0.7, ohwx_pose_v015-000005:0.3-0.5"
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path

from .. import manifest
from ..config import Project

DEFAULT_CONCEPT_RANGE = (0.4, 0.55, 0.7)
DEFAULT_PROMPTS = ("{trigger}, portrait, looking at camera, studio lighting",)
DEFAULT_SEEDS = (4242,)


def parse_concept_ranges(spec: str) -> list[tuple[str, tuple[float, ...]]]:
    """'name:0.4-0.7, name2:0.5' -> [(name,(0.4,0.55,0.7)), (name2,(0.5,))].

    A range 'lo-hi' expands to (lo, mid, hi); a single value stays one point;
    a bare name uses the default range.
    """
    out: list[tuple[str, tuple[float, ...]]] = []
    for tok in str(spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            name, rng = tok.rsplit(":", 1)
            name = name.strip(); rng = rng.strip()
            if "-" in rng:
                try:
                    lo, hi = (float(x) for x in rng.split("-", 1))
                    out.append((name, (round(lo, 3), round((lo + hi) / 2, 3), round(hi, 3))))
                    continue
                except ValueError:
                    pass
            try:
                out.append((name, (float(rng),)))
                continue
            except ValueError:
                pass
            out.append((name, DEFAULT_CONCEPT_RANGE))
        else:
            out.append((tok, DEFAULT_CONCEPT_RANGE))
    return out


def build_weight_grid(identity_lora: str, identity_weights: tuple[float, ...],
                      concepts: list[tuple[str, tuple[float, ...]]], cap: int = 40
                      ) -> list[dict]:
    """Cartesian product of identity x each concept's values -> weight dicts.

    Each dict maps lora-stem -> weight. If the grid exceeds ``cap``, it is
    deterministically strided down to <= cap combinations.
    """
    import itertools
    axes = [[(identity_lora, w) for w in identity_weights]]
    for name, vals in concepts:
        axes.append([(name, v) for v in vals])
    combos = [dict(items) for items in itertools.product(*axes)]
    if cap and len(combos) > cap:
        stride = (len(combos) + cap - 1) // cap
        combos = combos[::stride][:cap]
    return combos


def rank_stacks(rows: list[dict]) -> list[dict]:
    """Sort scored stacks by mean likeness, best first (None last)."""
    def key(r):
        ml = r.get("mean_likeness")
        return (ml is not None, ml if ml is not None else -1.0)
    return sorted(rows, key=key, reverse=True)


def _fmt_stack(weights: dict) -> str:
    return " ".join(f"<lora:{k}:{v:g}>" for k, v in weights.items())


def _mean(vals: list[float]) -> float | None:
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 4) if v else None


@dataclass
class StackTuneConfig:
    output_base: Path
    identity_lora: str
    concepts: list[tuple[str, tuple[float, ...]]]
    identity_weights: tuple[float, ...] = (0.8, 0.9, 1.0)
    checkpoint: str = ""
    forge_url: str = "http://127.0.0.1:7860"
    trigger: str = "ohwx person"
    prompts: tuple[str, ...] = DEFAULT_PROMPTS
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    cap: int = 40
    pony_prefix: bool = True
    generate_fn: Callable | None = None    # (prompt, seed, weights_dict) -> png
    score_fn: Callable | None = None       # (png_path) -> float|None


def stack_tune_generator(prj: Project, cfg: StackTuneConfig
                         ) -> Generator[str, None, None]:
    from .matrix import compose_prompt, likeness_scorer

    conn = manifest.connect(cfg.output_base)
    scorer = cfg.score_fn or (likeness_scorer(conn) if cfg.generate_fn is None else None)

    grid = build_weight_grid(cfg.identity_lora, cfg.identity_weights, cfg.concepts, cfg.cap)
    yield (f"Tuning {len(grid)} weight combination(s) "
           f"(identity {list(cfg.identity_weights)} x {len(cfg.concepts)} concept(s)).")

    gen = cfg.generate_fn
    client = None
    if gen is None:
        from .forge_api import ForgeClient
        client = ForgeClient(cfg.forge_url)
        if not client.alive():
            yield f"Forge not reachable at {cfg.forge_url}. Start reForge with --api."
            return
        if scorer is None:
            yield "No anchor / [ai] extras - cannot score likeness. Run `anchor` first."
            return
        if cfg.checkpoint:
            client.set_options({"sd_model_checkpoint": cfg.checkpoint})

    prompts = [compose_prompt(p.replace("{trigger}", cfg.trigger), prj, cfg.pony_prefix)
               for p in cfg.prompts]
    out_dir = cfg.output_base / "eval" / f"stack_tune_{cfg.identity_lora}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for ci, weights in enumerate(grid):
        scores: list[float] = []
        for pi, prompt in enumerate(prompts):
            for seed in cfg.seeds:
                try:
                    if gen is not None:
                        png = gen(prompt, seed, weights)
                    else:
                        png = client.txt2img(prompt=prompt, seed=seed,
                                             loras=list(weights.items()))
                    p = out_dir / f"c{ci:03d}_p{pi}_s{seed}.png"
                    p.write_bytes(png)
                    s = scorer(p) if scorer else None
                    if s is not None:
                        scores.append(s)
                except Exception as exc:                    # pragma: no cover
                    yield f"  gen/score failed (combo {ci}): {exc}"
        rows.append({"weights": weights, "mean_likeness": _mean(scores),
                     "n_scored": len(scores)})
        yield f"  [{ci + 1}/{len(grid)}] {_fmt_stack(weights)} -> {rows[-1]['mean_likeness']}"

    ranked = rank_stacks(rows)
    yield "\n=== STACK RANKING (best likeness first) ==="
    for r in ranked[:10]:
        yield f"  {str(r['mean_likeness']):>8}  {_fmt_stack(r['weights'])}"
    best = next((r for r in ranked if r["mean_likeness"] is not None), None)
    if best:
        yield f"\nBEST stack (likeness {best['mean_likeness']}):\n  {_fmt_stack(best['weights'])}"
