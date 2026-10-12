"""Phase 8.1: Best-Epoch Sweep - stop guessing which checkpoint to ship.

kohya saves one checkpoint per epoch (name-000001.safetensors ... name.safetensors).
The sweep runs the eval matrix on EVERY epoch, aggregates anchor-likeness and
flexibility per epoch, and recommends the best one:

    maximize likeness, subject to overfit guard
    (likeness - flexibility gap must stay under `max_gap`)

Resumable: epochs whose eval label already has rows are skipped, so an
interrupted sweep continues where it stopped. Results are stored in the
manifest meta (`sweep:<run>`) and printed as a table.

    pixel-curator sweep --run tok_character_v001
    pixel-curator sweep --run tok_character_v001 --backend diffusers
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

from . import manifest
from .config import Project
from .util import now_iso, setup_logging

EPOCH_RE = re.compile(r"-(\d{6})$")


@dataclass
class SweepConfig:
    output_base: Path
    run: str                                 # run name (checkpoint stem prefix)
    backend: str = "forge"
    forge_url: str = "http://127.0.0.1:7860"
    checkpoint: str = ""
    weight: float = 0.85
    categories: list[str] = field(default_factory=lambda: ["likeness", "flexibility"])
    seeds: list[int] = field(default_factory=lambda: [42, 1042])
    max_gap: float = 0.15                    # overfit guard
    limit_epochs: int = 0                    # 0 = all
    generate_fn: Optional[object] = None     # test hook (passed to matrix)


def epoch_number(stem: str) -> Optional[int]:
    """'name-000004' -> 4; final 'name' (no suffix) -> None (means last)."""
    m = EPOCH_RE.search(stem)
    return int(m.group(1)) if m else None


def find_epoch_files(lora_dir: Path, run: str) -> list[tuple[int, Path]]:
    """All checkpoints for a run, ordered by epoch (final file = highest+1)."""
    files = sorted(lora_dir.glob(f"{run}*.safetensors"))
    epochs: list[tuple[int, Path]] = []
    finals: list[Path] = []
    for f in files:
        if f.stem == run:
            finals.append(f)
            continue
        n = epoch_number(f.stem)
        if n is not None and f.stem == f"{run}-{n:06d}":
            epochs.append((n, f))
    epochs.sort()
    last = (epochs[-1][0] if epochs else 0) + 1
    for f in finals:
        epochs.append((last, f))
    return epochs


def epoch_stats(conn, label: str) -> dict:
    """Aggregate eval scores for one label -> {category: avg, 'n': count}."""
    out: dict = {"n": 0}
    for cat, n, avg in conn.execute(
        "SELECT category, COUNT(*), AVG(likeness) FROM evals "
        "WHERE label = ? GROUP BY category", (label,),
    ):
        out[cat] = round(avg, 4) if avg is not None else None
        out["n"] += n
    return out


def recommend_epoch(rows: list[dict], max_gap: float = 0.15) -> Optional[dict]:
    """rows: [{epoch, label, likeness, flexibility}]. Best = highest likeness
    whose likeness-flexibility gap is acceptable; falls back to highest
    likeness overall if every epoch is overfit (with a warning flag)."""
    scored = [r for r in rows if r.get("likeness") is not None]
    if not scored:
        return None
    ok = [r for r in scored
          if r.get("flexibility") is None
          or (r["likeness"] - r["flexibility"]) <= max_gap]
    pool = ok or scored
    best = max(pool, key=lambda r: r["likeness"])
    return {**best, "overfit_warning": not ok}


def sweep_generator(prj: Project, cfg: SweepConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)

    lora_dir = Path(prj.lora_output_dir
                    or (Path(prj.output_base) / "LORA_OUTPUT")).expanduser()
    epochs = find_epoch_files(lora_dir, cfg.run)
    if not epochs:
        yield (f"FATAL: no checkpoints matching '{cfg.run}*' in {lora_dir}. "
               "Train first, or check the run name (pixel-curator runs).")
        return
    if cfg.limit_epochs > 0:
        epochs = epochs[-cfg.limit_epochs:]

    yield (f"BEST-EPOCH SWEEP '{cfg.run}' - {len(epochs)} checkpoints, "
           f"categories {cfg.categories}, seeds {cfg.seeds}\n")

    from .eval.matrix import MatrixConfig, matrix_generator
    rows: list[dict] = []
    for epoch, path in epochs:
        label = path.stem
        existing = conn.execute(
            "SELECT COUNT(*) FROM evals WHERE label = ?", (label,)
        ).fetchone()[0]
        if existing > 0:
            yield f"[epoch {epoch}] {label}: already evaluated ({existing} rows) - skip"
        else:
            yield f"[epoch {epoch}] evaluating {path.name}..."
            mcfg = MatrixConfig(
                output_base=cfg.output_base, lora=str(path), label=label,
                categories=cfg.categories, seeds=cfg.seeds,
                backend=cfg.backend, forge_url=cfg.forge_url,
                checkpoint=cfg.checkpoint or prj.base_model,
                lora_weight=cfg.weight, generate_fn=cfg.generate_fn,
            )
            for update in matrix_generator(prj, mcfg):
                yield f"  {update}"
        s = epoch_stats(conn, label)
        rows.append({"epoch": epoch, "label": label,
                     "likeness": s.get("likeness"),
                     "flexibility": s.get("flexibility"), "n": s["n"]})

    # report
    yield "\nEPOCH SCOREBOARD"
    yield f"{'epoch':>6} {'likeness':>9} {'flexibility':>12} {'gap':>7}  label"
    for r in sorted(rows, key=lambda x: x["epoch"]):
        lk = r["likeness"]
        fx = r["flexibility"]
        gap = (f"{lk - fx:+.3f}" if lk is not None and fx is not None else "-")
        yield (f"{r['epoch']:>6} "
               f"{lk if lk is not None else '-':>9} "
               f"{fx if fx is not None else '-':>12} {gap:>7}  {r['label']}")

    best = recommend_epoch(rows, cfg.max_gap)
    if best is None:
        yield ("\nNo likeness scores recorded - build an identity anchor "
               "(pixel-curator anchor) and install [ai] extras, then re-run; "
               "evaluated images are cached so the re-score is instant.")
        manifest.meta_set(conn, f"sweep:{cfg.run}",
                          {"rows": rows, "best": None, "at": now_iso()})
        return

    warn = ("\nNOTE: every epoch exceeds the overfit gap - consider fewer "
            "epochs/repeats or more flexible data." if best["overfit_warning"] else "")
    manifest.meta_set(conn, f"sweep:{cfg.run}",
                      {"rows": rows, "best": best, "at": now_iso()})
    yield (
        f"\nRECOMMENDED EPOCH: {best['epoch']}  ({best['label']})\n"
        f"likeness {best['likeness']} | flexibility {best['flexibility']} "
        f"| evaluated images cached in evals/{warn}\n"
        f"\nShip it: {lora_dir / (best['label'] + '.safetensors')}\n"
        f"Or merge it: pixel-curator combo --character \"{lora_dir / (best['label'] + '.safetensors')}\" ..."
    )


def sweep_summary(conn, run: str) -> Optional[dict]:
    return manifest.meta_get(conn, f"sweep:{run}")
