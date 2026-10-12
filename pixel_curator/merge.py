"""Phase 7: Merging Lab - weighted + block-weight LoRA merging.

Method: kohya-style CONCAT merge. For each weight pair the scaled matrices
are concatenated along the rank axis, so LoRAs of *different* ranks merge
correctly (output rank = sum of input ranks) and the result is an ordinary
LoRA usable in Forge/Draw Things/diffusers.

For each input LoRA i with user weight w_i and stored alpha_i, rank_i:
    s_i  = w_i * (alpha_i / rank_i) * block_multiplier(key)
    down'_i = down_i * sqrt(|s_i|)
    up'_i   = up_i   * sqrt(|s_i|) * sign(s_i)        <- negative weights OK
    merged_down = concat(down'_i, axis=0)   merged_up = concat(up'_i, axis=1)
    merged_alpha = merged_rank   (scale folded in -> effective scale 1.0)

Block weights (coarse, practical groups):
    te    - text encoder modules        (identity/wording strength)
    down  - UNet down blocks            (composition / global structure)
    mid   - UNet mid block
    up    - UNet up blocks              (texture / detail / style surface)

Combo creator: one call merges character + style + outfit stacks into a
single deployable LoRA, e.g. character 1.0 with style 0.4 up-blocks-only.

Backend: safetensors.numpy + numpy only (no torch needed for merging).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

from . import manifest
from .util import now_iso, safe_slug, setup_logging


@dataclass
class MergeConfig:
    output_base: Path
    loras: list[tuple[str, float]] = field(default_factory=list)
    output_name: str = "merged"
    block_weights: dict = field(default_factory=dict)  # per-input overrides: {path: {te,down,mid,up}}
    default_blocks: dict = field(default_factory=lambda: {"te": 1.0, "down": 1.0,
                                                          "mid": 1.0, "up": 1.0})
    out_dir: str = ""                                  # default: lora_output_dir


def merge_available() -> tuple[bool, str]:
    try:
        import numpy  # noqa: F401
        import safetensors.numpy  # noqa: F401
        return True, "ok"
    except Exception as exc:
        return False, f"needs numpy + safetensors: {exc}"


# -----------------------------
# Key analysis (pure, testable)
# -----------------------------

def block_group(key: str) -> str:
    """Map a kohya LoRA key to a coarse block group."""
    k = key.lower()
    if k.startswith("lora_te"):
        return "te"
    if "down_blocks" in k or "input_blocks" in k:
        return "down"
    if "mid_block" in k or "middle_block" in k:
        return "mid"
    if "up_blocks" in k or "output_blocks" in k:
        return "up"
    return "mid"  # conservative default for odd keys


def module_bases(keys) -> set[str]:
    """Distinct module prefixes having .lora_down/.lora_up pairs."""
    bases = set()
    for k in keys:
        if k.endswith(".lora_down.weight"):
            bases.add(k[: -len(".lora_down.weight")])
    return bases


def parse_block_string(spec: str) -> dict:
    """'te=0,down=1,mid=1,up=0.5' -> dict (missing keys default 1.0)."""
    out = {"te": 1.0, "down": 1.0, "mid": 1.0, "up": 1.0}
    for part in (spec or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip().lower()
            if k in out:
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
    return out


# -----------------------------
# Merge core
# -----------------------------

def merge_state_dicts(dicts: list[dict], weights: list[float],
                      blocks: list[dict]) -> tuple[dict, dict]:
    """Concat-merge a list of kohya LoRA state dicts (numpy arrays).
    Returns (merged_state_dict, stats)."""
    import numpy as np

    merged: dict = {}
    stats = {"modules": 0, "skipped_unmatched": 0, "zeroed": 0}

    all_bases: list[str] = sorted(set().union(*[module_bases(d.keys()) for d in dicts]))
    for base in all_bases:
        downs, ups = [], []
        for d, w, blk in zip(dicts, weights, blocks):
            dk, uk, ak = (f"{base}.lora_down.weight", f"{base}.lora_up.weight",
                          f"{base}.alpha")
            if dk not in d or uk not in d:
                continue
            down = np.asarray(d[dk], dtype=np.float32)
            up = np.asarray(d[uk], dtype=np.float32)
            rank = down.shape[0]
            alpha = float(np.asarray(d.get(ak, rank)).reshape(-1)[0]) if ak in d else float(rank)
            s = w * (alpha / max(1, rank)) * float(blk.get(block_group(base), 1.0))
            if s == 0.0:
                stats["zeroed"] += 1
                continue
            root = float(np.sqrt(abs(s)))
            downs.append(down * root)
            ups.append(up * root * (1.0 if s > 0 else -1.0))
        if not downs:
            stats["skipped_unmatched"] += 1
            continue
        m_down = np.concatenate(downs, axis=0)
        m_up = np.concatenate(ups, axis=1)
        new_rank = m_down.shape[0]
        merged[f"{base}.lora_down.weight"] = m_down.astype(np.float16)
        merged[f"{base}.lora_up.weight"] = m_up.astype(np.float16)
        merged[f"{base}.alpha"] = np.asarray(float(new_rank), dtype=np.float16)
        stats["modules"] += 1
    return merged, stats


def merge_generator(prj, cfg: MergeConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    ok, reason = merge_available()
    if not ok:
        yield f"Merge unavailable: {reason}"
        return
    from safetensors.numpy import load_file, save_file

    if len(cfg.loras) < 2:
        yield "FATAL: need at least 2 LoRAs to merge."
        return

    dicts, weights, blocks, names = [], [], [], []
    for path, w in cfg.loras:
        p = Path(path).expanduser()
        if not p.exists():
            yield f"FATAL: not found: {p}"
            return
        dicts.append(load_file(str(p)))
        weights.append(float(w))
        blocks.append(cfg.block_weights.get(path, cfg.default_blocks))
        names.append(p.stem)
        yield f"  loaded {p.name} (w={w}, blocks={blocks[-1]})"

    yield "Merging (concat method - mixed ranks supported)..."
    merged, stats = merge_state_dicts(dicts, weights, blocks)
    if not merged:
        yield "FATAL: no mergeable modules found (are these kohya-format LoRAs?)"
        return

    out_dir = Path(cfg.out_dir or prj.lora_output_dir
                   or (Path(prj.output_base) / "LORA_OUTPUT")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_slug(cfg.output_name)}.safetensors"
    meta = {
        "pixel_curator_merge": json.dumps({
            "inputs": [{"name": n, "weight": w, "blocks": b}
                       for n, w, b in zip(names, weights, blocks)],
            "method": "concat", "created_at": now_iso(),
        })
    }
    save_file(merged, str(out_path), metadata=meta)

    # registry note in manifest
    conn = manifest.connect(cfg.output_base)
    manifest.meta_set(conn, f"merge:{cfg.output_name}", meta["pixel_curator_merge"])

    yield (
        f"\nMERGE DONE -> {out_path}\n"
        f"Modules merged: {stats['modules']} | skipped: {stats['skipped_unmatched']} "
        f"| zeroed-out: {stats['zeroed']}\n"
        "Test it: pixel-curator matrix --lora "
        f"\"{out_path}\" --label {safe_slug(cfg.output_name)}"
    )


def combo_generator(prj, output_base: Path, character: str, style: str = "",
                    outfit: str = "", name: str = "combo",
                    char_w: float = 1.0, style_w: float = 0.4,
                    outfit_w: float = 0.6) -> Generator[str, None, None]:
    """One-click 'final character' merge: identity at full strength, style
    surface-only (up blocks), outfit without TE interference."""
    loras: list[tuple[str, float]] = [(character, char_w)]
    block_weights: dict = {}
    if style:
        loras.append((style, style_w))
        block_weights[style] = {"te": 0.0, "down": 0.3, "mid": 0.5, "up": 1.0}
    if outfit:
        loras.append((outfit, outfit_w))
        block_weights[outfit] = {"te": 0.0, "down": 1.0, "mid": 1.0, "up": 1.0}
    cfg = MergeConfig(
        output_base=output_base, loras=loras, output_name=name,
        block_weights=block_weights,
    )
    yield from merge_generator(prj, cfg)
