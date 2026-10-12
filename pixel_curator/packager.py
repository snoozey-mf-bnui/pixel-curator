"""Dataset packaging: export curated frames into kohya/Forge-ready folders.

Phase 2 upgrades:
  - composition quotas across framing classes (closeup/portrait/upper/full)
  - diversity-aware selection: round-robin across CLIP clusters
  - per-frame WD14 captions from the captions table (fallback: static caption)
"""

from __future__ import annotations

import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Generator, Optional

from . import manifest
from .config import PackageConfig
from .util import now_iso, safe_slug, setup_logging


# -----------------------------
# Selection helpers (pure, testable)
# -----------------------------

def parse_quota(spec: str) -> dict[str, float]:
    """'closeup=0.3,portrait=0.3' -> {'closeup': 0.3, ...} (normalized)."""
    out: dict[str, float] = {}
    for part in (spec or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def allocate_quota(avail: dict[str, int], total: int,
                   quota: dict[str, float]) -> dict[str, int]:
    """Distribute `total` picks across framing buckets honoring quota ratios,
    redistributing shortfalls to buckets that still have frames."""
    alloc = {b: min(int(round(total * q)), avail.get(b, 0)) for b, q in quota.items()}
    # include non-quota buckets (e.g. 'none' / unscanned) as overflow targets
    remaining = total - sum(alloc.values())
    while remaining > 0:
        progressed = False
        for b in sorted(avail, key=lambda b: -(avail[b] - alloc.get(b, 0))):
            if alloc.get(b, 0) < avail[b]:
                alloc[b] = alloc.get(b, 0) + 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return {b: n for b, n in alloc.items() if n > 0}


def round_robin_clusters(frames: list, limit: int) -> list:
    """frames: rows whose [4] is cluster_id and [3] is sharpness.
    Interleave clusters (each ordered sharpest-first) for maximum diversity."""
    groups: dict = defaultdict(list)
    for f in frames:
        groups[f[4] if f[4] is not None else -1].append(f)
    for g in groups.values():
        g.sort(key=lambda r: -(r[3] or 0.0))
    order = sorted(groups, key=lambda c: -len(groups[c]))
    out: list = []
    i = 0
    while len(out) < limit:
        advanced = False
        for c in order:
            if i < len(groups[c]):
                out.append(groups[c][i])
                advanced = True
                if len(out) >= limit:
                    break
        if not advanced:
            break
        i += 1
    return out


# -----------------------------
# Packaging
# -----------------------------

def package_generator(cfg: PackageConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)

    rows = conn.execute(
        "SELECT f.frame_id, f.source_id, f.path, f.sharpness, f.cluster_id, "
        "d.framing FROM frames f "
        "LEFT JOIN detections d ON d.frame_id = f.frame_id "
        "WHERE f.status IN ('selected','packaged') "
        "ORDER BY f.source_id, f.sharpness DESC"
    ).fetchall()
    if not rows:
        yield "No selected frames. Run curate first."
        return

    has_framing = any(r[5] for r in rows)
    has_clusters = any(r[4] is not None for r in rows)
    quota = parse_quota(cfg.quota) if (cfg.quota and has_framing) else {}

    # 1) per-video cap with cluster-aware diversity inside each video
    per_source: dict[str, list] = defaultdict(list)
    for r in rows:
        per_source[r[1]].append(r)
    capped: list = []
    for sid, frames in per_source.items():
        cap = cfg.max_per_video if cfg.max_per_video > 0 else len(frames)
        if has_clusters:
            capped.extend(round_robin_clusters(frames, cap))
        else:
            capped.extend(frames[:cap])

    # 2) global framing quotas (needs a target total)
    mode = "legacy (sharpness)"
    if quota and cfg.max_total > 0:
        buckets: dict[str, list] = defaultdict(list)
        for r in capped:
            buckets[r[5] or "none"].append(r)
        avail = {b: len(fs) for b, fs in buckets.items()}
        alloc = allocate_quota(avail, min(cfg.max_total, len(capped)), quota)
        chosen = []
        for b, n in alloc.items():
            frames = buckets.get(b, [])
            chosen.extend(
                round_robin_clusters(frames, n) if has_clusters else frames[:n]
            )
        mode = f"quota {alloc}"
    elif cfg.max_total > 0:
        capped.sort(key=lambda r: -(r[3] or 0.0))
        chosen = capped[: cfg.max_total]
        mode = "sharpness top-N"
    else:
        chosen = capped
        mode = ("cluster round-robin per video" if has_clusters
                else "legacy (sharpness per video)")

    # captions lookup
    captions: dict[str, str] = {}
    if cfg.use_caption_table:
        captions = dict(conn.execute(
            "SELECT frame_id, caption_text FROM captions "
            "WHERE caption_text IS NOT NULL AND caption_text != ''"
        ).fetchall())

    static_caption = cfg.caption_text.strip() or f"{cfg.token} {cfg.class_word}".strip()
    folder = f"{cfg.repeats}_{safe_slug(cfg.token)} {safe_slug(cfg.class_word)}".strip()
    img_dir = cfg.output_base / cfg.dataset_name / "img" / folder
    img_dir.mkdir(parents=True, exist_ok=True)

    framing_mix: dict[str, int] = defaultdict(int)
    cluster_mix: set = set()
    for r in chosen:
        framing_mix[r[5] or "none"] += 1
        cluster_mix.add(r[4])

    yield (
        f"Packaging {len(chosen)} frames from {len(per_source)} sources\n"
        f"Selection mode: {mode}\n"
        f"Framing mix: {dict(framing_mix)}\n"
        f"Clusters represented: {len(cluster_mix)}\n"
        f"Captions: {len(captions)} per-frame (WD14) + static fallback "
        f"'{static_caption}'\n-> {img_dir}\n"
    )

    packaged = 0
    captioned_individually = 0
    errors = 0
    for frame_id, sid, path_str, _sharp, _cl, _fr in chosen:
        src = Path(path_str)
        if not src.exists():
            errors += 1
            continue
        dest = img_dir / f"{sid}_{src.name}"
        try:
            if not dest.exists():
                if cfg.link_mode == "hardlink":
                    try:
                        os.link(src, dest)
                    except OSError:
                        shutil.copy2(src, dest)
                else:
                    shutil.copy2(src, dest)
            if cfg.write_captions:
                text = captions.get(frame_id, static_caption)
                if frame_id in captions:
                    captioned_individually += 1
                dest.with_suffix(".txt").write_text(text, encoding="utf-8")
            conn.execute(
                "UPDATE frames SET status='packaged' WHERE frame_id=?", (frame_id,)
            )
            packaged += 1
            if packaged % 250 == 0:
                conn.commit()
                yield f"Packaged {packaged}/{len(chosen)}"
        except Exception as exc:
            errors += 1
            logging.error("Package failed for %s: %s", src, exc)
    conn.commit()

    notes = cfg.output_base / cfg.dataset_name / "README.txt"
    notes.write_text(
        f"""LoRA training dataset - generated {now_iso()}
Images: {packaged}
Selection: {mode}
Framing mix: {dict(framing_mix)}
Clusters represented: {len(cluster_mix)}
Per-frame WD14 captions: {captioned_individually} (rest: '{static_caption}')
Structure: img/{folder}/   (kohya_ss style: <repeats>_<token> <class>)

kohya_ss: point 'Image folder' at: {cfg.output_base / cfg.dataset_name / 'img'}
Suggested starting params (SDXL, character LoRA):
  network_dim 32, alpha 16, lr 1e-4 (unet) / 5e-5 (TE), cosine,
  batch 1-2, {cfg.repeats} repeats x ~10 epochs, min_snr_gamma 5.
""",
        encoding="utf-8",
    )

    # Phase 10: ready-to-train kohya config beside the package. Suggestion
    # layer only - failure never breaks packaging.
    tc_lines = ""
    if getattr(cfg, "training_config", True) and packaged:
        from .train.export_config import generate_for_export
        _, tc_msgs = generate_for_export(
            conn, cfg.output_base / cfg.dataset_name,
            frame_count=packaged, model_type="lora",
            output_name=cfg.dataset_name)
        tc_lines = "".join(f"{m}\n" for m in tc_msgs)

    yield (
        "\nPACKAGING DONE.\n"
        f"Packaged: {packaged} (per-frame captions: {captioned_individually})\n"
        f"Errors: {errors}\n"
        f"Dataset: {img_dir}\n"
        f"Notes: {notes}\n"
        f"{tc_lines}"
    )
