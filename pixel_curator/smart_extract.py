#!/usr/bin/env python3
"""smart_extract.py - bridge between pixel-curator_nsfw_v2.toml and the
pixel_curator extraction engine.

This module reads the rich ``[extraction]`` section from an NSFW project TOML
(e.g. ``pixel-curator_nsfw_v2.toml``) and converts it into the ``ExtractConfig``
that ``pixel_curator.extract.pipeline_generator`` consumes — so the NSFW TOML
drives the **same** extraction engine that ``python -m pixel_curator extract``
uses (manifest tracking, resume, adaptive FPS, HDR tonemapping, segmented
extraction, photo import).

Usage from Python::

    from pixel_curator.smart_extract import build_config_from_toml, run
    run("/path/to/pixel-curator_nsfw_v2.toml", dry_run=False)

Usage from the CLI (registered as ``smart-extract``)::

    python -m pixel_curator -p pixel-curator_nsfw_v2.toml smart-extract
    python -m pixel_curator -p pixel-curator_nsfw_v2.toml smart-extract --dry-run

The TOML schema (all fields optional; sensible defaults inherited from
ExtractConfig)::

    video_dirs        = ["/path/to/videos", ...]          # top-level
    output_base       = "/path/to/output"                 # top-level
    video_extensions  = [".mp4", ".mov", ...]             # top-level

    [extraction]
    ffmpeg_bin   = "ffmpeg-full"
    walk_subdirs = true

    [extraction.deduplication]
    enabled          = true
    log_duplicates   = true

    [extraction.fps_strategy]
    mode    = "adaptive_duration"
    min_fps = 0.1
    max_fps = 10
    rules = [
        { max_seconds = 10,   fps = 8 },
        { max_seconds = 30,   fps = 4 },
        { max_seconds = 60,   fps = 2 },
        { max_seconds = 180,  fps = 1 },
        { max_seconds = 600,  fps = 0.5 },
        { max_seconds = 1800, fps = 0.25 },
        { fps = 0.1 },           # catch-all (no max_seconds = infinity)
    ]

    [extraction.resolution]
    target_width  = 1024
    target_height = 0           # 0 = preserve aspect ratio

    [extraction.hdr]
    detect_hdr      = true
    on_hdr_detected = "tonemap_to_sdr"
    # tonemap_filter is honored: the engine uses zscale+tonemap when is_hdr;
    # if the TOML supplies a custom filter chain, it overrides the engine's
    # default tonemap chain.

    [extraction.flags]
    threads     = 0
    hwaccel     = "auto"        # "auto" | "videotoolbox" | "none"
    skip_corrupt = true
    overwrite   = false
    log_level   = "warning"
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .config import ExtractConfig, Project, DEFAULT_FPS_TIERS, load_project
from .extract import pipeline_generator
from .util import discover_videos

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TOML → ExtractConfig converter
# ---------------------------------------------------------------------------

def _load_toml_raw(path: str | Path) -> dict:
    """Load a TOML file and return the raw parsed dict (all sections)."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Project file not found: {p}")
    try:
        import tomllib
    except ImportError:  # Python 3.10
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            # Fall back to the studio's flat parser (handles simple schemas)
            from .config import _parse_flat_toml
            return _parse_flat_toml(p.read_text(encoding="utf-8"))
    with p.open("rb") as fh:
        return tomllib.load(fh)


def _convert_fps_rules(rules: list[dict], min_fps: float = 0.1
                       ) -> tuple[tuple[float, float, int], ...]:
    """Convert TOML fps_strategy.rules → ExtractConfig.fps_tiers format.

    TOML rule shape: ``{ max_seconds = 10, fps = 8 }`` (max_seconds optional;
    absent = infinity).

    ExtractConfig.fps_tiers shape: ``(max_seconds, fps, frame_cap)`` where
    frame_cap is a per-tier ceiling. The TOML doesn't specify caps, so we
    derive a sensible default: ``fps * max_seconds * 1.5`` (rounded up),
    which gives ~50% headroom over the nominal frame count. For the
    catch-all tier (no max_seconds), we use a large cap.
    """
    if not rules:
        return DEFAULT_FPS_TIERS

    tiers: list[tuple[float, float, int]] = []
    for rule in rules:
        fps = float(rule.get("fps", min_fps))
        max_sec = rule.get("max_seconds")
        if max_sec is None:
            # catch-all tier → infinity
            max_dur = float("inf")
            cap = 6000  # matches the engine's xlong default cap
        else:
            max_dur = float(max_sec)
            # nominal frames at this tier = fps * max_dur; add 50% headroom
            cap = int(fps * max_dur * 1.5) if max_dur > 0 else 120
            cap = max(cap, 30)  # floor so very short tiers still yield frames
        tiers.append((max_dur, fps, cap))

    # Sort ascending by max_duration so select_fps_tier() works correctly
    finite = [t for t in tiers if t[0] != float("inf")]
    infinite = [t for t in tiers if t[0] == float("inf")]
    finite.sort(key=lambda t: t[0])
    return tuple(finite + infinite)


def build_config_from_toml(
        toml_path: str | Path,
        *,
        dry_run: bool = False,
        overwrite: bool = False,
        limit_videos: int = 0,
        ) -> tuple[Project, ExtractConfig]:
    """Build a (Project, ExtractConfig) pair from an NSFW TOML.

    The Project carries video_dirs / output_base / trigger_token (top-level
    TOML fields). The ExtractConfig carries the per-extraction knobs from the
    ``[extraction]`` section, converted to the engine's native format.
    """
    data = _load_toml_raw(toml_path)
    prj = load_project(str(toml_path))

    # Make sure output_base is set (Project defaults to a fallback if the
    # TOML didn't specify one)
    output_base = Path(prj.output_base).expanduser()

    ext = data.get("extraction", {})
    fps_strat = ext.get("fps_strategy", {})
    resolution = ext.get("resolution", {})
    flags = ext.get("flags", {})
    dedup = ext.get("deduplication", {})

    # Build fps_tiers from the TOML's rules (or fall back to engine defaults)
    fps_tiers = _convert_fps_rules(
        fps_strat.get("rules", []),
        min_fps=float(fps_strat.get("min_fps", 0.1)),
    )

    # max_side: 0 = no scaling. If target_width is set, use it.
    max_side = int(resolution.get("target_width", 0) or 0)

    # hwaccel: "auto" → let the engine decide (it checks for videotoolbox).
    # "videotoolbox" → force on. "none" → force off.
    hwaccel = str(flags.get("hwaccel", "auto")).lower()

    cfg = ExtractConfig(
        output_base=output_base,
        fps=float(fps_strat.get("min_fps", 0.25)),  # fallback rate
        jpeg_quality=2,                              # engine default
        segment_seconds=300,                         # engine default
        use_personfromvid=False,
        import_photos=False,  # smart-extract is video-only; use `photos` for images
        photo_import_mode="hardlink",
        resume=True,          # always resume (skip completed videos)
        dry_run=dry_run,
        limit_videos=limit_videos,
        overwrite=overwrite or bool(flags.get("overwrite", False)),
        max_side=max_side,
        adaptive=True,        # the TOML's fps_strategy.rules IS adaptive
        fps_tiers=fps_tiers,
        max_frames_per_video=0,   # use per-tier caps
        min_frames_per_video=4,
    )

    # Stash the hwaccel preference + dedup flag on the config via a private
    # attribute so the engine can honor them. (The engine's `use_videotoolbox`
    # is computed from `check_dependencies()`; we override it below in `run()`.)
    cfg._nsfw_hwaccel = hwaccel  # type: ignore[attr-defined]
    cfg._nsfw_dedup = bool(dedup.get("enabled", True))  # type: ignore[attr-defined]
    cfg._nsfw_log_dupes = bool(dedup.get("log_duplicates", True))  # type: ignore[attr-defined]

    return prj, cfg


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(toml_path: str | Path,
        *,
        dry_run: bool = False,
        overwrite: bool = False,
        limit_videos: int = 0,
        ) -> int:
    """Run the NSFW smart extraction pipeline.

    Prints progress messages from the engine generator. Returns 0 on success,
    non-zero on error. This is the exact same engine ``python -m pixel_curator
    extract`` uses — the only difference is the config source (the NSFW TOML's
    ``[extraction]`` section instead of CLI flags).
    """
    prj, cfg = build_config_from_toml(
        toml_path, dry_run=dry_run, overwrite=overwrite,
        limit_videos=limit_videos,
    )

    video_dirs = prj.video_dirs or []
    if not video_dirs:
        print("ERROR: no video_dirs configured in the TOML")
        return 1

    print(f"=== Smart NSFW Extraction (pixel_curator engine) ===")
    print(f"  project    : {Path(toml_path).expanduser()}")
    print(f"  output_base: {cfg.output_base}")
    print(f"  video_dirs : {len(video_dirs)} director(ies)")
    for d in video_dirs:
        print(f"               - {d}")
    print(f"  fps_tiers  : {len(cfg.fps_tiers)} tier(s) from TOML")
    for max_dur, fps, cap in cfg.fps_tiers:
        dur_label = "inf" if max_dur == float("inf") else f"{max_dur:g}s"
        print(f"               - <= {dur_label}: {fps} fps (cap {cap})")
    print(f"  max_side   : {cfg.max_side or '(no scaling)'}")
    print(f"  dry_run    : {cfg.dry_run}")
    print(f"  overwrite  : {cfg.overwrite}")
    print()

    if dry_run:
        print("[DRY RUN] No frames will be written. The engine will report")
        print("          what it WOULD extract per video.\n")

    # Count discovered videos (for the header). The engine also does this
    # internally, but we do it here too so the user sees the count before
    # the generator starts yielding.
    try:
        all_videos = discover_videos(video_dirs)
        if limit_videos > 0:
            all_videos = all_videos[:limit_videos]
        print(f"Discovered {len(all_videos)} video file(s).\n")
    except Exception as exc:
        log.warning("video discovery failed: %s", exc)

    # Drive the engine. Each `yield` from pipeline_generator is a progress
    # message; we print it and continue.
    try:
        for update in pipeline_generator(video_dirs, "", cfg):
            print(update)
    except Exception as exc:
        print(f"\nFATAL: {exc}")
        return 1

    print(f"\nNext: python -m pixel_curator -p {Path(toml_path).name} curate --smart")
    return 0
