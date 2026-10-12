"""Frame extraction stage: ffprobe metadata, HDR-aware ffmpeg extraction,
segmented + resumable, optional personfromvid post-stage, photo import,
and the extraction pipeline generator."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

from . import manifest
from .config import DB_NAME, LOG_NAME, ExtractConfig, PrefilterConfig
from .util import (
    check_dependencies, discover_photos, discover_videos, ffmpeg_has_filter,
    ffmpeg_vsync_args, human_bytes, parse_fps, run_cmd, safe_slug,
    setup_logging, stable_id, which,
)


@dataclass
class VideoMeta:
    path: Path
    duration: float
    size_bytes: int
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    codec: Optional[str]
    pix_fmt: Optional[str]
    color_transfer: Optional[str]
    color_primaries: Optional[str]
    color_space: Optional[str]
    is_hdr: bool


# -----------------------------
# ffprobe
# -----------------------------

def ffprobe_json(video_path: Path) -> dict:
    proc = run_cmd([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path),
    ])
    return json.loads(proc.stdout)


def get_video_meta(video_path: Path) -> VideoMeta:
    info = ffprobe_json(video_path)
    fmt = info.get("format", {})
    vstream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if vstream is None:
        raise RuntimeError("No video stream found")

    duration = float(vstream.get("duration") or fmt.get("duration") or 0.0)
    size_bytes = int(fmt.get("size") or video_path.stat().st_size)
    color_transfer = vstream.get("color_transfer")
    color_primaries = vstream.get("color_primaries")
    color_space = vstream.get("color_space")
    pix_fmt = vstream.get("pix_fmt")
    haystack = " ".join(
        str(x or "").lower()
        for x in (color_transfer, color_primaries, color_space, pix_fmt)
    )
    is_hdr = any(
        tok in haystack
        for tok in ("smpte2084", "arib-std-b67", "bt2020", "yuv420p10le", "p010le")
    )
    return VideoMeta(
        path=video_path, duration=duration, size_bytes=size_bytes,
        width=vstream.get("width"), height=vstream.get("height"),
        fps=parse_fps(vstream.get("avg_frame_rate") or vstream.get("r_frame_rate")),
        codec=vstream.get("codec_name"), pix_fmt=pix_fmt,
        color_transfer=color_transfer, color_primaries=color_primaries,
        color_space=color_space, is_hdr=is_hdr,
    )


# -----------------------------
# Extraction
# -----------------------------

def build_video_output_dir(output_base: Path, video_path: Path) -> Path:
    sid = stable_id(video_path)
    stem = safe_slug(video_path.stem)
    ext = video_path.suffix.lower().lstrip(".")
    return output_base / "frames" / f"{stem}__{ext}__{sid}"


def count_jpegs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*.jpg") if p.is_file())


def build_filter(meta: VideoMeta, fps: float, max_side: int = 0) -> str:
    parts: list[str] = []
    if meta.is_hdr and ffmpeg_has_filter("zscale") and ffmpeg_has_filter("tonemap"):
        parts.extend([
            "zscale=t=linear:npl=100",
            "format=gbrpf32le",
            "zscale=p=bt709",
            "tonemap=tonemap=hable:desat=0",
            "zscale=t=bt709:m=bt709:r=tv",
            "format=yuv420p",
        ])
    else:
        if meta.is_hdr:
            logging.warning(
                "HDR source but ffmpeg lacks zscale/tonemap; output may look washed out: %s",
                meta.path.name,
            )
        parts.append("format=yuv420p")
    parts.append(f"fps={fps}")
    if max_side and max_side > 0:
        parts.append(
            f"scale='if(gt(iw,ih),{max_side},-2)':'if(gt(ih,iw),{max_side},-2)'"
        )
    return ",".join(parts)


def select_fps_tier(duration: float, cfg: ExtractConfig) -> tuple[float, int]:
    """Return ``(fps, max_frames)`` for the tier matching ``duration``.

    Walks ``cfg.fps_tiers`` (sorted ascending by max-duration) and returns the
    first tier whose ceiling is >= duration. Falls back to the base ``fps`` with
    no per-tier cap if no tier matches.
    """
    for max_dur, tier_fps, tier_cap in cfg.fps_tiers:
        if duration <= max_dur:
            return tier_fps, tier_cap
    return cfg.fps, 0


def compute_adaptive_fps(meta: VideoMeta, cfg: ExtractConfig) -> float:
    """Choose a sampling frame rate for one clip from its duration.

    The clip's duration selects a tier from ``cfg.fps_tiers`` (live / short /
    medium / long / extra-long), giving a target fps and a per-tier frame
    ceiling. The rate is then bounded so the clip yields between
    ``min_frames_per_video`` frames and the tier ceiling, and never exceeds the
    source frame rate. With ``adaptive=False`` the uniform base ``fps`` is
    returned for every clip (legacy behaviour).
    """
    base = max(cfg.fps, 0.01)
    duration = meta.duration if (meta.duration and meta.duration > 0) else 0.0
    if not cfg.adaptive or duration <= 0:
        return base

    fps, tier_cap = select_fps_tier(duration, cfg)

    # A global --max-frames-per-video (>0) overrides the tier's own ceiling.
    cap = cfg.max_frames_per_video if cfg.max_frames_per_video > 0 else tier_cap

    # Never sample faster than the source frame rate.
    if meta.fps and meta.fps > 0:
        fps = min(fps, meta.fps)

    # Ceiling: a long clip cannot explode past its frame cap.
    if cap and cap > 0 and fps * duration > cap:
        fps = cap / duration

    # Floor: a very short clip should still yield a minimum number of frames.
    if cfg.min_frames_per_video > 0 and fps * duration < cfg.min_frames_per_video:
        fps = cfg.min_frames_per_video / duration
        if meta.fps and meta.fps > 0:
            fps = min(fps, meta.fps)

    return max(fps, 0.01)


def extract_segment_ffmpeg(
    video_path: Path,
    output_dir: Path,
    meta: VideoMeta,
    fps: float,
    jpeg_quality: int,
    start_sec: float,
    duration_sec: float,
    segment_index: int,
    use_videotoolbox: bool,
    dry_run: bool,
    max_side: int = 0,
) -> int:
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = output_dir / f"seg{segment_index:04d}_frame_%06d.jpg"
    vf = build_filter(meta, fps, max_side)

    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    if use_videotoolbox and not meta.is_hdr:  # VTB + tonemap conflict on some builds
        cmd.extend(["-hwaccel", "videotoolbox"])
    cmd.extend([
        "-ss", f"{start_sec:.3f}",
        "-i", str(video_path),
        "-t", f"{duration_sec:.3f}",
        "-map", "0:v:0",
        "-an",  # ignore audio: Live Photo MOVs carry odd PCM tracks that
                # otherwise make ffmpeg exit non-zero on a video-only job
        "-vf", vf,
        "-q:v", str(jpeg_quality),
        *ffmpeg_vsync_args(),
        str(out_pattern),
    ])

    if dry_run:
        logging.info("[DRY RUN] %s", " ".join(cmd))
        return 0
    before = count_jpegs(output_dir)
    proc = run_cmd(cmd, check=False)
    produced = max(0, count_jpegs(output_dir) - before)
    if proc.returncode != 0:
        # Keep whatever frames were written; only treat as fatal if none came
        # out. Some containers report a non-zero exit after a clean decode.
        if produced == 0:
            raise RuntimeError((proc.stderr or "Unknown ffmpeg error")[-4000:])
        logging.warning(
            "ffmpeg exit %s but produced %d frame(s); keeping them: %s",
            proc.returncode, produced, video_path.name,
        )
    return produced


class ExtractionStopped(RuntimeError):
    """Raised when a should_stop callback halts extraction between
    segments (Curator Studio pause/stop). Frames already on disk are
    kept; segments are deterministic, so a re-run simply overwrites."""


def extract_video(
    conn: sqlite3.Connection,
    video_path: Path,
    cfg: ExtractConfig,
    use_videotoolbox: bool,
    should_stop=None,
) -> tuple[str, int]:
    sid = manifest.upsert_source(conn, video_path, "video")
    existing_status = manifest.get_status(conn, sid)
    out_dir = build_video_output_dir(cfg.output_base, video_path)

    if (cfg.resume and existing_status == "completed"
            and out_dir.exists() and not cfg.overwrite):
        frames = count_jpegs(out_dir)
        return f"SKIP completed: {video_path.name} ({frames} frames)", frames

    if cfg.overwrite and out_dir.exists() and not cfg.dry_run:
        shutil.rmtree(out_dir)
        conn.execute("DELETE FROM frames WHERE source_id = ?", (sid,))
        conn.commit()

    manifest.mark(conn, sid, "probing", output_dir=out_dir)
    meta = get_video_meta(video_path)
    meta_dict = {
        "duration": meta.duration, "size_bytes": meta.size_bytes,
        "width": meta.width, "height": meta.height, "fps": meta.fps,
        "codec": meta.codec, "pix_fmt": meta.pix_fmt,
        "color_transfer": meta.color_transfer,
        "color_primaries": meta.color_primaries,
        "color_space": meta.color_space, "is_hdr": meta.is_hdr,
    }
    manifest.mark(conn, sid, "extracting", output_dir=out_dir, meta=meta_dict)

    if meta.duration <= 0:
        raise RuntimeError("Video duration is zero or unavailable")

    effective_fps = compute_adaptive_fps(meta, cfg)
    manifest.event(
        conn, sid, "INFO",
        f"sampling fps={effective_fps:.3f} (duration={meta.duration:.2f}s, "
        f"adaptive={cfg.adaptive})",
    )

    seg_len = max(30, int(cfg.segment_seconds))
    # Phase G2: optional trim window. Defaults (0.0/0.0) keep the legacy
    # whole-video behaviour; segments are laid out inside [t0, t1) only.
    t0 = max(0.0, float(getattr(cfg, "trim_start", 0.0) or 0.0))
    t1 = float(getattr(cfg, "trim_end", 0.0) or 0.0)
    t1 = meta.duration if t1 <= 0 else min(t1, meta.duration)
    if t0 >= t1:
        t0, t1 = 0.0, meta.duration
    window = t1 - t0
    total_segments = int((window + seg_len - 1) // seg_len)

    for idx in range(total_segments):
        if should_stop is not None and should_stop():
            manifest.mark(conn, sid, "paused", output_dir=out_dir,
                          frames_extracted=count_jpegs(out_dir))
            manifest.event(conn, sid, "INFO",
                           f"paused after segment {idx}/{total_segments}")
            raise ExtractionStopped(
                f"stopped before segment {idx + 1}/{total_segments}")
        start = t0 + idx * seg_len
        duration = min(seg_len, t1 - start)
        if duration <= 0:
            continue
        new_count = extract_segment_ffmpeg(
            video_path=video_path, output_dir=out_dir, meta=meta,
            fps=effective_fps, jpeg_quality=cfg.jpeg_quality,
            start_sec=start, duration_sec=duration, segment_index=idx,
            use_videotoolbox=use_videotoolbox, dry_run=cfg.dry_run,
            max_side=cfg.max_side,
        )
        manifest.mark(conn, sid, "extracting", output_dir=out_dir,
                      frames_extracted=count_jpegs(out_dir))
        manifest.event(conn, sid, "INFO",
                       f"Segment {idx + 1}/{total_segments}: +{new_count} frames")

    total_frames = count_jpegs(out_dir)

    if not cfg.dry_run:
        rows = [
            (hashlib.sha1(str(p).encode()).hexdigest()[:16], sid, str(p))
            for p in sorted(out_dir.glob("*.jpg"))
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO frames(frame_id, source_id, path) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()

    if cfg.use_personfromvid:
        run_personfromvid(conn, sid, out_dir, cfg.dry_run)

    final_status = "dry_run" if cfg.dry_run else "completed"
    manifest.mark(conn, sid, final_status, output_dir=out_dir,
                  frames_extracted=total_frames)
    manifest.event(conn, sid, "INFO", f"{final_status}: {total_frames} frames")
    prefix = "DRY" if cfg.dry_run else "OK"
    return f"{prefix}: {video_path.name} -> {total_frames} frames", total_frames


def run_personfromvid(
    conn: sqlite3.Connection, source_id: str, frames_dir: Path, dry_run: bool
) -> None:
    if not which("personfromvid"):
        manifest.event(conn, source_id, "WARNING", "personfromvid not installed; skipping")
        return
    output_dir = frames_dir / "_personfromvid"
    candidates = [
        ["personfromvid", "--input", str(frames_dir), "--output", str(output_dir)],
        ["personfromvid", str(frames_dir), str(output_dir)],
    ]
    for cmd in candidates:
        if dry_run:
            logging.info("[DRY RUN] %s", " ".join(cmd))
            return
        proc = run_cmd(cmd, check=False)
        if proc.returncode == 0:
            manifest.event(conn, source_id, "INFO", "personfromvid completed")
            return
    manifest.event(conn, source_id, "WARNING", "personfromvid failed; extraction kept")


# -----------------------------
# Photo import
# -----------------------------

def photo_output_path(photo_out_root: Path, photo_path: Path) -> Path:
    sid = stable_id(photo_path)
    stem = safe_slug(photo_path.stem)
    return photo_out_root / f"{stem}__{sid}{photo_path.suffix.lower()}"


_EXIF_NORMALIZABLE = (".jpg", ".jpeg", ".png", ".webp")


def _photo_exif_orientation(path: Path) -> int:
    """EXIF orientation tag (1 = upright/none). Cheap; never raises.

    Delegates to the canonical reader in ``orient.py`` so there is a single
    EXIF-orientation source of truth across the pipeline."""
    try:
        from .orient import read_exif_orientation
        return read_exif_orientation(path)
    except Exception:
        return 1


def _write_exif_normalized(src: Path, dest: Path) -> bool:
    """Write `dest` as `src` with EXIF orientation baked into the pixels and the
    tag stripped (so stored frames are upright in raw pixels). Source untouched.
    Returns True on success."""
    try:
        from PIL import Image, ImageOps
        with Image.open(src) as im:
            fixed = ImageOps.exif_transpose(im)
            suffix = dest.suffix.lower()
            if suffix in (".jpg", ".jpeg"):
                fixed.convert("RGB").save(dest, format="JPEG", quality=95)
            elif suffix == ".webp":
                fixed.save(dest, format="WEBP", quality=95)
            else:
                fixed.save(dest)
        return True
    except Exception:
        return False


def import_one_photo(
    conn: sqlite3.Connection,
    photo_path: Path,
    output_base: Path,
    mode: str,
    resume: bool,
    dry_run: bool,
    prefilter_cfg: Optional[PrefilterConfig] = None,
) -> str:
    sid = manifest.upsert_source(conn, photo_path, "photo")
    status = manifest.get_status(conn, sid)
    photo_root = output_base / "photos_imported"
    dest = photo_output_path(photo_root, photo_path)

    if resume and status == "completed" and dest.exists():
        return f"SKIP photo: {photo_path.name}"
    if dry_run:
        manifest.event(conn, sid, "INFO", f"[DRY RUN] import photo -> {dest}")
        return f"DRY photo: {photo_path.name}"

    photo_root.mkdir(parents=True, exist_ok=True)
    try:
        if not dest.exists():
            # EXIF-at-ingest: if the photo carries a non-trivial orientation
            # tag, import it as a NORMALIZED real copy (upright raw pixels, tag
            # stripped) so no downstream tool has to wrestle EXIF. Source is
            # never modified. Already-upright photos keep the cheap link path.
            needs_norm = (dest.suffix.lower() in _EXIF_NORMALIZABLE
                          and _photo_exif_orientation(photo_path) != 1)
            if needs_norm and _write_exif_normalized(photo_path, dest):
                pass
            elif mode == "hardlink":
                try:
                    os.link(photo_path, dest)
                except OSError:
                    shutil.copy2(photo_path, dest)
            elif mode == "symlink":
                os.symlink(photo_path, dest)
            elif mode == "copy":
                shutil.copy2(photo_path, dest)
            else:
                raise ValueError(f"Unsupported photo import mode: {mode}")
        prefilter_reason = None
        if dest.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            frame_id = hashlib.sha1(str(dest).encode()).hexdigest()[:16]
            conn.execute(
                "INSERT OR IGNORE INTO frames(frame_id, source_id, path) VALUES (?, ?, ?)",
                (frame_id, sid, str(dest)),
            )
            # Phase 5 ingestion tier: zero-decode battery (size / resolution
            # header / bytes-per-pixel / truncation). Reversible via
            # `prefilter --restore`; never raises into the import path.
            if prefilter_cfg is not None:
                try:
                    from .prefilter import mark_rejected, prefilter_path
                    prefilter_reason = prefilter_path(dest, prefilter_cfg)
                    if prefilter_reason:
                        mark_rejected(conn, frame_id, prefilter_reason)
                except Exception:
                    prefilter_reason = None
        manifest.mark(conn, sid, "completed", output_dir=photo_root, frames_extracted=1)
        if prefilter_reason:
            return f"PREFILTER photo: {photo_path.name} ({prefilter_reason})"
        return f"OK photo: {photo_path.name}"
    except Exception as exc:
        manifest.mark(conn, sid, "failed", error=str(exc))
        return f"ERR photo: {photo_path.name}: {exc}"


# -----------------------------
# Pipeline
# -----------------------------

def pipeline_generator(
    video_dirs: list[str],
    photos_dir: str,
    cfg: ExtractConfig,
) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)

    deps = check_dependencies()
    use_vtb = deps["videotoolbox"]

    yield (
        "Dependency check:\n"
        f"- ffmpeg: {deps['ffmpeg']} | ffprobe: {deps['ffprobe']}\n"
        f"- VideoToolbox: {use_vtb} | zscale(HDR): {deps['zscale']}\n"
        f"- Pillow: {deps['pillow']} | insightface: {deps['insightface']}\n"
        f"- torch MPS: {deps['torch_mps']}\n"
    )
    if not deps["ffmpeg"] or not deps["ffprobe"]:
        yield "FATAL: Missing ffmpeg/ffprobe. Install with: brew install ffmpeg"
        return

    videos = discover_videos(video_dirs)
    if cfg.limit_videos > 0:
        videos = videos[: cfg.limit_videos]
    total_bytes = sum(v.stat().st_size for v in videos)
    yield f"Discovered {len(videos)} videos ({human_bytes(total_bytes)}).\n"

    processed = 0
    total_frames = 0
    failed = 0
    lines: list[str] = []

    for idx, vid in enumerate(videos, start=1):
        sid = stable_id(vid)
        try:
            msg, frames = extract_video(conn, vid, cfg, use_vtb)
            processed += 1
            total_frames += frames
            lines.append(f"[{idx}/{len(videos)}] {msg}")
        except Exception as exc:
            failed += 1
            err = str(exc)
            manifest.mark(conn, sid, "failed", error=err)
            manifest.event(conn, sid, "ERROR", err)
            lines.append(f"[{idx}/{len(videos)}] FAILED: {vid.name}: {err[:300]}")
        yield ("\n".join(lines[-25:])
               + f"\n\nProcessed: {processed} | Failed: {failed} | Frames: {total_frames}")

    if cfg.import_photos:
        photos = discover_photos(photos_dir)
        pf_cfg = (PrefilterConfig(output_base=cfg.output_base)
                  if getattr(cfg, "prefilter", True) else None)
        pf_note = (" (prefilter on)" if pf_cfg else " (prefilter off)")
        yield f"\nVideo stage done. Importing {len(photos)} photos{pf_note}...\n"
        imported = 0
        photo_failed = 0
        prefiltered = 0
        from .progress import ProgressTracker
        prog = ProgressTracker(len(photos), "photo import", every=100)
        for photo in photos:
            msg = import_one_photo(
                conn=conn, photo_path=photo, output_base=cfg.output_base,
                mode=cfg.photo_import_mode, resume=cfg.resume, dry_run=cfg.dry_run,
                prefilter_cfg=pf_cfg,
            )
            if msg.startswith("ERR"):
                photo_failed += 1
            else:
                imported += 1
                if msg.startswith("PREFILTER"):
                    prefiltered += 1
            line = prog.tick()
            if line:
                tail = f"ok/skip {imported}, failed {photo_failed}"
                if pf_cfg and prefiltered:
                    tail += f", prefiltered {prefiltered}"
                yield type(line)(f"{line} | {tail}")
        if photos:
            yield prog.summary(
                extra=f"ok/skip {imported}, failed {photo_failed}"
                      + (f", prefiltered {prefiltered}" if prefiltered
                         else ""))
        if prefiltered:
            yield (f"Prefilter rejected {prefiltered} photo(s) at ingestion "
                   "(tiny/low-res/heavy-compression/truncated). Review with "
                   "`pixel-curator insights intelligence`; undo with "
                   "`pixel-curator prefilter --restore`.")

    yield (
        "\nDONE.\n"
        f"Videos processed: {processed}\nVideo failures: {failed}\n"
        f"Extracted frames: {total_frames}\n"
        f"Output: {cfg.output_base}\n"
        f"Manifest: {cfg.output_base / DB_NAME}\n"
        f"Log: {cfg.output_base / LOG_NAME}\n"
        "\nNext: pixel-curator curate"
    )
