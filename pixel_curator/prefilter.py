"""Phase 5: Early Pre-filtering - fast rejection at ingestion.

Catches the files that would never survive curation before they cost the
pipeline any real work: thumbnail-sized junk, heavy social-media
re-compressions, truncated downloads, sub-training-resolution images, and
Snapchat-style caption-bar screenshots.

Two tiers, designed so no image is ever decoded twice:

* zero-decode battery (``prefilter_path``): stat() + a PIL *header-only*
  read. Runs inline at photo import and in the standalone ``prefilter``
  command. Checks: tiny_file, low_resolution, heavy_compression (bytes per
  pixel for JPEG/WebP), truncated (JPEG EOI tail read).
* decode tier: the text-overlay band detector (``detect_text_band``) and an
  extreme-blur floor. In ``curate`` these piggyback the 96x96 gray that
  scoring already loads; the standalone command decodes once itself.

Everything is reversible: rejected frames get status 'rejected_prefilter'
plus a machine-readable ``prefilter_reason``; ``restore_prefiltered`` flips
them back to 'new'. Each run writes a JSON report under
``output_base/prefilter_reports/`` for review. Source media is never
touched.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import manifest
from .config import PrefilterConfig
from .curate.basic import laplacian_variance, load_gray
from .util import HAVE_PIL, now_iso, setup_logging

logger = logging.getLogger(__name__)

if HAVE_PIL:
    from PIL import Image

STATUS = "rejected_prefilter"
REASONS = ("tiny_file", "low_resolution", "heavy_compression",
           "truncated", "text_overlay", "extreme_blur", "unreadable")

# extensions where bytes-per-pixel is a meaningful compression proxy
_BPP_EXTS = (".jpg", ".jpeg", ".webp")
_JPEG_EXTS = (".jpg", ".jpeg")

# --- text-overlay band detector tuning (conservative on purpose) ----------
_BAND_ROWS = 8            # strip height in rows on the 96x96 gray
_EDGE_THRESH = 40         # |gx| gradient magnitude that counts as an edge
_BAND_EDGE_MIN = 0.20     # strip edge density floor to even be considered
_BAND_EDGE_RATIO = 3.5    # strip must be this many times busier than median
_BAND_MEAN_DELTA = 40.0   # strip must sit on a distinctly dark/bright bar


# ---------------------------------------------------------------------------
# Zero-decode checks (stat + header only)
# ---------------------------------------------------------------------------

def check_file_size(path: Path, min_bytes: int) -> str | None:
    try:
        if path.stat().st_size < min_bytes:
            return "tiny_file"
    except OSError:
        return "unreadable"
    return None


def _header_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from the image header - PIL lazy open, no pixel
    decode. None when the header itself is unreadable."""
    if not HAVE_PIL:
        return None
    try:
        with Image.open(path) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return None


def check_resolution(path: Path, min_side: int,
                     size: tuple[int, int] | None = None) -> str | None:
    size = size if size is not None else _header_size(path)
    if size is None:
        return None                     # header trouble handled elsewhere
    if min(size) < min_side:
        return "low_resolution"
    return None


def check_compression(path: Path, min_bpp: float,
                      size: tuple[int, int] | None = None) -> str | None:
    """Bytes-per-pixel floor for lossy formats. A quality-2 ffmpeg JPEG runs
    ~0.5-1.5 bpp; social-media re-compression mush sits under ~0.05."""
    if path.suffix.lower() not in _BPP_EXTS or min_bpp <= 0:
        return None
    size = size if size is not None else _header_size(path)
    if size is None:
        return None
    w, h = size
    if w * h <= 0:
        return None
    try:
        bpp = path.stat().st_size / float(w * h)
    except OSError:
        return "unreadable"
    if bpp < min_bpp:
        return "heavy_compression"
    return None


def check_truncation(path: Path) -> str | None:
    """JPEG files end in the EOI marker FFD9; a missing tail means the
    download/copy was cut short. Two-byte read, no decode."""
    if path.suffix.lower() not in _JPEG_EXTS:
        return None
    try:
        with open(path, "rb") as fh:
            fh.seek(-2, 2)
            if fh.read(2) != b"\xff\xd9":
                return "truncated"
    except OSError:
        return "unreadable"
    except Exception:
        return None                     # <2-byte file already caught by size
    return None


def prefilter_path(path, cfg: PrefilterConfig) -> str | None:
    """Zero-decode battery. Returns the first rejection reason or None.
    Never raises."""
    path = Path(str(path))
    reason = check_file_size(path, cfg.min_bytes)
    if reason:
        return reason
    size = _header_size(path)
    reason = check_resolution(path, cfg.min_side, size=size)
    if reason:
        return reason
    reason = check_compression(path, cfg.min_bpp, size=size)
    if reason:
        return reason
    if cfg.check_truncation:
        reason = check_truncation(path)
        if reason:
            return reason
    return None


# ---------------------------------------------------------------------------
# Decode-tier checks
# ---------------------------------------------------------------------------

def detect_text_band(gray: list[int], w: int = 96, h: int = 96) -> bool:
    """Snapchat/social caption-bar signature on a small grayscale image:
    exactly the kind of horizontal strip that is (a) far busier with
    horizontal-gradient edges than the rest of the frame AND (b) sits on a
    background distinctly darker or brighter than the rest. Triple-gated so
    ordinary busy photos don't trip it. Pure function, never raises."""
    if not gray or len(gray) < w * h:
        return False
    n_bands = h // _BAND_ROWS
    if n_bands < 3:
        return False
    stats: list[tuple[float, float]] = []      # (mean, edge_density)
    for b in range(n_bands):
        y0 = b * _BAND_ROWS
        total = 0.0
        edges = 0
        count = 0
        for y in range(y0, min(y0 + _BAND_ROWS, h)):
            base = y * w
            row_prev = gray[base]
            total += row_prev
            for x in range(1, w):
                px = gray[base + x]
                total += px
                if abs(px - row_prev) >= _EDGE_THRESH:
                    edges += 1
                row_prev = px
                count += 1
        pixels = _BAND_ROWS * w
        stats.append((total / pixels, edges / max(1, count)))

    densities = sorted(d for _, d in stats)
    median_density = densities[len(densities) // 2]
    for i, (mean, density) in enumerate(stats):
        if density < _BAND_EDGE_MIN:
            continue
        if density < _BAND_EDGE_RATIO * max(median_density, 0.01):
            continue
        others = [m for j, (m, _) in enumerate(stats) if j != i]
        rest_mean = sum(others) / len(others)
        if abs(mean - rest_mean) >= _BAND_MEAN_DELTA:
            return True
    return False


def prefilter_decode(path, cfg: PrefilterConfig,
                     gray: list[int] | None = None) -> str | None:
    """Decode-tier battery. Accepts an already-loaded 96x96 gray so callers
    that decoded anyway (curate scoring) pay nothing extra. Never raises."""
    if gray is None:
        gray = load_gray(Path(str(path)), 96, 96)
    if gray is None:
        return "unreadable"
    if cfg.text_check and detect_text_band(gray, 96, 96):
        return "text_overlay"
    if cfg.extreme_blur > 0:
        if laplacian_variance(gray, 96, 96) < cfg.extreme_blur:
            return "extreme_blur"
    return None


# ---------------------------------------------------------------------------
# Manifest operations
# ---------------------------------------------------------------------------

def mark_rejected(conn, frame_id: str, reason: str) -> None:
    conn.execute(
        "UPDATE frames SET status=?, prefilter_reason=? WHERE frame_id=?",
        (STATUS, reason, frame_id),
    )


def restore_prefiltered(conn, reason: str | None = None) -> int:
    """Flip prefilter-rejected frames back to 'new' (optionally only one
    reason) so they re-enter curation. Returns the restored count."""
    if reason:
        cur = conn.execute(
            "UPDATE frames SET status='new', prefilter_reason=NULL "
            "WHERE status=? AND prefilter_reason=?", (STATUS, reason))
    else:
        cur = conn.execute(
            "UPDATE frames SET status='new', prefilter_reason=NULL "
            "WHERE status=?", (STATUS,))
    conn.commit()
    return cur.rowcount


def prefilter_summary(conn) -> dict:
    """Reason -> count for currently prefilter-rejected frames."""
    rows = conn.execute(
        "SELECT COALESCE(prefilter_reason,'unknown'), COUNT(*) FROM frames "
        "WHERE status=? GROUP BY prefilter_reason", (STATUS,)).fetchall()
    return {r: c for r, c in rows}


def _write_report(output_base: Path, entries: list[dict],
                  counts: dict, dry_run: bool) -> Path | None:
    """Reviewable JSON report of one prefilter run. Never raises."""
    try:
        rep_dir = Path(output_base) / "prefilter_reports"
        rep_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = rep_dir / f"prefilter_{stamp}.json"
        path.write_text(json.dumps({
            "created_at": now_iso(),
            "dry_run": dry_run,
            "counts": counts,
            "rejected": entries,
            "restore_hint": ("pixel-curator prefilter --restore "
                             "[--reason text_overlay]"),
        }, indent=2), encoding="utf-8")
        return path
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Standalone runner (CLI `prefilter` command)
# ---------------------------------------------------------------------------

def prefilter_generator(cfg: PrefilterConfig) -> Generator[str, None, None]:
    """Run the full battery over status='new' frames. Zero-decode checks
    always; decode checks (text band / extreme blur) when
    ``cfg.decode_checks``. Dry-run reports without writing."""
    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)

    rows = conn.execute(
        "SELECT frame_id, path FROM frames WHERE status='new' ORDER BY path"
    ).fetchall()
    if not rows:
        yield "No 'new' frames to prefilter. Run extract/photos first."
        return

    mode = ("zero-decode + decode checks" if cfg.decode_checks
            else "zero-decode only")
    yield (f"Prefiltering {len(rows)} frames ({mode}, "
           f"workers={cfg.workers}, dry_run={cfg.dry_run})...\n"
           f"Gates: min {cfg.min_bytes} bytes | min side {cfg.min_side}px | "
           f"min {cfg.min_bpp} bytes/px"
           + (f" | text bands | blur<{cfg.extreme_blur}"
              if cfg.decode_checks else "") + "\n")

    def _check(path_str: str) -> str | None:
        p = Path(path_str)
        reason = prefilter_path(p, cfg)
        if reason is None and cfg.decode_checks:
            reason = prefilter_decode(p, cfg)
        return reason

    entries: list[dict] = []
    counts: dict[str, int] = {}
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(_check, r[1]): r for r in rows}
        batch: list[tuple[str, str]] = []
        for fut in as_completed(futures):
            frame_id, path = futures[fut]
            try:
                reason = fut.result()
            except Exception:
                reason = None
            done += 1
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
                entries.append({"frame_id": frame_id, "path": path,
                                "reason": reason})
                if not cfg.dry_run:
                    batch.append((reason, frame_id))
            if batch and len(batch) >= 200:
                conn.executemany(
                    "UPDATE frames SET status=?, prefilter_reason=? "
                    "WHERE frame_id=?",
                    [(STATUS, r, f) for r, f in batch])
                conn.commit()
                batch.clear()
            if done % 500 == 0:
                rate = done / max(0.001, time.time() - t0)
                yield (f"Checked {done}/{len(rows)} ({rate:.0f} img/s, "
                       f"{len(entries)} flagged)")
        if batch and not cfg.dry_run:
            conn.executemany(
                "UPDATE frames SET status=?, prefilter_reason=? "
                "WHERE frame_id=?",
                [(STATUS, r, f) for r, f in batch])
            conn.commit()

    report = _write_report(cfg.output_base, entries, counts, cfg.dry_run)
    verb = "Would reject" if cfg.dry_run else "Rejected"
    lines = ["", "PREFILTER DONE." if not cfg.dry_run
             else "PREFILTER DRY-RUN (nothing written).",
             f"Checked: {done} | {verb}: {len(entries)}"]
    for reason in REASONS:
        if reason in counts:
            lines.append(f"  {reason:<18} {counts[reason]}")
    if report:
        lines.append(f"Report: {report}")
    if entries and not cfg.dry_run:
        lines.append("Undo:   pixel-curator prefilter --restore "
                     "[--reason <reason>]")
    lines.append("\nNext: pixel-curator curate")
    yield "\n".join(lines)


__all__ = [
    "STATUS", "REASONS", "PrefilterConfig",
    "check_file_size", "check_resolution", "check_compression",
    "check_truncation", "prefilter_path", "detect_text_band",
    "prefilter_decode", "mark_rejected", "restore_prefiltered",
    "prefilter_summary", "prefilter_generator",
]
