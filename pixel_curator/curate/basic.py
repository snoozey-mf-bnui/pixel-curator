"""Basic curation: dHash perceptual dedup, Laplacian blur scoring,
exposure filtering. Pure stdlib (Pillow used opportunistically)."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, Optional

from .. import manifest
from ..config import CurateConfig
from ..util import HAVE_PIL, now_iso, setup_logging

if HAVE_PIL:
    from PIL import Image


def load_gray(path: Path, w: int, h: int) -> Optional[list[int]]:
    """Grayscale w*h pixel list. Pillow if available, else ffmpeg rawvideo."""
    if HAVE_PIL:
        try:
            with Image.open(path) as im:
                im = im.convert("L").resize((w, h), Image.BILINEAR)
                return list(im.getdata())
        except Exception:
            return None
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", str(path),
             "-vf", f"scale={w}:{h}", "-pix_fmt", "gray",
             "-f", "rawvideo", "-"],
            capture_output=True,
        )
        data = proc.stdout
        if len(data) < w * h:
            return None
        return list(data[: w * h])
    except Exception:
        return None


def dhash_bits(gray9x8: list[int]) -> int:
    bits = 0
    for row in range(8):
        for col in range(8):
            left = gray9x8[row * 9 + col]
            right = gray9x8[row * 9 + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def laplacian_variance(gray: list[int], w: int, h: int) -> float:
    vals: list[float] = []
    for y in range(1, h - 1):
        base = y * w
        for x in range(1, w - 1):
            c = gray[base + x]
            lap = (gray[base + x - 1] + gray[base + x + 1]
                   + gray[base - w + x] + gray[base + w + x] - 4 * c)
            vals.append(float(lap))
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def score_frame(path_str: str) -> Optional[tuple[str, float, float]]:
    res = score_frame_ex(path_str, text_check=False)
    if res is None:
        return None
    return res[:3]


def score_frame_ex(
    path_str: str, text_check: bool = True
) -> Optional[tuple[str, float, float, Optional[str]]]:
    """score_frame plus the Phase 5 text-overlay band check, which reuses
    the 96x96 gray this function decodes anyway (zero extra I/O). Returns
    ``(dhash, sharpness, brightness, prefilter_reason_or_None)``."""
    p = Path(path_str)
    g9 = load_gray(p, 9, 8)
    if g9 is None:
        return None
    dh = dhash_bits(g9)
    g = load_gray(p, 96, 96)
    if g is None:
        return None
    sharp = laplacian_variance(g, 96, 96)
    bright = sum(g) / len(g)
    reason = None
    if text_check:
        try:
            from ..prefilter import detect_text_band
            if detect_text_band(g, 96, 96):
                reason = "text_overlay"
        except Exception:
            reason = None
    return (f"{dh:016x}", sharp, bright, reason)


def curate_generator(cfg: CurateConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)

    # rescore re-runs everything EXCEPT prefilter rejections: curate only
    # re-checks text bands, so letting tiny/heavy-compression junk back in
    # here would silently undo the ingestion battery. Those frames return
    # via `prefilter --restore` only.
    where = ("status = 'new'" if not cfg.rescore
             else "status != 'rejected_prefilter'")
    rows = conn.execute(
        f"SELECT frame_id, source_id, path FROM frames WHERE {where} "
        "ORDER BY source_id, path"
    ).fetchall()
    if not rows:
        yield "No frames to curate. Run extract/photos first."
        return

    text_check = bool(getattr(cfg, "prefilter_text", True))
    # Phase 9: visible implicit-resume (only 'new' frames are scored, so
    # a rerun after an interrupt continues exactly where it stopped)
    from ..progress import ProgressTracker
    already = 0
    if not cfg.rescore:
        already = conn.execute(
            "SELECT COUNT(*) FROM frames WHERE status NOT IN "
            "('new', 'rejected_prefilter')").fetchone()[0]
    prog = ProgressTracker(len(rows), "scoring", every=200,
                           already_done=already)
    banner = prog.resume_banner()
    if banner:
        yield banner
    yield (f"Scoring {len(rows)} frames (workers={cfg.workers}, "
           f"pillow={HAVE_PIL}, text_check={text_check})...\n")
    done = 0
    failed = 0
    text_rejected = 0

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(score_frame_ex, r[2], text_check): r
                   for r in rows}
        batch: list[tuple] = []
        for fut in as_completed(futures):
            frame_id, source_id, path = futures[fut]
            try:
                res = fut.result()
            except Exception:
                res = None
            if res is None:
                failed += 1
                batch.append((None, None, None, "decode_failed", now_iso(),
                              None, frame_id))
            else:
                dh, sharp, bright, pf_reason = res
                if pf_reason:
                    # Phase 5: caption-bar screenshot. Keep the scores for
                    # review but route it out of curation, reversibly.
                    text_rejected += 1
                    batch.append((dh, sharp, bright, "rejected_prefilter",
                                  now_iso(), pf_reason, frame_id))
                else:
                    batch.append((dh, sharp, bright, "scored", now_iso(),
                                  None, frame_id))
            done += 1
            if len(batch) >= 200:
                conn.executemany(
                    "UPDATE frames SET dhash=?, sharpness=?, brightness=?, "
                    "status=?, scored_at=?, prefilter_reason=? "
                    "WHERE frame_id=?",
                    batch,
                )
                conn.commit()          # durable batch: interrupt-safe
                batch.clear()
            line = prog.tick()
            if line:
                yield type(line)(f"{line} | {failed} failed") if failed \
                    else line
        if batch:
            conn.executemany(
                "UPDATE frames SET dhash=?, sharpness=?, brightness=?, "
                "status=?, scored_at=?, prefilter_reason=? "
                "WHERE frame_id=?",
                batch,
            )
            conn.commit()

    yield prog.summary(
        extra=f"{failed} failed" if failed else "") + \
        " Applying quality filters...\n"

    rejected_blur = conn.execute(
        "UPDATE frames SET status='rejected_blur' "
        "WHERE status='scored' AND ? > 0 AND sharpness < ?",
        (cfg.min_sharpness, cfg.min_sharpness),
    ).rowcount
    rejected_exposure = conn.execute(
        "UPDATE frames SET status='rejected_exposure' "
        "WHERE status='scored' AND (brightness < ? OR brightness > ?)",
        (cfg.min_brightness, cfg.max_brightness),
    ).rowcount
    conn.commit()

    yield "Deduplicating (dHash)...\n"
    dup_count = 0
    sources = [r[0] for r in conn.execute(
        "SELECT DISTINCT source_id FROM frames WHERE status='scored'"
    )]
    for sid in sources:
        frames = conn.execute(
            "SELECT frame_id, dhash, sharpness FROM frames "
            "WHERE source_id=? AND status='scored' ORDER BY path",
            (sid,),
        ).fetchall()
        kept: list[tuple[str, int, float]] = []
        dups: list[str] = []
        for frame_id, dh_hex, sharp in frames:
            if not dh_hex:
                continue
            h = int(dh_hex, 16)
            is_dup = any(
                hamming(h, kh) <= cfg.hamming_threshold
                for _, kh, _ in kept[-12:]
            )
            if is_dup:
                dups.append(frame_id)
            else:
                kept.append((frame_id, h, sharp or 0.0))
        if dups:
            conn.executemany(
                "UPDATE frames SET status='duplicate' WHERE frame_id=?",
                [(d,) for d in dups],
            )
            dup_count += len(dups)
        conn.executemany(
            "UPDATE frames SET status='selected' WHERE frame_id=?",
            [(k[0],) for k in kept],
        )
        conn.commit()

    sel = conn.execute("SELECT COUNT(*) FROM frames WHERE status='selected'").fetchone()[0]
    text_line = ""
    if text_rejected:
        text_line = (f"Rejected (text overlay): {text_rejected}"
                     "   (undo: pixel-curator prefilter --restore "
                     "--reason text_overlay)\n")
    yield (
        "\nBASIC CURATION DONE.\n"
        f"Selected: {sel}\n"
        f"Duplicates removed: {dup_count}\n"
        f"Rejected (blur): {rejected_blur}\n"
        f"Rejected (exposure): {rejected_exposure}\n"
        f"{text_line}"
        f"Decode failures: {failed}\n"
        "\nNext: pixel-curator curate --smart   (identity filtering, needs [ai] extras)"
    )
