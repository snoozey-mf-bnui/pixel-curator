"""Phase 3: WD14 auto-captioning with a Pony-aware rules engine.

Tagger: SmilingWolf/wd-swinv2-tagger-v3 (ONNX) via onnxruntime
(CoreML EP on Apple Silicon, CPU fallback). Booru-style tags - the native
caption dialect for Pony Diffusion training.

Install:  pip install 'pixel-curator[caption]'
Model (~450MB) auto-downloads to the HuggingFace cache on first run.
(The reForge DeepBooru threshold/formatting approach from modules/deepbooru.py
is the pattern reference; WD14 v3 is its modern successor.)

Rules engine (pure stdlib, unit-tested):
  - threshold filtering (general vs character tag categories)
  - blacklist: drop tags outright
  - remap: rename tags (e.g. "1girl:woman")
  - prune: remove permanent-trait tags so they bind to the trigger word
  - trigger + class word injection at position 0
  - optional Pony quality prefix (score_9, score_8_up, score_7_up)
Edited captions (edited=1) are never overwritten.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from pathlib import Path

from .. import manifest
from ..config import CaptionConfig
from ..util import now_iso, setup_logging

logger = logging.getLogger(__name__)

_TAGGER = None
_TAGGER_ERR: str | None = None

PONY_PREFIX = ["score_9", "score_8_up", "score_7_up"]


def caption_available() -> tuple[bool, str]:
    try:
        import huggingface_hub  # noqa: F401
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        from PIL import Image  # noqa: F401
        return True, "ok"
    except Exception as exc:
        return False, f"missing [caption] extras: {exc}"


# -----------------------------
# Rules engine (pure, testable)
# -----------------------------

def parse_rules(cfg: CaptionConfig) -> dict:
    blacklist = {t.strip().replace(" ", "_").lower()
                 for t in cfg.blacklist.split(",") if t.strip()}
    prune = {t.strip().replace(" ", "_").lower()
             for t in cfg.prune.split(",") if t.strip()}
    remap = {}
    for pair in cfg.remap.split(","):
        if ":" in pair:
            old, new = pair.split(":", 1)
            if old.strip():
                remap[old.strip().replace(" ", "_").lower()] = (
                    new.strip().replace(" ", "_").lower()
                )
    return {"blacklist": blacklist, "prune": prune, "remap": remap}


def build_caption(tags: list[tuple[str, float]], cfg: CaptionConfig,
                  rules: dict | None = None) -> str:
    """tags: [(tag_name, confidence), ...] already threshold-filtered.
    Returns final caption string (comma-space separated, booru style)."""
    rules = rules or parse_rules(cfg)
    out: list[str] = []
    seen: set[str] = set()
    for name, _conf in sorted(tags, key=lambda t: -t[1]):
        tag = name.strip().replace(" ", "_").lower()
        if tag in rules["blacklist"] or tag in rules["prune"]:
            continue
        tag = rules["remap"].get(tag, tag)
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
        if len(out) >= cfg.max_tags:
            break
    # presentation: spaces instead of underscores except score tags
    pretty = [t.replace("_", " ") for t in out]
    head: list[str] = []
    if cfg.pony_prefix:
        head.extend(PONY_PREFIX)
    trigger_part = f"{cfg.trigger} {cfg.class_word}".strip()
    if trigger_part:
        head.append(trigger_part)
    return ", ".join(head + pretty)


# -----------------------------
# WD14 tagger backend
# -----------------------------

def _load_tagger(repo_id: str):
    """Returns (session, input_name, input_size, tag_rows) where tag_rows is
    [(name, category), ...] aligned with model output order."""
    global _TAGGER, _TAGGER_ERR
    if _TAGGER is not None:
        return _TAGGER
    if _TAGGER_ERR is not None:
        raise RuntimeError(_TAGGER_ERR)
    try:
        import csv

        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx")
        tags_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv")

        from ..runtime import onnx_providers
        providers = onnx_providers()       # CUDA -> CoreML -> CPU
        session = ort.InferenceSession(model_path, providers=providers)
        inp = session.get_inputs()[0]
        size = inp.shape[1] if isinstance(inp.shape[1], int) else 448

        # WD14 swinv2 only partially partitions under CoreML and can fail
        # at inference. Validate with a dummy run; fall back to CPU-only.
        if providers[0] == "CoreMLExecutionProvider":
            try:
                import numpy as _np
                dummy = _np.zeros((1, size, size, 3), dtype=_np.float32)
                session.run(None, {inp.name: dummy})
            except Exception:
                logging.info("CoreML EP failed for WD14 - falling back "
                             "to CPUExecutionProvider")
                session = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"])
                inp = session.get_inputs()[0]

        tag_rows: list[tuple[str, int]] = []
        with open(tags_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                tag_rows.append((row["name"], int(row["category"])))

        _TAGGER = (session, inp.name, size, tag_rows)
        return _TAGGER
    except Exception as exc:
        _TAGGER_ERR = (f"WD14 tagger unavailable: {exc}\n"
                       "Install:  pip install 'pixel-curator[caption]'")
        raise RuntimeError(_TAGGER_ERR)


def _prep_image(path: Path, size: int):
    """WD14 preprocessing: RGB -> pad square (white) -> resize -> BGR float32 NHWC."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(im, ((side - w) // 2, (side - h) // 2))
        canvas = canvas.resize((size, size), Image.BICUBIC)
    arr = np.asarray(canvas, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR
    return arr[None, ...]


def tag_image(path: Path, cfg: CaptionConfig) -> list[tuple[str, float]] | None:
    """Run the tagger on one image. Returns threshold-filtered (tag, conf)."""
    session, input_name, size, tag_rows = _load_tagger(cfg.repo_id)
    try:
        batch = _prep_image(path, size)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return None
    probs = session.run(None, {input_name: batch})[0][0]
    out: list[tuple[str, float]] = []
    for (name, category), p in zip(tag_rows, probs.tolist()):
        if category == 9:        # rating tags -> skip
            continue
        thr = cfg.char_threshold if category == 4 else cfg.threshold
        if p >= thr:
            out.append((name, float(p)))
    return out


# -----------------------------
# Generator
# -----------------------------

def caption_generator(cfg: CaptionConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    ok, reason = caption_available()
    if not ok:
        yield f"Captioning skipped: {reason}"
        return
    conn = manifest.connect(cfg.output_base)

    cond = "" if cfg.force else "AND c.frame_id IS NULL"
    rows = conn.execute(
        f"SELECT f.frame_id, f.path FROM frames f "
        f"LEFT JOIN captions c ON c.frame_id = f.frame_id AND c.edited = 0 "
        f"WHERE f.status IN ('selected','packaged') {cond} ORDER BY f.path"
    ).fetchall()
    # never clobber manual edits
    edited = {r[0] for r in conn.execute("SELECT frame_id FROM captions WHERE edited=1")}
    rows = [r for r in rows if r[0] not in edited]

    if not rows:
        yield "Nothing to caption (all selected frames captioned; use --force to redo)."
        return

    # Backend selection (Phase 1): --tagger / project default_tagger / wd14.
    # The wd14 path delegates to the unchanged loader + tag_image below.
    from .taggers import resolve_tagger
    try:
        tag_fn, banner, actual = resolve_tagger(cfg)
    except RuntimeError as exc:
        yield str(exc)
        return

    rules = parse_rules(cfg)
    # Phase 9: visible implicit-resume + throttled percent/rate/ETA
    already = 0
    if not cfg.force:
        already = conn.execute(
            "SELECT COUNT(*) FROM frames f JOIN captions c "
            "ON c.frame_id = f.frame_id AND c.edited = 0 "
            "WHERE f.status IN ('selected','packaged')").fetchone()[0]
    from ..progress import ProgressTracker
    prog = ProgressTracker(len(rows), "captioning",
                           every=cfg.batch_log_every,
                           already_done=already)
    banner_resume = prog.resume_banner()
    if banner_resume:
        yield banner_resume
    yield (f"Captioning {len(rows)} frames | {banner}\n"
           f"thresholds gen={cfg.threshold} char={cfg.char_threshold} | "
           f"trigger='{cfg.trigger} {cfg.class_word}' | pony_prefix={cfg.pony_prefix}\n")

    done = 0
    failed = 0
    for frame_id, path_str in rows:
        tags = tag_fn(Path(path_str))
        if tags is None:
            failed += 1
        else:
            caption = build_caption(tags, cfg, rules)
            conn.execute(
                "INSERT OR REPLACE INTO captions(frame_id, tags_json, caption_text, "
                "edited, updated_at) VALUES (?,?,?,0,?)",
                (frame_id, json.dumps(tags), caption, now_iso()),
            )
        done += 1
        if done % cfg.batch_log_every == 0 or done == len(rows):
            conn.commit()          # durable batch: interrupt-safe resume
        line = prog.tick()
        if line:
            yield type(line)(f"{line} | {failed} failed") if failed \
                else line
    conn.commit()
    yield prog.summary(extra=f"{failed} failed" if failed else "")

    # tag frequency report (top 15) - the "prune what you train" signal.
    # Read caption_text (the pruned, training-facing caption), NOT tags_json
    # (which deliberately preserves the raw WD14 output for re-pruning). This
    # is what actually trains, so it reflects --prune / --blacklist / --remap.
    _FRAMEWORK = {t.replace(" ", "_").lower() for t in
                  (list(PONY_PREFIX) + [f"{cfg.trigger} {cfg.class_word}".strip(),
                                        cfg.trigger, cfg.class_word])}
    freq: dict[str, int] = {}
    rows_counted = 0
    for (cap,) in conn.execute(
        "SELECT caption_text FROM captions WHERE caption_text IS NOT NULL"
    ):
        rows_counted += 1
        seen_row = set()
        for part in str(cap).split(","):
            tag = part.strip().replace(" ", "_").lower()
            if not tag or tag in _FRAMEWORK or tag in seen_row:
                continue
            seen_row.add(tag)
            freq[tag] = freq.get(tag, 0) + 1
    total = max(1, rows_counted)
    top = sorted(freq.items(), key=lambda x: -x[1])[:15]
    report = "\n".join(f"  {n:<24} {c:>5}  ({100*c/total:.0f}%)" for n, c in top)
    yield (
        "\nCAPTIONING DONE.\n"
        f"Captioned: {done - failed} | failed: {failed}\n"
        f"\nTop tags in TRAINING captions (post-prune; consider --prune for "
        f"permanent traits >80%):\n{report}\n"
        "\nNote: raw WD14 tags are kept in tags_json for re-pruning; this "
        "report reflects caption_text (what trains).\n"
        "Next: review captions in the UI, then pixel-curator package"
    )


def _top_tag_report(freq: dict[str, int], total: int, n: int = 15) -> str:
    total = max(1, total)
    top = sorted(freq.items(), key=lambda x: -x[1])[:n]
    return "\n".join(f"  {name:<24} {c:>5}  ({100*c/total:.0f}%)"
                     for name, c in top)


def reprune_generator(cfg: CaptionConfig,
                      apply: bool = False,
                      preview: int = 8) -> Generator[str, None, None]:
    """Rebuild caption_text from the stored raw WD14 tags (tags_json) using the
    current prune / blacklist / remap / pony-prefix / trigger rules - WITHOUT
    re-running the tagger or touching any source image.

    Non-destructive and dry-run by default: with ``apply=False`` nothing is
    written; it reports what would change. ``apply=True`` rewrites caption_text
    for changed rows only. Manually edited captions (edited=1) are never
    touched. Rows without stored tags_json are skipped (they would need a real
    ``caption`` pass).
    """
    conn = manifest.connect(cfg.output_base)
    rules = parse_rules(cfg)

    rows = conn.execute(
        "SELECT c.frame_id, c.tags_json, c.caption_text "
        "FROM captions c JOIN frames f ON f.frame_id = c.frame_id "
        "WHERE c.edited = 0 AND f.status IN ('selected','packaged') "
        "ORDER BY f.path"
    ).fetchall()

    if not rows:
        yield "Nothing to re-prune (no auto captions found)."
        return

    mode = "APPLY" if apply else "DRY-RUN (preview only; pass --apply to write)"
    yield (f"Re-pruning {len(rows)} captions from stored tags | {mode}\n"
           f"prune={sorted(rules['prune']) or '-'} | "
           f"blacklist={sorted(rules['blacklist']) or '-'} | "
           f"remap={rules['remap'] or '-'} | pony_prefix={cfg.pony_prefix}\n")

    t0 = time.time()
    changed = unchanged = skipped = written = 0
    samples: list[str] = []
    freq: dict[str, int] = {}
    framework = {t.replace(" ", "_").lower() for t in
                 (list(PONY_PREFIX) + [f"{cfg.trigger} {cfg.class_word}".strip(),
                                       cfg.trigger, cfg.class_word])}

    for frame_id, tags_json, old_caption in rows:
        if not tags_json:
            skipped += 1
            continue
        try:
            tags = json.loads(tags_json)
        except Exception:
            skipped += 1
            continue
        new_caption = build_caption(tags, cfg, rules)

        # projected training-tag frequency (from the rebuilt caption)
        seen_row: set[str] = set()
        for part in new_caption.split(","):
            tag = part.strip().replace(" ", "_").lower()
            if not tag or tag in framework or tag in seen_row:
                continue
            seen_row.add(tag)
            freq[tag] = freq.get(tag, 0) + 1

        if new_caption == (old_caption or ""):
            unchanged += 1
            continue
        changed += 1
        if len(samples) < preview:
            samples.append(f"  - {old_caption}\n  + {new_caption}")
        if apply:
            conn.execute(
                "UPDATE captions SET caption_text = ?, updated_at = ? "
                "WHERE frame_id = ?",
                (new_caption, now_iso(), frame_id),
            )
            written += 1
            if written % max(1, cfg.batch_log_every) == 0:
                conn.commit()

    if apply:
        conn.commit()

    rate = len(rows) / max(0.001, time.time() - t0)
    report = _top_tag_report(freq, max(1, changed + unchanged))
    sample_block = ("\n\nSample changes (old -> new):\n" + "\n".join(samples)) \
        if samples else ""
    tail = (f"Wrote {written} rows." if apply else
            "Dry-run: no rows written. Re-run with --apply to commit.")
    yield (
        f"\nRE-PRUNE {'DONE' if apply else 'PREVIEW'}.\n"
        f"changed: {changed} | unchanged: {unchanged} | "
        f"skipped(no tags): {skipped} | {rate:.0f} rows/s\n"
        f"{tail}{sample_block}\n"
        f"\nProjected top tags in TRAINING captions:\n{report}\n"
    )
