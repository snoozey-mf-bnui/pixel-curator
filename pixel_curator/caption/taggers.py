"""Pluggable tagging backends for the caption pipeline (Phase 1).

Every backend implements the exact contract the rules engine already
consumes:

    tag_fn(path: Path) -> Optional[list[tuple[str, float]]]

so blacklist / remap / prune / trigger injection / pony prefix, the DB
writes, the frequency report and --reprune are all backend-agnostic.

Backends:

  wd14      The existing ONNX path (caption/wd14.py), completely unchanged.
            CoreML is attempted and validated; swinv2 usually falls back to
            CPU (a known CoreML partitioning limit). Default for
            backward compatibility.

  mps       The same SmilingWolf WD14-v3 models run natively on
            torch-MPS via timm (the wdv3-timm approach). Same
            selected_tags.csv, same thresholds, same tags out - just on the
            GPU. Defaults to the ViT variant (best MPS fit); --repo-id
            still overrides. Requires torch + timm; falls back to wd14
            with a logged reason when missing. Also runs on CUDA/CPU, so
            the same flag works on Colab.

Selection precedence: --tagger flag > project.toml default_tagger > "wd14".
Resolution is lazy: nothing heavy loads until the first image is tagged.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ..config import CaptionConfig

logger = logging.getLogger(__name__)

TaggerFn = Callable[[Path], list | None]

VALID_TAGGERS = ("wd14", "mps")

# ViT partitions cleanly on MPS/CoreML where swinv2 does not; use it as the
# mps-backend default unless the user explicitly picked another repo.
_MPS_DEFAULT_REPO = "SmilingWolf/wd-vit-tagger-v3"
_WD14_DEFAULT_REPO = "SmilingWolf/wd-swinv2-tagger-v3"


# ---------------------------------------------------------------------------
# Backend probes (cheap, import-only)
# ---------------------------------------------------------------------------
def mps_backend_available() -> tuple[bool, str]:
    try:
        import timm  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:
        return False, f"torch/timm not installed ({exc})"
    return True, "ok"


# ---------------------------------------------------------------------------
# wd14 (existing behavior, delegated untouched)
# ---------------------------------------------------------------------------
def _make_wd14(cfg: CaptionConfig) -> tuple[TaggerFn, str]:
    from . import wd14 as _wd14

    def tag(path: Path):
        return _wd14.tag_image(path, cfg)

    # Force the (lazy) load now so availability errors surface immediately,
    # exactly like the current caption_generator does.
    _wd14._load_tagger(cfg.repo_id)
    session = _wd14._TAGGER[0]
    provider = session.get_providers()[0] if session else "?"
    return tag, f"wd14/onnx [{provider}]"


# ---------------------------------------------------------------------------
# mps (torch + timm, WD14-v3 weights on the GPU)
# ---------------------------------------------------------------------------
def _make_mps(cfg: CaptionConfig) -> tuple[TaggerFn, str]:
    import csv

    import timm
    import torch
    from huggingface_hub import hf_hub_download
    from timm.data import create_transform, resolve_data_config

    from ..runtime import get_device_name

    repo = cfg.repo_id if cfg.repo_id != _WD14_DEFAULT_REPO else _MPS_DEFAULT_REPO
    device = get_device_name("auto")           # cuda -> mps -> cpu
    model = timm.create_model(f"hf-hub:{repo}", pretrained=True)
    model = model.to(device).eval()
    transform = create_transform(**resolve_data_config(model.pretrained_cfg,
                                                       model=model))
    tags_path = hf_hub_download(repo_id=repo, filename="selected_tags.csv")
    tag_rows: list[tuple[str, int]] = []
    with open(tags_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tag_rows.append((row["name"], int(row["category"])))

    def tag(path: Path):
        from PIL import Image
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                # WD14 preprocessing: pad to square on white before resize
                w, h = im.size
                side = max(w, h)
                canvas = Image.new("RGB", (side, side), (255, 255, 255))
                canvas.paste(im, ((side - w) // 2, (side - h) // 2))
                x = transform(canvas).unsqueeze(0)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            return None
        x = x[:, [2, 1, 0]]                    # RGB -> BGR (WD14 convention)
        with torch.inference_mode():
            probs = torch.sigmoid(model(x.to(device)))[0].float().cpu()
        out: list[tuple[str, float]] = []
        for (name, category), p in zip(tag_rows, probs.tolist()):
            if category == 9:                  # rating tags -> skip
                continue
            thr = cfg.char_threshold if category == 4 else cfg.threshold
            if p >= thr:
                out.append((name, float(p)))
        return out

    return tag, f"wd14-v3/timm [{device}] {repo.split('/')[-1]}"


# ---------------------------------------------------------------------------
# Resolution with graceful fallback
# ---------------------------------------------------------------------------
_FACTORIES = {"wd14": _make_wd14, "mps": _make_mps}


def resolve_tagger(cfg: CaptionConfig) -> tuple[TaggerFn, str, str]:
    """Build the requested backend; fall back to wd14 with a logged reason.

    Returns (tag_fn, banner, actual_name). Raises RuntimeError only when even
    the wd14 fallback is unavailable (same failure mode as today).
    """
    requested = (getattr(cfg, "tagger", "") or "wd14").strip().lower()
    if requested not in _FACTORIES:
        logging.warning("Unknown tagger '%s' - using wd14", requested)
        requested = "wd14"

    if requested != "wd14":
        ok, reason = mps_backend_available()
        if not ok:
            logging.info("Tagger '%s' unavailable (%s) - falling back to wd14",
                         requested, reason)
            requested = "wd14"

    try:
        fn, banner = _FACTORIES[requested](cfg)
        return fn, banner, requested
    except Exception as exc:
        if requested == "wd14":
            raise
        logging.info("Tagger '%s' failed to initialize (%s) - falling back "
                     "to wd14", requested, exc)
        # Fall back through the registry so the wd14 factory stays the single
        # source of truth (honors any override in _FACTORIES).
        fn, banner = _FACTORIES["wd14"](cfg)
        return fn, banner, "wd14"
