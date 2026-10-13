"""Model asset registry and first-run auto-download.

Every heavy model the pipeline can touch is catalogued here with where it
lives and how to fetch it. Three acquisition kinds:

* ``hf``        - files from a HuggingFace repo via ``huggingface_hub``
                  (lands in the standard HF cache; already the natural
                  behavior of the WD14 taggers and HF CLIP fallback)
* ``open_clip`` - an open_clip pretrained tag (open_clip manages its own
                  download/cache)
* ``insightface`` - an InsightFace model bundle (auto-downloads to
                  ``~/.insightface/models`` on first ``FaceAnalysis`` init)
* ``url``       - a direct file download into the repo-local ``models/``
                  directory (Real-ESRGAN / GFPGAN weights)

Usage from pipeline code::

    from .model_assets import ensure_model
    path = ensure_model("realesrgan-x4plus")   # downloads on first run

CLI::

    pixel-curator models            # status table
    pixel-curator models download   # fetch everything missing
    pixel-curator models download --only realesrgan-x4plus,gfpgan-v1.4

Network access only ever happens inside ``ensure_model`` /
``download_missing``; importing this module and calling ``status()`` are
always offline and side-effect free.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent
MODELS_DIR = _PACKAGE_ROOT / "models"


@dataclass(frozen=True)
class ModelAsset:
    """One downloadable model."""

    key: str
    kind: str                     # hf | open_clip | insightface | url
    purpose: str                  # human-readable, shown in `models` status
    size_hint_mb: int = 0
    # kind == "hf"
    repo_id: str = ""
    files: tuple[str, ...] = ()
    # kind == "open_clip"
    arch: str = ""
    pretrained: str = ""
    # kind == "insightface"
    bundle: str = ""
    # kind == "url"
    url: str = ""
    dest: str = ""                # path relative to MODELS_DIR
    sha256: str = ""              # optional integrity check
    optional: bool = False        # not needed for the core pipeline


MODEL_REGISTRY: dict[str, ModelAsset] = {a.key: a for a in (
    ModelAsset(
        key="wd14-swinv2",
        kind="hf",
        purpose="WD14 SwinV2 tagger (default captioning backend, ONNX)",
        size_hint_mb=450,
        repo_id="SmilingWolf/wd-swinv2-tagger-v3",
        files=("model.onnx", "selected_tags.csv"),
    ),
    ModelAsset(
        key="wd14-vit",
        kind="hf",
        purpose="WD14 ViT v3 tagger (MPS/torch captioning backend)",
        size_hint_mb=850,
        repo_id="SmilingWolf/wd-vit-tagger-v3",
        files=("selected_tags.csv",),   # weights pulled by timm hf-hub:
        optional=True,
    ),
    ModelAsset(
        key="insightface-buffalo-l",
        kind="insightface",
        bundle="buffalo_l",
        purpose="InsightFace buffalo_l: SCRFD face detector + ArcFace "
                "identity embeddings (smart curation, anchors)",
        size_hint_mb=330,
    ),
    ModelAsset(
        key="clip-openclip",
        kind="open_clip",
        arch="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        purpose="open_clip ViT-B-32 (clustering, semantic filter)",
        size_hint_mb=350,
    ),
    ModelAsset(
        key="clip-hf",
        kind="hf",
        purpose="openai/clip-vit-base-patch32 (fallback when open_clip "
                "is unavailable)",
        size_hint_mb=600,
        repo_id="openai/clip-vit-base-patch32",
        files=(),                       # snapshot; resolved lazily
        optional=True,
    ),
    ModelAsset(
        key="realesrgan-x4plus",
        kind="url",
        purpose="Real-ESRGAN x4 (garment/fabric detail upscale)",
        size_hint_mb=64,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.1.0/RealESRGAN_x4plus.pth",
        dest="realesrgan/RealESRGAN_x4plus.pth",
        optional=True,
    ),
    ModelAsset(
        key="gfpgan-v1.4",
        kind="url",
        purpose="GFPGAN v1.4 (conservative face restoration)",
        size_hint_mb=333,
        url="https://github.com/TencentARC/GFPGAN/releases/download/"
            "v1.3.4/GFPGANv1.4.pth",
        dest="gfpgan/GFPGANv1.4.pth",
        optional=True,
    ),
)}

# Face-swap (inswapper_128.onnx) is deliberately NOT in the registry: its
# distribution terms require the user to obtain it themselves; the pipeline
# only consumes it when an explicit path is configured.


# ---------------------------------------------------------------------------
# status (offline)
# ---------------------------------------------------------------------------

def _hf_cache_paths(asset: ModelAsset) -> list[Path]:
    """Best-effort: would these HF files be served from the local cache?"""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return []
    found = []
    targets = asset.files or ("config.json",)
    for f in targets:
        p = try_to_load_from_cache(asset.repo_id, f)
        if isinstance(p, str):
            found.append(Path(p))
    return found


def asset_path(asset: ModelAsset) -> Path | None:
    """Where the asset lives (or would live) on disk; None if not local."""
    if asset.kind == "url":
        return MODELS_DIR / asset.dest
    if asset.kind == "insightface":
        return Path.home() / ".insightface" / "models" / asset.bundle
    cached = _hf_cache_paths(asset)
    return cached[0] if cached else None


def is_installed(asset: ModelAsset) -> bool:
    if asset.kind == "url":
        p = MODELS_DIR / asset.dest
        return p.is_file() and p.stat().st_size > 0
    if asset.kind == "insightface":
        d = Path.home() / ".insightface" / "models" / asset.bundle
        return d.is_dir() and any(d.glob("*.onnx"))
    if asset.kind == "open_clip":
        # open_clip caches under ~/.cache/clip or HF hub cache depending on
        # version; treat "locally resolvable" as unknown (False = will
        # auto-fetch quickly from cache if present).
        try:
            import open_clip  # noqa: F401
            from huggingface_hub import try_to_load_from_cache
            p = try_to_load_from_cache(
                "laion/CLIP-ViT-B-32-laion2B-s34B-b79k", "open_clip_pytorch_model.bin")
            return isinstance(p, str)
        except Exception:
            return False
    # hf
    return bool(_hf_cache_paths(asset))


def status() -> list[dict]:
    """Offline status table for every registered model."""
    rows = []
    for asset in MODEL_REGISTRY.values():
        ok = is_installed(asset)
        rows.append({
            "key": asset.key,
            "installed": ok,
            "path": str(asset_path(asset)) if ok else None,
            "purpose": asset.purpose,
            "size_mb": asset.size_hint_mb,
            "optional": asset.optional,
        })
    return rows


# ---------------------------------------------------------------------------
# download (online, explicit)
# ---------------------------------------------------------------------------

def _download_url(asset: ModelAsset, progress) -> Path:
    dest = MODELS_DIR / asset.dest
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    progress(f"downloading {asset.key} "
             f"(~{asset.size_hint_mb} MB) -> {dest}")
    req = urllib.request.Request(asset.url, headers={"User-Agent":
                                                     "pixel-curator"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    if asset.sha256:
        import hashlib
        h = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if h != asset.sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{asset.key}: sha256 mismatch "
                               f"(got {h}, expected {asset.sha256})")
    os.replace(tmp, dest)
    logger.info("downloaded %s -> %s", asset.key, dest)
    return dest


def _download_hf(asset: ModelAsset, progress) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download
    progress(f"fetching {asset.key} from {asset.repo_id} "
             f"(~{asset.size_hint_mb} MB)")
    if asset.files:
        for f in asset.files:
            hf_hub_download(repo_id=asset.repo_id, filename=f)
    else:
        snapshot_download(repo_id=asset.repo_id)


def _download_open_clip(asset: ModelAsset, progress) -> None:
    import open_clip
    progress(f"fetching {asset.key} ({asset.arch}/{asset.pretrained}, "
             f"~{asset.size_hint_mb} MB)")
    open_clip.create_model_and_transforms(asset.arch,
                                          pretrained=asset.pretrained)


def _download_insightface(asset: ModelAsset, progress) -> None:
    from insightface.app import FaceAnalysis
    progress(f"fetching {asset.key} bundle (~{asset.size_hint_mb} MB)")
    FaceAnalysis(name=asset.bundle)
    # no .prepare() here - device selection stays with the caller


def ensure_model(key: str, progress=print) -> Path | None:
    """Download ``key`` on first use; return its local path if file-based.

    Raises KeyError for unknown keys and propagates network errors with
    context. Idempotent: already-present assets return immediately.
    """
    asset = MODEL_REGISTRY.get(key)
    if asset is None:
        raise KeyError(f"unknown model '{key}'; known: "
                       f"{', '.join(sorted(MODEL_REGISTRY))}")
    if is_installed(asset):
        return asset_path(asset)
    try:
        if asset.kind == "url":
            return _download_url(asset, progress)
        if asset.kind == "hf":
            _download_hf(asset, progress)
        elif asset.kind == "open_clip":
            _download_open_clip(asset, progress)
        elif asset.kind == "insightface":
            _download_insightface(asset, progress)
    except Exception as exc:
        raise RuntimeError(
            f"failed to download model '{key}' ({asset.purpose}): {exc}\n"
            f"Check your network connection, then retry - or download it "
            f"manually and place it under {MODELS_DIR}.") from exc
    return asset_path(asset)


def download_missing(keys: list[str] | None = None,
                     include_optional: bool = True,
                     progress=print) -> dict[str, bool]:
    """Fetch every missing model (or just ``keys``). Returns key -> ok."""
    wanted = keys or list(MODEL_REGISTRY)
    results = {}
    for key in wanted:
        asset = MODEL_REGISTRY.get(key)
        if asset is None:
            progress(f"unknown model: {key}")
            results[key] = False
            continue
        if asset.optional and not include_optional and not keys:
            continue
        if is_installed(asset):
            progress(f"OK (cached): {key}")
            results[key] = True
            continue
        try:
            ensure_model(key, progress=progress)
            results[key] = True
        except Exception as exc:
            progress(f"FAILED: {key}: {exc}")
            results[key] = False
    return results
