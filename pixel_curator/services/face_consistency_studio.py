"""Shared service - Face Consistency Studio.

Local-first, consent-aware face restoration + identity-consistency for the
user's OWN generated images. Given target image(s) and a high-quality
consenting source-face reference, it (optionally) applies inswapper identity
transfer, GFPGAN restoration, and RealESRGAN upscaling, writing improved copies
into an output folder that MIRRORS the input folder structure, alongside
before/after previews and provenance manifests.

Design rules honored here:
  * NO heavy imports at module top - torch / onnxruntime / insightface / gfpgan
    / realesrgan / cv2 / PIL are imported lazily inside the functions that need
    them, so importing this module (and the GUI) never loads a model.
  * Backend detection does BOUNDED globs inside known model folders only - it
    never walks the home dir / SSD root (that used to hang the UI).
  * Models are loaded ONCE and cached (inswapper FaceAnalysis + swapper), never
    re-instantiated per image.
  * Source images are NEVER modified, moved, or deleted. Every operation runs
    on a COPY written into the output folder.
  * Models are detected, never downloaded. Missing backends degrade to a clear
    warning and the pipeline continues safely.

All processing is local and user-controlled.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

SERVICE_NAME = "face_consistency_studio"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
OUTPUT_FILES = {
    "manifest": "face_fix_manifest.json",
    "summary": "face_fix_summary.json",
    "card": "face_fix_card.md",
}

# module-level caches (load once, reuse across images - never per-image reload)
_INSWAP: dict[str, Any] = {"app": None, "swapper": None, "model": None,
                           "err_logged": False}
_POST: dict[str, Any] = {"proc": None, "key": None}


def _ensure_post(esr_path, gfp_path):
    """Cache the GFPGAN/RealESRGAN PostProcessor so its models load ONCE."""
    key = (str(esr_path), str(gfp_path))
    if _POST["proc"] is not None and _POST["key"] == key:
        return _POST["proc"]
    from .. import post_processing as pp
    _POST["proc"] = pp.PostProcessor(esr_path, gfp_path)
    _POST["key"] = key
    return _POST["proc"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FaceConsistencyConfig:
    service_version: str = "face_consistency_studio_v1"
    output_dir: str = ""
    mode: str = "auto"                       # auto | single | batch
    forge_models_root: str = ""              # e.g. <reForge>/models
    link_originals: bool = False
    overwrite: bool = False
    preserve_structure: bool = True          # mirror input subfolders in output
    face_backend: str = "auto"               # auto | onnx | disabled
    face_model_path: str = ""                # explicit inswapper*.onnx
    face_detection_model_path: str = ""
    apply_face_alignment: bool = True
    apply_identity_transfer: bool = True
    apply_gfpgan: bool = True
    gfpgan_strength: float = 0.30
    gfpgan_model_path: str = ""              # explicit GFPGAN*.pth
    apply_realesrgan: bool = True
    realesrgan_model_path: str = ""          # explicit RealESRGAN*.pth
    realesrgan_tile: int = 0
    realesrgan_scale: int = 2
    save_before_after_preview: bool = True
    save_manifest: bool = True
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class FaceFixImageResult:
    target_image: str
    output_image: str = ""
    preview_image: str = ""
    source_face_image: str = ""
    ok: bool = False
    faces_detected_target: int = 0
    faces_detected_source: int = 0
    steps_applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_image": self.target_image,
            "source_face_image": self.source_face_image,
            "output_image": self.output_image,
            "preview_image": self.preview_image,
            "ok": self.ok,
            "faces_detected_target": self.faces_detected_target,
            "faces_detected_source": self.faces_detected_source,
            "steps_applied": list(self.steps_applied),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }


@dataclass
class FaceFixBatchResult:
    output_dir: str
    total_images: int = 0
    processed_images: int = 0
    failed_images: int = 0
    skipped_images: int = 0
    results: list[FaceFixImageResult] = field(default_factory=list)
    manifest_path: str = ""
    summary_path: str = ""
    card_path: str = ""
    backends: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": SERVICE_NAME,
            "output_dir": self.output_dir,
            "dry_run": self.dry_run,
            "total_images": self.total_images,
            "processed_images": self.processed_images,
            "failed_images": self.failed_images,
            "skipped_images": self.skipped_images,
            "manifest_path": self.manifest_path,
            "summary_path": self.summary_path,
            "card_path": self.card_path,
            "backends": self.backends,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Backend detection (BOUNDED - never walks home/SSD root, no model loading)
# ---------------------------------------------------------------------------
def _spec_available(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _model_roots(config: Optional["FaceConsistencyConfig"]) -> list[Path]:
    home = Path.home()
    roots: list[Path] = []
    if config and config.forge_models_root:
        roots.append(Path(config.forge_models_root).expanduser())
    roots += [
        home / "FORGE_REPO" / "stable-diffusion-webui-reForge" / "models",
        home / "FORGE_REPO" / "stable-diffusion-webui-forge" / "models",
        home / "stable-diffusion-webui-reForge" / "models",
        home / "stable-diffusion-webui-forge" / "models",
        home / "stable-diffusion-webui" / "models",
        home / "PIXEL-CURATOR" / "models",
    ]
    try:
        from .. import post_processing as pp
        roots += [Path(pp.DEFAULT_REALESRGAN_MODEL).parent,
                  Path(pp.DEFAULT_GFPGAN_MODEL).parent]
    except Exception:
        pass
    # de-dup, keep order
    return list(dict.fromkeys(roots))


def _find_model(roots: list[Path], subdirs: tuple[str, ...],
                patterns: tuple[str, ...], min_bytes: int = 1_000_000) -> str:
    """Bounded search: glob (depth 1) inside specific model subfolders only.
    Skips empty/placeholder files (< min_bytes) and prefers the largest real
    match, so a 0-byte download is never reported as a usable model."""
    for root in roots:
        for sub in subdirs:
            d = root / sub if sub else root
            try:
                if not d.is_dir():
                    continue
                for pat in patterns:
                    hits = sorted(
                        (p for p in d.glob(pat)
                         if p.is_file() and p.stat().st_size >= min_bytes),
                        key=lambda p: p.stat().st_size, reverse=True)
                    if hits:
                        return str(hits[0])
            except Exception:
                continue
    return ""


def detect_face_consistency_backends(
        search_roots: list[Path] | None = None,
        config: FaceConsistencyConfig | None = None) -> dict[str, Any]:
    """Detect which backends are available WITHOUT importing or loading any
    model. Returns a structured status dict (incl. resolved model paths)."""
    cfg = config or FaceConsistencyConfig()
    roots = search_roots or _model_roots(cfg)

    insw = cfg.face_model_path or _find_model(
        roots, ("insightface", ""), ("inswapper*.onnx", "*inswapper*.onnx"))
    identity = {
        "name": "identity_transfer (inswapper)",
        "model_path": insw,
        "model_found": bool(insw) and Path(insw).is_file(),
        "runtime": _spec_available("insightface") and _spec_available("onnxruntime"),
    }
    identity["available"] = bool(identity["model_found"] and identity["runtime"])

    gfp = cfg.gfpgan_model_path or _find_model(
        roots, ("GFPGAN", "gfpgan", ""),
        ("GFPGAN*.pth", "*GFPGAN*.pth", "gfpgan*.pth"))
    gfpgan = {
        "name": "gfpgan_restore",
        "model_path": gfp,
        "model_found": bool(gfp) and Path(gfp).is_file(),
        "runtime": _spec_available("gfpgan"),
    }
    gfpgan["available"] = bool(gfpgan["model_found"] and gfpgan["runtime"])

    esr = cfg.realesrgan_model_path or _find_model(
        roots, ("ESRGAN", "RealESRGAN", "realesrgan", ""),
        ("RealESRGAN_x4plus*.pth", "RealESRGAN*.pth", "*x4plus*.pth"))
    realesrgan = {
        "name": "realesrgan_upscale",
        "model_path": esr,
        "model_found": bool(esr) and Path(esr).is_file(),
        "runtime": _spec_available("realesrgan") and _spec_available("basicsr"),
    }
    realesrgan["available"] = bool(realesrgan["model_found"] and realesrgan["runtime"])

    any_avail = any(b["available"] for b in (identity, gfpgan, realesrgan))
    notes = []
    if not identity["available"]:
        notes.append("identity transfer unavailable: "
                     + ("inswapper model not found" if not identity["model_found"]
                        else "insightface/onnxruntime runtime missing"))
    if not gfpgan["available"]:
        notes.append("GFPGAN unavailable: "
                     + ("GFPGAN*.pth not found" if not gfpgan["model_found"]
                        else "gfpgan runtime missing"))
    if not realesrgan["available"]:
        notes.append("RealESRGAN unavailable: "
                     + ("RealESRGAN*.pth not found" if not realesrgan["model_found"]
                        else "realesrgan/basicsr runtime missing"))
    return {
        "service": SERVICE_NAME,
        "any_backend_available": any_avail,
        "identity_transfer": identity,
        "gfpgan": gfpgan,
        "realesrgan": realesrgan,
        "searched_roots": [str(r) for r in roots],
        "notes": notes,
        "local_processing_only": True,
        "downloads_disabled": True,
    }


# ---------------------------------------------------------------------------
# inswapper identity transfer (explicit model path, cached, lazy)
# ---------------------------------------------------------------------------
def _rt_providers() -> list[str]:
    try:
        from ..identity_integration import _rt_providers as _p
        return _p()
    except Exception:
        try:
            import onnxruntime as ort
            avail = ort.get_available_providers()
            for pref in ("CoreMLExecutionProvider", "CUDAExecutionProvider"):
                if pref in avail:
                    return [pref, "CPUExecutionProvider"]
        except Exception:
            pass
        return ["CPUExecutionProvider"]


def _ensure_inswap(model_path: str):
    if _INSWAP["swapper"] is not None and _INSWAP["model"] == model_path:
        return _INSWAP
    import insightface
    from insightface.app import FaceAnalysis
    providers = _rt_providers()
    # cache the analysis app ONCE (so a swapper-load failure can't reload it)
    if _INSWAP["app"] is None:
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0, det_size=(640, 640))
        _INSWAP["app"] = app
    # load the swapper - try with providers kwarg, then without (version-safe)
    swapper, last = None, None
    for kw in ({"providers": providers, "download": False, "download_zip": False},
               {"providers": providers}, {}):
        try:
            swapper = insightface.model_zoo.get_model(model_path, **kw)
            if swapper is not None:
                break
        except Exception as exc:
            last = exc
    if swapper is None:
        if not _INSWAP["err_logged"]:
            print(f"[face-studio] inswapper failed to load from {model_path}: "
                  f"{last!r}", file=sys.stderr)
            _INSWAP["err_logged"] = True
        raise RuntimeError(f"inswapper load failed: {last}")
    _INSWAP.update(swapper=swapper, model=model_path)
    return _INSWAP


def _run_inswap(image_path: Path, ref_path: Path, model_path: str,
                warnings: list[str]) -> tuple[bool, int, int]:
    """Swap the reference identity onto the (already-copied) image in place.
    Returns (ok, n_target_faces, n_ref_faces). Never raises."""
    try:
        import cv2
        eng = _ensure_inswap(model_path)
        img = cv2.imread(str(image_path))
        ref = cv2.imread(str(ref_path))
        if img is None or ref is None:
            warnings.append("identity: could not read target/reference image")
            return (False, 0, 0)
        tfaces = eng["app"].get(img)
        rfaces = eng["app"].get(ref)
        if not tfaces:
            warnings.append("identity: no face detected in target")
            return (False, 0, len(rfaces))
        if not rfaces:
            warnings.append("identity: no face detected in reference")
            return (False, len(tfaces), 0)
        # most prominent target face (largest bbox)
        tfaces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True)
        out = eng["swapper"].get(img, tfaces[0], rfaces[0], paste_back=True)
        cv2.imwrite(str(image_path), out)
        return (True, len(tfaces), len(rfaces))
    except Exception as exc:
        warnings.append(f"identity transfer failed (skipped): {exc}")
        return (False, 0, 0)


# ---------------------------------------------------------------------------
# Target collection (recursive - walks subfolders to any depth)
# ---------------------------------------------------------------------------
def collect_target_images(target_path: str | Path) -> list[Path]:
    """Single image OR a folder walked RECURSIVELY (any depth). Sorted.
    Returns [] for a missing path."""
    p = Path(target_path).expanduser()
    if not p.exists():
        return []
    if p.is_file():
        return [p] if p.suffix.lower() in IMAGE_EXTS else []
    out = [q for q in p.rglob("*")
           if q.is_file() and q.suffix.lower() in IMAGE_EXTS
           and "_previews" not in q.parts and "face_fixed" not in q.parts]
    return sorted(out)


def _resolve_output_dir(target_path: str | Path,
                        config: FaceConsistencyConfig) -> Path:
    if config.output_dir:
        return Path(config.output_dir).expanduser()
    p = Path(target_path).expanduser()
    parent = p.parent if p.is_file() else p
    return parent / "face_fixed"


def _provenance(target: Path, source_face: Path,
                config: FaceConsistencyConfig) -> dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "service_version": config.service_version,
        "source_face_path": str(source_face),
        "target_image_path": str(target),
        "processing_mode": config.mode,
        "user_provided_reference": True,
        "local_processing_only": True,
        "downloads_disabled": True,
    }


# ---------------------------------------------------------------------------
# Core processing (one image) - never touches the source
# ---------------------------------------------------------------------------
def _make_preview(before: Path, after: Path, dest: Path,
                  warnings: list[str]) -> str:
    try:
        from PIL import Image
        a = Image.open(before).convert("RGB")
        b = Image.open(after).convert("RGB")
        h = min(a.height, b.height, 768) or 1

        def _rs(im):
            w = max(1, int(im.width * (h / im.height)))
            return im.resize((w, h))
        a, b = _rs(a), _rs(b)
        canvas = Image.new("RGB", (a.width + b.width + 8, h), (20, 20, 20))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width + 8, 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dest, quality=90)
        return str(dest)
    except Exception as exc:
        warnings.append(f"preview skipped: {exc}")
        return ""


def _process_one(target: Path, source_face: Path, config: FaceConsistencyConfig,
                 out_path: Path, preview_path: Path,
                 backends: dict[str, Any]) -> FaceFixImageResult:
    res = FaceFixImageResult(
        target_image=str(target), source_face_image=str(source_face))
    res.metadata = _provenance(target, source_face, config)

    if not target.is_file():
        res.errors.append(f"target image not found: {target}")
        return res
    if not source_face.is_file():
        res.errors.append(f"source face image not found: {source_face}")
        return res

    bid, bg, be = (backends["identity_transfer"], backends["gfpgan"],
                   backends["realesrgan"])
    planned: list[str] = []
    if config.apply_identity_transfer:
        planned.append("identity_transfer" if bid["available"]
                       else "identity_transfer(unavailable->skip)")
    if config.apply_gfpgan:
        planned.append(f"gfpgan_restore_strength_{config.gfpgan_strength:.2f}"
                       if bg["available"] else "gfpgan_restore(unavailable->skip)")
    if config.apply_realesrgan:
        planned.append(f"realesrgan_upscale_tile_{config.realesrgan_tile}"
                       if be["available"] else "realesrgan_upscale(unavailable->skip)")

    if config.dry_run:
        res.ok = True
        res.steps_applied = planned
        res.warnings.append("dry-run: no image written")
        res.output_image = str(out_path)
        res.metadata["dry_run"] = True
        return res

    if out_path.exists() and not config.overwrite:
        res.warnings.append(f"output exists, skipped (use overwrite): {out_path}")
        res.output_image = str(out_path)
        res.ok = True
        return res

    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_path.parent / f"_work_{out_path.stem}{target.suffix.lower() or '.png'}"
    try:
        shutil.copy2(target, work)
    except Exception as exc:
        res.errors.append(f"could not stage working copy: {exc}")
        return res

    applied: list[str] = []

    # 1) identity transfer (inswapper) on the COPY, with the resolved model path
    if config.apply_identity_transfer and bid["available"]:
        ok, nt, ns = _run_inswap(work, source_face, bid["model_path"], res.warnings)
        res.faces_detected_target, res.faces_detected_source = nt, ns
        if ok:
            applied.append("identity_transfer")
    elif config.apply_identity_transfer:
        res.warnings.append("identity transfer backend unavailable; skipped")

    # 2) GFPGAN + 3) RealESRGAN via the shared PostProcessor (resolved paths)
    if (config.apply_gfpgan and bg["available"]) or \
       (config.apply_realesrgan and be["available"]):
        try:
            from .. import post_processing as pp
            esr_path = be["model_path"] or config.realesrgan_model_path or \
                pp.DEFAULT_REALESRGAN_MODEL
            gfp_path = bg["model_path"] or config.gfpgan_model_path or \
                pp.DEFAULT_GFPGAN_MODEL
            processor = _ensure_post(esr_path, gfp_path)
            pr = processor.process(
                work,
                use_realesrgan=config.apply_realesrgan and be["available"],
                use_gfpgan=config.apply_gfpgan and bg["available"],
                gfpgan_weight=config.gfpgan_strength)
            res.warnings.extend(pr.warnings)
            if pr.gfpgan_applied:
                applied.append(f"gfpgan_restore_strength_{config.gfpgan_strength:.2f}")
            if pr.realesrgan_applied:
                applied.append(f"realesrgan_upscale_tile_{config.realesrgan_tile}")
            produced = Path(pr.post_processed_path)
            if produced != work and produced.is_file():
                shutil.move(str(produced), str(out_path))
                if work.exists():
                    work.unlink()
            else:
                shutil.move(str(work), str(out_path))
        except Exception as exc:
            res.warnings.append(f"post-processing failed (kept prior result): {exc}")
            if work.exists():
                shutil.move(str(work), str(out_path))
    else:
        if work.exists():
            shutil.move(str(work), str(out_path))

    res.output_image = str(out_path)
    res.steps_applied = applied or ["copy_only(no_backend_applied)"]
    res.ok = out_path.is_file()
    if not applied:
        res.warnings.append("no enhancement backend available; output is a "
                            "faithful copy of the source target")

    if config.save_before_after_preview and res.ok:
        res.preview_image = _make_preview(target, out_path, preview_path, res.warnings)
    return res


def process_face_consistency_single(
        target_image: str | Path, source_face_image: str | Path,
        config: FaceConsistencyConfig | None = None) -> FaceFixImageResult:
    cfg = config or FaceConsistencyConfig()
    target = Path(target_image).expanduser()
    source = Path(source_face_image).expanduser()
    out_dir = _resolve_output_dir(target, cfg)
    backends = detect_face_consistency_backends(config=cfg)
    out_path = out_dir / "images" / f"{target.stem}.png"
    preview_path = out_dir / "_previews" / f"{target.stem}.jpg"
    return _process_one(target, source, cfg, out_path, preview_path, backends)


def process_face_consistency_batch(
        target_path: str | Path, source_face_image: str | Path,
        config: FaceConsistencyConfig | None = None) -> FaceFixBatchResult:
    cfg = config or FaceConsistencyConfig()
    source = Path(source_face_image).expanduser()
    base = Path(target_path).expanduser()
    out_dir = _resolve_output_dir(target_path, cfg)
    res = FaceFixBatchResult(output_dir=str(out_dir), dry_run=cfg.dry_run)
    res.backends = detect_face_consistency_backends(config=cfg)

    if not source.is_file():
        res.errors.append(f"source face image not found: {source}")
        return res
    targets = collect_target_images(target_path)
    if not targets:
        res.errors.append(f"no target images found at: {target_path}")
        return res

    res.total_images = len(targets)
    images_root = out_dir / "images"
    previews_root = out_dir / "_previews"
    t0 = time.time()
    try:
        for idx, t in enumerate(targets, 1):
            # mirror the input subfolder structure under the output dir
            rel = t.relative_to(base).parent if base.is_dir() else Path(".")
            out_path = images_root / rel / f"{t.stem}.png"
            preview_path = previews_root / rel / f"{t.stem}.jpg"
            r = _process_one(t, source, cfg, out_path, preview_path, res.backends)
            res.results.append(r)
            if r.errors:
                res.failed_images += 1
            elif any("skipped" in w for w in r.warnings):
                res.skipped_images += 1
                res.processed_images += 1
            else:
                res.processed_images += 1
            if idx % 10 == 0 or idx == res.total_images:
                el = time.time() - t0
                rate = idx / max(0.001, el)
                eta = int((res.total_images - idx) / max(0.001, rate))
                print(f"\rFace-fix {idx}/{res.total_images} "
                      f"({rate:.1f} img/s, {res.failed_images} failed, "
                      f"{res.skipped_images} skipped, ETA "
                      f"{eta//3600:d}:{eta%3600//60:02d}:{eta%60:02d})  ",
                      end="", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        res.warnings.append(
            f"interrupted at {len(res.results)}/{res.total_images}; re-run the "
            f"same command (without --overwrite) to resume the remaining images")
        print(f"\ninterrupted at {len(res.results)}/{res.total_images} - "
              f"manifest written for completed images; re-run to resume",
              file=sys.stderr)
    print("", file=sys.stderr)

    if cfg.save_manifest and not cfg.dry_run and res.results:
        paths = write_face_fix_manifest(res, out_dir)
        res.manifest_path = str(paths.get("manifest", ""))
        res.summary_path = str(paths.get("summary", ""))
        res.card_path = str(paths.get("card", ""))
    return res


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_face_fix_manifest(result: FaceFixBatchResult,
                            output_dir: Path) -> dict[str, Path]:
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    manifest = {
        "service": SERVICE_NAME, "output_dir": str(out),
        "local_processing_only": True, "user_provided_reference": True,
        "counts": {"total": result.total_images,
                   "processed": result.processed_images,
                   "failed": result.failed_images,
                   "skipped": result.skipped_images},
        "backends": result.backends,
        "images": [r.to_dict() for r in result.results],
    }
    mp = out / OUTPUT_FILES["manifest"]
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["manifest"] = mp

    summary = {
        "service": SERVICE_NAME, "output_dir": str(out),
        "total": result.total_images, "processed": result.processed_images,
        "failed": result.failed_images, "skipped": result.skipped_images,
        "backend_available": result.backends.get("any_backend_available", False),
        "manifest_path": str(mp), "preview_dir": str(out / "_previews"),
        "warnings": result.warnings,
    }
    sp = out / OUTPUT_FILES["summary"]
    sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["summary"] = sp

    card = (f"# Face Consistency Studio\n\n"
            f"- Output: `{out}`\n"
            f"- Total target images: {result.total_images}\n"
            f"- Processed: {result.processed_images} | "
            f"failed: {result.failed_images} | skipped: {result.skipped_images}\n"
            f"- Backends available: "
            f"{result.backends.get('any_backend_available', False)}\n\n"
            f"All processing was local and user-controlled. Source images were "
            f"never modified. Identity reference was user-provided. Output mirrors "
            f"the input folder structure.\n")
    cp = out / OUTPUT_FILES["card"]
    cp.write_text(card, encoding="utf-8")
    paths["card"] = cp
    return paths


def write_ui_status_summary(result: FaceFixBatchResult,
                            output_base: Path) -> Path:
    base = Path(output_base).expanduser() / "ui_status"
    base.mkdir(parents=True, exist_ok=True)
    payload = {
        "service": SERVICE_NAME,
        "processed": result.processed_images,
        "failed": result.failed_images,
        "skipped": result.skipped_images,
        "total": result.total_images,
        "backend_availability": result.backends.get("any_backend_available", False),
        "backends": result.backends,
        "output_dir": result.output_dir,
        "manifest_path": result.manifest_path,
        "preview_dir": str(Path(result.output_dir) / "_previews"),
        "warnings": result.warnings,
        "dry_run": result.dry_run,
    }
    p = base / "face_consistency_summary.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


def result_to_dataset_explorer_fields(
        result: FaceFixImageResult) -> dict[str, Any]:
    return {
        Path(result.target_image).name: {
            "face_fixed_available": result.ok,
            "face_fixed_output_path": result.output_image,
            "face_fix_status": "ok" if result.ok else (
                "error" if result.errors else "skipped"),
            "face_fix_steps": list(result.steps_applied),
        }
    }


__all__ = [
    "FaceConsistencyConfig", "FaceFixImageResult", "FaceFixBatchResult",
    "SERVICE_NAME", "OUTPUT_FILES",
    "detect_face_consistency_backends", "collect_target_images",
    "process_face_consistency_single", "process_face_consistency_batch",
    "write_face_fix_manifest", "write_ui_status_summary",
    "result_to_dataset_explorer_fields",
]
