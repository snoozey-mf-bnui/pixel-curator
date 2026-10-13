"""Priority 10 - Kohya / Colab dataset packager.

Turns a curated, validated export (``images/`` + ``captions/`` +
``dataset_manifest.json``) into a training-ready package whose layout matches
the project's SDXL DreamBooth / LoRA Colab notebooks exactly:

  * DreamBooth: ``<out>/{dataset_config.toml, instance/000001.jpg + .txt, ...}``
  * LoRA (multi-concept): ``<out>/concept_<target>/{dataset_config.toml,
    instance/...}`` for each target, zipped together.

Contract: deterministic, stdlib-only, fully source-safe (curated exports are
read, never modified/moved/deleted), default DRY-RUN (report only; ``apply``
copies images + writes ``.txt`` captions/config/manifest/card/zip). Captions are
assembled from existing metadata - no AI models, no network, no GPU. Never
raises for ordinary input.
"""
from __future__ import annotations

import json
import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "kohya_package_v1"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)  # fixed timestamp -> reproducible archives


@dataclass
class CaptionTemplateConfig:
    identity_token: str = "spikbnu person"
    class_token: str = "person"
    append_quality_tags: bool = True
    append_framing_tags: bool = True
    append_region_tags: bool = True
    append_pose_tags: bool = True
    max_caption_words: int = 75
    lowercase: bool = False
    architecture: str = "sdxl"   # SDXL/Pony only

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "identity_token", "class_token", "append_quality_tags",
            "append_framing_tags", "append_region_tags", "append_pose_tags",
            "max_caption_words", "lowercase", "architecture")}


@dataclass
class KohyaPackageConfig:
    mode: str = "dreambooth"            # dreambooth | lora
    identity_token: str = "spikbnu person"
    class_token: str = "person"
    concept_name: str = "core_identity"
    image_ext: str = ".jpg"
    overwrite_captions: bool = False
    copy_images: bool = True
    create_zip: bool = True
    stable_filenames: bool = True
    start_index: int = 1
    digits: int = 6
    resolution: int = 1024
    repeats: int = 1
    dry_run: bool = True
    caption_template: CaptionTemplateConfig = field(
        default_factory=CaptionTemplateConfig)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode, "identity_token": self.identity_token,
            "class_token": self.class_token, "concept_name": self.concept_name,
            "image_ext": self.image_ext,
            "overwrite_captions": self.overwrite_captions,
            "copy_images": self.copy_images, "create_zip": self.create_zip,
            "stable_filenames": self.stable_filenames,
            "start_index": self.start_index, "digits": self.digits,
            "resolution": self.resolution, "repeats": self.repeats,
            "dry_run": self.dry_run,
            "caption_template": self.caption_template.to_dict(),
        }


@dataclass
class PackagedSample:
    index: int
    source_image: str
    source_caption: str | None
    package_image: str
    package_caption: str
    caption_text: str
    lora_target: str | None = None
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index, "source_image": self.source_image,
            "source_caption": self.source_caption,
            "package_image": self.package_image,
            "package_caption": self.package_caption,
            "caption_text": self.caption_text, "lora_target": self.lora_target,
            "warnings": list(self.warnings),
        }


@dataclass
class KohyaPackageReport:
    source_dataset: str
    output_dir: str
    mode: str
    concept_name: str
    sample_count: int
    dry_run: bool
    zip_path: str | None = None
    dataset_config_path: str | None = None
    package_manifest_path: str | None = None
    package_card_path: str | None = None
    samples: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_dataset": self.source_dataset, "output_dir": self.output_dir,
            "mode": self.mode, "concept_name": self.concept_name,
            "sample_count": self.sample_count, "dry_run": self.dry_run,
            "zip_path": self.zip_path,
            "dataset_config_path": self.dataset_config_path,
            "package_manifest_path": self.package_manifest_path,
            "package_card_path": self.package_card_path,
            "samples": [s.to_dict() if isinstance(s, PackagedSample) else s
                        for s in self.samples],
            "warnings": list(self.warnings), "summary": self.summary,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def registry_files(report: KohyaPackageReport) -> list:
    """dataset_version_files entries for a packaged report (Phase 2).

    Gives kohya packages the same path-based membership as other folder
    exports, so ``pixel-curator diff`` and per-version change summaries work
    for them too."""
    files: list = []
    for s in report.samples:
        d = s.to_dict() if isinstance(s, PackagedSample) else dict(s)
        if d.get("package_image"):
            files.append({"rel_path": d["package_image"],
                          "source_path": d.get("source_image"),
                          "caption": d.get("caption_text")})
    return files


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_curated_dataset(dataset_dir) -> list:
    """Load curated samples from ``dataset_manifest.json`` (preferred) or, if
    absent, by pairing ``images/`` with ``captions/``. Never raises."""
    dataset_dir = Path(dataset_dir)
    man = dataset_dir / "dataset_manifest.json"
    if man.is_file():
        try:
            data = json.loads(man.read_text(encoding="utf-8"))
            samples = data.get("samples") if isinstance(data, dict) else None
            if samples:
                return list(samples)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass
    # fallback: scan images/ + captions/
    out = []
    img_dir = dataset_dir / "images"
    cap_dir = dataset_dir / "captions"
    if img_dir.is_dir():
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                cap = cap_dir / f"{p.stem}.txt"
                out.append({
                    "image_target": f"images/{p.name}",
                    "caption": cap.read_text(encoding="utf-8").strip()
                    if cap.is_file() else None})
    return out


# ---------------------------------------------------------------------------
# Caption assembly (deterministic, no models)
# ---------------------------------------------------------------------------
def _flat(sample: dict) -> dict:
    out = dict(sample or {})
    for nest in ("signals", "metadata"):
        blk = (sample or {}).get(nest)
        if isinstance(blk, dict):
            for k, v in blk.items():
                out.setdefault(k, v)
    return out


def _sanitize_token(t: str) -> str:
    return " ".join(str(t).replace(",", " ").split()).strip()


_FRAMING_PHRASE = {
    "full_body": "full body", "torso_focus": "torso focus",
    "upper_body": "upper body", "close_crop": "close crop",
    "face_closeup": "face closeup", "detail_crop": "detail crop",
}
_ANGLE_PHRASE = {"front": "front view", "three_quarter": "three quarter view",
                 "side": "side view", "back": "back view"}
_POSE_PHRASE = {"standing": "standing pose", "seated": "seated pose",
                "reclined": "reclined pose", "dynamic": "dynamic pose",
                "full_body": "full body pose"}


def _target_descriptors(target: str, f: dict, cfg: CaptionTemplateConfig
                        ) -> list:
    framing = _FRAMING_PHRASE.get(str(f.get("framing_label")
                                      or f.get("coverage_label") or "").lower())
    angle = _ANGLE_PHRASE.get(str(f.get("body_angle_bucket") or "").lower())
    pose = _POSE_PHRASE.get(str(f.get("pose_label")
                                or f.get("pose_type") or "").lower())
    clothing = f.get("clothing_type_label")
    clothing = None if clothing in (None, "unknown") else \
        str(clothing).replace("_", " ")
    lighting = None
    ls = f.get("lighting_score")
    if ls is not None:
        lighting = "studio lighting" if float(ls) >= 0.6 else "natural lighting"
    out: list = []
    t = (target or "").lower()
    if t == "core_identity":
        out += [framing or "portrait framing", "sharp face"]
        if (f.get("region_balance_score") or 0) and \
                float(f.get("region_balance_score")) >= 0.55:
            out.append("balanced composition")
    elif t == "outfit_clothing":
        out += [framing or "torso focus", clothing or "outfit",
                "garment detail"]
        if f.get("texture_preservation_score"):
            out.append("natural texture")
    elif t == "pose_action":
        out += [framing or "full body", pose or "dynamic pose", angle]
    elif t == "style_aesthetic":
        out += [lighting or "soft lighting", "color mood", "composition"]
    elif t == "garment_detail":
        out += ["detail crop", "garment texture", clothing or "material"]
    elif t == "catalog_reference":
        out += [framing or "full body", clothing or "outfit",
                lighting or "natural lighting", pose or "standing pose"]
    else:
        out += [framing, pose, angle]
    # optional appended tag groups
    if cfg.append_framing_tags and framing and framing not in out:
        out.append(framing)
    if cfg.append_pose_tags and pose and pose not in out:
        out.append(pose)
    if cfg.append_region_tags and angle and angle not in out:
        out.append(angle)
    if cfg.append_quality_tags and lighting and lighting not in out:
        out.append(lighting)
    return [d for d in out if d]


def build_caption(sample: dict, config: CaptionTemplateConfig) -> str:
    """Deterministically assemble a clean, comma-separated training caption from
    existing metadata. Always returns a non-empty string. Never raises."""
    cfg = config or CaptionTemplateConfig()
    f = _flat(sample)
    target = sample.get("lora_target") or f.get("lora_target") or ""
    parts: list = [_sanitize_token(cfg.identity_token),
                   _sanitize_token(cfg.class_token)]
    for d in _target_descriptors(target, f, cfg):
        parts.append(_sanitize_token(d))
    # fold in any extra concepts from an existing caption (deduped)
    existing = sample.get("caption") or f.get("caption")
    if isinstance(existing, str) and existing.strip():
        for seg in existing.split(","):
            s = _sanitize_token(seg)
            if s:
                parts.append(s)
    # dedup preserving order, drop empties
    seen, ordered = set(), []
    for p in parts:
        key = p.lower()
        if p and key not in seen:
            seen.add(key)
            ordered.append(p)
    caption = ", ".join(ordered)
    # word cap
    words = caption.split()
    if len(words) > cfg.max_caption_words:
        caption = " ".join(words[:cfg.max_caption_words]).rstrip(", ")
    if cfg.lowercase:
        caption = caption.lower()
    return caption or _sanitize_token(cfg.identity_token) or "person"


# ---------------------------------------------------------------------------
# dataset_config.toml (kohya array-of-tables; stdlib writer)
# ---------------------------------------------------------------------------
def _q(s) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_dataset_config_toml(image_dir: str, class_tokens: str,
                               resolution: int = 1024, batch_size: int = 1,
                               num_repeats: int = 1, keep_tokens: int = 1
                               ) -> str:
    """kohya sd-scripts dataset_config.toml (deterministic). The notebooks run
    from inside the dataset folder, so ``image_dir`` is the relative
    ``instance`` subfolder."""
    return (
        "[general]\n"
        "shuffle_caption = true\n"
        'caption_extension = ".txt"\n'
        f"keep_tokens = {int(keep_tokens)}\n\n"
        "[[datasets]]\n"
        f"resolution = {int(resolution)}\n"
        f"batch_size = {int(batch_size)}\n\n"
        "  [[datasets.subsets]]\n"
        f"  image_dir = {_q(image_dir)}\n"
        f"  class_tokens = {_q(class_tokens)}\n"
        f"  num_repeats = {int(num_repeats)}\n"
    )


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------
def _ordered_samples(samples: list, dataset_dir: Path) -> list:
    """Manifest order if present; else stable sort by source path."""
    def key(s):
        return str(s.get("image_target") or s.get("source_path")
                   or s.get("image") or "")
    if all(s.get("image_target") for s in samples):
        return list(samples)               # already in manifest order
    return sorted(samples, key=key)


def _resolve_source_image(sample: dict, dataset_dir: Path) -> Path | None:
    rel = sample.get("image_target") or sample.get("image")
    if rel:
        p = dataset_dir / rel
        if p.is_file():
            return p
    sp = sample.get("source_path")
    if sp and Path(sp).is_file():
        return Path(sp)
    # last resort: images/<stem>
    if rel:
        p = dataset_dir / "images" / Path(rel).name
        if p.is_file():
            return p
    return None


def _build_packaged_samples(samples, dataset_dir, cfg) -> list:
    out = []
    for i, s in enumerate(_ordered_samples(samples, dataset_dir),
                          start=cfg.start_index):
        warnings = []
        src_img = _resolve_source_image(s, dataset_dir)
        src_suffix = (src_img.suffix.lower() if src_img else
                      (cfg.image_ext if cfg.image_ext.startswith(".")
                       else f".{cfg.image_ext}"))
        stem = f"{i:0{cfg.digits}d}" if cfg.stable_filenames else \
            (Path(s.get("image_target") or f"img{i}").stem)
        pkg_img = f"instance/{stem}{src_suffix}"
        pkg_cap = f"instance/{stem}.txt"
        caption = build_caption(s, cfg.caption_template)
        if src_img is None:
            warnings.append("source_image_missing")
        out.append(PackagedSample(
            index=i, source_image=str(src_img) if src_img else "",
            source_caption=(s.get("caption") if isinstance(s.get("caption"),
                            str) else None),
            package_image=pkg_img, package_caption=pkg_cap,
            caption_text=caption,
            lora_target=s.get("lora_target"), warnings=warnings))
    return out


def _caption_stats(packaged: list) -> dict:
    caps = [p.caption_text for p in packaged]
    n = len(caps)
    if not n:
        return {"sample_count": 0, "empty_captions": 0, "avg_words": 0.0,
                "duplicate_captions": 0, "caption_extension": ".txt"}
    empty = sum(1 for c in caps if not c.strip())
    words = [len(c.split()) for c in caps]
    uniq = len(set(c.strip() for c in caps))
    return {
        "sample_count": n, "empty_captions": empty,
        "avg_words": round(sum(words) / n, 3),
        "duplicate_captions": n - uniq, "caption_extension": ".txt",
    }


def _concept_dir_name(target: str | None) -> str:
    t = (target or "concept").strip() or "concept"
    return f"concept_{t}"


def package_kohya_dataset(dataset_dir, output_dir,
                          config: KohyaPackageConfig | None = None
                          ) -> KohyaPackageReport:
    """Package a curated export into a Kohya/Colab training package. Default
    dry-run reports only; ``config.dry_run=False`` writes the package. Source
    dataset is never modified. Never raises for ordinary input."""
    cfg = config or KohyaPackageConfig()
    dataset_dir = Path(dataset_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    samples = load_curated_dataset(dataset_dir)
    report = KohyaPackageReport(
        source_dataset=str(dataset_dir), output_dir=str(output_dir),
        mode=cfg.mode, concept_name=cfg.concept_name,
        sample_count=len(samples), dry_run=cfg.dry_run)
    if not samples:
        report.warnings.append("no_samples_found")

    # LoRA multi-concept when samples span >1 target; else single instance.
    multi = cfg.mode == "lora" and len({s.get("lora_target")
                                        for s in samples
                                        if s.get("lora_target")}) > 1
    groups: dict = {}
    if multi:
        for s in samples:
            groups.setdefault(s.get("lora_target") or "concept", []).append(s)
    else:
        groups[None] = samples

    all_packaged: list = []
    concept_dirs: dict = {}
    for target, grp in groups.items():
        sub = _concept_dir_name(target) if multi else ""
        packaged = _build_packaged_samples(grp, dataset_dir, cfg)
        # prefix package paths with concept subfolder for multi-concept
        if sub:
            for p in packaged:
                p.package_image = f"{sub}/{p.package_image}"
                p.package_caption = f"{sub}/{p.package_caption}"
        concept_dirs[sub] = (target, packaged)
        all_packaged += packaged
    report.samples = all_packaged
    report.summary = {"caption_stats": _caption_stats(all_packaged),
                      "concepts": [k for k in concept_dirs if k] or
                      [cfg.concept_name]}

    if not cfg.dry_run:
        _write_package(report, dataset_dir, output_dir, cfg, concept_dirs, multi)
    return report


def _copy_image(src: Path, dest: Path, warnings: list):
    try:
        if src and src.is_file() and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    except Exception as exc:                             # never crash
        warnings.append(f"copy_failed: {src}: {exc}")


def _write_package(report, dataset_dir, output_dir, cfg, concept_dirs, multi):
    output_dir.mkdir(parents=True, exist_ok=True)
    class_tokens = cfg.identity_token
    config_paths = []
    for sub, (target, packaged) in concept_dirs.items():
        base = output_dir / sub if sub else output_dir
        inst = base / "instance"
        inst.mkdir(parents=True, exist_ok=True)
        for p in packaged:
            # package_image is relative to output_dir; strip concept prefix
            rel = p.package_image[len(sub) + 1:] if sub else p.package_image
            dest_img = base / rel
            dest_cap = base / (rel.rsplit(".", 1)[0] + ".txt")
            if cfg.copy_images and p.source_image:
                _copy_image(Path(p.source_image), dest_img, report.warnings)
            if cfg.overwrite_captions or not dest_cap.exists():
                try:
                    dest_cap.parent.mkdir(parents=True, exist_ok=True)
                    dest_cap.write_text(p.caption_text + "\n", encoding="utf-8")
                except Exception as exc:
                    report.warnings.append(f"caption_write_failed: {exc}")
        toml = render_dataset_config_toml(
            "instance", class_tokens, resolution=cfg.resolution,
            num_repeats=cfg.repeats)
        cfg_path = base / "dataset_config.toml"
        cfg_path.write_text(toml, encoding="utf-8")
        config_paths.append(str(cfg_path))
    report.dataset_config_path = config_paths[0] if config_paths else None

    # manifest + card at package root
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(dataset_dir), "mode": cfg.mode,
        "concept_name": cfg.concept_name, "identity_token": cfg.identity_token,
        "class_token": cfg.class_token, "sample_count": report.sample_count,
        "dataset_config": "dataset_config.toml", "instance_dir": "instance",
        "multi_concept": multi,
        "concepts": [s for s in concept_dirs if s] or [cfg.concept_name],
        "samples": [p.to_dict() for p in report.samples],
        "caption_stats": report.summary.get("caption_stats", {}),
        "source_dataset_card": str(dataset_dir / "dataset_card.json")
        if (dataset_dir / "dataset_card.json").is_file() else None,
        "source_dataset_manifest": str(dataset_dir / "dataset_manifest.json")
        if (dataset_dir / "dataset_manifest.json").is_file() else None,
        "config": cfg.to_dict(), "warnings": report.warnings,
    }
    man_path = output_dir / "package_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report.package_manifest_path = str(man_path)
    card_path = output_dir / "package_card.md"
    card_path.write_text(_render_card(report, cfg, multi), encoding="utf-8")
    report.package_card_path = str(card_path)

    if cfg.create_zip:
        report.zip_path = _make_zip(output_dir, cfg)


def _make_zip(output_dir: Path, cfg: KohyaPackageConfig) -> str | None:
    zip_name = f"{cfg.concept_name}_{cfg.mode}_kohya_package.zip"
    zip_path = output_dir / zip_name
    try:
        files = sorted(p for p in output_dir.rglob("*")
                       if p.is_file() and p.name != zip_name
                       and not p.name.startswith(".")
                       and "__MACOSX" not in p.parts)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in files:
                zi = zipfile.ZipInfo(str(p.relative_to(output_dir).as_posix()),
                                     date_time=_ZIP_DATE)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(zi, p.read_bytes())
        return str(zip_path)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return None


def _render_card(report: KohyaPackageReport, cfg: KohyaPackageConfig,
                 multi: bool) -> str:
    cs = report.summary.get("caption_stats", {})
    L = [f"# Kohya / Colab Package: {cfg.concept_name}", "",
         "## Summary",
         f"- Source dataset: {report.source_dataset}",
         f"- Mode: {report.mode}",
         f"- Concept: {cfg.concept_name}",
         f"- Identity token: {cfg.identity_token}",
         f"- Class token: {cfg.class_token}",
         f"- Sample count: {report.sample_count}",
         f"- Dry run: {report.dry_run}",
         f"- Zip: {report.zip_path or '(not created)'}", "",
         "## Folder Layout"]
    if multi:
        L += ["```", "<package>/", "  concept_<target>/",
              "    dataset_config.toml", "    instance/000001.jpg + .txt",
              "  package_manifest.json", "  package_card.md", "```", ""]
    else:
        L += ["```", "<package>/", "  dataset_config.toml",
              "  instance/000001.jpg + 000001.txt ...",
              "  package_manifest.json", "  package_card.md", "```", ""]
    L += ["## Caption Summary",
          f"- Empty captions: {cs.get('empty_captions')}",
          f"- Average words: {cs.get('avg_words')}",
          f"- Duplicate captions: {cs.get('duplicate_captions')}",
          f"- Caption extension: {cs.get('caption_extension')}", "",
          "## Files",
          "| # | Image | Caption Preview | Source |",
          "|---:|---|---|---|"]
    for p in report.samples[:50]:
        prev = (p.caption_text[:60] + "...") if len(p.caption_text) > 60 \
            else p.caption_text
        L.append(f"| {p.index} | {p.package_image} | {prev} | "
                 f"{Path(p.source_image).name if p.source_image else ''} |")
    if len(report.samples) > 50:
        L.append("")
        L.append(f"_Showing 50 of {len(report.samples)}; see "
                 "package_manifest.json for the full list._")
    L += ["", "## Colab Instructions",
          "1. Upload the package zip to Colab (or Drive).",
          "2. Extract the package.",
          "3. Point the notebook `dataset_config` / `train_data_dir` at this "
          "folder (run from inside it so `image_dir = \"instance\"` resolves).",
          "4. Start training.", ""]
    return "\n".join(L)


__all__ = [
    "CaptionTemplateConfig", "KohyaPackageConfig", "PackagedSample",
    "KohyaPackageReport", "load_curated_dataset", "build_caption",
    "render_dataset_config_toml", "package_kohya_dataset",
]
