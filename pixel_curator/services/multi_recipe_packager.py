"""Multi-Recipe Training Package Generator.

Converts Specialized Recipe routing metadata into ready-to-train package folders:
one main DreamBooth/base package plus multiple specialized LoRA packages, all
from the same curated dataset. Metadata-only planning + non-destructive

materialization (hardlink by default; copy/symlink optional) - it never loads
models, never launches training, and never modifies source images.

Professional, SFW package names only. Age-uncertain and severe-review records are
excluded from packages by default.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PACKAGER_VERSION = "multi_recipe_packager_v1"

DREAMBOOTH_PACKAGE = "dreambooth_main"
SPECIALIZED_PACKAGES = (
    "character_core", "portrait_reference", "upper_body_reference",
    "lower_body_reference", "full_body_reference", "full_body_pose_reference",
    "dynamic_pose_reference", "figure_composition_reference",
    "outfit_fabric_reference", "texture_detail_reference",
    "lighting_composition_reference", "captioning_reference")
# routed-but-not-trainable buckets (never materialized as training packages)
NON_TRAINING_RECIPES = ("review_required", "review_set", "reject_low_quality",
                        "captioning_reference")

RECIPE_TO_TYPE = {
    "character_core": "character_core", "portrait_reference": "portrait",
    "upper_body_reference": "upper_body", "lower_body_reference": "lower_body",
    "full_body_reference": "full_body", "full_body_pose_reference": "pose_dynamic",
    "dynamic_pose_reference": "pose_dynamic",
    "figure_composition_reference": "figure_composition",
    "outfit_fabric_reference": "outfit",
    "texture_detail_reference": "detail_texture",
    "lighting_composition_reference": "lighting_composition",
    "captioning_reference": "captioning"}

# DreamBooth prefers these recipes for identity/reference strength
_DREAMBOOTH_PREFERRED = ("character_core", "portrait_reference",
                         "full_body_reference", "upper_body_reference")

_SEVERE_FLAGS = ("corrupt_or_missing_input", "no_clear_subject", "corrupt_image",
                 "unsupported_format")
_BLOCKING_FLAGS = ("review_required_age_uncertain",) + _SEVERE_FLAGS
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")

OUTPUT_FILES = {
    "plan": "multi_recipe_package_plan.json",
    "summary": "multi_recipe_package_summary.json",
    "card": "multi_recipe_package_card.md"}


# ============================ dataclasses ===================================
@dataclass
class MultiRecipePackagerConfig:
    packager_version: str = PACKAGER_VERSION
    link_mode: str = "hardlink"          # hardlink | copy | symlink
    dry_run: bool = True
    overwrite: bool = False
    include_dreambooth_main: bool = True
    include_lora_packages: bool = True
    max_images_per_package: int | None = None
    min_images_per_package: int = 10
    dreambooth_min_suitability: float = 75.0
    lora_min_suitability: float = 55.0
    specialized_min_confidence: float = 0.55
    exclude_review_flags: bool = True
    exclude_age_uncertain: bool = True
    allow_review_records: bool = False
    caption_priority: list = field(default_factory=lambda: [
        "caption_wd14", "caption_natural", "caption", "txt_sidecar"])
    identity_token: str = "spikbnui person"
    class_token: str = "person"
    architecture_default: str = "sdxl"


@dataclass
class PackageImageRecord:
    image_path: str
    caption: str = ""
    source_record: dict = field(default_factory=dict)
    suitability_score: float = 0.0
    recipe_confidence: float = 0.0
    primary_recipe: str = ""
    assigned_packages: list = field(default_factory=list)
    routing_tags: list = field(default_factory=list)
    review_flags: list = field(default_factory=list)
    architecture: str = "sdxl"
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"image_path": self.image_path, "caption": self.caption,
                "suitability_score": round(self.suitability_score, 2),
                "recipe_confidence": round(self.recipe_confidence, 4),
                "primary_recipe": self.primary_recipe,
                "assigned_packages": list(self.assigned_packages),
                "routing_tags": list(self.routing_tags),
                "review_flags": list(self.review_flags),
                "architecture": self.architecture,
                "warnings": list(self.warnings)}


@dataclass
class TrainingPackageSpec:
    package_name: str
    package_type: str                    # dreambooth | lora
    recipe_type: str
    training_target: str
    architecture: str = "sdxl"
    output_dir: str = ""
    image_records: list = field(default_factory=list)
    min_suitability: float = 0.0
    min_confidence: float = 0.0
    caption_style: str = "auto"
    identity_token: str = ""
    class_token: str = ""
    warnings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"package_name": self.package_name,
                "package_type": self.package_type,
                "recipe_type": self.recipe_type,
                "training_target": self.training_target,
                "architecture": self.architecture, "output_dir": self.output_dir,
                "image_count": len(self.image_records),
                "min_suitability": self.min_suitability,
                "min_confidence": self.min_confidence,
                "caption_style": self.caption_style,
                "identity_token": self.identity_token,
                "class_token": self.class_token,
                "image_records": [r.to_dict() for r in self.image_records],
                "warnings": list(self.warnings), "metadata": self.metadata}


@dataclass
class MultiRecipePackagePlan:
    plan_id: str
    packager_version: str
    output_dir: str
    packages: list = field(default_factory=list)
    skipped_records: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict:
        return {"plan_id": self.plan_id,
                "packager_version": self.packager_version,
                "output_dir": self.output_dir,
                "packages": [p.to_dict() for p in self.packages],
                "skipped_records": list(self.skipped_records),
                "summary": self.summary, "warnings": list(self.warnings),
                "created_at": self.created_at}


# ============================ helpers =======================================
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _resolve_caption(rec: dict, cfg: MultiRecipePackagerConfig) -> str:
    """Caption priority: wd14 -> natural -> caption -> .txt sidecar -> fallback."""
    fl = rec.get("captions") or {}
    sources = {
        "caption_wd14": rec.get("caption_wd14") or rec.get("wd14_caption"),
        "caption_natural": rec.get("caption_natural") or fl.get("caption"),
        "caption": rec.get("caption"),
    }
    for key in cfg.caption_priority:
        if key == "txt_sidecar":
            img = rec.get("image_path") or rec.get("path") or ""
            side = Path(img).with_suffix(".txt") if img else None
            if side and side.is_file():
                try:
                    txt = side.read_text(encoding="utf-8").strip()
                    if txt:
                        return txt
                except Exception:
                    logger.debug("suppressed exception", exc_info=True)
                    pass
        else:
            v = sources.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    # safe fallback (uses configured tokens; never explicit)
    tok = (cfg.identity_token or "").strip()
    return (f"{tok}, professional portrait reference, clean composition"
            if tok else "professional portrait reference, clean composition")


def normalize_package_records(records: list, config=None) -> list:
    """Normalize assignment / explorer / visual-focus dicts into package records.
    Never raises for ordinary input."""
    cfg = config or MultiRecipePackagerConfig()
    out = []
    for r in records or []:
        rr = r or {}
        recipes = []
        conf = _num(rr.get("recipe_confidence")) or _num(rr.get("confidence")) or 0.0
        for c in (rr.get("recommended_recipes") or []):
            if isinstance(c, dict):
                if not c.get("blocked"):
                    recipes.append(c.get("name"))
            elif isinstance(c, str):
                recipes.append(c)
        primary = rr.get("primary_recipe") or (recipes[0] if recipes else "")
        arch = rr.get("architecture") or rr.get("architecture_target") \
            or cfg.architecture_default
        out.append(PackageImageRecord(
            image_path=rr.get("image_path") or rr.get("path") or rr.get("filename") or "",
            caption=_resolve_caption(rr, cfg),
            source_record=rr,
            suitability_score=_num(rr.get("suitability_score")) or 0.0,
            recipe_confidence=conf, primary_recipe=primary or "",
            assigned_packages=[x for x in recipes if x],
            routing_tags=rr.get("routing_tags") or [],
            review_flags=[str(f) for f in (rr.get("review_flags") or [])],
            architecture=arch if arch in ("sdxl", "unknown")
            else cfg.architecture_default))
    return out


def _excluded(rec: PackageImageRecord, cfg) -> str | None:
    if cfg.exclude_age_uncertain and "review_required_age_uncertain" in rec.review_flags:
        return "age_uncertain"
    if cfg.exclude_review_flags and not cfg.allow_review_records:
        sev = [f for f in rec.review_flags if f in _SEVERE_FLAGS]
        if sev:
            return f"severe_flag:{sev[0]}"
    if rec.primary_recipe == "reject_low_quality":
        return "reject_low_quality"
    if not rec.image_path:
        return "missing_image_path"
    return None


# ============================ selection / grouping ==========================
def select_dreambooth_main_records(records: list, config) -> list:
    cfg = config
    elig = []
    for r in records:
        if _excluded(r, cfg):
            continue
        if r.suitability_score < cfg.dreambooth_min_suitability:
            continue
        elig.append(r)

    def key(r):
        pref = any(p in r.assigned_packages or r.primary_recipe == p
                   for p in _DREAMBOOTH_PREFERRED)
        return (1 if pref else 0, r.suitability_score)
    elig.sort(key=key, reverse=True)
    if cfg.max_images_per_package:
        elig = elig[:cfg.max_images_per_package]
    return elig


def group_records_by_specialized_recipe(records: list, config) -> dict:
    cfg = config
    groups: dict = {name: [] for name in SPECIALIZED_PACKAGES}
    for r in records:
        if _excluded(r, cfg):
            continue
        if r.suitability_score < cfg.lora_min_suitability:
            continue
        if r.recipe_confidence and r.recipe_confidence < cfg.specialized_min_confidence:
            pass  # confidence is per-image-primary; tag/route members may lack it
        for name in r.assigned_packages:
            if name in NON_TRAINING_RECIPES or name not in groups:
                continue
            groups[name].append(r)
    # cap + drop empties
    out = {}
    for name, recs in groups.items():
        recs.sort(key=lambda x: x.suitability_score, reverse=True)
        if cfg.max_images_per_package:
            recs = recs[:cfg.max_images_per_package]
        if recs:
            out[name] = recs
    return out


def _dominant_arch(recs: list, cfg) -> tuple:
    archs = {}
    for r in recs:
        archs[r.architecture] = archs.get(r.architecture, 0) + 1
    if not archs:
        return cfg.architecture_default, []
    dom = max(archs, key=archs.get)
    warnings = []
    if len([a for a in archs if a != "unknown"]) > 1:
        warnings.append(f"mixed architecture in package ({archs}); using '{dom}'")
    return dom, warnings


# ============================ plan builder ==================================
def build_multi_recipe_package_plan(records, output_dir, config=None
                                    ) -> MultiRecipePackagePlan:
    cfg = config or MultiRecipePackagerConfig()
    out = Path(output_dir)
    plan = MultiRecipePackagePlan(
        plan_id=f"pkgplan_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
        packager_version=cfg.packager_version, output_dir=str(out),
        created_at=_now())
    recs = normalize_package_records(records, cfg)

    # track skips
    for r in recs:
        why = _excluded(r, cfg)
        if why:
            plan.skipped_records.append({"image_path": r.image_path,
                                         "reason": why,
                                         "review_flags": r.review_flags})

    if cfg.include_dreambooth_main:
        db = select_dreambooth_main_records(recs, cfg)
        arch, w = _dominant_arch(db, cfg)
        spec = TrainingPackageSpec(
            package_name=DREAMBOOTH_PACKAGE, package_type="dreambooth",
            recipe_type="character_core", training_target="core_character_likeness",
            architecture=arch, output_dir=str(out / DREAMBOOTH_PACKAGE),
            image_records=db, min_suitability=cfg.dreambooth_min_suitability,
            identity_token=cfg.identity_token, class_token=cfg.class_token,
            warnings=w)
        if len(db) < cfg.min_images_per_package:
            spec.warnings.append(
                f"only {len(db)} images (< min {cfg.min_images_per_package})")
        plan.packages.append(spec)

    if cfg.include_lora_packages:
        groups = group_records_by_specialized_recipe(recs, cfg)
        for name, grp in groups.items():
            arch, w = _dominant_arch(grp, cfg)
            spec = TrainingPackageSpec(
                package_name=name, package_type="lora",
                recipe_type=RECIPE_TO_TYPE.get(name, "review"),
                training_target=name, architecture=arch,
                output_dir=str(out / name), image_records=grp,
                min_suitability=cfg.lora_min_suitability,
                min_confidence=cfg.specialized_min_confidence,
                identity_token=cfg.identity_token, class_token=cfg.class_token,
                warnings=w)
            if len(grp) < cfg.min_images_per_package:
                spec.warnings.append(
                    f"only {len(grp)} images (< min {cfg.min_images_per_package})")
            plan.packages.append(spec)

    plan.summary = summarize_package_plan(plan)
    return plan


def summarize_package_plan(plan: MultiRecipePackagePlan) -> dict:
    return {
        "plan_id": plan.plan_id,
        "total_packages": len(plan.packages),
        "dreambooth_packages": sum(1 for p in plan.packages
                                   if p.package_type == "dreambooth"),
        "lora_packages": sum(1 for p in plan.packages if p.package_type == "lora"),
        "image_counts": {p.package_name: len(p.image_records)
                         for p in plan.packages},
        "architecture_by_package": {p.package_name: p.architecture
                                    for p in plan.packages},
        "skipped": len(plan.skipped_records),
        "skip_reasons": _count([s["reason"] for s in plan.skipped_records]),
        "warnings": list(plan.warnings)
        + [w for p in plan.packages for w in p.warnings],
    }


def _count(items) -> dict:
    d = {}
    for x in items:
        d[x] = d.get(x, 0) + 1
    return dict(sorted(d.items(), key=lambda kv: -kv[1]))


# ============================ materialization ===============================
def _link_one(src: Path, dest: Path, mode: str) -> tuple:
    """Return (ok, warning). hardlink falls back to copy across volumes."""
    try:
        if mode == "copy":
            shutil.copy2(src, dest)
        elif mode == "symlink":
            os.symlink(src, dest)
        else:  # hardlink
            try:
                os.link(src, dest)
            except OSError:
                shutil.copy2(src, dest)
                return True, f"hardlink failed for {src.name}; copied instead"
        return True, ""
    except Exception as exc:
        return False, f"{src.name}: {exc}"


def _subdir_for(spec: TrainingPackageSpec) -> str:
    return "instance" if spec.package_type == "dreambooth" else "images"


def materialize_multi_recipe_packages(plan, config=None) -> dict:
    """Create package folders + linked images + captions + manifests/cards/
    suggestions. Honors dry_run (plan only). Never mutates source images."""
    cfg = config or MultiRecipePackagerConfig()
    res = {"dry_run": cfg.dry_run, "materialized": {}, "warnings": []}
    if cfg.dry_run:
        res["note"] = "dry-run: no folders/links created. Re-run with apply."
        for p in plan.packages:
            res["materialized"][p.package_name] = {
                "planned_images": len(p.image_records), "applied": False}
        return res
    for spec in plan.packages:
        pkg_dir = Path(spec.output_dir)
        sub = pkg_dir / _subdir_for(spec)
        info = {"dir": str(pkg_dir), "linked": 0, "captions": 0, "applied": True,
                "warnings": list(spec.warnings)}
        try:
            sub.mkdir(parents=True, exist_ok=True)
            images_meta = []
            for i, rec in enumerate(spec.image_records, 1):
                src = Path(rec.image_path)
                ext = src.suffix.lower() if src.suffix.lower() in IMAGE_EXTS else ".jpg"
                stem = f"image_{i:06d}"
                dest = sub / f"{stem}{ext}"
                cap_dest = sub / f"{stem}.txt"
                if not src.is_file():
                    info["warnings"].append(f"missing source: {src.name}")
                    continue
                if dest.exists() and not cfg.overwrite:
                    pass
                else:
                    if dest.exists():
                        try:
                            dest.unlink()
                        except Exception:
                            logger.debug("suppressed exception", exc_info=True)
                            pass
                    ok, w = _link_one(src, dest, cfg.link_mode)
                    if w:
                        info["warnings"].append(w)
                    if not ok:
                        continue
                    info["linked"] += 1
                # package-local caption (never touches source)
                cap_dest.write_text((rec.caption or "") + "\n", encoding="utf-8")
                info["captions"] += 1
                images_meta.append({
                    "package_image": f"{_subdir_for(spec)}/{stem}{ext}",
                    "caption_file": f"{_subdir_for(spec)}/{stem}.txt",
                    "source_image": str(src),
                    "caption": rec.caption or "",
                    "suitability_score": round(rec.suitability_score, 2),
                    "recipe_confidence": round(rec.recipe_confidence, 4),
                    "routing_tags": list(rec.routing_tags),
                    "review_flags": list(rec.review_flags)})
            _write_package_files(spec, pkg_dir, images_meta, plan, cfg)
            info["images"] = images_meta
        except Exception as exc:
            info["warnings"].append(f"package error: {exc}")
        res["materialized"][spec.package_name] = info
    return res


def register_materialized_packages(plan, materialized, manifest_base, *,
                                   parent_version=None, note="") -> list:
    """Register every applied package as a dataset version (Phase 2).

    Each materialized package (DreamBooth main + specialized LoRA packages)
    becomes a ``multi_recipe_package`` version with path-based file
    membership, a ``version.json`` sidecar and optional parent lineage -
    so `pixel-curator datasets`, `history` and `diff` see these exports.
    Never raises: registration failures degrade to per-package warnings."""
    from ..dataset_versioning import KIND_MULTI_RECIPE_PACKAGE, register_folder_export
    records: list = []
    mat = (materialized or {}).get("materialized") or {}
    for spec in plan.packages:
        info = mat.get(spec.package_name) or {}
        if not info.get("applied"):
            continue
        files = [{"rel_path": m.get("package_image"),
                  "source_path": m.get("source_image"),
                  "caption": m.get("caption")}
                 for m in info.get("images") or []
                 if m.get("package_image")]
        rec = register_folder_export(
            manifest_base, kind=KIND_MULTI_RECIPE_PACKAGE,
            name=spec.package_name, export_dir=spec.output_dir,
            files=files, image_count=len(files),
            params={"package_type": spec.package_type,
                    "recipe_type": spec.recipe_type,
                    "training_target": spec.training_target,
                    "architecture": spec.architecture,
                    "plan_id": plan.plan_id,
                    "identity_token": spec.identity_token},
            note=note, parent_version=parent_version)
        rec["package_name"] = spec.package_name
        records.append(rec)
    return records


def _write_package_files(spec, pkg_dir, images_meta, plan, cfg):
    pkg_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "package_name": spec.package_name, "package_type": spec.package_type,
        "recipe_type": spec.recipe_type, "training_target": spec.training_target,
        "architecture": spec.architecture, "created_at": _now(),
        "image_count": len(images_meta),
        "caption_count": sum(1 for m in images_meta if m.get("caption_file")),
        "source_plan": plan.plan_id, "link_mode": cfg.link_mode,
        "identity_token": spec.identity_token, "class_token": spec.class_token,
        "selection_criteria": {"min_suitability": spec.min_suitability,
                               "min_confidence": spec.min_confidence},
        "images": images_meta, "warnings": list(spec.warnings)}
    (pkg_dir / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    card = {"package_name": spec.package_name, "package_type": spec.package_type,
            "training_target": spec.training_target,
            "architecture": spec.architecture, "image_count": len(images_meta),
            "identity_token": spec.identity_token, "class_token": spec.class_token,
            "created_at": _now()}
    (pkg_dir / "dataset_card.json").write_text(json.dumps(card, indent=2),
                                               encoding="utf-8")
    (pkg_dir / "dataset_card.md").write_text(
        f"# {spec.package_name}\n\n- type: {spec.package_type}\n"
        f"- target: {spec.training_target}\n- architecture: {spec.architecture}\n"
        f"- images: {len(images_meta)}\n- identity token: "
        f"`{spec.identity_token}`\n", encoding="utf-8")
    (pkg_dir / "training_plan_suggestion.json").write_text(
        json.dumps(_training_suggestion(spec), indent=2), encoding="utf-8")
    (pkg_dir / "README.md").write_text(
        f"# {spec.package_name}\n\nGenerated by {PACKAGER_VERSION}. "
        f"{len(images_meta)} images, captions in `{_subdir_for(spec)}/*.txt`.\n\n"
        "Source images were hardlinked/copied/symlinked — never modified. "
        "No training was launched.\n", encoding="utf-8")


def _training_suggestion(spec: TrainingPackageSpec) -> dict:
    base = {"training_type": spec.package_type, "package_name": spec.package_name,
            "architecture": spec.architecture, "recommended_resolution": 1024,
            "recommended_caption_source": "package_txt"}
    if spec.package_type == "dreambooth":
        base["notes"] = ["Use as the high-quality base identity/reference package."]
    else:
        base.update(network_rank=16, network_alpha=16,
                    notes=["Focused specialized LoRA package generated from "
                           "routing metadata."])
    return base


# ============================ persistence ===================================
def write_multi_recipe_package_plan(plan, output_dir, overwrite=False) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    files = {
        OUTPUT_FILES["plan"]: json.dumps(plan.to_dict(), indent=2),
        OUTPUT_FILES["summary"]: json.dumps(plan.summary, indent=2),
        OUTPUT_FILES["card"]: render_package_card(plan)}
    for name, content in files.items():
        dest = out / name
        if dest.exists() and not overwrite:
            written[name] = dest
            continue
        try:
            dest.write_text(content, encoding="utf-8")
            written[name] = dest
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass
    return written


def load_multi_recipe_package_plan(path) -> MultiRecipePackagePlan:
    p = Path(path)
    if p.is_dir():
        p = p / OUTPUT_FILES["plan"]
    data = {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    return MultiRecipePackagePlan(
        plan_id=data.get("plan_id", ""),
        packager_version=data.get("packager_version", PACKAGER_VERSION),
        output_dir=data.get("output_dir", ""),
        skipped_records=data.get("skipped_records", []),
        summary=data.get("summary", {}), warnings=data.get("warnings", []),
        created_at=data.get("created_at", ""))


def render_package_card(plan: MultiRecipePackagePlan) -> str:
    s = plan.summary or summarize_package_plan(plan)
    lines = ["# Multi-Recipe Training Packages", "",
             f"Plan: `{plan.plan_id}`", "",
             f"- packages: **{s.get('total_packages', 0)}** "
             f"(dreambooth {s.get('dreambooth_packages', 0)}, "
             f"lora {s.get('lora_packages', 0)})",
             f"- skipped records: {s.get('skipped', 0)}", "",
             "## Images per package"]
    for name, n in s.get("image_counts", {}).items():
        arch = s.get("architecture_by_package", {}).get(name, "")
        lines.append(f"- {name} ({arch}): {n}")
    if s.get("skip_reasons"):
        lines += ["", "## Skip reasons"]
        for k, v in s["skip_reasons"].items():
            lines.append(f"- {k}: {v}")
    for w in s.get("warnings", []):
        lines.append(f"\n> ⚠ {w}")
    lines += ["", "_Planning/packaging only — source images never modified, no "
              "training launched._"]
    return "\n".join(lines)


def result_to_dataset_explorer_fields(plan: MultiRecipePackagePlan) -> dict:
    """basename -> assigned packages (for Dataset Explorer merge)."""
    out = {}
    for spec in plan.packages:
        for rec in spec.image_records:
            name = Path(rec.image_path).name
            if not name:
                continue
            entry = out.setdefault(name, {
                "assigned_training_packages": [], "primary_training_package": "",
                "package_types": [], "multi_recipe_package_plan_id": plan.plan_id})
            entry["assigned_training_packages"].append(spec.package_name)
            entry["package_types"].append(spec.package_type)
            if spec.package_type == "dreambooth" or not entry["primary_training_package"]:
                entry["primary_training_package"] = spec.package_name
    return out


def write_ui_status_summary(plan: MultiRecipePackagePlan, output_base) -> Path:
    base = Path(output_base).expanduser()
    d = base / "ui_status"
    d.mkdir(parents=True, exist_ok=True)
    dest = d / "multi_recipe_package_summary.json"
    try:
        dest.write_text(json.dumps(plan.summary, indent=2, default=str),
                        encoding="utf-8")
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    return dest


__all__ = [
    "PACKAGER_VERSION", "DREAMBOOTH_PACKAGE", "SPECIALIZED_PACKAGES",
    "OUTPUT_FILES", "MultiRecipePackagerConfig", "PackageImageRecord",
    "TrainingPackageSpec", "MultiRecipePackagePlan",
    "build_multi_recipe_package_plan", "materialize_multi_recipe_packages",
    "write_multi_recipe_package_plan", "load_multi_recipe_package_plan",
    "normalize_package_records", "select_dreambooth_main_records",
    "group_records_by_specialized_recipe", "summarize_package_plan",
    "render_package_card", "result_to_dataset_explorer_fields",
    "write_ui_status_summary"]
