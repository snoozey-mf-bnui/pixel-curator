"""Signal-based non-character LoRA categorizer (outfit / pose / detail).

Reads the framing label + advanced-quality signals stored per frame and assigns
each curated frame its best-fit *non-character* LoRA category, then ranks within
each category by image quality. Character/identity frames (portraits, face
close-ups) are routed to ``identity_skip`` because identity is handled by the
DreamBooth, not a separate LoRA. Style is intentionally omitted - a personal
self-dataset has no trainable artistic "style".

Read-only and non-destructive: it only reads the manifest + writes plans /
symlinks on explicit export. Never raises for ordinary input.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

NON_CHARACTER = ("outfit", "pose", "detail")
CATEGORIES = NON_CHARACTER + ("identity_skip", "needs_review")

# coverage_intensity training-use hints -> our category keys
_HINT_MAP = {"outfit": "outfit", "pose": "pose", "detail": "detail",
             "identity": "identity_skip", "catalog_reference": "outfit",
             "review": "needs_review"}


def _num(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _parse(signals_json) -> dict:
    try:
        return json.loads(signals_json) if signals_json else {}
    except Exception:
        return {}


def _framing_label(framing: str, s: dict) -> str:
    """Prefer the coverage-intensity framing label; fall back to the detection
    framing bucket."""
    fl = s.get("framing_label")
    if not fl:
        ci = s.get("coverage_intensity")
        if isinstance(ci, dict):
            fl = ci.get("framing_label")
    if fl and fl != "unknown":
        return str(fl).lower()
    return str(framing or "").lower()


def _quality_score(s: dict) -> float:
    """Identity-independent image-quality score for ranking within a category."""
    score, n = 0.0, 0
    for k, w in (("region_balance_score", 0.30), ("face_region_score", 0.10),
                 ("torso_region_score", 0.15),
                 ("subject_integrity_score", 0.20),
                 ("aesthetic_score", 0.10),
                 ("texture_preservation_score", 0.15)):
        v = _num(s.get(k))
        if v is not None:
            score += w * v
            n += w
    base = (score / n) if n else 0.5
    blur = _num(s.get("motion_blur_score"))
    if blur is not None:
        base -= 0.15 * blur
    occ = _num(s.get("face_occlusion_score"))
    if occ is not None:
        base -= 0.10 * occ
    pd = s.get("pose_duplicate")
    if isinstance(pd, dict) and pd.get("near_duplicate") and \
            not pd.get("representative", True):
        base -= 0.15
    return round(max(0.0, min(1.0, base)), 4)


@dataclass
class CategoryResult:
    frame_id: str
    path: str
    category: str
    fit: float
    quality: float
    rank_score: float
    framing: str
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"frame_id": self.frame_id, "path": self.path,
                "category": self.category, "fit": round(self.fit, 4),
                "quality": round(self.quality, 4),
                "rank_score": round(self.rank_score, 4),
                "framing": self.framing, "reasons": list(self.reasons)}


def categorize_frame(framing: str, signals: dict) -> CategoryResult:
    """Assign one frame its best non-character category from framing + signals."""
    s = signals or {}
    fl = _framing_label(framing, s)
    hint = s.get("training_use_hint")
    clothing = s.get("clothing_type_label")
    clothing_visible = bool(clothing and clothing != "unknown")
    cov_ratio = _num(s.get("clothing_type_score"))   # garment coverage ratio
    full_vis = _num(s.get("full_body_region_score")) or _num(
        s.get("full_body_score"))
    torso = _num(s.get("torso_region_score"))
    tex = _num(s.get("texture_preservation_score"))
    dyn = _num(s.get("dynamic_pose_score"))
    reasons = []

    # --- per-category fit scores ---
    # detail: tight crops with strong texture / garment close-ups
    detail = 0.0
    if fl in ("close_crop", "detail_crop", "closeup", "face"):
        detail += 0.6
    if tex is not None:
        detail += 0.4 * tex
    if clothing_visible and fl in ("close_crop", "detail_crop", "closeup"):
        detail += 0.2

    # pose: full-body, good body visibility, dynamic posture
    pose = 0.0
    if fl in ("full_body", "full body"):
        pose += 0.6
    if full_vis is not None:
        pose += 0.4 * full_vis
    if dyn is not None:
        pose += 0.2 * dyn

    # outfit: torso / upper / full body with a visible garment
    outfit = 0.0
    if fl in ("torso_focus", "upper_body", "full_body", "half_body"):
        outfit += 0.5
    if clothing_visible:
        outfit += 0.3
        reasons.append(f"clothing:{clothing}")
    if torso is not None:
        outfit += 0.2 * torso
    if cov_ratio is not None and 0.3 <= cov_ratio <= 0.97:
        outfit += 0.15

    cats = {"outfit": outfit, "pose": pose, "detail": detail}

    # explicit coverage-intensity hint nudges its category
    mapped = _HINT_MAP.get(str(hint)) if hint else None
    if mapped in cats:
        cats[mapped] += 0.35
        reasons.append(f"hint:{hint}")
    elif mapped == "identity_skip":
        return CategoryResult("", "", "identity_skip", 1.0, _quality_score(s),
                              0.0, fl, ["hint:identity"])

    # face-centric framings with no garment/pose signal -> identity (skip)
    if fl in ("face_closeup", "portrait", "headshot", "face") \
            and max(cats.values()) < 0.5:
        return CategoryResult("", "", "identity_skip", 1.0, _quality_score(s),
                              0.0, fl, ["face-centric framing"])

    best = max(cats, key=cats.get)
    fit = round(cats[best], 4)
    quality = _quality_score(s)
    if fit < 0.5:
        cat = "needs_review"
        reasons.append("low category fit")
    else:
        cat = best
        reasons.append(f"{best} fit={fit}")
    rank = round(0.4 * fit + 0.6 * quality, 4)
    return CategoryResult("", "", cat, fit, quality, rank, fl, reasons)


def categorize_pass(conn, *, statuses=("selected", "packaged")) -> dict:
    """Categorize every curated frame. Returns counts + per-category ranked
    lists. Reads only; never raises for ordinary input."""
    has_adv = True
    try:
        conn.execute("SELECT 1 FROM frame_advanced_quality LIMIT 1").fetchone()
    except Exception:
        has_adv = False
    status_in = ",".join("?" for _ in statuses)
    if has_adv:
        sql = ("SELECT f.frame_id, f.path, d.framing, a.signals_json "
               "FROM frames f JOIN detections d ON d.frame_id=f.frame_id "
               "LEFT JOIN frame_advanced_quality a ON a.frame_id=f.frame_id "
               f"WHERE f.status IN ({status_in})")
    else:
        sql = ("SELECT f.frame_id, f.path, d.framing, NULL "
               "FROM frames f JOIN detections d ON d.frame_id=f.frame_id "
               f"WHERE f.status IN ({status_in})")
    cats: dict = {c: [] for c in CATEGORIES}
    warnings = []
    if not has_adv:
        warnings.append("frame_advanced_quality not populated - categorizing by "
                        "framing only; run the advanced 'quality' pass for "
                        "signal-backed routing")
    for fid, path, framing, sj in conn.execute(sql, tuple(statuses)):
        res = categorize_frame(framing, _parse(sj))
        res.frame_id, res.path = fid, path
        cats[res.category].append(res)
    for c in cats:
        cats[c].sort(key=lambda r: r.rank_score, reverse=True)
    counts = {c: len(v) for c, v in cats.items()}
    return {"counts": counts, "categories": cats, "has_advanced": has_adv,
            "warnings": warnings,
            "total": sum(len(v) for v in cats.values())}


def render_categorize(report: dict, top: int = 5) -> str:
    c = report["counts"]
    lines = ["", "Non-character LoRA categorization", "=" * 56,
             f"total curated frames: {report['total']}",
             f"signal-backed: {report['has_advanced']}", ""]
    for w in report.get("warnings", []):
        lines.append(f"  ! {w}")
    lines.append("")
    lines.append(f"{'category':<16}{'count':>10}")
    lines.append("-" * 26)
    for cat in ("outfit", "pose", "detail", "needs_review", "identity_skip"):
        lines.append(f"{cat:<16}{c.get(cat, 0):>10}")
    lines.append("")
    for cat in NON_CHARACTER:
        items = report["categories"][cat][:top]
        if items:
            lines.append(f"top {cat}:")
            for r in items:
                lines.append(f"  {r.rank_score:.3f}  {Path(r.path).name}  "
                             f"[{r.framing}]  {', '.join(r.reasons)}")
            lines.append("")
    return "\n".join(lines)


# per-category descriptor used to seed caption stubs (enriched later by the VLM)
_CATEGORY_DESCRIPTOR = {
    "outfit": "full outfit, clothing focus",
    "pose": "full body, dynamic pose",
    "detail": "close-up, garment detail, fabric texture",
}
PONY_PREFIX = "score_9, score_8_up, score_7_up"


def _caption_stub(category: str, architecture: str, token: str,
                  class_word: str) -> str:
    """Caption scaffold the VLM notebook later enriches: Pony-XL score
    prefix + optional identity/class + category descriptor."""
    desc = _CATEGORY_DESCRIPTOR.get(category, category)
    tok = (token or "").strip()
    cls = (class_word or "").strip()
    lead = [PONY_PREFIX]
    if tok:
        lead.append(f"{tok} {cls}".strip())
    elif cls:
        lead.append(cls)
    return ", ".join(lead + [desc])


def export_categories(report: dict, output_dir, *, count: int = 0,
                      link_mode: str = "symlink", dry_run: bool = True,
                      architecture: str = None, token: str = "",
                      class_word: str = "person") -> dict:
    """Write the top frames of each non-character category into
    ``<output_dir>/<category>/`` (+ a per-category manifest). When
    ``architecture`` is "pony", also writes a per-image ``.txt``
    score-prefixed caption stub so each set is drop-in ready for the
    trainer. Non-destructive: dry-run plans only; apply symlinks/copies
    (never moves the source). Never raises for ordinary input."""
    import os
    import shutil
    out = Path(output_dir).expanduser()
    arch = (architecture or "").lower() or None
    plan = {"output_dir": str(out), "dry_run": dry_run, "link_mode": link_mode,
            "architecture": arch, "exported": {}, "warnings": []}
    for cat in NON_CHARACTER:
        items = report["categories"][cat]
        if count:
            items = items[:count]
        cat_dir = out / cat
        stub = _caption_stub(cat, arch, token, class_word) if arch else None
        manifest = {"category": cat, "count": len(items),
                    "architecture": arch, "caption_stub": stub,
                    "samples": [r.to_dict() for r in items]}
        plan["exported"][cat] = {"dir": str(cat_dir), "count": len(items),
                                 "caption_stub": stub}
        if dry_run:
            continue
        try:
            (cat_dir / "images").mkdir(parents=True, exist_ok=True)
            for i, r in enumerate(items, 1):
                src = Path(r.path)
                if not src.is_file():
                    continue
                dest = cat_dir / "images" / f"{i:05d}{src.suffix.lower() or '.jpg'}"
                if not dest.exists():
                    try:
                        if link_mode == "copy":
                            shutil.copy2(src, dest)
                        elif link_mode == "hardlink":
                            os.link(src, dest)
                        else:
                            os.symlink(src, dest)
                    except Exception as exc:
                        plan["warnings"].append(f"{cat}: {src.name}: {exc}")
                        continue
                if stub is not None:
                    txt = dest.with_suffix(".txt")
                    if not txt.exists():
                        txt.write_text(stub + "\n", encoding="utf-8")
            (cat_dir / "category_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception as exc:
            plan["warnings"].append(f"{cat}: {exc}")
    return plan


__all__ = ["NON_CHARACTER", "CATEGORIES", "CategoryResult", "categorize_frame",
           "categorize_pass", "render_categorize", "export_categories"]
