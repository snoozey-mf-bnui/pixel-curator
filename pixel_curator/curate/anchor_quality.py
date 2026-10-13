"""Priority 9 - identity-anchor enrichment & automated outlier diagnostics.

A signal-based, dependency-light complement to the ArcFace-based
``clean_anchor_refs.py``. It scores each reference image for identity-anchor
suitability from already-computed advanced-quality metadata (face-region
quality, subject integrity, sharpness, occlusion, blur, duplicate / pose
duplicate, orientation, framing) and produces an ``accept`` / ``review`` /
``reject`` decision with explicit reason codes.

Safety contract (hard requirements):
  * source files are NEVER modified, moved or deleted
  * ``reject`` means "excluded from the rebuilt anchor output", never deleted
  * default mode is DRY-RUN: report/manifest/card only, no file copies
  * apply mode COPIES (never moves) into accepted/ review/ rejected/ folders
  * deterministic, torch/cv2-free, never raises for ordinary input
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DECISIONS = ("accept", "review", "reject")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


@dataclass
class AnchorQualityConfig:
    enabled: bool = True
    min_face_score: float = 0.60
    min_subject_integrity: float = 0.60
    min_face_region_score: float = 0.60
    min_sharpness_score: float = 0.50
    min_identity_confidence: float = 0.55
    max_occlusion_score: float = 0.45
    max_motion_blur_score: float = 0.50
    max_duplicate_score: float = 0.92
    require_face: bool = True
    require_canonical_orientation: bool = True
    allow_close_crop: bool = True
    allow_full_body: bool = True
    dry_run: bool = True
    max_examples: int = 20

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "enabled", "min_face_score", "min_subject_integrity",
            "min_face_region_score", "min_sharpness_score",
            "min_identity_confidence", "max_occlusion_score",
            "max_motion_blur_score", "max_duplicate_score", "require_face",
            "require_canonical_orientation", "allow_close_crop",
            "allow_full_body", "dry_run", "max_examples")}

    def config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class AnchorQualityResult:
    path: str
    decision: str = "review"
    score: float = 0.0
    identity_readiness_score: float = 0.0
    quality_score: float = 0.0
    face_score: float | None = None
    face_region_score: float | None = None
    subject_integrity_score: float | None = None
    duplicate_score: float | None = None
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    output_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "source_path": self.path,
            "output_path": self.output_path,
            "decision": self.decision,
            "anchor_score": round(float(self.score), 4),
            "identity_readiness_score": round(
                float(self.identity_readiness_score), 4),
            "quality_score": round(float(self.quality_score), 4),
            "face_score": _r(self.face_score),
            "face_region_score": _r(self.face_region_score),
            "subject_integrity_score": _r(self.subject_integrity_score),
            "duplicate_score": _r(self.duplicate_score),
            "reason_codes": list(self.reasons),
            "warnings": list(self.warnings),
            "metrics": self.metrics,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass
class AnchorRebuildReport:
    source_dir: str
    output_dir: str
    total: int
    accepted: int
    review: int
    rejected: int
    dry_run: bool = True
    results: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_dir": self.source_dir, "output_dir": self.output_dir,
            "total": self.total, "accepted": self.accepted,
            "review": self.review, "rejected": self.rejected,
            "dry_run": self.dry_run,
            "results": [r.to_dict() if isinstance(r, AnchorQualityResult) else r
                        for r in self.results],
            "summary": self.summary, "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def _r(v):
    return round(float(v), 4) if isinstance(v, (int, float)) else None


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------
def _flatten(sample: dict) -> dict:
    out = dict(sample or {})
    for nest in ("signals", "metadata", "advanced_quality", "region_readiness",
                 "coverage_intensity"):
        blk = (sample or {}).get(nest)
        if isinstance(blk, dict):
            for k, v in blk.items():
                out.setdefault(k, v)
    rq = out.get("region_quality")
    if isinstance(rq, dict):
        out.setdefault("face_quality", rq.get("face_quality"))
    return out


def _num(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _first(d, *keys):
    for k in keys:
        v = _num(d.get(k))
        if v is not None:
            return v
    return None


def evaluate_anchor_sample(sample: dict,
                           config: AnchorQualityConfig | None = None
                           ) -> AnchorQualityResult:
    """Score one anchor reference for identity-anchor suitability. Never raises;
    missing data yields ``review`` with explicit reason codes."""
    cfg = config or AnchorQualityConfig()
    f = _flatten(sample)
    path = str(sample.get("path") or sample.get("source_path")
               or sample.get("sample_path") or "")
    res = AnchorQualityResult(path=path)

    face = _first(f, "face_score", "face_quality", "face_region_score")
    face_region = _first(f, "face_region_score")
    integ = _first(f, "subject_integrity_score", "figure_integrity_score",
                   "integrity_score")
    sharp = _first(f, "sharpness_score", "sharpness")
    blur = _first(f, "motion_blur_score")
    if sharp is None and blur is not None:
        sharp = 1.0 - blur
    occ = _first(f, "face_occlusion_score")
    dup = _first(f, "duplicate_detection_score")
    pose_dup = sample.get("pose_duplicate")
    ident = _first(f, "identity_confidence", "identity_similarity",
                   "identity_sim")
    framing = f.get("framing_label") or f.get("coverage_label")
    has_face_bbox = bool(sample.get("face_bbox") or f.get("face_bbox"))
    orientation_norm = f.get("exif_normalized") or f.get("orientation_normalized")
    exif = _num(f.get("exif_orientation"))

    res.face_score = face
    res.face_region_score = face_region
    res.subject_integrity_score = integ
    res.duplicate_score = dup
    res.metrics = {k: v for k, v in (
        ("face_score", face), ("face_region_score", face_region),
        ("subject_integrity", integ), ("sharpness", sharp),
        ("occlusion", occ), ("motion_blur", blur),
        ("duplicate_score", dup), ("identity_confidence", ident)
    ) if v is not None}

    reasons, warns, hard_reject = [], [], False

    # face presence
    has_face_signal = face is not None or face_region is not None or has_face_bbox
    if not has_face_signal:
        reasons.append("face_missing")
        if cfg.require_face:
            hard_reject = True
    if face is not None and face < cfg.min_face_score:
        reasons.append("face_too_small" if face < cfg.min_face_score * 0.6
                       else "face_region_quality_low")
    if face_region is not None and face_region < cfg.min_face_region_score:
        reasons.append("face_region_quality_low")
    if integ is not None and integ < cfg.min_subject_integrity:
        reasons.append("subject_integrity_low")
    if sharp is not None and sharp < cfg.min_sharpness_score:
        reasons.append("face_blurry")
    if occ is not None and occ > cfg.max_occlusion_score:
        reasons.append("face_occluded")
    if blur is not None and blur > cfg.max_motion_blur_score:
        reasons.append("motion_blur_high")
    if _first(f, "exposure_outlier_score") is not None:
        exo = f.get("exposure_outlier")
        if isinstance(exo, dict) and (exo.get("is_overexposed")
                                      or exo.get("is_underexposed")):
            reasons.append("exposure_outlier")
    if dup is not None and dup >= cfg.max_duplicate_score:
        reasons.append("duplicate_anchor")
        hard_reject = True
    if isinstance(pose_dup, dict) and pose_dup.get("near_duplicate") \
            and not pose_dup.get("representative", True):
        reasons.append("pose_duplicate_anchor")
    if ident is None:
        warns.append("identity_confidence_missing")
    elif ident < cfg.min_identity_confidence:
        reasons.append("identity_confidence_low")
    if cfg.require_canonical_orientation and exif is not None \
            and exif not in (0, 1) and not orientation_norm:
        reasons.append("orientation_mismatch")
    if framing is not None:
        fl = str(framing).lower()
        if fl in ("detail_crop", "needs_review", "unknown"):
            reasons.append("coverage_ambiguous")
        if fl == "full_body" and not cfg.allow_full_body:
            reasons.append("bad_crop")
        if fl in ("close_crop", "face_closeup") and not cfg.allow_close_crop:
            reasons.append("bad_crop")
    if not res.metrics:
        reasons.append("quality_signal_missing")
        warns.append("metadata_incomplete")

    # scores
    quality_parts = [v for v in (face_region, integ, sharp,
                                 (1.0 - occ) if occ is not None else None,
                                 (1.0 - blur) if blur is not None else None)
                     if v is not None]
    res.quality_score = round(sum(quality_parts) / len(quality_parts), 4) \
        if quality_parts else 0.0
    identity_parts = [v for v in (face, face_region, integ, ident)
                      if v is not None]
    res.identity_readiness_score = round(
        sum(identity_parts) / len(identity_parts), 4) if identity_parts else 0.0
    res.score = round(0.6 * res.identity_readiness_score
                      + 0.4 * res.quality_score, 4)

    # decision
    soft = {"face_too_small", "face_region_quality_low", "subject_integrity_low",
            "face_blurry", "face_occluded", "motion_blur_high",
            "exposure_outlier", "identity_confidence_low", "orientation_mismatch",
            "coverage_ambiguous", "pose_duplicate_anchor", "bad_crop"}
    n_soft = len(set(reasons) & soft)
    if hard_reject or "face_missing" in reasons and cfg.require_face:
        res.decision = "reject"
    elif res.score >= 0.62 and n_soft == 0:
        res.decision = "accept"
    elif res.score < 0.40 or n_soft >= 3:
        res.decision = "reject"
    else:
        res.decision = "review"
    res.reasons = sorted(set(reasons))
    res.warnings = sorted(set(warns))
    return res


# ---------------------------------------------------------------------------
# Rebuild workflow
# ---------------------------------------------------------------------------
def _load_sidecar(path: Path) -> dict:
    sc = path.with_suffix(".json")
    if sc.is_file():
        try:
            data = json.loads(sc.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _stable_name(src: Path) -> str:
    h = hashlib.sha256(str(src).encode()).hexdigest()[:8]
    return f"{src.stem}_{h}{src.suffix.lower()}"


def rebuild_anchor_set(source_dir, output_dir,
                       config: AnchorQualityConfig | None = None
                       ) -> AnchorRebuildReport:
    """Evaluate every reference image in ``source_dir`` and produce an anchor
    rebuild report. Default dry-run writes report/manifest/card only; apply mode
    additionally COPIES images into accepted/ review/ rejected/ folders. Source
    files are never modified, moved or deleted. Never raises for ordinary input.
    """
    cfg = config or AnchorQualityConfig()
    source_dir = Path(source_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    warnings: list = []
    images: list = []
    if source_dir.is_dir():
        images = sorted(p for p in source_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    else:
        warnings.append(f"source_dir not found: {source_dir}")

    results: list = []
    for p in images:
        meta = _load_sidecar(p)
        sample = dict(meta)
        sample["path"] = str(p)
        res = evaluate_anchor_sample(sample, cfg)
        results.append(res)

    accepted = [r for r in results if r.decision == "accept"]
    review = [r for r in results if r.decision == "review"]
    rejected = [r for r in results if r.decision == "reject"]

    # apply mode: COPY (never move) into decision folders
    if not cfg.dry_run and images:
        for bucket, rs in (("accepted", accepted), ("review", review),
                           ("rejected", rejected)):
            dest_dir = output_dir / bucket
            dest_dir.mkdir(parents=True, exist_ok=True)
            for r in rs:
                src = Path(r.path)
                dest = dest_dir / _stable_name(src)
                try:
                    if src.is_file() and not dest.exists():
                        shutil.copy2(src, dest)
                    r.output_path = str(dest)
                except Exception as exc:                 # never crash
                    warnings.append(f"copy failed for {src}: {exc}")

    report = AnchorRebuildReport(
        source_dir=str(source_dir), output_dir=str(output_dir),
        total=len(results), accepted=len(accepted), review=len(review),
        rejected=len(rejected), dry_run=cfg.dry_run, results=results,
        warnings=warnings)
    report.summary = _summarize(report, cfg)
    if output_dir:
        _write_outputs(report, output_dir, cfg)
    return report


def _summarize(report: AnchorRebuildReport, cfg: AnchorQualityConfig) -> dict:
    from collections import Counter
    reasons = Counter()
    for r in report.results:
        for rc in r.reasons:
            reasons[rc] += 1
    n = max(1, report.total)

    def _avg(attr):
        vals = [getattr(r, attr) for r in report.results
                if getattr(r, attr) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {
        "total": report.total,
        "accepted": report.accepted, "review": report.review,
        "rejected": report.rejected,
        "accept_ratio": round(report.accepted / n, 4),
        "review_ratio": round(report.review / n, 4),
        "reject_ratio": round(report.rejected / n, 4),
        "avg_anchor_score": _avg("score"),
        "avg_face_score": _avg("face_score"),
        "avg_subject_integrity": _avg("subject_integrity_score"),
        "avg_face_region_score": _avg("face_region_score"),
        "top_reasons": dict(reasons.most_common(10)),
        "config_hash": cfg.config_hash(),
    }


def _write_outputs(report: AnchorRebuildReport, output_dir: Path,
                   cfg: AnchorQualityConfig) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "anchor_manifest_v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_dir": report.source_dir, "output_dir": report.output_dir,
            "dry_run": report.dry_run, "config": cfg.to_dict(),
            "config_hash": cfg.config_hash(),
            "samples": [r.to_dict() for r in report.results],
        }
        (output_dir / "anchor_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        (output_dir / "anchor_report.json").write_text(
            report.to_json(), encoding="utf-8")
        (output_dir / "anchor_card.md").write_text(
            render_anchor_card(report, cfg), encoding="utf-8")
    except Exception as exc:                             # never crash
        report.warnings.append(f"could not write anchor outputs: {exc}")


def render_anchor_card(report: AnchorRebuildReport,
                       cfg: AnchorQualityConfig | None = None) -> str:
    cfg = cfg or AnchorQualityConfig()
    s = report.summary
    name = Path(report.output_dir).name or "anchor_set"
    n = max(1, report.total)
    L = [f"# Anchor Set Card: {name}", "",
         "## Summary",
         f"- Source: {report.source_dir}",
         f"- Output: {report.output_dir}",
         f"- Total: {report.total}",
         f"- Accepted: {report.accepted}",
         f"- Review: {report.review}",
         f"- Rejected: {report.rejected}",
         f"- Dry run: {report.dry_run}", "",
         "## Score Summary",
         f"- Average anchor score: {s.get('avg_anchor_score')}",
         f"- Average face score: {s.get('avg_face_score')}",
         f"- Average subject integrity: {s.get('avg_subject_integrity')}",
         f"- Average face-region score: {s.get('avg_face_region_score')}", "",
         "## Decision Breakdown",
         "| Decision | Count | Ratio |", "|---|---:|---:|",
         f"| accept | {report.accepted} | {round(report.accepted / n, 3)} |",
         f"| review | {report.review} | {round(report.review / n, 3)} |",
         f"| reject | {report.rejected} | {round(report.rejected / n, 3)} |", "",
         "## Top Rejection / Review Reasons", "| Reason | Count |", "|---|---:|"]
    for reason, c in (s.get("top_reasons") or {}).items():
        L.append(f"| {reason} | {c} |")
    L.append("")
    for title, decision in (("Accepted", "accept"), ("Review", "review"),
                            ("Rejected", "reject")):
        L.append(f"## {title} Examples")
        L.append("| File | Score | Reasons |")
        L.append("|---|---:|---|")
        ex = [r for r in report.results if r.decision == decision]
        for r in ex[:cfg.max_examples]:
            fn = Path(r.path).name
            L.append(f"| {fn} | {round(r.score, 3)} | "
                     f"{', '.join(r.reasons) or '-'} |")
        L.append("")
    L.append("## Reproducibility")
    L.append(f"- Generated at: {datetime.now().isoformat(timespec='seconds')}")
    L.append(f"- Config hash: {s.get('config_hash')}")
    L.append(f"- Source root: {report.source_dir}")
    L.append("")
    return "\n".join(L)


def check_anchor_report(report, *, min_accept_count: int = 20,
                        max_review_ratio: float = 0.35,
                        max_duplicate_ratio: float = 0.20) -> list:
    """Warning-only anchor diagnostics over a report (object or dict). Returns
    ``(code, message)`` tuples; never blocks. Never raises."""
    if isinstance(report, AnchorRebuildReport):
        report = report.to_dict()
    out = []
    total = max(1, report.get("total", 0))
    results = report.get("results", [])
    if report.get("accepted", 0) < min_accept_count:
        out.append(("anchor_accept_count_low",
                    f"only {report.get('accepted', 0)} accepted anchors "
                    f"(< {min_accept_count} recommended)"))
    if report.get("review", 0) / total > max_review_ratio:
        out.append(("anchor_review_ratio_high",
                    f"{report.get('review', 0)}/{total} anchors need review "
                    f"(> {max_review_ratio:.0%})"))
    reason_counts: dict = {}
    ident_missing = 0
    for r in results:
        for rc in r.get("reason_codes", []):
            reason_counts[rc] = reason_counts.get(rc, 0) + 1
        if "identity_confidence_missing" in r.get("warnings", []):
            ident_missing += 1
    code_map = {
        "face_missing": "anchor_face_missing",
        "face_region_quality_low": "anchor_face_region_low",
        "subject_integrity_low": "anchor_subject_integrity_low",
        "orientation_mismatch": "anchor_orientation_mismatch",
    }
    for rc, code in code_map.items():
        if reason_counts.get(rc):
            out.append((code, f"{reason_counts[rc]} anchors flagged {rc}"))
    dup = reason_counts.get("duplicate_anchor", 0)
    if dup / total > max_duplicate_ratio:
        out.append(("anchor_duplicate_high",
                    f"{dup}/{total} anchors are duplicates "
                    f"(> {max_duplicate_ratio:.0%})"))
    if reason_counts.get("pose_duplicate_anchor"):
        out.append(("anchor_pose_duplicate_high",
                    f"{reason_counts['pose_duplicate_anchor']} pose-duplicate "
                    "anchors"))
    if ident_missing == total:
        out.append(("anchor_identity_confidence_missing",
                    "no anchor carries an identity-confidence signal"))
    return out


__all__ = [
    "DECISIONS", "AnchorQualityConfig", "AnchorQualityResult",
    "AnchorRebuildReport", "evaluate_anchor_sample", "rebuild_anchor_set",
    "render_anchor_card", "check_anchor_report",
]
