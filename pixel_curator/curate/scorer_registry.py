"""Scorer registry: one catalog of every curation scorer.

Adding a new scorer is a one-entry change here (module + import probe);
``pixel-curator curate --list-scorers`` shows what is available in the
current environment and what is missing.

Availability checks are import-only and offline - nothing heavy loads.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScorerEntry:
    """One scorer in the catalog."""

    key: str
    module: str                 # e.g. "pixel_curator.curate.basic"
    purpose: str
    needs: tuple[str, ...] = ()  # third-party imports to probe
    stage: str = "quality"       # quality | face | pose | lighting | dedup ...


_CATALOG: tuple[ScorerEntry, ...] = (
    ScorerEntry("sharpness", "pixel_curator.curate.basic",
                "Laplacian-variance sharpness + brightness + dHash",
                stage="core"),
    ScorerEntry("prefilter", "pixel_curator.prefilter",
                "Technical reject pass (tiny, corrupt, text overlays)",
                stage="core"),
    ScorerEntry("duplicates", "pixel_curator.curate.duplicates",
                "Perceptual-hash duplicate / near-duplicate grouping",
                stage="dedup"),
    ScorerEntry("pose-duplicates", "pixel_curator.curate.pose_duplicate_detector",
                "Pose near-duplicate collapse", stage="dedup"),
    ScorerEntry("quality-signals", "pixel_curator.curate.quality_signals",
                "Deep quality: face sharpness, detail retention, blockiness, "
                "integrity", needs=("numpy",), stage="quality"),
    ScorerEntry("aesthetic", "pixel_curator.curate.aesthetic_scorer",
                "Aesthetic scoring (color harmony, composition)",
                needs=("cv2",), stage="quality"),
    ScorerEntry("exposure", "pixel_curator.curate.exposure_outlier",
                "Exposure outlier detection", needs=("cv2",), stage="quality"),
    ScorerEntry("motion-blur", "pixel_curator.curate.motion_blur_detector",
                "Directional motion-blur detection", needs=("cv2",),
                stage="quality"),
    ScorerEntry("color-cast", "pixel_curator.curate.color_cast_detector",
                "Color-cast / white-balance drift", needs=("cv2",),
                stage="quality"),
    ScorerEntry("natural-texture", "pixel_curator.curate.natural_texture",
                "Over-smoothing / plastic-skin texture signal",
                needs=("cv2",), stage="quality"),
    ScorerEntry("hand-quality", "pixel_curator.curate.hand_quality",
                "Hand deformity heuristics (bone ratios)", stage="anatomy"),
    ScorerEntry("body-symmetry", "pixel_curator.curate.body_symmetry",
                "Left/right body symmetry from pose keypoints",
                stage="anatomy"),
    ScorerEntry("body-proportion", "pixel_curator.curate.body_proportion",
                "Body proportion plausibility", stage="anatomy"),
    ScorerEntry("eye-openness", "pixel_curator.curate.eye_openness",
                "Eye aspect-ratio openness", stage="face"),
    ScorerEntry("face-occlusion", "pixel_curator.curate.face_occlusion",
                "Face occlusion detection", stage="face"),
    ScorerEntry("expression", "pixel_curator.curate.expression_quality",
                "Expression quality / neutrality", stage="face"),
    ScorerEntry("gaze", "pixel_curator.curate.gaze_direction",
                "Gaze direction bucketing", stage="face"),
    ScorerEntry("identity", "pixel_curator.curate.identity_confidence",
                "Identity confidence vs anchor embeddings",
                needs=("insightface",), stage="face"),
    ScorerEntry("identity-drift", "pixel_curator.curate.identity_drift_sentinel",
                "Identity embedding drift / mode detection",
                needs=("numpy",), stage="face"),
    ScorerEntry("lighting", "pixel_curator.curate.lighting_analysis",
                "Lighting direction / quality", needs=("cv2",),
                stage="lighting"),
    ScorerEntry("lighting-consistency", "pixel_curator.curate.lighting_consistency",
                "Cross-frame lighting consistency", stage="lighting"),
    ScorerEntry("color-mood", "pixel_curator.curate.color_mood",
                "Palette / mood bucketing", stage="lighting"),
    ScorerEntry("pose-angle", "pixel_curator.curate.pose_angle",
                "Camera/pose angle estimation", stage="pose"),
    ScorerEntry("pose-diversity", "pixel_curator.curate.pose_diversity",
                "Pose coverage / diversity scoring", stage="pose"),
    ScorerEntry("dynamic-pose", "pixel_curator.curate.dynamic_pose_scorer",
                "Dynamic pose interest scoring", stage="pose"),
    ScorerEntry("framing", "pixel_curator.curate.framing",
                "Framing / coverage classification", stage="composition"),
    ScorerEntry("full-body", "pixel_curator.curate.full_body_detector",
                "Full-body visibility detection", stage="composition"),
    ScorerEntry("background", "pixel_curator.curate.background_quality",
                "Background cleanliness", needs=("cv2",), stage="composition"),
    ScorerEntry("figure", "pixel_curator.curate.figure_quality",
                "Figure-study specific quality", stage="composition"),
    ScorerEntry("outfit", "pixel_curator.curate.outfit",
                "Outfit / garment separation signals", stage="wardrobe"),
    ScorerEntry("fabric", "pixel_curator.curate.fabric_analysis",
                "Fabric texture / garment detail", needs=("cv2",),
                stage="wardrobe"),
    ScorerEntry("skin-rendering", "pixel_curator.curate.skin_rendering",
                "Skin rendering artifacts", needs=("cv2",), stage="quality"),
    ScorerEntry("region-quality", "pixel_curator.curate.region_quality",
                "Per-region quality breakdown", stage="quality"),
    ScorerEntry("anatomy-detail", "pixel_curator.curate.anatomy_detail",
                "Anatomy detail plausibility", stage="anatomy"),
)

SCORERS: dict[str, ScorerEntry] = {e.key: e for e in _CATALOG}


def probe(entry: ScorerEntry) -> tuple[bool, str]:
    """Import-only availability check. Never raises."""
    for mod in entry.needs:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            return False, f"missing {mod} ({exc})"
    try:
        importlib.import_module(entry.module)
    except Exception as exc:
        return False, f"import failed ({exc})"
    return True, "ok"


def list_scorers() -> list[dict]:
    """Status row per registered scorer (offline)."""
    rows = []
    for entry in SCORERS.values():
        ok, reason = probe(entry)
        rows.append({"key": entry.key, "available": ok, "stage": entry.stage,
                     "purpose": entry.purpose,
                     "reason": "" if ok else reason})
    return rows


def render_scorers(rows: list[dict] | None = None) -> str:
    rows = rows if rows is not None else list_scorers()
    lines = ["", "Curation scorers", "=" * 64]
    current = None
    for r in rows:
        if r["stage"] != current:
            current = r["stage"]
            lines.append(f"\n[{current}]")
        mark = "ok " if r["available"] else "-- "
        suffix = "" if r["available"] else f"  ({r['reason']})"
        lines.append(f"  {mark}{r['key']:<22} {r['purpose']}{suffix}")
    missing = [r for r in rows if not r["available"]]
    if missing:
        lines.append(f"\n{len(missing)} unavailable - install the matching "
                     f"extra (see pyproject.toml: cv / face / all).")
    return "\n".join(lines)
