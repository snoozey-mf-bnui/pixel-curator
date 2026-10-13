#!/usr/bin/env python3
"""specialized_recipe_builder.py

Builds specialized training recipes based on visual focus classification,
WD14 tags, and quality signals. Routes each image to one or more specialized
recipe targets (full_body_pose_reference, outfit_reference, portrait_reference,
figure_composition_reference, outfit_fabric_reference, etc.) and aggregates
them into a plan that can drive dataset builds and training recipes.

Design principles:
  - Metadata-only: never reads pixel data, never builds datasets, never
    launches training. All output is JSON + TOML suggestions.
  - Safety: age-uncertain images are routed to a blocked "review_set";
    low-quality images are routed to "reject_low_quality". Recipe names are
    professional and contain no unsafe vocabulary.
  - Caption-independent: when visual_focus/caption metadata is absent,
    routing falls back to WD14 tags + quality signals.
  - Dry-run by default: ``build_specialized_recipe_plan`` writes plan files
    but does not build datasets; ``generate_dataset_build_plan`` only
    suggests commands.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    from pixel_curator import nsfw_router
except ImportError:
    nsfw_router = None


# ============================ taxonomy ======================================

#: Approved specialized recipe target names. Professional, catalog-style.
#: Must NEVER contain unsafe vocabulary (nude, sexual, nsfw, explicit, porn).
SPECIALIZED_RECIPE_NAMES = (
    "full_body_pose_reference",
    "outfit_reference",
    "outfit_fabric_reference",
    "portrait_reference",
    "figure_composition_reference",
    "pose_dynamic_reference",
    "gesture_pose_reference",
    "detail_texture_reference",
    "lighting_reference",
    "composition_reference",
    "catalog_reference",
    "style_aesthetic_reference",
    "review_set",
    "reject_low_quality",
)

#: Mapping from visual-focus primary labels to recommended recipe targets.
_FOCUS_TO_RECIPE = {
    "full_body": "full_body_pose_reference",
    "full_body_pose": "full_body_pose_reference",
    "upper_body": "outfit_reference",
    "upper_body_focus": "outfit_reference",
    "torso_composition": "outfit_reference",
    "outfit": "outfit_reference",
    "outfit_focus": "outfit_reference",
    "portrait": "portrait_reference",
    "face_portrait": "portrait_reference",
    "head_and_shoulders": "portrait_reference",
    "detail_texture": "detail_texture_reference",
    "close_crop_study": "detail_texture_reference",
    "hands_detail": "detail_texture_reference",
    "feet_detail": "detail_texture_reference",
    "composition": "composition_reference",
    "environmental_composition": "composition_reference",
    "lighting_composition": "lighting_reference",
    "figure_composition": "figure_composition_reference",
    "figure_study_composition": "figure_composition_reference",
    "pose_dynamic": "pose_dynamic_reference",
    "dynamic_pose": "pose_dynamic_reference",
    "gesture_forward": "gesture_pose_reference",
}

#: WD14 tag → recipe target hints (used when visual_focus is absent).
_WD14_TAG_HINTS = {
    "full_body": "full_body_pose_reference",
    "standing": "full_body_pose_reference",
    "outfit": "outfit_reference",
    "dress": "outfit_reference",
    "clothing": "outfit_reference",
    "wardrobe": "outfit_reference",
    "fabric": "outfit_fabric_reference",
    "texture": "outfit_fabric_reference",
    "portrait": "portrait_reference",
    "face": "portrait_reference",
    "close-up": "detail_texture_reference",
    "macro": "detail_texture_reference",
}

#: Output filenames written by ``write_specialized_recipe_plan``.
OUTPUT_FILES = {
    "plan": "specialized_recipe_plan.json",
    "assignments": "specialized_recipe_assignments.jsonl",
    "summary": "specialized_recipe_summary.json",
    "toml": "specialized_recipe_toml_suggestions.json",
    "build_plan": "specialized_recipe_build_plan.json",
}

#: Unsafe vocabulary — tags matching these are dropped during normalization
#: and never appear in recipe names or routing.
_UNSAFE_TOKENS = ("nsfw", "nude", "naked", "sexual", "explicit", "porn",
                  "nipple", "genital", "penis", "vagina", "cum", "sex",
                  "fuck", "cock", "dick", "pussy", "loli", "shota",
                  "child", "underage", "minor", "baby", "toddler")


# ============================ dataclasses ===================================

@dataclass
class RecipeCandidate:
    """A single recommended recipe for an image."""
    name: str
    confidence: float
    recipe_type: str = "specialized"
    reason: str = ""
    required_tags: list = field(default_factory=list)
    source: str = "visual_focus"   # visual_focus | wd14 | quality | fallback

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImageAssignment:
    """Routing result for a single image."""
    image_path: str
    primary_recipe: str
    recommended_recipes: list = field(default_factory=list)
    suggested_version_name: str = ""
    blocked: bool = False
    source_tags: list = field(default_factory=list)
    suitability_score: float = 0.0
    suitability_tier: str = "review"
    routing_confidence: float = 0.0
    architecture: str = "sdxl"
    routing_summary: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "primary_recipe": self.primary_recipe,
            "recommended_recipes": [c.to_dict() if isinstance(c, RecipeCandidate)
                                    else c for c in self.recommended_recipes],
            "suggested_version_name": self.suggested_version_name,
            "blocked": self.blocked,
            "source_tags": list(self.source_tags),
            "suitability_score": round(self.suitability_score, 2),
            "suitability_tier": self.suitability_tier,
            "routing_confidence": round(self.routing_confidence, 4),
            "architecture": self.architecture,
            "routing_summary": self.routing_summary,
            "metadata": dict(self.metadata),
        }


@dataclass
class SpecializedRecipePlan:
    """Aggregated plan across all images."""
    plan_id: str
    architecture: str = "sdxl"
    assignments: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    project_toml_suggestions: dict = field(default_factory=dict)
    created_at: str = ""
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "architecture": self.architecture,
            "assignments": [a.to_dict() if isinstance(a, ImageAssignment) else a
                            for a in self.assignments],
            "warnings": list(self.warnings),
            "summary": dict(self.summary),
            "project_toml_suggestions": dict(self.project_toml_suggestions),
            "created_at": self.created_at,
            "config": dict(self.config),
        }


@dataclass
class SpecializedRecipeBuilderConfig:
    """Configuration for the recipe builder."""
    min_recipe_confidence: float = 0.55
    min_image_suitability: float = 50.0
    max_recipes_per_image: int = 4
    architecture_default: str = "sdxl"
    allow_mixed_architecture: bool = False
    overwrite: bool = False
    use_wd14_tags: bool = True
    use_quality_signals: bool = True
    use_visual_focus: bool = True
    use_captions_optional: bool = True


# ============================ helpers =======================================

def _norm_tag(t: str) -> str:
    """Lowercase, underscore-joined, trimmed."""
    return " ".join(str(t or "").strip().lower().replace("_", " ").split())


def _is_unsafe(t: str) -> bool:
    low = t.lower()
    return any(tok in low for tok in _UNSAFE_TOKENS)


def _normalize_tags(rec: dict) -> list:
    """Extract + normalize WD14 tags from a record.

    Accepts ``tags`` (dict tag→score), ``wd14_tags`` (list), or
    ``caption_wd14`` (comma-separated string). Returns a list of lowercase,
    underscore-joined tags with unsafe vocabulary dropped.
    """
    tags_dict = rec.get("tags")
    if isinstance(tags_dict, dict):
        items = sorted(tags_dict.items(),
                       key=lambda kv: -float(kv[1] if kv[1] is not None else 0))
        out = []
        for raw_tag, _score in items:
            t = _norm_tag(raw_tag).replace(" ", "_")
            if t and not _is_unsafe(t):
                out.append(t)
        return out
    # wd14_tags as a list
    wd14_list = rec.get("wd14_tags")
    if isinstance(wd14_list, list):
        out = []
        for raw in wd14_list:
            t = _norm_tag(raw).replace(" ", "_")
            if t and not _is_unsafe(t):
                out.append(t)
        return out
    cap = rec.get("caption_wd14") or rec.get("caption") or ""
    if isinstance(cap, list):
        cap = ",".join(str(x) for x in cap)
    out = []
    for raw in str(cap).split(","):
        t = _norm_tag(raw).replace(" ", "_")
        if t and not _is_unsafe(t):
            out.append(t)
    return out


def _first_focus(rec: dict) -> str:
    """Extract the primary focus label from a record (handles multiple shapes)."""
    vf = rec.get("visual_focus") or {}
    if isinstance(vf, dict):
        pf = vf.get("primary_focus")
        if isinstance(pf, dict):
            return str(pf.get("label", "")).lower()
        if isinstance(pf, str):
            return pf.lower()
    pf = rec.get("primary_focus")
    if isinstance(pf, dict):
        return str(pf.get("label", "")).lower()
    if isinstance(pf, str):
        return pf.lower()
    return ""


def _get_targets(rec: dict) -> list:
    """Extract recommended_targets from a record (handles multiple shapes)."""
    vf = rec.get("visual_focus") or {}
    targets = []
    if isinstance(vf, dict):
        raw = vf.get("recommended_targets", [])
    else:
        raw = rec.get("recommended_targets", [])
    for t in raw:
        if isinstance(t, dict):
            targets.append(str(t.get("target", "")))
        elif isinstance(t, str):
            targets.append(t)
    return [t for t in targets if t]


def _get_review_flags(rec: dict) -> list:
    vf = rec.get("visual_focus") or {}
    if isinstance(vf, dict):
        return [str(f) for f in vf.get("review_flags", [])]
    return [str(f) for f in rec.get("review_flags", [])]


def _get_suitability(rec: dict) -> tuple:
    """Return (score, tier) from the record."""
    vf = rec.get("visual_focus") or {}
    if isinstance(vf, dict):
        score = vf.get("suitability_score", rec.get("suitability_score", 0))
        tier = vf.get("suitability_tier", rec.get("suitability_tier", "review"))
    else:
        score = rec.get("suitability_score", 0)
        tier = rec.get("suitability_tier", "review")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0
    return score, str(tier)


def _get_quality_signals(rec: dict) -> dict:
    qs = rec.get("quality_signals")
    if isinstance(qs, dict):
        return qs
    vf = rec.get("visual_focus") or {}
    if isinstance(vf, dict):
        qs = vf.get("quality_signals")
        if isinstance(qs, dict):
            return qs
    return {}


def _version_name(primary: str, image_path: str) -> str:
    """Generate a professional, safe version name ending in _v1."""
    base = Path(str(image_path)).stem or "image"
    # sanitize: only alnum + underscore
    safe = re.sub(r"[^a-zA-Z0-9_]", "", base).lower() or "image"
    recipe_slug = re.sub(r"[^a-zA-Z0-9_]", "", primary).lower()
    return f"{recipe_slug}_{safe}_v1"


# ============================ core routing ==================================

def assign_image_to_recipes(rec: dict,
                            config: SpecializedRecipeBuilderConfig | None = None
                            ) -> ImageAssignment:
    """Route a single image record to specialized recipes.

    The record may contain:
      - ``visual_focus``: dict with primary_focus, recommended_targets,
        suitability_score, review_flags
      - ``wd14_tags`` / ``caption_wd14``: tag list/string for fallback routing
      - ``quality_signals``: dict with integrity, region_clarity, etc.
      - ``architecture_target``: "sdxl" | other
    """
    cfg = config or SpecializedRecipeBuilderConfig()
    image_path = str(rec.get("image_path", "/unknown.jpg"))
    arch = str(rec.get("architecture_target", cfg.architecture_default)).lower()
    warnings = []
    source_tags = _normalize_tags(rec)
    score, tier = _get_suitability(rec)
    flags = _get_review_flags(rec)
    targets = _get_targets(rec)
    focus = _first_focus(rec)
    qs = _get_quality_signals(rec)

    # Derive a suitability score from quality signals when no explicit score
    # was provided (e.g. WD14-only records).
    if score == 0 and qs:
        integrity = float(qs.get("integrity", 0) or 0)
        region_clarity = float(qs.get("region_clarity", 0) or 0)
        score = round((integrity + region_clarity) / 2 * 100, 1)
        if score >= 70:
            tier = "good"
        elif score >= 50:
            tier = "review"
        else:
            tier = "reject"

    # --- 1. Age-uncertain safety block (highest priority) ---
    if "review_required_age_uncertain" in flags:
        return ImageAssignment(
            image_path=image_path,
            primary_recipe="review_set",
            recommended_recipes=[RecipeCandidate(
                name="review_set", confidence=1.0, recipe_type="review",
                reason="age_uncertain - specialized routing blocked pending review",
                source="safety")],
            suggested_version_name="review_set_v1",
            blocked=True,
            source_tags=source_tags,
            suitability_score=score,
            suitability_tier=tier,
            routing_confidence=0.0,
            architecture=arch,
            routing_summary="blocked: age_uncertain review required",
            metadata={"warnings": ["age_uncertain_review_required"],
                      "review_flags": flags},
        )

    # --- 2. Low-quality reject ---
    if tier == "reject" or (score > 0 and score < 20):
        return ImageAssignment(
            image_path=image_path,
            primary_recipe="reject_low_quality",
            recommended_recipes=[RecipeCandidate(
                name="reject_low_quality", confidence=1.0,
                recipe_type="reject",
                reason=f"suitability {score:.1f} below usable threshold",
                source="quality")],
            suggested_version_name="reject_v1",
            blocked=False,
            source_tags=source_tags,
            suitability_score=score,
            suitability_tier=tier,
            routing_confidence=0.0,
            architecture=arch,
            routing_summary=f"rejected: {tier} tier (score {score:.1f})",
            metadata={"warnings": [], "review_flags": flags},
        )

    # --- 3. Visual-focus-based routing (primary path) ---
    candidates = []
    if cfg.use_visual_focus and targets:
        for tgt in targets:
            recipe_name = tgt if tgt in SPECIALIZED_RECIPE_NAMES else \
                _FOCUS_TO_RECIPE.get(tgt, _FOCUS_TO_RECIPE.get(focus, ""))
            if not recipe_name or recipe_name not in SPECIALIZED_RECIPE_NAMES:
                continue
            conf = 0.7 + min(0.25, max(0.0, (score - 50) / 200))
            candidates.append(RecipeCandidate(
                name=recipe_name, confidence=round(conf, 4),
                recipe_type="specialized",
                reason=f"visual_focus target: {tgt}",
                required_tags=source_tags[:5],
                source="visual_focus"))
    elif cfg.use_visual_focus and focus:
        recipe_name = _FOCUS_TO_RECIPE.get(focus)
        if recipe_name and recipe_name in SPECIALIZED_RECIPE_NAMES:
            conf = 0.65 + min(0.2, max(0.0, (score - 50) / 250))
            candidates.append(RecipeCandidate(
                name=recipe_name, confidence=round(conf, 4),
                recipe_type="specialized",
                reason=f"visual_focus primary: {focus}",
                required_tags=source_tags[:5],
                source="visual_focus"))

    # --- 4. WD14 tag-based fallback routing ---
    if not candidates and cfg.use_wd14_tags and source_tags:
        for tag in source_tags:
            hint = _WD14_TAG_HINTS.get(tag) or _WD14_TAG_HINTS.get(
                tag.replace("_", " "))
            if hint and hint in SPECIALIZED_RECIPE_NAMES:
                candidates.append(RecipeCandidate(
                    name=hint, confidence=0.6,
                    recipe_type="specialized",
                    reason=f"WD14 tag hint: {tag}",
                    required_tags=[tag],
                    source="wd14"))
                break  # one WD14 hint is enough

    # --- 5. Quality-only fallback ---
    if not candidates and cfg.use_quality_signals and qs:
        integrity = float(qs.get("integrity", 0) or 0)
        if integrity < 0.3:
            candidates.append(RecipeCandidate(
                name="reject_low_quality", confidence=0.8,
                recipe_type="reject",
                reason=f"quality integrity {integrity:.2f} too low",
                source="quality"))
        else:
            candidates.append(RecipeCandidate(
                name="review_set", confidence=0.5,
                recipe_type="review",
                reason="insufficient metadata for specialized routing",
                source="fallback"))

    # --- 6. Final fallback: review_set ---
    if not candidates:
        warnings.append("no routing signals found; defaulting to review_set")
        candidates.append(RecipeCandidate(
            name="review_set", confidence=0.4,
            recipe_type="review",
            reason="no visual_focus, WD14 tags, or quality signals available",
            source="fallback"))

    # Cap at max_recipes_per_image
    candidates = candidates[:cfg.max_recipes_per_image]
    primary = candidates[0].name
    version = _version_name(primary, image_path)

    return ImageAssignment(
        image_path=image_path,
        primary_recipe=primary,
        recommended_recipes=candidates,
        suggested_version_name=version,
        blocked=False,
        source_tags=source_tags,
        suitability_score=score,
        suitability_tier=tier,
        routing_confidence=candidates[0].confidence,
        architecture=arch,
        routing_summary=", ".join(c.name for c in candidates),
        metadata={"warnings": warnings, "review_flags": flags,
                  "focus": focus, "targets": targets},
    )


# ============================ plan building =================================

def _detect_architecture(assignments: list, cfg: SpecializedRecipeBuilderConfig
                         ) -> tuple:
    """Return (architecture, warnings)."""
    archs = {a.architecture for a in assignments if a.architecture}
    if len(archs) > 1:
        if not cfg.allow_mixed_architecture:
            return "mixed", [
                f"mixed architecture detected ({', '.join(sorted(archs))}) - "
                "pass --allow-mixed-architecture to proceed"]
        return "mixed", [
            f"mixed architecture allowed ({', '.join(sorted(archs))})"]
    return (sorted(archs)[0] if archs else cfg.architecture_default), []


def build_specialized_recipe_plan(
        records: list,
        output_dir=None,
        config: SpecializedRecipeBuilderConfig | None = None
        ) -> SpecializedRecipePlan:
    """Build a specialized recipe plan from a list of image records.

    ``records`` is a list of dicts (visual_focus results, WD14 tag records,
    or mixed). ``output_dir`` is accepted for API compatibility but the plan
    itself is not written here — use ``write_specialized_recipe_plan``.
    """
    cfg = config or SpecializedRecipeBuilderConfig()
    assignments = [assign_image_to_recipes(r, cfg) for r in records]
    arch, arch_warnings = _detect_architecture(assignments, cfg)
    plan_id = f"srp_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    plan = SpecializedRecipePlan(
        plan_id=plan_id,
        architecture=arch,
        assignments=assignments,
        warnings=list(arch_warnings),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        config={"min_recipe_confidence": cfg.min_recipe_confidence,
                "min_image_suitability": cfg.min_image_suitability,
                "max_recipes_per_image": cfg.max_recipes_per_image,
                "architecture_default": cfg.architecture_default},
    )
    plan.summary = summarize_recipe_plan(plan)
    return plan


def summarize_recipe_plan(plan: SpecializedRecipePlan) -> dict:
    """Return a summary dict with counts per primary recipe."""
    by_recipe: dict = {}
    for a in plan.assignments:
        by_recipe[a.primary_recipe] = by_recipe.get(a.primary_recipe, 0) + 1
    return {
        "total_images": len(plan.assignments),
        "images_by_primary_recipe": by_recipe,
        "architecture": plan.architecture,
        "blocked_count": sum(1 for a in plan.assignments if a.blocked),
        "plan_id": plan.plan_id,
    }


# ============================ output writers ================================

def write_specialized_recipe_plan(plan: SpecializedRecipePlan,
                                  output_dir,
                                  overwrite: bool = False) -> dict:
    """Write plan files to ``output_dir``. Returns dict of written paths."""
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    plan_dict = plan.to_dict()

    # plan JSON
    plan_path = out / OUTPUT_FILES["plan"]
    if overwrite or not plan_path.exists():
        plan_path.write_text(json.dumps(plan_dict, indent=2, default=str),
                             encoding="utf-8")
    written["plan"] = str(plan_path)

    # assignments JSONL
    assign_path = out / OUTPUT_FILES["assignments"]
    if overwrite or not assign_path.exists():
        with assign_path.open("w", encoding="utf-8") as fh:
            for a in plan.assignments:
                fh.write(json.dumps(a.to_dict(), default=str) + "\n")
    written["assignments"] = str(assign_path)

    # summary JSON
    summary_path = out / OUTPUT_FILES["summary"]
    if overwrite or not summary_path.exists():
        summary_path.write_text(
            json.dumps(plan.summary, indent=2, default=str),
            encoding="utf-8")
    written["summary"] = str(summary_path)

    # TOML suggestions
    toml_sug = generate_project_toml_recipe_suggestions(plan, out)
    toml_path = out / OUTPUT_FILES["toml"]
    if overwrite or not toml_path.exists():
        toml_path.write_text(json.dumps(toml_sug, indent=2, default=str),
                             encoding="utf-8")
    written["toml"] = str(toml_path)

    # build plan
    bplan = generate_dataset_build_plan(plan, out)
    bp_path = out / OUTPUT_FILES["build_plan"]
    if overwrite or not bp_path.exists():
        bp_path.write_text(json.dumps(bplan, indent=2, default=str),
                           encoding="utf-8")
    written["build_plan"] = str(bp_path)

    return written


def load_specialized_recipe_plan(output_dir) -> SpecializedRecipePlan:
    """Load a previously-written plan from ``output_dir``."""
    out = Path(output_dir).expanduser()
    plan_path = out / OUTPUT_FILES["plan"]
    if not plan_path.exists():
        # also check specialized_recipes/ subdir
        plan_path = out / "specialized_recipes" / OUTPUT_FILES["plan"]
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    assignments = []
    for a in data.get("assignments", []):
        recipes = [RecipeCandidate(**c) if isinstance(c, dict) else c
                   for c in a.get("recommended_recipes", [])]
        assignments.append(ImageAssignment(
            image_path=a.get("image_path", ""),
            primary_recipe=a.get("primary_recipe", "review_set"),
            recommended_recipes=recipes,
            suggested_version_name=a.get("suggested_version_name", ""),
            blocked=a.get("blocked", False),
            source_tags=a.get("source_tags", []),
            suitability_score=a.get("suitability_score", 0),
            suitability_tier=a.get("suitability_tier", "review"),
            routing_confidence=a.get("routing_confidence", 0),
            architecture=a.get("architecture", "sdxl"),
            routing_summary=a.get("routing_summary", ""),
            metadata=a.get("metadata", {}),
        ))
    return SpecializedRecipePlan(
        plan_id=data.get("plan_id", ""),
        architecture=data.get("architecture", "sdxl"),
        assignments=assignments,
        warnings=data.get("warnings", []),
        summary=data.get("summary", {}),
        project_toml_suggestions=data.get("project_toml_suggestions", {}),
        created_at=data.get("created_at", ""),
        config=data.get("config", {}),
    )


def results_by_filename(path) -> dict:
    """Load assignments JSONL and return a dict keyed by image basename.

    ``path`` may be a directory (searched for the assignments file) or the
    assignments JSONL file directly.
    """
    p = Path(path).expanduser()
    if p.is_dir():
        for cand in (p / OUTPUT_FILES["assignments"],
                     p / "specialized_recipes" / OUTPUT_FILES["assignments"]):
            if cand.exists():
                p = cand
                break
    if not p.exists() or p.is_dir():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        basename = Path(rec.get("image_path", "")).name
        if basename:
            out[basename] = rec
    return out


def result_to_dataset_explorer_fields(assignment: ImageAssignment) -> dict:
    """Map an assignment to the fields consumed by the dataset explorer."""
    top = assignment.recommended_recipes[0] if assignment.recommended_recipes else None
    return {
        "recommended_recipes": [c.name if isinstance(c, RecipeCandidate)
                                else c.get("name", "")
                                for c in assignment.recommended_recipes],
        "primary_recipe": assignment.primary_recipe,
        "recipe_confidence": round(top.confidence, 4) if top else 0.0,
        "suggested_version_name": assignment.suggested_version_name,
        "routing_summary": assignment.routing_summary,
        "architecture": assignment.architecture,
        "blocked": assignment.blocked,
        "suitability_score": round(assignment.suitability_score, 2),
    }


def generate_project_toml_recipe_suggestions(plan: SpecializedRecipePlan,
                                             output_dir=None) -> dict:
    """Generate [recipes.*] TOML suggestions for project.toml.

    Returns a dict with ``recipes`` (list of recipe suggestion dicts) and
    ``toml_preview`` (a string containing the TOML block).
    """
    seen = {}
    for a in plan.assignments:
        for c in a.recommended_recipes:
            if c.name not in seen and c.name in SPECIALIZED_RECIPE_NAMES:
                seen[c.name] = {
                    "name": c.name,
                    "type": c.recipe_type,
                    "target": c.name,
                    "confidence": round(c.confidence, 4),
                }
    recipes = list(seen.values())
    lines = ["# Specialized recipe suggestions (generated by pixel_curator)"]
    for r in recipes:
        slug = r["name"]
        lines.append("")
        lines.append(f"[recipes.{slug}]")
        lines.append(f'type = "{r["type"]}"')
        lines.append(f'target = "{r["target"]}"')
        lines.append(f'confidence = {r["confidence"]}')
    return {
        "recipes": recipes,
        "toml_preview": "\n".join(lines),
    }


def generate_dataset_build_plan(plan: SpecializedRecipePlan,
                                output_dir) -> dict:
    """Generate a dry-run dataset build plan with suggested commands.

    No datasets are built; this only suggests commands for the user to run.
    """
    recipes = sorted({c.name for a in plan.assignments
                      for c in a.recommended_recipes
                      if c.name in SPECIALIZED_RECIPE_NAMES
                      and c.name not in ("review_set", "reject_low_quality")})
    steps = []
    for recipe in recipes:
        version = f"{recipe}_v1"
        steps.append({
            "recipe": recipe,
            "version": version,
            "image_count": sum(
                1 for a in plan.assignments
                if any(c.name == recipe for c in a.recommended_recipes)),
            "suggested_command": (
                f"python -m pixel_curator build --recipe {recipe} "
                f"--version {version} --apply"),
            "dry_run": True,
        })
    return {
        "dry_run": True,
        "note": "dry-run only - no datasets are built; run the suggested "
                "commands manually to build",
        "steps": steps,
        "plan_id": plan.plan_id,
    }


def render_recipe_card(plan: SpecializedRecipePlan) -> str:
    """Render a short markdown summary card for the plan."""
    s = plan.summary
    lines = [
        "# Specialized Recipe Plan",
        "",
        f"- **plan_id**: `{plan.plan_id}`",
        f"- **architecture**: {plan.architecture}",
        f"- **total images**: {s.get('total_images', 0)}",
        f"- **blocked**: {s.get('blocked_count', 0)}",
        "",
        "## Images by primary recipe",
        "",
    ]
    for recipe, count in sorted(s.get("images_by_primary_recipe", {}).items()):
        lines.append(f"- **{recipe}**: {count}")
    if plan.warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for w in plan.warnings:
            lines.append(f"- {w}")
    return "\n".join(lines)


# ============================ legacy API (kept for compatibility) ===========

@dataclass
class RecipeConfig:
    """Legacy config — use SpecializedRecipeBuilderConfig for new code."""
    use_nsfw_routing: bool = False
    min_confidence_for_specialization: float = 0.55
    allow_explicit_routing: bool = False


@dataclass
class SpecializedRecipe:
    """Legacy recipe dataclass — use RecipeCandidate for new code."""
    recipe_name: str
    category: str
    confidence: float
    recommended_targets: list = field(default_factory=list)
    prompt_boosters: list = field(default_factory=list)
    nsfw_tags: list = field(default_factory=list)
    notes: str = ""


def _get_nsfw_category(tags: list) -> str:
    if nsfw_router:
        return nsfw_router.get_nsfw_category(tags)
    return "general"


def _should_route_nsfw(tags: list) -> bool:
    if not nsfw_router:
        return False
    return nsfw_router.should_use_nsfw_recipe(tags)


def build_specialized_recipe(visual_focus_result: dict,
                             existing_tags: list = None,
                             config: RecipeConfig | None = None
                             ) -> SpecializedRecipe:
    """Legacy single-image recipe builder — kept for backwards compat."""
    rec = {"visual_focus": visual_focus_result,
           "wd14_tags": existing_tags or []}
    assignment = assign_image_to_recipes(rec)
    top = assignment.recommended_recipes[0] if assignment.recommended_recipes else None
    return SpecializedRecipe(
        recipe_name=top.name if top else "review_set",
        category=top.source if top else "fallback",
        confidence=top.confidence if top else 0.0,
        recommended_targets=[c.name for c in assignment.recommended_recipes],
        prompt_boosters=[],
        nsfw_tags=[],
        notes=top.reason if top else "",
    )


def build_recipe_from_tags(tags: list,
                           config: RecipeConfig | None = None
                           ) -> SpecializedRecipe:
    """Legacy tag-only recipe builder — kept for backwards compat."""
    rec = {"wd14_tags": tags}
    assignment = assign_image_to_recipes(rec)
    top = assignment.recommended_recipes[0] if assignment.recommended_recipes else None
    return SpecializedRecipe(
        recipe_name=top.name if top else "review_set",
        category=top.source if top else "fallback",
        confidence=top.confidence if top else 0.0,
        recommended_targets=[c.name for c in assignment.recommended_recipes],
        prompt_boosters=[],
        nsfw_tags=[],
        notes=top.reason if top else "",
    )


if __name__ == "__main__":
    print("specialized_recipe_builder.py loaded (metadata-only routing)")
